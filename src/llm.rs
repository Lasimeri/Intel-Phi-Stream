//! One model, split three ways, and one context over llama.cpp's C API
//! (`sys.rs`): the backends (CUDA, the CPU, the cards' `libggml_phi.so`
//! through `GGML_BACKEND_PATH`), the model with the experts of the host
//! blocks overridden into host memory (`split.rs`), one context with a
//! unified KV cache and several sequences, decoding of a cycle made of
//! lanes (a token or a chunk per sequence), the sampler chain, and the
//! memory operations the engine composes sequences with. llama.cpp is
//! used as a library and never changed. See llm.md.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::{bail, Context as _, Result};

use crate::capture::{eval_callback, Capture, CaptureConfig};
use crate::remote::Remote;
use crate::split::{self, Split};
use crate::sys;

/// The tensors kept in host memory whatever the plan (`split::Sizes::host_only`),
/// as llama.cpp's override pattern.
const HOST_ONLY: &CStr = c"per_layer_token_embd\\.weight";

/// How the model and its context are set up.
#[derive(Clone, Debug)]
pub struct Options {
    pub model: String,
    /// Where the backends are loaded from (the llama.cpp build's `bin`).
    pub backend_dir: String,
    /// Cells of the context (one unified pool for every sequence).
    pub ctx: u32,
    /// The most tokens one cycle carries (the live token and a chunk).
    pub batch: u32,
    /// Threads for what llama.cpp's own CPU backend still computes (the
    /// token embedding lookups); the cards' backend has its own pool.
    pub threads: i32,
    /// Blocks whose experts stay on the GPU; none: as many as fit.
    pub gpu_blocks: Option<usize>,
    /// Blocks whose experts go to the GPU first (the blocks the mind reads).
    pub keep_on_gpu: Vec<usize>,
    /// K and V as 8-bit blocks instead of float16 (half the cells' bytes).
    pub kv_q8: bool,
    /// The KV cache and the attention over it in host memory
    /// (llama.cpp's `offload_kqv` off): none of it reserved on a GPU.
    pub kv_host: bool,
    /// llama.cpp's `op_offload`: a batch over host weights copied to a GPU.
    pub op_offload: bool,
    /// Each GPU's room for llama.cpp's working buffers, bytes; none: 1.25
    /// GiB plus 2 MiB per batch token.
    pub gpu_headroom: Option<u64>,
    /// No GPU: the model on the host (and the cards, when their backend is
    /// loaded), so a second model can have the GPU.
    pub cpu: bool,
    /// Sequences the context can hold apart.
    pub n_seq: u32,
    /// One pool of cells for every sequence (off: a stream per sequence).
    pub kv_unified: bool,
    /// Show llama.cpp's informational log.
    pub verbose: bool,
    /// Read the residual stream through llama.cpp's eval callback
    /// (`capture.rs`); none: no callback is installed at all.
    pub capture: Option<CaptureConfig>,
    /// GPU memory to leave free beyond the context's own needs (the
    /// readout's backend and transports), bytes.
    pub extra_reserve: u64,
}

static VERBOSE: AtomicBool = AtomicBool::new(false);
static LAST_LEVEL: AtomicU32 = AtomicU32::new(0);

/// llama.cpp's log: warnings and errors always, the rest with `verbose`.
/// A continuation line follows its message's level.
unsafe extern "C" fn log_cb(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    let level: u32 = level;
    // ggml_log_level: 1 debug, 2 info, 3 warn, 4 error, 5 continuation
    let shown = if level == 5 {
        LAST_LEVEL.load(Ordering::Relaxed)
    } else {
        level
    };
    if level != 5 {
        LAST_LEVEL.store(level, Ordering::Relaxed);
    }
    if shown >= 3 || VERBOSE.load(Ordering::Relaxed) {
        // SAFETY: llama.cpp passes a NUL-terminated string.
        let s = unsafe { CStr::from_ptr(text) };
        eprint!("{}", s.to_string_lossy());
    }
}

/// Sampling as the model's header recommends unless overridden.
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    /// Tokens under this share of the likeliest one's probability dropped
    /// (0: off).
    pub min_p: f32,
    /// DRY (llama.cpp's sampler against repeated sequences): its multiplier
    /// (0: off), base, the length repeated freely, and how far back it
    /// looks (-1: the whole context).
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_last_n: i32,
    pub seed: u32,
    /// Tokens seen again in the last `repeat_last_n` are divided by this
    /// (1: off).
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    /// Never sample a token that carries an em or en dash.
    pub ban_dashes: bool,
}

/// One sequence's part of a cycle: `tokens` at positions `pos0..`, the
/// logits of the last one kept when `logits` is set.
pub struct Lane<'a> {
    pub seq: i32,
    pub tokens: &'a [i32],
    pub pos0: i32,
    pub logits: bool,
}

/// The loaded model and its context; or, with `--remote`, a server that
/// holds the one sequence (`remote.rs`): then there is no model, context,
/// vocabulary pointer or sampler here, and every method below goes to it.
pub struct Llm {
    model: Option<NonNull<sys::llama_model>>,
    ctx: Option<NonNull<sys::llama_context>>,
    remote: Option<Box<Remote>>,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    batch: sys::llama_batch,
    batch_cap: usize,
    sampler: *mut sys::llama_sampler,
    /// Tokens whose text carries an em or en dash.
    dash_tokens: Vec<i32>,
    /// The context length the model was trained on (DRY's reach).
    n_ctx_train: i32,
    /// Tokens the engine asked never to sample (a frame's control tokens).
    banned: Vec<i32>,
    // Kept alive for the model, which keeps the pointers.
    _devices: Vec<sys::ggml_backend_dev_t>,
    _pattern: Option<CString>,
    _overrides: Box<[sys::llama_model_tensor_buft_override; 3]>,
    _tensor_split: Vec<f32>,
    pub split: Split,
    pub sizes: split::Sizes,
    /// GPU memory free and total when the plan was made, bytes, summed
    /// over every GPU.
    pub vram: (u64, u64),
    /// Blocks per GPU (llama.cpp's layer split; the last GPU also holds
    /// the output layer). One entry for one GPU.
    pub gpu_layers: Vec<usize>,
    pub opts: Options,
    eog: Vec<i32>,
    /// The callback's state; boxed so its address is stable for llama.cpp.
    capture: Option<Box<Capture>>,
}

// SAFETY: the context is driven by one thread at a time (the engine's);
// llama.cpp's objects are not tied to the thread that made them.
unsafe impl Send for Llm {}

impl Llm {
    /// Register the backends, plan the split from the GPU's free memory,
    /// load the model and make the context.
    pub fn load(opts: Options, sampling: &Sampling) -> Result<Self> {
        VERBOSE.store(opts.verbose, Ordering::Relaxed);
        let dir = CString::new(opts.backend_dir.as_str())?;
        let file = CString::new(opts.model.as_str())?;
        let sizes = split::sizes(&opts.model)?;
        // SAFETY: plain calls into llama.cpp with valid NUL-terminated
        // strings and structures that outlive their use (the model keeps
        // the device list and the overrides: boxed into `Llm`).
        unsafe {
            sys::llama_log_set(Some(log_cb), std::ptr::null_mut());
            sys::ggml_backend_load_all_from_path(dir.as_ptr());
            sys::llama_backend_init();
            // Every CUDA device (CUDA0, CUDA1, ...; PHI_STREAM_GPUS=N keeps
            // the first N), none with --cpu.
            let mut gpus: Vec<sys::ggml_backend_dev_t> = Vec::new();
            if !opts.cpu {
                for i in 0..16 {
                    let name = CString::new(format!("CUDA{i}"))?;
                    let d = sys::ggml_backend_dev_by_name(name.as_ptr());
                    if d.is_null() {
                        break;
                    }
                    gpus.push(d);
                }
                if let Some(k) = std::env::var("PHI_STREAM_GPUS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    gpus.truncate(k.max(1));
                }
                if gpus.is_empty() {
                    bail!("no CUDA0 device: is {} the CUDA build?", opts.backend_dir);
                }
            }
            let mem: Vec<(u64, u64)> = gpus
                .iter()
                .map(|&d| {
                    let (mut f, mut t) = (0usize, 0usize);
                    sys::ggml_backend_dev_memory(d, &mut f, &mut t);
                    (f as u64, t as u64)
                })
                .collect();
            let free: u64 = mem.iter().map(|m| m.0).sum();
            let total: u64 = mem.iter().map(|m| m.1).sum();
            let kv_per_token = if opts.kv_q8 {
                sizes.kv_per_token_f16 * 17 / 32
            } else {
                sizes.kv_per_token_f16
            };
            // Each GPU's own needs: the compute buffers of a cycle and a
            // margin for CUDA's own allocations.
            let per_gpu = opts
                .gpu_headroom
                .unwrap_or(512 * (1 << 20) + opts.batch as u64 * 2 * (1 << 20) + 768 * (1 << 20));
            // The context's: the cells, and every sequence slot's
            // recurrent state (62.8 MiB each for the 35B), which the K and
            // V term does not count; none of it on a GPU with --kv-host
            // (llama.cpp keeps both caches where offload_kqv says).
            let on_gpu_cache = if opts.kv_host { 0 } else { 1 };
            let reserve = on_gpu_cache
                * (kv_per_token * opts.ctx as u64 + sizes.recurrent_per_seq * opts.n_seq as u64)
                + per_gpu * gpus.len().max(1) as u64
                + opts.extra_reserve;
            let budget = free.saturating_sub(reserve);
            // With --cpu no block goes to a GPU (and none is listed below).
            let want = if opts.cpu { Some(0) } else { opts.gpu_blocks };
            let plan = split::plan(&sizes, budget, want, &opts.keep_on_gpu);
            // Which blocks each GPU holds: llama.cpp splits by layer count,
            // so the counts are chosen to balance what each block leaves on
            // a GPU (a block whose experts went to the host is small), in
            // proportion to each GPU's free memory.
            let frees: Vec<u64> = mem.iter().map(|m| m.0).collect();
            let cache = if opts.kv_host {
                Vec::new()
            } else {
                split::block_cache(&sizes, kv_per_token, opts.ctx as u64, opts.n_seq as u64)
            };
            let gpu_layers = split::layer_split(&sizes, &plan, &frees, &cache);
            let mut tensor_split = vec![0f32; sys::llama_max_devices().max(1)];
            for (i, &n) in gpu_layers.iter().enumerate().take(tensor_split.len()) {
                tensor_split[i] = n as f32;
            }

            let mut devices = gpus.clone();
            devices.push(std::ptr::null_mut());
            let pattern = plan.pattern.as_deref().map(CString::new).transpose()?;
            let none = sys::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            };
            // The host blocks' experts, then the host-only tables (the
            // per-layer token embeddings), then the list's end.
            let mut overrides = Box::new([none, none, none]);
            let mut k = 0;
            if let Some(p) = pattern.as_ref().filter(|_| !opts.cpu) {
                overrides[k].pattern = p.as_ptr();
                overrides[k].buft = sys::ggml_backend_cpu_buffer_type();
                k += 1;
            }
            if sizes.host_only > 0 && !opts.cpu {
                overrides[k].pattern = HOST_ONLY.as_ptr();
                overrides[k].buft = sys::ggml_backend_cpu_buffer_type();
            }
            let mut mp = sys::llama_model_default_params();
            mp.devices = devices.as_mut_ptr();
            if gpus.len() > 1 {
                mp.tensor_split = tensor_split.as_ptr();
            }
            mp.tensor_buft_overrides = overrides.as_ptr();
            mp.n_gpu_layers = if opts.cpu { 0 } else { 999 };
            // No repacking: a repacked weight is never offered to the cards.
            mp.use_extra_bufts = false;
            let m = NonNull::new(sys::llama_model_load_from_file(file.as_ptr(), mp))
                .with_context(|| format!("could not load {}", opts.model))?;

            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = opts.ctx;
            cp.n_batch = opts.batch;
            cp.n_ubatch = opts.batch;
            cp.n_seq_max = opts.n_seq;
            cp.n_rs_seq = 0;
            cp.n_threads = opts.threads;
            cp.n_threads_batch = opts.threads;
            cp.flash_attn_type = sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.kv_unified = opts.kv_unified;
            cp.offload_kqv = !opts.kv_host;
            cp.op_offload = opts.op_offload;
            let mut capture = opts.capture.clone().map(|c| Box::new(Capture::new(c)));
            if let Some(c) = capture.as_mut() {
                cp.cb_eval = Some(eval_callback);
                cp.cb_eval_user_data = c.as_mut() as *mut Capture as *mut c_void;
            }
            let kv_type = if opts.kv_q8 {
                sys::ggml_type_GGML_TYPE_Q8_0
            } else {
                sys::ggml_type_GGML_TYPE_F16
            };
            cp.type_k = kv_type;
            cp.type_v = kv_type;
            let c = match NonNull::new(sys::llama_init_from_model(m.as_ptr(), cp)) {
                Some(c) => c,
                None => {
                    sys::llama_model_free(m.as_ptr());
                    bail!("could not make a context of {} cells", opts.ctx);
                }
            };
            let vocab = sys::llama_model_get_vocab(m.as_ptr());
            let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;
            let batch_cap = opts.batch as usize;
            let batch = sys::llama_batch_init(batch_cap as i32, 0, 1);
            let eog = (0..n_vocab as i32)
                .filter(|&t| sys::llama_vocab_is_eog(vocab, t))
                .collect();
            let dash_tokens = dash_tokens(vocab, n_vocab);
            let n_ctx_train = sys::llama_model_n_ctx_train(m.as_ptr());
            let sampler = make_sampler(sampling, &dash_tokens, &[], vocab, n_ctx_train);
            Ok(Self {
                model: Some(m),
                ctx: Some(c),
                remote: None,
                vocab,
                n_vocab,
                batch,
                batch_cap,
                sampler,
                dash_tokens,
                n_ctx_train,
                banned: Vec::new(),
                _devices: devices,
                _pattern: pattern,
                _overrides: overrides,
                _tensor_split: tensor_split,
                split: plan,
                sizes,
                vram: (free, total),
                gpu_layers,
                opts,
                eog,
                capture,
            })
        }
    }

    /// The model served by the llama-server at `url` (`remote.rs`): its
    /// slot `slot` holds the one sequence, the vocabulary is read from
    /// `vocab` (default: the one kept for the server's model) and checked
    /// against the server's. Nothing is loaded here; `opts.model` becomes
    /// the server's model file and `opts.ctx` its slot's context.
    pub fn remote(
        mut opts: Options,
        sampling: &Sampling,
        url: &str,
        vocab: Option<&str>,
        slot: i32,
    ) -> Result<Self> {
        let r = Remote::connect(url, vocab, slot, sampling)?;
        opts.model = r.model.clone();
        opts.ctx = r.n_ctx;
        opts.n_seq = 1;
        opts.capture = None;
        let n_vocab = r.vocab.len();
        let eog = r.vocab.eog.clone();
        let dash_tokens = [
            r.vocab.containing("\u{2014}".as_bytes()),
            r.vocab.containing("\u{2013}".as_bytes()),
        ]
        .concat();
        let n_layer = r.vocab.n_layer.max(0) as usize;
        let mut me = Self {
            model: None,
            ctx: None,
            remote: Some(Box::new(r)),
            vocab: std::ptr::null(),
            n_vocab,
            // SAFETY: a one-token batch, freed in `drop`; never decoded.
            batch: unsafe { sys::llama_batch_init(1, 0, 1) },
            // A whole sequence goes to the server at once.
            batch_cap: opts.ctx as usize,
            sampler: std::ptr::null_mut(),
            dash_tokens,
            n_ctx_train: opts.ctx as i32,
            banned: Vec::new(),
            _devices: Vec::new(),
            _pattern: None,
            _overrides: Box::new(
                [sys::llama_model_tensor_buft_override {
                    pattern: std::ptr::null(),
                    buft: std::ptr::null_mut(),
                }; 3],
            ),
            _tensor_split: Vec::new(),
            split: Split {
                n_blocks: n_layer,
                gpu_blocks: 0,
                gpu_set: Vec::new(),
                gpu_bytes: 0,
                host_bytes: 0,
                pattern: None,
            },
            sizes: split::Sizes {
                block: Vec::new(),
                experts: Vec::new(),
                other: 0,
                host_only: 0,
                kv_per_token_f16: 0,
                recurrent_per_seq: 0,
                cache_kind: Vec::new(),
                arch: String::new(),
            },
            vram: (0, 0),
            gpu_layers: Vec::new(),
            opts,
            eog,
            capture: None,
        };
        me.set_sampling(sampling);
        Ok(me)
    }

    /// Whether sequences beside the live one can be read, forked and
    /// composed (a model in this process); not with `--remote`.
    pub fn forks(&self) -> bool {
        self.remote.is_none()
    }

    /// With `--remote`: where the model runs and its name, the streams the
    /// server was asked for and the tokens they gave.
    pub fn remote_info(&self) -> Option<(String, String, u64, u64)> {
        self.remote
            .as_ref()
            .map(|r| (r.url(), r.vocab.name.clone(), r.streams, r.tokens))
    }

    /// With `--remote`: `tokens` read on the server's prefill engine beside
    /// the slot's stream (`Remote::prefetch`); an error with a model here,
    /// whose readings beside the live sequence need no server.
    pub fn prefetch(&self, tokens: &[i32]) -> Result<crate::remote::PrefetchStart> {
        match &self.remote {
            Some(r) => r.prefetch(tokens),
            None => bail!("no prefetch: the model is in this process"),
        }
    }

    /// With `--remote`: the state of the prefetch `id`.
    pub fn prefetch_state(&self, id: u64) -> Result<crate::remote::Prefetch> {
        match &self.remote {
            Some(r) => r.prefetch_state(id),
            None => bail!("no prefetch: the model is in this process"),
        }
    }

    fn model_ptr(&self) -> *mut sys::llama_model {
        self.model.expect("a model in this process").as_ptr()
    }

    fn ctx_ptr(&self) -> *mut sys::llama_context {
        self.ctx.expect("a context in this process").as_ptr()
    }

    /// The capture, when one is installed.
    pub fn capture(&mut self) -> Option<&mut Capture> {
        self.capture.as_deref_mut()
    }

    /// The number of decoder blocks the main pass runs. llama.cpp's count
    /// already leaves out the extra prediction block (`hparams.n_layer()`
    /// against `n_layer_all`, llama-hparams.cpp): 40 for this model, whose
    /// file holds 41.
    pub fn n_layer(&self) -> i32 {
        if let Some(r) = &self.remote {
            return r.vocab.n_layer;
        }
        // SAFETY: a plain query of the model.
        unsafe { sys::llama_model_n_layer(self.model_ptr()) }
    }

    pub fn n_ctx(&self) -> u32 {
        if let Some(r) = &self.remote {
            return r.n_ctx;
        }
        // SAFETY: a plain query of the context.
        unsafe { sys::llama_n_ctx(self.ctx_ptr()) }
    }

    pub fn batch_cap(&self) -> usize {
        self.batch_cap
    }

    /// Sequences the context holds apart (`--n-seq`).
    pub fn n_seq(&self) -> u32 {
        if self.remote.is_some() {
            return 1;
        }
        // SAFETY: a plain query of the context.
        unsafe { sys::llama_n_seq_max(self.ctx_ptr()) }
    }

    /// The vocabulary's tokens whose text holds `needle` (a scan of every
    /// token's piece, as for the dashes).
    pub fn tokens_containing(&self, needle: &str) -> Vec<i32> {
        if let Some(r) = &self.remote {
            return r.vocab.containing(needle.as_bytes());
        }
        tokens_with(self.vocab, self.n_vocab, needle.as_bytes())
    }

    /// How many tokens the sampler never draws for carrying a dash.
    pub fn dash_tokens_banned(&self) -> usize {
        self.dash_tokens.len()
    }

    /// Never sample these tokens (a frame's control tokens); the chain is
    /// rebuilt with the current sampling.
    pub fn ban_tokens(&mut self, tokens: &[i32], s: &Sampling) {
        self.banned.extend_from_slice(tokens);
        self.set_sampling(s);
    }

    /// The banned tokens replaced (a gate opened or closed while it runs);
    /// the sampler rebuilt with `recent` accepted, so its penalties stand.
    pub fn set_banned(&mut self, tokens: &[i32], s: &Sampling, recent: &[i32]) {
        self.banned = tokens.to_vec();
        self.reset_sampler(s, s.seed, recent);
    }

    /// Text to tokens; `special` parses the template's control tokens.
    pub fn tokenize(&self, text: &str, special: bool) -> Result<Vec<i32>> {
        if let Some(r) = &self.remote {
            return r.tokenize(text, special);
        }
        let mut out = vec![0i32; text.len() + 8];
        for _ in 0..2 {
            // SAFETY: `out` has room for `out.len()` tokens.
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    out.as_mut_ptr(),
                    out.len() as i32,
                    false,
                    special,
                )
            };
            if n >= 0 {
                out.truncate(n as usize);
                return Ok(out);
            }
            out.resize((-n) as usize, 0);
        }
        bail!("tokenize failed")
    }

    /// The one token `text` is, parsed as a control token, if it is one.
    pub fn special(&self, text: &str) -> Option<i32> {
        match self.tokenize(text, true) {
            Ok(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// A token's bytes (control tokens included when `special`).
    pub fn piece(&self, token: i32, special: bool, out: &mut Vec<u8>) {
        if let Some(r) = &self.remote {
            return r.vocab.piece(token, special, out);
        }
        let mut buf = [0u8; 256];
        // SAFETY: `buf` holds 256 bytes.
        let n = unsafe {
            sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                special,
            )
        };
        if n > 0 {
            out.extend_from_slice(&buf[..n as usize]);
        } else if n < 0 {
            let mut big = vec![0u8; (-n) as usize];
            // SAFETY: `big` has the length llama.cpp asked for.
            let m = unsafe {
                sys::llama_token_to_piece(
                    self.vocab,
                    token,
                    big.as_mut_ptr() as *mut c_char,
                    big.len() as i32,
                    0,
                    special,
                )
            };
            if m > 0 {
                out.extend_from_slice(&big[..m as usize]);
            }
        }
    }

    /// Tokens as text, control tokens left out (what a program sees).
    pub fn text_plain(&self, tokens: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            self.piece(t, false, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Tokens as text, control tokens shown.
    pub fn text(&self, tokens: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            self.piece(t, true, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn is_eog(&self, token: i32) -> bool {
        self.eog.contains(&token)
    }

    pub fn eot(&self) -> i32 {
        if let Some(r) = &self.remote {
            return r.vocab.eot;
        }
        // SAFETY: a plain query of the vocabulary.
        unsafe { sys::llama_vocab_eot(self.vocab) }
    }

    /// One `llama_decode` over the lanes (at most `batch` tokens in all).
    /// Returns, for each lane that asked, the batch row its last token's
    /// logits are read from, in lane order.
    pub fn decode(&mut self, lanes: &[Lane]) -> Result<Vec<i32>> {
        let n: usize = lanes.iter().map(|l| l.tokens.len()).sum();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Remote: the tokens are kept in order; the server decodes them with
        // the next stream (`sample`). Its one row stands for "the last".
        if let Some(r) = self.remote.as_mut() {
            let mut rows = Vec::new();
            for l in lanes {
                r.place(l.seq, l.tokens, l.pos0)?;
                if l.logits {
                    rows.push(0);
                }
            }
            return Ok(rows);
        }
        if n > self.batch_cap {
            bail!(
                "a cycle of {n} tokens exceeds the batch of {}",
                self.batch_cap
            );
        }
        let b = &mut self.batch;
        b.n_tokens = n as i32;
        let mut rows = Vec::new();
        let mut i = 0usize;
        for l in lanes {
            for (j, &t) in l.tokens.iter().enumerate() {
                let last = j + 1 == l.tokens.len();
                // SAFETY: the batch was made for `batch_cap` tokens, one
                // sequence id each.
                unsafe {
                    *b.token.add(i) = t;
                    *b.pos.add(i) = l.pos0 + j as i32;
                    *b.n_seq_id.add(i) = 1;
                    *(*b.seq_id.add(i)) = l.seq;
                    *b.logits.add(i) = (l.logits && last) as i8;
                }
                if l.logits && last {
                    rows.push(i as i32);
                }
                i += 1;
            }
        }
        // SAFETY: the batch is filled for its `n_tokens`.
        let r = unsafe { sys::llama_decode(self.ctx_ptr(), self.batch) };
        if r != 0 {
            bail!("llama_decode failed ({r})");
        }
        Ok(rows)
    }

    /// The logits of batch row `row` of the last decode.
    pub fn logits(&self, row: i32) -> Result<&[f32]> {
        if self.remote.is_some() {
            bail!("no logits: the model is served remotely (--remote) and samples on the server");
        }
        // SAFETY: the row asked for logits in the last decode: n_vocab floats.
        unsafe {
            let p = sys::llama_get_logits_ith(self.ctx_ptr(), row);
            if p.is_null() {
                bail!("no logits for row {row}");
            }
            Ok(std::slice::from_raw_parts(p, self.n_vocab))
        }
    }

    /// The sampler chain's choice for batch row `row` (accepted into its
    /// own history), with a pump for the wait behind a remote server's
    /// first token (`Remote::sample_pumped`); the in-process chain answers
    /// at once, so its pump is never called.
    pub fn sample_pumped(
        &mut self,
        row: i32,
        pump: &mut dyn FnMut(Option<(usize, usize)>) -> Result<bool>,
    ) -> Result<i32> {
        if let Some(r) = self.remote.as_mut() {
            return r.sample_pumped(pump);
        }
        let _ = pump;
        // SAFETY: the row asked for logits in the last decode.
        Ok(unsafe { sys::llama_sampler_sample(self.sampler, self.ctx_ptr(), row) })
    }

    /// The next token from logits given here (one per vocabulary entry)
    /// through the same chain as `sample` (its bans, penalties, DRY, its
    /// random state), accepted once: what `llama_sampler_sample` does with a
    /// row of the context (llama-sampling.cpp), for a row the engine made
    /// (the guide's mix).
    pub fn sample_logits(&mut self, logits: &[f32]) -> Result<i32> {
        if self.remote.is_some() {
            bail!("no sampling from given logits: the model is served remotely (--remote)");
        }
        let mut data: Vec<sys::llama_token_data> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| sys::llama_token_data {
                id: i as i32,
                logit: l,
                p: 0.0,
            })
            .collect();
        let mut arr = sys::llama_token_data_array {
            data: data.as_mut_ptr(),
            size: data.len(),
            selected: -1,
            sorted: false,
        };
        // SAFETY: the array points at `data`, alive for the call; the chain
        // works in place within its size and sets `selected` within it.
        unsafe {
            sys::llama_sampler_apply(self.sampler, &mut arr);
            let t = if arr.selected >= 0 && (arr.selected as usize) < arr.size {
                (*arr.data.add(arr.selected as usize)).id
            } else {
                // No selection (a chain without a final pick): the likeliest.
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i as i32)
            };
            sys::llama_sampler_accept(self.sampler, t);
            Ok(t)
        }
    }

    /// Replace the chain (new temperature or seed).
    pub fn set_sampling(&mut self, s: &Sampling) {
        if let Some(r) = self.remote.as_mut() {
            let mut never = self.banned.clone();
            if s.ban_dashes {
                never.extend_from_slice(&self.dash_tokens);
            }
            r.set_sampling(s, &never);
            return;
        }
        // SAFETY: the old chain is freed once, the new one made once.
        unsafe {
            sys::llama_sampler_free(self.sampler);
            self.sampler = make_sampler(
                s,
                &self.dash_tokens,
                &self.banned,
                self.vocab,
                self.n_ctx_train,
            );
        }
    }

    /// The greedy choice for batch row `row`, end-of-generation tokens
    /// skipped when `ban_eog`.
    pub fn greedy(&self, row: i32, ban_eog: bool) -> Result<i32> {
        let v = self.logits(row)?;
        Ok(if ban_eog {
            argmax_without(v, &self.eog)
        } else {
            argmax(v)
        })
    }

    fn mem(&self) -> sys::llama_memory_t {
        // SAFETY: a plain query of the context.
        unsafe { sys::llama_get_memory(self.ctx_ptr()) }
    }

    /// Give `dst` the cells of `src` in `[p0, p1)` (`-1`: open) and
    /// `src`'s recurrent state.
    /// Remote: nothing is copied (one sequence), and a decode into `dst`
    /// is refused there.
    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        if self.remote.is_some() {
            return;
        }
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_seq_cp(self.mem(), src, dst, p0, p1) }
    }

    /// Drop `seq`'s cells in `[p0, p1)`; false when a partial range cannot go.
    pub fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
        if let Some(r) = self.remote.as_mut() {
            return r.remove(seq, p0, p1);
        }
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_seq_rm(self.mem(), seq, p0, p1) }
    }

    /// The highest position `seq` holds; -1 when it holds none.
    pub fn seq_pos_max(&self, seq: i32) -> i32 {
        if let Some(r) = &self.remote {
            return r.pos_max(seq);
        }
        // SAFETY: a plain query of this context's memory.
        unsafe { sys::llama_memory_seq_pos_max(self.mem(), seq) }
    }

    /// Put `token` into the sampler chain's history as if the chain had
    /// chosen it (a forced token: the penalties count it).
    pub fn accept(&mut self, token: i32) {
        // Remote: the server's sampler counts the prompt it is sent.
        if self.remote.is_some() {
            return;
        }
        // SAFETY: the chain lives as long as `self`.
        unsafe { sys::llama_sampler_accept(self.sampler, token) }
    }

    /// A fresh chain with seed `seed` whose history is `recent` (after a
    /// rewind the tokens taken back no longer count toward the penalties).
    pub fn reset_sampler(&mut self, s: &Sampling, seed: u32, recent: &[i32]) {
        let s = Sampling { seed, ..s.clone() };
        self.set_sampling(&s);
        let n = if s.repeat_last_n < 0 {
            recent.len()
        } else {
            recent.len().min(s.repeat_last_n as usize)
        };
        for &t in &recent[recent.len() - n..] {
            self.accept(t);
        }
    }

    pub fn clear(&mut self) {
        if let Some(r) = self.remote.as_mut() {
            return r.clear();
        }
        // SAFETY: plain calls on this context's memory.
        unsafe { sys::llama_memory_clear(self.mem(), true) }
    }
}

/// The vocabulary's tokens whose text carries U+2014 or U+2013.
fn dash_tokens(vocab: *const sys::llama_vocab, n_vocab: usize) -> Vec<i32> {
    let mut out = Vec::new();
    let mut buf = [0u8; 64];
    for t in 0..n_vocab as i32 {
        // SAFETY: `buf` holds 64 bytes; a longer piece is cut, which is
        // fine for a test of its bytes.
        let n = unsafe {
            sys::llama_token_to_piece(
                vocab,
                t,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                false,
            )
        };
        let n = if n < 0 { buf.len() } else { n as usize };
        let s = &buf[..n.min(buf.len())];
        if s.windows(3)
            .any(|w| w == "\u{2014}".as_bytes() || w == "\u{2013}".as_bytes())
        {
            out.push(t);
        }
    }
    out
}

unsafe fn make_sampler(
    s: &Sampling,
    dashes: &[i32],
    banned: &[i32],
    vocab: *const sys::llama_vocab,
    n_ctx_train: i32,
) -> *mut sys::llama_sampler {
    let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
    let mut never: Vec<i32> = banned.to_vec();
    if s.ban_dashes {
        never.extend_from_slice(dashes);
    }
    if !never.is_empty() {
        let biases: Vec<sys::llama_logit_bias> = never
            .iter()
            .map(|&token| sys::llama_logit_bias {
                token,
                bias: f32::NEG_INFINITY,
            })
            .collect();
        sys::llama_sampler_chain_add(
            chain,
            sys::llama_sampler_init_logit_bias(0, biases.len() as i32, biases.as_ptr()),
        );
    }
    // A window of -1 is the whole trained context, resolved here: since
    // llama.cpp a6aa6f545 (2026-08-04) the samplers clamp a negative
    // window to 0, which turns them off, and take no context length
    // (build.rs sets llama_samplers_v2 for that API).
    let window = |n: i32| if n < 0 { n_ctx_train } else { n };
    if s.repeat_penalty > 1.0 && s.repeat_last_n != 0 {
        #[cfg(llama_samplers_v2)]
        let penalties = sys::llama_sampler_init_penalties(
            sys::llama_vocab_n_tokens(vocab),
            window(s.repeat_last_n),
            s.repeat_penalty,
            0.0,
            0.0,
        );
        #[cfg(not(llama_samplers_v2))]
        let penalties =
            sys::llama_sampler_init_penalties(window(s.repeat_last_n), s.repeat_penalty, 0.0, 0.0);
        sys::llama_sampler_chain_add(chain, penalties);
    }
    // Repeated sequences penalized (llama.cpp's order: after the
    // penalties, before top-k), with its usual breakers but the newline:
    // with it, a line written again and again ("```" on every line, on the
    // live service) was never a repeat to DRY.
    if s.dry_multiplier > 0.0 {
        let breakers: Vec<std::ffi::CString> = [":", "\"", "*"]
            .iter()
            .map(|b| std::ffi::CString::new(*b).unwrap())
            .collect();
        let ptrs: Vec<*const c_char> = breakers.iter().map(|b| b.as_ptr()).collect();
        #[cfg(llama_samplers_v2)]
        let dry = sys::llama_sampler_init_dry(
            vocab,
            s.dry_multiplier,
            s.dry_base,
            s.dry_allowed_length,
            window(s.dry_last_n),
            ptrs.as_ptr() as *mut *const c_char,
            ptrs.len(),
        );
        #[cfg(not(llama_samplers_v2))]
        let dry = sys::llama_sampler_init_dry(
            vocab,
            n_ctx_train,
            s.dry_multiplier,
            s.dry_base,
            s.dry_allowed_length,
            s.dry_last_n,
            ptrs.as_ptr() as *mut *const c_char,
            ptrs.len(),
        );
        sys::llama_sampler_chain_add(chain, dry);
    }
    if s.top_k > 0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_k(s.top_k));
    }
    if s.top_p < 1.0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_p(s.top_p, 1));
    }
    if s.min_p > 0.0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_min_p(s.min_p, 1));
    }
    if s.temp > 0.0 {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_temp(s.temp));
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_dist(s.seed));
    } else {
        sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_greedy());
    }
    chain
}

// `dash_tokens` scans the vocabulary once for tokens whose text carries
// U+2014 or U+2013; `make_sampler` puts a logit bias of minus infinity
// on every one of them at the head of the chain when `ban_dashes` is
// set, so the rule the persona states is also enforced by the sampler.

impl Drop for Llm {
    fn drop(&mut self) {
        // SAFETY: each was made once in `load` (or `remote`: the batch
        // alone) and is freed once here.
        unsafe {
            if !self.sampler.is_null() {
                sys::llama_sampler_free(self.sampler);
            }
            sys::llama_batch_free(self.batch);
            if let Some(c) = self.ctx {
                sys::llama_free(c.as_ptr());
            }
            if let Some(m) = self.model {
                sys::llama_model_free(m.as_ptr());
            }
        }
    }
}

/// `argmax` over the tokens not in `ban` (a short list).
pub fn argmax_without(v: &[f32], ban: &[i32]) -> i32 {
    let best = argmax(v);
    if !ban.contains(&best) {
        return best;
    }
    let mut best = -1;
    let mut top = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > top && !ban.contains(&(i as i32)) {
            top = x;
            best = i as i32;
        }
    }
    best
}

/// The index of the largest value (the first of equals, as llama.cpp's
/// greedy sampler takes it).
pub fn argmax(v: &[f32]) -> i32 {
    let mut best = 0;
    let mut top = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > top {
            top = x;
            best = i;
        }
    }
    best as i32
}

/// The vocabulary's tokens whose text holds `needle`.
fn tokens_with(vocab: *const sys::llama_vocab, n_vocab: usize, needle: &[u8]) -> Vec<i32> {
    let mut out = Vec::new();
    let mut buf = [0u8; 64];
    for t in 0..n_vocab as i32 {
        // SAFETY: `buf` holds 64 bytes; a longer piece is cut, which is
        // fine for a test of its bytes.
        let n = unsafe {
            sys::llama_token_to_piece(
                vocab,
                t,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                false,
            )
        };
        let n = if n < 0 { buf.len() } else { n as usize };
        if buf[..n.min(buf.len())]
            .windows(needle.len())
            .any(|w| w == needle)
        {
            out.push(t);
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_the_first_of_equals() {
        assert_eq!(argmax(&[0.0, 3.0, 1.0, 3.0]), 1);
        assert_eq!(argmax_without(&[0.0, 3.0, 1.0, 3.0], &[1]), 3);
        assert_eq!(argmax_without(&[0.0, 3.0, 1.0, 2.0], &[1, 3]), 2);
    }
}
