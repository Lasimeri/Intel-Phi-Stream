//! `phi-stream`: one model split over the GPU, the cards and host memory,
//! thinking without pause while it reads what it is given, over llama.cpp
//! used as a library and never changed. The model is owned by a service
//! (`serve`); the terminal and the one-shot commands are its clients. See
//! main.md.

mod agent;
mod capture;
mod check;
mod client;
mod clock;
mod code;
mod engine;
mod eval;
mod feeds;
mod format;
mod gate;
mod improve;
mod lens;
mod llm;
mod mcp;
mod mind;
mod playout;
mod probe;
mod readout;
mod reflect;
mod remote;
mod rotlog;
mod screen;
mod serve;
mod situation;
mod split;
mod sys;
mod term;
mod torch;
mod tui;
mod verify;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context as _, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::client::{default_socket, escape, parse, Client, Msg};
use crate::engine::{compose, Command, Config, Engine, Event, Frame, Kind, DEFAULT_BASE};
use crate::llm::{Llm, Options, Sampling};

#[derive(Parser)]
#[command(
    name = "phi-stream",
    about = "A continuous thought stream over llama.cpp: the GPU, the cards and host memory together"
)]
struct Cli {
    #[command(flatten)]
    model: ModelArgs,
    /// The service's socket (also PHI_STREAM_SOCKET; default under XDG_RUNTIME_DIR).
    #[arg(long, global = true)]
    socket: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// The model file (gguf).
    #[arg(
        global = true,
        short = 'm',
        long,
        env = "PHI_STREAM_MODEL",
        default_value = "~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q6_K.gguf"
    )]
    model: String,
    /// Where llama.cpp's backends are (the CUDA build's bin).
    #[arg(global = true, long, env = "PHI_STREAM_BACKEND_DIR", default_value = env!("PHI_STREAM_LLAMA_BUILD_DIR"))]
    backend_dir: String,
    /// Cells of the context, shared by every sequence.
    #[arg(global = true, short = 'c', long, default_value_t = 32768)]
    ctx: u32,
    /// The most tokens a cycle carries (the live token and a chunk).
    #[arg(global = true, long, default_value_t = 129)]
    batch: u32,
    /// Blocks whose experts stay on the GPU (default: as many as fit).
    #[arg(global = true, long)]
    gpu_blocks: Option<usize>,
    /// Give the GPU the experts of the blocks the mind reads first (reading a
    /// block whose experts run on the host or the cards costs the stream).
    #[arg(global = true, long)]
    read_blocks_on_gpu: bool,
    /// K and V in 8-bit blocks (half the cells' bytes).
    #[arg(global = true, long)]
    kv_q8: bool,
    /// The KV cache (and the attention over it) in host memory, not on the
    /// GPUs: a longer context, and the GPUs' memory for weights.
    #[arg(global = true, long)]
    kv_host: bool,
    /// A batch over host weights computed where they are (the host and the
    /// cards: the prompts on the cards), not copied to a GPU per batch.
    #[arg(global = true, long)]
    no_op_offload: bool,
    /// Each GPU's room for llama.cpp's working buffers (the compute buffers
    /// and the pool), GiB; default: 1.25 plus 2 MiB per batch token.
    #[arg(global = true, long)]
    gpu_headroom: Option<f64>,
    /// No GPU: the model on the host and the cards, leaving the GPU to another
    /// model (a second instance, src/llm.md).
    #[arg(global = true, long)]
    cpu: bool,
    /// Threads for llama.cpp's own CPU work (the cards' backend has its pool).
    #[arg(global = true, short = 't', long, default_value_t = 8)]
    threads: i32,
    /// Sequences the context holds apart.
    #[arg(global = true, long, default_value_t = 4)]
    n_seq: u32,
    /// A KV stream per sequence instead of one unified pool (measurement only).
    #[arg(global = true, long)]
    kv_split: bool,
    /// Sampling temperature (0: greedy).
    #[arg(global = true, long, default_value_t = 1.0)]
    temp: f32,
    #[arg(global = true, long, default_value_t = 20)]
    top_k: i32,
    #[arg(global = true, long, default_value_t = 0.95)]
    top_p: f32,
    /// Tokens under this share of the likeliest one's probability dropped (0: off).
    #[arg(global = true, long, default_value_t = 0.0)]
    min_p: f32,
    /// DRY against repeated sequences (llama.cpp's sampler): its multiplier (0: off).
    #[arg(global = true, long, default_value_t = 0.0)]
    dry_multiplier: f32,
    #[arg(global = true, long, default_value_t = 1.75)]
    dry_base: f32,
    #[arg(global = true, long, default_value_t = 2)]
    dry_allowed_length: i32,
    /// How far back DRY looks (-1: the whole context).
    #[arg(global = true, long, default_value_t = -1)]
    dry_last_n: i32,
    /// The sampler's seed (default: from the clock, so each start is its own).
    #[arg(global = true, long)]
    seed: Option<u32>,
    /// Tokens seen again in the last --repeat-last-n are divided by this (1: off).
    #[arg(global = true, long, default_value_t = 1.05)]
    repeat_penalty: f32,
    #[arg(global = true, long, default_value_t = 256)]
    repeat_last_n: i32,
    /// Let em and en dashes through (by default no token carrying one is ever sampled).
    #[arg(global = true, long)]
    allow_dashes: bool,
    /// Show llama.cpp's informational log.
    #[arg(global = true, short = 'v', long)]
    verbose: bool,
    /// The model served by a llama-server elsewhere instead of one loaded
    /// here (http://HOST:PORT; the GPU rack's is http://192.168.0.39:8001):
    /// its slot holds the one sequence and samples it (src/remote.md). The
    /// mind, the checks, the second chain, the guide and the goal probe need
    /// the model's state in this process and are off.
    #[arg(global = true, long, env = "PHI_STREAM_REMOTE")]
    remote: Option<String>,
    /// The remote model's vocabulary, its GGUF metadata without the weights
    /// (default: ~/.local/share/phi-stream/remote/STEM.vocab.gguf for the
    /// server's model; scripts/remote-vocab.sh fetches it).
    #[arg(global = true, long, env = "PHI_STREAM_REMOTE_VOCAB")]
    remote_vocab: Option<String>,
    /// The server's slot the sequence is kept in (pinned, so another client
    /// does not evict its cache).
    #[arg(
        global = true,
        long,
        env = "PHI_STREAM_REMOTE_SLOT",
        default_value_t = 1
    )]
    remote_slot: i32,
}

#[derive(Clone, Copy, ValueEnum)]
enum FrameArg {
    /// One continuous first-person text, no turns; « from outside, » said aloud.
    Journal,
    /// The model's chat template: thoughts in <think>, speech after.
    Chat,
    /// The chat template with the model's own tool calls (src/agent.md): it
    /// reasons in <think>, then calls its tools; their results come back before
    /// its next turn.
    Agent,
}

#[derive(Args, Clone)]
struct StreamArgs {
    /// How the text is framed.
    #[arg(long, value_enum, default_value_t = FrameArg::Journal)]
    frame: FrameArg,
    /// Where the persona, the notes and the log live.
    #[arg(
        long,
        env = "PHI_STREAM_WORKSPACE",
        default_value = "~/.local/share/phi-stream"
    )]
    workspace: String,
    /// The base of the personality: a person's standing instructions (default: ~/CLAUDE.md when it exists).
    #[arg(long, env = "PHI_STREAM_PERSONALITY")]
    personality: Option<String>,
    /// The whole persona verbatim, composed with nothing (an experiment's override).
    #[arg(long)]
    system: Option<String>,
    /// The first thing from outside (default: the frame's own).
    #[arg(long)]
    seed_text: Option<String>,
    /// The journal's first words in its own voice, after the seed.
    #[arg(long, default_value = "Where was I. ")]
    first_words: String,
    /// Said things up to this many tokens are heard at once.
    #[arg(long, default_value_t = 48)]
    direct_max: usize,
    /// Tokens read beside the live token each cycle; 0 adapts.
    #[arg(long, default_value_t = 0)]
    chunk: usize,
    /// Roll the context over past this share of it.
    #[arg(long, default_value_t = 0.6)]
    rollover_at: f32,
    /// Roll over past this many cells instead of the share (the stream is told
    /// its memory is nearly full only then).
    #[arg(long)]
    rollover_tokens: Option<usize>,
    /// Put the wall clock into the chain after this many seconds with
    /// nothing from outside (0: never).
    #[arg(long, default_value_t = 60.0)]
    time_every: f64,
    /// Nudge circling thoughts at most once in this many seconds.
    #[arg(long, default_value_t = 60.0)]
    nudge_every: f64,
    /// Show the text this many seconds behind its placement, at an even pace
    /// (the playout absorbs the placement's jumps; 0: as placed; default 0,
    /// or 1 with --reflect).
    #[arg(long)]
    horizon: Option<f64>,
    /// Hand over a file at the start.
    #[arg(long)]
    feed: Option<String>,
    /// Develop the repository at REPO with Claude (docs/dev.md): the persona says
    /// so, [read: PATH] resolves there, [prefer: ...] lines are kept.
    #[arg(long, value_name = "REPO")]
    dev: Option<String>,
    /// A terminal for the stream (src/term.md): `[run: COMMAND]` lines run in a
    /// sandbox (the repository read-only, the workspace read-write, no
    /// network, one CPU at the lowest priority), their output handed back.
    #[arg(long)]
    terminal: bool,
    /// Do not hold its output until it has an objective (by default it only
    /// thinks until one is given: `phi-stream objective TEXT`).
    #[arg(long)]
    no_objective_gate: bool,
    /// The second chain (src/engine.md): beside the live token, a reflection on
    /// each line it ends, given the J-space words the line had on its mind;
    /// the reflection joins the journal at a later line.
    #[arg(long)]
    second_chain: bool,
    /// The second chain against each line instead (src/engine.md): with an
    /// objective set, it argues against the line as a step toward it, and
    /// the main chain is told to answer; implies --second-chain.
    #[arg(long)]
    chain_against: bool,
    /// The second chain audits each thinking token instead (src/engine.md):
    /// the line's tokens, each with what was on the stream's mind as it was
    /// chosen (its J-space reading), checked against the objective; implies
    /// --chain-against.
    #[arg(long)]
    chain_audit: bool,
    /// The goal probe (src/engine.md): at most every 30 s, at a thinking
    /// line's end, whether it serves the objective, as P(yes), in goal.log.
    #[arg(long)]
    goal_probe: bool,
    /// The guide lane (src/engine.md): beside every thinking token, the
    /// distribution with the last reflection in mind, measured against the
    /// live one (shadow; takes a fifth sequence, --n-seq 5 is implied).
    #[arg(long)]
    guide: bool,
    /// The self-improvement loop (src/improve.md): with --dev and --frame
    /// agent, the propose tool; a change from its working copy is built and
    /// tested in a sandbox (make check), the outcome told to it and kept in
    /// improve.log; one that passes goes to Claude for review.
    #[arg(long)]
    improve: bool,
    #[command(flatten)]
    mind: MindArgs,
}

/// The mind and the reflection loop, the same for the service and for
/// `code stream` (so a measurement runs what the stream runs).
#[derive(Args, Clone)]
struct MindArgs {
    /// Read what is on its mind at every token it places, through the
    /// Jacobian lens (mind.md); off: no eval callback is installed.
    #[arg(long)]
    mind: bool,
    /// The lens (scripts/fetch-lens.sh makes it).
    #[arg(long, default_value = "~/models/jlens/qwen3.6-35B-A3B/lens.jlens")]
    lens: String,
    /// The blocks read through the lens.
    #[arg(long, value_delimiter = ',', default_value = "27,29,31")]
    mind_layers: Vec<i32>,
    /// Words shown per block.
    #[arg(long, default_value_t = 6)]
    mind_k: usize,
    /// Check the tokens it places: where it doubts a word or its mind
    /// lights words of error, a deliberation beside the live token decides
    /// keep or write another, and a change rewinds onto a copy made before
    /// the token (reflect.md; needs --mind; sets --horizon 1 unless given).
    #[arg(long)]
    reflect: bool,
    /// As --reflect, read-only: every deliberation runs, no token changes.
    #[arg(long, conflicts_with = "reflect")]
    reflect_dry: bool,
    /// A check keeps when keep's share of its choice is at least this (0.45,
    /// the stream's own choice: a write needs 0.55; above 1 every check
    /// writes: a test of the rewind, reflect.md).
    #[arg(long, default_value_t = 0.45)]
    reflect_keep_at: f32,
}

impl MindArgs {
    /// The mind's and the loop's settings.
    fn configs(&self) -> Result<(Option<mind::MindConfig>, Option<reflect::ReflectConfig>)> {
        let reflecting = self.reflect || self.reflect_dry;
        if reflecting && !self.mind {
            anyhow::bail!("--reflect reads the mind: add --mind");
        }
        let m = self.mind.then(|| mind::MindConfig {
            lens: expand_home(&self.lens),
            layers: self.mind_layers.clone(),
            k: self.mind_k,
            final_block: None,
        });
        let r = reflecting.then(|| reflect::ReflectConfig {
            dry: self.reflect_dry,
            keep_at: self.reflect_keep_at,
            ..Default::default()
        });
        Ok((m, r))
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Own the model and listen on the socket; the terminal and the commands below talk to it.
    Serve {
        #[command(flatten)]
        stream: StreamArgs,
    },
    /// The terminal, talking to the service (it waits for one, and reconnects after a restart).
    Tui {
        /// Reload onto each new build of this binary, keeping the view and the line being typed
        /// (`scripts/phi-stream.sh attach --follow`; developing the terminal while using it).
        #[arg(long)]
        follow: bool,
    },
    /// Say something to the stream.
    Say {
        /// Who is speaking (the stream hears the name; default: no name).
        #[arg(long = "as", value_name = "NAME")]
        who: Option<String>,
        text: Vec<String>,
    },
    /// A message from Claude that waits for its answer: the stream's
    /// `tell_claude` that answers it, printed on stdout (the agent frame;
    /// docs/dev.md). With `--spoken`, said as anyone (`--as`) and answered
    /// by its next spoken line instead (the journal and chat frames).
    Ask {
        #[arg(long = "as", value_name = "NAME")]
        who: Option<String>,
        /// Wait for its next spoken line rather than a `tell_claude`.
        #[arg(long)]
        spoken: bool,
        /// Give up after this many seconds.
        #[arg(long, default_value_t = 180)]
        timeout: u64,
        /// Also print its thoughts since the message (on stderr).
        #[arg(long)]
        thoughts: bool,
        text: Vec<String>,
    },
    /// The management interface as an MCP server on stdin and stdout (an
    /// agent reads and types into the real terminal interface, in tmux;
    /// src/mcp.md).
    Mcp,
    /// What it says aloud, notes, prefers, and the checks that changed a
    /// word, one line each as it happens (for a monitor; docs/dev.md).
    Listen,
    /// Hand a file over.
    Feed {
        path: String,
    },
    /// Print the stream as it happens (the recent text first).
    Tail {
        /// Status lines too.
        #[arg(long)]
        status: bool,
        /// The readings of its mind too (when the service reads them).
        #[arg(long)]
        mind: bool,
    },
    /// The last status line.
    Status,
    /// The checks' keep threshold, live: a check keeps at a keep share of
    /// at least P (a write needs 1 - P; reflect.md).
    KeepAt {
        p: f32,
    },
    /// What it works toward (none given: it only thinks; its speech and tool
    /// lines wait). `-` clears it.
    Objective {
        #[arg(required = true, num_args = 1..)]
        text: Vec<String>,
    },
    /// Tokens read beside the live token each cycle (0 adapts).
    Chunk {
        n: usize,
    },
    /// The second chain, live (src/engine.md): on (reflecting on each
    /// line), off, against (arguing against each line as a step toward
    /// the objective, the main chain told to answer it), or audit (each
    /// thinking token of the line checked against what was on its mind as
    /// it was chosen and the objective).
    Chain {
        #[arg(value_parser = ["on", "off", "against", "audit"])]
        state: String,
    },
    /// The goal probe, live (src/engine.md): every 30 s at most, whether
    /// the last thinking line serves the objective, as P(yes), to goal.log.
    Goal {
        #[arg(value_parser = ["on", "off"])]
        state: String,
    },
    /// The guide lane, live (started with --guide; src/engine.md): on or
    /// off, or where its asides come from: chain (the second chain's
    /// reflections), lens (the J-lens words a line does not say) or placebo
    /// (as many of those the line does say), or ab (lens and placebo by
    /// turns, fork by fork)
    Guide {
        #[arg(value_parser = ["on", "off", "chain", "lens", "placebo", "ab"])]
        state: String,
    },
    /// The experts each token is routed to, captured at the mind's blocks,
    /// live (the guide compares the guided token's with the live one's).
    Experts {
        #[arg(value_parser = ["on", "off"])]
        state: String,
    },
    /// The line-loop breaker, live.
    Breaker {
        #[arg(value_parser = ["on", "off"])]
        state: String,
    },
    /// The harness's nudges (circling, tool reminders), live.
    Nudges {
        #[arg(value_parser = ["on", "off"])]
        state: String,
    },
    /// A sampling setting, live: temp, top-k, top-p, min-p, dry or repeat-penalty.
    Set {
        key: String,
        value: f32,
    },
    /// The temperature.
    Temp {
        t: f32,
    },
    /// A new persona from a file; the context rolls over onto it.
    Persona {
        path: String,
    },
    Pause,
    Resume,
    /// Stop the service.
    Quit,
    /// The placement and the rates on this machine (no service).
    Probe {
        /// A prompt file to read (default: a built-in paragraph).
        prompt: Option<String>,
        /// Tokens to generate alone.
        #[arg(long, default_value_t = 64)]
        gen: usize,
        /// Chunk sizes for reading while thinking.
        #[arg(long, value_delimiter = ',', default_value = "8,16,32,64")]
        chunks: Vec<usize>,
        /// Install the eval callback but ask it for nothing (its cost alone).
        #[arg(long)]
        capture_idle: bool,
        /// Capture these blocks and run the per-token readout after every
        /// decode (synthetic transports: the cost, not a lens).
        #[arg(long, value_delimiter = ',')]
        mind_layers: Vec<i32>,
    },
    /// Does composing a reading give the straight sequence's next token? (no service)
    Gate {
        /// Thoughts generated while reading.
        #[arg(long, default_value_t = 48)]
        thoughts: usize,
        #[arg(long, default_value_t = 16)]
        chunk: usize,
        /// Greedy tokens compared after the composition.
        #[arg(long, default_value_t = 32)]
        compare: usize,
        /// Instead: whether a sequence copied from the live one keeps its
        /// state while the live one goes on (gate.md).
        #[arg(long)]
        snapshot: bool,
        /// Instead: whether the reflection loop's lanes (a snapshot and a
        /// deliberation beside the live sequence, kept or rewound) keep
        /// every state, every captured row its own lane's (gate.md).
        #[arg(long)]
        reflect: bool,
        /// Episodes of `--reflect`, kept and rewound in turn.
        #[arg(long, default_value_t = 8)]
        episodes: usize,
    },
    /// Whether the stream writes code that works: MultiPL-E's HumanEval in
    /// Rust, compiled and tested in a sandbox (code.md; no service).
    Code {
        #[command(subcommand)]
        cmd: CodeCmd,
    },
    /// A candidate built by hand, as the propose tool builds one (improve.md;
    /// no service): the files of UPPER over the repository's HEAD, refused
    /// on a denied path, else staged and run through make check in the
    /// sandbox; the outcome printed, the candidate under ~/.cache/phi-stream/improve.
    Improve {
        /// A layer of changed files, laid out as the repository.
        #[arg(long)]
        upper: PathBuf,
        /// One line: what the change does.
        #[arg(long)]
        title: String,
        /// The repository (default: the current directory).
        #[arg(long, default_value = ".")]
        repo: PathBuf,
    },
    /// The lens: its checks and readouts (no service).
    Lens {
        #[command(subcommand)]
        cmd: LensCmd,
    },
    /// The stream on stdout, lines on stdin said to it (no service; for scripts).
    Run {
        #[command(flatten)]
        stream: StreamArgs,
        /// Stop after this many live tokens (0: never).
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
    },
}

#[derive(Subcommand)]
enum CodeCmd {
    /// The benchmark's own protocol: the raw prompt, greedy, stopped at the
    /// dataset's stop sequence; the number comparable with published ones.
    Anchor {
        /// The tasks (scripts/fetch-code-eval.sh makes them).
        #[arg(long, default_value = "~/models/code-eval/humaneval-rs.jsonl")]
        tasks: String,
        /// Only the first N tasks.
        #[arg(long)]
        first: Option<usize>,
        /// Only these tasks, by name.
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// The most tokens a completion may run to.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
    /// The stream's own way: each task through the engine (chat frame,
    /// thinking, greedy), the answer's code extracted by the stated rule.
    Stream {
        #[arg(long, default_value = "~/models/code-eval/humaneval-rs.jsonl")]
        tasks: String,
        #[arg(long)]
        first: Option<usize>,
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// The persona's base: claude-md (~/CLAUDE.md, as the stream runs),
        /// neutral (a built-in paragraph), or a file.
        #[arg(long, default_value = "neutral")]
        base: String,
        /// Thinking tokens before </think> is placed (0: no limit).
        #[arg(long, default_value_t = 2048)]
        think_budget: usize,
        /// The repetition penalty (1: off).
        #[arg(long, default_value_t = 1.0)]
        penalty: f32,
        #[command(flatten)]
        mind: MindArgs,
    },
}

#[derive(Subcommand)]
enum LensCmd {
    /// Gate: the captured residual of the token being placed, decoded by
    /// the readout, reproduces the model's own logits, alone, beside a
    /// reading and at the end of an injection (check.md).
    Check,
    /// Convert the reference implementation's lens.pt to a .jlens file
    /// (lens.md); scripts/fetch-lens.sh fetches and converts.
    Convert {
        /// The torch.save file.
        src: String,
        /// The .jlens file to write.
        out: String,
    },
    /// What a .jlens file holds.
    Info { path: String },
    /// Gate C: the lens against the plain logit lens on the reference's
    /// evaluation sets, scored as the paper scores them (eval.md).
    Eval {
        /// The .jlens file.
        #[arg(long, default_value = "~/models/jlens/qwen3.6-35B-A3B/lens.jlens")]
        lens: String,
        /// The evaluation sets (lens-eval-*.json); default: every set in
        /// ~/models/jlens/eval.
        sets: Vec<String>,
    },
}

fn print_lens(h: &lens::Header, path: &str) {
    println!(
        "{path}: d_model {}, {} blocks ({}..{}), fitted over {} prompts, from sha256 {}",
        h.d_model,
        h.layers.len(),
        h.layers.first().copied().unwrap_or(-1),
        h.layers.last().copied().unwrap_or(-1),
        h.n_prompts,
        h.source_sha256
    );
    println!("||J||_F / sqrt(d) per block:");
    for (l, n) in h.layers.iter().zip(&h.norms) {
        println!("  block {l:>2}: {n:.4}");
    }
}

pub fn expand_home(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => p.to_string(),
    }
}

/// `GGML_BACKEND_PATH` names the cards' backend; when unset, the library
/// beside this binary is used when there is one.
fn backend_path() {
    if std::env::var_os("GGML_BACKEND_PATH").is_some() {
        return;
    }
    if let Ok(exe) = std::env::current_exe() {
        let lib = exe.with_file_name("libggml_phi.so");
        if lib.is_file() {
            std::env::set_var("GGML_BACKEND_PATH", &lib);
        }
    }
}

fn sampling(m: &ModelArgs) -> Sampling {
    Sampling {
        temp: m.temp,
        top_k: m.top_k,
        top_p: m.top_p,
        min_p: m.min_p,
        dry_multiplier: m.dry_multiplier,
        dry_base: m.dry_base,
        dry_allowed_length: m.dry_allowed_length,
        dry_last_n: m.dry_last_n,
        seed: m.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as u32)
                .unwrap_or(42)
        }),
        repeat_penalty: m.repeat_penalty,
        repeat_last_n: m.repeat_last_n,
        ban_dashes: !m.allow_dashes,
    }
}

fn load(m: &ModelArgs) -> Result<Llm> {
    load_with(m, None, 0)
}

/// GPU memory the readout needs beside the model: its own backend's
/// buffers and the logits of a few columns (`readout.md`).
const READOUT_RESERVE: u64 = 256 << 20;

/// The model with a capture installed (`capture.md`) and `extra` bytes of
/// GPU memory kept free for the readout.
fn load_with(m: &ModelArgs, capture: Option<capture::CaptureConfig>, extra: u64) -> Result<Llm> {
    if m.remote.is_none() {
        backend_path();
    }
    let opts = Options {
        model: expand_home(&m.model),
        backend_dir: m.backend_dir.clone(),
        ctx: m.ctx,
        batch: m.batch,
        threads: m.threads,
        gpu_blocks: m.gpu_blocks,
        keep_on_gpu: if m.read_blocks_on_gpu {
            capture
                .as_ref()
                .map(|c| {
                    c.layers
                        .iter()
                        .filter(|&&l| l >= 0)
                        .map(|&l| l as usize)
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        },
        kv_q8: m.kv_q8,
        kv_host: m.kv_host,
        op_offload: !m.no_op_offload,
        gpu_headroom: m.gpu_headroom.map(|g| (g * (1u64 << 30) as f64) as u64),
        cpu: m.cpu,
        n_seq: m.n_seq,
        kv_unified: !m.kv_split,
        verbose: m.verbose,
        capture,
        extra_reserve: extra,
    };
    match &m.remote {
        Some(url) => Llm::remote(
            opts,
            &sampling(m),
            url,
            m.remote_vocab.as_deref().map(expand_home).as_deref(),
            m.remote_slot,
        ),
        None => Llm::load(opts, &sampling(m)),
    }
}

/// With a remote model (`--remote`) what needs the model's state in this
/// process is turned off, each said once on stderr: the mind and the
/// checks on it, the second chain, the guide, the goal probe.
fn remote_off(cfg: &mut Config) {
    let mut off = Vec::new();
    if cfg.mind.take().is_some() {
        off.push("the mind (--mind)");
    }
    if cfg.reflect.take().is_some() {
        off.push("the checks (--reflect)");
    }
    if cfg.second_chain {
        off.push("the second chain (--second-chain, --chain-against, --chain-audit)");
    }
    if cfg.guide {
        off.push("the guide (--guide)");
    }
    if cfg.goal_probe {
        off.push("the goal probe (--goal-probe)");
    }
    cfg.second_chain = false;
    cfg.chain_against = false;
    cfg.chain_audit = false;
    cfg.guide = false;
    cfg.goal_probe = false;
    if !off.is_empty() {
        eprintln!(
            "phi-stream: the model is remote: off, since each needs the model's state in this process: {}",
            off.join(", ")
        );
    }
}

const PARAGRAPH: &str = "The scheduler assigns every operation of the graph to the backend that reports support for it, in priority order, and splits the graph where the assignment changes so that each backend computes a contiguous run of nodes; tensors crossing a split are copied between buffers before the next run starts. ";

fn config(s: &StreamArgs, sampling: Sampling) -> Result<Config> {
    let frame = match s.frame {
        FrameArg::Journal => Frame::Journal,
        FrameArg::Chat | FrameArg::Agent => Frame::Chat,
    };
    let agent = matches!(s.frame, FrameArg::Agent);
    let workspace = PathBuf::from(expand_home(&s.workspace));
    // The repository it develops with Claude (docs/dev.md), checked now.
    let dev = match &s.dev {
        Some(r) => Some(
            std::fs::canonicalize(expand_home(r))
                .with_context(|| format!("--dev {r}: no such directory"))?,
        ),
        None => None,
    };
    let system = match &s.system {
        Some(p) => {
            std::fs::read_to_string(expand_home(p)).with_context(|| format!("reading {p}"))?
        }
        None => {
            let base = match &s.personality {
                Some(p) => std::fs::read_to_string(expand_home(p))
                    .with_context(|| format!("reading {p}"))?,
                None => {
                    let home = expand_home("~/CLAUDE.md");
                    match std::fs::read_to_string(&home) {
                        Ok(t) if !t.trim().is_empty() => {
                            eprintln!("phi-stream: the personality's base is {home}");
                            t
                        }
                        _ => DEFAULT_BASE.to_string(),
                    }
                }
            };
            compose(&base, frame, dev.as_deref())
        }
    };
    let (mind, reflect) = s.mind.configs()?;
    // A check needs its token held while it deliberates: a second by
    // default, so a kept token never shows as a stall.
    let horizon = s
        .horizon
        .unwrap_or(if reflect.is_some() { 1.0 } else { 0.0 });
    let seed = s.seed_text.clone().unwrap_or_else(|| match frame {
        Frame::Journal => "(the room is quiet; nothing has been said. The journal goes on from wherever its thoughts were.)".to_string(),
        Frame::Chat if agent => "Begin: reason about your objective, then act on it with a tool.".to_string(),
        Frame::Chat => "[The stream begins. Nobody has spoken yet.]".to_string(),
    });
    Ok(Config {
        frame,
        system,
        seed,
        first_words: s.first_words.clone(),
        direct_max: s.direct_max,
        chunk: s.chunk,
        rollover_at: s.rollover_at,
        rollover_tokens: s.rollover_tokens,
        summary_max: 1024,
        sampling,
        status_every: 8,
        time_every_us: (s.time_every * 1e6) as i64,
        nudge_every_us: (s.nudge_every * 1e6) as i64,
        horizon_us: (horizon * 1e6) as i64,
        task: false,
        think_budget: 0,
        workspace,
        mind,
        reflect,
        dev,
        terminal: s.terminal,
        gate_output: !s.no_objective_gate,
        summary_on_quit: false,
        second_chain: s.second_chain || s.chain_against || s.chain_audit,
        chain_against: s.chain_against || s.chain_audit,
        chain_audit: s.chain_audit,
        goal_probe: s.goal_probe,
        guide: s.guide,
        improve: s.improve,
        agent,
    })
}

fn info_line(llm: &Llm, cfg: &Config) -> String {
    format!(
        "info model={} gpu_blocks={} n_blocks={} gpu_gib={:.2} host_gib={:.2} gpu_split={} n_ctx={} frame={} workspace={} started={}",
        escape(
            &std::path::Path::new(&llm.opts.model)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ),
        llm.split.gpu_blocks,
        llm.split.n_blocks,
        split::gib(llm.split.gpu_bytes),
        split::gib(llm.split.host_bytes),
        llm.gpu_layers
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("+"),
        llm.n_ctx(),
        if cfg.agent { "agent" } else { cfg.frame.name() },
        escape(&cfg.workspace.display().to_string()),
        clock::now_us()
    )
}

/// The model as a stream loads it: with the capture and the readout's
/// GPU memory set aside when the mind is read.
fn load_mind(m: &ModelArgs, a: &MindArgs) -> Result<Llm> {
    if !a.mind || m.remote.is_some() {
        return load(m);
    }
    let lens = lens::Lens::open(&expand_home(&a.lens))?;
    for l in &a.mind_layers {
        if !lens.header.layers.contains(l) {
            anyhow::bail!(
                "the lens has no block {l} (it has {:?})",
                lens.header.layers
            );
        }
    }
    let d = lens.header.d_model as u64;
    let extra = READOUT_RESERVE + a.mind_layers.len() as u64 * d * d * 2;
    load_with(
        m,
        Some(capture::CaptureConfig {
            layers: a.mind_layers.clone(),
            all_rows: false,
            keep_logits: false,
        }),
        extra,
    )
}

/// The service: the engine on its thread, the socket on this one.
fn serve_cmd(m: &ModelArgs, s: &StreamArgs, socket: PathBuf) -> Result<()> {
    match &m.remote {
        Some(url) => eprintln!("phi-stream: the model is served by {url}; asking it what it holds"),
        None => eprintln!("phi-stream: placing the model and loading it; the cards upload their shares at the first multiply"),
    }
    let mut cfg = config(s, sampling(m))?;
    // A restart of the service resumes from the summary its quit writes.
    cfg.summary_on_quit = true;
    // The guide lane takes a fifth sequence.
    let mut m = m.clone();
    if s.guide {
        m.n_seq = m.n_seq.max(5);
    }
    let llm = load_mind(&m, &s.mind)?;
    if !llm.forks() {
        remote_off(&mut cfg);
    }
    let info = info_line(&llm, &cfg);
    serve::asks_from(cfg.workspace.join("asks"));
    let (etx, erx) = mpsc::channel();
    let (ctx, crx) = mpsc::channel();
    if let Some(f) = &s.feed {
        let text =
            std::fs::read_to_string(expand_home(f)).with_context(|| format!("reading {f}"))?;
        ctx.send(Command::Feed(
            text,
            format!("the file {f}"),
            clock::now_us(),
        ))
        .ok();
    }
    let engine = Engine::new(llm, cfg, etx, crx)?;
    let worker = std::thread::spawn(move || engine.run());
    let r = serve::serve(erx, ctx, &socket, info);
    match worker.join() {
        Ok(Ok(_)) => r,
        Ok(Err(e)) => Err(e),
        Err(_) => anyhow::bail!("the engine thread panicked"),
    }
}

/// The stream on stdout, status and notes on stderr, stdin lines said to it.
fn run_cmd(m: &ModelArgs, s: &StreamArgs, max_tokens: usize) -> Result<()> {
    let mut cfg = config(s, sampling(m))?;
    let llm = load_mind(m, &s.mind)?;
    if !llm.forks() {
        remote_off(&mut cfg);
    }
    let (etx, erx) = mpsc::channel();
    let (ctx, crx) = mpsc::channel();
    if let Some(f) = &s.feed {
        let text =
            std::fs::read_to_string(expand_home(f)).with_context(|| format!("reading {f}"))?;
        ctx.send(Command::Feed(
            text,
            format!("the file {f}"),
            clock::now_us(),
        ))
        .ok();
    }
    let engine = Engine::new(llm, cfg, etx, crx)?;
    let worker = std::thread::spawn(move || engine.run());
    let stdin_tx = ctx.clone();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            if let Some(p) = line.strip_prefix("/feed ") {
                match std::fs::read_to_string(expand_home(p.trim())) {
                    Ok(t) => {
                        stdin_tx
                            .send(Command::Feed(
                                t,
                                format!("the file {}", p.trim()),
                                clock::now_us(),
                            ))
                            .ok();
                    }
                    Err(e) => eprintln!("/feed: {e}"),
                }
            } else if line == "/quit" {
                stdin_tx.send(Command::Quit).ok();
                break;
            } else if let Some(c) = line.strip_prefix("/chunk ") {
                if let Ok(n) = c.trim().parse() {
                    stdin_tx.send(Command::Chunk(n)).ok();
                }
            } else {
                stdin_tx.send(Command::Say(line, clock::now_us())).ok();
            }
        }
    });
    let mut out = std::io::stdout();
    let mut live = 0usize;
    loop {
        match erx.recv() {
            Ok(Event::Text(t, kind, _, _)) => {
                match kind {
                    Kind::Given => write!(out, "\x1b[33m{t}\x1b[0m")?,
                    Kind::Speak => write!(out, "\x1b[1m{t}\x1b[0m")?,
                    Kind::Think => {
                        write!(out, "{t}")?;
                        live += 1;
                    }
                }
                out.flush()?;
                if max_tokens > 0 && live >= max_tokens {
                    ctx.send(Command::Quit).ok();
                }
            }
            Ok(Event::Status(st)) => eprintln!("\x1b[2m[{}]\x1b[0m", engine::status_text(&st)),
            Ok(Event::Note(n)) => eprintln!("\x1b[2m[{n}]\x1b[0m"),
            // The diagnostics are for the terminal and the stream (diag.md).
            Ok(Event::Diag(_)) => {}
            Ok(Event::Mind(r)) => eprintln!("\x1b[2mmind {}\x1b[0m", mind::line(&r)),
            Ok(Event::Reflect(e)) => eprintln!("\x1b[2mreflect {}\x1b[0m", reflect::line(&e)),
            Ok(Event::Objective(_, t)) => eprintln!("\x1b[2mobjective: {t}\x1b[0m"),
            Ok(Event::Guide(_)) => {}
            Ok(Event::ToClaude(m)) => eprintln!("\x1b[2mto Claude m{}: {}\x1b[0m", m.id, m.text),
            Ok(Event::TermStart(_, _, c)) => eprintln!("\x1b[2m$ {c}\x1b[0m"),
            Ok(Event::TermEnd(_, r)) => eprintln!("\x1b[2m{}\x1b[0m", r.out),
            Ok(Event::Act(a)) => eprintln!(
                "\x1b[2m{} {}\x1b[0m",
                if a.end { "  ->" } else { a.kind.as_str() },
                a.text
            ),
            Ok(Event::Delib(d)) => eprintln!("\x1b[2mbeside: {}\x1b[0m", d.text),
            Ok(Event::Done { .. }) => {}
            Ok(Event::Stopped) | Err(_) => break,
        }
    }
    match worker.join() {
        Ok(r) => r.map(|_| ()),
        Err(_) => anyhow::bail!("the engine thread panicked"),
    }
}

/// A one-shot command to the service: its reply on stdout.
fn ask(socket: &Path, line: &str) -> Result<()> {
    let mut c = Client::connect(socket)?;
    let reply = c.ask(line)?;
    println!("{reply}");
    Ok(())
}

/// The stream as it happens, on stdout; status on stderr when asked.
fn tail(socket: &Path, with_status: bool, with_mind: bool) -> Result<()> {
    let mut c = Client::connect(socket)?;
    c.send("tail")?;
    let mut out = std::io::stdout();
    while let Some(line) = c.line()? {
        match parse(&line) {
            Msg::Text(t, Kind::Given, _, _) => write!(out, "\x1b[33m{t}\x1b[0m")?,
            Msg::Text(t, Kind::Speak, _, _) => write!(out, "\x1b[1m{t}\x1b[0m")?,
            Msg::Text(t, Kind::Think, _, _) => write!(out, "{t}")?,
            Msg::Status(st) => {
                if with_status {
                    eprintln!("\x1b[2m[{}]\x1b[0m", engine::status_text(&st));
                }
            }
            Msg::Note(n) => eprintln!("\x1b[2m[{n}]\x1b[0m"),
            Msg::Mind(r) => {
                if with_mind {
                    eprintln!("\x1b[2mmind {}\x1b[0m", mind::line(&r));
                }
            }
            Msg::Reflect(e) => eprintln!("\x1b[2mreflect {}\x1b[0m", reflect::line(&e)),
            Msg::Bye => break,
            _ => {}
        }
        out.flush()?;
    }
    Ok(())
}

/// The socket line that says `text`, named when `who` is given (`docs/dev.md`).
fn say_line(who: Option<&str>, text: &str) -> String {
    match who {
        Some(w) => format!(
            "say-as {} {}",
            w.split_whitespace().collect::<Vec<_>>().join("_"),
            escape(text)
        ),
        None => format!("say {}", escape(text)),
    }
}

/// The service's lines, on a thread, into a channel (so the caller can
/// stop at a deadline).
fn lines_of(mut c: Client) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        while let Ok(Some(l)) = c.line() {
            if tx.send(l).is_err() {
                break;
            }
        }
    });
    rx
}

/// `ask`: a message from Claude, and the stream's answer to it
/// (`client::ask_claude`) on stdout; its ids and the wait on stderr.
fn ask_claude(socket: &Path, text: &str, timeout: u64) -> Result<()> {
    let t0 = std::time::Instant::now();
    let (id, answer) = client::ask_claude(socket, text, std::time::Duration::from_secs(timeout))?;
    match answer {
        Some(m) => {
            eprintln!(
                "{id} answered by m{} in {:.1} s",
                m.id,
                t0.elapsed().as_secs_f32()
            );
            println!("{}", m.text);
            Ok(())
        }
        None => anyhow::bail!("no answer to {id} within {timeout} s (it waits for its next turn)"),
    }
}

/// `ask`: say `text`, then print the next line the stream says aloud
/// (what it placed after the message was heard, its spoken pieces up to
/// the end of their line). Its thoughts meanwhile go to stderr when asked.
fn converse(
    socket: &Path,
    who: Option<&str>,
    text: &str,
    timeout: u64,
    thoughts: bool,
) -> Result<()> {
    let mut c = Client::connect(socket)?;
    let t0 = clock::now_us();
    c.ask(&say_line(who, text))?;
    c.send("tail")?;
    let rx = lines_of(c);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout);
    let mut speech = String::new();
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let line = match rx.recv_timeout(left) {
            Ok(l) => l,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                anyhow::bail!("it did not speak within {timeout} s (it went on thinking)")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("the service closed the connection")
            }
        };
        match parse(&line) {
            Msg::Text(t, Kind::Speak, at, _) if at >= t0 => {
                speech.push_str(&t);
                if let Some(i) = speech.find('\n') {
                    let said = speech[..i].trim().trim_start_matches('»').trim();
                    if !said.is_empty() {
                        println!("{said}");
                        return Ok(());
                    }
                    speech = speech[i + 1..].to_string();
                }
            }
            Msg::Text(t, Kind::Think, at, _) if at >= t0 && thoughts => eprint!("{t}"),
            Msg::Bye => anyhow::bail!("the service stopped"),
            _ => {}
        }
    }
}

/// `listen`: from now on, what it says aloud, its notes and preferences,
/// what it hears, and the checks that changed a word, one line each with
/// the time, flushed line by line (a monitor's events).
fn listen(socket: &Path) -> Result<()> {
    let mut c = Client::connect(socket)?;
    let t0 = clock::now_us();
    c.send("tail")?;
    let mut out = std::io::stdout();
    let (mut speech, mut speech_t) = (String::new(), 0i64);
    let (mut heard, mut heard_t) = (String::new(), 0i64);
    while let Some(line) = c.line()? {
        match parse(&line) {
            Msg::Text(t, Kind::Speak, at, _) if at >= t0 => {
                if speech.is_empty() {
                    speech_t = at;
                }
                speech.push_str(&t);
                while let Some(i) = speech.find('\n') {
                    let said = speech[..i]
                        .trim()
                        .trim_start_matches('»')
                        .trim()
                        .to_string();
                    if !said.is_empty() {
                        writeln!(out, "{} said: {said}", clock::hms(speech_t))?;
                    }
                    speech = speech[i + 1..].to_string();
                    speech_t = at;
                }
            }
            Msg::Text(t, Kind::Given, at, _) if at >= t0 => {
                // What it heard or was handed: its first line.
                if heard.is_empty() {
                    heard_t = at;
                }
                heard.push_str(&t);
                if let Some(l) = heard.lines().map(str::trim).find(|l| !l.is_empty()) {
                    let l: String = l.chars().take(200).collect();
                    writeln!(out, "{} heard: {l}", clock::hms(heard_t))?;
                }
                heard.clear();
            }
            Msg::Note(n) => writeln!(out, "{} {n}", clock::hms(clock::now_us()))?,
            // What it sent Claude, whole (one line, newlines shown as \n).
            Msg::ToClaude(m) if m.t_us >= t0 => writeln!(
                out,
                "{} to Claude m{}{}: {}",
                clock::hms(m.t_us),
                m.id,
                m.re.map(|r| format!(" (answering {r})"))
                    .unwrap_or_default(),
                m.text.replace('\n', "\\n")
            )?,
            Msg::Reflect(e) if e.outcome == reflect::Outcome::Changed => writeln!(
                out,
                "{} changed {:?} to {:?} at position {} ({})",
                clock::hms(e.t_us),
                e.chosen.trim(),
                e.to.trim(),
                e.pos,
                e.why.name()
            )?,
            Msg::Bye => break,
            _ => {}
        }
        out.flush()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let socket = cli.socket.clone().unwrap_or_else(default_socket);
    match cli.cmd {
        Cmd::Serve { stream } => serve_cmd(&cli.model, &stream, socket),
        Cmd::Tui { follow } => tui::run(&socket, follow),
        Cmd::Say { who, text } => ask(&socket, &say_line(who.as_deref(), &text.join(" "))),
        Cmd::Ask {
            who,
            spoken,
            timeout,
            thoughts,
            text,
        } if spoken => converse(&socket, who.as_deref(), &text.join(" "), timeout, thoughts),
        Cmd::Ask { timeout, text, .. } => ask_claude(&socket, &text.join(" "), timeout),
        Cmd::Listen => listen(&socket),
        Cmd::Mcp => mcp::run(&socket),
        Cmd::Feed { path } => ask(&socket, &format!("feed {}", expand_home(&path))),
        Cmd::KeepAt { p } => ask(&socket, &format!("keep-at {p}")),
        Cmd::Objective { text } => {
            let t = text.join(" ");
            let t = t.trim();
            ask(
                &socket,
                &format!("objective {}", if t == "-" { "" } else { t }),
            )
        }
        Cmd::Tail { status, mind } => tail(&socket, status, mind),
        Cmd::Status => {
            let mut c = Client::connect(&socket)?;
            c.send("status")?;
            while let Some(line) = c.line()? {
                match parse(&line) {
                    Msg::Status(st) => println!("{}", engine::status_text(&st)),
                    Msg::Ok(_) | Msg::Err(_) | Msg::Bye => break,
                    _ => {}
                }
            }
            Ok(())
        }
        Cmd::Chunk { n } => ask(&socket, &format!("chunk {n}")),
        Cmd::Chain { state } => ask(&socket, &format!("chain {state}")),
        Cmd::Goal { state } => ask(&socket, &format!("goal {state}")),
        Cmd::Guide { state } => ask(&socket, &format!("guide {state}")),
        Cmd::Experts { state } => ask(&socket, &format!("experts {state}")),
        Cmd::Breaker { state } => ask(&socket, &format!("breaker {state}")),
        Cmd::Nudges { state } => ask(&socket, &format!("nudges {state}")),
        Cmd::Set { key, value } => ask(&socket, &format!("set {key} {value}")),
        Cmd::Temp { t } => ask(&socket, &format!("temp {t}")),
        Cmd::Persona { path } => ask(&socket, &format!("persona {}", expand_home(&path))),
        Cmd::Pause => ask(&socket, "pause"),
        Cmd::Resume => ask(&socket, "resume"),
        Cmd::Quit => ask(&socket, "quit"),
        Cmd::Probe {
            prompt,
            gen,
            chunks,
            capture_idle,
            mind_layers,
        } => {
            let text = match prompt {
                Some(p) => std::fs::read_to_string(expand_home(&p))
                    .with_context(|| format!("reading {p}"))?,
                None => PARAGRAPH.repeat(20),
            };
            let capture =
                (capture_idle || !mind_layers.is_empty()).then(|| capture::CaptureConfig {
                    layers: mind_layers.clone(),
                    all_rows: false,
                    keep_logits: false,
                });
            // The readout's GPU memory and its transports (8 MiB each) are
            // set aside only when the readout runs.
            let extra = if mind_layers.is_empty() {
                0
            } else {
                READOUT_RESERVE + mind_layers.len() as u64 * (8 << 20)
            };
            let mut llm = load_with(&cli.model, capture, extra)?;
            if capture_idle && mind_layers.is_empty() {
                if let Some(c) = llm.capture() {
                    c.enabled = false;
                }
            }
            let mut mind = (!mind_layers.is_empty()).then(|| probe::MindCost::new(mind_layers));
            probe::probe(&mut llm, &text, gen, &chunks, &mut mind)
        }
        Cmd::Gate {
            thoughts,
            chunk,
            compare,
            snapshot,
            reflect,
            episodes,
        } => {
            if reflect {
                // The capture as the engine has it when it reflects: the
                // band's blocks and the final one.
                let mut llm = load_with(
                    &cli.model,
                    Some(capture::CaptureConfig {
                        layers: Vec::new(),
                        all_rows: false,
                        keep_logits: false,
                    }),
                    READOUT_RESERVE,
                )?;
                let last = llm.n_layer() - 1;
                if let Some(c) = llm.capture() {
                    c.cfg.layers = vec![27, 29, 31, last];
                }
                let a = format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    PARAGRAPH.repeat(3)
                );
                let q = reflect::question(
                    Frame::Chat,
                    Some(clock::now_us()),
                    " scheduler",
                    &["backend", "graph", "split"],
                );
                return gate::gate_reflect(&mut llm, &a, &q, episodes, 12, 16);
            }
            let mut llm = load(&cli.model)?;
            if snapshot {
                let a = format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    PARAGRAPH.repeat(3)
                );
                return gate::gate_snapshot(&mut llm, &a, 24, 24);
            }
            let a = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n[The stream begins.]<|im_end|>\n<|im_start|>assistant\n<think>\n",
                compose(DEFAULT_BASE, Frame::Chat, None)
            );
            let b = format!("\n[they hand you a document:\n{}\n]\n", PARAGRAPH.repeat(6));
            gate::gate(&mut llm, &a, &b, thoughts, chunk, compare)
        }
        Cmd::Code { cmd } => match cmd {
            CodeCmd::Stream {
                tasks,
                first,
                only,
                base,
                think_budget,
                penalty,
                mind,
            } => {
                let all = code::load_tasks(&expand_home(&tasks))?;
                let sel = code::select(all, first, &only)?;
                let (base_text, label) = match base.as_str() {
                    "neutral" => (engine::DEFAULT_BASE.to_string(), "neutral".to_string()),
                    "claude-md" => (
                        std::fs::read_to_string(expand_home("~/CLAUDE.md"))
                            .context("reading ~/CLAUDE.md")?,
                        "claude-md".to_string(),
                    ),
                    f => (
                        std::fs::read_to_string(expand_home(f))
                            .with_context(|| format!("reading {f}"))?,
                        f.to_string(),
                    ),
                };
                let (mind_cfg, reflect_cfg) = mind.configs()?;
                let dir = code::run_dir(&format!("stream-{}", label.replace('/', "_")))?;
                let arm = if mind.reflect {
                    "reflect"
                } else if mind.reflect_dry {
                    "reflect-dry"
                } else if mind.mind {
                    "mind"
                } else {
                    "plain"
                };
                std::fs::write(
                    dir.join("run.json"),
                    serde_json::json!({"base": label, "think_budget": think_budget, "repeat_penalty": penalty, "arm": arm, "mind_layers": if mind.mind { mind.mind_layers.clone() } else { Vec::new() }, "temp": 0, "frame": "chat", "tasks": sel.len(), "model": cli.model.model}).to_string(),
                )?;
                println!("{} tasks through the stream's engine: base {label}, thinking budget {think_budget}, penalty {penalty}, greedy, {arm}; run directory {}", sel.len(), dir.display());
                let llm = load_mind(&cli.model, &mind)?;
                let opts = code::StreamOpts {
                    base: base_text,
                    base_label: label,
                    think_budget,
                    repeat_penalty: penalty,
                    mind: mind_cfg,
                    reflect: reflect_cfg,
                };
                let (out, _llm) = code::stream(llm, &sel, &opts, &dir)?;
                let s = code::summary(&out);
                std::fs::write(
                    dir.join("summary.txt"),
                    format!("base {}\n{s}\n", opts.base_label),
                )?;
                println!("{s}");
                Ok(())
            }
            CodeCmd::Anchor {
                tasks,
                first,
                only,
                max_tokens,
            } => {
                let all = code::load_tasks(&expand_home(&tasks))?;
                let sel = code::select(all, first, &only)?;
                let dir = code::run_dir("anchor")?;
                println!("{} tasks, the benchmark's protocol (raw prompt, greedy, at most {max_tokens} tokens); run directory {}", sel.len(), dir.display());
                let mut llm = load(&cli.model)?;
                let out = code::anchor(&mut llm, &sel, max_tokens, &dir)?;
                let s = code::summary(&out);
                std::fs::write(dir.join("summary.txt"), format!("{s}\n"))?;
                println!("{s}");
                Ok(())
            }
        },
        Cmd::Improve { upper, title, repo } => {
            let home = std::env::var("HOME").unwrap_or_default();
            let mut imp = improve::Improver::new(improve::ImproveConfig {
                repo: repo.canonicalize()?,
                upper: upper.canonicalize()?,
                root: PathBuf::from(&home).join(".cache/phi-stream/improve"),
                mirror: PathBuf::from(&home).join(".cache/phi-stream/improve/by-hand"),
            });
            let id = imp
                .propose(&title, None)
                .map_err(|b| anyhow::anyhow!("candidate {b} is building"))?;
            eprintln!("candidate {id}: staging and building (make check in the sandbox)");
            loop {
                if let Some(o) = imp.poll() {
                    println!(
                        "candidate {} {} in {:.0} s: {}\nfiles: {}\nbase {}\ndir {}\n{}",
                        o.id,
                        o.verdict.word(),
                        o.secs,
                        o.title,
                        o.files.join(", "),
                        o.base,
                        o.dir.display(),
                        o.summary
                    );
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
        Cmd::Lens { cmd } => match cmd {
            LensCmd::Convert { src, out } => {
                let h = lens::convert(&expand_home(&src), &expand_home(&out))?;
                print_lens(&h, &out);
                Ok(())
            }
            LensCmd::Eval { lens, sets } => {
                let mut lens = lens::Lens::open(&expand_home(&lens))?;
                let sets = if sets.is_empty() {
                    let dir = expand_home("~/models/jlens/eval");
                    let mut v: Vec<String> = std::fs::read_dir(&dir)
                        .with_context(|| {
                            format!("reading {dir} (scripts/fetch-lens.sh fetches the sets)")
                        })?
                        .filter_map(|e| e.ok().map(|e| e.path().display().to_string()))
                        .filter(|p| p.ends_with(".json") && p.contains("lens-eval-"))
                        .collect();
                    v.sort();
                    v
                } else {
                    sets.iter().map(|s| expand_home(s)).collect()
                };
                let n_layers = lens.header.layers.len() as u64;
                let d = lens.header.d_model as u64;
                // The transports, the readout's own buffers, and the logits
                // of one item (two columns per block and the final block).
                let extra =
                    READOUT_RESERVE + n_layers * d * d * 2 + (2 * n_layers + 1) * 248_320 * 4 * 2;
                let mut layers: Vec<i32> = lens.header.layers.clone();
                let mut llm = load_with(
                    &cli.model,
                    Some(capture::CaptureConfig {
                        layers: Vec::new(),
                        all_rows: false,
                        keep_logits: false,
                    }),
                    extra,
                )?;
                let final_layer = llm.n_layer() - 1;
                if layers.iter().any(|&l| l >= final_layer) {
                    anyhow::bail!(
                        "the lens has blocks {:?}; this model's final block is {final_layer}",
                        layers
                    );
                }
                let mut capture_layers = layers.clone();
                capture_layers.push(final_layer);
                if let Some(c) = llm.capture() {
                    c.cfg.layers = capture_layers;
                }
                let mut readout = eval::readout_for(&mut llm)?;
                layers = eval::load_lens(&mut readout, &mut lens)?;
                println!(
                    "lens {} ({} blocks, from sha256 {}), model {}, final block {final_layer}",
                    lens.path,
                    layers.len(),
                    &lens.header.source_sha256[..16],
                    cli.model.model
                );
                for path in &sets {
                    let (name, position, items) = eval::load_set(path)?;
                    let r = eval::run_set(
                        &mut llm,
                        &mut readout,
                        &name,
                        position,
                        &items,
                        &layers,
                        final_layer,
                    )?;
                    eval::report(&r, &layers);
                }
                Ok(())
            }
            LensCmd::Info { path } => {
                let l = lens::Lens::open(&expand_home(&path))?;
                print_lens(&l.header, &path);
                Ok(())
            }
            LensCmd::Check => {
                let mut llm = load_with(
                    &cli.model,
                    Some(capture::CaptureConfig {
                        layers: Vec::new(),
                        all_rows: false,
                        keep_logits: true,
                    }),
                    READOUT_RESERVE,
                )?;
                let last = llm.n_layer() - 1;
                if let Some(c) = llm.capture() {
                    c.cfg.layers = vec![last];
                }
                check::check(&mut llm, &PARAGRAPH.repeat(6), &PARAGRAPH.repeat(4))
            }
        },
        Cmd::Run { stream, max_tokens } => run_cmd(&cli.model, &stream, max_tokens),
    }
}
