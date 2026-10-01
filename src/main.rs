//! `phi-stream`: one model split over the GPU, the cards and host memory,
//! thinking without pause while it reads what it is given, over llama.cpp
//! used as a library and never changed. The model is owned by a service
//! (`serve`); the terminal and the one-shot commands are its clients. See
//! main.md.

mod capture;
mod check;
mod client;
mod clock;
mod engine;
mod eval;
mod gate;
mod lens;
mod llm;
mod mind;
mod probe;
mod readout;
mod serve;
mod split;
mod sys;
mod torch;
mod tui;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context as _, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::client::{default_socket, escape, parse, Client, Msg};
use crate::engine::{compose, Command, Config, Engine, Event, Frame, Kind, Mode, DEFAULT_BASE};
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

#[derive(Args)]
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
}

#[derive(Clone, Copy, ValueEnum)]
enum FrameArg {
    /// One continuous first-person text, no turns; « from outside, » said aloud.
    Journal,
    /// The model's chat template: thoughts in <think>, speech after.
    Chat,
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
    /// Put the wall clock into the chain after this many seconds with
    /// nothing from outside (0: never).
    #[arg(long, default_value_t = 60.0)]
    time_every: f64,
    /// Nudge circling thoughts at most once in this many seconds.
    #[arg(long, default_value_t = 60.0)]
    nudge_every: f64,
    /// Hand over a file at the start.
    #[arg(long)]
    feed: Option<String>,
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
}

#[derive(Subcommand)]
enum Cmd {
    /// Own the model and listen on the socket; the terminal and the commands below talk to it.
    Serve {
        #[command(flatten)]
        stream: StreamArgs,
    },
    /// The terminal, talking to the service.
    Tui,
    /// Say something to the stream.
    Say {
        text: Vec<String>,
    },
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
    /// Tokens read beside the live token each cycle (0 adapts).
    Chunk {
        n: usize,
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
    backend_path();
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
        n_seq: m.n_seq,
        kv_unified: !m.kv_split,
        verbose: m.verbose,
        capture,
        extra_reserve: extra,
    };
    Llm::load(opts, &sampling(m))
}

const PARAGRAPH: &str = "The scheduler assigns every operation of the graph to the backend that reports support for it, in priority order, and splits the graph where the assignment changes so that each backend computes a contiguous run of nodes; tensors crossing a split are copied between buffers before the next run starts. ";

fn config(s: &StreamArgs, sampling: Sampling) -> Result<Config> {
    let frame = match s.frame {
        FrameArg::Journal => Frame::Journal,
        FrameArg::Chat => Frame::Chat,
    };
    let workspace = PathBuf::from(expand_home(&s.workspace));
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
            compose(&base, frame)
        }
    };
    let seed = s.seed_text.clone().unwrap_or_else(|| match frame {
        Frame::Journal => "(the room is quiet; nothing has been said. The journal goes on from wherever its thoughts were.)".to_string(),
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
        summary_max: 1024,
        sampling,
        status_every: 8,
        time_every_us: (s.time_every * 1e6) as i64,
        nudge_every_us: (s.nudge_every * 1e6) as i64,
        workspace,
        mind: s.mind.then(|| mind::MindConfig {
            lens: expand_home(&s.lens),
            layers: s.mind_layers.clone(),
            k: s.mind_k,
        }),
    })
}

fn info_line(llm: &Llm, cfg: &Config) -> String {
    format!(
        "info model={} gpu_blocks={} n_blocks={} gpu_gib={:.2} host_gib={:.2} n_ctx={} frame={} workspace={}",
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
        llm.n_ctx(),
        cfg.frame.name(),
        escape(&cfg.workspace.display().to_string())
    )
}

/// The service: the engine on its thread, the socket on this one.
/// The model as a stream loads it: with the capture and the readout's
/// GPU memory set aside when the mind is read.
fn load_stream(m: &ModelArgs, s: &StreamArgs) -> Result<Llm> {
    if !s.mind {
        return load(m);
    }
    let lens_path = expand_home(&s.lens);
    let lens = lens::Lens::open(&lens_path)?;
    for l in &s.mind_layers {
        if !lens.header.layers.contains(l) {
            anyhow::bail!(
                "the lens has no block {l} (it has {:?})",
                lens.header.layers
            );
        }
    }
    let d = lens.header.d_model as u64;
    let extra = READOUT_RESERVE + s.mind_layers.len() as u64 * d * d * 2;
    load_with(
        m,
        Some(capture::CaptureConfig {
            layers: s.mind_layers.clone(),
            all_rows: false,
            keep_logits: false,
        }),
        extra,
    )
}

fn serve_cmd(m: &ModelArgs, s: &StreamArgs, socket: PathBuf) -> Result<()> {
    eprintln!("phi-stream: placing the model and loading it; the cards upload their shares at the first multiply");
    let llm = load_stream(m, s)?;
    let cfg = config(s, sampling(m))?;
    let info = info_line(&llm, &cfg);
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
        Ok(Ok(())) => r,
        Ok(Err(e)) => Err(e),
        Err(_) => anyhow::bail!("the engine thread panicked"),
    }
}

/// The stream on stdout, status and notes on stderr, stdin lines said to it.
fn run_cmd(m: &ModelArgs, s: &StreamArgs, max_tokens: usize) -> Result<()> {
    let llm = load_stream(m, s)?;
    let cfg = config(s, sampling(m))?;
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
            Ok(Event::Text(t, kind, _)) => {
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
            Ok(Event::Status(st)) => eprintln!("\x1b[2m[{}]\x1b[0m", status_text(&st)),
            Ok(Event::Note(n)) => eprintln!("\x1b[2m[{n}]\x1b[0m"),
            Ok(Event::Mind(r)) => eprintln!("\x1b[2mmind {}\x1b[0m", mind::line(&r)),
            Ok(Event::Stopped) | Err(_) => break,
        }
    }
    match worker.join() {
        Ok(r) => r,
        Err(_) => anyhow::bail!("the engine thread panicked"),
    }
}

fn status_text(st: &engine::Status) -> String {
    let mode = match st.mode {
        Mode::Thinking => "thinking".to_string(),
        Mode::Speaking => "speaking".to_string(),
        Mode::Reading { done, total } => format!("reading {done}/{total}"),
        Mode::CatchingUp { done, total } => format!("catching up {done}/{total}"),
        Mode::Summarizing { tokens } => format!("summarizing ({tokens})"),
        Mode::Paused => "paused".to_string(),
    };
    format!(
        "{mode}; stream {:.1} tok/s, beside {:.1} tok/s, cycle {:.0} ms; {}/{} cells; queued {}; notes {}; {} frame",
        st.stream_tps, st.side_tps, st.cycle_ms, st.pos, st.n_ctx, st.queued, st.notes, st.frame
    )
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
            Msg::Text(t, Kind::Given, _) => write!(out, "\x1b[33m{t}\x1b[0m")?,
            Msg::Text(t, Kind::Speak, _) => write!(out, "\x1b[1m{t}\x1b[0m")?,
            Msg::Text(t, Kind::Think, _) => write!(out, "{t}")?,
            Msg::Status(st) => {
                if with_status {
                    eprintln!("\x1b[2m[{}]\x1b[0m", status_text(&st));
                }
            }
            Msg::Note(n) => eprintln!("\x1b[2m[{n}]\x1b[0m"),
            Msg::Mind(r) => {
                if with_mind {
                    eprintln!("\x1b[2mmind {}\x1b[0m", mind::line(&r));
                }
            }
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
        Cmd::Tui => tui::run(&socket),
        Cmd::Say { text } => ask(&socket, &format!("say {}", escape(&text.join(" ")))),
        Cmd::Feed { path } => ask(&socket, &format!("feed {}", expand_home(&path))),
        Cmd::Tail { status, mind } => tail(&socket, status, mind),
        Cmd::Status => {
            let mut c = Client::connect(&socket)?;
            c.send("status")?;
            while let Some(line) = c.line()? {
                match parse(&line) {
                    Msg::Status(st) => println!("{}", status_text(&st)),
                    Msg::Ok(_) | Msg::Err(_) | Msg::Bye => break,
                    _ => {}
                }
            }
            Ok(())
        }
        Cmd::Chunk { n } => ask(&socket, &format!("chunk {n}")),
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
        } => {
            let mut llm = load(&cli.model)?;
            let a = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n[The stream begins.]<|im_end|>\n<|im_start|>assistant\n<think>\n",
                compose(DEFAULT_BASE, Frame::Chat)
            );
            let b = format!("\n[they hand you a document:\n{}\n]\n", PARAGRAPH.repeat(6));
            gate::gate(&mut llm, &a, &b, thoughts, chunk, compare)
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
