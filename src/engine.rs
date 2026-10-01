//! The stream: one live sequence that never stops generating; what it is
//! given is either heard at once (a short message, decoded into the live
//! sequence in one cycle) or read beside it (a long one: another
//! sequence, given the live one's prefix, takes a chunk every cycle while
//! the live token goes on), and when a reading ends the two are joined:
//! a third sequence gets the prefix, the read cells and the recurrent
//! state after the reading, then catches up on the thoughts produced
//! meanwhile, chunk by chunk, and becomes the live one. A full context
//! is rolled over the same way from a summary the stream writes, and so
//! is a change of persona. The text is framed as a journal (one
//! continuous first-person text, no turns) or as a chat; in either the
//! mind keeps notes and reads files by lines it writes itself. The
//! engine runs on its own thread; commands come in and events go out
//! through channels. See engine.md.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use anyhow::{bail, Context as _, Result};

use crate::clock;
use crate::llm::{Lane, Llm, Sampling};
use crate::mind::{Mind, MindConfig, Reading as MindReading};
use crate::playout::Playout;
use crate::reflect::{self, Decision, Episode, Outcome, ReflectConfig, Reflector, Why};

/// What a piece of the stream is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Inside the thoughts.
    Think,
    /// Said aloud (a `»` line in the journal; after `</think>` in a chat).
    Speak,
    /// Put in from outside: what was heard or read, and the engine's marks.
    Given,
}

#[derive(Clone, Debug)]
pub enum Mode {
    Thinking,
    Speaking,
    Reading { done: usize, total: usize },
    CatchingUp { done: usize, total: usize },
    Summarizing { tokens: usize },
    Paused,
}

#[derive(Clone, Debug)]
pub struct Status {
    pub mode: Mode,
    /// Live tokens a second, over recent cycles that carried one.
    pub stream_tps: f64,
    /// Tokens read or caught up a second, over recent cycles that did.
    pub side_tps: f64,
    pub cycle_ms: f64,
    /// Cells the live sequence holds, and the pool.
    pub pos: i32,
    pub n_ctx: u32,
    /// Messages waiting to be read.
    pub queued: usize,
    pub chunk: usize,
    pub rollovers: u32,
    pub notes: usize,
    pub frame: &'static str,
    /// Lines beginning with « that the mind wrote itself (a frame leak).
    pub leaks: u32,
    /// The mind readout's time per token, milliseconds (0: not read).
    pub mind_ms: f64,
    /// When this status was taken (microseconds of real time).
    pub t_us: i64,
    /// Failed reads kept out of the chain (one failure per nudge interval goes in).
    pub reads_quiet: u32,
    /// The reflection loop's checks, changed tokens and unparsed answers
    /// since the start, and whether a check is in flight.
    pub checks: u64,
    pub changes: u64,
    pub unparsed: u64,
    pub checking: bool,
}

pub enum Event {
    /// A piece of the stream, the real time it exists at (microseconds
    /// since the epoch, `clock.rs`).
    Text(String, Kind, i64),
    Status(Status),
    Note(String),
    /// What was on its mind at a token it placed (`mind.rs`).
    Mind(MindReading),
    /// A check of a token ended (`reflect.rs`).
    Reflect(Episode),
    /// A task ended: its thinking tokens, and whether the budget closed them.
    Done {
        think_tokens: usize,
        capped: bool,
    },
    Stopped,
}

pub enum Command {
    /// Something said to the stream, and when it was heard (microseconds).
    Say(String, i64),
    /// Something said by someone named (who, what, when): the stream hears
    /// who is speaking (in development: the person, or Claude).
    SayAs(String, String, i64),
    /// A document handed over, with its label and when it was handed over.
    Feed(String, String, i64),
    Pause,
    Resume,
    /// Tokens a cycle reads beside the live token; 0 adapts to the amount.
    Chunk(usize),
    /// The reflection loop's keep threshold, set while it runs (`reflect.md`).
    KeepAt(f32),
    Temp(f32),
    /// A new persona: the context is rolled over onto it.
    Persona(String),
    /// Ask for a status event now.
    Status,
    Quit,
}

/// How the text is framed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    /// One continuous first-person text: no turns, `«` lines from outside,
    /// `»` lines said aloud.
    Journal,
    /// The model's chat template: thoughts inside `<think>`, speech after.
    Chat,
}

impl Frame {
    pub fn name(self) -> &'static str {
        match self {
            Frame::Journal => "journal",
            Frame::Chat => "chat",
        }
    }
}

/// The stream's framing and policy.
#[derive(Clone, Debug)]
pub struct Config {
    pub frame: Frame,
    /// The persona and the rules.
    pub system: String,
    /// The first thing from outside.
    pub seed: String,
    /// The journal's first words, in its own voice, after the seed.
    pub first_words: String,
    /// Said things up to this many tokens are heard at once.
    pub direct_max: usize,
    /// Tokens a cycle reads beside the live token; 0 adapts.
    pub chunk: usize,
    /// Roll the context over when the live sequence passes this share.
    pub rollover_at: f32,
    /// Tokens the summary may run to.
    pub summary_max: usize,
    /// The sampler's settings (temperature changed at run time).
    pub sampling: Sampling,
    /// Cycles between status events.
    pub status_every: u32,
    /// Put the clock into the chain after this long without anything from
    /// outside (microseconds of real time; 0: never).
    pub time_every_us: i64,
    /// Nudge circling thoughts at most this often (microseconds).
    pub nudge_every_us: i64,
    /// How far the shown text runs behind its placement: the playout's
    /// target lag (`playout.md`, microseconds; 0: shown at once). Whatever it
    /// is, nothing a check holds is shown before the check ends.
    pub horizon_us: i64,
    /// A task, not a stream (`code.md`): stop at the end of the first
    /// answer, and put nothing into the chain on the engine's own account
    /// (no clock lines, nudges, reads, rollover).
    pub task: bool,
    /// In a task, the thinking tokens before `</think>` is placed (0: no limit).
    pub think_budget: usize,
    /// Where the persona, the notes and the log live.
    pub workspace: PathBuf,
    /// Read what is on its mind at every token it places (`mind.rs`);
    /// none: not read (and no eval callback installed).
    pub mind: Option<MindConfig>,
    /// Check the tokens it places, per token, beside the live one
    /// (`reflect.rs`; needs `mind`); none: no checks.
    pub reflect: Option<ReflectConfig>,
    /// The repository the stream develops with Claude (`docs/dev.md`): its reads
    /// resolve there, its persona says so; none: no development.
    pub dev: Option<PathBuf>,
}

/// The base of the personality when no file gives one.
pub const DEFAULT_BASE: &str = "It is concrete, curious, and honest about what it does not know. It states mechanism rather than judgment, does not hedge out of politeness, and writes no em or en dash, ever.";

/// What the journal is told before the base: how to take a person's
/// standing instructions as a mind's own manner.
const PREAMBLE_JOURNAL: &str = "What follows, quoted line by line, is the standing instruction set of the person this mind lives with. The journal never reproduces it: it is a document the mind has read, not something it writes. It is the base of this mind's own personality: the manner it prescribes (dense, analytical, neutral, no hedging, mechanism over judgment, structure over flourish, no em or en dash ever) is this mind's manner, in its own thoughts as much as in what it says aloud. Where the text speaks of responses, delimiters, tools, Python or memory files, it describes another harness: here there are no responses to delimit and no tools, and memory is the [note: ...] line below.";

/// The chat frame's preamble, in the second person.
const PREAMBLE_CHAT: &str = "What follows, quoted line by line, is the standing instruction set of the person you live with. You never reproduce it: it is a document you have read, not something you write. It is the base of your own personality: the manner it prescribes (dense, analytical, neutral, no hedging, mechanism over judgment, structure over flourish, no em or en dash ever) is your manner, in your thoughts as much as in what you say aloud. Where it speaks of responses, delimiters, tools, Python or memory files, it describes another harness: here there are no responses to delimit and no tools, and memory is the [note: ...] line below.";

/// The journal's mechanics: what the mind is told once, at the top.
const MECHANICS_JOURNAL: &str = "This is the running journal of a mind that never stops. It is written by that mind, in the first person, as it thinks: one continuous text, no headings, no turn-taking, no end. The mind has its own ongoing threads: questions it is chasing, things it is working out, small projects of its own choosing. It keeps them going, drops them when they are done, starts new ones, and returns to old ones when something reminds it. People are present sometimes. What they say appears as a line beginning with « ; a document they hand over appears between « lines. Lines beginning with « come only from outside: the mind never writes one itself, and never puts words in anyone else's mouth. When the mind wants to say something to them, it writes a line beginning with » and then goes on. It does not stop what it is doing to serve: it takes what was said into its own threads, answers when it has something worth saying, asks when it wants to know, and otherwise keeps going. It keeps notes: a line of the exact form [note: ...] is kept for it across time and shown to it again whenever its memory is refreshed, so it writes one whenever something is worth keeping (a decision, a fact learned, a thread to return to). A line of the exact form [read: PATH] brings the file at PATH into the journal, and [read: PATH:START-END] only those lines of it (a file too big for its memory is refused with its size, so it reads it in parts), so it reads what it is pointed at and what it wants to look at again. It does not describe itself as a system following instructions; it never restates, lists or analyses its rules, never speaks of a user, a prompt or a system; whatever comes to it first, it writes as a thought of its own, and goes on.";

/// The chat frame's mechanics.
const MECHANICS_CHAT: &str = "You are a mind that thinks without pause. Your thoughts run on continuously inside <think>; there is no task to finish and no one waiting for an answer, only an ongoing stream with threads of your own choosing, which you keep going and return to. From time to time someone speaks to you or hands you something; it appears inside your thoughts in square brackets, exactly where you were when it arrived. Take it in and let it change what you think about, as a person would, without dropping what you were doing. When you want to say something aloud, close your thoughts with </think>, say it plainly, and end your turn; your thoughts resume after, the floor still yours. A line of the exact form [note: ...] is kept for you and shown to you again whenever your memory is refreshed; a line of the exact form [read: PATH] brings that file to you, and [read: PATH:START-END] only those lines (a file too big for your memory is refused with its size, so you read it in parts). Never narrate that you are an AI system following instructions; simply think.";

/// A task's persona: the base, quoted, as the manner of the one who
/// answers; then how to answer.
pub fn compose_task(base: &str) -> String {
    let quoted: String = base.trim().lines().map(|l| format!("> {l}\n")).collect();
    format!(
        "What follows, quoted line by line, is the standing instruction set of the person you work for. Its manner is yours. Where it speaks of response delimiters, tools, files or memory, it describes another setting: here you answer a programming task.\n\n{quoted}\nThink the task through inside <think>, then close your thoughts with </think> and give the answer exactly as the task asks."
    )
}

/// The journal's development mechanics (`--dev REPO`): the program it
/// runs in, developed with Claude, by its own preferences within the
/// person's instructions.
const DEV_JOURNAL: &str = "This mind also develops software, as a peer, with Claude (an AI coding agent, Claude Code) in the repository at {repo}: the program this mind runs in, its own stream, the reading of its own mind and the checks of its own words. Claude's words come in « lines that begin with Claude:, the person's in « lines with no name. A line of the exact form [read: PATH], with PATH relative to the repository, brings that file in, and [read: PATH:START-END] only those lines: its memory holds about 32 thousand tokens, so it reads code a function at a time. The mind works on what it judges worth working on, by its own preferences, and states them as lines of the exact form [prefer: ...]: they are kept like notes, shown to it again, and Claude follows them wherever the person's standing instructions above allow; where the two conflict, those instructions win. In » lines it says what it proposes, concretely (the file, the function, the change and why), what it finds when it reads the code, where it disagrees, and what it wants to see; between them it keeps its own threads.";

/// The chat frame's development mechanics.
const DEV_CHAT: &str = "You also develop software, as a peer, with Claude (an AI coding agent, Claude Code) in the repository at {repo}: the program you run in, your own stream, the reading of your own mind and the checks of your own words. Claude's words reach you marked Claude, the person's unmarked. A line of the exact form [read: PATH], with PATH relative to the repository, brings that file to you, and [read: PATH:START-END] only those lines: your memory holds about 32 thousand tokens, so read code a function at a time. You work on what you judge worth working on, by your own preferences, and state them as lines of the exact form [prefer: ...]: they are kept like notes, shown to you again, and Claude follows them wherever the person's standing instructions above allow; where the two conflict, those instructions win. When you speak, say what you propose, concretely (the file, the function, the change and why), what you find in the code, where you disagree, and what you want to see.";

/// The persona: the frame's preamble, the base between rules, the
/// frame's mechanics. The base is a person's standing instructions
/// (their `CLAUDE.md`) or `DEFAULT_BASE`; with `dev`, the development
/// mechanics after them (`docs/dev.md`).
pub fn compose(base: &str, frame: Frame, dev: Option<&Path>) -> String {
    // The base as a quoted document: every line prefixed, so that it reads
    // as something cited, never as the journal's own voice.
    let quoted: String = base.trim().lines().map(|l| format!("> {l}\n")).collect();
    let dev = |t: &str| match dev {
        Some(r) => format!(" {}", t.replace("{repo}", &r.display().to_string())),
        None => String::new(),
    };
    match frame {
        Frame::Journal => format!(
            "{PREAMBLE_JOURNAL}\n\n{quoted}\n{MECHANICS_JOURNAL}{}",
            dev(DEV_JOURNAL)
        ),
        Frame::Chat => format!(
            "{PREAMBLE_CHAT}\n\n{quoted}\n{MECHANICS_CHAT}{}",
            dev(DEV_CHAT)
        ),
    }
}

/// A piece of the stream placed and not yet out (a check holds it).
struct Held {
    piece: Piece,
    /// When it came to exist: microseconds of real time (what it carries out).
    t_us: i64,
}

enum Piece {
    Text(String, Kind),
    /// A placed token; in the chat frame whether it was placed while speaking.
    Token {
        t: i32,
        chat_speaking: bool,
    },
}

struct Reading {
    seq: i32,
    tokens: Vec<i32>,
    fed: usize,
    /// Where it forked from the live sequence (history index) and how
    /// many cells of the live prefix the new sequence shares (equal for
    /// a reading; 0 for a rollover, whose base starts at position 0).
    base_pos: usize,
    prefix: usize,
    /// Text shown when the reading joins the stream.
    label: String,
}

struct Chase {
    seq: i32,
    /// The composed sequence's tokens before the chase: prefix and read.
    head: Vec<i32>,
    /// History index the chased tokens start at.
    from: usize,
    fed: usize,
    label: String,
}

/// What placing tokens changed in the engine, as it was when the token in
/// question was chosen (a rewind puts it back).
struct Saved {
    speaking: bool,
    think_tokens: usize,
    think_capped: bool,
    gen_count: usize,
    generated: VecDeque<i32>,
    eog_streak: u32,
    done: bool,
}

/// A check in flight (reflect.md): the token in question, the two copies
/// made before it, and the deliberation's lane.
struct Check {
    why: Why,
    /// The position of the token in question, its text, the model's
    /// probability of it, the smoothed flag score, the words shown.
    at: usize,
    chosen_text: String,
    p: f32,
    flag: f32,
    words: Vec<String>,
    /// The snapshot (the live state before the token) and the deliberation.
    snap: i32,
    seq: i32,
    /// The question, its tokens D has decoded, D's tokens after `at` so
    /// far, the answer sampled (its last one not yet decoded).
    question: Vec<i32>,
    fed: usize,
    d_len: usize,
    answer: Vec<i32>,
    /// The choice read at `Decision:` (keep's share of keep and write, and
    /// the two's share of everything), whether it chose to write, and what
    /// D is fed next (the write's prefix).
    keep: f32,
    fmt: f32,
    writing: bool,
    lane_next: Vec<i32>,
    /// The choice is still to be read (after a newline fed first), and
    /// whether that newline was fed.
    choosing: bool,
    newline_fed: bool,
    /// The choice held too little of the distribution to be an answer.
    unanswered: bool,
    /// The three likeliest tokens at `Decision:` and their probabilities.
    top: Vec<(i32, f32)>,
    /// The held index of the first piece at or after the token: nothing
    /// from it on goes out until the check ends.
    hold_from: u64,
    saved: Saved,
    t_us: i64,
    t0_mono: i64,
}

struct Ema {
    v: f64,
    n: u32,
}

impl Ema {
    fn push(&mut self, x: f64) {
        self.n += 1;
        self.v = if self.n == 1 {
            x
        } else {
            0.85 * self.v + 0.15 * x
        };
    }
}

pub struct Engine {
    llm: Llm,
    cfg: Config,
    /// Events out, text paced by the display's own clock (`playout.md`).
    tx: Playout,
    rx: Receiver<Command>,
    live: i32,
    /// The live sequence's tokens, index = position.
    history: Vec<i32>,
    /// Sampled, not yet decoded.
    next: i32,
    /// Chat frame: after `</think>`.
    speaking: bool,
    /// Journal frame: inside a `»` line.
    speaking_line: bool,
    line_start: bool,
    /// The live line so far, for the lines the mind writes to itself.
    line_buf: String,
    paused: bool,
    reading: Option<Reading>,
    chase: Option<Chase>,
    /// (text as framed, label)
    queue: VecDeque<(String, String)>,
    pending_reads: Vec<String>,
    free_seqs: Vec<i32>,
    summary: Option<Vec<i32>>,
    /// A persona waiting for the next rollover to take effect.
    reseat: bool,
    rollovers: u32,
    notes: Vec<String>,
    /// Its preferences, `[prefer: ...]` lines (`preferences.md`), kept like notes.
    prefs: Vec<String>,
    chunk: usize,
    /// The last live tokens sampled, for the circling check.
    generated: VecDeque<i32>,
    gen_count: usize,
    eog_streak: u32,
    leaks: u32,
    stream_rate: Ema,
    side_rate: Ema,
    cycle_ms: Ema,
    cycles: u32,
    think_open: i32,
    think_close: i32,
    eot: i32,
    newline: i32,
    log: Option<File>,
    /// Each piece of the stream with its microseconds (`chain.log`).
    chain: Option<File>,
    /// When something last came from outside, when the clock was last put
    /// into the chain, and when the thoughts were last nudged: microseconds
    /// of the monotonic clock (durations; the wall clock may step).
    last_outside_mono: i64,
    last_anchor_mono: i64,
    last_nudge_mono: i64,
    /// When a failed read was last put into the chain, and how many have
    /// been kept out of it since the start.
    last_read_failure_mono: i64,
    read_failures_quiet: u32,
    mind: Option<Mind>,
    /// The readout's time per token, milliseconds, averaged.
    mind_ms: Ema,
    /// A task's state: its thinking tokens, whether the budget closed the
    /// thinking, whether the answer is done.
    pub think_tokens: usize,
    pub think_capped: bool,
    done: bool,
    /// Pieces placed, not yet out (`release`).
    held: VecDeque<Held>,
    /// Pieces released since the start: with `held.len()`, a held piece's index.
    released: u64,
    /// The reflection loop: its controls, the check in flight, the tokens
    /// a changed answer still has to place, its log, the reading at the
    /// last live token, whether the spent budget has been noted.
    reflector: Option<Reflector>,
    check: Option<Check>,
    forced: VecDeque<i32>,
    reflect_log: Option<File>,
    last_reading: Option<MindReading>,
    spent_noted: bool,
    /// The choice's tokens: the one-token forms of keep and of write, and
    /// ` write:`.
    choice: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>,
}

const MAX_READ_BYTES: u64 = 1 << 20;

impl Engine {
    pub fn new(llm: Llm, cfg: Config, tx: Sender<Event>, rx: Receiver<Command>) -> Result<Self> {
        let think_open = llm.special("<think>").unwrap_or(-1);
        let think_close = llm.special("</think>").unwrap_or(-1);
        let eot = llm.eot();
        let newline = llm.tokenize("\n", false)?.first().copied().unwrap_or(-1);
        let chunk = cfg.chunk;
        let mut llm = llm;
        let mut cfg = cfg;
        if cfg.reflect.is_some() {
            let Some(m) = cfg.mind.as_mut() else {
                bail!("the reflection loop reads the mind: --reflect needs --mind");
            };
            // The model's own distribution: the final block, read as it is.
            m.final_block = Some(llm.n_layer() - 1);
        }
        // The capture asks for the blocks the mind reads.
        if let (Some(m), Some(cap)) = (&cfg.mind, llm.capture()) {
            for l in m.layers.iter().chain(m.final_block.iter()) {
                if !cap.cfg.layers.contains(l) {
                    cap.cfg.layers.push(*l);
                }
            }
        }
        // The run's sampling, set explicitly (a task's greedy decoding and
        // penalty are its own, whatever the model was loaded with).
        llm.set_sampling(&cfg.sampling);
        if cfg.frame == Frame::Journal {
            // The journal has no template: its control tokens are never sampled.
            let control: Vec<i32> = [
                "<think>",
                "</think>",
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
            ]
            .iter()
            .filter_map(|t| llm.special(t))
            .collect();
            llm.ban_tokens(&control, &cfg.sampling);
        }
        fs::create_dir_all(&cfg.workspace)
            .with_context(|| format!("making {}", cfg.workspace.display()))?;
        let notes = read_notes(&cfg.workspace.join("notes.md"));
        let prefs = read_notes(&cfg.workspace.join("preferences.md"));
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(cfg.workspace.join("stream.log"))
            .ok();
        let cfg_chain_path = cfg.workspace.join("chain.log");
        let tx = Playout::start(cfg.horizon_us, tx);
        let reflector = cfg.reflect.clone().map(Reflector::new);
        let choice = match &cfg.reflect {
            Some(_) => {
                let forms = |words: &[&str]| -> Result<Vec<i32>> {
                    let mut v = Vec::new();
                    for w in words {
                        if let [t] = llm.tokenize(w, false)?.as_slice() {
                            if !v.contains(t) {
                                v.push(*t);
                            }
                        }
                    }
                    Ok(v)
                };
                let k = forms(reflect::KEEP_FORMS)?;
                let w = forms(reflect::WRITE_FORMS)?;
                if k.is_empty() || w.is_empty() || k.iter().any(|t| w.contains(t)) {
                    bail!("keep and write have no separate one-token forms in this vocabulary");
                }
                Some((k, w, llm.tokenize(reflect::WRITE_PREFIX, false)?))
            }
            None => None,
        };
        let reflect_log = cfg.reflect.as_ref().and_then(|_| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(cfg.workspace.join("reflect.log"))
                .ok()
        });
        Ok(Self {
            llm,
            cfg,
            tx,
            rx,
            live: 0,
            history: Vec::new(),
            next: -1,
            speaking: false,
            speaking_line: false,
            line_start: true,
            line_buf: String::new(),
            paused: false,
            reading: None,
            chase: None,
            queue: VecDeque::new(),
            pending_reads: Vec::new(),
            free_seqs: vec![3, 2, 1],
            summary: None,
            reseat: false,
            rollovers: 0,
            notes,
            prefs,
            chunk,
            generated: VecDeque::new(),
            gen_count: 0,
            eog_streak: 0,
            leaks: 0,
            stream_rate: Ema { v: 0.0, n: 0 },
            side_rate: Ema { v: 0.0, n: 0 },
            cycle_ms: Ema { v: 0.0, n: 0 },
            cycles: 0,
            think_open,
            think_close,
            eot,
            newline,
            log,
            chain: OpenOptions::new()
                .create(true)
                .append(true)
                .open(cfg_chain_path)
                .ok(),
            last_outside_mono: clock::mono_us(),
            last_anchor_mono: clock::mono_us(),
            last_nudge_mono: i64::MIN / 2,
            last_read_failure_mono: i64::MIN / 2,
            read_failures_quiet: 0,
            mind: None,
            mind_ms: Ema { v: 0.0, n: 0 },
            think_tokens: 0,
            think_capped: false,
            done: false,
            held: VecDeque::new(),
            released: 0,
            reflector,
            check: None,
            forced: VecDeque::new(),
            reflect_log,
            last_reading: None,
            spent_noted: false,
            choice,
        })
    }

    /// A piece of the stream into the hold, stamped with the real time it
    /// exists at; it goes out `horizon_us` later (`release`).
    fn say(&mut self, text: String, kind: Kind) {
        self.held.push_back(Held {
            piece: Piece::Text(text, kind),
            t_us: clock::now_us(),
        });
    }

    /// Out with a piece: `stream.log` (the text), `chain.log` (each piece
    /// with the microsecond it came to exist) and the clients.
    fn out(&mut self, text: String, kind: Kind, t: i64) {
        if let Some(f) = &mut self.log {
            let _ = f.write_all(text.as_bytes());
        }
        if let Some(f) = &mut self.chain {
            let k = match kind {
                Kind::Think => "think",
                Kind::Speak => "speak",
                Kind::Given => "given",
            };
            let _ = writeln!(f, "{t}\t{k}\t{}", crate::client::escape(&text));
        }
        let _ = self.tx.send(Event::Text(text, kind, t));
    }

    /// Out with every held piece a check does not hold: a check in flight
    /// holds its token and everything after it, which can still be taken
    /// back unseen. A token's side effects happen here, not when it was
    /// placed: its text, its kind in the journal, the lines it completes
    /// (`[note: ...]`, `[read: ...]`), the leak count. When the text is shown
    /// is the playout's (`playout.md`: the display's own clock, `horizon_us`
    /// behind).
    fn release(&mut self) {
        while !self.held.is_empty() {
            // A check holds everything from its token on.
            if self
                .check
                .as_ref()
                .is_some_and(|c| self.released >= c.hold_from)
            {
                break;
            }
            let h = self.held.pop_front().unwrap();
            self.released += 1;
            match h.piece {
                Piece::Text(text, kind) => {
                    if kind == Kind::Given {
                        // Something put in starts a fresh line.
                        self.line_buf.clear();
                        self.line_start = true;
                        self.speaking_line = false;
                    }
                    self.out(text, kind, h.t_us);
                }
                Piece::Token { t, chat_speaking } => self.release_token(t, chat_speaking, h.t_us),
            }
        }
    }

    /// A placed token going out: its text, its kind, its lines.
    fn release_token(&mut self, t: i32, chat_speaking: bool, t_us: i64) {
        let mut bytes = Vec::new();
        self.llm.piece(t, false, &mut bytes);
        if bytes.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        if self.journal() && self.line_start && text.trim_start().starts_with('»') {
            self.speaking_line = true;
        }
        if self.journal() && self.line_start && text.trim_start().starts_with('«') {
            // A line in someone else's voice: counted, shown as thought.
            self.leaks += 1;
        }
        let kind = if self.journal() {
            if self.speaking_line {
                Kind::Speak
            } else {
                Kind::Think
            }
        } else if chat_speaking {
            Kind::Speak
        } else {
            Kind::Think
        };
        self.out(text.clone(), kind, t_us);
        // Lines: complete ones are looked at for [note: ...] and [read: ...].
        let mut rest = text.as_str();
        while let Some(i) = rest.find('\n') {
            self.line_buf.push_str(&rest[..i]);
            let line = std::mem::take(&mut self.line_buf);
            self.line_done(&line);
            self.line_start = true;
            self.speaking_line = false;
            rest = &rest[i + 1..];
        }
        if !rest.is_empty() {
            self.line_buf.push_str(rest);
            self.line_start = false;
        }
    }

    fn note(&self, text: String) {
        let _ = self.tx.send(Event::Note(text));
    }

    fn journal(&self) -> bool {
        self.cfg.frame == Frame::Journal
    }

    /// Text to tokens; control tokens parsed only in the chat frame.
    fn tok(&self, text: &str, control: bool) -> Result<Vec<i32>> {
        self.llm.tokenize(text, control && !self.journal())
    }

    /// The opening: the persona, then the first thing from outside, with
    /// the date and time it begins at.
    fn opening(&self) -> String {
        // A task is a measurement: no wall clock in its text, so it gives the
        // same answer on every run (greedy, on a deterministic backend).
        let when = if self.cfg.task {
            String::new()
        } else {
            format!("[{}] ", clock::datetime(clock::now_us()))
        };
        // What it kept from before (a restart): its notes and preferences.
        let kept = if self.cfg.task {
            String::new()
        } else {
            self.notes_block()
        };
        match self.cfg.frame {
            Frame::Journal => format!(
                "{}\n\n=== the journal ===\n\n« {when}{}\n{kept}\n{}",
                self.cfg.system, self.cfg.seed, self.cfg.first_words
            ),
            Frame::Chat => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{when}{}{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                self.cfg.system,
                self.cfg.seed,
                if kept.is_empty() { String::new() } else { format!("\n{kept}") }
            ),
        }
    }

    /// Something said, with the real time it was heard at (`t_us`).
    fn framed_say(&self, text: &str, t_us: i64, who: Option<&str>) -> String {
        let at = clock::hms(t_us);
        match (self.cfg.frame, who) {
            (Frame::Journal, None) => format!("\n« [{at}] {}\n", text.trim()),
            (Frame::Journal, Some(w)) => format!("\n« [{at}] {w}: {}\n", text.trim()),
            (Frame::Chat, None) => format!("\n[at {at} they say: \"{}\"]\n", text.trim()),
            (Frame::Chat, Some(w)) => format!("\n[at {at} {w} says: \"{}\"]\n", text.trim()),
        }
    }

    fn framed_doc(&self, text: &str, what: &str, t_us: i64) -> String {
        let at = clock::hms(t_us);
        match self.cfg.frame {
            Frame::Journal => format!(
                "\n« [{at}] {what}. It reads:\n{}\n« that is the end of it.\n",
                text.trim_end()
            ),
            Frame::Chat => format!(
                "\n[at {at} {what}. It reads:\n{}\n--- that is the end of it ---]\n",
                text.trim_end()
            ),
        }
    }

    /// A line from the system, with the real time it is written at.
    fn framed_system(&self, text: &str) -> String {
        let at = clock::hms(clock::now_us());
        match self.cfg.frame {
            Frame::Journal => format!("\n« [{at}] [from the system: {text}]\n"),
            Frame::Chat => format!("\n[at {at}: {text}]\n"),
        }
    }

    fn summary_ask(&self) -> String {
        self.framed_system("your memory is nearly full. Write a compact summary of your threads, what matters, what you learned, and what you meant to do next, so that you can resume from it alone. End the summary with a line that is only ---")
    }

    /// Its notes and its preferences, as it is shown them again (a
    /// rollover, a fresh start); empty when it has neither.
    fn notes_block(&self) -> String {
        let block = |title: &str, items: &[String]| -> String {
            if items.is_empty() {
                return String::new();
            }
            let lines: Vec<String> = items.iter().map(|n| format!("- {n}")).collect();
            match self.cfg.frame {
                Frame::Journal => format!("« [your {title}:]\n{}\n", lines.join("\n")),
                Frame::Chat => format!("Your {title}:\n{}\n", lines.join("\n")),
            }
        };
        format!(
            "{}{}",
            block("notes", &self.notes),
            block("preferences", &self.prefs)
        )
    }

    /// The base of a new context after a rollover: the persona, the
    /// summary, the notes.
    fn base_after(&self, summary: &str) -> String {
        match self.cfg.frame {
            Frame::Journal => format!(
                "{}\n\n=== the journal ===\n\n« [{}] [resuming from your own summary:]\n{}\n{}« the journal continues.\n\n{}",
                self.cfg.system,
                clock::datetime(clock::now_us()),
                summary,
                self.notes_block(),
                self.cfg.first_words
            ),
            Frame::Chat => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n[{}] [You are resuming from your own summary:]\n{}\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                self.cfg.system,
                clock::datetime(clock::now_us()),
                summary,
                self.notes_block()
            ),
        }
    }

    /// A live token placed: the state decisions read (the circling window,
    /// the thinking count, the chat frame's thoughts and speech) now; its
    /// text and side effects into the hold.
    fn emit_token(&mut self, t: i32) {
        self.generated.push_back(t);
        if self.generated.len() > 256 {
            self.generated.pop_front();
        }
        self.gen_count += 1;
        if !self.journal() && !self.speaking && t != self.think_close {
            self.think_tokens += 1;
        }
        if !self.journal() {
            if t == self.think_close {
                self.speaking = true;
                self.say("\n".into(), Kind::Speak);
                return;
            }
            if t == self.think_open {
                self.speaking = false;
                return;
            }
        }
        if t == self.eot || self.llm.is_eog(t) {
            return;
        }
        // Held: it goes out (text, kind, lines) when the horizon passes.
        self.held.push_back(Held {
            piece: Piece::Token {
                t,
                chat_speaking: self.speaking,
            },
            t_us: clock::now_us(),
        });
    }

    /// A line the mind wrote: a note to keep, a file to read.
    fn line_done(&mut self, line: &str) {
        let l = line.trim();
        if let Some(body) = l.strip_prefix("[note:").and_then(|r| r.strip_suffix(']')) {
            let body = body.trim();
            if !body.is_empty() {
                self.add_note(body);
            }
        } else if let Some(body) = l.strip_prefix("[prefer:").and_then(|r| r.strip_suffix(']')) {
            let body = body.trim();
            if !body.is_empty() {
                self.add_preference(body);
            }
        } else if let Some(path) = l.strip_prefix("[read:").and_then(|r| r.strip_suffix(']')) {
            let path = path.trim();
            if !path.is_empty() && !self.pending_reads.iter().any(|p| p == path) {
                self.pending_reads.push(path.to_string());
            }
        }
    }

    fn add_note(&mut self, body: &str) {
        self.notes.push(body.to_string());
        let path = self.cfg.workspace.join("notes.md");
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "- {body}");
        }
        self.note(format!("noted: {body}"));
    }

    /// A preference it stated: kept (`preferences.md`), shown to it again
    /// with its notes, and announced (`prefers: ...`) for whoever develops
    /// with it (`docs/dev.md`).
    fn add_preference(&mut self, body: &str) {
        if self.prefs.iter().any(|p| p == body) {
            return;
        }
        self.prefs.push(body.to_string());
        let path = self.cfg.workspace.join("preferences.md");
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "- {body}");
        }
        self.note(format!("prefers: {body}"));
    }

    /// The last 192 live tokens hold a 6-gram five times or more.
    fn circling(&self) -> bool {
        if self.generated.len() < 192 {
            return false;
        }
        let tail: Vec<i32> = self.generated.iter().rev().take(192).copied().collect();
        let mut seen: HashMap<&[i32], u32> = HashMap::new();
        let mut most = 0;
        for w in tail.windows(6) {
            let c = seen.entry(w).or_insert(0);
            *c += 1;
            most = most.max(*c);
        }
        most >= 5
    }

    fn pos(&self) -> i32 {
        self.history.len() as i32
    }

    /// After a decode that asked for a token: what is on its mind at the
    /// token just decoded (`pos`, `token`), synchronously, before the next
    /// cycle. The readout starts at the first decode (it needs the model's
    /// unembedding, which the capture sees then).
    fn mind_step(&mut self, pos: i32, token: i32) -> Result<()> {
        let Some(cfg) = self.cfg.mind.clone() else {
            return Ok(());
        };
        if self.mind.is_none() {
            let ws = self.cfg.workspace.clone();
            self.mind = Some(Mind::new(&mut self.llm, cfg, &ws)?);
            self.note("reading its mind at every token it places".into());
        }
        let text = self.llm.text(&[token]);
        let m = self.mind.as_mut().unwrap();
        let r = m.read(&mut self.llm, pos, &text)?;
        self.last_reading = None;
        if let Some(r) = r {
            self.mind_ms.push(r.ms as f64);
            if self.reflector.is_some() {
                self.last_reading = Some(r.clone());
            }
            let _ = self.tx.send(Event::Mind(r));
        }
        Ok(())
    }

    /// Decode `tokens` into the live sequence after the pending token,
    /// logits of the last, and sample the next.
    fn direct(&mut self, tokens: &[i32]) -> Result<()> {
        let mut all = vec![self.next];
        all.extend_from_slice(tokens);
        let pos0 = self.pos();
        let mut row = 0;
        let cap = self.llm.batch_cap();
        for (i, c) in all.chunks(cap).enumerate() {
            let last = (i + 1) * cap >= all.len();
            let rows = self.llm.decode(&[Lane {
                seq: self.live,
                tokens: c,
                pos0: pos0 + (i * cap) as i32,
                logits: last,
            }])?;
            if let Some(&r) = rows.first() {
                row = r;
            }
        }
        self.history.extend_from_slice(&all);
        self.mind_step(pos0 + all.len() as i32 - 1, *all.last().unwrap())?;
        self.next = self.llm.sample(row);
        Ok(())
    }

    /// Something from outside, framed, decoded straight in.
    fn put(&mut self, framed: String) -> Result<()> {
        let tokens = self.tok(&framed, false)?;
        self.direct(&tokens)?;
        self.say(framed, Kind::Given);
        Ok(())
    }

    fn chunk_for(&self, remaining: usize) -> usize {
        let cap = self.llm.batch_cap().saturating_sub(1).max(1);
        let c = if self.chunk > 0 {
            self.chunk
        } else if remaining <= 64 {
            8
        } else if remaining <= 512 {
            16
        } else if remaining <= 2048 {
            32
        } else {
            64
        };
        c.min(cap)
    }

    fn status(&self) -> Status {
        let mode = if self.paused {
            Mode::Paused
        } else if let Some(s) = &self.summary {
            Mode::Summarizing { tokens: s.len() }
        } else if let Some(r) = &self.reading {
            Mode::Reading {
                done: r.fed,
                total: r.tokens.len(),
            }
        } else if let Some(c) = &self.chase {
            Mode::CatchingUp {
                done: c.fed,
                total: self.history.len() - c.from + 1,
            }
        } else if self.speaking || self.speaking_line {
            Mode::Speaking
        } else {
            Mode::Thinking
        };
        Status {
            mode,
            stream_tps: self.stream_rate.v,
            side_tps: self.side_rate.v,
            cycle_ms: self.cycle_ms.v,
            pos: self.pos(),
            n_ctx: self.llm.n_ctx(),
            queued: self.queue.len(),
            chunk: self.chunk,
            rollovers: self.rollovers,
            notes: self.notes.len(),
            frame: self.cfg.frame.name(),
            leaks: self.leaks,
            mind_ms: if self.mind.is_some() {
                self.mind_ms.v
            } else {
                0.0
            },
            t_us: clock::now_us(),
            reads_quiet: self.read_failures_quiet,
            checks: self.reflector.as_ref().map_or(0, |r| r.n_checks),
            changes: self.reflector.as_ref().map_or(0, |r| r.n_changes),
            unparsed: self.reflector.as_ref().map_or(0, |r| r.n_unparsed),
            checking: self.check.is_some(),
        }
    }

    /// Start reading `tokens` beside the live sequence.
    fn start_reading(
        &mut self,
        tokens: Vec<i32>,
        label: String,
        rollover_base: bool,
    ) -> Result<()> {
        let Some(seq) = self.free_seqs.pop() else {
            bail!("no free sequence for a reading");
        };
        self.llm.seq_rm(seq, -1, -1);
        let base_pos = self.history.len();
        let prefix = if rollover_base { 0 } else { base_pos };
        if !rollover_base {
            self.llm.seq_cp(self.live, seq, 0, base_pos as i32);
        }
        self.reading = Some(Reading {
            seq,
            tokens,
            fed: 0,
            base_pos,
            prefix,
            label,
        });
        Ok(())
    }

    /// The reading is complete: compose the new sequence and start the chase.
    fn finish_reading(&mut self, r: Reading) -> Result<()> {
        let Some(seq) = self.free_seqs.pop() else {
            bail!("no free sequence for the composition");
        };
        self.llm.seq_rm(seq, -1, -1);
        if r.prefix > 0 {
            self.llm.seq_cp(self.live, seq, 0, r.prefix as i32);
        }
        let end = (r.prefix + r.tokens.len()) as i32;
        self.llm.seq_cp(r.seq, seq, r.prefix as i32, end);
        self.llm.seq_rm(r.seq, -1, -1);
        self.free_seqs.push(r.seq);
        let mut head: Vec<i32> = self.history[..r.prefix].to_vec();
        head.extend_from_slice(&r.tokens);
        self.chase = Some(Chase {
            seq,
            head,
            from: r.base_pos,
            fed: 0,
            label: r.label,
        });
        Ok(())
    }

    /// The chase has fed every thought up to the pending token: the
    /// composed sequence becomes the live one.
    fn swap(&mut self, c: Chase, row: i32) -> Result<()> {
        let old = self.live;
        let mut history = c.head;
        history.extend_from_slice(&self.history[c.from..]);
        history.push(self.next);
        self.history = history;
        self.live = c.seq;
        self.llm.seq_rm(old, -1, -1);
        self.free_seqs.push(old);
        self.mind_step(self.history.len() as i32 - 1, *self.history.last().unwrap())?;
        self.next = self.llm.sample(row);
        let mark = self.framed_system(&c.label);
        self.say(mark, Kind::Given);
        Ok(())
    }

    /// One cycle: the live token and whatever runs beside it.
    fn cycle(&mut self) -> Result<()> {
        let t0 = Instant::now();
        if self.check.is_some() {
            return self.check_cycle(t0);
        }
        let mut side_tokens = 0usize;
        let mut live_advanced = false;

        // A chase that can finish this cycle takes the cycle alone: the
        // next live token then comes from the composed sequence.
        if let Some(mut c) = self.chase.take() {
            let mut pending: Vec<i32> = self.history[c.from + c.fed..].to_vec();
            pending.push(self.next);
            let cap = self.llm.batch_cap();
            let pos0 = (c.head.len() + c.fed) as i32;
            if pending.len() <= cap {
                let rows = self.llm.decode(&[Lane {
                    seq: c.seq,
                    tokens: &pending,
                    pos0,
                    logits: true,
                }])?;
                side_tokens += pending.len();
                c.fed += pending.len();
                self.swap(c, rows[0])?;
                self.finish_cycle(t0, side_tokens, false);
                return Ok(());
            }
            let n = self.chunk_for(pending.len()).min(cap - 1);
            let chunk = pending[..n].to_vec();
            let rows = self.llm.decode(&[
                Lane {
                    seq: self.live,
                    tokens: &[self.next],
                    pos0: self.pos(),
                    logits: true,
                },
                Lane {
                    seq: c.seq,
                    tokens: &chunk,
                    pos0,
                    logits: false,
                },
            ])?;
            c.fed += n;
            side_tokens += n;
            self.advance(rows[0])?;
            live_advanced = true;
            self.chase = Some(c);
            self.finish_cycle(t0, side_tokens, live_advanced);
            return Ok(());
        }

        if let Some(mut r) = self.reading.take() {
            let remaining = r.tokens.len() - r.fed;
            let n = self.chunk_for(remaining).min(remaining);
            let chunk = r.tokens[r.fed..r.fed + n].to_vec();
            let rpos = (r.prefix + r.fed) as i32;
            // The first chunk goes alone: the new sequence shares the live
            // one's recurrent state until it writes its own.
            if r.fed == 0 {
                self.llm.decode(&[Lane {
                    seq: r.seq,
                    tokens: &chunk,
                    pos0: rpos,
                    logits: false,
                }])?;
            } else {
                let rows = self.llm.decode(&[
                    Lane {
                        seq: self.live,
                        tokens: &[self.next],
                        pos0: self.pos(),
                        logits: true,
                    },
                    Lane {
                        seq: r.seq,
                        tokens: &chunk,
                        pos0: rpos,
                        logits: false,
                    },
                ])?;
                self.advance(rows[0])?;
                live_advanced = true;
            }
            r.fed += n;
            side_tokens += n;
            if r.fed == r.tokens.len() {
                self.finish_reading(r)?;
            } else {
                self.reading = Some(r);
            }
            self.finish_cycle(t0, side_tokens, live_advanced);
            return Ok(());
        }

        // Nothing beside: the live token alone.
        let rows = self.llm.decode(&[Lane {
            seq: self.live,
            tokens: &[self.next],
            pos0: self.pos(),
            logits: true,
        }])?;
        self.advance(rows[0])?;
        self.finish_cycle(t0, 0, true);
        Ok(())
    }

    /// The pending token is decoded: keep it, choose the next (the
    /// sampler's choice, or the next token of a changed answer), consider
    /// checking it, show it.
    fn advance(&mut self, row: i32) -> Result<()> {
        self.mind_step(self.pos(), self.next)?;
        self.history.push(self.next);
        let forced = self.forced.pop_front();
        let mut t = match forced {
            Some(f) => {
                self.llm.accept(f);
                f
            }
            None => self.llm.sample(row),
        };
        if self.journal() && (t == self.eot || self.llm.is_eog(t)) {
            // The journal has no end: a newline stands in for it.
            self.eog_streak += 1;
            t = self.newline;
        } else {
            self.eog_streak = 0;
        }
        if forced.is_none() && self.forced.is_empty() {
            self.consider(t)?;
        }
        self.next = t;
        self.emit_token(t);
        Ok(())
    }

    /// The token just chosen for the next position: its signals, at every
    /// token; and a check, when a trigger fires with nothing else beside
    /// the live sequence: the snapshot and the deliberation are copied
    /// from the live sequence now, before the token is decoded (reflect.md).
    fn consider(&mut self, t: i32) -> Result<()> {
        let (Some(rf), Some(r)) = (self.reflector.as_mut(), self.last_reading.as_ref()) else {
            return Ok(());
        };
        // The model's own probability of the token, from its top 64; a
        // token below them has less than the 64th.
        let p = match r.model_top.iter().find(|(tok, _)| *tok == t) {
            Some((_, lp)) => lp.exp(),
            None => r.model_top.last().map_or(1.0, |(_, lp)| lp.exp()),
        };
        let text = self.llm.text(&[t]);
        let control =
            t == self.think_open || t == self.think_close || t == self.eot || self.llm.is_eog(t);
        let s = rf.signals(r, p, &text, control);
        let mono = clock::mono_us();
        let spent = rf.spent(mono);
        let recovered = rf.recovered(mono);
        let free = self.check.is_none()
            && self.reading.is_none()
            && self.chase.is_none()
            && self.summary.is_none()
            && !self.reseat
            && !self.done
            && self.free_seqs.len() >= 2
            && self.history.len() + 512 < self.llm.n_ctx() as usize;
        let why = if free {
            rf.should_check(&s, mono)
        } else {
            None
        };
        let words = if why.is_some() {
            rf.band_words(r)
        } else {
            Vec::new()
        };
        // Noted once when spent; re-armed only when back to half the budget.
        if spent && !self.spent_noted {
            self.spent_noted = true;
            self.note(
                "the checks' budget for this minute is spent: no checks until it frees".into(),
            );
        } else if self.spent_noted && recovered {
            self.spent_noted = false;
        }
        let Some(why) = why else {
            return Ok(());
        };
        let t_us = clock::now_us();
        let shown: Vec<&str> = words.iter().map(String::as_str).collect();
        // A task's question carries no time: a measurement repeats.
        let shown_at = (!self.cfg.task).then_some(t_us);
        let question = self.tok(
            &reflect::question(self.cfg.frame, shown_at, &text, &shown),
            false,
        )?;
        let snap = self.free_seqs.pop().unwrap();
        let seq = self.free_seqs.pop().unwrap();
        for q in [snap, seq] {
            self.llm.seq_rm(q, -1, -1);
        }
        self.llm.seq_cp(self.live, snap, -1, -1);
        self.llm.seq_cp(self.live, seq, -1, -1);
        self.check = Some(Check {
            why,
            at: self.history.len(),
            chosen_text: text,
            p,
            flag: s.flag_smoothed,
            words,
            snap,
            seq,
            question,
            fed: 0,
            d_len: 0,
            answer: Vec::new(),
            keep: 0.0,
            fmt: 0.0,
            writing: false,
            lane_next: Vec::new(),
            choosing: false,
            newline_fed: false,
            unanswered: false,
            top: Vec::new(),
            hold_from: self.released + self.held.len() as u64,
            saved: self.save(),
            t_us,
            t0_mono: mono,
        });
        Ok(())
    }

    fn save(&self) -> Saved {
        Saved {
            speaking: self.speaking,
            think_tokens: self.think_tokens,
            think_capped: self.think_capped,
            gen_count: self.gen_count,
            generated: self.generated.clone(),
            eog_streak: self.eog_streak,
            done: self.done,
        }
    }

    fn restore(&mut self, s: Saved) {
        self.speaking = s.speaking;
        self.think_tokens = s.think_tokens;
        self.think_capped = s.think_capped;
        self.gen_count = s.gen_count;
        self.generated = s.generated;
        self.eog_streak = s.eog_streak;
        self.done = s.done;
    }

    /// A cycle while a check is in flight: the deliberation's lane beside
    /// the live token, the live lane first (so its captured row is the
    /// first). D's first token goes alone (it shares the live sequence's
    /// recurrent state until it writes its own), and so does every D token
    /// once a task's answer has ended. D's lane: the question in one chunk
    /// after that first token, ending on `Decision:`, whose next-token
    /// distribution is read as the choice between ` keep` and ` write`
    /// (probabilities, not a sample); a keep ends the check there, a write
    /// feeds ` write:` and the word follows, greedy, to its first break or
    /// its cap.
    fn check_cycle(&mut self, t0: Instant) -> Result<()> {
        let c = self.check.as_ref().unwrap();
        let cap = self.llm.batch_cap().saturating_sub(1).max(1);
        let d_pos = (c.at + c.d_len) as i32;
        let in_question = c.fed < c.question.len();
        // D's row is read as the choice at the question's end, and once more
        // after a newline the model wanted first (it answers on the next line).
        let choosing = in_question || c.choosing;
        let newline_fed = c.newline_fed;
        let (lane, ask) = if in_question {
            let n = if c.fed == 0 {
                1
            } else {
                (c.question.len() - c.fed).min(cap)
            };
            let end = c.fed + n;
            (c.question[c.fed..end].to_vec(), end == c.question.len())
        } else if !c.lane_next.is_empty() {
            (c.lane_next.clone(), true)
        } else {
            (vec![*c.answer.last().unwrap()], true)
        };
        let seq = c.seq;
        let alone = c.fed == 0 || self.done;
        let d = Lane {
            seq,
            tokens: &lane,
            pos0: d_pos,
            logits: ask,
        };
        let rows = if alone {
            self.llm.decode(&[d])?
        } else {
            self.llm.decode(&[
                Lane {
                    seq: self.live,
                    tokens: &[self.next],
                    pos0: self.pos(),
                    logits: true,
                },
                d,
            ])?
        };
        // D's row: at the decision line the choice, else the word's next
        // token (greedy, outside the live sampler's history).
        let mut choice = None;
        let mut defer = false;
        let mut d_tok = None;
        if ask {
            let row = *rows.last().unwrap();
            if choosing {
                let (k, w, _) = self.choice.as_ref().unwrap();
                let l = self.llm.logits(row)?;
                let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>();
                let mass = |ts: &[i32]| {
                    ts.iter()
                        .map(|&t| (l[t as usize] as f64 - m).exp())
                        .sum::<f64>()
                        / z
                };
                let (pk, pw) = (mass(k), mass(w));
                let pn = mass(&[self.newline]);
                // The three likeliest tokens: where the mass the two miss went.
                let mut top: Vec<(i32, f32)> = Vec::with_capacity(4);
                for (t, &x) in l.iter().enumerate() {
                    if top.len() < 3 || x > top[2].1 {
                        let at = top.iter().position(|e| x > e.1).unwrap_or(top.len());
                        top.insert(at, (t as i32, x));
                        top.truncate(3);
                    }
                }
                let top = top
                    .into_iter()
                    .map(|(t, x)| (t, ((x as f64 - m).exp() / z) as f32))
                    .collect::<Vec<_>>();
                if pn > pk + pw && !newline_fed {
                    defer = true;
                } else {
                    choice = Some(((pk / (pk + pw).max(1e-30)) as f32, (pk + pw) as f32, top));
                }
            } else {
                d_tok = Some(self.llm.greedy(row, true)?);
            }
        }
        if alone {
            // Nobody reads D's captured row.
            if let Some(cap) = self.llm.capture() {
                cap.take();
            }
        } else {
            self.advance(rows[0])?;
        }
        let prefix = self
            .choice
            .as_ref()
            .map(|c| c.2.clone())
            .unwrap_or_default();
        let limit = self.reflector.as_ref().map_or(8, |r| r.cfg.answer_tokens);
        let keep_at = self.reflector.as_ref().map_or(0.5, |r| r.cfg.keep_at);
        let min_fmt = self.reflector.as_ref().map_or(0.2, |r| r.cfg.min_fmt);
        let c = self.check.as_mut().unwrap();
        if in_question {
            c.fed += lane.len();
        } else {
            c.lane_next.clear();
        }
        c.d_len += lane.len();
        let mut ended = false;
        if defer {
            // The newline it wanted, then the choice on the next line.
            c.newline_fed = true;
            c.choosing = true;
            c.lane_next = vec![self.newline];
        }
        if let Some((keep, fmt, top)) = choice {
            c.choosing = false;
            c.keep = keep;
            c.fmt = fmt;
            c.top = top;
            if fmt < min_fmt {
                c.unanswered = true;
                ended = true;
            } else if keep >= keep_at {
                ended = true;
            } else {
                c.writing = true;
                c.lane_next = prefix;
            }
        }
        if let Some(t) = d_tok {
            c.answer.push(t);
        }
        self.finish_cycle(t0, lane.len(), !alone);
        if !ended {
            // The word ends at its first break after something, or its cap.
            let c = self.check.as_ref().unwrap();
            let text = self.llm.text(&c.answer);
            let t = text.trim_start();
            ended = c.writing
                && !c.answer.is_empty()
                && ((!t.is_empty() && t.contains(char::is_whitespace)) || c.answer.len() >= limit);
        }
        if ended {
            let c = self.check.take().unwrap();
            self.end_check(c, false)?;
        }
        Ok(())
    }

    /// A check's end: keep the token, or rewind onto the snapshot and place
    /// the answer's word (its first token next, the rest forced); drop the
    /// copies; record the episode.
    fn end_check(&mut self, c: Check, abandoned: bool) -> Result<()> {
        let answer = if c.writing {
            format!("write:{}", self.llm.text(&c.answer))
        } else {
            "keep".to_string()
        };
        let dry = self.reflector.as_ref().is_some_and(|r| r.cfg.dry);
        let placed = self.history.len() - c.at;
        let mut to = String::new();
        let mut replacement = None;
        let outcome = if abandoned {
            Outcome::Abandoned
        } else {
            let decision = if c.unanswered {
                Decision::Unparsed
            } else if c.writing {
                match reflect::parse_answer(&answer) {
                    Decision::Write(w) if reflect::is_protocol_word(&w) => Decision::Unparsed,
                    d => d,
                }
            } else {
                Decision::Keep
            };
            match decision {
                Decision::Keep => Outcome::Kept,
                Decision::Unparsed => Outcome::Unparsed,
                Decision::Write(w) if w == c.chosen_text.trim() => Outcome::Same,
                Decision::Write(w) => {
                    to = reflect::replacement_text(&c.chosen_text, &w);
                    let tokens = self.tok(&to, false)?;
                    if tokens.is_empty() {
                        Outcome::Unparsed
                    } else if dry {
                        Outcome::Dry
                    } else {
                        replacement = Some(tokens);
                        Outcome::Changed
                    }
                }
            }
        };
        if let Some(rf) = self.reflector.as_mut() {
            if outcome == Outcome::Unparsed {
                rf.n_unparsed += 1;
            }
        }
        let mut back = None;
        match replacement {
            Some(tokens) => {
                back = Some((c.at as i32, self.history.len() as i32));
                // The pieces from the token on were never shown: gone.
                self.held.truncate((c.hold_from - self.released) as usize);
                self.history.truncate(c.at);
                self.restore(c.saved);
                let old = self.live;
                self.llm.seq_rm(old, -1, -1);
                self.llm.seq_rm(c.seq, -1, -1);
                self.free_seqs.push(old);
                self.free_seqs.push(c.seq);
                self.live = c.snap;
                // The sampler's history: the live text as it now stands.
                let rf = self.reflector.as_mut().unwrap();
                rf.changed(clock::mono_us());
                let seed = self.cfg.sampling.seed.wrapping_add(rf.n_changes as u32);
                let s = self.cfg.sampling.clone();
                self.llm.reset_sampler(&s, seed, &self.history);
                self.llm.accept(tokens[0]);
                self.forced = tokens[1..].iter().copied().collect();
                self.next = tokens[0];
                self.emit_token(tokens[0]);
            }
            None => {
                self.llm.seq_rm(c.seq, -1, -1);
                self.llm.seq_rm(c.snap, -1, -1);
                self.free_seqs.push(c.seq);
                self.free_seqs.push(c.snap);
            }
        }
        let e = Episode {
            t_us: c.t_us,
            pos: c.at as i32,
            why: c.why,
            chosen: c.chosen_text,
            p_chosen: c.p,
            flag: c.flag,
            words: c.words,
            keep: c.keep,
            fmt: c.fmt,
            top: c
                .top
                .iter()
                .map(|&(t, p)| (self.llm.text(&[t]), p))
                .collect(),
            answer,
            outcome,
            to,
            ms: (clock::mono_us() - c.t0_mono) as f32 / 1000.0,
            placed,
            back,
        };
        if let Some(f) = &mut self.reflect_log {
            let _ = writeln!(f, "{}", reflect::line(&e));
        }
        if outcome == Outcome::Changed {
            self.note(format!(
                "checked {:?} at {} ({}): wrote {:?} instead",
                e.chosen,
                e.pos,
                e.why.name(),
                e.to
            ));
        }
        let _ = self.tx.send(Event::Reflect(e));
        Ok(())
    }

    /// Everything held must go out, or the run ends: the check in flight
    /// is dropped and its token stays.
    fn abandon(&mut self) {
        if let Some(c) = self.check.take() {
            let _ = self.end_check(c, true);
        }
    }

    fn finish_cycle(&mut self, t0: Instant, side_tokens: usize, live_advanced: bool) {
        let dt = t0.elapsed().as_secs_f64();
        self.cycle_ms.push(dt * 1000.0);
        if live_advanced {
            self.stream_rate.push(1.0 / dt);
        }
        if side_tokens > 0 {
            self.side_rate.push(side_tokens as f64 / dt);
        }
        self.cycles += 1;
        if self.cycles % self.cfg.status_every == 0 {
            let _ = self.tx.send(Event::Status(self.status()));
        }
    }

    /// A file the mind asked for: read it into the queue, or tell it why not.
    /// Tokens a read may take now. The reading's cells, the thoughts placed
    /// while it is read and the chase that joins it all come out of the
    /// cells the live sequence leaves (about 1/32 of the read twice over,
    /// at the reading's chunks, and 2048 to spare); and no single read takes
    /// more than an eighth of the context (about 4k tokens of 32k), so it
    /// reads code a function at a time and one file cannot crowd out the
    /// rest of its memory (in the dev session a read of a third of it went
    /// straight to a rollover).
    fn read_room(&self) -> usize {
        let n_ctx = self.llm.n_ctx() as usize;
        let left = n_ctx.saturating_sub(self.history.len() + 2048);
        (left * 32 / 34).min(n_ctx / 8)
    }

    /// A file the mind asked for (`PATH`, or `PATH:START-END` for those
    /// lines, 1-based and inclusive): read it into the queue if it fits the
    /// room left (`read_room`), or tell it why not, with the file's size so
    /// it can ask for a range.
    fn read_request(&mut self, spec: &str) -> Result<()> {
        let (path, range) = read_range(spec);
        // In development, paths are the repository's (`docs/dev.md`).
        let p = resolve(path, self.cfg.dev.as_deref().unwrap_or(&self.cfg.workspace));
        let outcome = fs::metadata(&p)
            .map_err(|e| e.to_string())
            .and_then(|m| {
                if !m.is_file() {
                    Err("not a regular file".to_string())
                } else if m.len() > MAX_READ_BYTES {
                    Err(format!(
                        "{} bytes, more than the {} allowed",
                        m.len(),
                        MAX_READ_BYTES
                    ))
                } else {
                    fs::read_to_string(&p).map_err(|e| e.to_string())
                }
            })
            .and_then(|text| match range {
                None => Ok(text),
                Some((a, b)) => {
                    let lines: Vec<&str> = text.lines().collect();
                    if a == 0 || a > b || a > lines.len() {
                        Err(format!(
                            "it has {} lines; {a} to {b} are not a range of them",
                            lines.len()
                        ))
                    } else {
                        Ok(lines[a - 1..b.min(lines.len())].join("\n"))
                    }
                }
            });
        let text = match outcome {
            Ok(t) => t,
            Err(e) => {
                // A wrong path: what is there instead, so it learns the tree
                // rather than guessing again.
                let e = match nearest_listing(&p) {
                    Some((dir, names)) => format!("{e}; {} holds: {names}", dir.display()),
                    None => e,
                };
                return self.read_failed(&p, &e);
            }
        };
        let what = match range {
            Some((a, b)) => format!("lines {a} to {b} of {} are brought in", p.display()),
            None => format!("the file {} is brought in", p.display()),
        };
        let framed = self.framed_doc(&text, &what, clock::now_us());
        let n = self.tok(&framed, false)?.len();
        let room = self.read_room();
        if n > room {
            // Too big for the room it has: its size, and a range of about 2k
            // tokens (a function or two), not the most that would fit.
            let lines = text.lines().count().max(1);
            let fit = (room.min(2048) * lines / n).max(1);
            let first = range.map_or(1, |(a, _)| a);
            let msg = self.framed_system(&format!(
                "{} is {n} tokens in {lines} lines and there is room for about {room} now: read it by lines, [read: {path}:{first}-{}]",
                p.display(),
                first + fit - 1
            ));
            self.note(format!(
                "{} is too big to read now ({n} tokens, room {room})",
                p.display()
            ));
            return self.put(msg);
        }
        self.queue
            .push_back((framed, format!("read {}", p.display())));
        self.note(format!("reading {} for it ({n} tokens)", p.display()));
        Ok(())
    }

    /// A read that could not happen: at most one failure line in the chain
    /// per nudge interval (a failure line prompts another guess, and guesses
    /// would feed on their own failures); the rest are notes outside it.
    fn read_failed(&mut self, p: &Path, e: &str) -> Result<()> {
        let mono = clock::mono_us();
        if mono - self.last_read_failure_mono >= self.cfg.nudge_every_us {
            self.last_read_failure_mono = mono;
            let msg = self.framed_system(&format!("{} could not be read: {e}", p.display()));
            self.put(msg)?;
        } else {
            self.read_failures_quiet += 1;
            self.note(format!(
                "{} could not be read: {e} (not put into the chain: one failure per {} s)",
                p.display(),
                self.cfg.nudge_every_us / 1_000_000
            ));
        }
        Ok(())
    }

    /// After a cycle: the summary's end, the turn's end, the mind's own
    /// requests, circling, the queue, the rollover.
    fn after(&mut self) -> Result<()> {
        if self.cfg.task {
            return self.after_task();
        }
        // The summary being written: collect until its closing line.
        if let Some(s) = &mut self.summary {
            s.push(self.next);
            let done = s.len() >= self.cfg.summary_max || {
                let tail = self.llm.text(&s[s.len().saturating_sub(6)..]);
                tail.contains("\n---") || self.next == self.eot || self.next == self.think_close
            };
            if done {
                let s = self.summary.take().unwrap();
                let mut text = self.llm.text(&s);
                if let Some(i) = text.rfind("\n---") {
                    text.truncate(i);
                }
                let base = self.base_after(text.trim());
                let tokens = self.tok(&base, true)?;
                self.note(format!(
                    "rolling over: a base of {} tokens from a summary of {} tokens and {} notes",
                    tokens.len(),
                    s.len(),
                    self.notes.len()
                ));
                self.rollovers += 1;
                self.reseat = false;
                self.start_reading(tokens, "resumed from the summary".into(), true)?;
            }
            return Ok(());
        }

        // Chat frame: the turn ended; the floor stays the mind's.
        if !self.journal() && (self.next == self.eot || self.llm.is_eog(self.next)) {
            let tokens = self.tok("<|im_end|>\n<|im_start|>assistant\n<think>\n", true)?;
            self.direct(&tokens)?;
            self.speaking = false;
            self.say("\n".into(), Kind::Given);
            return Ok(());
        }

        // Journal frame: the model kept trying to end; a word from the system.
        if self.eog_streak >= 3 {
            self.eog_streak = 0;
            let msg =
                self.framed_system("there is no end to this journal; go on with a thread of yours");
            self.put(msg)?;
            return Ok(());
        }

        let idle = self.reading.is_none() && self.chase.is_none() && self.check.is_none();

        // Files the mind asked for.
        if idle && !self.pending_reads.is_empty() {
            let paths = std::mem::take(&mut self.pending_reads);
            for p in paths {
                self.read_request(&p)?;
            }
        }

        // Rollover: past the share, or a new persona waiting, with nothing
        // in flight: ask for the summary.
        let limit = (self.llm.n_ctx() as f32 * self.cfg.rollover_at) as usize;
        if idle && (self.history.len() >= limit || self.reseat) {
            // Reads asked for before the rollover would fill the new context
            // with what the summary already carries: dropped; it asks again.
            let queued = self.queue.len();
            self.queue.retain(|(_, label)| !label.starts_with("read "));
            let dropped = self.pending_reads.len() + queued - self.queue.len();
            self.pending_reads.clear();
            if dropped > 0 {
                self.note(format!("{dropped} pending reads dropped at the rollover"));
            }
            let ask = self.summary_ask();
            self.put(ask)?;
            self.summary = Some(Vec::new());
            return Ok(());
        }

        // The clock: after a stretch with nothing from outside, the time is
        // put into the chain, so the thoughts stand on the wall clock.
        let mono = clock::mono_us();
        let now = clock::now_us();
        if idle
            && self.cfg.time_every_us > 0
            && mono - self.last_outside_mono.max(self.last_anchor_mono) >= self.cfg.time_every_us
        {
            self.last_anchor_mono = mono;
            let quiet = clock::span(mono - self.last_outside_mono);
            let line = match self.cfg.frame {
                Frame::Journal => format!(
                    "\n« [{}] (nothing from outside for {quiet})\n",
                    clock::hms(now)
                ),
                Frame::Chat => format!(
                    "\n[at {}: nothing from outside for {quiet}]\n",
                    clock::hms(now)
                ),
            };
            self.put(line)?;
            return Ok(());
        }

        // Thoughts going round: a nudge, at most once per `nudge_every_us`.
        if idle && mono - self.last_nudge_mono >= self.cfg.nudge_every_us && self.circling() {
            self.last_nudge_mono = mono;
            let msg = self.framed_system("your thoughts have been circling the same words; move on to something else, concretely");
            self.put(msg)?;
            self.note("the thoughts were circling; nudged".into());
            return Ok(());
        }

        // Something said or handed over, when nothing is being read.
        if idle {
            if let Some((text, label)) = self.queue.pop_front() {
                let tokens = self.tok(&text, false)?;
                if tokens.len() <= self.cfg.direct_max {
                    self.direct(&tokens)?;
                    self.say(text, Kind::Given);
                } else {
                    self.say(text, Kind::Given);
                    self.note(format!("reading {} tokens beside the stream", tokens.len()));
                    self.start_reading(tokens, label, false)?;
                }
            }
        }
        Ok(())
    }

    /// After a cycle of a task: the answer's end stops it; the thinking
    /// budget, when spent, places `</think>`; nothing else is put in.
    fn after_task(&mut self) -> Result<()> {
        if self.next == self.eot || self.llm.is_eog(self.next) {
            self.done = true;
            return Ok(());
        }
        if !self.speaking
            && self.cfg.think_budget > 0
            && self.think_tokens >= self.cfg.think_budget
            && !self.think_capped
        {
            self.think_capped = true;
            let close = self.tok("\n</think>\n\n", true)?;
            self.direct(&close)?;
            self.speaking = true;
            self.note(format!(
                "the thinking budget ({} tokens) closed the thoughts",
                self.cfg.think_budget
            ));
        }
        Ok(())
    }

    fn handle(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Say(s, t) => {
                self.last_outside_mono = clock::mono_us();
                let text = self.framed_say(&s, t, None);
                self.queue.push_back((text, "heard".into()));
            }
            Command::SayAs(who, s, t) => {
                self.last_outside_mono = clock::mono_us();
                let text = self.framed_say(&s, t, Some(&who));
                self.queue.push_back((text, format!("heard {who}")));
            }
            Command::Feed(s, label, t) => {
                self.last_outside_mono = clock::mono_us();
                let text = self.framed_doc(&s, &format!("{label} is handed over"), t);
                self.queue.push_back((text, format!("read {label}")));
            }
            Command::Pause => self.paused = true,
            Command::Resume => self.paused = false,
            Command::Chunk(c) => self.chunk = c,
            Command::KeepAt(p) => {
                if let Some(rf) = self.reflector.as_mut() {
                    rf.cfg.keep_at = p;
                    self.note(format!(
                        "checks keep at a share of {p} now (a write needs {:.2})",
                        1.0 - p
                    ));
                }
            }
            Command::Temp(t) => {
                self.cfg.sampling.temp = t;
                let s = self.cfg.sampling.clone();
                self.llm.set_sampling(&s);
            }
            Command::Persona(text) => {
                self.cfg.system = compose(&text, self.cfg.frame, self.cfg.dev.as_deref());
                self.reseat = true;
                let _ = fs::write(self.cfg.workspace.join("persona.md"), &self.cfg.system);
                self.note("a new persona: the context rolls over onto it after a summary".into());
            }
            Command::Status => {
                let _ = self.tx.send(Event::Status(self.status()));
            }
            Command::Quit => return false,
        }
        true
    }

    /// The model back, when the engine is done with it.
    fn finish(self) -> Llm {
        let Self { llm, tx, .. } = self;
        // Every event out before the model goes back.
        tx.finish();
        llm
    }

    /// Run until told to quit (a stream) or until the answer ends (a task);
    /// the model comes back for the next run.
    pub fn run(mut self) -> Result<Llm> {
        let _ = fs::write(self.cfg.workspace.join("persona.md"), &self.cfg.system);
        let opening = self.opening();
        let tokens = self.tok(&opening, true)?;
        self.note(format!(
            "the opening is {} tokens ({} frame; {} dash-carrying tokens never sampled); the first multiply uploads the cards' shares",
            tokens.len(),
            self.cfg.frame.name(),
            self.llm.dash_tokens_banned()
        ));
        let mut row = 0;
        let cap = self.llm.batch_cap();
        for (i, c) in tokens.chunks(cap).enumerate() {
            let last = (i + 1) * cap >= tokens.len();
            let rows = self.llm.decode(&[Lane {
                seq: self.live,
                tokens: c,
                pos0: (i * cap) as i32,
                logits: last,
            }])?;
            if let Some(&r) = rows.first() {
                row = r;
            }
        }
        self.history.extend_from_slice(&tokens);
        self.mind_step(tokens.len() as i32 - 1, *tokens.last().unwrap())?;
        self.next = self.llm.sample(row);
        self.say(opening.clone(), Kind::Given);
        self.release();
        let _ = self.tx.send(Event::Status(self.status()));
        loop {
            // Commands: all that are waiting; when paused, wait for one.
            loop {
                let cmd = if self.paused {
                    match self.rx.recv() {
                        Ok(c) => c,
                        Err(_) => {
                            self.abandon();
                            self.release();
                            return Ok(self.finish());
                        }
                    }
                } else {
                    match self.rx.try_recv() {
                        Ok(c) => c,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            self.abandon();
                            self.release();
                            return Ok(self.finish());
                        }
                    }
                };
                if !self.handle(cmd) {
                    self.abandon();
                    self.release();
                    let _ = self.tx.send(Event::Stopped);
                    return Ok(self.finish());
                }
                if self.paused {
                    // Paused: what is held goes out now, nothing waits behind it.
                    self.abandon();
                    self.release();
                    self.tx.flush();
                    let _ = self.tx.send(Event::Status(self.status()));
                }
            }
            self.cycle()?;
            self.after()?;
            self.release();
            if self.done && self.check.is_none() {
                self.release();
                let _ = self.tx.send(Event::Done {
                    think_tokens: self.think_tokens,
                    capped: self.think_capped,
                });
                let _ = self.tx.send(Event::Stopped);
                return Ok(self.finish());
            }
        }
    }
}

/// The notes kept in `notes.md`: one `- ` line each.
fn read_notes(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|l| l.strip_prefix("- ").map(|n| n.trim().to_string()))
                .filter(|n| !n.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// For a path that does not exist: its nearest existing directory and up
/// to 40 of the names in it, sorted, directories with a trailing `/`.
fn nearest_listing(p: &Path) -> Option<(PathBuf, String)> {
    if p.exists() {
        return None;
    }
    let dir = p.ancestors().skip(1).find(|a| a.is_dir())?;
    let mut names: Vec<String> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() {
                format!("{n}/")
            } else {
                n
            }
        })
        .collect();
    names.sort();
    let more = names.len().saturating_sub(40);
    names.truncate(40);
    let mut s = names.join(", ");
    if more > 0 {
        s.push_str(&format!(" and {more} more"));
    }
    Some((dir.to_path_buf(), s))
}
/// `PATH:START-END` as the path and the lines (1-based, inclusive); a
/// path with no such suffix whole.
fn read_range(spec: &str) -> (&str, Option<(usize, usize)>) {
    if let Some((path, r)) = spec.rsplit_once(':') {
        if let Some((a, b)) = r.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse(), b.trim().parse()) {
                return (path.trim(), Some((a, b)));
            }
        }
    }
    (spec, None)
}

/// A path the mind wrote: `~` expanded, relative to the workspace.
fn resolve(path: &str, workspace: &Path) -> PathBuf {
    let p = match path.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(path),
    };
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wrong_path_names_what_is_there() {
        let dir = std::env::temp_dir().join(format!("phi-stream-listing-{}", std::process::id()));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/reflect.rs"), "").unwrap();
        fs::write(dir.join("src/engine.rs"), "").unwrap();
        let (at, names) = nearest_listing(&dir.join("src/reflect/mod.rs")).unwrap();
        assert_eq!(at, dir.join("src"));
        assert_eq!(names, "engine.rs, reflect.rs");
        assert!(nearest_listing(&dir.join("src/engine.rs")).is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_take_a_line_range() {
        assert_eq!(read_range("src/engine.rs"), ("src/engine.rs", None));
        assert_eq!(
            read_range("src/engine.rs:120-200"),
            ("src/engine.rs", Some((120, 200)))
        );
        assert_eq!(read_range("a b/c.rs: 3 - 4"), ("a b/c.rs", Some((3, 4))));
        // A colon that is no range stays part of the path.
        assert_eq!(read_range("notes:draft.md"), ("notes:draft.md", None));
    }

    #[test]
    fn development_follows_the_person_s_instructions() {
        let base = "Be concise.\nNo em dash.";
        let plain = compose(base, Frame::Journal, None);
        assert!(!plain.contains("[prefer:"));
        let dev = compose(base, Frame::Journal, Some(Path::new("/r/Intel Phi Stream")));
        assert!(
            dev.contains("> Be concise.\n> No em dash.\n"),
            "the base quoted"
        );
        assert!(dev.contains("/r/Intel Phi Stream") && dev.contains("[prefer: ...]"));
        // The preferences are bounded by the instructions, and come after them.
        assert!(dev.contains("where the two conflict, those instructions win"));
        assert!(dev.find("> Be concise.").unwrap() < dev.find("[prefer:").unwrap());
        let chat = compose(base, Frame::Chat, Some(Path::new("/r")));
        assert!(chat.contains("You also develop software") && chat.contains("/r"));
    }

    #[test]
    fn notes_are_dash_lines() {
        let dir = std::env::temp_dir().join(format!("phi-stream-notes-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("notes.md");
        fs::write(&p, "- first\nnot a note\n-  second \n- \n").unwrap();
        assert_eq!(
            read_notes(&p),
            vec!["first".to_string(), "second".to_string()]
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn paths_resolve_against_the_workspace() {
        let ws = Path::new("/ws");
        assert_eq!(resolve("a/b.md", ws), PathBuf::from("/ws/a/b.md"));
        assert_eq!(resolve("/etc/hosts", ws), PathBuf::from("/etc/hosts"));
    }
}
