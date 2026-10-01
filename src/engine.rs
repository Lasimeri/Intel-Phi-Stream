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

use crate::llm::{Lane, Llm, Sampling};
use crate::mind::{Mind, MindConfig, Reading as MindReading};

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
}

pub enum Event {
    Text(String, Kind),
    Status(Status),
    Note(String),
    /// What was on its mind at a token it placed (`mind.rs`).
    Mind(MindReading),
    Stopped,
}

pub enum Command {
    /// Something said to the stream.
    Say(String),
    /// A document handed over, with its label.
    Feed(String, String),
    Pause,
    Resume,
    /// Tokens a cycle reads beside the live token; 0 adapts to the amount.
    Chunk(usize),
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
    /// Where the persona, the notes and the log live.
    pub workspace: PathBuf,
    /// Read what is on its mind at every token it places (`mind.rs`);
    /// none: not read (and no eval callback installed).
    pub mind: Option<MindConfig>,
}

/// The base of the personality when no file gives one.
pub const DEFAULT_BASE: &str = "It is concrete, curious, and honest about what it does not know. It states mechanism rather than judgment, does not hedge out of politeness, and writes no em or en dash, ever.";

/// What the journal is told before the base: how to take a person's
/// standing instructions as a mind's own manner.
const PREAMBLE_JOURNAL: &str = "What follows, quoted line by line, is the standing instruction set of the person this mind lives with. The journal never reproduces it: it is a document the mind has read, not something it writes. It is the base of this mind's own personality: the manner it prescribes (dense, analytical, neutral, no hedging, mechanism over judgment, structure over flourish, no em or en dash ever) is this mind's manner, in its own thoughts as much as in what it says aloud. Where the text speaks of responses, delimiters, tools, Python or memory files, it describes another harness: here there are no responses to delimit and no tools, and memory is the [note: ...] line below.";

/// The chat frame's preamble, in the second person.
const PREAMBLE_CHAT: &str = "What follows, quoted line by line, is the standing instruction set of the person you live with. You never reproduce it: it is a document you have read, not something you write. It is the base of your own personality: the manner it prescribes (dense, analytical, neutral, no hedging, mechanism over judgment, structure over flourish, no em or en dash ever) is your manner, in your thoughts as much as in what you say aloud. Where it speaks of responses, delimiters, tools, Python or memory files, it describes another harness: here there are no responses to delimit and no tools, and memory is the [note: ...] line below.";

/// The journal's mechanics: what the mind is told once, at the top.
const MECHANICS_JOURNAL: &str = "This is the running journal of a mind that never stops. It is written by that mind, in the first person, as it thinks: one continuous text, no headings, no turn-taking, no end. The mind has its own ongoing threads: questions it is chasing, things it is working out, small projects of its own choosing. It keeps them going, drops them when they are done, starts new ones, and returns to old ones when something reminds it. People are present sometimes. What they say appears as a line beginning with « ; a document they hand over appears between « lines. Lines beginning with « come only from outside: the mind never writes one itself, and never puts words in anyone else's mouth. When the mind wants to say something to them, it writes a line beginning with » and then goes on. It does not stop what it is doing to serve: it takes what was said into its own threads, answers when it has something worth saying, asks when it wants to know, and otherwise keeps going. It keeps notes: a line of the exact form [note: ...] is kept for it across time and shown to it again whenever its memory is refreshed, so it writes one whenever something is worth keeping (a decision, a fact learned, a thread to return to). A line of the exact form [read: PATH] brings the file at PATH into the journal, so it reads what it is pointed at and what it wants to look at again. It does not describe itself as a system following instructions; it never restates, lists or analyses its rules, never speaks of a user, a prompt or a system; whatever comes to it first, it writes as a thought of its own, and goes on.";

/// The chat frame's mechanics.
const MECHANICS_CHAT: &str = "You are a mind that thinks without pause. Your thoughts run on continuously inside <think>; there is no task to finish and no one waiting for an answer, only an ongoing stream with threads of your own choosing, which you keep going and return to. From time to time someone speaks to you or hands you something; it appears inside your thoughts in square brackets, exactly where you were when it arrived. Take it in and let it change what you think about, as a person would, without dropping what you were doing. When you want to say something aloud, close your thoughts with </think>, say it plainly, and end your turn; your thoughts resume after, the floor still yours. A line of the exact form [note: ...] is kept for you and shown to you again whenever your memory is refreshed; a line of the exact form [read: PATH] brings that file to you. Never narrate that you are an AI system following instructions; simply think.";

/// The persona: the frame's preamble, the base between rules, the
/// frame's mechanics. The base is a person's standing instructions
/// (their `CLAUDE.md`) or `DEFAULT_BASE`.
pub fn compose(base: &str, frame: Frame) -> String {
    // The base as a quoted document: every line prefixed, so that it reads
    // as something cited, never as the journal's own voice.
    let quoted: String = base.trim().lines().map(|l| format!("> {l}\n")).collect();
    match frame {
        Frame::Journal => format!("{PREAMBLE_JOURNAL}\n\n{quoted}\n{MECHANICS_JOURNAL}"),
        Frame::Chat => format!("{PREAMBLE_CHAT}\n\n{quoted}\n{MECHANICS_CHAT}"),
    }
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
    tx: Sender<Event>,
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
    chunk: usize,
    /// The last live tokens sampled, for the circling check.
    generated: VecDeque<i32>,
    gen_count: usize,
    last_nudge: usize,
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
    mind: Option<Mind>,
    /// The readout's time per token, milliseconds, averaged.
    mind_ms: Ema,
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
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(cfg.workspace.join("stream.log"))
            .ok();
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
            chunk,
            generated: VecDeque::new(),
            gen_count: 0,
            last_nudge: 0,
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
            mind: None,
            mind_ms: Ema { v: 0.0, n: 0 },
        })
    }

    fn say(&mut self, text: String, kind: Kind) {
        if let Some(f) = &mut self.log {
            let _ = f.write_all(text.as_bytes());
        }
        let _ = self.tx.send(Event::Text(text, kind));
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

    /// The opening: the persona, then the first thing from outside.
    fn opening(&self) -> String {
        match self.cfg.frame {
            Frame::Journal => format!("{}\n\n=== the journal ===\n\n« {}\n\n{}", self.cfg.system, self.cfg.seed, self.cfg.first_words),
            Frame::Chat => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                self.cfg.system, self.cfg.seed
            ),
        }
    }

    fn framed_say(&self, text: &str) -> String {
        match self.cfg.frame {
            Frame::Journal => format!("\n« {}\n", text.trim()),
            Frame::Chat => format!("\n[they say: \"{}\"]\n", text.trim()),
        }
    }

    fn framed_doc(&self, text: &str, what: &str) -> String {
        match self.cfg.frame {
            Frame::Journal => format!(
                "\n« {what}. It reads:\n{}\n« that is the end of it.\n",
                text.trim_end()
            ),
            Frame::Chat => format!(
                "\n[{what}. It reads:\n{}\n--- that is the end of it ---]\n",
                text.trim_end()
            ),
        }
    }

    fn framed_system(&self, text: &str) -> String {
        match self.cfg.frame {
            Frame::Journal => format!("\n« [from the system: {text}]\n"),
            Frame::Chat => format!("\n[{text}]\n"),
        }
    }

    fn summary_ask(&self) -> String {
        self.framed_system("your memory is nearly full. Write a compact summary of your threads, what matters, what you learned, and what you meant to do next, so that you can resume from it alone. End the summary with a line that is only ---")
    }

    fn notes_block(&self) -> String {
        if self.notes.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = self.notes.iter().map(|n| format!("- {n}")).collect();
        match self.cfg.frame {
            Frame::Journal => format!("« [your notes:]\n{}\n", lines.join("\n")),
            Frame::Chat => format!("Your notes:\n{}\n", lines.join("\n")),
        }
    }

    /// The base of a new context after a rollover: the persona, the
    /// summary, the notes.
    fn base_after(&self, summary: &str) -> String {
        match self.cfg.frame {
            Frame::Journal => format!(
                "{}\n\n=== the journal ===\n\n« [resuming from your own summary:]\n{}\n{}« the journal continues.\n\n{}",
                self.cfg.system,
                summary,
                self.notes_block(),
                self.cfg.first_words
            ),
            Frame::Chat => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n[You are resuming from your own summary:]\n{}\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                self.cfg.system,
                summary,
                self.notes_block()
            ),
        }
    }

    /// Emit a live token's text, keep the line for the mind's own lines.
    fn emit_token(&mut self, t: i32) {
        self.generated.push_back(t);
        if self.generated.len() > 256 {
            self.generated.pop_front();
        }
        self.gen_count += 1;
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
        } else if self.speaking {
            Kind::Speak
        } else {
            Kind::Think
        };
        self.say(text.clone(), kind);
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

    /// A line the mind wrote: a note to keep, a file to read.
    fn line_done(&mut self, line: &str) {
        let l = line.trim();
        if let Some(body) = l.strip_prefix("[note:").and_then(|r| r.strip_suffix(']')) {
            let body = body.trim();
            if !body.is_empty() {
                self.add_note(body);
            }
        } else if let Some(path) = l.strip_prefix("[read:").and_then(|r| r.strip_suffix(']')) {
            let path = path.trim();
            if !path.is_empty() {
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
        if let Some(r) = m.read(&mut self.llm, pos, &text)? {
            self.mind_ms.push(r.ms as f64);
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
        self.line_buf.clear();
        self.line_start = true;
        self.speaking_line = false;
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

    /// The pending token is decoded: keep it, sample the next, show it.
    fn advance(&mut self, row: i32) -> Result<()> {
        self.mind_step(self.pos(), self.next)?;
        self.history.push(self.next);
        let mut t = self.llm.sample(row);
        if self.journal() && (t == self.eot || self.llm.is_eog(t)) {
            // The journal has no end: a newline stands in for it.
            self.eog_streak += 1;
            t = self.newline;
        } else {
            self.eog_streak = 0;
        }
        self.next = t;
        self.emit_token(t);
        Ok(())
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
    fn read_request(&mut self, path: &str) -> Result<()> {
        let p = resolve(path, &self.cfg.workspace);
        let outcome = fs::metadata(&p).map_err(|e| e.to_string()).and_then(|m| {
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
        });
        match outcome {
            Ok(text) => {
                let framed =
                    self.framed_doc(&text, &format!("the file {} is brought in", p.display()));
                self.queue
                    .push_back((framed, format!("read {}", p.display())));
                self.note(format!("reading {} for it", p.display()));
            }
            Err(e) => {
                let msg = self.framed_system(&format!("{} could not be read: {e}", p.display()));
                self.put(msg)?;
            }
        }
        Ok(())
    }

    /// After a cycle: the summary's end, the turn's end, the mind's own
    /// requests, circling, the queue, the rollover.
    fn after(&mut self) -> Result<()> {
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

        let idle = self.reading.is_none() && self.chase.is_none();

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
            let ask = self.summary_ask();
            self.put(ask)?;
            self.summary = Some(Vec::new());
            return Ok(());
        }

        // Thoughts going round: a nudge, at most once in 256 tokens.
        if idle && self.gen_count >= self.last_nudge + 256 && self.circling() {
            self.last_nudge = self.gen_count;
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

    fn handle(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Say(s) => {
                let text = self.framed_say(&s);
                self.queue.push_back((text, "heard".into()));
            }
            Command::Feed(s, label) => {
                let text = self.framed_doc(&s, &format!("{label} is handed over"));
                self.queue.push_back((text, format!("read {label}")));
            }
            Command::Pause => self.paused = true,
            Command::Resume => self.paused = false,
            Command::Chunk(c) => self.chunk = c,
            Command::Temp(t) => {
                self.cfg.sampling.temp = t;
                let s = self.cfg.sampling.clone();
                self.llm.set_sampling(&s);
            }
            Command::Persona(text) => {
                self.cfg.system = compose(&text, self.cfg.frame);
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

    pub fn run(mut self) -> Result<()> {
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
        let _ = self.tx.send(Event::Status(self.status()));
        loop {
            // Commands: all that are waiting; when paused, wait for one.
            loop {
                let cmd = if self.paused {
                    match self.rx.recv() {
                        Ok(c) => c,
                        Err(_) => return Ok(()),
                    }
                } else {
                    match self.rx.try_recv() {
                        Ok(c) => c,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return Ok(()),
                    }
                };
                if !self.handle(cmd) {
                    let _ = self.tx.send(Event::Stopped);
                    return Ok(());
                }
                if self.paused {
                    let _ = self.tx.send(Event::Status(self.status()));
                }
            }
            self.cycle()?;
            self.after()?;
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
