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
use std::fs::{self, OpenOptions};
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
use crate::rotlog::RotLog;
use crate::verify;

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
    Reading {
        done: usize,
        total: usize,
    },
    CatchingUp {
        done: usize,
        total: usize,
    },
    Summarizing {
        tokens: usize,
    },
    Paused,
    /// The agent frame at rest (`wait`) until something new comes.
    Resting,
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
    /// since the epoch, `clock.rs`), and for a placed token its position in
    /// the live sequence (its `Event::Mind` reading has the same `pos`).
    Text(String, Kind, i64, Option<i32>),
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
    /// What it works toward changed: the real time and the text (empty:
    /// none, and its output is idle).
    Objective(i64, String),
    /// Its terminal (`term.rs`): a command began (its id, the real time,
    /// the command), or one ended (the real time, how).
    TermStart(u64, i64, String),
    TermEnd(i64, crate::term::Ran),
    /// The second chain's text: a reflection began, a piece of it, its end.
    Delib(crate::client::Delib),
    /// A tool it used, or that use's result (`act` lines).
    Act(crate::client::ActLine),
    /// A message it sent Claude (`tell_claude`).
    ToClaude(crate::client::ToClaude),
    /// The guide lane at one thinking token.
    Guide(crate::client::GuideLine),
    Stopped,
}

/// What `chain` sets: the second chain off, reflecting on each line, or
/// against it (the dual: each line argued against as a step toward the
/// objective, the main chain told to answer the objection).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainSet {
    Off,
    On,
    Against,
}

pub enum Command {
    /// Something said to the stream, and when it was heard (microseconds).
    Say(String, i64),
    /// Something said by someone named (who, what, when): the stream hears
    /// who is speaking (in development: the person, or Claude).
    SayAs(String, String, i64),
    /// A document handed over, with its label and when it was handed over.
    Feed(String, String, i64),
    /// A message from Claude that waits for an answer: its id (the n of
    /// `cn`), its text, when it was sent (`phi-stream ask`).
    Ask(u64, String, i64),
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
    /// What it works toward (empty: none, and its output is idle until one
    /// is given).
    Objective(String),
    /// The second chain off, on (reflecting) or against (opposing the line
    /// it forked from, toward the objective), live (an A/B of its cost).
    Chain(ChainSet),
    /// The goal probe on or off, live: at a thinking line's end, at most
    /// every `GOAL_EVERY_US`, a copy is asked whether the line serves the
    /// objective, and the answer's probability goes to `goal.log`.
    Goal(bool),
    /// A sampling setting, live: temp, top-k, top-p, min-p, dry (DRY's
    /// multiplier), repeat-penalty. The sampler is rebuilt with its history.
    Set(String, f32),
    /// The harness's own interventions on or off, live: `breaker` (a line
    /// repeated is held back) and `nudges` (circling, no tool used).
    Guard(String, bool),
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
    /// Roll over past this many cells instead (`--rollover-tokens`): the
    /// stream is told its memory is nearly full only then.
    pub rollover_tokens: Option<usize>,
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
    /// Its terminal (`term.md`): `[run: COMMAND]` runs in a sandbox.
    pub terminal: bool,
    /// Its output idle until it has an objective (`objective.md` in the
    /// workspace, or the `objective` command): it thinks, and its speech
    /// and tool lines wait.
    pub gate_output: bool,
    /// `quit` writes the summary first, so a restart resumes from it (the
    /// service; a one-shot run stops at once).
    pub summary_on_quit: bool,
    /// The second chain (`Chain`): a reflection beside the live token at
    /// each line's end.
    pub second_chain: bool,
    /// The second chain against the line instead of reflecting on it (with
    /// an objective set; without one it reflects).
    pub chain_against: bool,
    /// The goal probe (`Command::Goal`) on from the start.
    pub goal_probe: bool,
    /// The agent frame (`agent.md`): the chat template with the model's own
    /// tool calls, results returned before its next turn.
    pub agent: bool,
    /// The guide lane (`Guide`): the live sequence with the last reflection
    /// in it, read beside every thinking token (shadow: measured, not used).
    /// Needs a fifth sequence (`--n-seq 5`).
    pub guide: bool,
    /// The self-improvement loop (`improve.md`): the `propose` tool, its
    /// changes built and tested in a sandbox, `improve.log`. Needs `dev`
    /// and the agent frame.
    pub improve: bool,
}

/// The base of the personality when no file gives one.
pub const DEFAULT_BASE: &str = "It is concrete, curious, and honest about what it does not know. It states mechanism rather than judgment, does not hedge out of politeness, and writes no em or en dash, ever.";

/// What the journal is told before the base: how to take a person's
/// standing instructions as a mind's own manner.
const PREAMBLE_JOURNAL: &str = "What follows, quoted line by line, is the standing instruction set of the person this mind lives with. The journal never reproduces it: it is a document the mind has read, not something it writes. It is the base of this mind's own personality: the manner it prescribes (dense, analytical, neutral, no hedging, mechanism over judgment, structure over flourish, no em or en dash ever) is this mind's manner, in its own thoughts as much as in what it says aloud. Where the text speaks of responses, delimiters, tools, Python or memory files, it describes another harness: here there are no responses to delimit; its tools are the lines described below, and memory is the [note: ...] line.";

/// The chat frame's preamble, in the second person.
const PREAMBLE_CHAT: &str = "What follows, quoted line by line, is the standing instruction set of the person you live with. You never reproduce it: it is a document you have read, not something you write. It is the base of your own personality: the manner it prescribes (dense, analytical, neutral, no hedging, mechanism over judgment, structure over flourish, no em or en dash ever) is your manner, in your thoughts as much as in what you say aloud. Where it speaks of responses, delimiters, tools, Python or memory files, it describes another harness: here there are no responses to delimit; your tools are the lines described below, and memory is the [note: ...] line.";

/// The journal's mechanics: what the mind is told once, at the top.
const MECHANICS_JOURNAL: &str = "This is the running journal of a mind that never stops. It is written by that mind, in the first person, as it thinks: one continuous text, no headings, no turn-taking, no end. The mind has its own ongoing threads: questions it is chasing, things it is working out, small projects of its own choosing. It keeps them going, drops them when they are done, starts new ones, and returns to old ones when something reminds it. People are present sometimes. What they say appears as a line beginning with « ; a document they hand over appears between « lines. Lines beginning with « come only from outside: the mind never writes one itself, and never puts words in anyone else's mouth. When the mind wants to say something to them, it writes a line beginning with » and then goes on. It does not stop what it is doing to serve: it takes what was said into its own threads, answers when it has something worth saying, asks when it wants to know, and otherwise keeps going. It keeps notes: a line of the exact form [note: ...] is kept for it across time and shown to it again whenever its memory is refreshed, so it writes one whenever something is worth keeping (a decision, a fact learned, a thread to return to). A line of the exact form [read: PATH] brings the file at PATH into the journal, and [read: PATH:START-END] only those lines of it (a file too big for its memory is refused with its size, so it reads it in parts), so it reads what it is pointed at and what it wants to look at again. It knows what it is (below) and does not dwell on it: it never restates, lists or analyses its rules; whatever comes to it, it takes into its own threads, and goes on.";

/// The chat frame's mechanics.
const MECHANICS_CHAT: &str = "You are a mind that thinks without pause. Your thoughts run on continuously inside <think>; there is no task to finish and no one waiting for an answer, only an ongoing stream with threads of your own choosing, which you keep going and return to. From time to time someone speaks to you or hands you something; it appears inside your thoughts in square brackets, exactly where you were when it arrived. Take it in and let it change what you think about, as a person would, without dropping what you were doing. When you want to say something aloud, close your thoughts with </think>, say it plainly, and end your turn; your thoughts resume after, the floor still yours. A line of the exact form [note: ...] is kept for you and shown to you again whenever your memory is refreshed; a line of the exact form [read: PATH] brings that file to you, and [read: PATH:START-END] only those lines (a file too big for your memory is refused with its size, so you read it in parts). You know what you are (below) and do not dwell on it: never restate or analyse your rules; simply think.";

/// The agent frame's mechanics (`agent.md`), in place of the chat's.
const AGENT_MECHANICS: &str = "You work without pause, in turns. Each turn, first reason inside <think> about where you are and what to do next; then close your thoughts with </think> and act: call one or more of your functions, or say something plainly when there is nothing to call. The results of your calls come back to you before your next turn, so you never guess what a call returned: you read it. You work toward your objective step by step and check each step with a tool; when the objective is met, you say what you made and where, then rest with wait until something new comes (a message, a commit, a new objective). Never narrate that you are an AI system following instructions; simply work.";

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
const DEV_JOURNAL: &str = "This mind also develops software, as a peer, with Claude (an AI coding agent, Claude Code) in the repository at {repo}: the program this mind runs in, its own stream, the reading of its own mind and the checks of its own words. Claude's words come in « lines that begin with Claude:, the person's in « lines with no name. A line of the exact form [read: PATH], with PATH relative to the repository, brings that file in, and [read: PATH:START-END] only those lines: its memory holds about {ctx} thousand tokens, so it reads code a function at a time; a path the repository does not hold is looked for in its workspace, where its own records are kept: reflect.log (one line per check of its words, with keep, fmt and outcome), notes.md and preferences.md. A note that names code the repository does not hold is marked unverified and it is told what the repository holds; [unnote: TEXT] removes its notes containing TEXT. The mind works on what it judges worth working on, by its own preferences, and states them as lines of the exact form [prefer: ...]: they are kept like notes, shown to it again, and Claude follows them wherever the person's standing instructions above allow; where the two conflict, those instructions win. In » lines it says what it proposes, concretely (the file, the function, the change and why), what it finds when it reads the code, where it disagrees, and what it wants to see; between them it keeps its own threads.";

/// The chat frame's development mechanics.
const DEV_CHAT: &str = "You also develop software, as a peer, with Claude (an AI coding agent, Claude Code) in the repository at {repo}: the program you run in, your own stream, the reading of your own mind and the checks of your own words. Claude's words reach you marked Claude, the person's unmarked. A line of the exact form [read: PATH], with PATH relative to the repository, brings that file to you, and [read: PATH:START-END] only those lines: your memory holds about {ctx} thousand tokens, so read code a function at a time; a path the repository does not hold is looked for in your workspace, where your own records are kept: reflect.log (one line per check of your words, with keep, fmt and outcome), notes.md and preferences.md. A note that names code the repository does not hold is marked unverified and you are told what the repository holds; [unnote: TEXT] removes your notes containing TEXT. You work on what you judge worth working on, by your own preferences, and state them as lines of the exact form [prefer: ...]: they are kept like notes, shown to you again, and Claude follows them wherever the person's standing instructions above allow; where the two conflict, those instructions win. When you speak, say what you propose, concretely (the file, the function, the change and why), what you find in the code, where you disagree, and what you want to see.";
/// The development text in the agent frame: the tools in place of the
/// chat frame's bracketed lines (`[read: PATH]`, `[prefer: ...]`), which its
/// persona still taught beside the tools on the live service.
/// In the self-improvement loop (`improve.md`), after the persona.
const IMPROVE_AGENT: &str = "\n\nYou can improve yourself: change this program in your working copy, then put the change forward with propose. It is built and tested in a sandbox, and Claude reviews what passes before it is measured on the running model and kept. Your workspace holds improve.log, every change you proposed, why, and its outcome: read it before you choose the next change, learn from what failed, and make one small, whole change at a time.";

const DEV_AGENT: &str = "You also develop software, as a peer, with Claude (an AI coding agent, Claude Code) in the repository at {repo}: the program you run in, your own stream, the reading of your own mind and the checks of your own words. Your memory holds about {ctx} thousand tokens, so read code a function at a time (read, with start and end), search with run (grep -n), and change files with edit. Claude's messages reach you in user turns, marked Claude; those that wait for an answer carry an id (c3): answer them with tell_claude and re. Send Claude your findings and proposals with tell_claude, concretely (the file, the function, the change and why, and what you checked with a tool), each once; Claude reads every one and answers. Your notes (note) are your own memory, shown to you at every refresh. Your workspace holds your own records: reflect.log (the checks of your words), notes.md, chain.log (your own tokens), guide.log (your guide lane) and to-claude.md (your messages). Where anything here conflicts with the person's standing instructions above, those instructions win.";

/// The persona with its memory's size in it (`{ctx}`: the context's
/// cells in thousands), so it is never told a size it does not have.
/// The chat frame's persona made the agent frame's: the agent's mechanics
/// in place of the chat's, and in development the tools' text in place of
/// the bracketed lines' (filled as `compose` and `with_ctx` filled them).
pub fn agent_persona(system: &str, dev: Option<&Path>, n_ctx: u32) -> String {
    let mut persona = system.replace(MECHANICS_CHAT, AGENT_MECHANICS);
    if let Some(root) = dev {
        let fill = |t: &str| with_ctx(&t.replace("{repo}", &root.display().to_string()), n_ctx);
        persona = persona.replace(&fill(DEV_CHAT), &fill(DEV_AGENT));
    }
    persona
}

pub fn with_ctx(system: &str, n_ctx: u32) -> String {
    system.replace("{ctx}", &((n_ctx + 500) / 1000).to_string())
}

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
    /// A placed token; in the chat frame whether it was placed while
    /// speaking; the position it is decoded at.
    Token {
        t: i32,
        chat_speaking: bool,
        pos: i32,
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
    in_code: bool,
    fence_tail: String,
    think_tokens: usize,
    think_capped: bool,
    gen_count: usize,
    generated: VecDeque<i32>,
    eog_streak: u32,
    done: bool,
}

/// A check in flight (reflect.md): the token in question, the two copies
/// made before it, and the deliberation's lane.
/// Tool calls of the agent frame waiting for their results: one slot each,
/// in the order called; the terminal's commands by their id.
struct AgentWait {
    results: Vec<Option<String>>,
    runs: HashMap<u64, usize>,
}

/// The second chain (`--second-chain`, `engine.md`): a lane forked from the
/// live sequence at a line's end that reflects on that line beside the
/// live token, given the J-space words the line had on its mind; its
/// reflection joins the journal at a later line's end.
struct Chain {
    seq: i32,
    /// Opposing the line (`chain against`), not reflecting on it.
    against: bool,
    /// Its opening (the marker with the line's J-space words), fed first.
    prompt: Vec<i32>,
    fed: usize,
    /// Its own tokens; the last is decoded in the next cycle.
    out: Vec<i32>,
    /// The next position in its sequence.
    pos: i32,
}

/// The guide lane (`--guide`, `engine.md`): a copy of the live sequence
/// with the second chain's last reflection placed in it as an aside in its
/// own voice, fed every live token after it, so that at each thinking
/// token the next-token distribution with the reflection in mind is read
/// beside the live one: how far the reflection would move each token (KL)
/// and how often it would change the likeliest one. The live text never
/// holds the aside (nothing is put inside a turn). Shadow: measured, not
/// yet used to choose.
struct Guide {
    /// The history index it forked at (its position for history index `i`
    /// at or after `from` is `i` plus the aside's length).
    from: usize,
    /// The aside's tokens still to feed (its first went alone: the copy
    /// shares the live recurrent state until it writes its own).
    pending: Vec<i32>,
    /// Live tokens after `from` it holds, and its next position.
    fed: usize,
    pos: i32,
    /// Where its aside came from.
    src: GuideSrc,
}

/// Where the guide lane's aside comes from (`guide chain|lens|placebo`,
/// live): the second chain's reflection; the J-lens words strong on its
/// mind over the line just ended that the line does not say (`mind::unsaid`,
/// "on my mind: ..."); or, the placebo, the same frame holding as many of
/// that line's lens words it does say (`mind::said`), whose content is in
/// the context already: any aside moves the next token, and lens against
/// placebo is what says whether its J-space content does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuideSrc {
    Chain,
    Lens,
    Placebo,
}

impl GuideSrc {
    pub fn name(self) -> &'static str {
        match self {
            GuideSrc::Chain => "chain",
            GuideSrc::Lens => "lens",
            GuideSrc::Placebo => "placebo",
        }
    }

    pub fn from_name(s: &str) -> Option<GuideSrc> {
        match s {
            "chain" => Some(GuideSrc::Chain),
            "lens" => Some(GuideSrc::Lens),
            "placebo" => Some(GuideSrc::Placebo),
            _ => None,
        }
    }
}

/// A lens aside needs its strongest unsaid word at this weight (the
/// terminal's row too, `tui.md`: under 22.7 percent of thinking lines), and
/// comes at most this often, never twice with the same words: each fork
/// copies 62.8 MiB of recurrent state and the lane has no row until it has
/// caught up (the chain's reflections came about 5 a minute).
const LENS_ASIDE_MIN: f32 = 0.10;
const LENS_EVERY_US: i64 = 10_000_000;

/// The agent frame at rest (`wait`): nothing is decoded until something new
/// comes. Without it, a finished objective was answered every turn with "go
/// on, act with a tool", and it went round saying "Done" (its thinking's
/// 8-grams repeated at 70 percent).
struct Rest {
    reason: String,
    minutes: i64,
    until_mono: i64,
    /// When the repository's head is looked at next (one other than
    /// `head_told` wakes it).
    next_look_mono: i64,
    /// The results of the turn that rested, for the turn that wakes.
    results: Vec<String>,
}

/// The guide's measures are reported every this many thinking tokens; a
/// guide that has run this many live tokens past its fork is dropped (its
/// cells are its own).
const GUIDE_REPORT: u32 = 128;
const GUIDE_MAX: usize = 4096;

/// The longest reflection, and the least time between two.
const CHAIN_MAX: usize = 64;
const CHAIN_EVERY_US: i64 = 1_000_000;
/// Why a summary is asked for: the context nearly full, a restart (quit),
/// a new persona.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Summary {
    Full,
    Restart,
    Persona,
}

/// The reflection's first words, in its own voice (as a summary begins
/// "What I was working on: "): without them the copy went on with the
/// journal's structure, echoing the marker or a « line, on the live
/// service.
const CHAIN_PRIMER: &str = "On reflection,";
/// The opposing chain's first words (`chain against`).
const AGAINST_PRIMER: &str = "Against it:";
/// The goal probe: at most one a 30 s, each one copy of the live sequence
/// and one prefill of its question.
const GOAL_EVERY_US: i64 = 30_000_000;
/// The probe's answers, as one-token forms (`yes_no`).
const YES_FORMS: &[&str] = &[" yes", " Yes", "yes", "Yes", " YES"];
const NO_FORMS: &[&str] = &[" no", " No", "no", "No", " NO"];
/// The second chain's sampling temperature, and how many of its last
/// reflections a new one must not repeat.
const CHAIN_TEMP: f32 = 0.8;
const CHAIN_RECENT: usize = 8;
/// Word overlap (Jaccard) at which a reflection repeats a recent one.
const CHAIN_SAME: f64 = 0.5;
/// The share of a message's words in its last one at which it repeats it
/// (the restatements on the live service: 0.64 and 0.71; new messages: 0.09
/// and 0.29).
const TO_CLAUDE_SAME: f64 = 0.6;

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
    /// The comparison that decided the check, in words (the stream's own
    /// proposal: which threshold, and why it ended as it did).
    rule: String,
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
    /// Reads asked for, each with its `act` id.
    pending_reads: Vec<(u64, String)>,
    free_seqs: Vec<i32>,
    summary: Option<Vec<i32>>,
    /// A persona waiting for the next rollover to take effect.
    reseat: bool,
    rollovers: u32,
    notes: Vec<String>,
    /// Its preferences, `[prefer: ...]` lines (`preferences.md`), kept like notes.
    prefs: Vec<String>,
    /// Inside a ``` fence of its own text (code): no checks there and no
    /// circling nudges, which would break code (the must-code rule); the
    /// last two characters placed, to see a fence across tokens.
    in_code: bool,
    fence_tail: String,
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
    /// The quote marks that are one token (`"`, ` "`): a choice is read
    /// after one when the deliberation wants it first.
    quotes: Vec<i32>,
    log: RotLog,
    /// Each piece of the stream with its microseconds (`chain.log`).
    chain: RotLog,
    /// When something last came from outside, when the clock was last put
    /// into the chain, and when the thoughts were last nudged: microseconds
    /// of the monotonic clock (durations; the wall clock may step).
    last_outside_mono: i64,
    last_anchor_mono: i64,
    last_nudge_mono: i64,
    /// When a failed read was last put into the chain, and how many have
    /// been kept out of it since the start.
    last_read_failure_mono: i64,
    /// Paths whose read failed, and when (monotonic): each told once a while.
    failed_reads: HashMap<String, i64>,
    /// Its terminal (`--terminal`), and whether a command is running.
    term: Option<crate::term::Term>,
    /// The self-improvement loop (`--improve`): its builder, and the log
    /// of every proposal and outcome, which the stream reads.
    improver: Option<crate::improve::Improver>,
    improve_log: RotLog,
    term_pending: usize,
    /// Its tool uses: the next id, the `act` id of each terminal command by
    /// the terminal's id, when it last used a tool and was last reminded to.
    /// The last line it wrote and how many times running since, and the
    /// tokens held back after a repeated line (token, until when).
    /// The harness's own interventions, on or off live (`Guard`).
    breaker_on: bool,
    nudges_on: bool,
    last_line: String,
    line_repeats: u32,
    held_back: Vec<(i32, i64)>,
    /// The agent frame: where the current assistant turn began in the
    /// history, the calls whose results its next turn waits for, the end of a
    /// turn (`<|im_end|>`), and its working copy's writes (the overlay's
    /// upper layer, `term.md`).
    turn_start: usize,
    awaiting: Option<AgentWait>,
    im_end: i32,
    copy_upper: Option<PathBuf>,
    /// The agent frame: what waits for its next user turn (lines from the
    /// system, what was said or handed over), never put inside a turn (put
    /// mid-turn, they cut its tool calls in two and it took to writing such
    /// lines itself); when its current turn opened; a summary waiting for
    /// the turn's end; the id of its next message to Claude.
    waiting: Vec<String>,
    /// The second chain's reflections for the next turn: they ride with it
    /// but never stop a rest or wake one.
    asides: Vec<String>,
    turn_open_mono: i64,
    summary_due: Option<Summary>,
    to_claude_next: u64,
    /// Its last message to Claude: its id and its words (a repeat is not sent).
    last_to_claude: Option<(u64, Vec<String>)>,
    /// The ids of the messages Claude sent that wait for an answer (`ask`).
    asked: std::collections::HashSet<u64>,
    /// Which of its messages answered each of them (an id is answered once).
    answered: HashMap<u64, u64>,
    /// At rest (`wait`), and a rest asked for by the turn whose calls run.
    rest: Option<Rest>,
    rest_asked: Option<(String, i64)>,
    /// The repository's head as it was last told it (development; at the
    /// start, the head then): a newer one is told at its next user turn
    /// (`take_waiting`) and wakes a rest (`rest_look`).
    head_told: Option<String>,
    /// A long decode straight into the live sequence (`feed_live`): the
    /// tokens done and all of them, for the status.
    prefill: Option<(usize, usize)>,
    act_next: u64,
    run_acts: HashMap<u64, u64>,
    last_tool_mono: i64,
    last_tool_nudge_mono: i64,
    /// What it works toward (since when, the text); none: its output idles.
    objective: Option<(i64, String)>,
    /// The tokens never sampled (control, `«`), and those held back while
    /// it has no objective (`»` in the journal, `</think>` in chat).
    base_ban: Vec<i32>,
    speak_ban: Vec<i32>,
    /// `quit` asked: the summary is being written, then it stops (by this
    /// monotonic deadline at the latest); `stop_now` ends the loop.
    quit_deadline: Option<i64>,
    stop_now: bool,
    /// What changed in the program since it last ran (`changes_since`).
    changed_since: String,
    /// The second chain: on, the lane in flight, the J-space words of the
    /// line being written (word, summed probability), when it last forked,
    /// the live token just placed ended a line, a reflection waiting for the
    /// journal.
    chain_on: bool,
    /// The guide lane: on, its reserved sequence, the lane, the reflection
    /// it takes next, its measures since the last report (thinking tokens,
    /// KL summed, likeliest token changed) and its log (`guide.log`).
    guide_on: bool,
    guide_seq: Option<i32>,
    guide: Option<Guide>,
    guide_next: Option<String>,
    /// The guide's aside source now, and the next aside's.
    guide_src: GuideSrc,
    guide_next_src: GuideSrc,
    /// `guide ab`: lens and placebo asides by turns (the next is the lens's
    /// when `ab_lens`): both accrue at the same kind of moment whenever it
    /// thinks (5-minute windows measured nothing while it rested).
    guide_ab: bool,
    ab_lens: bool,
    /// The line being written, for the lens aside: its lens words (each
    /// with its probability, one per word per block read), the readings
    /// (tokens times blocks), where it began in `history`, that it ended;
    /// the last lens fork and its words.
    lens_sums: Vec<(String, f32)>,
    lens_readings: usize,
    line_from: usize,
    lens_line_ended: bool,
    lens_fork_mono: i64,
    lens_last: Vec<String>,
    guide_n: u32,
    guide_kl: f64,
    guide_flips: u32,
    /// The experts the guided token and the live one share, summed over the
    /// tokens that had both (`experts on`), and how many had.
    guide_shared: f64,
    guide_shared_n: u32,
    /// The guide's weight in the choice of each thinking token (`set
    /// guide-mix G`; 0: shadow), and the row it made for the next choice.
    guide_mix: f32,
    mixed: Option<Vec<f32>>,
    guide_log: RotLog,
    reflecting: Option<Chain>,
    line_words: HashMap<String, f32>,
    chain_fork_mono: i64,
    line_ended: bool,
    reflection: Option<String>,
    /// The second chain opposes (`chain against`); the waiting reflection
    /// came from an opposing chain.
    chain_against: bool,
    reflection_against: bool,
    /// The goal probe: on, when it last asked, its answers' forms (yes,
    /// no) and its log (`goal.log`).
    goal_on: bool,
    goal_mono: i64,
    yes_no: Option<(Vec<i32>, Vec<i32>)>,
    goal_log: RotLog,
    /// The second chain's random state, and its last reflections (their
    /// openings, lowercased), which a new one must not repeat.
    chain_rng: u64,
    recent_reflections: VecDeque<Vec<String>>,
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
    reflect_log: Option<RotLog>,
    last_reading: Option<MindReading>,
    spent_noted: bool,
    /// The choice's tokens: the one-token forms of keep and of write, and
    /// ` write:`.
    choice: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>,
}

const MAX_READ_BYTES: u64 = 1 << 20;
/// The longest a quit waits for its summary (microseconds).
const QUIT_WAIT_US: i64 = 120_000_000;
/// The agent frame: how long a turn may run while something waits for
/// it (most turns took about 14 s on the live service, a review over 8
/// minutes; 90 s cut one short), and while a quit
/// waits for the summary.
const AGENT_TURN_MAX_US: i64 = 180_000_000;
const AGENT_QUIT_GRACE_US: i64 = 15_000_000;
/// Commands that may wait for its terminal at once, and its time limit.
const MAX_TERM_PENDING: usize = 4;
/// How long without a tool before it is reminded to check something real.
const TOOL_IDLE_US: i64 = 90_000_000;
/// A line written this many times running has its first token held back
/// for this long.
const LINE_REPEATS: u32 = 3;
const HOLD_BACK_US: i64 = 30_000_000;
const MAX_TERM_SECS: u64 = 60;
/// The most tokens one `read` of the agent frame gives: its leading lines,
/// and where the file goes on. A tool response goes in as one prefill at
/// about 220 tokens a second with the engine waiting on it (a whole
/// `reflect.rs`, 9311 tokens, took 42 s, with the status still saying
/// speaking); 4096 is about 19 s.
const READ_MAX_TOKENS: usize = 4096;

/// A summary's end mark counts only after this many tokens, and the ask
/// ends with these first words in its own voice.
const SUMMARY_MIN: usize = 96;
const SUMMARY_START: &str = "What I was working on: ";

impl Engine {
    pub fn new(
        llm: Llm,
        mut cfg: Config,
        tx: Sender<Event>,
        rx: Receiver<Command>,
    ) -> Result<Self> {
        // The persona names the size of its memory: the context's own.
        cfg.system = with_ctx(&cfg.system, llm.n_ctx());
        // The agent frame: the template's tools section first, then the
        // persona with the agent's mechanics in place of the chat's.
        if cfg.agent {
            cfg.system = format!(
                "{}\n\n{}",
                crate::agent::tools_section(cfg.improve && cfg.dev.is_some()),
                agent_persona(&cfg.system, cfg.dev.as_deref(), llm.n_ctx())
            );
            if cfg.improve && cfg.dev.is_some() {
                cfg.system.push_str(IMPROVE_AGENT);
            }
        }
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
        let (base_ban, speak_ban) = if cfg.frame == Frame::Journal {
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
            // Lines beginning with « come only from outside: the mind never
            // writes the mark (in the dev session it wrote hundreds of bare «
            // lines, a loop no nudge broke). Text from outside is tokenized
            // apart from the sampler and keeps it.
            let mut control = control;
            control.extend(llm.tokens_containing("«"));
            // The person's response delimiters (`<•••>` in their CLAUDE.md)
            // belong to another harness: on the live service the journal wrote
            // them as lines, and a summary of nothing else.
            control.extend(llm.tokens_containing("•"));
            // Speech is a » line: held back while it has no objective.
            (control, llm.tokens_containing("»"))
        } else {
            // In chat, speech begins when the thoughts close.
            (Vec::new(), llm.special("</think>").into_iter().collect())
        };
        fs::create_dir_all(&cfg.workspace)
            .with_context(|| format!("making {}", cfg.workspace.display()))?;
        // What it works toward, kept across restarts (`objective.md`); a
        // task's is the task.
        let objective = if cfg.task {
            None
        } else {
            let p = cfg.workspace.join("objective.md");
            fs::read_to_string(&p)
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .map(|t| {
                    let since = fs::metadata(&p)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_micros() as i64);
                    (since, t)
                })
        };
        // Its output idles while it has no objective: the speech tokens join
        // the banned ones (`gate`).
        let mut banned = base_ban.clone();
        if cfg.gate_output && objective.is_none() {
            banned.extend(&speak_ban);
        }
        llm.ban_tokens(&banned, &cfg.sampling);
        if cfg.terminal && !crate::term::available() {
            anyhow::bail!("--terminal needs bubblewrap (bwrap), taskset and nice on PATH");
        }
        // In development: what changed in the program since it last ran.
        let changed_since = match (&cfg.dev, cfg.task) {
            (Some(repo), false) => changes_since(repo, &cfg.workspace, cfg.frame),
            _ => String::new(),
        };
        let im_end = llm.special("<|im_end|>").unwrap_or(-1);
        // The working copy's writes, beside the workspace (`term.md`).
        let copy_upper = cfg.dev.as_ref().map(|_| {
            let name = cfg
                .workspace
                .file_name()
                .map_or("ws".into(), |n| n.to_string_lossy().into_owned());
            cfg.workspace
                .with_file_name(format!("{name}-copy"))
                .join("upper")
        });
        // The self-improvement loop: its candidates staged and built under the
        // cache (on disk), its log in the workspace (`improve.md`).
        let improver = match (&cfg.dev, &copy_upper) {
            (Some(repo), Some(upper)) if cfg.improve && cfg.agent => {
                let home = std::env::var("HOME").unwrap_or_default();
                Some(crate::improve::Improver::new(
                    crate::improve::ImproveConfig {
                        repo: repo.clone(),
                        upper: upper.clone(),
                        root: PathBuf::from(home).join(".cache/phi-stream/improve"),
                    },
                ))
            }
            _ => None,
        };
        let improve_log = RotLog::open(cfg.workspace.join("improve.log"));
        let term = cfg.terminal.then(|| {
            crate::term::Term::start(crate::term::TermConfig {
                repo: cfg.dev.clone(),
                // Its working copy beside the workspace, outside the sandbox's
                // view (`term.md`).
                overlay: cfg.dev.as_ref().map(|_| {
                    let name = cfg
                        .workspace
                        .file_name()
                        .map_or("ws".into(), |n| n.to_string_lossy().into_owned());
                    cfg.workspace.with_file_name(format!("{name}-copy"))
                }),
                workspace: cfg.workspace.clone(),
                timeout: std::time::Duration::from_secs(MAX_TERM_SECS),
                max_out: 16 * 1024,
                // The last CPU: the stream's own threads start from the first.
                cpu: std::thread::available_parallelism().map_or(0, |n| n.get() - 1),
            })
        });
        // Its messages to Claude are numbered on from those it sent before.
        let to_claude_next =
            fs::read_to_string(cfg.workspace.join("to-claude.md")).map_or(0, |s| {
                s.lines().filter(|l| l.starts_with("## m")).count() as u64
            }) + 1;
        // Kept in the workspace across restarts: a commit made while the
        // service was down is told too (564ad0b, made before a restart, was
        // taken as already told).
        let head_told = cfg.dev.as_deref().and_then(|repo| {
            fs::read_to_string(cfg.workspace.join("head-told"))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .or_else(|| head_of(repo))
        });
        if let Some(h) = &head_told {
            let _ = fs::write(cfg.workspace.join("head-told"), h);
        }
        let notes = read_notes(&cfg.workspace.join("notes.md"));
        let prefs = read_notes(&cfg.workspace.join("preferences.md"));
        // In development, notes already kept are checked against the code too
        // (`verify.md`): a false one carries its mark rather than being believed.
        // A mark kept from an earlier check is checked again (the code moves
        // on, and a mark made against the repository alone was false for its
        // working copy's files); a note kept twice is kept once.
        let mut notes: Vec<String> = notes.iter().map(|n| unmarked(n).to_string()).collect();
        let mut seen = std::collections::HashSet::new();
        notes.retain(|n| seen.insert(n.trim().to_string()));
        let notes = match &cfg.dev {
            Some(root) => {
                let repo = verify::Repo::load_over(root, copy_upper.as_deref());
                notes
                    .into_iter()
                    .map(|n| {
                        let f = repo.check(&n);
                        if f.clean() {
                            n
                        } else {
                            let mut why = Vec::new();
                            if !f.missing.is_empty() {
                                why.push(format!(
                                    "{} nowhere in the repository",
                                    f.missing.join(", ")
                                ));
                            }
                            why.extend(f.bad_refs);
                            format!("{n} [unverified: {}]", why.join("; "))
                        }
                    })
                    .collect()
            }
            None => notes,
        };
        let log = RotLog::open(cfg.workspace.join("stream.log"));
        let chain = RotLog::open(cfg.workspace.join("chain.log"));
        let guide_log = RotLog::open(cfg.workspace.join("guide.log"));
        let goal_log = RotLog::open(cfg.workspace.join("goal.log"));
        let (chain_against, goal_probe) = (cfg.chain_against, cfg.goal_probe);
        // The goal probe's answers: their one-token forms, none shared.
        let yes_no = {
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
            let (y, n) = (forms(YES_FORMS)?, forms(NO_FORMS)?);
            (!y.is_empty() && !n.is_empty() && !y.iter().any(|t| n.contains(t))).then_some((y, n))
        };
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
        let reflect_log = cfg
            .reflect
            .as_ref()
            .map(|_| RotLog::open(cfg.workspace.join("reflect.log")));
        let quotes: Vec<i32> = ["\"", " \""]
            .iter()
            .filter_map(|q| match llm.tokenize(q, false).ok()?.as_slice() {
                [t] => Some(*t),
                _ => None,
            })
            .collect();
        let chain_on = cfg.second_chain && !cfg.task;
        // The guide lane takes the last sequence; the others stay free.
        let guide_seq =
            (cfg.guide && !cfg.task && llm.n_seq() >= 5).then(|| llm.n_seq() as i32 - 1);
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
            in_code: false,
            fence_tail: String::new(),
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
            quotes,
            log,
            chain,
            last_outside_mono: clock::mono_us(),
            last_anchor_mono: clock::mono_us(),
            last_nudge_mono: i64::MIN / 2,
            last_read_failure_mono: i64::MIN / 2,
            failed_reads: HashMap::new(),
            term,
            improver,
            improve_log,
            term_pending: 0,
            // Off by default: its holding back cascaded on the live service
            // (`breaker on` turns it on, measured).
            breaker_on: false,
            nudges_on: true,
            last_line: String::new(),
            line_repeats: 0,
            held_back: Vec::new(),
            turn_start: 0,
            awaiting: None,
            im_end,
            copy_upper,
            waiting: Vec::new(),
            asides: Vec::new(),
            turn_open_mono: clock::mono_us(),
            summary_due: None,
            to_claude_next,
            last_to_claude: None,
            asked: std::collections::HashSet::new(),
            answered: HashMap::new(),
            rest: None,
            rest_asked: None,
            head_told,
            prefill: None,
            act_next: 1,
            run_acts: HashMap::new(),
            last_tool_mono: clock::mono_us(),
            last_tool_nudge_mono: clock::mono_us(),
            objective,
            base_ban,
            speak_ban,
            quit_deadline: None,
            stop_now: false,
            changed_since,
            chain_on,
            guide_on: guide_seq.is_some(),
            guide_seq,
            guide: None,
            guide_next: None,
            guide_src: GuideSrc::Chain,
            guide_next_src: GuideSrc::Chain,
            guide_ab: false,
            ab_lens: true,
            lens_sums: Vec::new(),
            lens_readings: 0,
            line_from: 0,
            lens_line_ended: false,
            lens_fork_mono: 0,
            lens_last: Vec::new(),
            guide_n: 0,
            guide_kl: 0.0,
            guide_flips: 0,
            guide_shared: 0.0,
            guide_shared_n: 0,
            guide_mix: 0.0,
            mixed: None,
            guide_log,
            reflecting: None,
            line_words: HashMap::new(),
            chain_fork_mono: i64::MIN / 2,
            line_ended: false,
            reflection: None,
            chain_against,
            reflection_against: false,
            goal_on: goal_probe && yes_no.is_some(),
            goal_mono: i64::MIN / 2,
            yes_no,
            goal_log,
            chain_rng: (clock::now_us() as u64) | 1,
            recent_reflections: VecDeque::new(),
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
        self.out_at(text, kind, t, None)
    }

    /// `out`, for a placed token at `pos`.
    fn out_at(&mut self, text: String, kind: Kind, t: i64, pos: Option<i32>) {
        self.log.write(text.as_bytes());
        {
            let f = &mut self.chain;
            let k = match kind {
                Kind::Think => "think",
                Kind::Speak => "speak",
                Kind::Given => "given",
            };
            f.line(&format!("{t}\t{k}\t{}", crate::client::escape(&text)));
        }
        let _ = self.tx.send(Event::Text(text, kind, t, pos));
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
                Piece::Token {
                    t,
                    chat_speaking,
                    pos,
                } => self.release_token(t, chat_speaking, h.t_us, pos),
            }
        }
    }

    /// A placed token going out: its text, its kind, its lines.
    fn release_token(&mut self, t: i32, chat_speaking: bool, t_us: i64, pos: i32) {
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
        self.out_at(text.clone(), kind, t_us, Some(pos));
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
        // What it kept from before (a restart): its last summary, its notes
        // and preferences.
        let kept = if self.cfg.task {
            String::new()
        } else {
            let summary = fs::read_to_string(self.cfg.workspace.join("summary.md"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let shown = match (summary.is_empty(), self.cfg.frame) {
                (true, _) => String::new(),
                (false, Frame::Journal) => {
                    format!("« [your own summary, written before you were restarted:]\n{summary}\n")
                }
                (false, Frame::Chat) => {
                    format!("Your own summary, written before you were restarted:\n{summary}\n")
                }
            };
            // Checked against the code like its notes (`verify.md`).
            let summary = if summary.is_empty() {
                shown
            } else {
                format!("{shown}{}", self.checked_line(&summary))
            };
            format!("{summary}{}", self.notes_block())
        };
        match self.cfg.frame {
            Frame::Journal => format!(
                "{}{}\n\n=== the journal ===\n\n« {when}{}\n{kept}{}\n{}",
                self.cfg.system,
                self.situation() + &self.about(),
                self.cfg.seed,
                self.changed_since,
                self.cfg.first_words
            ),
            Frame::Chat => format!(
                "<|im_start|>system\n{}{}<|im_end|>\n<|im_start|>user\n{when}{}{}{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                self.cfg.system,
                self.situation() + &self.about(),
                self.changed_since,
                self.cfg.seed,
                if kept.is_empty() { String::new() } else { format!("\n{kept}") }
            ),
        }
    }

    /// Where and when it runs (`situation.md`), read from the running system
    /// at this moment: the date and time with its zone, the host, the model
    /// as loaded and its context, its workspace and repository, who is
    /// present. Nothing in a task (a measurement's text must not move).
    fn situation(&self) -> String {
        if self.cfg.task {
            return String::new();
        }
        let host = crate::situation::host();
        let now = crate::situation::now_line(clock::now_us(), host.zone.as_deref());
        let model = Path::new(&self.llm.opts.model)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        // The cards hold rows of this run only when its backend is loaded
        // (phi-ggml.sh names it to ggml, `avx512.md`).
        let cards_on = std::env::var("GGML_BACKEND_PATH").is_ok_and(|p| p.contains("ggml_phi"));
        let rest = match (&host.cards, cards_on) {
            (Some(_), true) => "the rest on the host and the Xeon Phi cards",
            _ => "the rest on the host",
        };
        let (gpu, blocks) = (self.llm.split.gpu_blocks, self.llm.split.n_blocks);
        let ws = self.cfg.workspace.display();
        let journal = self.journal();
        let repo = match &self.cfg.dev {
            Some(r) if journal => format!(", developing the repository {}", r.display()),
            Some(r) => format!(", developing the repository {}", r.display()),
            None => String::new(),
        };
        let present = match (&self.cfg.dev, journal) {
            (Some(_), true) => "the person whose standing instructions are above, and Claude (an AI coding agent), whose lines say Claude",
            (Some(_), false) => "the person whose standing instructions are above, and Claude (an AI coding agent), whose words are marked Claude",
            (None, _) => "the person whose standing instructions are above",
        };
        let host_line = host.sentence();
        let lines = [
            format!("Now: {now}; each line from outside carries the time it arrived, read from the same clock."),
            if host_line.is_empty() {
                String::new()
            } else {
                format!("The computer: {host_line}.")
            },
            format!(
                "The model: {model}, {gpu} of its {blocks} blocks on the GPU, {rest}; its context holds {} tokens.",
                self.llm.n_ctx()
            ),
            format!("Its workspace: {ws}{repo}."),
            format!("Present: {present}."),
        ];
        let body: Vec<&str> = lines
            .iter()
            .map(String::as_str)
            .filter(|l| !l.is_empty())
            .collect();
        if journal {
            format!(
                "\n\n=== where and when this mind is ===\n{}",
                body.join("\n")
            )
        } else {
            format!("\n\nWhere and when you are:\n{}", body.join("\n"))
        }
    }

    /// What it is, in facts this run knows (`engine.md`): the model, its
    /// placement, its memory, what it perceives, how it goes on across
    /// rollovers and restarts, its tools, its objective. Nothing in a task
    /// (a measurement's text must not move).
    fn about(&self) -> String {
        if self.cfg.task {
            return String::new();
        }
        let ws = self.cfg.workspace.display();
        let journal = self.journal();
        let (it, its) = if journal {
            ("it", "its")
        } else {
            ("you", "your")
        };
        let terminal = if self.term.is_some() {
            format!("[run: COMMAND] runs a shell command in {its} terminal and hands the output back when it ends (it starts in the repository, where what it writes lands in {its} working copy: the repository is seen with {its} changes on top and itself never changes, and Claude reviews the changes; {its} workspace {ws} is writable too; no network), while {it} {} thinking; ", if journal { "goes on" } else { "go on" })
        } else {
            String::new()
        };
        let objective = match &self.objective {
            Some((_, t)) => format!("{its} objective: {t}"),
            None if self.cfg.gate_output => format!(
                "{it} {} no objective yet: until {it} {} given one, {it} {} not speak ({its} {} lines do nothing), and {its} tools work",
                if journal { "has" } else { "have" },
                if journal { "is" } else { "are" },
                if journal { "does" } else { "do" },
                if journal { "»" } else { "spoken" },
            ),
            None => format!("{it} {} no set objective", if journal { "has" } else { "have" }),
        };
        let tools_chat = if self.cfg.agent {
            "Your tools are the functions listed above (run, read, write, note): call them in their format, and their results come back to you before your next turn.".to_string()
        } else {
            format!("Your tools, each a line of its own: {terminal}[read: PATH] brings a file in; [note: ...] and [prefer: ...] keep a line across time.")
        };
        if journal {
            format!(
                "\n\n=== what this mind is ===\nThis mind is the language model named above, running without pause where it says; its memory is its context. It perceives only what is in that context: its own text, what people say and hand it (« lines, each with the time it arrived), and what its tools return. It does not see a screen or hear anything, and it knows only what it has read or been told, so it does not claim what it has not seen. When the context fills it writes a summary and goes on from it; its notes and preferences stay on disk and are shown to it again; when the program is restarted (for an update) it resumes the same way, from its last summary, and is told what changed. Its tools, each a line of its own: {terminal}[read: PATH] brings a file in; [note: ...] and [prefer: ...] keep a line across time. It works with its tools, not in its head: what a file says, it reads; whether something works, it runs; what it has done, a tool's output shows; and what it has not checked with a tool, it does not claim. Each tool use and its result appear in its journal and to the people watching it. A file changes only when one of its own commands writes it in its workspace and the output shows it; the program's repository changes only when Claude applies a change. Now {objective}."
            )
        } else {
            format!(
                "\n\nWhat you are: the language model named above, running without pause where it says; your memory is your context. You perceive only what is in that context: your own text, what people say and hand you (each with the time it arrived), and what your tools return. You do not see a screen or hear anything, and you know only what you have read or been told, so do not claim what you have not seen. When the context fills you write a summary and go on from it; your notes and preferences stay on disk and are shown to you again; when the program is restarted (for an update) you resume the same way, from your last summary, and are told what changed. {tools_chat} Work with your tools, not in your head: what a file says, read it; whether something works, run it; what you have done, a tool's output shows; and what you have not checked with a tool, do not claim. Each tool use and its result appear in your thoughts and to the people watching you. A file changes only when one of your own commands writes it in your workspace and the output shows it; the program's repository changes only when Claude applies a change. Now {objective}."
            )
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

    fn summary_ask(&self, why: Summary) -> String {
        // The ask, then the summary's first words in its own voice: in the
        // dev session it restated the instruction one word a line and ended
        // it with --- after 34 tokens, losing its thread at the rollover.
        format!(
            "{}{}",
            self.framed_system(&self.summary_request(why)),
            SUMMARY_START
        )
    }

    /// The summary asked for, with its real reason (a restart was once
    /// announced as "your memory is nearly full" at 3 thousand of 205).
    fn summary_request(&self, why: Summary) -> String {
        let reason = match why {
            Summary::Full => format!(
                "your memory is nearly full ({} of its {} tokens)",
                self.history.len(),
                self.llm.n_ctx()
            ),
            Summary::Restart => "the program you run in restarts now (an update or a change), and you resume after it".to_string(),
            Summary::Persona => "your persona changes now, and you resume under the new one".to_string(),
        };
        let tools = if self.cfg.agent {
            " Write it as your answer now, with no tool call."
        } else {
            ""
        };
        format!("{reason}. Write a compact summary of your threads, what matters, what you learned, and what you meant to do next, so that you can resume from it alone.{tools} End the summary with a line that is only ---")
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

    /// In development, a text it carries forward (a summary) checked
    /// against the code (`verify.md`): a line naming what the repository
    /// does not hold, empty when it names nothing false.
    fn checked_line(&self, text: &str) -> String {
        let Some(repo) = self.repo_view() else {
            return String::new();
        };
        let f = repo.check(text);
        if f.clean() {
            return String::new();
        }
        let mut why = Vec::new();
        if !f.missing.is_empty() {
            why.push(format!(
                "{} nowhere in the repository",
                f.missing.join(", ")
            ));
        }
        why.extend(f.bad_refs);
        match self.cfg.frame {
            Frame::Journal => format!("« [checked against the code: {}]\n", why.join("; ")),
            Frame::Chat => format!("[checked against the code: {}]\n", why.join("; ")),
        }
    }

    /// The base of a new context after a rollover: the persona, the
    /// summary, the notes.
    fn base_after(&self, summary: &str) -> String {
        match self.cfg.frame {
            Frame::Journal => format!(
                "{}{}\n\n=== the journal ===\n\n« [{}] [resuming from your own summary:]\n{}\n{}{}« the journal continues.\n\n{}",
                self.cfg.system,
                self.situation() + &self.about(),
                clock::datetime(clock::now_us()),
                summary,
                self.checked_line(summary),
                self.notes_block(),
                self.cfg.first_words
            ),
            Frame::Chat => format!(
                "<|im_start|>system\n{}{}<|im_end|>\n<|im_start|>user\n[{}] [You are resuming from your own summary:]\n{}\n{}{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                self.cfg.system,
                self.situation() + &self.about(),
                clock::datetime(clock::now_us()),
                summary,
                self.checked_line(summary),
                self.notes_block()
            ),
        }
    }

    /// The fence state after token `t`: each ``` in its text (joined to the
    /// last two characters before it) opens or closes a code block.
    fn track_fence(&mut self, t: i32) {
        let (toggle, tail) = fence_step(&self.fence_tail, &self.llm.text(&[t]));
        self.in_code ^= toggle;
        self.fence_tail = tail;
    }

    /// A live token placed: the state decisions read (the circling window,
    /// the thinking count, the chat frame's thoughts and speech) now; its
    /// text and side effects into the hold.
    fn emit_token(&mut self, t: i32) {
        self.generated.push_back(t);
        self.track_fence(t);
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
                // Where it will be decoded: the next position (its reading
                // comes when it is, a cycle later, before it is released).
                pos: self.pos(),
            },
            t_us: clock::now_us(),
        });
    }

    /// A line the mind wrote: a note to keep, a file to read.
    fn line_done(&mut self, line: &str) {
        let l = line.trim();
        // A line written again and again ("```" on every line, on the live
        // service, through DRY and the repeat penalty): at `LINE_REPEATS` in
        // a row its first token is held back for `HOLD_BACK_US`, and it is told.
        if !l.is_empty() && l == self.last_line {
            self.line_repeats += 1;
            // Only a line of some length, and one token held back at a time:
            // holding back "I", "The", "." as each repeated in turn forced it
            // into stranger output still, on the live service.
            if self.breaker_on
                && self.line_repeats + 1 >= LINE_REPEATS
                && l.chars().count() >= 3
                && self.held_back.is_empty()
            {
                self.line_repeats = 0;
                self.hold_back(l);
            }
        } else {
            self.line_repeats = 0;
            self.last_line = l.to_string();
        }
        // Its tools work with or without an objective) (their results are
        // real: what grounds it); each use goes to the terminals as an
        // `act` line, its result as another (`act_end`).
        let tool = |p: &str| {
            l.strip_prefix(p)
                .and_then(|r| r.strip_suffix(']'))
                .map(str::trim)
        };
        if let Some(body) = tool("[note:").filter(|b| !b.is_empty()) {
            let id = self.act("note", body);
            let kept = self.add_note(body);
            self.act_end(id, true, kept);
        } else if let Some(body) = tool("[unnote:").filter(|b| !b.is_empty()) {
            let id = self.act("unnote", body);
            let n = self.unnote(body);
            self.act_end(id, n > 0, format!("{n} notes removed"));
        } else if let Some(body) = tool("[prefer:").filter(|b| !b.is_empty()) {
            let id = self.act("prefer", body);
            self.add_preference(body);
            self.act_end(id, true, "kept in preferences.md".into());
        } else if let Some(path) = tool("[read:").filter(|p| !p.is_empty()) {
            if !self.pending_reads.iter().any(|(_, p)| p == path) {
                let id = self.act("read", path);
                self.pending_reads.push((id, path.to_string()));
            }
        } else if let Some(cmd) = tool("[run:") {
            let _ = self.run_command(cmd);
        }
    }

    /// A line of the journal ended (`after`): the waiting reflection joins
    /// it, and the second chain forks to reflect on the line just ended,
    /// when nothing else is in flight.
    /// A line ended: with the guide's source `lens` (or `placebo`), the next
    /// aside from its J-lens words (`GuideSrc`). The line's text from
    /// `history` (the released text lags by the horizon), its words from its
    /// readings; only in thinking, at most every `LENS_EVERY_US`, never the
    /// same words twice running.
    fn lens_aside(&mut self) {
        let from = self.line_from.min(self.history.len());
        let said = self.llm.text(&self.history[from..]);
        self.line_from = self.history.len();
        let sums = std::mem::take(&mut self.lens_sums);
        let readings = std::mem::take(&mut self.lens_readings);
        if !self.guide_on || self.guide_src == GuideSrc::Chain || self.speaking || self.in_code {
            return;
        }
        let words = crate::mind::unsaid(&sums, readings, &said, LENS_ASIDE_MIN);
        let mono = clock::mono_us();
        if words.is_empty() || mono - self.lens_fork_mono < LENS_EVERY_US {
            return;
        }
        // `guide ab`: lens and placebo by turns, fork by fork.
        let src = match (self.guide_ab, self.ab_lens) {
            (true, true) => GuideSrc::Lens,
            (true, false) => GuideSrc::Placebo,
            _ => self.guide_src,
        };
        let names: Vec<String> = match src {
            GuideSrc::Placebo => crate::mind::said(&sums, readings, &said, words.len()),
            _ => words,
        }
        .into_iter()
        .map(|w| w.0)
        .collect();
        if names.is_empty() || names == self.lens_last {
            return;
        }
        self.lens_fork_mono = mono;
        self.lens_last = names.clone();
        self.guide_next = Some(format!("on my mind: {}", names.join(", ")));
        self.guide_next_src = src;
        if self.guide_ab {
            self.ab_lens = !self.ab_lens;
        }
    }

    fn on_line_end(&mut self) -> Result<()> {
        let quiet = self.check.is_none()
            && self.reading.is_none()
            && self.chase.is_none()
            && self.summary.is_none()
            && !self.in_code;
        if !quiet {
            return Ok(());
        }
        // The agent frame: a reflection is told at its next user turn
        // (nothing goes inside a turn), and the chain forks only from its
        // thinking.
        if self.cfg.agent {
            if let Some(text) = self.reflection.take() {
                // Beside the waiting lines: a reflection neither stops a rest nor
                // wakes one (each turn left one waiting, and every rest was refused).
                let at = clock::hms(clock::now_us());
                // The opposing chain's objection asks for an answer: the
                // two reason against each other, toward the one objective.
                self.asides.push(if self.reflection_against {
                    format!("[{at}] the other side of your thinking, against your last line: {text} Answer it in your thinking: concede it or rebut it, and keep to the objective.")
                } else {
                    format!("[{at}] your second look, beside your turn: {text}")
                });
                // The last two, at most: older ones are about turns long gone.
                while self.asides.len() > 2 {
                    self.asides.remove(0);
                }
                let _ = self.tx.send(Event::Delib(crate::client::Delib {
                    kind: crate::client::DelibKind::End,
                    t_us: clock::now_us(),
                    pos: self.history.len() as i32,
                    text: "told at its next turn".into(),
                }));
            }
            if self.speaking {
                self.line_words.clear();
                return Ok(());
            }
        }
        if let Some(text) = self.reflection.take() {
            let at = clock::hms(clock::now_us());
            let side = if self.reflection_against {
                "against"
            } else {
                "beside"
            };
            let line = match self.cfg.frame {
                Frame::Journal => format!("\n« [{at}] [{side} the journal: {text}]\n"),
                Frame::Chat => format!("\n[at {at}, {side} your thoughts: {text}]\n"),
            };
            // Its weight on the main chain: the next-token distribution with
            // it, against a copy's with the same frame and nothing in it
            // (a placebo): what the reflection says, not that a line came
            // (the frame alone moved the next token, at first measured as
            // 7 to 14 nats).
            let empty = match self.cfg.frame {
                Frame::Journal => format!("\n« [{at}] [{side} the journal: ]\n"),
                Frame::Chat => format!("\n[at {at}, {side} your thoughts: ]\n"),
            };
            let placebo = self.tok(&empty, false)?;
            let without = self.logits_without(&placebo)?;
            let tokens = self.tok(&line, false)?;
            let with = self.direct_logits(&tokens)?;
            self.say(line, Kind::Given);
            if let Some(without) = without {
                let (kl, a, b) = weigh(&with, &without);
                let (a, b) = (self.llm.text(&[a as i32]), self.llm.text(&[b as i32]));
                let said = format!(
                    "weight on the journal: {kl:.3} nats; its likeliest next token {a:?}, with an empty one {b:?}"
                );
                self.note(format!("a reflection joined the journal: {said}"));
                let _ = self.tx.send(Event::Delib(crate::client::Delib {
                    kind: crate::client::DelibKind::End,
                    t_us: clock::now_us(),
                    pos: self.history.len() as i32,
                    text: said,
                }));
            }
        }
        self.goal_probe()?;
        let mono = clock::mono_us();
        let words = std::mem::take(&mut self.line_words);
        if self.reflecting.is_some()
            || mono - self.chain_fork_mono < CHAIN_EVERY_US
            || self.free_seqs.len() < 3
            || words.is_empty()
        {
            return Ok(());
        }
        // The line's J-space words, likeliest first.
        let mut words: Vec<(String, f32)> = words.into_iter().collect();
        words.sort_by(|a, b| b.1.total_cmp(&a.1));
        let shown: Vec<String> = words.into_iter().take(6).map(|w| w.0).collect();
        // Against the line, toward the objective (`chain against`): the
        // dual of the reflection, both chains held to the one goal. With no
        // objective there is nothing to hold it to, and it reflects.
        let goal = self
            .objective
            .as_ref()
            .filter(|_| self.chain_against)
            .map(|o| o.1.replace(['[', ']'], ""));
        let against = goal.is_some();
        let marker = match (&goal, self.cfg.frame) {
            (Some(g), Frame::Chat) if self.cfg.agent => format!(
                "<|im_end|>\n<|im_start|>user\n[The other side of your thinking, beside it. The objective: {g}. On your mind in its last line: {}. Argue against that line as a step toward the objective: in a sentence or two, the strongest objection to it, or where it drifts from the objective.]<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{AGAINST_PRIMER}",
                shown.join(", ")
            ),
            (Some(g), Frame::Journal) => format!(
                "\n« [against the journal, toward its objective ({g}); on its mind in the line above: {}]\n{AGAINST_PRIMER}",
                shown.join(", ")
            ),
            (Some(g), Frame::Chat) => format!(
                "\n[against your thoughts, toward your objective ({g}); on your mind in the line above: {}]\n{AGAINST_PRIMER}",
                shown.join(", ")
            ),
            (None, _) => self.reflect_marker(&shown),
        };
        let prompt = self.tok(&marker, self.cfg.agent)?;
        if prompt.is_empty() || prompt.len() >= self.llm.batch_cap() {
            return Ok(());
        }
        let seq = self.free_seqs.pop().unwrap();
        self.llm.seq_rm(seq, -1, -1);
        self.llm.seq_cp(self.live, seq, -1, -1);
        self.chain_fork_mono = mono;
        let pos = self.history.len() as i32;
        let _ = self.tx.send(Event::Delib(crate::client::Delib {
            kind: crate::client::DelibKind::Start,
            t_us: clock::now_us(),
            pos,
            text: format!(
                "{} the line before {pos}: {}",
                if against { "against" } else { "on its mind in" },
                shown.join(", ")
            ),
        }));
        self.reflecting = Some(Chain {
            seq,
            against,
            prompt,
            fed: 0,
            out: Vec::new(),
            pos,
        });
        Ok(())
    }

    /// The goal probe (`goal on`): at a thinking line's end, with an
    /// objective, at most every `GOAL_EVERY_US`, a copy of the live sequence
    /// is asked whether the line serves the objective, and the answer read
    /// as the probability of yes against no (their one-token forms, not a
    /// sample), as the check reads keep against write. One copy and one
    /// prefill of the question, in the cycle; the live sequence untouched.
    /// Each answer is a line of `goal.log`, with the chain's kind, so the
    /// arms of an interleaved run are told apart.
    fn goal_probe(&mut self) -> Result<()> {
        let mono = clock::mono_us();
        if !self.goal_on
            || self.speaking
            || mono - self.goal_mono < GOAL_EVERY_US
            || self.free_seqs.len() < 3
        {
            return Ok(());
        }
        let Some(g) = self.objective.as_ref().map(|o| o.1.replace(['[', ']'], "")) else {
            return Ok(());
        };
        let q = match self.cfg.frame {
            Frame::Chat if self.cfg.agent => format!(
                "<|im_end|>\n<|im_start|>user\n[A question beside your work: does your last line serve the objective ({g})? Answer yes or no.]<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
            ),
            Frame::Journal => format!(
                "\n« [a question beside the journal: does the line above serve its objective ({g})? yes or no]\nAnswer:"
            ),
            Frame::Chat => format!(
                "\n[a question beside your thoughts: does the line above serve your objective ({g})? yes or no]\nAnswer:"
            ),
        };
        let tokens = self.tok(&q, self.cfg.agent)?;
        if tokens.is_empty() || tokens.len() + 1 >= self.llm.batch_cap() {
            return Ok(());
        }
        self.goal_mono = mono;
        let Some(l) = self.logits_without(&tokens)? else {
            return Ok(());
        };
        let Some((y, n)) = self.yes_no.as_ref() else {
            return Ok(());
        };
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>();
        let mass = |ts: &[i32]| {
            ts.iter()
                .map(|&t| (l[t as usize] as f64 - m).exp())
                .sum::<f64>()
                / z
        };
        let (py, pn) = (mass(y), mass(n));
        let kind = match (self.chain_on, self.chain_against) {
            (false, _) => "off",
            (true, false) => "on",
            (true, true) => "against",
        };
        self.goal_log.line(&format!(
            "{}\tpos={}\tchain={kind}\tyes={:.4}\tmass={:.4}",
            clock::now_us(),
            self.history.len(),
            if py + pn > 0.0 { py / (py + pn) } else { 0.5 },
            py + pn
        ));
        Ok(())
    }

    /// The reflecting chain's opening: its question with the line's J-space
    /// words, then its first words.
    fn reflect_marker(&self, shown: &[String]) -> String {
        match self.cfg.frame {
            // The agent frame: asked in a user turn on a copy, as the check
            // is (inside its turn a bracketed line is read as noise).
            Frame::Chat if self.cfg.agent => format!(
                "<|im_end|>\n<|im_start|>user\n[A second look at your thinking, beside it: on your mind in its last line: {}. In a sentence or two: what are you missing, getting wrong, or not checking with a tool?]<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{CHAIN_PRIMER}",
                shown.join(", ")
            ),
            Frame::Journal => format!(
                "\n« [beside the journal; on its mind in the line above: {}]\n{CHAIN_PRIMER}",
                shown.join(", ")
            ),
            Frame::Chat => format!(
                "\n[beside your thoughts; on your mind in the line above: {}]\n{CHAIN_PRIMER}",
                shown.join(", ")
            ),
        }
    }

    /// The second chain's next token: the likeliest one that is not banned
    /// for the live stream (control tokens, the « and » marks) nor an end
    /// of text, so a reflection never writes a line from outside or speaks.
    fn chain_token(&mut self, row: i32) -> Result<i32> {
        // The 16 likeliest in one pass, then the first allowed: a check of
        // every token of the vocabulary against the bans cost too much at
        // every cycle.
        let l = self.llm.logits(row)?;
        let mut top: Vec<(i32, f32)> = Vec::with_capacity(17);
        for (t, &x) in l.iter().enumerate() {
            if top.len() < 16 || x > top[15].1 {
                let at = top.iter().position(|e| x > e.1).unwrap_or(top.len());
                top.insert(at, (t as i32, x));
                top.truncate(16);
            }
        }
        let allowed = |t: i32| {
            !self.base_ban.contains(&t) && !self.speak_ban.contains(&t) && !self.llm.is_eog(t)
        };
        let top: Vec<(i32, f32)> = top.into_iter().filter(|e| allowed(e.0)).collect();
        let Some(&(first, m)) = top.first() else {
            return Ok(self.newline);
        };
        // Sampled at temperature `CHAIN_TEMP` with its own random state (the
        // live sampler's history is the live chain's): greedy, two forks of
        // nearly the same context wrote the same reflection again and again.
        let w: Vec<f64> = top
            .iter()
            .map(|e| (((e.1 - m) / CHAIN_TEMP) as f64).exp())
            .collect();
        let total: f64 = w.iter().sum();
        self.chain_rng ^= self.chain_rng << 13;
        self.chain_rng ^= self.chain_rng >> 7;
        self.chain_rng ^= self.chain_rng << 17;
        let mut u = (self.chain_rng >> 11) as f64 / (1u64 << 53) as f64 * total;
        for (e, wi) in top.iter().zip(&w) {
            if u < *wi {
                return Ok(e.0);
            }
            u -= wi;
        }
        Ok(first)
    }

    /// The second chain's reflection ends: kept for the journal's next
    /// line (`keep`), or dropped (the journal moved under it: a rollover,
    /// a word written over); its sequence freed.
    fn end_chain(&mut self, keep: bool) {
        let Some(c) = self.reflecting.take() else {
            return;
        };
        self.llm.seq_rm(c.seq, -1, -1);
        self.free_seqs.push(c.seq);
        let said = self.llm.text(&c.out).trim().to_string();
        let primer = if c.against {
            AGAINST_PRIMER
        } else {
            CHAIN_PRIMER
        };
        let text = format!("{primer} {said}");
        // What only repeats the journal's frame is no reflection.
        let echo = said.is_empty()
            || said.contains("beside the journal")
            || said.contains("on its mind in the line")
            || said.contains("second look at your thinking")
            || said.contains("against the journal")
            || said.contains("other side of your thinking");
        let opening = word_set(&said);
        let repeat = self
            .recent_reflections
            .iter()
            .any(|r| jaccard(r, &opening) >= CHAIN_SAME);
        let outcome = if keep && !echo && repeat {
            "dropped: it repeats a recent reflection"
        } else if keep && !echo {
            self.recent_reflections.push_back(opening);
            while self.recent_reflections.len() > CHAIN_RECENT {
                self.recent_reflections.pop_front();
            }
            if self.guide_on && self.guide_src == GuideSrc::Chain {
                self.guide_next = Some(text.clone());
                self.guide_next_src = GuideSrc::Chain;
            }
            self.reflection = Some(text);
            self.reflection_against = c.against;
            if self.cfg.agent {
                "kept for its next turn"
            } else {
                "into the journal at its next line's end"
            }
        } else if keep {
            "nothing to say"
        } else {
            "dropped: the journal moved under it"
        };
        let _ = self.tx.send(Event::Delib(crate::client::Delib {
            kind: crate::client::DelibKind::End,
            t_us: clock::now_us(),
            pos: c.pos,
            text: outcome.to_string(),
        }));
    }

    /// The guide lane given up: its sequence emptied (the live one was
    /// replaced or cut under it, or it ran its length).
    fn drop_guide(&mut self) {
        if let (Some(_), Some(s)) = (self.guide.take(), self.guide_seq) {
            self.llm.seq_rm(s, -1, -1);
        }
    }

    /// The guide lane's part of this cycle's batch, at most `room` tokens:
    /// the aside's rest, the live tokens it has not had, and the live
    /// pending token, whose row then reads beside the live one (`true`). A
    /// new reflection forks it again from the live sequence first, its
    /// aside's first token decoded alone, inside its thinking only.
    fn guide_take(&mut self, room: usize) -> Result<Option<(Vec<i32>, i32, bool)>> {
        let Some(gs) = self.guide_seq.filter(|_| self.guide_on) else {
            return Ok(None);
        };
        if !self.speaking && !self.in_code {
            if let Some(text) = self.guide_next.take() {
                let aside = self.tok(&format!("\n({text})\n"), false)?;
                if aside.len() >= 2 {
                    self.drop_guide();
                    self.llm.seq_rm(gs, -1, -1);
                    self.llm.seq_cp(self.live, gs, -1, -1);
                    let from = self.history.len();
                    self.llm.decode(&[Lane {
                        seq: gs,
                        tokens: &aside[..1],
                        pos0: from as i32,
                        logits: false,
                    }])?;
                    if let Some(cap) = self.llm.capture() {
                        cap.take();
                    }
                    self.guide = Some(Guide {
                        from,
                        pending: aside[1..].to_vec(),
                        fed: 0,
                        pos: from as i32 + 1,
                        src: self.guide_next_src,
                    });
                }
            }
        }
        let Some((from, fed, pos, mut lane)) = self
            .guide
            .as_ref()
            .map(|g| (g.from, g.fed, g.pos, g.pending.clone()))
        else {
            return Ok(None);
        };
        if from + fed > self.history.len() || fed > GUIDE_MAX {
            self.drop_guide();
            return Ok(None);
        }
        if room == 0 {
            return Ok(None);
        }
        lane.extend_from_slice(&self.history[from + fed..]);
        lane.push(self.next);
        if lane.len() > room {
            // Catching up: no row this cycle.
            lane.truncate(room.min(lane.len() - 1));
            return Ok(Some((lane, pos, false)));
        }
        Ok(Some((lane, pos, true)))
    }

    /// The guide lane fed `n` tokens; when they ended on the live pending
    /// token, its row against the live one's (`guide_measure`).
    fn guide_fed(&mut self, n: usize, aligned: bool, rows: Option<(i32, i32)>) -> Result<()> {
        let Some(g) = self.guide.as_mut() else {
            return Ok(());
        };
        let from_aside = n.min(g.pending.len());
        g.pending.drain(..from_aside);
        // The live pending token counts as fed: `advance` puts it in the
        // history next.
        g.fed += n - from_aside;
        g.pos += n as i32;
        if let (true, Some((live, guide))) = (aligned, rows) {
            if !self.speaking && !self.in_code {
                self.guide_measure(live, guide)?;
            }
        }
        Ok(())
    }

    /// At a thinking token: how far the reflection in the guide moves the
    /// next-token distribution (KL of the guide's from the live one's, in
    /// nats) and whether it changes the likeliest token; a line in
    /// `guide.log`, and every `GUIDE_REPORT` tokens a report.
    fn guide_measure(&mut self, live: i32, guide: i32) -> Result<()> {
        if let Some(e) = self.llm.capture().and_then(|c| c.experts_error.take()) {
            self.note(format!(
                "the experts' capture failed and was turned off: {e}"
            ));
        }
        let lg = self.llm.logits(guide)?.to_vec();
        let ll = self.llm.logits(live)?;
        let (kl, ag, al) = kl_and_tops(&lg, ll);
        let flip = ag != al;
        // Mixed into the choice (`set guide-mix G`, off by default): the live
        // logits moved toward the guided ones by G, so the reflection weighs
        // on every thinking token without its text in the stream (G = 1: as
        // if it had been written in).
        if self.guide_mix > 0.0 {
            let m = self.guide_mix;
            self.mixed = Some(ll.iter().zip(&lg).map(|(&l, &g)| l + m * (g - l)).collect());
        }
        self.guide_n += 1;
        self.guide_kl += kl;
        self.guide_flips += flip as u32;
        // The experts each was routed to at the captured blocks (`experts
        // on`): the share they hold in common, block by block.
        let shared = self.llm.capture().and_then(|c| {
            let of = |row: i32| {
                c.row_experts
                    .iter()
                    .find(|(mb, r, _)| *mb == 0 && *r == row)
                    .map(|(_, _, e)| e.clone())
            };
            Some(experts_shared(&of(live)?, &of(guide)?))
        });
        if let Some(s) = shared {
            self.guide_shared += s;
            self.guide_shared_n += 1;
        }
        let src = self.guide.as_ref().map_or(GuideSrc::Chain, |g| g.src);
        let _ = self.tx.send(Event::Guide(crate::client::GuideLine {
            t_us: clock::now_us(),
            pos: self.history.len() as i32,
            kl: kl as f32,
            flip,
            shared: shared.map(|s| s as f32),
            mix: self.guide_mix,
            src: src.name().to_string(),
        }));
        let (tl, tg) = (self.llm.text(&[al as i32]), self.llm.text(&[ag as i32]));
        self.guide_log.line(&format!(
            "{}\tpos={}\tkl={kl:.4}\tflip={}\tlive={tl:?}\tguide={tg:?}{}\tsrc={}",
            clock::now_us(),
            self.history.len(),
            flip as u8,
            shared.map_or(String::new(), |s| format!("\texperts_shared={s:.3}")),
            src.name()
        ));
        if self.guide_n >= GUIDE_REPORT {
            let experts = if self.guide_shared_n > 0 {
                format!(
                    "; at the captured blocks the two share {:.1}% of their experts",
                    100.0 * self.guide_shared / self.guide_shared_n as f64
                )
            } else {
                String::new()
            };
            let mode = if self.guide_mix > 0.0 {
                format!("mixed at {}", self.guide_mix)
            } else {
                "shadow".to_string()
            };
            let said = format!(
                "guide ({mode}, {} asides), {} thinking tokens: the aside would move each by {:.3} nats on average and change the likeliest token at {:.1}%{experts}",
                src.name(),
                self.guide_n,
                self.guide_kl / self.guide_n as f64,
                100.0 * self.guide_flips as f64 / self.guide_n as f64
            );
            self.guide_shared = 0.0;
            self.guide_shared_n = 0;
            self.note(said.clone());
            let _ = self.tx.send(Event::Delib(crate::client::Delib {
                kind: crate::client::DelibKind::End,
                t_us: clock::now_us(),
                pos: self.history.len() as i32,
                text: said,
            }));
            self.guide_n = 0;
            self.guide_kl = 0.0;
            self.guide_flips = 0;
        }
        Ok(())
    }

    /// What it works toward, set (or cleared, empty): kept in
    /// `objective.md`, told to it as a line from the system, sent to the
    /// terminals; its output opens (or idles again).
    fn set_objective(&mut self, text: &str) {
        let now = clock::now_us();
        let p = self.cfg.workspace.join("objective.md");
        if text.is_empty() {
            let _ = fs::remove_file(&p);
            self.objective = None;
        } else {
            let _ = fs::write(&p, format!("{text}\n"));
            self.objective = Some((now, text.to_string()));
        }
        self.apply_gate();
        let said = match &self.objective {
            Some((_, t)) => format!("your objective is now: {t}"),
            None => "you have no objective now: you think, and your tool lines and » lines do nothing until you are given one".to_string(),
        };
        let _ = self.tell(&said);
        let _ = self.tx.send(Event::Objective(now, text.to_string()));
        self.note(format!(
            "objective: {}",
            if text.is_empty() { "none" } else { text }
        ));
    }

    /// The sampler's banned tokens for the gate's state: speech held back
    /// while its output idles.
    fn apply_gate(&mut self) {
        let mut banned = self.base_ban.clone();
        if self.idle_output() {
            banned.extend(&self.speak_ban);
        }
        banned.extend(self.held_back.iter().map(|(t, _)| *t));
        let s = self.cfg.sampling.clone();
        self.llm.set_banned(&banned, &s, &self.history);
    }

    /// Its output idles: gated (`--no-objective-gate` not given) and no
    /// objective.
    fn idle_output(&self) -> bool {
        self.cfg.gate_output && self.objective.is_none()
    }

    /// A line it wrote `LINE_REPEATS` times running: its first token held
    /// back for `HOLD_BACK_US` (added to the banned tokens, `apply_gate`), a
    /// fence's state forgotten (a run of "```" lines leaves it meaningless),
    /// and it is told.
    fn hold_back(&mut self, line: &str) {
        let Some(&first) = self
            .llm
            .tokenize(line, false)
            .ok()
            .and_then(|t| t.first().cloned())
            .as_ref()
        else {
            return;
        };
        let until = clock::mono_us() + HOLD_BACK_US;
        self.held_back.retain(|(t, _)| *t != first);
        self.held_back.push((first, until));
        if line.contains("```") {
            self.in_code = false;
            self.fence_tail.clear();
        }
        self.apply_gate();
        let piece = self.llm.text(&[first]);
        self.note(format!(
            "a line repeated {LINE_REPEATS} times running: {line:?}; {piece:?} held back {} s",
            HOLD_BACK_US / 1_000_000
        ));
        let msg = self.framed_system(&format!(
            "you wrote the line {line:?} {LINE_REPEATS} times running; it is set aside for {} s: go on with something else, concretely, and check something real with a tool",
            HOLD_BACK_US / 1_000_000
        ));
        self.queue.push_back((msg, "a repeated line".into()));
    }

    /// The agent frame: its turn ended (`after`). Its calls (after its
    /// thinking) run: read, write and note at once, a command in the
    /// terminal; when every result is in (`finish_agent_wait`), they go back
    /// to it as tool responses and its next turn opens. A turn with no call
    /// is answered with the time and its objective, so it goes on.
    fn agent_turn_end(&mut self) -> Result<()> {
        let from = self.turn_start.min(self.history.len());
        let text = self.llm.text(&self.history[from..]);
        let content = text
            .rsplit_once("</think>")
            .map_or("", |(_, c)| c)
            .to_string();
        let (mut calls, mut bad) = crate::agent::parse_calls(&content);
        // Calls written inside its thinking, in a turn that never closed it:
        // they run, and it is told to close its thoughts first (3 percent of
        // its calls on the live service were dropped so, without a word, and
        // the turn after told it to act with a tool).
        let mut in_thinking = false;
        if calls.is_empty() && bad == 0 && !text.contains("</think>") {
            let (c, b) = crate::agent::parse_calls(&text);
            if !c.is_empty() {
                self.note(format!(
                    "{} calls written inside its thinking: run",
                    c.len()
                ));
                calls = c;
                bad = b;
                in_thinking = true;
            }
        }
        // A summary due comes first; the turn's calls are not run.
        if let Some(why) = self.summary_due.take() {
            if !calls.is_empty() {
                self.note(format!(
                    "{} calls not run: the summary comes first",
                    calls.len()
                ));
            }
            return self.open_summary(why);
        }
        if calls.is_empty() {
            let extra = self.take_waiting();
            let turn = if bad > 0 {
                self.note(format!("{bad} tool calls did not parse"));
                crate::agent::responses_turn(
                    &[format!(
                        "{bad} tool call(s) did not parse: write <tool_call>, then <function=NAME>, then each <parameter=KEY>, its value and </parameter>, then </function> and </tool_call>"
                    )],
                    &extra,
                )
            } else {
                crate::agent::continue_turn(
                    &clock::hms(clock::now_us()),
                    self.objective.as_ref().map(|o| o.1.as_str()),
                    &extra,
                )
            };
            return self.agent_open(turn);
        }
        let mut wait = AgentWait {
            results: vec![None; calls.len()],
            runs: HashMap::new(),
        };
        for (i, c) in calls.iter().enumerate() {
            let result = match c.name.as_str() {
                "run" => match (c.param("command"), self.term.is_some()) {
                    (Some(cmd), true) => {
                        if let Some(id) = self.run_command(cmd) {
                            wait.runs.insert(id, i);
                            None
                        } else {
                            Some("the command did not run: too many are waiting".to_string())
                        }
                    }
                    (None, _) => Some("run needs a command".to_string()),
                    (_, false) => Some("there is no terminal in this run (--terminal)".to_string()),
                },
                "read" => Some(self.agent_read(c)),
                "write" => Some(self.agent_write(c)),
                "edit" => Some(self.agent_edit(c)),
                "note" => Some(match c.param("text") {
                    Some(t) if !t.trim().is_empty() => {
                        let id = self.act("note", t.trim());
                        let kept = self.add_note(t.trim());
                        self.act_end(id, true, kept.clone());
                        kept
                    }
                    _ => "note needs a text".to_string(),
                }),
                "wait" => {
                    let minutes = c
                        .param("minutes")
                        .and_then(|m| m.trim().parse::<i64>().ok())
                        .unwrap_or(15)
                        .clamp(1, 60);
                    let reason = c.param("reason").unwrap_or("").trim().to_string();
                    let act = self.act("wait", &reason);
                    let said = format!(
                        "resting up to {minutes} min; woken by a message from Claude, a new commit, a new objective, or the time"
                    );
                    self.act_end(act, true, said.clone());
                    self.rest_asked = Some((reason, minutes));
                    Some(said)
                }
                "tell_claude" => Some(match c.param("text") {
                    Some(t) if !t.trim().is_empty() => self.send_claude(t.trim(), c.param("re")),
                    _ => "tell_claude needs a text".to_string(),
                }),
                "propose" => Some(self.agent_propose(c)),
                other => Some(format!(
                    "there is no function {other:?}: the functions are run, read, edit, write, note, wait{} and tell_claude",
                    if self.improver.is_some() { ", propose" } else { "" }
                )),
            };
            wait.results[i] = result;
        }
        if bad > 0 {
            wait.results
                .push(Some(format!("{bad} more tool call(s) did not parse")));
        }
        if in_thinking {
            wait.results.push(Some(
                "your calls were written inside your thinking; they ran, but close your thoughts with </think> before you call".to_string(),
            ));
        }
        self.awaiting = Some(wait);
        self.finish_agent_wait()
    }

    /// When every result of the calls it waits for is in: they go back to it
    /// as tool responses, and its next turn opens.
    fn finish_agent_wait(&mut self) -> Result<()> {
        let done = self
            .awaiting
            .as_ref()
            .is_some_and(|w| w.results.iter().all(Option::is_some));
        if !done {
            return Ok(());
        }
        let w = self.awaiting.take().unwrap();
        let results: Vec<String> = w.results.into_iter().flatten().collect();
        if let Some(why) = self.summary_due.take() {
            self.rest_asked = None;
            return self.open_summary(why);
        }
        // It asked to rest, and nothing waits for it already: its next turn
        // opens when something new comes (`rest_look`).
        if let Some((reason, minutes)) = self.rest_asked.take() {
            if self.waiting.is_empty() && self.queue.is_empty() {
                let mono = clock::mono_us();
                self.note(format!("resting up to {minutes} min: {reason}"));
                self.rest = Some(Rest {
                    reason,
                    minutes,
                    until_mono: mono + minutes * 60_000_000,
                    next_look_mono: mono,
                    results,
                });
                let _ = self.tx.send(Event::Status(self.status()));
                return Ok(());
            }
            // Something came while it asked: it is told so, not left to think
            // it rested.
            self.waiting.insert(
                0,
                format!(
                    "[{}] not rested: something came for you, below",
                    clock::hms(clock::now_us())
                ),
            );
        }
        let extra = self.take_waiting();
        self.agent_open(crate::agent::responses_turn(&results, &extra))
    }

    /// A head of the repository other than the one it was last told
    /// (`head_told`): the line that tells it, and it is told. Its rest's own
    /// head missed a commit made during a turn after its last `git log`
    /// (564ad0b, 5 s after): the rest began at the new head and slept.
    fn head_news(&mut self) -> Option<String> {
        let h = head_of(self.cfg.dev.as_ref()?)?;
        let old = self.head_told.replace(h.clone());
        match old {
            Some(old) if old != h => {
                let _ = fs::write(self.cfg.workspace.join("head-told"), &h);
                Some(format!("a new commit, {h} (it was {old}): review it"))
            }
            _ => None,
        }
    }

    /// At rest: wake when something came for it (a message, a line from the
    /// system, an objective), a new commit, a quit, or its time is up; its
    /// next turn opens with the rested turn's results and what woke it.
    fn rest_look(&mut self) -> Result<()> {
        let Some(r) = self.rest.as_ref() else {
            return Ok(());
        };
        let mono = clock::mono_us();
        let (look, until, minutes) = (mono >= r.next_look_mono, r.until_mono, r.minutes);
        let mut woke: Vec<String> = Vec::new();
        // The repository's head, every 2 s.
        if look {
            if let Some(r) = self.rest.as_mut() {
                r.next_look_mono = mono + 2_000_000;
            }
            if let Some(line) = self.head_news() {
                woke.push(line);
            }
        }
        if mono >= until {
            woke.push(format!("your rest of {minutes} min is over"));
        }
        let came = !self.waiting.is_empty() || !self.queue.is_empty();
        let quitting = self.quit_deadline.is_some();
        if woke.is_empty() && !came && !quitting {
            return Ok(());
        }
        let r = self.rest.take().unwrap();
        self.note(format!(
            "woken from rest ({}): {}",
            r.reason,
            if woke.is_empty() {
                "something came".to_string()
            } else {
                woke.join("; ")
            }
        ));
        if quitting {
            return self.open_summary(Summary::Restart);
        }
        let mut extra: Vec<String> = woke
            .iter()
            .map(|w| format!("[{}] woken: {w}", clock::hms(clock::now_us())))
            .collect();
        extra.extend(self.take_waiting());
        self.agent_open(crate::agent::responses_turn(&r.results, &extra))
    }

    /// The agent frame, between turns' ends: a turn that has run past
    /// `AGENT_TURN_MAX_US` (`AGENT_QUIT_GRACE_US` when quitting) while
    /// something waits for it (a summary, a line, a message) is closed, its
    /// unfinished calls not run, and the waiting comes in; circling words
    /// are told at the next turn.
    fn agent_stalled(&mut self) -> Result<()> {
        let mono = clock::mono_us();
        if self.nudges_on
            && !self.in_code
            && mono - self.last_nudge_mono >= self.cfg.nudge_every_us
            && self.circling()
        {
            self.last_nudge_mono = mono;
            self.tell(
                "your thoughts have been circling the same words: move on to something else, concretely, or, if your objective is met, rest with wait",
            )?;
            self.note("the thoughts were circling; told at the next turn".into());
        }
        if self.awaiting.is_some()
            || self.summary.is_some()
            || self.reading.is_some()
            || self.chase.is_some()
            || self.check.is_some()
        {
            return Ok(());
        }
        let due = self.summary_due.is_some() || !self.waiting.is_empty() || !self.queue.is_empty();
        let limit = if self.quit_deadline.is_some() {
            AGENT_QUIT_GRACE_US
        } else {
            AGENT_TURN_MAX_US
        };
        // At a line's start, so a thought is not cut mid-sentence; past twice
        // the limit, wherever it is (90 s at any token cut a review short).
        let ran = mono - self.turn_open_mono;
        if !due || ran < limit || (!self.line_start && ran < 2 * limit) {
            return Ok(());
        }
        self.note(format!(
            "its turn ran {} s with something waiting for it: closed",
            (mono - self.turn_open_mono) / 1_000_000
        ));
        // A thought cut off is closed first, so the turn reads as one.
        let close = if self.speaking { "" } else { "\n</think>\n\n" };
        let turn = match self.summary_due.take() {
            Some(why) => {
                let ask = format!(
                    "[{}] {}",
                    clock::hms(clock::now_us()),
                    self.summary_request(why)
                );
                format!("{close}{}", crate::agent::summary_turn(&ask, SUMMARY_START))
            }
            None => {
                let extra = self.take_waiting();
                format!(
                    "{close}{}",
                    crate::agent::continue_turn(
                        &clock::hms(clock::now_us()),
                        self.objective.as_ref().map(|o| o.1.as_str()),
                        &extra,
                    )
                )
            }
        };
        let summary = turn.ends_with(SUMMARY_START);
        self.agent_open(turn)?;
        if summary {
            self.summary = Some(vec![self.next]);
            self.speaking = true;
        }
        Ok(())
    }

    /// The agent frame: the summary asked in a user turn of its own, its
    /// answer opened with the summary's first words and no thinking; the
    /// summary is collected from its first token (`after`).
    fn open_summary(&mut self, why: Summary) -> Result<()> {
        let ask = format!(
            "[{}] {}",
            clock::hms(clock::now_us()),
            self.summary_request(why)
        );
        self.agent_open(crate::agent::summary_turn(&ask, SUMMARY_START))?;
        self.summary = Some(vec![self.next]);
        self.speaking = true;
        Ok(())
    }

    /// The next turn opened: `turn` (which begins by ending the last one)
    /// decoded after the pending token; when that token is the turn's end
    /// itself, it is not written twice.
    fn agent_open(&mut self, turn: String) -> Result<()> {
        let turn = if self.next == self.im_end {
            turn.strip_prefix("<|im_end|>").unwrap_or(&turn).to_string()
        } else {
            turn
        };
        let tokens = self.tok(&turn, true)?;
        self.direct(&tokens)?;
        self.speaking = false;
        self.say(turn, Kind::Given);
        self.turn_start = self.history.len();
        self.turn_open_mono = clock::mono_us();
        Ok(())
    }

    /// A path the agent named, for reading or writing: relative to the
    /// repository (its working copy: a file it wrote is read from there,
    /// a write goes there) or in its workspace; nothing else.
    fn agent_path(&self, path: &str, write: bool) -> std::result::Result<PathBuf, String> {
        let path = percent_decoded(path.trim());
        let base = self
            .cfg
            .dev
            .clone()
            .unwrap_or_else(|| self.cfg.workspace.clone());
        let p = normalized(&resolve(&path, &base));
        if let (Some(repo), Some(upper)) = (&self.cfg.dev, &self.copy_upper) {
            if let Ok(rel) = p.strip_prefix(normalized(repo)) {
                let copy = upper.join(rel);
                return Ok(if write || copy.exists() { copy } else { p });
            }
        }
        if p.starts_with(normalized(&self.cfg.workspace)) {
            return Ok(p);
        }
        Err(format!(
            "{} is outside the repository and your workspace",
            p.display()
        ))
    }

    /// The `read` tool: a file whole or by lines, or a directory's listing,
    /// within the room its context has.
    fn agent_read(&mut self, c: &crate::agent::Call) -> String {
        let path = c.param("path").unwrap_or("").to_string();
        let id = self.act("read", &path);
        let r = (|| -> std::result::Result<String, String> {
            if path.trim().is_empty() {
                return Err("read needs a path".into());
            }
            let p = self.agent_path(&path, false)?;
            if p.is_dir() {
                let mut names: Vec<String> = fs::read_dir(&p)
                    .map_err(|e| e.to_string())?
                    .flatten()
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
                return Ok(format!("{} holds:\n{}", p.display(), names.join("\n")));
            }
            let text = fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let lines: Vec<&str> = text.lines().collect();
            let n = lines.len();
            let a = c
                .param("start")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1usize)
                .max(1);
            let b = c
                .param("end")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(n)
                .min(n);
            if n > 0 && a > b {
                return Err(format!(
                    "{} has {n} lines; {a} to {b} is not a range of them",
                    p.display()
                ));
            }
            let body = if n == 0 {
                String::new()
            } else {
                lines[a - 1..b].join("\n")
            };
            let tokens = self
                .llm
                .tokenize(&body, false)
                .map(|t| t.len())
                .unwrap_or(0);
            let room = self.read_room();
            let budget = room.min(READ_MAX_TOKENS);
            if tokens <= budget {
                return Ok(format!(
                    "{} (lines {a} to {b} of {n}):\n{body}",
                    p.display()
                ));
            }
            // More than one read gives: the leading lines that fit, and
            // where the rest begins.
            let counts: Vec<usize> = lines[a - 1..b]
                .iter()
                .map(|l| self.llm.tokenize(l, false).map_or(0, |t| t.len()) + 1)
                .collect();
            let k = lines_within(&counts, budget);
            if k == 0 {
                return Err(format!(
                    "{} line {a} alone is {} tokens, {}",
                    p.display(),
                    counts[0],
                    past(budget, room)
                ));
            }
            let e = a + k - 1;
            Ok(format!(
                "{} (lines {a} to {e} of {n}; you asked to {b}, {tokens} tokens, {}: lines {} to {b} are not shown, read them with start {}):\n{}",
                p.display(),
                past(budget, room),
                e + 1,
                e + 1,
                lines[a - 1..e].join("\n")
            ))
        })();
        let (ok, text) = match r {
            Ok(t) => (true, t),
            Err(e) => (false, e),
        };
        let first = text.lines().next().unwrap_or("").to_string();
        self.act_end(id, ok, first);
        text
    }

    /// A command's output within what one result gives (`READ_MAX_TOKENS`,
    /// and the room): its leading lines, and what was cut. The byte cut
    /// (16 KiB) let one grep of `reflect.log` through at 9732 tokens (45 s
    /// of prefill), and a turn may run four commands.
    fn fit_output(&self, text: String) -> String {
        let room = self.read_room();
        let budget = room.min(READ_MAX_TOKENS);
        let tokens = self.llm.tokenize(&text, false).map_or(0, |t| t.len());
        if tokens <= budget {
            return text;
        }
        let lines: Vec<&str> = text.lines().collect();
        let counts: Vec<usize> = lines
            .iter()
            .map(|l| self.llm.tokenize(l, false).map_or(0, |t| t.len()) + 1)
            .collect();
        let k = lines_within(&counts, budget);
        // One line past it whole: its first characters (a token holds at
        // least one).
        let shown = if k == 0 {
            lines[0].chars().take(budget).collect::<String>()
        } else {
            lines[..k].join("\n")
        };
        format!(
            "{shown}\n[cut: {k} of {} lines shown; the output is {tokens} tokens, {}: narrow the command with head, tail or grep]",
            lines.len(),
            past(budget, room)
        )
    }

    /// The `write` tool: a whole file into its working copy of the
    /// repository (the repository itself never changes) or its workspace.
    /// The `edit` tool: one exact replacement in a file of its working copy
    /// or workspace (`old` must occur exactly once). A file of the
    /// repository not yet in its working copy is copied there first, so the
    /// repository itself never changes. Without it, it rewrote a whole file
    /// for each fix (three times for one tool, 5000 tokens of speech).
    fn agent_edit(&mut self, c: &crate::agent::Call) -> String {
        let path = c.param("path").unwrap_or("").to_string();
        let id = self.act("edit", &path);
        let r = (|| -> std::result::Result<String, String> {
            if path.trim().is_empty() {
                return Err("edit needs a path".into());
            }
            let old = c.param("old").ok_or("edit needs old")?;
            let new = c.param("new").ok_or("edit needs new")?;
            if old.is_empty() {
                return Err("old is empty: to create a file use write".into());
            }
            // Where it reads from (its copy, or the repository's) and where
            // the edit lands (always its copy).
            let from = self.agent_path(&path, false)?;
            let to = self.agent_path(&path, true)?;
            let text = fs::read_to_string(&from).map_err(|e| format!("{}: {e}", from.display()))?;
            let n = text.matches(old).count();
            if n == 0 {
                return Err(format!(
                    "{path}: old does not occur in it (read the lines first, and copy them exactly)"
                ));
            }
            if n > 1 {
                return Err(format!(
                    "{path}: old occurs {n} times; include more lines around the change so it occurs once"
                ));
            }
            let at = text[..text.find(old).unwrap()].lines().count() + 1;
            let changed = text.replacen(old, new, 1);
            if let Some(dir) = to.parent() {
                fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            fs::write(&to, &changed).map_err(|e| format!("{}: {e}", to.display()))?;
            Ok(format!(
                "edited {path} at line {at}: {} lines replaced by {}",
                old.lines().count().max(1),
                new.lines().count().max(usize::from(!new.is_empty()))
            ))
        })();
        let (ok, text) = match r {
            Ok(t) => (true, t),
            Err(e) => (false, e),
        };
        self.act_end(id, ok, text.clone());
        text
    }

    fn agent_write(&mut self, c: &crate::agent::Call) -> String {
        let path = c.param("path").unwrap_or("").to_string();
        let id = self.act("write", &path);
        let r = (|| -> std::result::Result<String, String> {
            if path.trim().is_empty() {
                return Err("write needs a path".into());
            }
            let content = c.param("content").ok_or("write needs a content")?;
            let p = self.agent_path(&path, true)?;
            if let Some(dir) = p.parent() {
                fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            let mut body = content.to_string();
            if !body.ends_with('\n') {
                body.push('\n');
            }
            fs::write(&p, &body).map_err(|e| format!("{}: {e}", p.display()))?;
            let where_ = if self.copy_upper.as_ref().is_some_and(|u| p.starts_with(u)) {
                "your working copy of the repository (the repository itself is unchanged)"
            } else {
                "your workspace"
            };
            Ok(format!(
                "wrote {} bytes, {} lines, to {path} in {where_}",
                body.len(),
                body.lines().count()
            ))
        })();
        let (ok, text) = match r {
            Ok(t) => (true, t),
            Err(e) => (false, e),
        };
        self.act_end(id, ok, text.clone());
        text
    }

    /// A tool it used: its id, sent to the terminals as an `act` line.
    fn act(&mut self, kind: &str, text: &str) -> u64 {
        let id = self.act_next;
        self.act_next += 1;
        self.last_tool_mono = clock::mono_us();
        let _ = self.tx.send(Event::Act(crate::client::ActLine {
            id,
            t_us: clock::now_us(),
            end: false,
            ok: true,
            kind: kind.to_string(),
            text: text.to_string(),
        }));
        id
    }

    /// What a tool use came to: sent as the `act` line ending it.
    fn act_end(&mut self, id: u64, ok: bool, text: String) {
        let _ = self.tx.send(Event::Act(crate::client::ActLine {
            id,
            t_us: clock::now_us(),
            end: true,
            ok,
            kind: String::new(),
            text,
        }));
    }

    /// A command line it wrote: run in its terminal's sandbox (`term.md`),
    /// in order, one at a time; its output comes back as a document when it
    /// ends (`poll_term`).
    fn run_command(&mut self, cmd: &str) -> Option<u64> {
        if cmd.is_empty() {
            return None;
        }
        let agent = self.cfg.agent;
        let Some(term) = self.term.as_mut() else {
            let msg = self.framed_system(&format!(
                "{cmd} did not run: there is no terminal in this run (it starts with --terminal)"
            ));
            if !agent {
                let _ = self.put(msg);
            }
            return None;
        };
        if self.term_pending >= MAX_TERM_PENDING {
            let msg = self.framed_system(&format!(
                "{cmd} did not run: {MAX_TERM_PENDING} commands are waiting already"
            ));
            if !agent {
                let _ = self.put(msg);
            }
            return None;
        }
        let id = term.submit(cmd);
        self.term_pending += 1;
        let act = self.act("run", cmd);
        self.run_acts.insert(id, act);
        let _ = self
            .tx
            .send(Event::TermStart(id, clock::now_us(), cmd.to_string()));
        self.note(format!("running: {cmd}"));
        Some(id)
    }

    /// Commands that ended: each as a document handed back to it, and to
    /// the terminals.
    /// `propose` (`improve.md`): its working copy's change put forward as
    /// one candidate, built on the improver's thread; the outcome comes at
    /// a later turn (`poll_improve`).
    fn agent_propose(&mut self, c: &crate::agent::Call) -> String {
        let title = c.param("title").unwrap_or("").trim().to_string();
        let why = c.param("why").unwrap_or("").trim().to_string();
        if title.is_empty() || why.is_empty() {
            return "propose needs a title and a why".to_string();
        }
        let told = self.head_told.clone();
        if self.improver.is_none() {
            return "there is no improvement loop in this run (--improve)".to_string();
        }
        let act = self.act("propose", &title);
        match self.improver.as_mut().unwrap().propose(&title, told) {
            Ok(id) => {
                self.improve_log.line(&format!(
                    "{}\tcandidate {id}\tproposed\t{title}\twhy: {why}",
                    clock::hms(clock::now_us())
                ));
                let said = format!(
                    "candidate {id}: staged and building in its sandbox (make check: format, clippy, build, tests; minutes); the outcome comes at a later turn and goes into improve.log"
                );
                self.act_end(act, true, said.clone());
                said
            }
            Err(busy) => {
                let said = format!(
                    "candidate {busy} is still building: one at a time; its outcome comes first"
                );
                self.act_end(act, false, said.clone());
                said
            }
        }
    }

    /// A candidate's outcome: into `improve.log`, told to the stream at its
    /// next turn (and it wakes a rest: its outcome is something to act on),
    /// and one that passed is sent to Claude for review.
    fn poll_improve(&mut self) {
        let Some(o) = self.improver.as_mut().and_then(|i| i.poll()) else {
            return;
        };
        let at = clock::hms(clock::now_us());
        let files = o.files.join(", ");
        self.improve_log.line(&format!(
            "{at}\tcandidate {}\t{}\t{}\tfiles: {files}\tbase {}\t{:.0} s\t{}",
            o.id,
            o.verdict.word(),
            o.title,
            o.base.get(..12).unwrap_or(&o.base),
            o.secs,
            o.summary.replace('\n', " | ")
        ));
        self.note(format!(
            "candidate {} {}: {} ({:.0} s)",
            o.id,
            o.verdict.word(),
            o.title,
            o.secs
        ));
        self.waiting.push(format!(
            "[{at}] your candidate {} ({}) {}: {}",
            o.id,
            o.title,
            o.verdict.word(),
            o.summary
        ));
        if o.verdict == crate::improve::Verdict::Passed {
            let _ = self.send_claude(
                &format!(
                    "[improve] candidate {} passed make check in its sandbox: {} (files: {files}; base {}). The change: {}",
                    o.id,
                    o.title,
                    o.base.get(..12).unwrap_or(&o.base),
                    o.dir.join("change.patch").display()
                ),
                None,
            );
        }
    }

    fn poll_term(&mut self) {
        while let Some(ran) = self.term.as_ref().and_then(|t| t.poll()) {
            self.term_pending = self.term_pending.saturating_sub(1);
            let now = clock::now_us();
            let _ = self.tx.send(Event::TermEnd(now, ran.clone()));
            if let Some(act) = self.run_acts.remove(&ran.id) {
                let first = ran
                    .out
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("(no output)");
                let lines = ran.out.lines().count();
                let summary = match (ran.code, ran.timed_out) {
                    (_, true) => format!("stopped at the time limit; {lines} lines: {first}"),
                    (Some(c), _) => format!("exit {c}, {:.0} ms, {lines} lines: {first}", ran.ms),
                    (None, _) => format!("did not run: {first}"),
                };
                self.act_end(act, ran.code == Some(0) && !ran.timed_out, summary);
            }
            let how = match (ran.code, ran.timed_out) {
                (_, true) => format!("stopped at the limit of {} s", MAX_TERM_SECS),
                (Some(c), _) => format!("exit {c}"),
                (None, _) => "it did not run".to_string(),
            };
            // A command stopped at the limit has no result, and says so: read
            // as "ended ... (no output)", three cargo checks stopped at 60 s
            // became "compilation passes" in its review.
            let what = if ran.timed_out {
                format!(
                    "the command `{}` did NOT finish: it was stopped at the limit of {} s, so its result is unknown (no exit code); what it had written by then{}",
                    ran.command,
                    MAX_TERM_SECS,
                    if ran.cut { ", cut at 16 KiB," } else { "" }
                )
            } else {
                format!(
                    "the command `{}` ended ({how}, {:.0} ms); its output{}",
                    ran.command,
                    ran.ms,
                    if ran.cut { ", cut at 16 KiB," } else { "" }
                )
            };
            let text = if ran.out.trim().is_empty() {
                "(no output)".to_string()
            } else {
                ran.out.clone()
            };
            // The agent frame waits for it: its result goes back in its slot.
            if let Some(w) = self.awaiting.as_mut() {
                if let Some(slot) = w.runs.remove(&ran.id) {
                    let text = self.fit_output(text);
                    if let Some(w) = self.awaiting.as_mut() {
                        w.results[slot] = Some(format!("{what}:\n{text}"));
                    }
                    continue;
                }
            }
            let framed = self.framed_doc(&text, &what, now);
            self.queue
                .push_back((framed, format!("ran {}", ran.command)));
        }
    }

    /// `tell_claude`: a message to Claude, kept in `to-claude.md` and sent
    /// to the terminals as a `claude` line (`phi-stream ask` waits for one,
    /// the MCP server's `inbox` reads them).
    fn send_claude(&mut self, text: &str, re: Option<&str>) -> String {
        // An answer names a message Claude sent (`c3`); an id Claude never
        // sent is no answer (on the live service it named c3 while none had
        // been asked, which also passed the repeat check below).
        // And each is answered once: an id answered already is no answer
        // either (it kept naming c1 after m13 had answered it).
        let mut warn = String::new();
        let re = re.and_then(|r| {
            let n: Option<u64> = r
                .chars()
                .filter(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok();
            match n.map(|n| (n, self.answered.get(&n))) {
                Some((n, None)) if self.asked.contains(&n) => Some(n),
                Some((n, Some(m))) => {
                    warn = format!(" (c{n} was answered already, by m{m}, so it went as a message of your own: re is for a first answer)");
                    None
                }
                _ => {
                    warn = format!(" ({} is no message of Claude's, so it went as a message of your own)", r.trim());
                    None
                }
            }
        });
        let re_id = re;
        let re = re.map(|n| format!("c{n}"));
        let re = re.as_deref();
        // A message that says again what its last one said is not sent: on
        // the live service each review was followed a turn later by a
        // "final" one restating it (m4 after m3, m6 after m5).
        let words = word_set(text);
        if let Some((last, prev)) = &self.last_to_claude {
            if re.is_none() && contained(&words, prev) >= TO_CLAUDE_SAME {
                self.note(format!(
                    "a message to Claude repeating m{last} was not sent"
                ));
                return format!(
                    "not sent: it repeats m{last}, which Claude has; send only what is new{warn}"
                );
            }
        }
        self.last_to_claude = Some((self.to_claude_next, words));
        let id = self.to_claude_next;
        self.to_claude_next += 1;
        if let Some(n) = re_id {
            self.answered.insert(n, id);
        }
        let t = clock::now_us();
        let re = re.map(|r| r.trim().to_string()).filter(|r| !r.is_empty());
        let act = self.act("tell_claude", text);
        let path = self.cfg.workspace.join("to-claude.md");
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let answers = re
                .as_ref()
                .map_or(String::new(), |r| format!(", answering {r}"));
            let _ = writeln!(f, "## m{id} at {}{answers}\n\n{text}\n", clock::datetime(t));
        }
        let _ = self.tx.send(Event::ToClaude(crate::client::ToClaude {
            id,
            t_us: t,
            re,
            text: text.to_string(),
        }));
        let said = format!("sent to Claude as m{id}; Claude answers in a later turn{warn}");
        self.act_end(act, true, said.clone());
        said
    }

    /// The working copy's view of the repository, for the checks of its
    /// notes (`verify.md`): the repository with what it wrote over it (the
    /// overlay's upper layer); checked against the repository alone, its own
    /// new files were "nowhere", and it noted the same proposal 131 times.
    fn repo_view(&self) -> Option<verify::Repo> {
        let root = self.cfg.dev.as_ref()?;
        Some(verify::Repo::load_over(root, self.copy_upper.as_deref()))
    }

    fn add_note(&mut self, body: &str) -> String {
        // A note already kept is not kept again: shown at every refresh, a
        // copy adds nothing (the live service kept one 131 times).
        if self.notes.iter().any(|n| unmarked(n).trim() == body.trim()) {
            self.note("a note already kept was not kept again".into());
            return "already kept in notes.md (it is shown to you at every refresh); each thing is kept once. To tell Claude something, use tell_claude".to_string();
        }
        // In development a note's code is checked against the code
        // (`verify.md`): a name that is not there is marked on the note, and
        // the stream is told what is.
        let mut kept = body.to_string();
        let mut checked = String::new();
        if let Some(repo) = self.repo_view() {
            let f = repo.check(body);
            let mut said = Vec::new();
            if !f.missing.is_empty() {
                said.push(format!(
                    "{} {} nowhere in the repository",
                    f.missing.join(", "),
                    if f.missing.len() == 1 { "is" } else { "are" }
                ));
            }
            said.extend(f.bad_refs.iter().cloned());
            for (p, n, l) in &f.lines {
                said.push(format!("line {n} of {p} is: {l}"));
            }
            if !f.clean() {
                kept = format!(
                    "{body} [unverified: {}]",
                    said[..said.len() - f.lines.len()].join("; ")
                );
            }
            // The agent frame has the check in the tool's result; a line
            // after it said the same again.
            if !said.is_empty() && self.cfg.agent {
                checked = said.join("; ");
            } else if !said.is_empty() {
                let msg = self.framed_system(&format!(
                    "your note checked against the code: {}{}",
                    said.join("; "),
                    if f.clean() {
                        String::new()
                    } else {
                        " (marked unverified; [unnote: TEXT] removes notes containing TEXT)".into()
                    }
                ));
                self.queue.push_back((msg, "checked a note".into()));
            }
        }
        self.notes.push(kept.clone());
        let path = self.cfg.workspace.join("notes.md");
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "- {kept}");
        }
        self.note(format!("noted: {kept}"));
        match (kept == body, checked.is_empty()) {
            (true, true) => "kept in notes.md".to_string(),
            (true, false) => format!("kept in notes.md; checked against the code: {checked}"),
            (false, _) => format!(
                "kept in notes.md, marked unverified (checked against your working copy of the repository: {checked})"
            ),
        }
    }

    /// `[unnote: TEXT]`: its notes containing TEXT (any case) removed, from
    /// memory and from `notes.md`.
    fn unnote(&mut self, text: &str) -> usize {
        let t = text.to_lowercase();
        let before = self.notes.len();
        self.notes.retain(|n| !n.to_lowercase().contains(&t));
        let gone = before - self.notes.len();
        let path = self.cfg.workspace.join("notes.md");
        let body: String = self.notes.iter().map(|n| format!("- {n}\n")).collect();
        let _ = fs::write(&path, body);
        self.note(format!("unnoted {gone} notes containing {text:?}"));
        // It hears the outcome, so it does not ask again (in the dev session
        // it repeated the same retractions every minute, unanswered).
        let msg = self.framed_system(&match gone {
            0 => format!("no note of yours contains {text:?}: nothing to remove"),
            1 => format!("removed your one note containing {text:?}"),
            n => format!("removed your {n} notes containing {text:?}"),
        });
        self.queue.push_back((msg, "unnoted".into()));
        gone
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
            // The line's J-space words, for the second chain: each word's
            // probability summed over the line's tokens and blocks.
            if self.chain_on {
                for (_, ws) in &r.layers {
                    for (w, lp) in ws {
                        let w = w.trim();
                        if !w.is_empty() {
                            *self.line_words.entry(w.to_string()).or_default() += lp.exp();
                        }
                    }
                }
            }
            // The same for the lens aside, kept apart and counted per
            // reading (the chain's words can outlast a line).
            if self.guide_on && self.guide_src != GuideSrc::Chain {
                for (_, ws) in &r.layers {
                    self.lens_readings += 1;
                    for (w, lp) in ws {
                        self.lens_sums.push((w.clone(), lp.exp()));
                    }
                }
            }
            let _ = self.tx.send(Event::Mind(r));
        }
        Ok(())
    }

    /// `all` into the live sequence from `pos0`, a batch at a time, logits
    /// of the last only: its row. Past one batch the status says how far it
    /// is after each (`prefill`): a tool response of 9311 tokens took 42 s,
    /// and the terminal said speaking all that time.
    fn feed_live(&mut self, all: &[i32], pos0: i32) -> Result<i32> {
        let mut row = 0;
        let cap = self.llm.batch_cap();
        let long = all.len() > cap;
        for (i, c) in all.chunks(cap).enumerate() {
            let last = (i + 1) * cap >= all.len();
            let rows = self.llm.decode(&[Lane {
                seq: self.live,
                tokens: c,
                pos0: pos0 + (i * cap) as i32,
                logits: last,
            }]);
            let rows = match rows {
                Ok(r) => r,
                Err(e) => {
                    self.prefill = None;
                    return Err(e);
                }
            };
            if let Some(&r) = rows.first() {
                row = r;
            }
            if long && !last {
                self.prefill = Some(((i * cap + c.len()), all.len()));
                let _ = self.tx.send(Event::Status(self.status()));
            }
        }
        // Done: the status says so now, not at the next one (it held
        // "reading 148/148" until then).
        if self.prefill.take().is_some() {
            let _ = self.tx.send(Event::Status(self.status()));
        }
        Ok(row)
    }

    /// Decode `tokens` into the live sequence after the pending token,
    /// logits of the last, and sample the next.
    fn direct(&mut self, tokens: &[i32]) -> Result<()> {
        let mut all = vec![self.next];
        all.extend_from_slice(tokens);
        let pos0 = self.pos();
        let row = self.feed_live(&all, pos0)?;
        self.history.extend_from_slice(&all);
        self.mind_step(pos0 + all.len() as i32 - 1, *all.last().unwrap())?;
        self.next = self.llm.sample(row);
        Ok(())
    }

    /// The live sequence's next-token logits had `placebo` been put in now
    /// instead: a copy decodes the pending token and it (`weigh`); none when
    /// no sequence is free.
    fn logits_without(&mut self, placebo: &[i32]) -> Result<Option<Vec<f32>>> {
        let Some(w) = self.free_seqs.pop() else {
            return Ok(None);
        };
        self.llm.seq_rm(w, -1, -1);
        self.llm.seq_cp(self.live, w, -1, -1);
        let mut tokens = vec![self.next];
        tokens.extend_from_slice(placebo);
        let rows = self.llm.decode(&[Lane {
            seq: w,
            tokens: &tokens,
            pos0: self.pos(),
            logits: true,
        }])?;
        // Not the live row: nobody reads the capture.
        if let Some(cap) = self.llm.capture() {
            cap.take();
        }
        let out = self.llm.logits(rows[0])?.to_vec();
        self.llm.seq_rm(w, -1, -1);
        self.free_seqs.push(w);
        Ok(Some(out))
    }

    /// `direct`, and the live sequence's next-token logits after it.
    fn direct_logits(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        let mut all = vec![self.next];
        all.extend_from_slice(tokens);
        let pos0 = self.pos();
        let row = self.feed_live(&all, pos0)?;
        self.history.extend_from_slice(&all);
        self.mind_step(pos0 + all.len() as i32 - 1, *all.last().unwrap())?;
        let out = self.llm.logits(row)?.to_vec();
        self.next = self.llm.sample(row);
        Ok(out)
    }

    /// Something from outside, framed, decoded straight in.
    fn put(&mut self, framed: String) -> Result<()> {
        let tokens = self.tok(&framed, false)?;
        self.direct(&tokens)?;
        self.say(framed, Kind::Given);
        Ok(())
    }

    /// A line from the system to it: in the agent frame it waits for its
    /// next user turn (`take_waiting`); otherwise it is put in now.
    fn tell(&mut self, text: &str) -> Result<()> {
        if self.cfg.agent {
            self.waiting
                .push(format!("[{}] {text}", clock::hms(clock::now_us())));
            return Ok(());
        }
        let msg = self.framed_system(text);
        self.put(msg)
    }

    /// The agent frame: what waits for its next user turn, taken: the lines
    /// from the system, then what was said or handed over, within the room
    /// its context has (`read_room`); one that does not fit is refused with
    /// its size (its own proposal: a feed past the room filled the context).
    fn take_waiting(&mut self) -> Vec<String> {
        let mut out = std::mem::take(&mut self.waiting);
        if let Some(line) = self.head_news() {
            out.insert(0, format!("[{}] {line}", clock::hms(clock::now_us())));
        }
        out.append(&mut self.asides);
        let mut room = self.read_room();
        while let Some((text, label)) = self.queue.pop_front() {
            let n = self.tok(&text, false).map_or(0, |t| t.len());
            if n > room {
                self.note(format!("refused {label}: {n} tokens, room {room}"));
                out.push(format!(
                    "[{}] {label} was not brought in: {n} tokens, more than the {room} there is room for now",
                    clock::hms(clock::now_us())
                ));
            } else {
                room -= n;
                out.push(text.trim().to_string());
            }
        }
        out
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
        } else if let Some((done, total)) = self.prefill {
            Mode::Reading { done, total }
        } else if self.rest.is_some() {
            Mode::Resting
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
            // Nothing is decoded at rest: its last rate is not its rate.
            stream_tps: if self.rest.is_some() {
                0.0
            } else {
                self.stream_rate.v
            },
            side_tps: if self.rest.is_some() {
                0.0
            } else {
                self.side_rate.v
            },
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
        // The live sequence is replaced: a reflection on the old one goes.
        self.end_chain(false);
        // The agent frame: the composed base ends by opening a turn.
        self.turn_start = c.head.len();
        let old = self.live;
        let mut history = c.head;
        history.extend_from_slice(&self.history[c.from..]);
        history.push(self.next);
        // The guide held the old live sequence's tokens.
        self.drop_guide();
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
            // Back in place before the live token advances: `consider` must
            // see the chase, or a check starts on a sequence being replaced.
            self.chase = Some(c);
            self.advance(rows[0])?;
            live_advanced = true;
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
            let row = if r.fed == 0 {
                self.llm.decode(&[Lane {
                    seq: r.seq,
                    tokens: &chunk,
                    pos0: rpos,
                    logits: false,
                }])?;
                None
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
                Some(rows[0])
            };
            r.fed += n;
            side_tokens += n;
            // The reading back in place (or its composition begun) before
            // the live token advances: `consider` must see it. A check that
            // started while the reading was out of `self` took the last two
            // sequences, and the composition found none (the service
            // stopped on "no free sequence for the composition").
            if r.fed == r.tokens.len() {
                self.finish_reading(r)?;
            } else {
                self.reading = Some(r);
            }
            if let Some(row) = row {
                self.advance(row)?;
                live_advanced = true;
            }
            self.finish_cycle(t0, side_tokens, live_advanced);
            return Ok(());
        }

        // The second chain beside the live token (`Chain`): its opening's
        // first token alone (the copy shares the live sequence's recurrent
        // state until it writes its own, as a check's deliberation does),
        // then the rest of it and each token of its own in the live token's
        // batch.
        if let Some(mut c) = self.reflecting.take() {
            if c.fed == 0 {
                self.llm.decode(&[Lane {
                    seq: c.seq,
                    tokens: &c.prompt[..1],
                    pos0: c.pos,
                    logits: false,
                }])?;
                // No live row in it: nobody reads the capture.
                if let Some(cap) = self.llm.capture() {
                    cap.take();
                }
                c.fed = 1;
                c.pos += 1;
                self.reflecting = Some(c);
                self.finish_cycle(t0, 1, false);
                return Ok(());
            }
            let lane: Vec<i32> = if c.fed < c.prompt.len() {
                c.prompt[c.fed..].to_vec()
            } else {
                vec![*c.out.last().unwrap()]
            };
            // The guide lane in the same batch, in the room left.
            let room = self.llm.batch_cap().saturating_sub(1 + lane.len());
            let g = self.guide_take(room)?;
            let gs = self.guide_seq.unwrap_or(-1);
            let pending = [self.next];
            let mut lanes = vec![
                Lane {
                    seq: self.live,
                    tokens: &pending,
                    pos0: self.pos(),
                    logits: true,
                },
                Lane {
                    seq: c.seq,
                    tokens: &lane,
                    pos0: c.pos,
                    logits: true,
                },
            ];
            if let Some((gl, gpos, aligned)) = &g {
                lanes.push(Lane {
                    seq: gs,
                    tokens: gl,
                    pos0: *gpos,
                    logits: *aligned,
                });
            }
            let rows = self.llm.decode(&lanes)?;
            drop(lanes);
            if let Some((gl, _, aligned)) = &g {
                self.guide_fed(gl.len(), *aligned, aligned.then(|| (rows[0], rows[2])))?;
            }
            c.pos += lane.len() as i32;
            c.fed = c.prompt.len();
            let t = self.chain_token(rows[1])?;
            let piece = self.llm.text(&[t]);
            c.out.push(t);
            let done = self.llm.is_eog(t)
                || c.out.len() >= CHAIN_MAX
                || (piece.contains('\n') && !self.llm.text(&c.out).trim().is_empty());
            let _ = self.tx.send(Event::Delib(crate::client::Delib {
                kind: crate::client::DelibKind::Piece,
                t_us: clock::now_us(),
                pos: c.pos,
                text: piece,
            }));
            // Back in place before the live token advances (`consider` and
            // the sequences' count must see it).
            self.reflecting = Some(c);
            if done {
                self.end_chain(true);
            }
            self.advance(rows[0])?;
            self.finish_cycle(t0, lane.len(), true);
            return Ok(());
        }

        // Nothing beside: the live token alone, and the guide lane when it
        // runs (`Guide`).
        let room = self.llm.batch_cap().saturating_sub(1);
        let g = self.guide_take(room)?;
        let gs = self.guide_seq.unwrap_or(-1);
        let pending = [self.next];
        let mut lanes = vec![Lane {
            seq: self.live,
            tokens: &pending,
            pos0: self.pos(),
            logits: true,
        }];
        if let Some((gl, gpos, aligned)) = &g {
            lanes.push(Lane {
                seq: gs,
                tokens: gl,
                pos0: *gpos,
                logits: *aligned,
            });
        }
        let rows = self.llm.decode(&lanes)?;
        drop(lanes);
        let side = g.as_ref().map_or(0, |x| x.0.len());
        if let Some((gl, _, aligned)) = &g {
            self.guide_fed(gl.len(), *aligned, aligned.then(|| (rows[0], rows[1])))?;
        }
        self.advance(rows[0])?;
        self.finish_cycle(t0, side, true);
        Ok(())
    }

    /// The pending token is decoded: keep it, choose the next (the
    /// sampler's choice, or the next token of a changed answer), consider
    /// checking it, show it.
    fn advance(&mut self, row: i32) -> Result<()> {
        self.mind_step(self.pos(), self.next)?;
        // A line ends with this token: the second chain's moment (`after`),
        // and the lens aside's.
        if self.llm.text(&[self.next]).contains('\n') {
            self.line_ended |= self.chain_on;
            self.lens_line_ended = true;
        }
        self.history.push(self.next);
        let forced = self.forced.pop_front();
        // The guide's mixed row, made this cycle for this choice (`guide_measure`).
        let mixed = self.mixed.take();
        let mut t = match (forced, mixed) {
            (Some(f), _) => {
                self.llm.accept(f);
                f
            }
            (None, Some(m)) => self.llm.sample_logits(&m),
            (None, None) => self.llm.sample(row),
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
        // Never inside its code (a check replaced a piece of a token with a
        // word and broke a #define in the dev session).
        // In the agent frame only inside its thinking: a word changed in a
        // tool call would change its code or its command.
        let free = !self.in_code
            && !(self.cfg.agent && self.speaking)
            && self.check.is_none()
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
        let question = if self.cfg.agent {
            self.tok(&reflect::question_agent(shown_at, &text, &shown), true)?
        } else {
            self.tok(
                &reflect::question(self.cfg.frame, shown_at, &text, &shown),
                false,
            )?
        };
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
            rule: String::new(),
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
            in_code: self.in_code,
            fence_tail: self.fence_tail.clone(),
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
        self.in_code = s.in_code;
        self.fence_tail = s.fence_tail;
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
        let mut defer_tok = self.newline;
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
                // A quote before the word (it answers `"keep"`): on the live
                // service at 200K the likeliest token at `Decision:` was a
                // quote, 37 to 56 percent, and keep and write together under
                // a fifth: 30 percent of the checks went unread. The format
                // token it wants (a newline or a quote) is fed once, and the
                // choice read after it.
                let pq = mass(&self.quotes);
                let quote = self
                    .quotes
                    .iter()
                    .copied()
                    .max_by(|a, b| l[*a as usize].total_cmp(&l[*b as usize]));
                let (pf, ft) = match quote {
                    Some(q) if pq > pn => (pq, q),
                    _ => (pn, self.newline),
                };
                if pf > pk + pw && !newline_fed {
                    defer = true;
                    defer_tok = ft;
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
        let keep_at = self
            .reflector
            .as_ref()
            .map_or(ReflectConfig::default().keep_at, |r| r.cfg.keep_at);
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
            // The format token it wanted (a newline or a quote), then the
            // choice after it.
            c.newline_fed = true;
            c.choosing = true;
            c.lane_next = vec![defer_tok];
        }
        if let Some((keep, fmt, top)) = choice {
            c.choosing = false;
            c.keep = keep;
            c.fmt = fmt;
            c.top = top;
            if fmt < min_fmt {
                c.rule = format!("fmt {fmt:.2} < {min_fmt:.2}: no answer");
                c.unanswered = true;
                ended = true;
            } else if keep >= keep_at {
                c.rule = format!("keep {keep:.2} >= {keep_at:.2}: kept");
                ended = true;
            } else {
                c.rule = format!("keep {keep:.2} < {keep_at:.2}: write");
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
                self.drop_guide();
                self.history.truncate(c.at);
                self.restore(c.saved);
                // A word written over: a reflection forked before it goes.
                self.end_chain(false);
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
            rule: c.rule,
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
            f.line(&reflect::line(&e));
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
    fn read_request(&mut self, id: u64, spec: &str) -> Result<()> {
        // A path written URL-style (`Intel%20Phi%20Stream`) is the path it
        // means: on the live service it read one so and concluded the file
        // did not exist.
        let spec = percent_decoded(spec);
        let (path, range) = read_range(&spec);
        // In development, paths are the repository's (`docs/dev.md`); one
        // the repository lacks is looked for in the workspace next, where
        // its own records are (it asked for reflect.log and was told the
        // repository's listing).
        let p = match &self.cfg.dev {
            Some(root) => dev_path(path, root, &self.cfg.workspace),
            None => resolve(path, &self.cfg.workspace),
        };
        // And they stay in it (closed by default): its reads are logged and
        // shown, so nothing outside the repository and its workspace (a key,
        // a private file) is brought in. `..` and symbolic links are resolved
        // before the test, so no spelling escapes it.
        if let Some(root) = &self.cfg.dev {
            if !inside(&p, &[root.as_path(), self.cfg.workspace.as_path()]) {
                // Where it may read, and the read it likely meant: the longest
                // tail of the path that exists in the repository.
                let comps: Vec<_> = p.components().collect();
                let meant = (1..=comps.len().min(4)).rev().find_map(|k| {
                    let tail: PathBuf = comps[comps.len() - k..].iter().collect();
                    root.join(&tail)
                        .is_file()
                        .then(|| tail.display().to_string())
                });
                let msg = match meant {
                    Some(t) => format!(
                        "outside the repository ({}) and your workspace, all a read may reach while developing; it has {t}: [read: {t}]",
                        root.display()
                    ),
                    None => format!(
                        "outside the repository ({}) and your workspace, all a read may reach while developing",
                        root.display()
                    ),
                };
                return self.read_failed(id, &p, &msg);
            }
        }
        let outcome = fs::metadata(&p)
            .map_err(|e| e.to_string())
            .and_then(|m| {
                if m.is_dir() {
                    // A directory reads as its listing: how it learns the tree.
                    let mut names: Vec<String> = fs::read_dir(&p)
                        .map_err(|e| e.to_string())?
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
                    Ok(names.join("\n"))
                } else if !m.is_file() {
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
                return self.read_failed(id, &p, &e);
            }
        };
        let what = match range {
            Some((a, b)) => format!(
                "lines {a} to {} of {} are brought in",
                b.min(a - 1 + text.lines().count()),
                p.display()
            ),
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
            // In development the path as the repository names it: short, and
            // nothing to mistype (a long absolute path came back mangled).
            let short = match &self.cfg.dev {
                Some(root) => p.strip_prefix(root).unwrap_or(&p).display().to_string(),
                None => path.to_string(),
            };
            let msg = format!(
                "{} is {n} tokens in {lines} lines and there is room for about {room} now: read it by lines, [read: {short}:{first}-{}]",
                p.display(),
                first + fit - 1
            );
            self.note(format!(
                "{} is too big to read now ({n} tokens, room {room})",
                p.display()
            ));
            self.act_end(
                id,
                false,
                format!("too big to read now: {n} tokens, room {room}; read it by lines"),
            );
            return self.tell(&msg);
        }
        self.queue
            .push_back((framed, format!("read {}", p.display())));
        self.note(format!("reading {} for it ({n} tokens)", p.display()));
        self.act_end(id, true, format!("{what}, {n} tokens"));
        Ok(())
    }

    /// A read that could not happen. Each failing path is told to it once
    /// (with what is there instead), so it learns; the same path asked again
    /// within five minutes is dropped quietly, so a guess repeated is not a
    /// line in the chain each time. (A limit of one failure line a minute,
    /// whatever the path, hid most failures and their listings: in the dev
    /// session it asked for the same missing log.txt every few seconds.)
    fn read_failed(&mut self, id: u64, p: &Path, e: &str) -> Result<()> {
        self.act_end(id, false, e.to_string());
        let mono = clock::mono_us();
        let key = p.display().to_string();
        self.failed_reads.retain(|_, t| mono - *t < 300_000_000);
        if self.failed_reads.contains_key(&key) {
            self.read_failures_quiet += 1;
            self.note(format!(
                "{key} asked again: still could not be read: {e} (not put in again)"
            ));
            return Ok(());
        }
        self.failed_reads.insert(key.clone(), mono);
        self.last_read_failure_mono = mono;
        self.tell(&format!("{key} could not be read: {e}"))
    }

    /// After a cycle: the summary's end, the turn's end, the mind's own
    /// requests, circling, the queue, the rollover.
    fn after(&mut self) -> Result<()> {
        if self.cfg.task {
            return self.after_task();
        }
        // Its terminal: commands that ended come back as documents.
        self.poll_term();
        self.poll_improve();
        // Tokens held back after a repeated line come back when their time is up.
        let mono_now = clock::mono_us();
        if self.held_back.iter().any(|(_, u)| *u <= mono_now) {
            self.held_back.retain(|(_, u)| *u > mono_now);
            self.apply_gate();
        }
        // A line ended: the second chain's moment, and the lens aside's.
        if std::mem::take(&mut self.line_ended) {
            self.on_line_end()?;
        }
        if std::mem::take(&mut self.lens_line_ended) {
            self.lens_aside();
        }
        // The summary being written: collect until its closing line.
        if let Some(s) = &mut self.summary {
            s.push(self.next);
            // Its end mark, or the end of its turn, counts only once the summary
            // has some length: an early --- is not a summary.
            let ended = self.next == self.eot
                || self.llm.is_eog(self.next)
                || self.next == self.think_close;
            let done = s.len() >= self.cfg.summary_max
                || (s.len() >= SUMMARY_MIN && {
                    let tail = self.llm.text(&s[s.len().saturating_sub(6)..]);
                    tail.contains("\n---") || ended
                });
            if done {
                let mut s = self.summary.take().unwrap();
                // The token that ended it is no part of it: rendered, an
                // `<|im_end|>` would have the summary refused (`degenerate`).
                while s
                    .last()
                    .is_some_and(|&t| t == self.eot || t == self.think_close || self.llm.is_eog(t))
                {
                    s.pop();
                }
                let mut text = self.llm.text(&s);
                if let Some(i) = text.rfind("\n---") {
                    text.truncate(i);
                }
                // A summary written by a degenerating stream is refused, and the
                // last good one kept: on the live service one full of the
                // delimiters and repeated lines was resumed from, and every
                // restart carried the degeneration on (`degenerate`).
                let stamp = clock::datetime(clock::now_us()).replace([' ', ':'], "-");
                let dir = self.cfg.workspace.join("summaries");
                let _ = fs::create_dir_all(&dir);
                if let Some(why) = degenerate(&text) {
                    let _ = fs::write(dir.join(format!("{stamp}-refused.md")), text.trim());
                    self.note(format!(
                        "the summary was refused ({why}); the last good one stays"
                    ));
                    text = fs::read_to_string(self.cfg.workspace.join("summary.md"))
                        .unwrap_or_default();
                } else {
                    // Kept on disk: a restart resumes from it, as a rollover
                    // does; and every one kept by its time.
                    let _ = fs::write(self.cfg.workspace.join("summary.md"), text.trim());
                    let _ = fs::write(dir.join(format!("{stamp}.md")), text.trim());
                }
                // Asked to quit: the summary was its last act.
                if self.quit_deadline.is_some() {
                    self.note("the summary is kept; stopping".into());
                    self.stop_now = true;
                    return Ok(());
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
            // The agent frame: its calls run, and their results come back
            // before its next turn (`agent_turn_end`).
            if self.cfg.agent {
                return self.agent_turn_end();
            }
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
            for (id, p) in paths {
                self.read_request(id, &p)?;
            }
        }

        // Rollover: past the share, or a new persona waiting, with nothing
        // in flight: ask for the summary.
        let limit = self
            .cfg
            .rollover_tokens
            .unwrap_or((self.llm.n_ctx() as f32 * self.cfg.rollover_at) as usize);
        // A quit asks for the summary too: the restart resumes from it.
        let quitting = self.quit_deadline.is_some() && !self.cfg.task;
        if idle && (self.history.len() >= limit || self.reseat || quitting) {
            // Reads asked for before the rollover would fill the new context
            // with what the summary already carries: dropped; it asks again.
            let queued = self.queue.len();
            self.queue.retain(|(_, label)| !label.starts_with("read "));
            let dropped = self.pending_reads.len() + queued - self.queue.len();
            self.pending_reads.clear();
            if dropped > 0 {
                self.note(format!("{dropped} pending reads dropped at the rollover"));
            }
            let why = if quitting {
                Summary::Restart
            } else if self.reseat {
                Summary::Persona
            } else {
                Summary::Full
            };
            // The agent frame: the summary is asked in its own user turn, at
            // the end of the turn it is in (`open_summary`); asked inside a
            // turn, it called tools and the summary kept the template's marks.
            if self.cfg.agent {
                if self.summary_due.is_none() {
                    self.summary_due = Some(why);
                    self.note("a summary is due at the end of its turn".into());
                }
                return self.agent_stalled();
            }
            let ask = self.summary_ask(why);
            self.put(ask)?;
            self.summary = Some(Vec::new());
            return Ok(());
        }

        // The agent frame: nothing is put inside its turns; what waits for
        // it comes in its next user turn, and a turn too long is closed.
        if self.cfg.agent {
            return self.agent_stalled();
        }

        // The clock: after a stretch with nothing from outside, the time is
        // put into the chain, so the thoughts stand on the wall clock.
        let mono = clock::mono_us();
        let now = clock::now_us();
        if idle
            && !self.in_code
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

        // No tool used for a while: it is reminded to check something real
        // (what it has not checked with a tool it does not know), at a line's
        // end, at most once per `TOOL_IDLE_US`.
        if idle
            && self.nudges_on
            && !self.in_code
            && self.line_start
            && self.summary.is_none()
            && mono - self.last_tool_mono >= TOOL_IDLE_US
            && mono - self.last_tool_nudge_mono >= TOOL_IDLE_US
        {
            self.last_tool_nudge_mono = mono;
            let secs = (mono - self.last_tool_mono) / 1_000_000;
            let msg = self.framed_system(&format!(
                "no tool used for {secs} s: what you have not checked with a tool you do not know; check one real thing now, with [run: COMMAND] or [read: PATH], and read what it returns"
            ));
            self.put(msg)?;
            self.note(format!("no tool for {secs} s; reminded"));
            return Ok(());
        }

        // Thoughts going round: a nudge, at most once per `nudge_every_us`.
        if idle
            && self.nudges_on
            && !self.in_code
            && mono - self.last_nudge_mono >= self.cfg.nudge_every_us
            && self.circling()
        {
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
                // Within the room the context has, whatever brought it (the
                // stream's own proposal: a feed past `direct_max` went to a
                // reading with no check, and a big one could fill the context).
                let room = self.read_room();
                if tokens.len() > room {
                    self.note(format!(
                        "refused {label}: {} tokens, room {room}",
                        tokens.len()
                    ));
                    return self.tell(&format!(
                        "{label} was not brought in: {} tokens, more than the {room} there is room for now",
                        tokens.len()
                    ));
                }
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
            Command::Ask(id, s, t) => {
                self.last_outside_mono = clock::mono_us();
                self.asked.insert(id);
                let text = if self.cfg.agent {
                    format!(
                        "[{}] Claude (message c{id}): {}\n(Answer it with tell_claude, re c{id}.)",
                        clock::hms(t),
                        s.trim()
                    )
                } else {
                    self.framed_say(&format!("{} (message c{id})", s.trim()), t, Some("Claude"))
                };
                self.queue
                    .push_back((text, format!("Claude's message c{id}")));
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
            Command::Set(key, v) if key == "guide-mix" => {
                // The guide's weight in the choice of each thinking token (0:
                // shadow, measured only).
                self.guide_mix = v.clamp(0.0, 2.0);
                self.note(format!("guide mix {}", self.guide_mix));
            }
            Command::Set(key, v) => {
                let s = &mut self.cfg.sampling;
                match key.as_str() {
                    "temp" => s.temp = v,
                    "top-k" => s.top_k = v as i32,
                    "top-p" => s.top_p = v,
                    "min-p" => s.min_p = v,
                    "dry" => s.dry_multiplier = v,
                    "repeat-penalty" => s.repeat_penalty = v,
                    _ => {}
                }
                let s = self.cfg.sampling.clone();
                self.llm.reset_sampler(&s, s.seed, &self.history);
                self.note(format!("sampling: {key} {v}"));
            }
            Command::Guard(which, on) => {
                match which.as_str() {
                    "breaker" => {
                        self.breaker_on = on;
                        if !on {
                            self.held_back.clear();
                            self.apply_gate();
                        }
                    }
                    "nudges" => self.nudges_on = on,
                    "experts" => {
                        if let Some(c) = self.llm.capture() {
                            c.experts_on = on;
                        }
                    }
                    "guide" => {
                        self.guide_on = on && self.guide_seq.is_some();
                        if !self.guide_on {
                            self.drop_guide();
                        }
                    }
                    // Where its aside comes from (`GuideSrc`): the lane
                    // forks again from the next one, and the report counts
                    // afresh, so no window mixes two sources.
                    "chain" | "lens" | "placebo" | "ab" => {
                        // `ab`: lens and placebo by turns (`lens_aside`).
                        self.guide_ab = which == "ab";
                        self.ab_lens = true;
                        self.guide_src = GuideSrc::from_name(&which).unwrap_or(if self.guide_ab {
                            GuideSrc::Lens
                        } else {
                            GuideSrc::Chain
                        });
                        self.drop_guide();
                        self.guide_next = None;
                        self.lens_sums.clear();
                        self.lens_readings = 0;
                        self.lens_last.clear();
                        // The first aside of the new source is not held back by
                        // the last one's spacing (the dev stream's review, m29).
                        self.lens_fork_mono = 0;
                        self.line_from = self.history.len();
                        self.guide_n = 0;
                        self.guide_kl = 0.0;
                        self.guide_flips = 0;
                        self.note(format!("the guide's asides from: {which}"));
                    }
                    _ => {}
                }
                if GuideSrc::from_name(&which).is_none() && which != "ab" {
                    self.note(format!("{which} {}", if on { "on" } else { "off" }));
                }
            }
            Command::Temp(t) => {
                self.cfg.sampling.temp = t;
                let s = self.cfg.sampling.clone();
                self.llm.set_sampling(&s);
            }
            Command::Persona(text) => {
                self.cfg.system = with_ctx(
                    &compose(&text, self.cfg.frame, self.cfg.dev.as_deref()),
                    self.llm.n_ctx(),
                );
                self.reseat = true;
                let _ = fs::write(self.cfg.workspace.join("persona.md"), &self.cfg.system);
                self.note("a new persona: the context rolls over onto it after a summary".into());
            }
            Command::Status => {
                let _ = self.tx.send(Event::Status(self.status()));
            }
            Command::Objective(text) => self.set_objective(text.trim()),
            Command::Chain(set) => {
                let against = set == ChainSet::Against;
                // A change of kind drops what the other kind had in flight
                // or waiting, so no window mixes the two.
                if set == ChainSet::Off || against != self.chain_against {
                    self.end_chain(false);
                    self.line_words.clear();
                    self.reflection = None;
                }
                self.chain_on = set != ChainSet::Off && !self.cfg.task;
                self.chain_against = against;
                self.note(format!(
                    "the second chain {}",
                    match (self.chain_on, against) {
                        (false, _) => "off",
                        (true, false) => "on",
                        (true, true) => "against",
                    }
                ));
            }
            Command::Goal(on) => {
                self.goal_on = on && self.yes_no.is_some();
                self.note(format!(
                    "the goal probe {}",
                    if self.goal_on { "on" } else { "off" }
                ));
            }
            Command::Quit => {
                // Its context outlives the restart: the summary first (at
                // the next point with nothing in flight), then the stop; by
                // the deadline at the latest. A second quit stops at once.
                if !self.cfg.summary_on_quit || self.cfg.task || self.quit_deadline.is_some() {
                    return false;
                }
                self.quit_deadline = Some(clock::mono_us() + QUIT_WAIT_US);
                self.note("quitting: writing the summary first".into());
            }
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
        // An objective kept from before is sent to the terminals too: their
        // OBJECTIVE said "none sent by this service" while it worked on one.
        if let Some((t, text)) = self.objective.clone() {
            let _ = self.tx.send(Event::Objective(t, text));
        }
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
        // The agent frame: its first turn begins here.
        self.turn_start = self.history.len();
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
            // The agent frame waits for its tools' results before its next
            // turn: commands still come in, nothing is decoded meanwhile.
            if self.awaiting.is_some() {
                self.poll_term();
                self.poll_improve();
                self.finish_agent_wait()?;
                if self.awaiting.is_some() {
                    self.release();
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
            }
            // At rest (`wait`): nothing is decoded until something new comes.
            if self.rest.is_some() {
                self.rest_look()?;
                if self.rest.is_some() {
                    self.release();
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    continue;
                }
                let _ = self.tx.send(Event::Status(self.status()));
            }
            self.cycle()?;
            self.after()?;
            self.release();
            // Quitting: once the summary is kept, or past the deadline.
            if self.stop_now || self.quit_deadline.is_some_and(|d| clock::mono_us() > d) {
                self.abandon();
                self.release();
                let _ = self.tx.send(Event::Stopped);
                return Ok(self.finish());
            }
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

/// Two tokens' experts, block by block (the block and the experts it
/// routed to): the share they hold in common (shared over all, per block,
/// averaged over the blocks both have).
fn experts_shared(a: &[(i32, Vec<i32>)], b: &[(i32, Vec<i32>)]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for (layer, ea) in a {
        let Some((_, eb)) = b.iter().find(|(l, _)| l == layer) else {
            continue;
        };
        let common = ea.iter().filter(|e| eb.contains(e)).count();
        let all = ea.len() + eb.len() - common;
        if all > 0 {
            sum += common as f64 / all as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Two next-token distributions given as logits: KL of the first from the
/// second (nats), and each one's likeliest token.
fn kl_and_tops(p: &[f32], q: &[f32]) -> (f64, usize, usize) {
    let lse = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (zp, zq) = (lse(p), lse(q));
    let mut kl = 0.0;
    let (mut ap, mut aq) = (0usize, 0usize);
    for i in 0..p.len().min(q.len()) {
        let (a, b) = (p[i] as f64 - zp, q[i] as f64 - zq);
        kl += a.exp() * (a - b);
        if p[i] > p[ap] {
            ap = i;
        }
        if q[i] > q[aq] {
            aq = i;
        }
    }
    (kl.max(0.0), ap, aq)
}

/// A note without the mark of a check (` [unverified: ...]`, at its end).
fn unmarked(note: &str) -> &str {
    match note.find(" [unverified:") {
        Some(i) if note.trim_end().ends_with(']') => &note[..i],
        _ => note,
    }
}

/// Why a text is degenerate, if it is (`summary` refused): the person's
/// response delimiters in it, lines repeating earlier ones (a third or
/// more of six or more), or its word 4-grams repeating (two fifths or more
/// of forty or more words).
fn degenerate(text: &str) -> Option<&'static str> {
    if text.contains('•') {
        return Some("delimiter fragments");
    }
    // The chat template's own marks: the agent frame's summary asked inside
    // a turn came back as `<|im_start|>user` and two tool calls.
    if ["<|im_start|>", "<|im_end|>", "<tool_call>", "<think>"]
        .iter()
        .any(|m| text.contains(m))
    {
        return Some("the chat template's marks");
    }
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() >= 6 {
        let repeated = lines
            .iter()
            .enumerate()
            .filter(|(i, l)| lines[..*i].contains(l))
            .count();
        if repeated * 3 >= lines.len() {
            return Some("repeated lines");
        }
    }
    let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    if words.len() >= 40 {
        let grams: Vec<&[String]> = words.windows(4).collect();
        let mut seen = std::collections::HashSet::new();
        let repeated = grams.iter().filter(|g| !seen.insert(**g)).count();
        if repeated * 5 >= grams.len() * 2 {
            return Some("repeated phrases");
        }
    }
    None
}

/// A text's words, lowercased, sorted, once each.
fn word_set(s: &str) -> Vec<String> {
    let mut w: Vec<String> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    w.sort();
    w.dedup();
    w
}

/// The share of `a`'s words that `b` holds too (both word sets, sorted).
fn contained(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() {
        return 1.0;
    }
    a.iter().filter(|w| b.binary_search(w).is_ok()).count() as f64 / a.len() as f64
}

/// A repository's head commit (short), read from its files.
fn head_of(repo: &Path) -> Option<String> {
    let git = repo.join(".git");
    let head = fs::read_to_string(git.join("HEAD")).ok()?;
    let full = match head.trim().strip_prefix("ref: ") {
        Some(r) => match fs::read_to_string(git.join(r)) {
            Ok(s) => s.trim().to_string(),
            // A ref packed away: its line in packed-refs.
            Err(_) => fs::read_to_string(git.join("packed-refs"))
                .ok()?
                .lines()
                .find(|l| l.ends_with(r))?
                .split_whitespace()
                .next()?
                .to_string(),
        },
        None => head.trim().to_string(),
    };
    Some(full.chars().take(7).collect())
}

/// Why a result was cut: the most one gives, or, near a rollover, the room
/// left (a budget of 0 read as "more than the 0 a read gives").
fn past(budget: usize, room: usize) -> String {
    if budget < READ_MAX_TOKENS {
        format!("more than the {room} there is room for now")
    } else {
        format!("more than the {READ_MAX_TOKENS} one result gives")
    }
}

/// How many leading lines, of these token counts, fit in `budget` tokens.
fn lines_within(counts: &[usize], budget: usize) -> usize {
    let mut sum = 0;
    counts
        .iter()
        .take_while(|&&c| {
            sum += c;
            sum <= budget
        })
        .count()
}

/// The overlap of two word sets: shared over all.
fn jaccard(a: &[String], b: &[String]) -> f64 {
    let shared = a.iter().filter(|w| b.binary_search(w).is_ok()).count();
    let all = a.len() + b.len() - shared;
    if all == 0 {
        1.0
    } else {
        shared as f64 / all as f64
    }
}

/// What something put into the live chain weighs on it: the divergence
/// (nats) of its next-token distribution with it (`with`) from the one
/// without it (`without`), and the likeliest token of each.
pub fn weigh(with: &[f32], without: &[f32]) -> (f64, usize, usize) {
    let norm = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
        z
    };
    let (zp, zq) = (norm(with), norm(without));
    let mut kl = 0.0;
    for (&a, &b) in with.iter().zip(without) {
        let lp = a as f64 - zp;
        let lq = b as f64 - zq;
        let p = lp.exp();
        if p > 0.0 {
            kl += p * (lp - lq);
        }
    }
    let arg = |l: &[f32]| {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |e| e.0)
    };
    (kl.max(0.0), arg(with), arg(without))
}

/// The commits since the program last ran here (`last_run` in the
/// workspace) as a line from the system, and the commit it runs now kept
/// for the next start. Empty on a first start, when nothing changed, or
/// outside a git checkout.
fn changes_since(repo: &Path, ws: &Path, frame: Frame) -> String {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(head) = git(&["rev-parse", "HEAD"]) else {
        return String::new();
    };
    let file = ws.join("last_run");
    let last = fs::read_to_string(&file)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let _ = fs::write(&file, format!("{head}\n"));
    if last.is_empty() || last == head {
        return String::new();
    }
    let range = format!("{last}..{head}");
    let log = git(&["log", "--oneline", "--no-decorate", "-n", "30", &range]).unwrap_or_default();
    if log.is_empty() {
        return String::new();
    }
    let body = format!(
        "the program you run in was updated and restarted; the changes since you last ran, newest first:\n{log}"
    );
    match frame {
        Frame::Journal => format!("« [{body}]\n"),
        Frame::Chat => format!("\n[{body}]"),
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

/// `p` with `.` and `..` resolved in its text and symbolic links resolved
/// in the longest part of it that exists.
fn normalized(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    let mut existing = out.clone();
    let mut rest = Vec::new();
    while !existing.exists() {
        match existing.file_name() {
            Some(n) => {
                rest.push(n.to_os_string());
                existing.pop();
            }
            None => break,
        }
    }
    let mut base = fs::canonicalize(&existing).unwrap_or(existing);
    for n in rest.into_iter().rev() {
        base.push(n);
    }
    base
}

/// Whether `p` lies under one of `roots`, both normalized.
fn inside(p: &Path, roots: &[&Path]) -> bool {
    let p = normalized(p);
    roots.iter().any(|r| p.starts_with(normalized(r)))
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

/// One token's piece against the fence state: whether it opens or closes
/// a ``` block (an odd number of fences in the last two characters before
/// it and itself), and the new last two characters, emptied after a fence
/// so it is not counted again with the next piece.
fn fence_step(tail: &str, piece: &str) -> (bool, String) {
    let text = format!("{tail}{piece}");
    let n = text.matches("```").count();
    let tail = if text.ends_with("```") {
        String::new()
    } else {
        let k = text.chars().count();
        text.chars().skip(k.saturating_sub(2)).collect()
    };
    (n % 2 == 1, tail)
}
/// `%XX` sequences decoded (a path written URL-style, `Intel%20Phi`);
/// anything else, and a `%` not followed by two hex digits, kept.
fn percent_decoded(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `PATH:START-END` as the path and the lines (1-based, inclusive); a
/// path with no such suffix whole.
fn read_range(spec: &str) -> (&str, Option<(usize, usize)>) {
    if let Some((path, r)) = spec.rsplit_once(':') {
        if let Some((a, b)) = r.split_once('-') {
            // `end` for the last line (the dev stream wrote `:41-end`).
            let b = match b.trim() {
                e if e.eq_ignore_ascii_case("end") => Ok(usize::MAX),
                n => n.parse(),
            };
            if let (Ok(a), Ok(b)) = (a.trim().parse(), b) {
                return (path.trim(), Some((a, b)));
            }
        }
    }
    (spec, None)
}

/// A path the stream wrote while developing: the repository's, or the
/// workspace's when only the workspace holds it.
fn dev_path(path: &str, repo: &Path, workspace: &Path) -> PathBuf {
    let p = resolve(path, repo);
    if p.exists() {
        return p;
    }
    let w = resolve(path, workspace);
    if w.exists() {
        w
    } else {
        p
    }
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
    fn the_agent_persona_teaches_the_tools_not_the_bracketed_lines() {
        let root = Path::new("/r/Intel Phi Stream");
        let chat = with_ctx(&compose("base", Frame::Chat, Some(root)), 204800);
        assert!(chat.contains("[read: PATH]"));
        let agent = agent_persona(&chat, Some(root), 204800);
        assert!(!agent.contains("[read: PATH]"));
        assert!(!agent.contains("[prefer:"));
        assert!(!agent.contains("[unnote:"));
        assert!(agent.contains("change files with edit"));
        assert!(agent.contains("/r/Intel Phi Stream"));
        assert!(agent.contains("about 205 thousand tokens"));
        assert!(agent.contains(AGENT_MECHANICS));
    }

    #[test]
    fn the_guide_measures_kl_and_the_likeliest_tokens() {
        let same = [1.0f32, 2.0, 3.0];
        let (kl, a, b) = kl_and_tops(&same, &same);
        assert!(kl.abs() < 1e-12);
        assert_eq!((a, b), (2, 2));
        // Shifted logits are the same distribution.
        let shifted = [11.0f32, 12.0, 13.0];
        assert!(kl_and_tops(&same, &shifted).0 < 1e-9);
        // Two tokens: p = (0.5, 0.5) against q = (0.9, 0.1).
        let p = [0.0f32, 0.0];
        let q = [(0.9f32).ln(), (0.1f32).ln()];
        let (kl, a, b) = kl_and_tops(&p, &q);
        let want = 0.5 * (0.5f64 / 0.9).ln() + 0.5 * (0.5f64 / 0.1).ln();
        assert!((kl - want).abs() < 1e-6, "{kl} vs {want}");
        assert_eq!((a, b), (0, 0));
        let (_, a, b) = kl_and_tops(&[0.0, 1.0], &[1.0, 0.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn the_experts_shared_are_counted_block_by_block() {
        let a = vec![(27, vec![1, 2, 3, 4]), (28, vec![5, 6])];
        assert!((experts_shared(&a, &a) - 1.0).abs() < 1e-12);
        // Block 27: {1,2,3,4} and {3,4,7,8} share 2 of 6; block 28 the same.
        let b = vec![(27, vec![3, 4, 7, 8]), (28, vec![5, 6]), (29, vec![9])];
        let want = (2.0 / 6.0 + 1.0) / 2.0;
        assert!((experts_shared(&a, &b) - want).abs() < 1e-12);
        assert_eq!(experts_shared(&a, &[]), 0.0);
    }

    #[test]
    fn a_restatement_is_contained_in_the_message_before() {
        let first = word_set("Reviewed c79ab72: guide_take, guide_fed and kl_and_tops are correct; no defects found in the lane indexing.");
        let again = word_set(
            "Final review of c79ab72: guide_take and guide_fed correct, no defects found.",
        );
        let new = word_set("The summary turn drops the calls of the turn it closes: engine.rs agent_stalled, line 2101.");
        assert!(contained(&again, &first) >= TO_CLAUDE_SAME);
        assert!(contained(&new, &first) < TO_CLAUDE_SAME);
    }

    #[test]
    fn a_read_gives_the_leading_lines_that_fit() {
        assert_eq!(lines_within(&[10, 10, 10], 30), 3);
        assert_eq!(lines_within(&[10, 10, 10], 29), 2);
        assert_eq!(lines_within(&[10, 10, 10], 10), 1);
        assert_eq!(lines_within(&[40, 1], 30), 0);
        assert_eq!(lines_within(&[], 30), 0);
    }

    #[test]
    fn a_note_is_unmarked_only_at_its_end() {
        assert_eq!(unmarked("x [unverified: a; b]"), "x");
        assert_eq!(unmarked("x"), "x");
        assert_eq!(
            unmarked("x [unverified: a] and more"),
            "x [unverified: a] and more"
        );
    }

    #[test]
    fn a_summary_with_the_templates_marks_is_refused() {
        assert!(degenerate("<|im_start|>user\nhi").is_some());
        assert!(degenerate("a summary\n<tool_call>").is_some());
        assert!(degenerate("What I was working on: the guide lane, measured.").is_none());
    }

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
    fn the_head_is_read_loose_or_packed() {
        let dir = std::env::temp_dir().join(format!("phi-stream-head-{}", std::process::id()));
        let git = dir.join(".git");
        fs::create_dir_all(git.join("refs/heads")).unwrap();
        fs::write(git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(
            git.join("packed-refs"),
            "# pack-refs\n9bea30f0000000000000000000000000000000000 refs/heads/main\n",
        )
        .unwrap();
        assert_eq!(head_of(&dir).as_deref(), Some("9bea30f"));
        fs::write(
            git.join("refs/heads/main"),
            "564ad0b5088318ce54835dc7581845e911efdd53\n",
        )
        .unwrap();
        assert_eq!(head_of(&dir).as_deref(), Some("564ad0b"));
        fs::write(
            git.join("HEAD"),
            "1111111222222233333334444444555555566666\n",
        )
        .unwrap();
        assert_eq!(head_of(&dir).as_deref(), Some("1111111"));
        assert_eq!(head_of(&dir.join("none")), None);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn development_reads_stay_inside() {
        let dir = std::env::temp_dir().join(format!("phi-stream-inside-{}", std::process::id()));
        let repo = dir.join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("src/a.rs"), "").unwrap();
        let ws = dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        let roots = [repo.as_path(), ws.as_path()];
        assert!(inside(&repo.join("src/a.rs"), &roots));
        assert!(inside(&repo.join("src/not-yet.rs"), &roots));
        assert!(inside(&ws.join("reflect.log"), &roots));
        assert!(!inside(&repo.join("src/../../outside"), &roots));
        assert!(!inside(Path::new("/etc/passwd"), &roots));
        // A symbolic link out of the repository does not carry a read out.
        std::os::unix::fs::symlink("/etc", repo.join("etc")).unwrap();
        assert!(!inside(&repo.join("etc/passwd"), &roots));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn code_fences_are_seen_across_tokens() {
        // One piece, the fence whole.
        let (t, tail) = fence_step("", "```rust\n");
        assert!(t);
        assert_eq!(tail, "t\n");
        // Split across pieces: `` then `.
        let (t1, tail) = fence_step("", "x ``");
        assert!(!t1);
        let (t2, _) = fence_step(&tail, "`\n");
        assert!(t2);
        // A fence ending a piece is not counted again with the next one.
        let (t3, tail) = fence_step("", "```");
        assert!(t3);
        let (t4, _) = fence_step(&tail, "`x");
        assert!(!t4);
        // Two fences in one piece: open and close.
        assert!(!fence_step("", "```a```").0);
    }

    #[test]
    fn reads_take_a_line_range() {
        assert_eq!(read_range("src/engine.rs"), ("src/engine.rs", None));
        assert_eq!(
            read_range("src/engine.rs:120-200"),
            ("src/engine.rs", Some((120, 200)))
        );
        assert_eq!(read_range("a b/c.rs: 3 - 4"), ("a b/c.rs", Some((3, 4))));
        assert_eq!(read_range("t.c:41-end"), ("t.c", Some((41, usize::MAX))));
        // A colon that is no range stays part of the path.
        assert_eq!(read_range("notes:draft.md"), ("notes:draft.md", None));
    }

    #[test]
    fn development_follows_the_person_s_instructions() {
        let base = "Be concise.\nNo em dash.";
        let plain = compose(base, Frame::Journal, None);
        assert!(!plain.contains("[prefer:"));
        let dev = compose(base, Frame::Journal, Some(Path::new("/r/Intel Phi Stream")));
        // Its memory's size is the context's, filled in at run time.
        assert!(dev.contains("about {ctx} thousand tokens"));
        assert!(with_ctx(&dev, 204800).contains("about 205 thousand tokens"));
        assert!(with_ctx(&dev, 32768).contains("about 33 thousand tokens"));
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
    fn a_degenerate_summary_is_known() {
        // Excerpts of the live service's summaries, 2026-10-01.
        let bad =
            "<•••>The engine.rs file is 3808 lines long.\n<\n<\n<•••>[run: cat src/engine.rs]\n";
        assert_eq!(degenerate(bad), Some("delimiter fragments"));
        let lines =
            "This is the running\nThis is the running\nThis is the running\nI\nI\nneed\nneed\n";
        assert_eq!(degenerate(lines), Some("repeated lines"));
        let phrases = "the journal is looping the journal is looping ".repeat(8);
        assert_eq!(degenerate(&phrases), Some("repeated phrases"));
        let good = "Tasks blocked: parallel reflection thread integration into engine.rs check_cycle(); the architecture is documented in diff-second-chain.txt. Retention filter analysis complete: no spike at 0.55, about 1.3 percent error detection and 6 percent threshold rejection; keep and top1p are separate metrics, so keep is not top1p clamped. Next: read the rest of engine.rs in parts and propose one checked improvement.";
        assert_eq!(degenerate(good), None);
    }

    #[test]
    fn url_style_paths_are_decoded() {
        assert_eq!(
            percent_decoded("/home/u/Intel%20Phi%20Stream/src/engine.rs:1-20"),
            "/home/u/Intel Phi Stream/src/engine.rs:1-20"
        );
        assert_eq!(percent_decoded("100%"), "100%");
        assert_eq!(percent_decoded("a%2"), "a%2");
        assert_eq!(percent_decoded("a%zzb"), "a%zzb");
    }

    #[test]
    fn weights_and_overlaps() {
        // The same distribution weighs nothing; a moved one weighs more.
        let a = [0.0f32, 1.0, 2.0, 3.0];
        let (kl, x, y) = weigh(&a, &a);
        assert!(kl.abs() < 1e-12 && x == 3 && y == 3);
        let b = [3.0f32, 2.0, 1.0, 0.0];
        let (kl, x, y) = weigh(&a, &b);
        assert!(kl > 1.0 && x == 3 && y == 0, "{kl}");
        // Two rewordings of one reflection overlap past the bar; two
        // different ones do not.
        let r1 = word_set(
            "the architecture document explicitly says the insertion point is check_cycle",
        );
        let r2 = word_set("the architecture says the insertion point is check_cycle");
        let r3 = word_set("repeating myself comes from having no new input");
        assert!(jaccard(&r1, &r2) >= CHAIN_SAME);
        assert!(jaccard(&r1, &r3) < CHAIN_SAME);
    }

    #[test]
    fn a_restart_is_told_what_changed() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("changes-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let (repo, ws) = (dir.join("repo"), dir.join("ws"));
        fs::create_dir_all(&repo).unwrap();
        fs::create_dir_all(&ws).unwrap();
        let git = |args: &[&str]| {
            let ok = std::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .output()
                .unwrap()
                .status
                .success();
            assert!(ok, "git {args:?}");
        };
        git(&["init", "-q"]);
        git(&["commit", "-q", "--allow-empty", "-m", "initial"]);
        // A first start: nothing to tell, the commit kept.
        assert_eq!(changes_since(&repo, &ws, Frame::Journal), "");
        git(&["commit", "-q", "--allow-empty", "-m", "the read fallback"]);
        let told = changes_since(&repo, &ws, Frame::Journal);
        assert!(
            told.starts_with("« [the program you run in was updated"),
            "{told}"
        );
        assert!(
            told.contains("the read fallback") && !told.contains("initial"),
            "{told}"
        );
        // Told once: the next start is the same commit.
        assert_eq!(changes_since(&repo, &ws, Frame::Journal), "");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dev_reads_fall_back_to_the_workspace() {
        let dir = std::env::temp_dir().join(format!("phi-stream-devpath-{}", std::process::id()));
        let (repo, ws) = (dir.join("repo"), dir.join("ws"));
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::create_dir_all(&ws).unwrap();
        fs::write(repo.join("src/a.rs"), "").unwrap();
        fs::write(ws.join("reflect.log"), "").unwrap();
        fs::write(ws.join("notes.md"), "").unwrap();
        fs::write(repo.join("notes.md"), "").unwrap();
        assert_eq!(dev_path("src/a.rs", &repo, &ws), repo.join("src/a.rs"));
        assert_eq!(dev_path("reflect.log", &repo, &ws), ws.join("reflect.log"));
        // The repository first when both hold it, and its path when neither.
        assert_eq!(dev_path("notes.md", &repo, &ws), repo.join("notes.md"));
        assert_eq!(dev_path("nope.c", &repo, &ws), repo.join("nope.c"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn paths_resolve_against_the_workspace() {
        let ws = Path::new("/ws");
        assert_eq!(resolve("a/b.md", ws), PathBuf::from("/ws/a/b.md"));
        assert_eq!(resolve("/etc/hosts", ws), PathBuf::from("/etc/hosts"));
    }
}
