//! `phi-stream tui`: the stream as something alive in the terminal. The
//! thoughts flow in the middle; what it says aloud stands out; what you
//! type is heard where the stream is when you press Enter and appears
//! there; a strip shows what it is doing (thinking, reading what you
//! gave it, catching up, summarizing to roll its context over) with its
//! rates and how full its context is. crossterm only, the family's
//! palette. See tui.md.

use std::fs;
use std::io::{self, BufRead, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event as TEvent, KeyCode, KeyEvent, KeyModifiers};
use crossterm::style::{Color, ResetColor};
use crossterm::{cursor, execute, queue, terminal};

use crate::client::{escape, parse, unescape, ActLine, Client, Delib, DelibKind, Msg, TermLine};
use crate::engine::{Kind, Mode, Status};
use crate::format::{self, Class};
use crate::mind::Reading;
use crate::reflect::{Episode, Outcome};
use crate::screen::{self, text_columns, Glyphs, Rect, Screen, Style, Weight};
use std::collections::VecDeque;

/// seaof.glass's palette, as Mechanical Jev's tui draws it.
mod theme {
    use crossterm::style::Color;
    pub const BG: Color = Color::Rgb {
        r: 0x0a,
        g: 0x0a,
        b: 0x0f,
    };
    pub const SURFACE: Color = Color::Rgb {
        r: 0x12,
        g: 0x12,
        b: 0x1a,
    };
    pub const TEXT: Color = Color::Rgb {
        r: 0xc4,
        g: 0x94,
        b: 0x5a,
    };
    pub const DIM: Color = Color::Rgb {
        r: 0x8a,
        g: 0x6a,
        b: 0x3e,
    };
    pub const ACCENT_DIM: Color = Color::Rgb {
        r: 0x7a,
        g: 0x5c,
        b: 0x38,
    };
    /// Speech: the warm text lifted.
    pub const BRIGHT: Color = Color::Rgb {
        r: 0xe8,
        g: 0xd0,
        b: 0xa8,
    };
    /// What was given: cooler, so it reads as from outside.
    pub const GIVEN: Color = Color::Rgb {
        r: 0x8c,
        g: 0xa6,
        b: 0xc0,
    };
    /// The Machine's white (the management plan, 2.3): the assessment's
    /// frames and labels, 19.75:1 on the background.
    pub const WHITE: Color = Color::Rgb {
        r: 0xff,
        g: 0xff,
        b: 0xff,
    };
    /// Code blocks' background: a step above the background, so a block
    /// reads as one.
    pub const CODE_BG: Color = Color::Rgb {
        r: 0x16,
        g: 0x16,
        b: 0x1e,
    };
    /// The Machine's yellow (plan 2.3): keywords.
    pub const YELLOW: Color = Color::Rgb {
        r: 0xee,
        g: 0xe9,
        b: 0x3c,
    };
    /// Its green: strings.
    pub const GREEN: Color = Color::Rgb {
        r: 0x39,
        g: 0xb1,
        b: 0x1c,
    };
    /// Its red, 4.44:1: frames and marks only, never text.
    pub const RED: Color = Color::Rgb {
        r: 0xeb,
        g: 0x1c,
        b: 0x24,
    };
}

/// What the title line says about the placement (the service's `info`).
pub struct Placement {
    pub model: String,
    pub gpu_blocks: usize,
    pub n_blocks: usize,
    pub gpu_gib: f64,
    pub host_gib: f64,
    pub n_ctx: u32,
    pub frame: String,
    pub workspace: String,
    pub started: i64,
}

struct Piece {
    text: String,
    kind: Kind,
}

struct View {
    pieces: Vec<Piece>,
    chars: usize,
    /// Lines scrolled up from the bottom; 0 follows the stream.
    scroll: usize,
    input: String,
    status: Option<Status>,
    notes: Vec<String>,
    started: Instant,
    heard: u32,
    /// The real time of the newest piece of the stream (microseconds).
    last_t_us: i64,
    /// The last readings of its mind, newest last.
    minds: VecDeque<Reading>,
    /// The guide lane at the last thinking token (`guide` lines).
    guide: Option<crate::client::GuideLine>,
    /// The engine's diagnostics, the last sent (DIAGNOSTICS).
    diag: Option<String>,
    /// What the main compartment shows (Tab cycles it).
    view: Pane,
    /// The checks and the engine's notes, oldest first: (real time, text).
    log: VecDeque<(i64, String)>,
    /// Box drawing for the outlines (a UTF-8 locale), else ASCII.
    utf8: bool,
    /// The last checks of its tokens (`reflect.rs`), newest last.
    episodes: VecDeque<Episode>,
    /// Whether the service answers, and since when it has not.
    link: Link,
    socket: String,
    /// `/quit` was sent: its `bye` ends this terminal too.
    quitting: bool,
    /// `--follow`: the builds this terminal has reloaded onto.
    follow: Option<u32>,
    /// The deliberation's text: each check's question, its reasoning,
    /// its outcome, as pieces of the given, thought and spoken kinds.
    delib: Vec<Piece>,
    delib_chars: usize,
    /// What it said aloud: each utterance under its time, as pieces.
    output: Vec<Piece>,
    output_chars: usize,
    /// The last piece was speech (the next speech continues its utterance).
    speaking: bool,
    /// Its terminal: each command and its output, as pieces.
    term: Vec<Piece>,
    term_chars: usize,
    /// What the stream is working toward (`objective` lines), and since when.
    objective: Option<(i64, String)>,
    /// Kinds of line this terminal does not show, each noted once.
    unknown: std::collections::HashSet<String>,
    /// The lens under the reasoning (`/lens`): a row under a line of its
    /// thinking when the strongest word on its mind that the line does not
    /// hold weighs at least this much; none: not shown.
    lens: Option<f32>,
    /// The thinking line being written: its tokens' positions, its text.
    line_pos: Vec<i32>,
    line_text: String,
    /// Checks whose written-over word is shown already (their times).
    struck: VecDeque<i64>,
}

/// The lens under a line by default: 0.10 put a row under 22.7 percent of
/// its thinking lines (4784 lines of mind.log, 2026-10-02; 0.15: 9.2).
const LENS_MIN: f32 = 0.10;
/// Rows are kept from this weight up, so `/lens` can lower the bar later.
const LENS_FLOOR: f32 = 0.05;

/// A lens row's strongest weight: its first word's percentage.
fn lens_weight(row: &[(char, Kind)]) -> f32 {
    let s: String = row.iter().map(|c| c.0).collect();
    s.split_once('%')
        .and_then(|(a, _)| a.rsplit(' ').next()?.parse::<f32>().ok())
        .map_or(0.0, |p| p / 100.0)
}

/// What the J-lens read on its mind over a line it wrote: for each of its
/// tokens (`pos`, the newest reading at each), each block's words summed as
/// probability over the readings, and `mind::unsaid` (the one rule, the
/// guide lane's lens aside uses it too): words of fewer than three letters,
/// words the line holds (`said`) and forms of them left out; a row when the
/// strongest weighs `min` or more, it and the others of at least half its
/// weight (a tail of 3 percent words, seen at a 5 percent bar, said
/// nothing), four at most, each with its weight.
fn lens_row(minds: &VecDeque<Reading>, pos: &[i32], said: &str, min: f32) -> Option<String> {
    let mut sums: Vec<(String, f32)> = Vec::new();
    let mut readings = 0usize;
    for p in pos {
        let Some(r) = minds.iter().rev().find(|r| r.pos == *p) else {
            continue;
        };
        for (_, ws) in &r.layers {
            readings += 1;
            for (w, lp) in ws {
                sums.push((w.clone(), lp.exp()));
            }
        }
    }
    let shown: Vec<String> = crate::mind::unsaid(&sums, readings, said, min)
        .into_iter()
        .map(|(w, x)| format!("{w} {:.0}%", x * 100.0))
        .collect();
    (!shown.is_empty()).then(|| format!("on its mind: {}", shown.join(" · ")))
}

/// The service as this terminal sees it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Link {
    Up,
    /// Not answered since this terminal started.
    Connecting,
    /// Gone at this real time (microseconds): looked for every `RETRY`.
    Gone(i64),
}

const KEEP_MINDS: usize = 1024;
const KEEP_EPISODES: usize = 64;
const KEEP_LOG: usize = 500;
const KEEP_DELIB_CHARS: usize = 100_000;
/// How often a missing service is looked for.
const RETRY: Duration = Duration::from_secs(3);
/// How often `--follow` looks at the binary.
const LOOK: Duration = Duration::from_millis(500);
/// The view handed from a terminal to the build that replaces it.
const STATE_VAR: &str = "PHI_STREAM_TUI_STATE";

/// A connection: commands go out through `w`; the service's lines come
/// from `rx`, read on a thread of their own, which ends with `Bye`.
struct Conn {
    w: UnixStream,
    rx: mpsc::Receiver<Msg>,
}

fn connect(socket: &Path) -> Option<Conn> {
    let mut c = Client::connect(socket).ok()?;
    c.send("tail").ok()?;
    let (reader, w) = c.split();
    let (tx, rx) = mpsc::channel::<Msg>();
    std::thread::spawn(move || {
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if tx.send(parse(&line)).is_err() {
                break;
            }
        }
        tx.send(Msg::Bye).ok();
    });
    Some(Conn { w, rx })
}

/// A file's identity: a build replaces the binary with a new file at the
/// same path (a new inode), so this differs from the running one's.
fn identity(m: &fs::Metadata) -> (u64, u64, i64) {
    (m.ino(), m.len(), m.mtime() * 1_000_000_000 + m.mtime_nsec())
}

/// `--follow`: the binary this terminal runs, watched for a new build.
struct Build {
    path: PathBuf,
    running: (u64, u64, i64),
    /// A new file seen at the last look; it is run when the next look
    /// finds it unchanged (the build is done writing it).
    seen: Option<(u64, u64, i64)>,
    next_look: Instant,
}

impl Build {
    fn new() -> Option<Self> {
        // The path it was started from; "(deleted)" when a build has
        // already replaced it.
        let exe = fs::read_link("/proc/self/exe").ok()?;
        let exe = exe.to_string_lossy();
        let path = PathBuf::from(exe.strip_suffix(" (deleted)").unwrap_or(&exe));
        let running = identity(&fs::metadata("/proc/self/exe").ok()?);
        Some(Self {
            path,
            running,
            seen: None,
            next_look: Instant::now() + LOOK,
        })
    }

    /// Whether a new build is there and finished.
    fn ready(&mut self) -> bool {
        if Instant::now() < self.next_look {
            return false;
        }
        self.next_look = Instant::now() + LOOK;
        match fs::metadata(&self.path).ok().map(|m| identity(&m)) {
            Some(id) if id != self.running => {
                let done = self.seen == Some(id);
                self.seen = Some(id);
                done
            }
            _ => {
                self.seen = None;
                false
            }
        }
    }

    /// The new build in this terminal's place, with the same arguments and
    /// the view handed over. The alternate screen stays, so nothing
    /// flashes; raw mode is left first, so the new process records the
    /// terminal's own mode to restore when it ends. Returns only on failure.
    fn reload(&self, v: &View) -> io::Error {
        let _ = terminal::disable_raw_mode();
        let e = std::process::Command::new(&self.path)
            .args(std::env::args_os().skip(1))
            .env(STATE_VAR, v.state())
            .exec();
        let _ = terminal::enable_raw_mode();
        e
    }
}

/// A token as shown in the mind's rows: newlines and tabs visible.
fn shown(token: &str) -> String {
    token.replace('\n', "⏎").replace('\t', "⇥")
}

/// One reading as one line: the token, then each block's words.
fn mind_row(r: &Reading) -> String {
    let blocks: Vec<String> = r
        .layers
        .iter()
        .map(|(l, ws)| {
            let words: Vec<&str> = ws.iter().map(|(w, _)| w.as_str()).collect();
            format!("{l}: {}", words.join(" "))
        })
        .collect();
    format!(
        "{}  {:>14}  {}",
        crate::clock::hms(r.t_us),
        shown(&r.token),
        blocks.join("  ·  ")
    )
}

/// A check in a few words: why, the token, what became of it.
fn episode_short(e: &Episode) -> String {
    let what = match e.outcome {
        Outcome::Changed => format!("wrote {:?}", e.to.trim()),
        Outcome::Dry => format!("would write {:?}", e.to.trim()),
        o => o.name().to_string(),
    };
    format!(
        "check {} {:?} (p {:.2}): {what}, {:.0} ms",
        e.why.name(),
        e.chosen.trim(),
        e.p_chosen,
        e.ms
    )
}

const KEEP_CHARS: usize = 400_000;

impl View {
    /// A `term` line into the terminal's text: a command as `$ COMMAND`
    /// under its time, its end as its output and how it ended.
    fn term_push(&mut self, t: &TermLine) {
        let pieces: Vec<Piece> = if t.end {
            let how = match (t.code, t.timed_out) {
                (_, true) => "stopped at its time limit".to_string(),
                (Some(c), _) => format!("exit {c}"),
                (None, _) => "did not run".to_string(),
            };
            vec![
                Piece {
                    text: format!("{}\n", t.text.trim_end()),
                    kind: Kind::Think,
                },
                Piece {
                    text: format!(
                        "[{how}, {:.0} ms{}]\n",
                        t.ms,
                        if t.cut { ", output cut" } else { "" }
                    ),
                    kind: Kind::Given,
                },
            ]
        } else {
            vec![Piece {
                text: format!("\n[{}] $ {}\n", crate::clock::hms(t.t_us), t.text),
                kind: Kind::Speak,
            }]
        };
        for p in pieces {
            self.term_chars += p.text.chars().count();
            self.term.push(p);
        }
        while self.term_chars > KEEP_DELIB_CHARS && self.term.len() > 1 {
            let p = self.term.remove(0);
            self.term_chars -= p.text.chars().count();
        }
    }

    /// A tool use into the output: its kind and what it was given under its
    /// time, then what it came to.
    fn act_push(&mut self, a: &ActLine) {
        self.speaking = false;
        let p = if a.end {
            Piece {
                text: format!("  {} {}\n", if a.ok { "->" } else { "x>" }, a.text),
                kind: Kind::Given,
            }
        } else {
            Piece {
                text: format!("\n[{}] {}: {}\n", crate::clock::hms(a.t_us), a.kind, a.text),
                kind: Kind::Speak,
            }
        };
        self.output_chars += p.text.chars().count();
        self.output.push(p);
        while self.output_chars > KEEP_DELIB_CHARS && self.output.len() > 1 {
            let p = self.output.remove(0);
            self.output_chars -= p.text.chars().count();
        }
    }

    /// Speech into the output: an utterance begins under its time, the
    /// oldest dropped past `KEEP_DELIB_CHARS`.
    fn output_push(&mut self, text: &str, t_us: i64) {
        if !self.speaking {
            let head = format!("\n[{}]\n", crate::clock::hms(t_us));
            self.output_chars += head.chars().count();
            self.output.push(Piece {
                text: head,
                kind: Kind::Given,
            });
        }
        self.speaking = true;
        self.output_chars += text.chars().count();
        self.output.push(Piece {
            text: text.to_string(),
            kind: Kind::Speak,
        });
        while self.output_chars > KEEP_DELIB_CHARS && self.output.len() > 1 {
            let p = self.output.remove(0);
            self.output_chars -= p.text.chars().count();
        }
    }

    /// A `delib` line into the deliberation's text: a check's start as a
    /// header and the question it was asked (given), its pieces as thoughts,
    /// its end as the outcome (spoken), the oldest dropped past
    /// `KEEP_DELIB_CHARS`.
    fn delib_push(&mut self, d: Delib) {
        let (text, kind) = match d.kind {
            DelibKind::Start => (
                format!(
                    "\n--- {} at {} ---\n{}\n",
                    d.pos,
                    crate::clock::hms(d.t_us),
                    d.text.trim()
                ),
                Kind::Given,
            ),
            DelibKind::Piece => (d.text, Kind::Think),
            DelibKind::End => (format!("\n=> {}\n", d.text.trim()), Kind::Speak),
        };
        self.delib_chars += text.chars().count();
        self.delib.push(Piece { text, kind });
        while self.delib_chars > KEEP_DELIB_CHARS && self.delib.len() > 1 {
            let p = self.delib.remove(0);
            self.delib_chars -= p.text.chars().count();
        }
    }

    /// A line of the log, the oldest dropped past `KEEP_LOG`.
    fn log_push(&mut self, t_us: i64, text: String) {
        self.log.push_back((t_us, text));
        while self.log.len() > KEEP_LOG {
            self.log.pop_front();
        }
    }

    /// What a reload hands over (`STATE_VAR`): the view, the counts and
    /// the line being typed, last and escaped, so it may hold anything.
    fn state(&self) -> String {
        format!(
            "mind={} scroll={} heard={} up={} reloads={} lens={} input={}",
            match self.view {
                Pane::Feed => 0,
                Pane::Mind => 1,
                Pane::Log => 2,
                Pane::Delib => 3,
                Pane::Output => 4,
                Pane::Term => 5,
                Pane::Diag => 6,
            },
            self.scroll,
            self.heard,
            self.started.elapsed().as_secs(),
            self.follow.unwrap_or(0),
            // In percent; 0: off.
            self.lens.map_or(0, |m| (m * 100.0).round() as u64),
            escape(&self.input)
        )
    }

    /// The state a reload handed over, back in place.
    fn restore(&mut self, s: &str) {
        let (fields, input) = s.split_once(" input=").unwrap_or((s, ""));
        self.input = unescape(input);
        for f in fields.split(' ') {
            let Some((k, val)) = f.split_once('=') else {
                continue;
            };
            let n: u64 = val.parse().unwrap_or(0);
            match k {
                "mind" => {
                    self.view = match n {
                        1 => Pane::Mind,
                        2 => Pane::Log,
                        3 => Pane::Delib,
                        4 => Pane::Output,
                        5 => Pane::Term,
                        6 => Pane::Diag,
                        _ => Pane::Feed,
                    }
                }
                "scroll" => self.scroll = n as usize,
                "heard" => self.heard = n as u32,
                "up" => {
                    self.started = Instant::now()
                        .checked_sub(Duration::from_secs(n))
                        .unwrap_or(self.started)
                }
                "reloads" => self.follow = self.follow.map(|_| n as u32 + 1),
                "lens" => self.lens = (n > 0).then_some(n as f32 / 100.0),
                _ => {}
            }
        }
    }

    fn push(&mut self, text: String, kind: Kind) {
        self.chars += text.chars().count();
        self.pieces.push(Piece { text, kind });
        while self.chars > KEEP_CHARS && self.pieces.len() > 1 {
            let p = self.pieces.remove(0);
            self.chars -= p.text.chars().count();
        }
    }

    /// A piece of the stream as it comes, a placed token with its position:
    /// a word a check wrote over goes in before the one that replaced it
    /// (its episode came first, the check holds the text until it ends);
    /// a thinking line, when it ends, gets the lens's row under it
    /// (`lens_row`, from the readings of its tokens; `rows` leaves out the
    /// rows inside a code block).
    fn push_stream(&mut self, text: String, kind: Kind, pos: Option<i32>) {
        if let Some(p) = pos {
            let over = self.episodes.iter().rev().find(|e| {
                e.pos == p && e.outcome == Outcome::Changed && !self.struck.contains(&e.t_us)
            });
            if let Some(e) = over {
                let (t, word) = (e.t_us, e.chosen.trim().to_string());
                self.struck.push_back(t);
                while self.struck.len() > KEEP_EPISODES {
                    self.struck.pop_front();
                }
                let lead = if text.starts_with(' ') { " " } else { "" };
                self.push(
                    format!("{lead}{}{word}{}", format::STRUCK, format::STRUCK),
                    kind,
                );
            }
            if kind == Kind::Think {
                self.line_pos.push(p);
            }
        }
        let mut rest = text.as_str();
        while let Some(i) = rest.find('\n') {
            self.line_text.push_str(&rest[..i]);
            let line = std::mem::take(&mut self.line_text);
            let pos = std::mem::take(&mut self.line_pos);
            let fence = format::is_fence(&line);
            // Inside a code block or not is the view's to say (`rows`): the
            // results it collapses hold fences too, and counted here they
            // left every later line inside one.
            let row = if kind == Kind::Think && !fence && pos.len() >= 3 {
                lens_row(&self.minds, &pos, &line, LENS_FLOOR)
            } else {
                None
            };
            // The newline first, then the row on a line of its own.
            let (head, tail) = rest.split_at(i + 1);
            if let Some(r) = row {
                self.push(head.to_string(), kind);
                self.push(format!("{}{r}\n", format::LENS_MARK), kind);
                rest = tail;
                if rest.is_empty() {
                    return;
                }
                continue;
            }
            self.push(head.to_string(), kind);
            rest = tail;
        }
        self.line_text.push_str(rest);
        if !rest.is_empty() {
            self.push(rest.to_string(), kind);
        }
    }

    /// The stream as rows of styled runs for `width` columns, set by
    /// `format.rs`: as a person reads it (`format::readable`: no template
    /// marks, one line per tool call and result), the lens's rows when on
    /// and strong enough, code blocks kept as written and highlighted,
    /// prose wrapped by words under its own indentation, Markdown marks
    /// shown as styles (a word may span pieces, since a token can end
    /// inside one).
    fn rows(&self, width: usize) -> Vec<Vec<(String, Kind, Class)>> {
        let mut lines: Vec<Vec<(char, Kind)>> = vec![Vec::new()];
        for p in &self.pieces {
            for ch in p.text.chars() {
                if ch == '\n' {
                    lines.push(Vec::new());
                } else {
                    lines.last_mut().unwrap().push((ch, p.kind));
                }
            }
        }
        // Inside a code block as shown (after `readable`), no lens row.
        let mut code = false;
        let lines: Vec<Vec<(char, Kind)>> = format::readable(&lines)
            .into_iter()
            .filter(|l| match l.first() {
                Some((c, _)) if *c == format::LENS_MARK => {
                    !code && self.lens.is_some_and(|min| lens_weight(l) >= min)
                }
                _ => {
                    let t: String = l.iter().map(|c| c.0).collect();
                    if format::is_fence(&t) {
                        code = !code;
                    }
                    true
                }
            })
            .collect();
        format::rows(&lines, width)
            .iter()
            .map(|r| runs(r))
            .collect()
    }
}

/// Pieces of text (the stream's, or the deliberation's) as rows of styled
/// runs for `width` columns, set by `format.rs`.
fn piece_rows(pieces: &[Piece], width: usize) -> Vec<Vec<(String, Kind, Class)>> {
    // Flatten into lines of characters with their kinds.
    let mut lines: Vec<Vec<(char, Kind)>> = vec![Vec::new()];
    for p in pieces {
        for ch in p.text.chars() {
            if ch == '\n' {
                lines.push(Vec::new());
            } else {
                lines.last_mut().unwrap().push((ch, p.kind));
            }
        }
    }
    format::rows(&lines, width)
        .iter()
        .map(|r| runs(r))
        .collect()
}

/// Characters of one row as runs of one kind and class.
fn runs(row: &[format::Cell]) -> Vec<(String, Kind, Class)> {
    let mut out: Vec<(String, Kind, Class)> = Vec::new();
    for &(c, k, cl) in row {
        match out.last_mut() {
            Some((s, kind, class)) if *kind == k && *class == cl => s.push(c),
            _ => out.push((c.to_string(), k, cl)),
        }
    }
    out
}

fn mode_line(s: &Status, tick: u64) -> (String, String) {
    let pulse = ["·", "•", "●", "•"][(tick / 4 % 4) as usize];
    let m = match &s.mode {
        Mode::Thinking => format!("{pulse} thinking"),
        Mode::Speaking => format!("{pulse} speaking"),
        Mode::Reading { done, total } => {
            format!("{pulse} reading {done}/{total} {}", bar(*done, *total, 12))
        }
        Mode::CatchingUp { done, total } => format!(
            "{pulse} taking it in {done}/{total} {}",
            bar(*done, *total, 12)
        ),
        Mode::Summarizing { tokens } => format!("{pulse} gathering its thoughts ({tokens} tokens)"),
        Mode::Paused => "paused".to_string(),
        Mode::Resting => {
            "resting until something new comes (a message, a commit, an objective)".to_string()
        }
    };
    let fill = bar(s.pos as usize, s.n_ctx as usize, 10);
    let rates = format!(
        "stream {:.1} tok/s · beside {:.1} tok/s · cycle {:.0} ms · context {} {:.1}k/{}k · queued {} · chunk {}{}",
        s.stream_tps,
        s.side_tps,
        s.cycle_ms,
        fill,
        s.pos as f64 / 1000.0,
        s.n_ctx / 1000,
        s.queued,
        if s.chunk == 0 { "auto".to_string() } else { s.chunk.to_string() },
        if s.rollovers > 0 {
            format!(" · rolled over {}x", s.rollovers)
        } else {
            String::new()
        }
    ) + &if s.checks > 0 || s.checking {
        format!(
            " · checks {} changed {}{}",
            s.checks,
            s.changes,
            if s.checking { " (one now)" } else { "" }
        )
    } else {
        String::new()
    };
    (m, rates)
}

fn bar(done: usize, total: usize, cells: usize) -> String {
    let filled = (done * cells + total / 2).checked_div(total).unwrap_or(0);
    let filled = filled.min(cells);
    format!("{}{}", "▇".repeat(filled), "▁".repeat(cells - filled))
}

fn plain(fg: Color, bg: Color) -> Style {
    Style {
        fg,
        bg,
        weight: Weight::Plain,
    }
}

/// The main compartment's views, cycled with Tab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pane {
    /// The stream.
    Feed,
    /// The readings of its mind, token by token.
    Mind,
    /// The checks and the engine's notes, over time.
    Log,
    /// The deliberation's own text (its compartment under the feed from
    /// `WIDE` columns; a view under it).
    Delib,
    /// What it says aloud, utterance by utterance (its compartment under the
    /// deliberation from `WIDE` columns; a view under it).
    Output,
    /// Its terminal: the commands it runs and their output (in the side
    /// column from `WIDE` columns; a view under it).
    Term,
    /// Its diagnostics: the one text the engine writes every few seconds,
    /// the same that the stream reads (`diag.md`, `report`).
    Diag,
}

impl Pane {
    fn name(self) -> &'static str {
        match self {
            Pane::Feed => "REASONING",
            Pane::Mind => "MIND",
            Pane::Log => "LOG",
            Pane::Delib => "DELIBERATION",
            Pane::Output => "OUTPUT",
            Pane::Term => "TERMINAL",
            Pane::Diag => "DIAGNOSTICS",
        }
    }

    fn next(self) -> Self {
        match self {
            Pane::Feed => Pane::Delib,
            Pane::Delib => Pane::Output,
            Pane::Output => Pane::Term,
            Pane::Term => Pane::Mind,
            Pane::Mind => Pane::Log,
            Pane::Log => Pane::Diag,
            Pane::Diag => Pane::Feed,
        }
    }
}

/// Where each compartment goes (`tui.md`), or none under the minimum.
#[derive(Debug)]
struct Layout {
    /// The view chosen (feed, deliberation, mind, log), framed.
    main: Rect,
    /// The deliberation under the view, when there is room for both.
    delib: Option<Rect>,
    /// The output under the deliberation, likewise.
    output: Option<Rect>,
    /// The terminal in the side column, between the assessment and the log.
    term: Option<Rect>,
    /// The last check, framed.
    assess: Rect,
    /// The log beside the view, framed, when there is room for a side column.
    log: Option<Rect>,
    mind: usize,
    status: usize,
    input: usize,
    hints: usize,
}

const MIN_W: usize = 80;
const MIN_H: usize = 24;
/// From this width the assessment and the log stand in a side column.
const WIDE: usize = 120;
const SIDE_W: usize = 40;

fn layout(w: usize, h: usize) -> Option<Layout> {
    if w < MIN_W || h < MIN_H {
        return None;
    }
    let (mind, status, input, hints) = (h - 4, h - 3, h - 2, h - 1);
    // Rows 1 up to the mind strip hold the compartments.
    let body = mind - 1;
    Some(if w >= WIDE {
        let left = w - SIDE_W;
        let assess_h = 12.min(body / 2);
        // Both reasoning chains side by side (the view on the left, the
        // deliberation on the right), what it says aloud under both.
        let out_h = (body / 4).max(5);
        let chains_h = body - out_h;
        let half = left / 2;
        Layout {
            main: Rect {
                top: 1,
                left: 0,
                h: chains_h,
                w: half,
            },
            delib: Some(Rect {
                top: 1,
                left: half,
                h: chains_h,
                w: left - half,
            }),
            output: Some(Rect {
                top: 1 + chains_h,
                left: 0,
                h: out_h,
                w: left,
            }),
            assess: Rect {
                top: 1,
                left,
                h: assess_h,
                w: SIDE_W,
            },
            // Under the assessment: the terminal, then the log, halves.
            term: Some(Rect {
                top: 1 + assess_h,
                left,
                h: (body - assess_h) / 2,
                w: SIDE_W,
            }),
            log: Some(Rect {
                top: 1 + assess_h + (body - assess_h) / 2,
                left,
                h: body - assess_h - (body - assess_h) / 2,
                w: SIDE_W,
            }),
            mind,
            status,
            input,
            hints,
        }
    } else {
        let assess_h = 6;
        Layout {
            main: Rect {
                top: 1,
                left: 0,
                h: body - assess_h,
                w,
            },
            delib: None,
            output: None,
            term: None,
            assess: Rect {
                top: 1 + body - assess_h,
                left: 0,
                h: assess_h,
                w,
            },
            log: None,
            mind,
            status,
            input,
            hints,
        }
    })
}

/// A probability as the assessment shows it: percent, two decimals.
fn pct(p: f32) -> String {
    format!("{:.2} %", p * 100.0)
}

/// A token as a label: its spaces and controls visible, quoted.
fn tok(t: &str) -> String {
    format!("{:?}", shown(t))
}

/// The lines of the assessment of check `e`; `narrow`: four longer
/// lines, for the compartment under the view at 80 columns.
fn assessment(e: &Episode, narrow: bool) -> Vec<String> {
    let label = match e.outcome {
        Outcome::Changed => "CHANGED",
        Outcome::Dry => "WOULD WRITE",
        Outcome::Kept => "KEPT",
        Outcome::Same => "SAME",
        Outcome::Unparsed => "UNPARSED",
        Outcome::Abandoned => "ABANDONED",
    };
    let wrote = match e.outcome {
        Outcome::Changed | Outcome::Dry => format!("  WROTE {}", tok(&e.to)),
        _ => String::new(),
    };
    let top: Vec<String> = e
        .top
        .iter()
        .map(|(t, p)| format!("{} {}", tok(t), pct(*p)))
        .collect();
    let words = if e.words.is_empty() {
        "nothing in particular".to_string()
    } else {
        e.words.join(", ")
    };
    let when = format!("{} pos {} {:.0} ms", crate::clock::hms(e.t_us), e.pos, e.ms);
    if narrow {
        vec![
            format!(
                "{label} {}  p {:.4} {}  KEEP {}  WRITE {}  ANSWERED {}{wrote}",
                tok(&e.chosen),
                e.p_chosen,
                e.why.name().to_uppercase(),
                pct(e.keep),
                pct(1.0 - e.keep),
                pct(e.fmt)
            ),
            format!("TOP  {}", top.join(" · ")),
            format!("ON ITS MIND  {words}"),
            format!("RULE {}  ·  {when}", e.rule),
        ]
    } else {
        let mut l = vec![
            format!("{label}{wrote}"),
            format!("KEEP     {}", pct(e.keep)),
            format!("WRITE    {}", pct(1.0 - e.keep)),
            format!("ANSWERED {}", pct(e.fmt)),
            format!("p {:.4} {}", e.p_chosen, e.why.name().to_uppercase()),
        ];
        for (i, t) in top.iter().enumerate() {
            l.push(format!("{} {t}", if i == 0 { "TOP" } else { "   " }));
        }
        l.push(format!("ON ITS MIND {words}"));
        l.push(format!("RULE {}", e.rule));
        l.push(when);
        l
    }
}

/// The frame around the token in question (plan 2.4: an outline with an
/// inward tick at each side's midpoint), 5 rows by 13 columns at
/// `top`, `left`: light for a check that kept it, heavy red for one that
/// changed it, a white outline with heavy red ticks for one that would
/// have (dry) or one in flight. Heavy glyphs, or `#` in ASCII, carry the
/// state without colour.
fn token_frame(
    s: &mut Screen,
    top: usize,
    left: usize,
    token: &str,
    state: Option<Outcome>,
    utf8: bool,
) {
    let white = plain(theme::WHITE, theme::BG);
    let red = plain(theme::RED, theme::BG);
    let (edge, ticks_style, heavy_ticks, heavy_edge) = match state {
        Some(Outcome::Changed) => (red, red, true, true),
        Some(Outcome::Dry) | None => (white, red, true, false),
        _ => (white, white, false, false),
    };
    let g = match (utf8, heavy_edge) {
        (false, true) => &Glyphs {
            h: '#',
            v: '#',
            tl: '#',
            tr: '#',
            bl: '#',
            br: '#',
        },
        (false, false) => &screen::ASCII,
        (true, true) => &screen::HEAVY,
        (true, false) => &screen::LIGHT,
    };
    let r = Rect {
        top,
        left,
        h: 5,
        w: 13,
    };
    s.frame(r, g, edge, "", edge);
    let (tt, tb, tl, tr) = match (utf8, heavy_ticks) {
        (true, true) => ('┳', '┻', '┣', '┫'),
        (true, false) => ('┬', '┴', '├', '┤'),
        (false, true) => ('#', '#', '#', '#'),
        (false, false) => ('+', '+', '+', '+'),
    };
    let mid = left + 6;
    s.put(top, mid, &tt.to_string(), ticks_style);
    s.put(top + 4, mid, &tb.to_string(), ticks_style);
    s.put(top + 2, left, &tl.to_string(), ticks_style);
    s.put(top + 2, left + 12, &tr.to_string(), ticks_style);
    let t = shown(token.trim());
    let t: String = t.chars().take(9).collect();
    let pad = (9usize.saturating_sub(text_columns(&t))) / 2;
    s.put_to(
        top + 2,
        left + 2 + pad,
        left + 11,
        &t,
        plain(theme::WHITE, theme::BG),
    );
}

/// `text` wrapped by words to `width` terminal columns, the lines after
/// the first indented by `indent`; a word longer than a line is left to be
/// clipped where it is drawn.
fn wrap(text: &str, width: usize, indent: usize) -> Vec<String> {
    let width = width.max(indent + 8);
    let mut rows = Vec::new();
    let mut line = String::new();
    let mut cols = 0;
    for word in text.split(' ') {
        let wc = text_columns(word);
        if cols + wc > width && cols > indent {
            rows.push(std::mem::take(&mut line));
            line = " ".repeat(indent);
            cols = indent;
        }
        line.push_str(word);
        line.push(' ');
        cols += wc + 1;
    }
    rows.push(line);
    rows
}

/// The log's lines for a compartment `width` wide, oldest first: each
/// entry its time, then its text, wrapped with a two-column indent.
fn log_rows(log: &VecDeque<(i64, String)>, width: usize) -> Vec<String> {
    log.iter()
        .flat_map(|(t, text)| wrap(&format!("{} {text}", crate::clock::hms(*t)), width, 2))
        .collect()
}

/// A run's style: its kind's for prose (thoughts warm, speech lifted and
/// bold, what was given cool and italic on the surface), the code palette
/// inside a block and for inline code (`format.rs`).
fn style_of(kind: Kind, class: Class) -> Style {
    let code = |fg| Style {
        fg,
        bg: theme::CODE_BG,
        weight: Weight::Plain,
    };
    match class {
        Class::Prose => match kind {
            Kind::Think => plain(theme::TEXT, theme::BG),
            Kind::Speak => Style {
                fg: theme::BRIGHT,
                bg: theme::BG,
                weight: Weight::Bold,
            },
            Kind::Given => Style {
                fg: theme::GIVEN,
                bg: theme::SURFACE,
                weight: Weight::Italic,
            },
        },
        Class::Bold => Style {
            weight: Weight::Bold,
            ..style_of(kind, Class::Prose)
        },
        Class::Heading => Style {
            fg: theme::BRIGHT,
            bg: theme::BG,
            weight: Weight::Bold,
        },
        Class::Code | Class::Plain => code(theme::BRIGHT),
        Class::Fence => code(theme::GIVEN),
        Class::Keyword => code(theme::YELLOW),
        Class::Str => code(theme::GREEN),
        Class::Number => code(theme::WHITE),
        Class::Comment => Style {
            fg: theme::GIVEN,
            bg: theme::CODE_BG,
            weight: Weight::Italic,
        },
        // What the lens read under a line: the cool colour of what comes
        // from outside the text, italic, on the plain background.
        Class::Lens => Style {
            fg: theme::GIVEN,
            bg: theme::BG,
            weight: Weight::Italic,
        },
        // A word written over: crossed out, the cool colour.
        Class::Struck => Style {
            fg: theme::GIVEN,
            bg: theme::BG,
            weight: Weight::Struck,
        },
    }
}

/// Styled rows into the compartment inside `inner`, the newest at its
/// bottom, `scroll` rows up from the end; a code block's background runs
/// to the compartment's edge.
fn draw_rows(s: &mut Screen, inner: Rect, rows: &[Vec<(String, Kind, Class)>], scroll: usize) {
    let end = inner.left + inner.w;
    let last = rows.len().saturating_sub(scroll);
    let first = last.saturating_sub(inner.h);
    for (r, row) in rows[first..last].iter().enumerate() {
        let mut col = inner.left + 1;
        for (text, kind, class) in row {
            col = s.put_to(inner.top + r, col, end, text, style_of(*kind, *class));
        }
        if row.first().is_some_and(|x| x.2.in_block()) && col < end {
            let pad = " ".repeat(end - col);
            s.put_to(
                inner.top + r,
                col,
                end,
                &pad,
                style_of(Kind::Think, Class::Plain),
            );
        }
    }
}

/// Its terminal: each command and its output; says so when there is none.
fn draw_term(s: &mut Screen, inner: Rect, v: &View, scroll: usize) {
    if v.term.is_empty() {
        let why =
            "no command since this terminal connected (the service runs them with --terminal)";
        for (r, l) in wrap(why, inner.w.saturating_sub(2), 0)
            .iter()
            .take(inner.h)
            .enumerate()
        {
            s.put_to(
                inner.top + r,
                inner.left + 1,
                inner.left + inner.w,
                l,
                plain(theme::GIVEN, theme::BG),
            );
        }
        return;
    }
    draw_rows(
        s,
        inner,
        &piece_rows(&v.term, inner.w.saturating_sub(2)),
        scroll,
    );
}

/// What it said aloud, each utterance under its time; says so when it
/// has said nothing yet.
fn draw_output(s: &mut Screen, inner: Rect, v: &View, scroll: usize) {
    if v.output.is_empty() {
        let why = "no tool use or speech since this terminal connected";
        s.put_to(
            inner.top,
            inner.left + 1,
            inner.left + inner.w,
            why,
            plain(theme::GIVEN, theme::BG),
        );
        return;
    }
    draw_rows(
        s,
        inner,
        &piece_rows(&v.output, inner.w.saturating_sub(2)),
        scroll,
    );
}

/// The deliberation: what the stream is working toward on top (its
/// `objective`), under it each check's question, its own reasoning and
/// its outcome. Says what it lacks rather than leaving it blank.
/// DIAGNOSTICS: the engine's one text, as the stream reads it; a section's
/// head (a line without indent) bright, its lines plain.
fn draw_diag(s: &mut Screen, inner: Rect, v: &View, scroll: usize) {
    let end = inner.left + inner.w;
    let width = inner.w.saturating_sub(2);
    let text = v.diag.as_deref().unwrap_or(
        "no diagnostics from this service yet: it sends them every few seconds (the same text the stream reads as diag.md)",
    );
    let mut rows: Vec<(String, bool)> = Vec::new();
    for l in text.lines() {
        let head = !l.starts_with(' ');
        for w in wrap(l, width, 4) {
            rows.push((w, head));
        }
    }
    let last = rows.len().saturating_sub(scroll);
    let first = last.saturating_sub(inner.h);
    for (r, (l, head)) in rows[first..last].iter().enumerate() {
        let style = if *head {
            plain(theme::WHITE, theme::BG)
        } else {
            plain(theme::GIVEN, theme::BG)
        };
        s.put_to(inner.top + r, inner.left + 1, end, l, style);
    }
}

fn draw_delib(s: &mut Screen, inner: Rect, v: &View, scroll: usize) {
    let end = inner.left + inner.w;
    let width = inner.w.saturating_sub(2);
    let head = match &v.objective {
        Some((t, text)) => format!("OBJECTIVE ({}) {text}", crate::clock::hms(*t)),
        None => "OBJECTIVE none sent by this service".to_string(),
    };
    let head = wrap(&head, width, 2);
    let top_rows = head.len().min(inner.h / 2).max(1);
    for (r, l) in head.iter().take(top_rows).enumerate() {
        s.put_to(
            inner.top + r,
            inner.left + 1,
            end,
            l,
            plain(theme::WHITE, theme::BG),
        );
    }
    let rest = Rect {
        top: inner.top + top_rows,
        h: inner.h.saturating_sub(top_rows),
        ..inner
    };
    if v.delib.is_empty() {
        let why = "no deliberation from this service yet: one that reasons sends each check's question, its reasoning and its outcome here";
        for (r, l) in wrap(why, width, 0).iter().take(rest.h).enumerate() {
            s.put_to(
                rest.top + r,
                rest.left + 1,
                end,
                l,
                plain(theme::GIVEN, theme::BG),
            );
        }
        return;
    }
    draw_rows(s, rest, &piece_rows(&v.delib, width), scroll);
}

/// One frame: drawn into a fresh screen, then only what changed since
/// `front` (the frame the terminal shows) is written (`screen.md`).
fn draw(
    out: &mut impl Write,
    v: &View,
    p: &Placement,
    tick: u64,
    front: &mut Option<Screen>,
) -> io::Result<()> {
    let (w, h) = terminal::size()?;
    let (w, h) = (w as usize, h as usize);
    let base = plain(theme::TEXT, theme::BG);
    let mut s = Screen::new(w, h, base);
    let Some(lay) = layout(w, h) else {
        // Under the minimum: the size it needs, and nothing else.
        s.line(
            0,
            &format!("phi-stream needs {MIN_W}x{MIN_H}; this terminal is {w}x{h}"),
            base,
        );
        queue!(out, cursor::Hide)?;
        s.diff(front.as_ref(), out)?;
        out.flush()?;
        *front = Some(s);
        return Ok(());
    };
    let g = if v.utf8 {
        &screen::LIGHT
    } else {
        &screen::ASCII
    };
    let edge = plain(theme::DIM, theme::BG);
    let label = Style {
        fg: theme::TEXT,
        bg: theme::BG,
        weight: Weight::Bold,
    };
    // Title.
    let title = format!(
        " phi-stream · {} · {} · GPU {}/{} blocks {:.1} GiB · cards+host {:.1} GiB · {}k cells · up {}m · heard {} · {}",
        p.model,
        p.frame,
        p.gpu_blocks,
        p.n_blocks,
        p.gpu_gib,
        p.host_gib,
        p.n_ctx / 1000,
        // The service's own time up (this terminal's, from an older one).
        if p.started > 0 {
            (crate::clock::now_us() - p.started).max(0) as u64 / 60_000_000
        } else {
            v.started.elapsed().as_secs() / 60
        },
        v.heard,
        if v.last_t_us > 0 { crate::clock::hms(v.last_t_us) } else { String::new() }
    );
    s.line(0, &title, plain(theme::DIM, theme::SURFACE));

    // The main compartment: the view chosen.
    let views = format!("{} · Tab: {}", v.view.name(), v.view.next().name());
    s.frame(lay.main, g, edge, &views, label);
    let inner = lay.main.inner();
    let end = inner.left + inner.w;
    match v.view {
        Pane::Feed => {
            let rows = v.rows(inner.w.saturating_sub(2));
            draw_rows(&mut s, inner, &rows, v.scroll);
        }
        Pane::Delib => draw_delib(&mut s, inner, v, v.scroll),
        Pane::Output => draw_output(&mut s, inner, v, v.scroll),
        Pane::Term => draw_term(&mut s, inner, v, v.scroll),
        Pane::Diag => draw_diag(&mut s, inner, v, v.scroll),
        Pane::Mind => {
            // The readings, newest at the bottom; a check beside the
            // reading it was asked from (the one before its token).
            let last = v.minds.len().saturating_sub(v.scroll);
            let first = last.saturating_sub(inner.h);
            for r in 0..inner.h {
                let Some(m) = v.minds.get(first + r) else {
                    continue;
                };
                let text = match v.episodes.iter().rev().find(|e| e.pos == m.pos + 1) {
                    Some(e) => format!("{}   [{}]", mind_row(m), episode_short(e)),
                    None => mind_row(m),
                };
                s.put_to(
                    inner.top + r,
                    inner.left + 1,
                    end,
                    &text,
                    plain(theme::GIVEN, theme::BG),
                );
            }
        }
        Pane::Log => {
            let rows = log_rows(&v.log, inner.w.saturating_sub(2));
            let last = rows.len().saturating_sub(v.scroll);
            let first = last.saturating_sub(inner.h);
            for (r, line) in rows[first..last].iter().enumerate() {
                s.put_to(
                    inner.top + r,
                    inner.left + 1,
                    end,
                    line,
                    plain(theme::GIVEN, theme::BG),
                );
            }
        }
    }

    // Both reasoning streams at once, when there is room: with the
    // deliberation in the main view, the reasoning beside it (seen through
    // the MCP screen: both columns showed the deliberation).
    if let Some(dr) = lay.delib {
        if v.view == Pane::Delib {
            s.frame(dr, g, edge, "REASONING", label);
            let inner = dr.inner();
            let rows = v.rows(inner.w.saturating_sub(2));
            draw_rows(&mut s, inner, &rows, 0);
        } else {
            s.frame(dr, g, edge, "DELIBERATION", label);
            draw_delib(&mut s, dr.inner(), v, 0);
        }
    }
    if let Some(tr) = lay.term {
        s.frame(tr, g, edge, "TERMINAL", label);
        draw_term(&mut s, tr.inner(), v, 0);
    }
    if let Some(or) = lay.output {
        s.frame(or, g, edge, "OUTPUT", label);
        draw_output(&mut s, or.inner(), v, 0);
    }

    // The assessment: the last check, and whether one is in flight.
    let checking = v.status.as_ref().is_some_and(|s| s.checking);
    let assess_label = if checking {
        "ASSESSMENT · ASSESSING NOW"
    } else {
        "ASSESSMENT"
    };
    s.frame(lay.assess, g, edge, assess_label, label);
    let ai = lay.assess.inner();
    let aend = ai.left + ai.w;
    let text = plain(theme::TEXT, theme::BG);
    match v.episodes.back() {
        None => {
            // What it knows, not a guess: the service does not say whether it
            // reflects, only how many checks it has run.
            let why = match &v.status {
                None => "no status from the service yet".to_string(),
                Some(s) => format!(
                    "no check has ended since this terminal connected ({} in all; checks run with --reflect)",
                    s.checks
                ),
            };
            for (r, l) in wrap(&why, ai.w.saturating_sub(2), 0)
                .iter()
                .take(ai.h)
                .enumerate()
            {
                s.put_to(ai.top + r, ai.left + 1, aend, l, text);
            }
        }
        Some(e) if lay.log.is_none() => {
            for (r, l) in assessment(e, true).iter().take(ai.h).enumerate() {
                s.put_to(ai.top + r, ai.left + 1, aend, l, text);
            }
        }
        Some(e) => {
            token_frame(
                &mut s,
                ai.top,
                ai.left + 1,
                &e.chosen,
                Some(e.outcome),
                v.utf8,
            );
            let lines = assessment(e, false);
            // Beside the frame (5 rows): the outcome and the branches.
            for (r, l) in lines.iter().take(5.min(ai.h)).enumerate() {
                let st = if r == 0 {
                    plain(theme::WHITE, theme::BG)
                } else {
                    text
                };
                s.put_to(ai.top + r, ai.left + 15, aend, l, st);
            }
            // Under it the rest, wrapped, as many rows as there are.
            let rest: Vec<String> = lines
                .iter()
                .skip(5)
                .flat_map(|l| wrap(l, ai.w.saturating_sub(2), 2))
                .collect();
            for (r, l) in rest.iter().take(ai.h.saturating_sub(5)).enumerate() {
                s.put_to(ai.top + 5 + r, ai.left + 1, aend, l, text);
            }
        }
    }

    // The log beside, when there is room.
    if let Some(lr) = lay.log {
        s.frame(lr, g, edge, "LOG", label);
        let li = lr.inner();
        let rows = log_rows(&v.log, li.w.saturating_sub(2));
        let first = rows.len().saturating_sub(li.h);
        for (r, line) in rows[first..].iter().enumerate() {
            s.put_to(
                li.top + r,
                li.left + 1,
                li.left + li.w,
                line,
                plain(theme::GIVEN, theme::BG),
            );
        }
    }

    // The mind strip: what was on its mind at the last token it placed,
    // after the guide lane at that token when it reads one (every thinking
    // token reasoned against its reflection: how far it moved, whether the
    // likeliest token changed, the experts the two share).
    let guide = v
        .guide
        .as_ref()
        .filter(|g| v.minds.back().is_some_and(|r| (r.pos - g.pos).abs() <= 2))
        .map(|g| {
            // Its aside's source (chain, lens, placebo) when not the chain.
            let src = if g.src == "chain" {
                String::new()
            } else {
                format!("{} ", g.src)
            };
            format!(
                "GUIDE {src}{} kl {:.2}{}{}   ",
                if g.mix > 0.0 {
                    format!("mix {}", g.mix)
                } else {
                    "shadow".to_string()
                },
                g.kl,
                if g.flip { " top changed" } else { "" },
                g.shared.map_or(String::new(), |s| format!(
                    " experts {:.0}% shared",
                    100.0 * s
                ))
            )
        })
        .unwrap_or_default();
    match v.minds.back() {
        Some(r) => s.line(
            lay.mind,
            &format!(
                " {guide}MIND  {}   ({:.1} ms)",
                mind_row(r).trim_start(),
                r.ms
            ),
            plain(theme::GIVEN, theme::SURFACE),
        ),
        None => {
            // Its status says whether it reads its mind: a reading time of 0.
            let why = match &v.status {
                Some(s) if s.mind_ms == 0.0 => {
                    "the service does not read its mind (it does with --mind)"
                }
                _ => "no reading yet",
            };
            s.line(
                lay.mind,
                &format!(" MIND  {why}"),
                plain(theme::GIVEN, theme::SURFACE),
            )
        }
    }
    // The strip: what it is doing, and the rates.
    let (mode, rates) = match (v.link, &v.status) {
        (Link::Up, Some(s)) => mode_line(s, tick),
        (Link::Up, None) => (
            "· waking (the model may be loading)".to_string(),
            String::new(),
        ),
        (Link::Connecting, _) => (
            "CONNECTING".to_string(),
            format!("no answer yet at {}; looking every 3 s", v.socket),
        ),
        (Link::Gone(t), _) => (
            "NO SERVICE".to_string(),
            format!(
                "gone at {}; looking every 3 s at {} (a restart reconnects here)",
                crate::clock::hms(t),
                v.socket
            ),
        ),
    };
    s.line(
        lay.status,
        &format!(" {mode}   {rates}"),
        plain(theme::TEXT, theme::SURFACE),
    );
    // Input.
    s.line(
        lay.input,
        &format!(" › {}", v.input),
        plain(theme::BRIGHT, theme::BG),
    );
    // Hints and the last note.
    let note = v.notes.last().cloned().unwrap_or_default();
    let follows = match v.follow {
        Some(0) => " follows the build ·".to_string(),
        Some(n) => format!(" follows the build ({n} reloaded) ·"),
        None => String::new(),
    };
    let hints = format!(
        "{follows} Enter speaks · Tab views · /objective TEXT · /feed FILE · /persona FILE · /pause /resume · /chunk N · /temp T · /lens [on|off|P%] · /chain on|off|against|audit · /goal on|off · /quit stops it · PgUp PgDn End · ^C leaves it running   {}   {note}",
        p.workspace
    );
    s.line(lay.hints, &hints, plain(theme::ACCENT_DIM, theme::BG));
    queue!(out, cursor::Hide)?;
    s.diff(front.as_ref(), out)?;
    let cx = (3 + text_columns(&v.input)).min(w - 1) as u16;
    queue!(out, cursor::MoveTo(cx, lay.input as u16), cursor::Show)?;
    out.flush()?;
    *front = Some(s);
    Ok(())
}

/// A typed line: a command, or something said; sent to the service
/// (none: nothing is sent, and it says so).
fn submit(line: &str, w: Option<&mut UnixStream>, v: &mut View) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let Some(w) = w else {
        v.notes.push(format!("no service: {line:?} was not sent"));
        return;
    };
    let msg = if let Some(p) = line.strip_prefix("/feed ") {
        v.notes.push(format!("handing over {}", p.trim()));
        format!("feed {}", crate::expand_home(p.trim()))
    } else if let Some(p) = line.strip_prefix("/persona ") {
        format!("persona {}", crate::expand_home(p.trim()))
    } else if line == "/pause" {
        "pause".to_string()
    } else if line == "/resume" {
        "resume".to_string()
    } else if let Some(c) = line.strip_prefix("/chunk ") {
        format!("chunk {}", c.trim())
    } else if let Some(t) = line.strip_prefix("/temp ") {
        format!("temp {}", t.trim())
    } else if line == "/mind" || line == "/log" || line == "/feed" {
        // A view by name (Tab cycles them); /mind again goes back to the
        // stream, as before.
        v.view = match line {
            "/mind" if v.view != Pane::Mind => Pane::Mind,
            "/log" => Pane::Log,
            _ => Pane::Feed,
        };
        v.scroll = 0;
        return;
    } else if line == "/lens" || line.starts_with("/lens ") {
        // The lens's rows under the reasoning: on and off, or the weight a
        // row needs, in percent (rows are kept from `LENS_FLOOR` up).
        let arg = line.trim_start_matches("/lens").trim();
        v.lens = match (arg, v.lens) {
            ("", Some(_)) | ("off", _) => None,
            ("", None) | ("on", _) => Some(LENS_MIN),
            (p, _) => match p.trim_end_matches('%').parse::<f32>() {
                Ok(x) if x > 0.0 => Some((x / 100.0).max(LENS_FLOOR)),
                _ => {
                    v.notes
                        .push(format!("/lens takes on, off or a percent, not {p:?}"));
                    return;
                }
            },
        };
        v.notes.push(match v.lens {
            Some(m) => format!(
                "the lens under the reasoning: a row when the strongest word on its mind that the line does not hold weighs {:.0}% or more",
                m * 100.0
            ),
            None => "the lens under the reasoning: off (/lens turns it on)".to_string(),
        });
        return;
    } else if matches!(
        line,
        "/chain on" | "/chain off" | "/chain against" | "/chain audit" | "/goal on" | "/goal off"
    ) {
        line.trim_start_matches('/').to_string()
    } else if line == "/objective" {
        v.notes.push(match &v.objective {
            Some((_, t)) => {
                format!("objective: {t} (/objective TEXT sets one, /objective - clears it)")
            }
            None => "no objective (/objective TEXT sets one)".to_string(),
        });
        return;
    } else if let Some(o) = line.strip_prefix("/objective ") {
        let o = o.trim();
        format!("objective {}", if o == "-" { "" } else { o })
    } else if line == "/quit" {
        v.quitting = true;
        "quit".to_string()
    } else if line.starts_with('/') {
        v.notes.push(format!("unknown command {line}"));
        return;
    } else {
        v.heard += 1;
        format!("say {}", escape(line))
    };
    if writeln!(w, "{msg}").is_err() {
        v.notes.push("the service is gone".into());
    }
}

/// The terminal, as a client of the service at `socket`. It outlives the
/// service: when it goes (a restart, a crash) the terminal says so and
/// reconnects when it is back. With `follow`, it also reloads onto each
/// new build of its own binary, keeping its view and the line being typed.
pub fn run(socket: &Path, follow: bool) -> Result<()> {
    let mut conn = connect(socket);
    let mut build = if follow { Build::new() } else { None };
    // Started by a reload: the terminal is already in the alternate screen.
    let handed = std::env::var(STATE_VAR).ok();
    // A panic leaves the terminal as it found it, so its message is legible.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = execute!(
            io::stdout(),
            ResetColor,
            cursor::Show,
            terminal::LeaveAlternateScreen
        );
        let _ = terminal::disable_raw_mode();
        default_hook(info);
    }));
    let mut placement = Placement {
        model: String::new(),
        gpu_blocks: 0,
        n_blocks: 0,
        gpu_gib: 0.0,
        host_gib: 0.0,
        n_ctx: 0,
        frame: String::new(),
        workspace: String::new(),
        started: 0,
    };
    let mut out = io::stdout();
    terminal::enable_raw_mode()?;
    if handed.is_none() {
        execute!(
            out,
            terminal::EnterAlternateScreen,
            terminal::Clear(terminal::ClearType::All)
        )?;
    }
    let result = (|| -> Result<()> {
        let mut v = View {
            pieces: Vec::new(),
            chars: 0,
            scroll: 0,
            input: String::new(),
            status: None,
            notes: Vec::new(),
            started: Instant::now(),
            heard: 0,
            last_t_us: 0,
            minds: VecDeque::new(),
            guide: None,
            diag: None,
            view: Pane::Feed,
            log: VecDeque::new(),
            utf8: screen::utf8_locale(|k| std::env::var(k).ok()),
            episodes: VecDeque::new(),
            link: if conn.is_some() {
                Link::Up
            } else {
                Link::Connecting
            },
            socket: socket.display().to_string(),
            quitting: false,
            follow: build.as_ref().map(|_| 0),
            delib: Vec::new(),
            delib_chars: 0,
            output: Vec::new(),
            output_chars: 0,
            speaking: false,
            term: Vec::new(),
            term_chars: 0,
            objective: None,
            unknown: Default::default(),
            lens: Some(LENS_MIN),
            line_pos: Vec::new(),
            line_text: String::new(),
            struck: VecDeque::new(),
        };
        if let Some(s) = &handed {
            v.restore(s);
            v.notes.push(format!(
                "reloaded onto the new build at {}",
                crate::clock::hms(crate::clock::now_us())
            ));
        } else if follow && build.is_none() {
            v.notes
                .push("--follow: its own binary could not be found; not following".into());
        }
        let mut last_try = Instant::now();
        let mut tick = 0u64;
        let mut dirty = true;
        // The frame the terminal shows (none: unknown, so all of it is drawn).
        let mut front: Option<Screen> = None;
        let mut last_draw = Instant::now();
        loop {
            // No service: looked for every RETRY; when it answers, its tail
            // replays the stream's last characters, so the old ones go.
            if conn.is_none() && last_try.elapsed() >= RETRY {
                last_try = Instant::now();
                if let Some(c) = connect(socket) {
                    conn = Some(c);
                    if let Link::Gone(_) = v.link {
                        v.notes.push(format!(
                            "the service is back at {}",
                            crate::clock::hms(crate::clock::now_us())
                        ));
                    }
                    v.link = Link::Up;
                    v.status = None;
                    v.pieces.clear();
                    v.chars = 0;
                    dirty = true;
                }
            }
            // A new build of this terminal: run it in this one's place.
            if let Some(b) = build.as_mut() {
                if b.ready() {
                    let e = b.reload(&v);
                    // Only on failure: the next finished build is tried.
                    b.seen = None;
                    v.notes.push(format!("the new build did not start: {e}"));
                    front = None;
                    dirty = true;
                }
            }
            // The service's messages, all that are waiting.
            let mut gone = false;
            while let Some(c) = &conn {
                match c.rx.try_recv() {
                    Ok(Msg::Info(i)) => {
                        placement = Placement {
                            model: i.model,
                            gpu_blocks: i.gpu_blocks,
                            n_blocks: i.n_blocks,
                            gpu_gib: i.gpu_gib,
                            host_gib: i.host_gib,
                            n_ctx: i.n_ctx,
                            frame: i.frame,
                            workspace: i.workspace,
                            started: i.started,
                        };
                        dirty = true;
                    }
                    Ok(Msg::Text(t, k, at, pos)) => {
                        // Speech is the output too; anything else ends an utterance.
                        if k == Kind::Speak {
                            v.output_push(&t, at);
                        } else {
                            v.speaking = false;
                        }
                        v.push_stream(t, k, pos);
                        v.last_t_us = at;
                        dirty = true;
                    }
                    Ok(Msg::Status(s)) => {
                        v.status = Some(s);
                        dirty = true;
                    }
                    Ok(Msg::Note(n)) => {
                        // A note carries no time: the time it arrived.
                        v.log_push(crate::clock::now_us(), n.clone());
                        v.notes.push(n);
                        dirty = true;
                    }
                    Ok(Msg::Guide(g)) => {
                        v.guide = Some(g);
                        dirty = true;
                    }
                    Ok(Msg::Diag(d)) => {
                        v.diag = Some(d);
                        dirty = true;
                    }
                    Ok(Msg::Mind(r)) => {
                        v.minds.push_back(r);
                        while v.minds.len() > KEEP_MINDS {
                            v.minds.pop_front();
                        }
                        dirty = true;
                    }
                    Ok(Msg::Reflect(e)) => {
                        v.log_push(e.t_us, episode_short(&e));
                        v.episodes.push_back(e);
                        while v.episodes.len() > KEEP_EPISODES {
                            v.episodes.pop_front();
                        }
                        dirty = true;
                    }
                    Ok(Msg::Err(e)) => {
                        v.notes.push(e);
                        dirty = true;
                    }
                    Ok(Msg::Ok(_)) => {}
                    Ok(Msg::Delib(d)) => {
                        v.delib_push(d);
                        dirty = true;
                    }
                    Ok(Msg::Act(a)) => {
                        if !a.end {
                            v.log_push(a.t_us, format!("{}: {}", a.kind, a.text));
                        }
                        v.act_push(&a);
                        dirty = true;
                    }
                    Ok(Msg::ToClaude(m)) => {
                        let re = m.re.map(|r| format!(", answering {r}")).unwrap_or_default();
                        v.log_push(m.t_us, format!("to Claude (m{}{re}): {}", m.id, m.text));
                        dirty = true;
                    }
                    Ok(Msg::Term(t)) => {
                        if !t.end {
                            v.log_push(t.t_us, format!("ran: {}", t.text));
                        }
                        v.term_push(&t);
                        dirty = true;
                    }
                    Ok(Msg::Objective(t, text)) => {
                        v.log_push(t, format!("objective: {text}"));
                        v.objective = Some((t, text));
                        dirty = true;
                    }
                    Ok(Msg::Other(l)) => {
                        // A newer service's line: noted once per kind, not
                        // once per line (one may come at every token).
                        let head = l.split(' ').next().unwrap_or("").to_string();
                        if v.unknown.insert(head.clone()) {
                            v.notes.push(format!(
                                "the service sends {head:?} lines, which this terminal does not show"
                            ));
                            dirty = true;
                        }
                    }
                    Ok(Msg::Bye) | Err(TryRecvError::Disconnected) => {
                        gone = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                }
            }
            if gone {
                conn = None;
                // Stopped by /quit from here: this terminal ends with it.
                if v.quitting {
                    return Ok(());
                }
                v.link = Link::Gone(crate::clock::now_us());
                v.notes
                    .push("the service went away; looking for it every 3 s".into());
                last_try = Instant::now();
                dirty = true;
            }
            if event::poll(Duration::from_millis(40))? {
                match event::read()? {
                    TEvent::Key(KeyEvent {
                        code, modifiers, ..
                    }) => {
                        dirty = true;
                        match code {
                            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                                return Ok(())
                            }
                            KeyCode::Char(ch) => v.input.push(ch),
                            KeyCode::Backspace => {
                                v.input.pop();
                            }
                            KeyCode::Enter => {
                                let line = std::mem::take(&mut v.input);
                                // /quit with no service: nothing to stop; it leaves.
                                if conn.is_none() && line.trim() == "/quit" {
                                    return Ok(());
                                }
                                submit(&line, conn.as_mut().map(|c| &mut c.w), &mut v);
                                v.scroll = 0;
                            }
                            KeyCode::Esc => v.input.clear(),
                            KeyCode::Tab => {
                                v.view = v.view.next();
                                v.scroll = 0;
                            }
                            KeyCode::PageUp => v.scroll += 10,
                            KeyCode::PageDown => v.scroll = v.scroll.saturating_sub(10),
                            KeyCode::End => v.scroll = 0,
                            _ => {}
                        }
                    }
                    TEvent::Resize(_, _) => {
                        // What the terminal shows is unknown now: all of it again.
                        front = None;
                        dirty = true;
                    }
                    _ => {}
                }
            }
            tick += 1;
            if dirty || last_draw.elapsed() > Duration::from_millis(250) {
                draw(&mut out, &v, &placement, tick, &mut front)?;
                dirty = false;
                last_draw = Instant::now();
            }
        }
    })();
    execute!(
        out,
        ResetColor,
        cursor::Show,
        terminal::LeaveAlternateScreen
    )
    .ok();
    terminal::disable_raw_mode().ok();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View {
            pieces: Vec::new(),
            chars: 0,
            scroll: 0,
            input: String::new(),
            status: None,
            notes: Vec::new(),
            started: Instant::now(),
            heard: 0,
            last_t_us: 0,
            minds: VecDeque::new(),
            guide: None,
            diag: None,
            view: Pane::Feed,
            log: VecDeque::new(),
            utf8: true,
            episodes: VecDeque::new(),
            link: Link::Up,
            socket: String::new(),
            quitting: false,
            follow: Some(0),
            delib: Vec::new(),
            delib_chars: 0,
            output: Vec::new(),
            output_chars: 0,
            speaking: false,
            term: Vec::new(),
            term_chars: 0,
            objective: None,
            unknown: Default::default(),
            lens: Some(LENS_MIN),
            line_pos: Vec::new(),
            line_text: String::new(),
            struck: VecDeque::new(),
        }
    }

    #[test]
    fn compartments_fit_and_never_overlap() {
        assert!(layout(79, 24).is_none());
        assert!(layout(80, 23).is_none());
        for (w, h) in [
            (80, 24),
            (100, 30),
            (119, 40),
            (120, 24),
            (150, 36),
            (200, 60),
        ] {
            let l = layout(w, h).unwrap();
            let mut rects = vec![l.main, l.assess];
            rects.extend(l.log);
            rects.extend(l.delib);
            rects.extend(l.output);
            rects.extend(l.term);
            for (i, a) in rects.iter().enumerate() {
                assert!(a.top >= 1 && a.top + a.h <= l.mind, "{w}x{h} {a:?}");
                assert!(a.left + a.w <= w, "{w}x{h} {a:?}");
                assert!(
                    a.h >= 5 && a.w >= 13,
                    "{w}x{h} {a:?}: a row inside at least"
                );
                for b in &rects[i + 1..] {
                    assert!(!a.overlaps(*b), "{w}x{h} {a:?} {b:?}");
                }
            }
            assert_eq!(l.log.is_some(), w >= WIDE);
            assert_eq!(l.delib.is_some(), w >= WIDE);
            assert_eq!(l.output.is_some(), w >= WIDE);
            assert_eq!(l.term.is_some(), w >= WIDE);
            assert!(l.assess.h >= 6, "{w}x{h}: room for the token frame");
            assert_eq!(
                (l.mind, l.status, l.input, l.hints),
                (h - 4, h - 3, h - 2, h - 1)
            );
        }
        // 80x24: the view in rows 1 to 13, the assessment's four lines in 15 to 18.
        let l = layout(80, 24).unwrap();
        assert_eq!(
            l.main,
            Rect {
                top: 1,
                left: 0,
                h: 13,
                w: 80
            }
        );
        assert_eq!(l.assess.inner().h, 4);
    }

    #[test]
    fn wrapping_keeps_to_the_width() {
        let rows = wrap("a check on the next word: you were about to write", 20, 2);
        assert!(rows.len() > 1);
        for r in &rows {
            assert!(text_columns(r.trim_end()) <= 20, "{r:?}");
        }
        assert!(rows[1].starts_with("  "));
        // CJK words count two columns each.
        for r in wrap("而不是 而非 而不是 而非", 10, 0) {
            assert!(text_columns(r.trim_end()) <= 10, "{r:?}");
        }
    }

    #[test]
    fn a_reload_hands_the_view_over() {
        let mut a = view();
        a.view = Pane::Log;
        a.scroll = 7;
        a.heard = 3;
        a.input = "half typed\nwith a newline \\ and a backslash".into();
        let mut b = view();
        b.restore(&a.state());
        assert_eq!((b.view, b.scroll, b.heard), (Pane::Log, 7, 3));
        assert_eq!(b.input, a.input);
        assert_eq!(b.follow, Some(1));
        // An older build handed over `mind=1` for the mind view.
        let mut c = view();
        c.restore("mind=1 scroll=0 heard=0 up=5 reloads=0 input=");
        assert_eq!(c.view, Pane::Mind);
        // The lens's setting goes over too; off is 0.
        a.lens = None;
        let mut d = view();
        d.restore(&a.state());
        assert_eq!(d.lens, None);
        a.lens = Some(0.15);
        d.restore(&a.state());
        assert_eq!(d.lens, Some(0.15));
    }

    /// A reading at `pos` whose three blocks each have these words (each
    /// at this probability).
    fn reading(pos: i32, token: &str, words: &[(&str, f32)]) -> Reading {
        let ws: Vec<(String, f32)> = words.iter().map(|(w, p)| (w.to_string(), p.ln())).collect();
        Reading {
            pos,
            token: token.into(),
            layers: vec![(27, ws.clone()), (29, ws.clone()), (31, ws)],
            model_top: Vec::new(),
            band: Vec::new(),
            ms: 1.0,
            t_us: 0,
        }
    }

    #[test]
    fn the_lens_row_weighs_what_the_line_does_not_say() {
        let mut minds = VecDeque::new();
        minds.push_back(reading(
            10,
            " idle",
            &[("again", 0.3), ("idle", 0.5), ("to", 0.9)],
        ));
        minds.push_back(reading(11, ".", &[("again", 0.1), ("waiting", 0.06)]));
        // A reading of position 10 taken back and read again: the newest counts.
        minds.push_front(reading(10, " busy", &[("never", 0.9)]));
        let row = lens_row(&minds, &[10, 11], "Idle.", 0.10).unwrap();
        // again: (0.3 + 0.1) * 3 blocks / 6 = 0.20; waiting 0.03; idle is said,
        // to is too short.
        assert_eq!(row, "on its mind: again 20%");
        assert_eq!(lens_row(&minds, &[10, 11], "Idle.", 0.25), None);
        assert_eq!(lens_row(&minds, &[99], "x", 0.0), None);
        // Forms of the line's words are not unsaid, and the weak tail goes
        // (the live service, under "3. Rebuild and test": testing 16%,
        // tests 7%, using 3%, rebuilt 3%).
        let mut m = VecDeque::new();
        m.push_back(reading(
            1,
            " Rebuild",
            &[
                ("testing", 0.5),
                ("tests", 0.2),
                ("rebuilt", 0.3),
                ("verify", 0.3),
                ("using", 0.05),
            ],
        ));
        m.push_back(reading(2, " and", &[]));
        m.push_back(reading(3, " test", &[]));
        assert_eq!(
            lens_row(&m, &[1, 2, 3], "3. Rebuild and test", 0.05),
            Some("on its mind: verify 10%".to_string())
        );
    }

    #[test]
    fn a_thinking_line_gets_its_row_and_a_written_word_shows_struck() {
        let mut v = view();
        for (p, t) in [(20, " resting"), (21, " for"), (22, " now")] {
            v.minds.push_back(reading(p, t, &[("again", 0.4)]));
        }
        v.episodes.push_back(Episode {
            t_us: 5,
            pos: 22,
            why: crate::reflect::Why::Doubt,
            chosen: " later".into(),
            p_chosen: 0.1,
            flag: 0.0,
            words: Vec::new(),
            keep: 0.3,
            fmt: 0.5,
            rule: String::new(),
            top: Vec::new(),
            answer: String::new(),
            outcome: Outcome::Changed,
            to: " now".into(),
            ms: 500.0,
            placed: 3,
            back: Some((22, 25)),
        });
        v.push_stream(" resting".into(), Kind::Think, Some(20));
        v.push_stream(" for".into(), Kind::Think, Some(21));
        v.push_stream(" now".into(), Kind::Think, Some(22));
        v.push_stream(".\n".into(), Kind::Think, Some(23));
        let text: Vec<String> = v
            .rows(80)
            .iter()
            .map(|r| r.iter().map(|(s, _, _)| s.as_str()).collect())
            .collect();
        assert_eq!(text[0].trim(), "resting for ~later~ now.");
        assert_eq!(text[1], "on its mind: again 40%");
        // Off: no row; inside a code block: made, and the view leaves it out.
        v.lens = None;
        assert_eq!(v.rows(80).len(), 2);
        v.lens = Some(LENS_MIN);
        v.push_stream("```\n".into(), Kind::Think, Some(24));
        for p in 25..29 {
            v.minds.push_back(reading(p, " x", &[("again", 0.9)]));
            v.push_stream(" x".into(), Kind::Think, Some(p));
        }
        v.push_stream("\n".into(), Kind::Think, Some(29));
        let shown = |v: &View| -> Vec<String> {
            v.rows(80)
                .iter()
                .map(|r| r.iter().map(|(s, _, _)| s.as_str()).collect())
                .collect()
        };
        assert_eq!(
            shown(&v)
                .iter()
                .filter(|r| r.starts_with("on its mind"))
                .count(),
            1
        );
        // A fence inside a tool's result, which the view collapses, does
        // not leave later lines inside a block (on the live service no row
        // ever showed so).
        let mut w = view();
        w.push_stream(
            "<tool_response>\n```\n</tool_response>\n".into(),
            Kind::Given,
            None,
        );
        for (p, t) in [(40, " a"), (41, " b"), (42, " c")] {
            w.minds.push_back(reading(p, t, &[("again", 0.4)]));
            w.push_stream(t.into(), Kind::Think, Some(p));
        }
        w.push_stream("\n".into(), Kind::Think, Some(43));
        assert!(
            shown(&w).iter().any(|r| r.starts_with("on its mind")),
            "{:?}",
            shown(&w)
        );
    }
}

#[cfg(test)]
mod contrast {
    use super::*;

    /// WCAG 2 contrast of two colours.
    fn ratio(a: Color, b: Color) -> f64 {
        let lum = |c: Color| {
            let Color::Rgb { r, g, b } = c else {
                panic!("not RGB")
            };
            let ch = |v: u8| {
                let s = v as f64 / 255.0;
                if s <= 0.03928 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)
        };
        let (x, y) = (lum(a), lum(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    #[test]
    fn every_text_style_reads_at_4_5_to_1() {
        use Class::*;
        for kind in [Kind::Think, Kind::Speak, Kind::Given] {
            for class in [
                Prose, Bold, Heading, Code, Fence, Plain, Keyword, Str, Comment, Number,
            ] {
                let s = style_of(kind, class);
                let r = ratio(s.fg, s.bg);
                assert!(r >= 4.5, "{kind:?} {class:?}: {r:.2}");
            }
        }
    }
}
