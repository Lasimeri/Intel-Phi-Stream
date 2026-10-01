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

use crate::client::{escape, parse, unescape, Client, Msg};
use crate::engine::{Kind, Mode, Status};
use crate::mind::Reading;
use crate::reflect::{Episode, Outcome};
use crate::screen::{self, columns, text_columns, Glyphs, Rect, Screen, Style, Weight};
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

const KEEP_MINDS: usize = 400;
const KEEP_EPISODES: usize = 64;
const KEEP_LOG: usize = 500;
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
            "mind={} scroll={} heard={} up={} reloads={} input={}",
            match self.view {
                Pane::Feed => 0,
                Pane::Mind => 1,
                Pane::Log => 2,
            },
            self.scroll,
            self.heard,
            self.started.elapsed().as_secs(),
            self.follow.unwrap_or(0),
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

    /// The stream as rows of styled runs, wrapped to `width` by words
    /// (a word may span pieces, since a token can end inside one).
    fn rows(&self, width: usize) -> Vec<Vec<(String, Kind)>> {
        let width = width.max(8);
        // Flatten into lines of styled characters.
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
        let mut rows: Vec<Vec<(String, Kind)>> = Vec::new();
        for line in lines {
            let mut row: Vec<(char, Kind)> = Vec::new();
            // Its width in terminal columns (a CJK character takes two).
            let mut cols = 0;
            // Words: a run of non-spaces with the spaces that follow it.
            let mut i = 0;
            while i < line.len() {
                let mut j = i;
                while j < line.len() && line[j].0 != ' ' {
                    j += 1;
                }
                while j < line.len() && line[j].0 == ' ' {
                    j += 1;
                }
                let word = &line[i..j];
                let w: usize = word
                    .iter()
                    .filter(|c| c.0 != ' ')
                    .map(|c| columns(c.0))
                    .sum();
                if cols + w > width && !row.is_empty() {
                    rows.push(runs(&row));
                    row.clear();
                    cols = 0;
                }
                for &c in word {
                    if cols + columns(c.0) > width {
                        rows.push(runs(&row));
                        row.clear();
                        cols = 0;
                    }
                    if c.0 == ' ' && row.is_empty() {
                        continue;
                    }
                    row.push(c);
                    cols += columns(c.0);
                }
                i = j;
            }
            rows.push(runs(&row));
        }
        rows
    }
}

/// Characters of one row as runs of one kind.
fn runs(row: &[(char, Kind)]) -> Vec<(String, Kind)> {
    let mut out: Vec<(String, Kind)> = Vec::new();
    for &(c, k) in row {
        match out.last_mut() {
            Some((s, kind)) if *kind == k => s.push(c),
            _ => out.push((c.to_string(), k)),
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
}

impl Pane {
    fn name(self) -> &'static str {
        match self {
            Pane::Feed => "FEED",
            Pane::Mind => "MIND",
            Pane::Log => "LOG",
        }
    }

    fn next(self) -> Self {
        match self {
            Pane::Feed => Pane::Mind,
            Pane::Mind => Pane::Log,
            Pane::Log => Pane::Feed,
        }
    }
}

/// Where each compartment goes (`tui.md`), or none under the minimum.
#[derive(Debug)]
struct Layout {
    /// The view chosen (feed, mind, log), framed.
    main: Rect,
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
        Layout {
            main: Rect {
                top: 1,
                left: 0,
                h: body,
                w: left,
            },
            assess: Rect {
                top: 1,
                left,
                h: assess_h,
                w: SIDE_W,
            },
            log: Some(Rect {
                top: 1 + assess_h,
                left,
                h: body - assess_h,
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
        v.started.elapsed().as_secs() / 60,
        v.heard,
        if v.last_t_us > 0 { crate::clock::hms(v.last_t_us) } else { String::new() }
    );
    s.line(0, &title, plain(theme::DIM, theme::SURFACE));

    // The main compartment: the view chosen.
    let views = format!("{} · Tab: {}", v.view.name(), v.view.next().name());
    s.frame(lay.main, g, edge, &views, label);
    let inner = lay.main.inner();
    let end = inner.left + inner.w;
    let given = Style {
        fg: theme::GIVEN,
        bg: theme::SURFACE,
        weight: Weight::Italic,
    };
    match v.view {
        Pane::Feed => {
            let rows = v.rows(inner.w.saturating_sub(2));
            let last = rows.len().saturating_sub(v.scroll);
            let first = last.saturating_sub(inner.h);
            for r in 0..inner.h {
                let Some(row) = rows.get(first + r) else {
                    continue;
                };
                let mut col = inner.left + 1;
                for (text, kind) in row {
                    let style = match kind {
                        Kind::Think => base,
                        Kind::Speak => Style {
                            fg: theme::BRIGHT,
                            bg: theme::BG,
                            weight: Weight::Bold,
                        },
                        Kind::Given => given,
                    };
                    col = s.put_to(inner.top + r, col, end, text, style);
                }
            }
        }
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

    // The mind strip: what was on its mind at the last token it placed.
    match v.minds.back() {
        Some(r) => s.line(
            lay.mind,
            &format!(" MIND  {}   ({:.1} ms)", mind_row(r).trim_start(), r.ms),
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
        "{follows} Enter speaks · Tab views · /feed FILE · /persona FILE · /pause /resume · /chunk N · /temp T · /quit stops it · PgUp PgDn End · ^C leaves it running   {}   {note}",
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
                        };
                        dirty = true;
                    }
                    Ok(Msg::Text(t, k, at)) => {
                        v.push(t, k);
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
                    Ok(Msg::Other(l)) => {
                        v.notes.push(format!("the service said: {l}"));
                        dirty = true;
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
            view: Pane::Feed,
            log: VecDeque::new(),
            utf8: true,
            episodes: VecDeque::new(),
            link: Link::Up,
            socket: String::new(),
            quitting: false,
            follow: Some(0),
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
            for (i, a) in rects.iter().enumerate() {
                assert!(a.top >= 1 && a.top + a.h <= l.mind, "{w}x{h} {a:?}");
                assert!(a.left + a.w <= w, "{w}x{h} {a:?}");
                assert!(
                    a.h >= 6 && a.w >= 13,
                    "{w}x{h} {a:?}: room for the token frame"
                );
                for b in &rects[i + 1..] {
                    assert!(!a.overlaps(*b), "{w}x{h} {a:?} {b:?}");
                }
            }
            assert_eq!(l.log.is_some(), w >= WIDE);
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
    }
}
