//! `phi-stream tui`: the stream as something alive in the terminal. The
//! thoughts flow in the middle; what it says aloud stands out; what you
//! type is heard where the stream is when you press Enter and appears
//! there; a strip shows what it is doing (thinking, reading what you
//! gave it, catching up, summarizing to roll its context over) with its
//! rates and how full its context is. crossterm only, the family's
//! palette. See tui.md.

use std::io::{self, BufRead, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::mpsc::{self, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event as TEvent, KeyCode, KeyEvent, KeyModifiers};
use crossterm::style::{
    Attribute, Print, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor,
};
use crossterm::{cursor, execute, queue, terminal};

use crate::client::{escape, parse, Client, Msg};
use crate::engine::{Kind, Mode, Status};
use crate::mind::Reading;
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
    /// The last readings of its mind, newest last.
    minds: VecDeque<Reading>,
    /// The main area shows the readings token by token instead of the stream.
    mind_view: bool,
}

const KEEP_MINDS: usize = 400;

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
    format!("{:>14}  {}", shown(&r.token), blocks.join("  ·  "))
}

const KEEP_CHARS: usize = 400_000;

impl View {
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
                let w = word.iter().filter(|c| c.0 != ' ').count();
                if row.len() + w > width && !row.is_empty() {
                    rows.push(runs(&row));
                    row.clear();
                }
                for &c in word {
                    if row.len() >= width {
                        rows.push(runs(&row));
                        row.clear();
                    }
                    if c.0 == ' ' && row.is_empty() {
                        continue;
                    }
                    row.push(c);
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
    );
    (m, rates)
}

fn bar(done: usize, total: usize, cells: usize) -> String {
    let filled = (done * cells + total / 2).checked_div(total).unwrap_or(0);
    let filled = filled.min(cells);
    format!("{}{}", "▇".repeat(filled), "▁".repeat(cells - filled))
}

fn pad(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.chars().take(width).collect()
    } else {
        format!("{s}{}", " ".repeat(width - n))
    }
}

fn draw(out: &mut impl Write, v: &View, p: &Placement, tick: u64) -> io::Result<()> {
    let (w, h) = terminal::size()?;
    let (w, h) = (w as usize, h as usize);
    if h < 6 {
        return Ok(());
    }
    queue!(out, cursor::Hide, SetBackgroundColor(theme::BG))?;
    // Title.
    let title = format!(
        " phi-stream · {} · {} · GPU {}/{} blocks {:.1} GiB · cards+host {:.1} GiB · {}k cells · up {}m · heard {}",
        p.model,
        p.frame,
        p.gpu_blocks,
        p.n_blocks,
        p.gpu_gib,
        p.host_gib,
        p.n_ctx / 1000,
        v.started.elapsed().as_secs() / 60,
        v.heard
    );
    queue!(
        out,
        cursor::MoveTo(0, 0),
        SetBackgroundColor(theme::SURFACE),
        SetForegroundColor(theme::DIM),
        Print(pad(&title, w))
    )?;
    // The stream.
    let has_mind = !v.minds.is_empty();
    let body_h = h - 4 - has_mind as usize;
    if v.mind_view {
        // The readings, token by token, newest at the bottom.
        let end = v.minds.len().saturating_sub(v.scroll);
        let start = end.saturating_sub(body_h);
        for r in 0..body_h {
            let text = v.minds.get(start + r).map(mind_row).unwrap_or_default();
            queue!(
                out,
                cursor::MoveTo(0, (1 + r) as u16),
                SetAttribute(Attribute::Reset),
                SetBackgroundColor(theme::BG),
                SetForegroundColor(theme::GIVEN),
                Print(pad(&format!(" {text}"), w))
            )?;
        }
    }
    let rows = if v.mind_view {
        Vec::new()
    } else {
        v.rows(w.saturating_sub(2))
    };
    let end = rows.len().saturating_sub(v.scroll);
    let start = end.saturating_sub(body_h);
    for r in 0..if v.mind_view { 0 } else { body_h } {
        queue!(
            out,
            cursor::MoveTo(0, (1 + r) as u16),
            SetBackgroundColor(theme::BG),
            ResetColor,
            SetBackgroundColor(theme::BG)
        )?;
        let mut col = 0usize;
        if let Some(row) = rows.get(start + r) {
            queue!(out, Print(" "))?;
            col += 1;
            for (text, kind) in row {
                match kind {
                    Kind::Think => queue!(
                        out,
                        SetAttribute(Attribute::Reset),
                        SetBackgroundColor(theme::BG),
                        SetForegroundColor(theme::TEXT)
                    )?,
                    Kind::Speak => queue!(
                        out,
                        SetAttribute(Attribute::Bold),
                        SetBackgroundColor(theme::BG),
                        SetForegroundColor(theme::BRIGHT)
                    )?,
                    Kind::Given => queue!(
                        out,
                        SetAttribute(Attribute::Italic),
                        SetBackgroundColor(theme::SURFACE),
                        SetForegroundColor(theme::GIVEN)
                    )?,
                }
                queue!(out, Print(text))?;
                col += text.chars().count();
            }
        }
        queue!(
            out,
            SetAttribute(Attribute::Reset),
            SetBackgroundColor(theme::BG),
            Print(" ".repeat(w.saturating_sub(col)))
        )?;
    }
    // The mind strip: what was on its mind at the last token it placed.
    if let Some(r) = v.minds.back() {
        let strip = format!(" mind  {}   ({:.1} ms)", mind_row(r).trim_start(), r.ms);
        queue!(
            out,
            cursor::MoveTo(0, (h - 4) as u16),
            SetAttribute(Attribute::Reset),
            SetBackgroundColor(theme::SURFACE),
            SetForegroundColor(theme::GIVEN),
            Print(pad(&strip, w))
        )?;
    }
    // The strip: what it is doing, and the rates.
    let (mode, rates) = match &v.status {
        Some(s) => mode_line(s, tick),
        None => ("· waking".to_string(), String::new()),
    };
    let note = v.notes.last().cloned().unwrap_or_default();
    let strip = format!(" {mode}   {rates}");
    queue!(
        out,
        cursor::MoveTo(0, (h - 3) as u16),
        SetBackgroundColor(theme::SURFACE),
        SetForegroundColor(theme::TEXT),
        Print(pad(&strip, w))
    )?;
    // Input.
    let prompt = format!(" › {}", v.input);
    queue!(
        out,
        cursor::MoveTo(0, (h - 2) as u16),
        SetBackgroundColor(theme::BG),
        SetForegroundColor(theme::BRIGHT),
        Print(pad(&prompt, w))
    )?;
    // Hints and the last note.
    let hints = format!(
        " Enter speaks · /feed FILE · /persona FILE · /mind · /pause /resume · /chunk N · /temp T · /quit stops it · PgUp PgDn End · ^C leaves it running   {}   {note}",
        p.workspace
    );
    queue!(
        out,
        cursor::MoveTo(0, (h - 1) as u16),
        SetBackgroundColor(theme::BG),
        SetForegroundColor(theme::ACCENT_DIM),
        Print(pad(&hints, w))
    )?;
    let cx = (3 + v.input.chars().count()).min(w - 1) as u16;
    queue!(out, cursor::MoveTo(cx, (h - 2) as u16), cursor::Show)?;
    out.flush()
}

/// A typed line: a command, or something said; sent to the service.
fn submit(line: &str, w: &mut UnixStream, v: &mut View) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
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
    } else if line == "/mind" {
        v.mind_view = !v.mind_view;
        v.scroll = 0;
        v.notes.push(if v.mind_view {
            "the readings of its mind, token by token (/mind again: the stream)".into()
        } else {
            "the stream".into()
        });
        return;
    } else if line == "/quit" {
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

/// The terminal, as a client of the service at `socket`.
pub fn run(socket: &Path) -> Result<()> {
    let mut c = Client::connect(socket)?;
    c.send("tail")?;
    let (reader, mut w) = c.split();
    // The service's lines, read on a thread of their own.
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
    execute!(
        out,
        terminal::EnterAlternateScreen,
        terminal::Clear(terminal::ClearType::All)
    )?;
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
            minds: VecDeque::new(),
            mind_view: false,
        };
        let mut tick = 0u64;
        let mut dirty = true;
        let mut last_draw = Instant::now();
        loop {
            // The service's messages, all that are waiting.
            loop {
                match rx.try_recv() {
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
                    Ok(Msg::Text(t, k)) => {
                        v.push(t, k);
                        dirty = true;
                    }
                    Ok(Msg::Status(s)) => {
                        v.status = Some(s);
                        dirty = true;
                    }
                    Ok(Msg::Note(n)) => {
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
                    Ok(Msg::Err(e)) => {
                        v.notes.push(e);
                        dirty = true;
                    }
                    Ok(Msg::Ok(_)) => {}
                    Ok(Msg::Other(l)) => {
                        v.notes.push(format!("the service said: {l}"));
                        dirty = true;
                    }
                    Ok(Msg::Bye) => return Ok(()),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
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
                                submit(&line, &mut w, &mut v);
                                v.scroll = 0;
                            }
                            KeyCode::Esc => v.input.clear(),
                            KeyCode::PageUp => v.scroll += 10,
                            KeyCode::PageDown => v.scroll = v.scroll.saturating_sub(10),
                            KeyCode::End => v.scroll = 0,
                            _ => {}
                        }
                    }
                    TEvent::Resize(_, _) => dirty = true,
                    _ => {}
                }
            }
            tick += 1;
            if dirty || last_draw.elapsed() > Duration::from_millis(250) {
                draw(&mut out, &v, &placement, tick)?;
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
