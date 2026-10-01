//! The wire between the service (`serve.rs`) and whoever talks to it:
//! one Unix socket, lines of UTF-8 each way, newlines inside a text
//! escaped. A client sends commands (`say`, `feed`, `tail`, `status`,
//! `chunk`, `temp`, `persona`, `pause`, `resume`, `quit`); the service
//! answers `ok` or `err`, sends `info` once on connect, and to a client
//! that said `tail` every `text`, `status` and `note` as they happen,
//! `bye` when it stops. See client.md.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};

use crate::engine::{Kind, Mode, Status};

/// Where the socket is: `PHI_STREAM_SOCKET`, else under `XDG_RUNTIME_DIR`,
/// else `/tmp/phi-stream-<uid>.sock`.
pub fn default_socket() -> PathBuf {
    if let Ok(p) = std::env::var("PHI_STREAM_SOCKET") {
        return PathBuf::from(p);
    }
    if let Ok(d) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(d).join("phi-stream.sock");
    }
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/phi-stream-{uid}.sock"))
}

/// A text as one line: backslashes, newlines and returns escaped.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// What the service holds, as its `info` line says.
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub model: String,
    pub gpu_blocks: usize,
    pub n_blocks: usize,
    pub gpu_gib: f64,
    pub host_gib: f64,
    pub n_ctx: u32,
    pub frame: String,
    pub workspace: String,
}

/// A line from the service, parsed.
#[derive(Debug)]
pub enum Msg {
    Info(Info),
    /// A piece of the stream, its kind and the real time it exists at (us).
    Text(String, Kind, i64),
    Status(Status),
    Note(String),
    /// What was on its mind at one token (`mind.rs`).
    Mind(crate::mind::Reading),
    /// A check of a token, start to end (`reflect.rs`).
    Reflect(crate::reflect::Episode),
    /// The deliberation's own text, as it is written beside the stream:
    /// a check's start (the question it was asked), each piece of its
    /// reasoning, its end (the outcome).
    Delib(Delib),
    /// What the stream is working toward, when it changes: its real time
    /// and text.
    Objective(i64, String),
    /// Its terminal: a command began, or ended with its output.
    Term(TermLine),
    Ok(String),
    Err(String),
    Bye,
    Other(String),
}

fn fields(s: &str) -> Vec<(String, String)> {
    s.split_whitespace()
        .filter_map(|kv| {
            kv.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect()
}

fn field<'a>(f: &'a [(String, String)], k: &str) -> &'a str {
    f.iter()
        .find(|(key, _)| key == k)
        .map(|(_, v)| v.as_str())
        .unwrap_or("")
}

pub fn parse(line: &str) -> Msg {
    let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
    match head {
        "info" => {
            let f = fields(rest);
            Msg::Info(Info {
                model: unescape(field(&f, "model")),
                gpu_blocks: field(&f, "gpu_blocks").parse().unwrap_or(0),
                n_blocks: field(&f, "n_blocks").parse().unwrap_or(0),
                gpu_gib: field(&f, "gpu_gib").parse().unwrap_or(0.0),
                host_gib: field(&f, "host_gib").parse().unwrap_or(0.0),
                n_ctx: field(&f, "n_ctx").parse().unwrap_or(0),
                frame: field(&f, "frame").to_string(),
                workspace: unescape(field(&f, "workspace")),
            })
        }
        "text" => {
            // text KIND t=MICROSECONDS TEXT
            let (kind, rest) = rest.split_once(' ').unwrap_or((rest, ""));
            let (t, text) = match rest.strip_prefix("t=").and_then(|r| r.split_once(' ')) {
                Some((t, text)) => (t.parse().unwrap_or(0), text),
                None => (0, rest),
            };
            let kind = match kind {
                "speak" => Kind::Speak,
                "given" => Kind::Given,
                _ => Kind::Think,
            };
            Msg::Text(unescape(text), kind, t)
        }
        "status" => {
            let f = fields(rest);
            let mode = field(&f, "mode");
            let pair = |s: &str| -> (usize, usize) {
                s.split_once('/')
                    .map(|(a, b)| (a.parse().unwrap_or(0), b.parse().unwrap_or(0)))
                    .unwrap_or((0, 0))
            };
            let mode = if let Some(p) = mode.strip_prefix("reading:") {
                let (done, total) = pair(p);
                Mode::Reading { done, total }
            } else if let Some(p) = mode.strip_prefix("catching:") {
                let (done, total) = pair(p);
                Mode::CatchingUp { done, total }
            } else if let Some(p) = mode.strip_prefix("summarizing:") {
                Mode::Summarizing {
                    tokens: p.parse().unwrap_or(0),
                }
            } else {
                match mode {
                    "speaking" => Mode::Speaking,
                    "paused" => Mode::Paused,
                    _ => Mode::Thinking,
                }
            };
            Msg::Status(Status {
                mode,
                stream_tps: field(&f, "stream").parse().unwrap_or(0.0),
                side_tps: field(&f, "beside").parse().unwrap_or(0.0),
                cycle_ms: field(&f, "cycle").parse().unwrap_or(0.0),
                pos: field(&f, "pos").parse().unwrap_or(0),
                n_ctx: field(&f, "ctx").parse().unwrap_or(0),
                queued: field(&f, "queued").parse().unwrap_or(0),
                chunk: field(&f, "chunk").parse().unwrap_or(0),
                rollovers: field(&f, "rollovers").parse().unwrap_or(0),
                notes: field(&f, "notes").parse().unwrap_or(0),
                frame: if field(&f, "frame") == "chat" {
                    "chat"
                } else {
                    "journal"
                },
                leaks: field(&f, "leaks").parse().unwrap_or(0),
                mind_ms: field(&f, "mind_ms").parse().unwrap_or(0.0),
                t_us: field(&f, "t").parse().unwrap_or(0),
                reads_quiet: field(&f, "reads_quiet").parse().unwrap_or(0),
                checks: field(&f, "checks").parse().unwrap_or(0),
                changes: field(&f, "changes").parse().unwrap_or(0),
                unparsed: field(&f, "unparsed").parse().unwrap_or(0),
                checking: field(&f, "checking") == "1",
            })
        }
        "note" => Msg::Note(unescape(rest)),
        "mind" => match crate::mind::parse_line(rest) {
            Some(r) => Msg::Mind(r),
            None => Msg::Other(line.to_string()),
        },
        "reflect" => match crate::reflect::parse_line(rest) {
            Some(e) => Msg::Reflect(e),
            None => Msg::Other(line.to_string()),
        },
        // delib start|piece|end t=US pos=P TEXT
        "delib" => {
            let (kind, rest) = rest.split_once(' ').unwrap_or((rest, ""));
            let f: Vec<&str> = rest.splitn(3, ' ').collect();
            let num = |i: usize, k: &str| -> i64 {
                f.get(i)
                    .and_then(|x| x.strip_prefix(k))
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0)
            };
            let kind = match kind {
                "start" => DelibKind::Start,
                "piece" => DelibKind::Piece,
                "end" => DelibKind::End,
                _ => return Msg::Other(line.to_string()),
            };
            Msg::Delib(Delib {
                kind,
                t_us: num(0, "t="),
                pos: num(1, "pos=") as i32,
                text: unescape(f.get(2).copied().unwrap_or("")),
            })
        }
        // term start t=US id=N COMMAND, or term end t=US id=N code=C ms=M cut=0|1 timeout=0|1 OUTPUT
        "term" => match parse_term(rest) {
            Some(t) => Msg::Term(t),
            None => Msg::Other(line.to_string()),
        },
        // objective t=US TEXT
        "objective" => match rest.strip_prefix("t=").and_then(|r| r.split_once(' ')) {
            Some((t, text)) => Msg::Objective(t.parse().unwrap_or(0), unescape(text)),
            None => Msg::Other(line.to_string()),
        },
        "ok" => Msg::Ok(rest.to_string()),
        "err" => Msg::Err(rest.to_string()),
        "bye" => Msg::Bye,
        _ => Msg::Other(line.to_string()),
    }
}

/// One `term` line: a command began (`end` false), or ended.
#[derive(Clone, Debug, PartialEq)]
pub struct TermLine {
    pub end: bool,
    pub t_us: i64,
    pub id: u64,
    /// The exit code, none when it was stopped or did not run.
    pub code: Option<i32>,
    pub ms: f32,
    pub cut: bool,
    pub timed_out: bool,
    /// The command (start), or its output (end).
    pub text: String,
}

/// A command's end as the service sends it.
pub fn term_end_line(t_us: i64, r: &crate::term::Ran) -> String {
    format!(
        "term end t={t_us} id={} code={} ms={:.0} cut={} timeout={} {}",
        r.id,
        r.code.map_or("-".to_string(), |c| c.to_string()),
        r.ms,
        r.cut as u8,
        r.timed_out as u8,
        escape(&r.out)
    )
}

fn parse_term(rest: &str) -> Option<TermLine> {
    let (kind, rest) = rest.split_once(' ')?;
    let end = match kind {
        "start" => false,
        "end" => true,
        _ => return None,
    };
    let n = if end { 6 } else { 2 };
    let f: Vec<&str> = rest.splitn(n + 1, ' ').collect();
    let get = |k: &str| f.iter().take(n).find_map(|x| x.strip_prefix(k));
    Some(TermLine {
        end,
        t_us: get("t=")?.parse().ok()?,
        id: get("id=")?.parse().ok()?,
        code: get("code=").and_then(|c| c.parse().ok()),
        ms: get("ms=").and_then(|m| m.parse().ok()).unwrap_or(0.0),
        cut: get("cut=") == Some("1"),
        timed_out: get("timeout=") == Some("1"),
        text: unescape(f.get(n).copied().unwrap_or("")),
    })
}

/// Which part of a deliberation a `delib` line carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelibKind {
    Start,
    Piece,
    End,
}

/// One `delib` line.
#[derive(Clone, Debug, PartialEq)]
pub struct Delib {
    pub kind: DelibKind,
    pub t_us: i64,
    pub pos: i32,
    pub text: String,
}

/// A `delib` line as the service sends it.
// Sent by the engine's next commit (the reasoning check); the terminal
// reads these lines first, so a reloaded terminal shows them on the
// service's restart.
#[allow(dead_code)]
pub fn delib_line(d: &Delib) -> String {
    let kind = match d.kind {
        DelibKind::Start => "start",
        DelibKind::Piece => "piece",
        DelibKind::End => "end",
    };
    format!(
        "delib {kind} t={} pos={} {}",
        d.t_us,
        d.pos,
        escape(&d.text)
    )
}

/// `Status` as the service sends it.
pub fn status_line(s: &Status) -> String {
    let mode = match &s.mode {
        Mode::Thinking => "thinking".to_string(),
        Mode::Speaking => "speaking".to_string(),
        Mode::Reading { done, total } => format!("reading:{done}/{total}"),
        Mode::CatchingUp { done, total } => format!("catching:{done}/{total}"),
        Mode::Summarizing { tokens } => format!("summarizing:{tokens}"),
        Mode::Paused => "paused".to_string(),
    };
    format!(
        "status mode={mode} stream={:.2} beside={:.2} cycle={:.1} pos={} ctx={} queued={} chunk={} rollovers={} notes={} frame={} leaks={} mind_ms={:.2} t={} reads_quiet={} checks={} changes={} unparsed={} checking={}",
        s.stream_tps, s.side_tps, s.cycle_ms, s.pos, s.n_ctx, s.queued, s.chunk, s.rollovers, s.notes, s.frame, s.leaks, s.mind_ms, s.t_us, s.reads_quiet, s.checks, s.changes, s.unparsed, u8::from(s.checking)
    )
}

pub fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Think => "think",
        Kind::Speak => "speak",
        Kind::Given => "given",
    }
}

/// A connection to the service.
pub struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Client {
    pub fn connect(socket: &Path) -> Result<Self> {
        let s = UnixStream::connect(socket).with_context(|| {
            format!(
                "no phi-stream service at {} (start one: scripts/phi-stream.sh start)",
                socket.display()
            )
        })?;
        let writer = s.try_clone()?;
        Ok(Self {
            reader: BufReader::new(s),
            writer,
        })
    }

    pub fn send(&mut self, line: &str) -> Result<()> {
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }

    /// The next line, `None` when the service closed the connection.
    pub fn line(&mut self) -> Result<Option<String>> {
        let mut s = String::new();
        let n = self.reader.read_line(&mut s)?;
        if n == 0 {
            return Ok(None);
        }
        while s.ends_with('\n') || s.ends_with('\r') {
            s.pop();
        }
        Ok(Some(s))
    }

    /// The reading half, for a thread of its own.
    pub fn split(self) -> (BufReader<UnixStream>, UnixStream) {
        (self.reader, self.writer)
    }

    /// Send a command and wait for its `ok` or `err` (the `info` line and
    /// anything else before it is skipped).
    pub fn ask(&mut self, line: &str) -> Result<String> {
        self.send(line)?;
        loop {
            match self.line()? {
                None => bail!("the service closed the connection"),
                Some(l) => match parse(&l) {
                    Msg::Ok(m) => return Ok(m),
                    Msg::Err(m) => bail!("{m}"),
                    Msg::Bye => bail!("the service stopped"),
                    _ => continue,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping_round_trips() {
        let s = "a\\b\nc\rd";
        assert_eq!(unescape(&escape(s)), s);
        assert_eq!(escape("x\ny"), "x\\ny");
    }

    #[test]
    fn status_lines_parse() {
        let st = Status {
            mode: Mode::Reading { done: 3, total: 10 },
            stream_tps: 7.5,
            side_tps: 120.0,
            cycle_ms: 130.0,
            pos: 1234,
            n_ctx: 32768,
            queued: 1,
            chunk: 0,
            rollovers: 2,
            notes: 4,
            frame: "journal",
            leaks: 0,
            mind_ms: 0.0,
            t_us: 0,
            reads_quiet: 0,
            checks: 12,
            changes: 3,
            unparsed: 1,
            checking: true,
        };
        match parse(&status_line(&st)) {
            Msg::Status(s) => {
                assert!(matches!(s.mode, Mode::Reading { done: 3, total: 10 }));
                assert_eq!(s.pos, 1234);
                assert_eq!(s.notes, 4);
                assert_eq!(s.frame, "journal");
                assert_eq!(
                    (s.checks, s.changes, s.unparsed, s.checking),
                    (12, 3, 1, true)
                );
            }
            other => panic!("{other:?}"),
        }
        match parse("text speak t=1790000000000001 hello\\nthere") {
            Msg::Text(t, Kind::Speak, at) => {
                assert_eq!(t, "hello\nthere");
                assert_eq!(at, 1_790_000_000_000_001);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn delib_and_objective_lines_round_trip() {
        for (kind, text) in [
            (DelibKind::Start, "a check on the next word:\n\"x\""),
            (DelibKind::Piece, " the"),
            (DelibKind::Piece, ""),
            (DelibKind::End, "kept"),
        ] {
            let d = Delib {
                kind,
                t_us: 1790893948209724,
                pos: 4011,
                text: text.to_string(),
            };
            match parse(&delib_line(&d)) {
                Msg::Delib(p) => assert_eq!(p, d),
                other => panic!("{other:?}"),
            }
        }
        match parse("objective t=17 finish the read fallback") {
            Msg::Objective(17, t) => assert_eq!(t, "finish the read fallback"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(parse("delib sideways t=1 pos=2 x"), Msg::Other(_)));
        let r = crate::term::Ran {
            id: 3,
            command: "ls".into(),
            code: Some(0),
            out: "a b\nc".into(),
            cut: false,
            timed_out: false,
            ms: 12.0,
        };
        match parse(&term_end_line(7, &r)) {
            Msg::Term(t) => {
                assert!(t.end && t.id == 3 && t.code == Some(0) && t.t_us == 7);
                assert_eq!(t.text, "a b\nc");
            }
            other => panic!("{other:?}"),
        }
        match parse("term start t=5 id=4 grep -n x src/a.rs") {
            Msg::Term(t) => assert!(!t.end && t.id == 4 && t.text == "grep -n x src/a.rs"),
            other => panic!("{other:?}"),
        }
    }
}
