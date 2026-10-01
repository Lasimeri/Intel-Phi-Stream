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
    Text(String, Kind),
    Status(Status),
    Note(String),
    /// What was on its mind at one token (`mind.rs`).
    Mind(crate::mind::Reading),
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
            let (kind, text) = rest.split_once(' ').unwrap_or((rest, ""));
            let kind = match kind {
                "speak" => Kind::Speak,
                "given" => Kind::Given,
                _ => Kind::Think,
            };
            Msg::Text(unescape(text), kind)
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
            })
        }
        "note" => Msg::Note(unescape(rest)),
        "mind" => match crate::mind::parse_line(rest) {
            Some(r) => Msg::Mind(r),
            None => Msg::Other(line.to_string()),
        },
        "ok" => Msg::Ok(rest.to_string()),
        "err" => Msg::Err(rest.to_string()),
        "bye" => Msg::Bye,
        _ => Msg::Other(line.to_string()),
    }
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
        "status mode={mode} stream={:.2} beside={:.2} cycle={:.1} pos={} ctx={} queued={} chunk={} rollovers={} notes={} frame={} leaks={} mind_ms={:.2}",
        s.stream_tps, s.side_tps, s.cycle_ms, s.pos, s.n_ctx, s.queued, s.chunk, s.rollovers, s.notes, s.frame, s.leaks, s.mind_ms
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
        };
        match parse(&status_line(&st)) {
            Msg::Status(s) => {
                assert!(matches!(s.mode, Mode::Reading { done: 3, total: 10 }));
                assert_eq!(s.pos, 1234);
                assert_eq!(s.notes, 4);
                assert_eq!(s.frame, "journal");
            }
            other => panic!("{other:?}"),
        }
        match parse("text speak hello\\nthere") {
            Msg::Text(t, Kind::Speak) => assert_eq!(t, "hello\nthere"),
            other => panic!("{other:?}"),
        }
    }
}
