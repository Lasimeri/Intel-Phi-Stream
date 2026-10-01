//! `phi-stream serve`: the service that owns the model. The engine runs
//! on its thread; this side keeps the recent text and the last status,
//! listens on a Unix socket, and for each connection reads commands and,
//! when asked, streams every event as it happens (`client.md` has the
//! lines). Several clients at once: the terminal, `say`, `tail`, a
//! script, an agent. The model stays loaded across all of them. See
//! serve.md.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context as _, Result};

use crate::client::{escape, kind_name, status_line};
use crate::engine::{Command, Event, Kind};

const KEEP_CHARS: usize = 400_000;
/// What a new `tail` is shown first.
const REPLAY_CHARS: usize = 12_000;
/// Readings of its mind kept and shown to a new `tail`.
const KEEP_MINDS: usize = 256;

struct Hub {
    recent: VecDeque<(String, Kind, i64)>,
    chars: usize,
    last_status: Option<String>,
    subs: Vec<Sender<String>>,
    /// The last readings of its mind, as lines.
    minds: VecDeque<String>,
}

impl Hub {
    fn push_text(&mut self, text: &str, kind: Kind, t_us: i64) {
        self.chars += text.chars().count();
        self.recent.push_back((text.to_string(), kind, t_us));
        while self.chars > KEEP_CHARS && self.recent.len() > 1 {
            let (t, _, _) = self.recent.pop_front().unwrap();
            self.chars -= t.chars().count();
        }
    }

    fn broadcast(&mut self, line: &str) {
        self.subs.retain(|s| s.send(line.to_string()).is_ok());
    }

    /// The last `REPLAY_CHARS` of text as lines, then the last status.
    fn replay(&self) -> Vec<String> {
        let mut n = 0;
        let mut start = self.recent.len();
        while start > 0 && n < REPLAY_CHARS {
            start -= 1;
            n += self.recent[start].0.chars().count();
        }
        let mut out: Vec<String> = self
            .recent
            .iter()
            .skip(start)
            .map(|(t, k, at)| format!("text {} t={at} {}", kind_name(*k), escape(t)))
            .collect();
        out.extend(self.minds.iter().cloned());
        if let Some(s) = &self.last_status {
            out.push(s.clone());
        }
        out
    }
}

/// Run the service until the engine stops. `info` is the line every
/// connection gets first.
pub fn serve(
    erx: Receiver<Event>,
    ctx: Sender<Command>,
    socket: &Path,
    info: String,
) -> Result<()> {
    if socket.exists() {
        // A socket nobody answers on is stale; one that answers is taken.
        if UnixStream::connect(socket).is_ok() {
            anyhow::bail!(
                "a phi-stream service is already listening at {}",
                socket.display()
            );
        }
        fs::remove_file(socket).ok();
    }
    if let Some(d) = socket.parent() {
        fs::create_dir_all(d).ok();
    }
    let listener =
        UnixListener::bind(socket).with_context(|| format!("listening at {}", socket.display()))?;
    listener.set_nonblocking(true)?;
    let hub = Arc::new(Mutex::new(Hub {
        recent: VecDeque::new(),
        chars: 0,
        last_status: None,
        subs: Vec::new(),
        minds: VecDeque::new(),
    }));
    let stopped = Arc::new(AtomicBool::new(false));

    // The engine's events: kept, and sent on to every tail.
    {
        let hub = hub.clone();
        let stopped = stopped.clone();
        thread::spawn(move || {
            while let Ok(ev) = erx.recv() {
                let mut h = hub.lock().unwrap();
                match ev {
                    Event::Text(t, k, at) => {
                        h.push_text(&t, k, at);
                        let line = format!("text {} t={at} {}", kind_name(k), escape(&t));
                        h.broadcast(&line);
                    }
                    Event::Status(s) => {
                        let line = status_line(&s);
                        h.last_status = Some(line.clone());
                        h.broadcast(&line);
                    }
                    Event::Note(n) => {
                        let line = format!("note {}", escape(&n));
                        h.broadcast(&line);
                    }
                    Event::Mind(r) => {
                        let line = format!("mind {}", crate::mind::line(&r));
                        h.minds.push_back(line.clone());
                        while h.minds.len() > KEEP_MINDS {
                            h.minds.pop_front();
                        }
                        h.broadcast(&line);
                    }
                    Event::Reflect(e) => {
                        let line = format!("reflect {}", crate::reflect::line(&e));
                        h.broadcast(&line);
                    }
                    Event::Done { .. } => {}
                    Event::Stopped => {
                        h.broadcast("bye");
                        break;
                    }
                }
            }
            stopped.store(true, Ordering::SeqCst);
        });
    }

    eprintln!("phi-stream: listening at {}", socket.display());
    while !stopped.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let hub = hub.clone();
                let ctx = ctx.clone();
                let info = info.clone();
                let stopped = stopped.clone();
                thread::spawn(move || {
                    if let Err(e) = connection(stream, hub, ctx, info, stopped) {
                        eprintln!("phi-stream: a connection ended: {e}");
                    }
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50))
            }
            Err(e) => {
                eprintln!("phi-stream: accept: {e}");
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
    fs::remove_file(socket).ok();
    Ok(())
}

/// One client: commands in, replies out, and a tail when asked.
fn connection(
    stream: UnixStream,
    hub: Arc<Mutex<Hub>>,
    ctx: Sender<Command>,
    info: String,
    stopped: Arc<AtomicBool>,
) -> Result<()> {
    let mut w = stream.try_clone()?;
    let reader = BufReader::new(stream);
    writeln!(w, "{info}")?;
    let mut tailing = false;
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if stopped.load(Ordering::SeqCst) {
            writeln!(w, "bye")?;
            break;
        }
        let (cmd, arg) = line.split_once(' ').unwrap_or((line, ""));
        let arg = arg.trim();
        let reply = match cmd {
            "say" => {
                if arg.is_empty() {
                    Err("say what?".to_string())
                } else {
                    ctx.send(Command::Say(crate::client::unescape(arg), crate::clock::now_us())).map(|_| "heard".to_string()).map_err(|_| "the engine is gone".to_string())
                }
            }
            "say-as" => match arg.split_once(' ') {
                Some((who, text)) if !text.trim().is_empty() => ctx
                    .send(Command::SayAs(who.to_string(), crate::client::unescape(text.trim()), crate::clock::now_us()))
                    .map(|_| format!("heard {who}"))
                    .map_err(|_| "the engine is gone".to_string()),
                _ => Err("say-as NAME TEXT".to_string()),
            },
            "feed" => match read_file(arg) {
                Ok((text, label)) => ctx
                    .send(Command::Feed(text, label.clone(), crate::clock::now_us()))
                    .map(|_| format!("handed over {label}"))
                    .map_err(|_| "the engine is gone".to_string()),
                Err(e) => Err(e),
            },
            "persona" => match read_file(arg) {
                Ok((text, label)) => ctx
                    .send(Command::Persona(text))
                    .map(|_| format!("persona from {label}; the context rolls over onto it"))
                    .map_err(|_| "the engine is gone".to_string()),
                Err(e) => Err(e),
            },
            "chunk" => match arg.parse::<usize>() {
                Ok(n) => ctx.send(Command::Chunk(n)).map(|_| format!("chunk {n}")).map_err(|_| "the engine is gone".to_string()),
                Err(_) => Err("chunk takes a number (0 adapts)".to_string()),
            },
            "temp" => match arg.parse::<f32>() {
                Ok(t) => ctx.send(Command::Temp(t)).map(|_| format!("temperature {t}")).map_err(|_| "the engine is gone".to_string()),
                Err(_) => Err("temp takes a number".to_string()),
            },
            "pause" => ctx.send(Command::Pause).map(|_| "paused".to_string()).map_err(|_| "the engine is gone".to_string()),
            "resume" => ctx.send(Command::Resume).map(|_| "resumed".to_string()).map_err(|_| "the engine is gone".to_string()),
            "status" => {
                let last = hub.lock().unwrap().last_status.clone();
                ctx.send(Command::Status).ok();
                match last {
                    Some(s) => {
                        writeln!(w, "{s}")?;
                        Ok("status".to_string())
                    }
                    None => Ok("no status yet".to_string()),
                }
            }
            "recent" => {
                let lines = hub.lock().unwrap().replay();
                for l in lines {
                    writeln!(w, "{l}")?;
                }
                Ok("recent".to_string())
            }
            "tail" => {
                if !tailing {
                    tailing = true;
                    let (tx, rx) = mpsc::channel::<String>();
                    let lines = {
                        let mut h = hub.lock().unwrap();
                        let lines = h.replay();
                        h.subs.push(tx);
                        lines
                    };
                    for l in lines {
                        writeln!(w, "{l}")?;
                    }
                    let mut w2 = w.try_clone()?;
                    thread::spawn(move || {
                        for l in rx {
                            if writeln!(w2, "{l}").is_err() {
                                break;
                            }
                        }
                    });
                }
                Ok("tailing".to_string())
            }
            "quit" => {
                ctx.send(Command::Quit).ok();
                Ok("stopping".to_string())
            }
            _ => Err(format!("unknown command {cmd}; say, feed, persona, chunk, temp, pause, resume, status, recent, tail, quit")),
        };
        match reply {
            Ok(m) => writeln!(w, "ok {m}")?,
            Err(m) => writeln!(w, "err {m}")?,
        }
    }
    Ok(())
}

/// A file named by a client, read here (the service's view of the
/// file system is the user's own): its text and a label.
fn read_file(arg: &str) -> std::result::Result<(String, String), String> {
    if arg.is_empty() {
        return Err("which file?".to_string());
    }
    let p = match arg.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(arg),
    };
    let text = fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let label = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| arg.to_string());
    Ok((text, format!("the file {label}")))
}
