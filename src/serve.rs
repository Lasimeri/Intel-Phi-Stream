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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context as _, Result};

use crate::client::{escape, status_line};
use crate::engine::{Command, Event, Kind};

/// The id of the next message from Claude that waits for an answer
/// (`ask`: c1, c2, ...), and the file it is kept in (`asks_from`).
static ASKS: AtomicU64 = AtomicU64::new(1);
static ASKS_FILE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Ids of asks go on from the last service's (kept in `path`): begun at c1
/// at each restart, a new c1 met the old c1 in the stream's summary, and it
/// answered the new one with a review the old one had asked for.
pub fn asks_from(path: PathBuf) {
    let next = fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(1)
        .max(1);
    ASKS.store(next, Ordering::SeqCst);
    let _ = ASKS_FILE.set(path);
}

const KEEP_CHARS: usize = 400_000;
/// What a new `tail` is shown first.
const REPLAY_CHARS: usize = 12_000;
/// Readings of its mind kept and shown to a new `tail` (about the last
/// screen of replayed text, whose lens rows they make, `tui.md`).
const KEEP_MINDS: usize = 1024;
/// Lines of its terminal replayed to a new tail.
const KEEP_TERM: usize = 64;
/// Lines of the second chain replayed to a new tail.
const KEEP_DELIB: usize = 400;

struct Hub {
    recent: VecDeque<(String, Kind, i64, Option<i32>)>,
    chars: usize,
    last_status: Option<String>,
    subs: Vec<Sender<String>>,
    /// The last readings of its mind, as lines.
    minds: VecDeque<String>,
    /// What it works toward (the last `objective` line), and the last
    /// lines of its terminal, replayed to a new tail.
    objective: Option<String>,
    term: VecDeque<String>,
    /// The second chain's last lines, replayed likewise.
    delib: VecDeque<String>,
}

impl Hub {
    fn push_delib(&mut self, line: String) {
        self.delib.push_back(line);
        while self.delib.len() > KEEP_DELIB {
            self.delib.pop_front();
        }
    }

    fn push_term(&mut self, line: String) {
        self.term.push_back(line);
        while self.term.len() > KEEP_TERM {
            self.term.pop_front();
        }
    }

    fn push_text(&mut self, text: &str, kind: Kind, t_us: i64, pos: Option<i32>) {
        self.chars += text.chars().count();
        self.recent.push_back((text.to_string(), kind, t_us, pos));
        while self.chars > KEEP_CHARS && self.recent.len() > 1 {
            let (t, _, _, _) = self.recent.pop_front().unwrap();
            self.chars -= t.chars().count();
        }
    }

    fn broadcast(&mut self, line: &str) {
        self.subs.retain(|s| s.send(line.to_string()).is_ok());
    }

    /// The readings of its mind kept, then the last `REPLAY_CHARS` of text
    /// as lines (each joins its reading by position, so the readings come
    /// first: after a reload the replayed lines had lost their lens rows),
    /// then the last status.
    fn replay(&self) -> Vec<String> {
        let mut n = 0;
        let mut start = self.recent.len();
        while start > 0 && n < REPLAY_CHARS {
            start -= 1;
            n += self.recent[start].0.chars().count();
        }
        let mut out: Vec<String> = self.minds.iter().cloned().collect();
        out.extend(
            self.recent
                .iter()
                .skip(start)
                .map(|(t, k, at, p)| crate::client::text_line(t, *k, *at, *p)),
        );
        out.extend(self.objective.iter().cloned());
        out.extend(self.term.iter().cloned());
        out.extend(self.delib.iter().cloned());
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
        objective: None,
        term: VecDeque::new(),
        delib: VecDeque::new(),
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
                    Event::Text(t, k, at, p) => {
                        h.push_text(&t, k, at, p);
                        let line = crate::client::text_line(&t, k, at, p);
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
                    Event::Objective(t, text) => {
                        let line = format!("objective t={t} {}", escape(&text));
                        h.objective = Some(line.clone());
                        h.broadcast(&line);
                    }
                    Event::TermStart(id, t, cmd) => {
                        let line = format!("term start t={t} id={id} {}", escape(&cmd));
                        h.push_term(line.clone());
                        h.broadcast(&line);
                    }
                    Event::Act(a) => {
                        let line = crate::client::act_line(&a);
                        h.push_term(line.clone());
                        h.broadcast(&line);
                    }
                    // Per token, as `mind` lines are, and not kept.
                    Event::Guide(g) => h.broadcast(&crate::client::guide_line(&g)),
                    Event::ToClaude(m) => {
                        let line = crate::client::to_claude_line(&m);
                        h.push_term(line.clone());
                        h.broadcast(&line);
                    }
                    Event::Delib(d) => {
                        let line = crate::client::delib_line(&d);
                        h.push_delib(line.clone());
                        h.broadcast(&line);
                    }
                    Event::TermEnd(t, r) => {
                        let line = crate::client::term_end_line(t, &r);
                        h.push_term(line.clone());
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
            // A message from Claude that waits for an answer: its id back,
            // which the answer names (`tell_claude` with re).
            "ask" => {
                if arg.is_empty() {
                    Err("ask what?".to_string())
                } else {
                    let id = ASKS.fetch_add(1, Ordering::SeqCst);
                    if let Some(p) = ASKS_FILE.get() {
                        let _ = fs::write(p, format!("{}\n", id + 1));
                    }
                    let t = crate::clock::now_us();
                    ctx.send(Command::Ask(id, crate::client::unescape(arg), t))
                        .map(|_| format!("asked c{id} t={t}"))
                        .map_err(|_| "the engine is gone".to_string())
                }
            }
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
            "keep-at" => match arg.parse::<f32>() {
                Ok(p) if (0.0..=2.0).contains(&p) => ctx.send(Command::KeepAt(p)).map(|_| format!("keep at {p}")).map_err(|_| "the engine is gone".to_string()),
                _ => Err("keep-at takes a share between 0 and 2 (above 1 every check writes)".to_string()),
            },
            "chunk" => match arg.parse::<usize>() {
                Ok(n) => ctx.send(Command::Chunk(n)).map(|_| format!("chunk {n}")).map_err(|_| "the engine is gone".to_string()),
                Err(_) => Err("chunk takes a number (0 adapts)".to_string()),
            },
            "set" => match arg.split_once(' ').map(|(k, v)| (k.trim(), v.trim().parse::<f32>())) {
                Some((k, Ok(v))) if ["temp", "top-k", "top-p", "min-p", "dry", "repeat-penalty", "guide-mix"].contains(&k) => ctx.send(Command::Set(k.to_string(), v)).map(|_| format!("{k} {v}")).map_err(|_| "the engine is gone".to_string()),
                _ => Err("set takes temp, top-k, top-p, min-p, dry, repeat-penalty or guide-mix and a number".to_string()),
            },
            "breaker" | "nudges" | "guide" | "experts" => match arg.trim() {
                "on" | "off" => ctx.send(Command::Guard(cmd.to_string(), arg.trim() == "on")).map(|_| format!("{cmd} {}", arg.trim())).map_err(|_| "the engine is gone".to_string()),
                // The guide's aside source (engine.md): one of three.
                s @ ("chain" | "lens" | "placebo" | "ab") if cmd == "guide" => ctx.send(Command::Guard(s.to_string(), true)).map(|_| format!("guide asides from {s}")).map_err(|_| "the engine is gone".to_string()),
                _ if cmd == "guide" => Err("guide takes on, off, chain, lens, placebo or ab".to_string()),
                _ => Err(format!("{cmd} takes on or off")),
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
            "chain" => match arg.trim() {
                "on" => ctx.send(Command::Chain(true)).map(|_| "the second chain on".to_string()).map_err(|_| "the engine is gone".to_string()),
                "off" => ctx.send(Command::Chain(false)).map(|_| "the second chain off".to_string()).map_err(|_| "the engine is gone".to_string()),
                _ => Err("chain takes on or off".to_string()),
            },
            "objective" => ctx
                .send(Command::Objective(arg.to_string()))
                .map(|_| if arg.trim().is_empty() { "objective cleared".to_string() } else { format!("objective set: {}", arg.trim()) })
                .map_err(|_| "the engine is gone".to_string()),
            "quit" => {
                ctx.send(Command::Quit).ok();
                Ok("stopping".to_string())
            }
            _ => Err(format!("unknown command {cmd}; say, say-as, ask, feed, persona, objective, chain, chunk, set, temp, pause, resume, status, recent, tail, quit")),
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
