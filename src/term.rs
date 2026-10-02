//! The stream's terminal: a command line it writes (`[run: COMMAND]`) run
//! by `sh -c` in a sandbox (bubblewrap) that sees the system's programs, the
//! repository read-only and its own workspace read-write, nothing else of
//! the home directory, no network; at the lowest priority on one CPU, with
//! a time limit and its output capped. One command at a time, on a thread
//! of its own, so the stream never waits for it. See term.md.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct TermConfig {
    /// The repository it develops (or none): where its commands start, read
    /// only, or through `overlay`.
    pub repo: Option<PathBuf>,
    /// Its working copy of the repository: a directory holding an overlay's
    /// `upper` (its writes) and `work`. The repository is seen as it is with
    /// its writes on top; the repository itself never changes. None: the
    /// repository read only.
    pub overlay: Option<PathBuf>,
    /// Read-write: its workspace, where its commands start.
    pub workspace: PathBuf,
    /// The longest a command may run.
    pub timeout: Duration,
    /// The most output kept (bytes, stdout and stderr together).
    pub max_out: usize,
    /// The CPU the command is pinned to.
    pub cpu: usize,
}

/// How a command ended.
#[derive(Clone, Debug, PartialEq)]
pub struct Ran {
    pub id: u64,
    pub command: String,
    /// The exit code; none when it was stopped (time) or could not start.
    pub code: Option<i32>,
    pub out: String,
    /// The output was cut at `max_out`.
    pub cut: bool,
    pub timed_out: bool,
    pub ms: f32,
}

/// The programs a command may not run: the person's standing instructions
/// forbid Python, so the interpreters are masked inside the sandbox.
fn masked() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/usr/bin") {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            let python = n.starts_with("python") || n == "pip" || n.starts_with("pip3");
            // A symlink (python3 to python3.14) resolves to the masked file;
            // bubblewrap cannot mount over the link itself.
            let file = e.path().symlink_metadata().is_ok_and(|m| m.is_file());
            if python && file {
                out.push(e.path());
            }
        }
    }
    out.sort();
    out
}

/// The sandbox's command line for `command` (`bwrap` and its arguments).
pub fn argv(cfg: &TermConfig, command: &str) -> Vec<String> {
    let ws = cfg.workspace.display().to_string();
    let mut a: Vec<String> = vec![
        "nice".into(),
        "-n".into(),
        "19".into(),
        "taskset".into(),
        "-c".into(),
        cfg.cpu.to_string(),
        "bwrap".into(),
        "--ro-bind".into(),
        "/usr".into(),
        "/usr".into(),
        "--symlink".into(),
        "usr/bin".into(),
        "/bin".into(),
        "--symlink".into(),
        "usr/bin".into(),
        "/sbin".into(),
        "--symlink".into(),
        "usr/lib".into(),
        "/lib".into(),
        "--symlink".into(),
        "usr/lib".into(),
        "/lib64".into(),
        "--ro-bind-try".into(),
        "/etc/ld.so.cache".into(),
        "/etc/ld.so.cache".into(),
        "--ro-bind-try".into(),
        "/etc/localtime".into(),
        "/etc/localtime".into(),
        "--proc".into(),
        "/proc".into(),
        "--dev".into(),
        "/dev".into(),
        "--tmpfs".into(),
        "/tmp".into(),
    ];
    for p in masked() {
        let p = p.display().to_string();
        a.extend(["--ro-bind".into(), "/dev/null".into(), p]);
    }
    if let Some(r) = &cfg.repo {
        let r = r.display().to_string();
        match &cfg.overlay {
            Some(o) => a.extend([
                "--overlay-src".into(),
                r.clone(),
                "--overlay".into(),
                o.join("upper").display().to_string(),
                o.join("work").display().to_string(),
                r,
            ]),
            None => a.extend(["--ro-bind".into(), r.clone(), r]),
        }
    }
    // Commands start in the repository when there is one (its paths are
    // what the stream reads by), else in the workspace.
    let start = cfg
        .repo
        .as_ref()
        .map_or(ws.clone(), |r| r.display().to_string());
    a.extend([
        "--bind".into(),
        ws.clone(),
        ws.clone(),
        "--chdir".into(),
        start,
        "--unshare-all".into(),
        "--die-with-parent".into(),
        "--new-session".into(),
        "--clearenv".into(),
        "--setenv".into(),
        "PATH".into(),
        "/usr/bin".into(),
        "--setenv".into(),
        "HOME".into(),
        ws,
        "--setenv".into(),
        "LANG".into(),
        "C.UTF-8".into(),
        "sh".into(),
        "-c".into(),
        command.into(),
    ]);
    a
}

/// Run one command in the sandbox and wait for it (the terminal's thread).
pub fn run(cfg: &TermConfig, id: u64, command: &str) -> Ran {
    let t0 = Instant::now();
    if let Some(o) = &cfg.overlay {
        let _ = std::fs::create_dir_all(o.join("upper"));
        let _ = std::fs::create_dir_all(o.join("work"));
    }
    let a = argv(cfg, command);
    let mut child = match Command::new(&a[0])
        .args(&a[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return Ran {
                id,
                command: command.into(),
                code: None,
                out: format!("could not start the sandbox: {e}"),
                cut: false,
                timed_out: false,
                ms: 0.0,
            }
        }
    };
    // Output read on two threads, so a full pipe never stalls the command.
    let max = cfg.max_out;
    let reader = |mut r: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let mut cut = false;
            while let Ok(n) = r.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if buf.len() < max {
                    let take = n.min(max - buf.len());
                    buf.extend_from_slice(&chunk[..take]);
                    cut |= take < n;
                } else {
                    cut = true;
                }
            }
            (buf, cut)
        })
    };
    let out_t = reader(Box::new(child.stdout.take().unwrap()));
    let err_t = reader(Box::new(child.stderr.take().unwrap()));
    let mut timed_out = false;
    let code = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s.code(),
            Ok(None) if t0.elapsed() >= cfg.timeout => {
                timed_out = true;
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => break None,
        }
    };
    let (mut out, cut_o) = out_t.join().unwrap_or_default();
    let (err, cut_e) = err_t.join().unwrap_or_default();
    out.extend_from_slice(&err);
    let cut = cut_o || cut_e || out.len() > max;
    out.truncate(max);
    Ran {
        id,
        command: command.into(),
        code,
        out: String::from_utf8_lossy(&out).into_owned(),
        cut,
        timed_out,
        ms: t0.elapsed().as_secs_f32() * 1000.0,
    }
}

/// The terminal: commands go to a thread that runs them one at a time.
pub struct Term {
    tx: Sender<(u64, String)>,
    rx: Receiver<Ran>,
    next: u64,
}

impl Term {
    pub fn start(cfg: TermConfig) -> Self {
        let (tx, cmd_rx) = mpsc::channel::<(u64, String)>();
        let (ran_tx, rx) = mpsc::channel::<Ran>();
        let c = cfg.clone();
        std::thread::spawn(move || {
            for (id, command) in cmd_rx {
                if ran_tx.send(run(&c, id, &command)).is_err() {
                    break;
                }
            }
        });
        Self { tx, rx, next: 1 }
    }

    /// Queue a command; its id.
    pub fn submit(&mut self, command: &str) -> u64 {
        let id = self.next;
        self.next += 1;
        let _ = self.tx.send((id, command.to_string()));
        id
    }

    /// A command that has ended, if any.
    pub fn poll(&self) -> Option<Ran> {
        self.rx.try_recv().ok()
    }
}

/// Whether the sandbox can run here (bubblewrap and taskset installed).
pub fn available() -> bool {
    ["bwrap", "taskset", "nice"].iter().all(|p| {
        std::env::var_os("PATH")
            .is_some_and(|path| std::env::split_paths(&path).any(|d| d.join(p).is_file()))
    })
}

/// A path for tests: a fresh directory on disk under the repository's
/// target directory (never tmpfs).
#[cfg(test)]
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("term-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[cfg(test)]
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ws: &Path) -> TermConfig {
        TermConfig {
            repo: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
            overlay: None,
            workspace: ws.to_path_buf(),
            timeout: Duration::from_secs(10),
            max_out: 4096,
            cpu: 0,
        }
    }

    #[test]
    fn the_sandbox_sees_the_repository_and_its_workspace_only() {
        if !available() {
            eprintln!("bwrap not installed: skipped");
            return;
        }
        let ws = scratch("see");
        let c = cfg(&ws);
        let repo = env!("CARGO_MANIFEST_DIR");
        // The repository reads; it does not write.
        let r = run(
            &c,
            1,
            &format!(
                "grep -c -F {q}[package]{q} {q}{repo}/Cargo.toml{q}",
                q = "'"
            ),
        );
        assert_eq!(r.code, Some(0), "{r:?}");
        assert_eq!(r.out.trim(), "1", "{r:?}");
        let r = run(&c, 2, &format!("touch '{repo}/x'"));
        assert_ne!(r.code, Some(0), "{r:?}");
        assert!(!Path::new(repo).join("x").exists());
        // The workspace writes; the command starts in the repository.
        let r = run(
            &c,
            3,
            &format!("echo hi > '{}/made.txt' && pwd", ws.display()),
        );
        assert_eq!(r.code, Some(0), "{r:?}");
        assert_eq!(r.out.trim(), repo, "{r:?}");
        assert_eq!(
            std::fs::read_to_string(ws.join("made.txt")).unwrap(),
            "hi\n"
        );
        // Nothing else of the home directory, and no Python.
        let home = std::env::var("HOME").unwrap();
        let r = run(
            &c,
            4,
            &format!("ls '{home}/.ssh' 2>&1; ls '{home}' 2>&1 | head -3"),
        );
        assert!(!r.out.contains("id_"), "{r:?}");
        let r = run(&c, 5, "python3 -c 'print(1)'");
        assert_ne!(r.code, Some(0), "{r:?}");
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn the_working_copy_takes_its_writes_and_the_repository_none() {
        if !available() {
            eprintln!("bwrap not installed: skipped");
            return;
        }
        let ws = scratch("copy-ws");
        let copy = scratch("copy");
        let mut c = cfg(&ws);
        c.overlay = Some(copy.clone());
        let repo = env!("CARGO_MANIFEST_DIR");
        // It reads the repository and writes into it, in its own view.
        let r = run(&c, 1, "echo edit > made-by-the-stream.txt && cat made-by-the-stream.txt && grep -c -F [package] Cargo.toml");
        assert_eq!(r.code, Some(0), "{r:?}");
        assert_eq!(r.out, "edit\n1\n", "{r:?}");
        // The write is in the working copy, and the repository has none.
        assert!(copy.join("upper/made-by-the-stream.txt").is_file());
        assert!(!Path::new(repo).join("made-by-the-stream.txt").exists());
        // A later command sees it still.
        let r = run(&c, 2, "cat made-by-the-stream.txt");
        assert_eq!(r.out, "edit\n", "{r:?}");
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&copy);
    }

    #[test]
    fn no_network_a_time_limit_and_a_cap() {
        if !available() {
            eprintln!("bwrap not installed: skipped");
            return;
        }
        let ws = scratch("limits");
        let mut c = cfg(&ws);
        // Only the loopback device exists in its own network namespace.
        let r = run(
            &c,
            1,
            "cat /proc/net/dev | tail -n +3 | cut -d: -f1 | tr -d ' '",
        );
        assert_eq!(r.out.trim(), "lo", "{r:?}");
        c.timeout = Duration::from_millis(300);
        let r = run(&c, 2, "sleep 5");
        assert!(r.timed_out && r.code.is_none(), "{r:?}");
        c.max_out = 100;
        let r = run(&c, 3, "yes | head -c 10000");
        assert!(r.cut && r.out.len() == 100, "{r:?}");
        let _ = std::fs::remove_dir_all(&ws);
    }
}
