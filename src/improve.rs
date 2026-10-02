//! The self-improvement loop, first stage (`--improve`, improve.md): a
//! change the stream proposes from its working copy is staged as a diff
//! against a recorded base commit, refused when it touches the loop, its
//! evaluator or the build, built and tested in a sandbox (`make check`, no
//! network), and its outcome told to the stream at its next turn and kept
//! in `improve.log`, which it reads to choose the next change. Claude
//! reviews every change that passes before anything of it runs outside the
//! sandbox.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;

/// Paths a proposal may not touch: the build (it runs what it builds:
/// build.rs, the manifest, the Makefile, cargo's own configuration), the
/// scripts that launch the service, and the loop with its evaluators (the
/// code benchmark's harness and the reflection gate). A change to any of
/// them goes through Claude and the person, never through this loop.
pub const DENY: &[&str] = &[
    "build.rs",
    "Cargo.toml",
    "Cargo.lock",
    "Makefile",
    ".cargo/",
    ".git/",
    "scripts/",
    "src/improve.rs",
    "src/improve.md",
    "src/code.rs",
    "src/code.md",
    "src/gate.rs",
    "src/gate.md",
];

/// The longest a build may take, and the CPUs it runs on (the stream's
/// threads start from the first; `nice -n 19` too).
const BUILD_LIMIT_S: u64 = 1200;
const BUILD_CPUS: &str = "12-15";
const BUILD_JOBS: &str = "4";

#[derive(Clone, Debug)]
pub struct ImproveConfig {
    /// The repository being developed, and the stream's working copy's
    /// upper layer (its writes, `term.md`).
    pub repo: PathBuf,
    pub upper: PathBuf,
    /// Where candidates are staged and built (on disk, never tmpfs).
    pub root: PathBuf,
    /// Where each candidate's record is copied for the stream to read (its
    /// workspace's `improve/`): the outcome, the diff, the whole build log
    /// and, once measured, the measurement.
    pub mirror: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Not built: it touches a denied path, a file changed under it, or it
    /// changes nothing.
    Refused,
    /// Built and tested, and `make check` failed.
    Failed,
    /// `make check` passed: Claude reviews it next.
    Passed,
}

impl Verdict {
    pub fn word(self) -> &'static str {
        match self {
            Verdict::Refused => "refused",
            Verdict::Failed => "failed",
            Verdict::Passed => "passed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub id: u64,
    pub title: String,
    pub base: String,
    pub files: Vec<String>,
    pub verdict: Verdict,
    /// What the stream is told: the reason, or the build's errors in short.
    pub summary: String,
    /// The candidate's directory (`change.patch`, `build.log`, `tree/`).
    pub dir: PathBuf,
    pub secs: f64,
}

/// One file of a candidate: its path in the repository and its new
/// content, or `None` for a deletion (an overlay whiteout).
type Change = (String, Option<Vec<u8>>);

fn git(repo: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// The commit `HEAD` names.
pub fn head(repo: &Path) -> Result<String, String> {
    Ok(String::from_utf8_lossy(&git(repo, &["rev-parse", "HEAD"])?)
        .trim()
        .to_string())
}

/// Whether a path is one a proposal may not touch.
pub fn denied(rel: &str) -> bool {
    DENY.iter().any(|d| {
        if d.ends_with('/') {
            rel.starts_with(d)
        } else {
            rel == *d
        }
    })
}

/// The files of the working copy's layer that differ from `base`: each
/// regular file whose bytes are not the base's, and each whiteout (a
/// character device, the overlay's mark of a deletion) of a file the base
/// has.
fn changes(cfg: &ImproveConfig, base: &str) -> Result<Vec<Change>, String> {
    let mut out = Vec::new();
    let mut stack = vec![cfg.upper.clone()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(rel) = p.strip_prefix(&cfg.upper) else {
                continue;
            };
            let rel = rel.to_string_lossy().into_owned();
            let Ok(meta) = std::fs::symlink_metadata(&p) else {
                continue;
            };
            let ft = meta.file_type();
            if ft.is_dir() {
                if rel != ".git" && rel != "target" {
                    stack.push(p);
                }
                continue;
            }
            let at_base = git(&cfg.repo, &["show", &format!("{base}:{rel}")]).ok();
            if ft.is_file() {
                let new = std::fs::read(&p).map_err(|e| format!("{rel}: {e}"))?;
                if at_base.as_deref() != Some(new.as_slice()) {
                    out.push((rel, Some(new)));
                }
            } else {
                use std::os::unix::fs::FileTypeExt;
                if ft.is_char_device() && at_base.is_some() {
                    out.push((rel, None));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Stage a candidate: the base exported from git (`git archive`) into
/// `tree/`, made a repository of its own with the base as its one commit,
/// the changes written over it, and the diff kept as `change.patch` (what
/// Claude reviews and applies).
fn stage(dir: &Path, cfg: &ImproveConfig, base: &str, ch: &[Change]) -> Result<(), String> {
    let tree = dir.join("tree");
    std::fs::create_dir_all(&tree).map_err(|e| format!("{}: {e}", tree.display()))?;
    let archive = Command::new("git")
        .arg("-C")
        .arg(&cfg.repo)
        .args(["archive", "--format=tar", base])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("git archive: {e}"))?;
    let tar = Command::new("tar")
        .arg("-x")
        .arg("-C")
        .arg(&tree)
        .stdin(archive.stdout.unwrap())
        .status()
        .map_err(|e| format!("tar: {e}"))?;
    if !tar.success() {
        return Err("tar refused the base's archive".into());
    }
    let id = [
        "-c",
        "user.name=improve",
        "-c",
        "user.email=improve@localhost",
    ];
    git(&tree, &["init", "-q"])?;
    git(&tree, &["add", "-A"])?;
    let mut commit: Vec<&str> = id.to_vec();
    let msg = format!("base {base}");
    commit.extend(["commit", "-qm", &msg]);
    git(&tree, &commit)?;
    for (rel, new) in ch {
        let p = tree.join(rel);
        match new {
            Some(bytes) => {
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("{rel}: {e}"))?;
                }
                std::fs::write(&p, bytes).map_err(|e| format!("{rel}: {e}"))?;
            }
            None => {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    git(&tree, &["add", "-A"])?;
    let patch = git(&tree, &["diff", "--cached"])?;
    std::fs::write(dir.join("change.patch"), patch).map_err(|e| format!("change.patch: {e}"))?;
    Ok(())
}

/// The build's sandbox (bubblewrap, as the terminal's, `term.md`): `/usr`
/// read-only, the staged tree read-write at its own path, the build's
/// target directory and cargo's home (the registry read-only inside it)
/// read-write, llama.cpp read-only at its own path (the headers bound and
/// the libraries linked, `build.md`), no network, Python masked, a cleared
/// environment with cargo offline; at the lowest priority on the last CPUs,
/// stopped after `BUILD_LIMIT_S`.
fn build_argv(tree: &Path, root: &Path) -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let llama = std::env::var("LLAMA_CPP_DIR").unwrap_or(format!("{home}/llama.cpp"));
    let t = tree.display().to_string();
    let target = root.join("target").display().to_string();
    let cargo = root.join("cargo-home").display().to_string();
    let tmp = root.join("tmp").display().to_string();
    let mut a: Vec<String> = [
        "timeout",
        "-k",
        "10",
        &BUILD_LIMIT_S.to_string(),
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        BUILD_CPUS,
        "bwrap",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/bin",
        "/sbin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
        "--ro-bind-try",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--ro-bind-try",
        "/etc/localtime",
        "/etc/localtime",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--bind",
        &tmp,
        "/tmp",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for p in crate::term::masked() {
        let p = p.display().to_string();
        a.extend(["--ro-bind".into(), "/dev/null".into(), p]);
    }
    a.extend(
        [
            "--ro-bind",
            &llama,
            &llama,
            "--bind",
            &t,
            &t,
            "--bind",
            &target,
            "/target",
            "--bind",
            &cargo,
            "/cargo",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    let registry = format!("{home}/.cargo/registry");
    if Path::new(&registry).is_dir() {
        a.extend(["--ro-bind".into(), registry, "/cargo/registry".into()]);
    }
    a.extend(
        [
            "--chdir",
            &t,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LLAMA_CPP_DIR",
            &llama,
            "--setenv",
            "CARGO_HOME",
            "/cargo",
            "--setenv",
            "CARGO_TARGET_DIR",
            "/target",
            "--setenv",
            "CARGO_NET_OFFLINE",
            "true",
            "--setenv",
            "CARGO_BUILD_JOBS",
            BUILD_JOBS,
            "make",
            "check",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    a
}

/// The build's errors in short, for the stream: the lines that say what
/// failed (cargo's errors with their places, failed tests, panics, the
/// docs check's and rustfmt's complaints), then the last lines; at most
/// about 3000 characters.
pub fn summarize(log: &str) -> String {
    let mut keep: Vec<&str> = Vec::new();
    let lines: Vec<&str> = log.lines().collect();
    for l in &lines {
        let t = l.trim_start();
        let hit = t.starts_with("error")
            || t.starts_with("--> ")
            || t.contains("FAILED")
            || t.contains("panicked at")
            || t.starts_with("Diff in")
            || t.starts_with("check-docs.sh:")
            || t.starts_with("make: ***");
        if hit {
            keep.push(l);
        }
    }
    keep.dedup();
    let mut s = keep.join("\n");
    if s.len() > 2400 {
        let mut cut = 2400;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("\n...");
    }
    let tail: Vec<&str> = lines.iter().rev().take(6).rev().copied().collect();
    format!("{s}\nthe build's last lines:\n{}", tail.join("\n"))
}

/// Collect, stage and build one proposal (on the improver's thread).
pub fn attempt(cfg: &ImproveConfig, id: u64, title: &str, told: Option<&str>) -> Outcome {
    let t0 = Instant::now();
    let mut o = Outcome {
        id,
        title: title.to_string(),
        base: String::new(),
        files: Vec::new(),
        verdict: Verdict::Refused,
        summary: String::new(),
        // Number 0 is a trial (`build`): staged and built the same way, in a
        // directory of its own, never a candidate.
        dir: cfg.root.join(cand_dir(id)),
        secs: 0.0,
    };
    let done = |mut o: Outcome, v: Verdict, s: String| {
        o.verdict = v;
        o.summary = s;
        o.secs = t0.elapsed().as_secs_f64();
        o
    };
    let base = match head(&cfg.repo) {
        Ok(b) => b,
        Err(e) => return done(o, Verdict::Refused, e),
    };
    o.base = base.clone();
    let ch = match changes(cfg, &base) {
        Ok(c) => c,
        Err(e) => return done(o, Verdict::Refused, e),
    };
    o.files = ch.iter().map(|c| c.0.clone()).collect();
    if ch.is_empty() {
        return done(
            o,
            Verdict::Refused,
            "your working copy changes nothing against the repository".into(),
        );
    }
    let bad: Vec<&str> = o
        .files
        .iter()
        .filter(|f| denied(f))
        .map(|f| f.as_str())
        .collect();
    if !bad.is_empty() {
        let s = format!(
            "it touches paths this loop may not change ({}): the build, the scripts and the loop with its evaluators go through Claude",
            bad.join(", ")
        );
        return done(o, Verdict::Refused, s);
    }
    // A file the repository changed since the stream was last told of a
    // commit: its copy was made from an older one, and building it would
    // revert that commit's work there.
    if let Some(t) = told.filter(|t| *t != base) {
        let range = format!("{t}..{base}");
        let mut args = vec!["log", "--format=%h", range.as_str(), "--"];
        args.extend(o.files.iter().map(|f| f.as_str()));
        match git(&cfg.repo, &args) {
            Ok(out) if !out.is_empty() => {
                let s = format!(
                    "these files changed in the repository since your copy was made (commits {}): read them again and make the change over the new version",
                    String::from_utf8_lossy(&out).split_whitespace().collect::<Vec<_>>().join(", ")
                );
                return done(o, Verdict::Refused, s);
            }
            Ok(_) => {}
            Err(e) => return done(o, Verdict::Refused, e),
        }
    }
    let _ = std::fs::remove_dir_all(&o.dir);
    for d in [
        o.dir.clone(),
        cfg.root.join("target"),
        cfg.root.join("cargo-home"),
        cfg.root.join("tmp"),
    ] {
        if let Err(e) = std::fs::create_dir_all(&d) {
            return done(o, Verdict::Refused, format!("{}: {e}", d.display()));
        }
    }
    if let Err(e) = stage(&o.dir, cfg, &base, &ch) {
        return done(o, Verdict::Refused, e);
    }
    let argv = build_argv(&o.dir.join("tree"), &cfg.root);
    let log_path = o.dir.join("build.log");
    let ran = std::fs::File::create(&log_path).and_then(|f| {
        let f2 = f.try_clone()?;
        Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(f)
            .stderr(f2)
            .status()
    });
    let log = std::fs::read_to_string(&log_path).unwrap_or_default();
    match ran {
        Ok(st) if st.success() => {
            let tests: Vec<&str> = log
                .lines()
                .filter(|l| l.starts_with("test result:"))
                .collect();
            // The binary it built, kept with it: the shared target directory
            // is the next candidate's, and measuring runs this one.
            let built = cfg.root.join("target/release/phi-stream");
            if let Err(e) = std::fs::copy(&built, o.dir.join("phi-stream")) {
                return done(
                    o,
                    Verdict::Failed,
                    format!("make check passed, but its binary was not kept: {e}"),
                );
            }
            let s = format!(
                "make check passed in its sandbox ({}); {}",
                tests.last().copied().unwrap_or("no test summary"),
                if id == 0 {
                    "a trial: nothing is sent; propose it when it is ready"
                } else {
                    "Claude reviews the change next"
                }
            );
            done(o, Verdict::Passed, s)
        }
        Ok(st) => {
            let why = if st.code() == Some(124) {
                format!("stopped at the limit ({BUILD_LIMIT_S} s)\n")
            } else {
                String::new()
            };
            let s = format!("make check failed:\n{why}{}", summarize(&log));
            done(o, Verdict::Failed, s)
        }
        Err(e) => done(o, Verdict::Refused, format!("the build did not start: {e}")),
    }
}

/// The working copy's change against the repository's `HEAD`, as a
/// unified diff (`diff`): what a proposal would hold now.
pub fn diff_text(cfg: &ImproveConfig) -> Result<String, String> {
    let base = head(&cfg.repo)?;
    let ch = changes(cfg, &base)?;
    if ch.is_empty() {
        return Ok("your working copy changes nothing against the repository".into());
    }
    let mut out = String::new();
    for (rel, new) in &ch {
        let old = cfg.repo.join(rel);
        let old = if old.is_file() {
            old
        } else {
            PathBuf::from("/dev/null")
        };
        let new = match new {
            Some(_) => cfg.upper.join(rel),
            None => PathBuf::from("/dev/null"),
        };
        let d = Command::new("git")
            .args(["diff", "--no-index", "--no-color", "--"])
            .arg(&old)
            .arg(&new)
            .output()
            .map_err(|e| format!("git diff: {e}"))?;
        // Paths as the repository names them, not where the files lie.
        let text = String::from_utf8_lossy(&d.stdout)
            .replace(&format!("a{}", old.display()), &format!("a/{rel}"))
            .replace(&format!("b{}", new.display()), &format!("b/{rel}"));
        out.push_str(&text);
    }
    if denied_any(&ch) {
        out.push_str("\n(note: it touches a path the loop may not change; propose refuses it)\n");
    }
    Ok(out)
}

fn denied_any(ch: &[Change]) -> bool {
    ch.iter().any(|c| denied(&c.0))
}

/// One file of the working copy back to the repository's version
/// (`revert`): its copy, or its deletion mark, taken out of the layer. The
/// stream cannot do it itself: a delete inside the overlay hides the
/// repository's file too (`term.md`). The path must be relative and stay
/// inside the layer.
pub fn revert(cfg: &ImproveConfig, rel: &str) -> Result<String, String> {
    let rel = rel.trim().trim_start_matches("./");
    let p = Path::new(rel);
    if rel.is_empty()
        || p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(format!(
            "{rel:?}: a path relative to the repository, without .. or a leading /"
        ));
    }
    let f = cfg.upper.join(p);
    match std::fs::symlink_metadata(&f) {
        Ok(m) if m.is_dir() => Err(format!("{rel} is a directory: revert its files one by one")),
        Ok(_) => {
            std::fs::remove_file(&f).map_err(|e| format!("{rel}: {e}"))?;
            Ok(format!(
                "{rel}: your copy is gone; you read the repository's version again"
            ))
        }
        Err(_) => Err(format!("{rel}: your working copy has no change to it")),
    }
}

/// A candidate's directory name: `cand-N`, or `trial` for number 0.
pub fn cand_dir(id: u64) -> String {
    if id == 0 {
        "trial".to_string()
    } else {
        format!("cand-{id}")
    }
}

/// A candidate's record: its `outcome` file beside its diff and build log,
/// and all three copied into the stream's workspace (`improve/cand-N/`),
/// where it reads them: the whole build log, not only the summary it is
/// told.
pub fn record(cfg: &ImproveConfig, o: &Outcome) {
    if std::fs::create_dir_all(&o.dir).is_err() {
        return;
    }
    let text = format!(
        "candidate {}\nverdict {}\nbase {}\ntitle {}\nfiles {}\nseconds {:.0}\n\n{}\n",
        o.id,
        o.verdict.word(),
        o.base,
        o.title,
        o.files.join(", "),
        o.secs,
        o.summary
    );
    let _ = std::fs::write(o.dir.join("outcome"), text);
    let m = cfg.mirror.join(cand_dir(o.id));
    if std::fs::create_dir_all(&m).is_ok() {
        for f in ["outcome", "change.patch", "build.log"] {
            let _ = std::fs::copy(o.dir.join(f), m.join(f));
        }
    }
}

/// The improver: one proposal at a time on a thread of its own, so the
/// stream keeps thinking while its change builds.
pub struct Improver {
    cfg: ImproveConfig,
    tx: Sender<Outcome>,
    rx: Receiver<Outcome>,
    busy: Option<u64>,
    next: u64,
}

impl Improver {
    pub fn new(cfg: ImproveConfig) -> Self {
        let (tx, rx) = channel();
        // Candidate numbers go on across restarts: the next after the
        // highest staged.
        let next = std::fs::read_dir(&cfg.root)
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .strip_prefix("cand-")
                            .and_then(|n| n.parse::<u64>().ok())
                    })
                    .max()
                    .map_or(1, |m| m + 1)
            })
            .unwrap_or(1);
        Improver {
            cfg,
            tx,
            rx,
            busy: None,
            next,
        }
    }

    /// Start one: its number, or the one still building.
    pub fn propose(&mut self, title: &str, told: Option<String>) -> Result<u64, u64> {
        if let Some(b) = self.busy {
            return Err(b);
        }
        let id = self.next;
        self.next += 1;
        self.busy = Some(id);
        let (cfg, tx, title) = (self.cfg.clone(), self.tx.clone(), title.to_string());
        thread::spawn(move || {
            let o = attempt(&cfg, id, &title, told.as_deref());
            record(&cfg, &o);
            let _ = tx.send(o);
        });
        Ok(id)
    }

    /// A trial build (`build`): the working copy's change staged and run
    /// through make check as a candidate is, but numbered 0, never sent to
    /// Claude; or the number still building.
    pub fn trial(&mut self, told: Option<String>) -> Result<(), u64> {
        if let Some(b) = self.busy {
            return Err(b);
        }
        self.busy = Some(0);
        let (cfg, tx) = (self.cfg.clone(), self.tx.clone());
        thread::spawn(move || {
            let o = attempt(&cfg, 0, "trial build", told.as_deref());
            record(&cfg, &o);
            let _ = tx.send(o);
        });
        Ok(())
    }

    pub fn config(&self) -> &ImproveConfig {
        &self.cfg
    }

    /// What is building: a candidate's number, 0 for a trial.
    pub fn building(&self) -> Option<u64> {
        self.busy
    }

    pub fn poll(&mut self) -> Option<Outcome> {
        let o = self.rx.try_recv().ok()?;
        self.busy = None;
        Some(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loop_and_the_build_are_denied() {
        for p in [
            "build.rs",
            "Cargo.toml",
            "scripts/phi-stream.sh",
            "src/improve.rs",
            "src/code.rs",
            ".cargo/config.toml",
        ] {
            assert!(denied(p), "{p}");
        }
        for p in [
            "src/engine.rs",
            "src/agent.md",
            "docs/dev.md",
            "tools/guide-analyze.c",
            "src/code_x.rs",
        ] {
            assert!(!denied(p), "{p}");
        }
    }

    #[test]
    fn a_summary_keeps_the_errors_and_their_places() {
        let log = "   Compiling phi-stream v0.1.0\nerror[E0308]: mismatched types\n   --> src/engine.rs:12:5\n    |\nwarning: unused\nerror: could not compile `phi-stream`\nmake: *** [Makefile:20: clippy] Error 101\n";
        let s = summarize(log);
        assert!(s.contains("error[E0308]: mismatched types"));
        assert!(s.contains("--> src/engine.rs:12:5"));
        assert!(s.contains("make: ***"));
        assert!(!s.contains("Compiling phi-stream v0.1.0\nerror"));
    }

    /// A real repository, a real layer, a real git: a change is staged as a
    /// diff, an unchanged copy is not a change, a denied path is refused.
    #[test]
    fn a_change_is_staged_against_its_base() {
        let root = std::env::temp_dir().join(format!("improve-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = root.join("repo");
        let upper = root.join("upper");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::create_dir_all(upper.join("src")).unwrap();
        std::fs::write(repo.join("src/a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(repo.join("src/b.rs"), "fn b() {}\n").unwrap();
        let id = ["-c", "user.name=t", "-c", "user.email=t@localhost"];
        git(&repo, &["init", "-q"]).unwrap();
        git(&repo, &["add", "-A"]).unwrap();
        let mut c: Vec<&str> = id.to_vec();
        c.extend(["commit", "-qm", "one"]);
        git(&repo, &c).unwrap();
        std::fs::write(upper.join("src/a.rs"), "fn a() { b() }\n").unwrap();
        std::fs::write(upper.join("src/b.rs"), "fn b() {}\n").unwrap();
        let cfg = ImproveConfig {
            repo: repo.clone(),
            upper: upper.clone(),
            root: root.join("improve"),
            mirror: root.join("mirror"),
        };
        let base = head(&repo).unwrap();
        let ch = changes(&cfg, &base).unwrap();
        assert_eq!(ch.len(), 1, "only the file that differs");
        assert_eq!(ch[0].0, "src/a.rs");
        let dir = cfg.root.join("cand-1");
        stage(&dir, &cfg, &base, &ch).unwrap();
        let patch = std::fs::read_to_string(dir.join("change.patch")).unwrap();
        assert!(
            patch.contains("-fn a() {}") && patch.contains("+fn a() { b() }"),
            "{patch}"
        );
        std::fs::write(upper.join("Cargo.toml"), "[package]\n").unwrap();
        let o = attempt(&cfg, 2, "touches the manifest", None);
        assert_eq!(o.verdict, Verdict::Refused);
        assert!(o.summary.contains("Cargo.toml"), "{}", o.summary);
        // The diff names the repository's paths, and flags the denied one.
        let d = diff_text(&cfg).unwrap();
        assert!(
            d.contains("a/src/a.rs") && d.contains("+fn a() { b() }"),
            "{d}"
        );
        assert!(d.contains("propose refuses it"), "{d}");
        // Revert: a path outside the layer is refused, a copy is dropped, and
        // the diff no longer holds it.
        for bad in ["../repo/src/a.rs", "/etc/hostname", "", "src/../../x"] {
            assert!(revert(&cfg, bad).is_err(), "{bad}");
        }
        assert!(revert(&cfg, "Cargo.toml").is_ok());
        assert!(revert(&cfg, "src/a.rs").is_ok());
        assert!(
            revert(&cfg, "src/a.rs").is_err(),
            "no change left to revert"
        );
        assert!(
            repo.join("src/a.rs").is_file(),
            "the repository's file stays"
        );
        let d = diff_text(&cfg).unwrap();
        assert!(d.contains("changes nothing"), "{d}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
