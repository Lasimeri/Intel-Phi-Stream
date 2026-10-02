//! `phi-stream code`: whether the stream writes code that works. The
//! tasks are MultiPL-E's HumanEval in Rust (`scripts/fetch-code-eval.sh`):
//! a prompt (doc comment and signature), the tests to append. A program is
//! assembled, compiled with `rustc` and run, each step sandboxed (no
//! network, a user namespace of its own, limits on memory, CPU time, file
//! size and wall time) in a run directory on disk, and classified: no code,
//! compile error, test failure, timeout, pass.
//!
//! `anchor` is the benchmark's own protocol, the number comparable with
//! published ones: the raw prompt, no chat template, greedy, stopped at the
//! dataset's stop sequence. See code.md.

use std::fs;
use std::io::Write as _;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context as _, Result};
use serde_json::Value;

use crate::llm::{Lane, Llm};

pub struct Task {
    pub name: String,
    pub prompt: String,
    pub tests: String,
    pub stops: Vec<String>,
}

pub fn load_tasks(path: &str) -> Result<Vec<Task>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading {path} (scripts/fetch-code-eval.sh fetches it)"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let v: Value =
            serde_json::from_str(line).with_context(|| format!("{path}:{}: not JSON", i + 1))?;
        let s = |k: &str| -> Result<String> {
            Ok(v.get(k)
                .and_then(Value::as_str)
                .with_context(|| format!("{path}:{}: no {k}", i + 1))?
                .to_string())
        };
        out.push(Task {
            name: s("name")?,
            prompt: s("prompt")?,
            tests: s("tests")?,
            stops: v
                .get("stop_tokens")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        });
    }
    Ok(out)
}

/// What became of one task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    NoCode,
    CompileError,
    TestFailure,
    Timeout,
    Pass,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::NoCode => "no code",
            Verdict::CompileError => "compile error",
            Verdict::TestFailure => "test failure",
            Verdict::Timeout => "timeout",
            Verdict::Pass => "pass",
        }
    }
}

/// How a sandboxed process ended.
pub struct Ran {
    pub status: Option<i32>,
    pub timed_out: bool,
    pub output: String,
}

/// Limits for one sandboxed process.
pub struct Limits {
    pub wall: Duration,
    pub cpu_secs: u64,
    pub mem_bytes: u64,
}

/// Run `argv` in `dir` with no network (`unshare -rn`: a user namespace and
/// a network namespace with only loopback), in a process group of its own,
/// with address space, CPU time, file size and core limits, killed (the
/// whole group) at the wall limit. Output kept up to 64 KiB.
pub fn sandboxed(argv: &[&str], dir: &Path, lim: &Limits) -> Result<Ran> {
    let out_path = dir.join(".out");
    let out = fs::File::create(&out_path)?;
    let err = out.try_clone()?;
    let mut cmd = Command::new("unshare");
    cmd.args(["-rn", "--"])
        .args(argv)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    let (mem, cpu) = (lim.mem_bytes, lim.cpu_secs);
    // SAFETY: only async-signal-safe calls between fork and exec.
    unsafe {
        cmd.pre_exec(move || {
            let set = |r, v: u64| {
                let l = libc::rlimit {
                    rlim_cur: v,
                    rlim_max: v,
                };
                libc::setrlimit(r, &l)
            };
            set(libc::RLIMIT_AS, mem);
            set(libc::RLIMIT_CPU, cpu);
            set(libc::RLIMIT_FSIZE, 64 << 20);
            set(libc::RLIMIT_CORE, 0);
            libc::setpgid(0, 0);
            Ok(())
        });
    }
    let mut child = cmd.spawn().context("starting unshare (util-linux)")?;
    let pid = child.id() as i32;
    let t0 = Instant::now();
    let (status, timed_out) = loop {
        if let Some(st) = child.try_wait()? {
            break (st.code(), false);
        }
        if t0.elapsed() >= lim.wall {
            // SAFETY: the child leads its own process group.
            unsafe { libc::kill(-pid, libc::SIGKILL) };
            let _ = child.wait();
            break (None, true);
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut output = fs::read_to_string(&out_path).unwrap_or_default();
    output.truncate(output.len().min(64 << 10));
    Ok(Ran {
        status,
        timed_out,
        output,
    })
}

/// Compile `source` and run it; the verdict and the output of the step
/// that decided it.
pub fn judge(source: &str, dir: &Path) -> Result<(Verdict, String)> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("prog.rs"), source)?;
    let compile = sandboxed(
        &[
            "rustc",
            "--edition",
            "2021",
            "-A",
            "warnings",
            "-C",
            "debuginfo=0",
            "-o",
            "prog",
            "prog.rs",
        ],
        dir,
        &Limits {
            wall: Duration::from_secs(120),
            cpu_secs: 120,
            mem_bytes: 4 << 30,
        },
    )?;
    if compile.timed_out {
        return Ok((Verdict::Timeout, compile.output));
    }
    if compile.status != Some(0) {
        return Ok((Verdict::CompileError, compile.output));
    }
    let run = sandboxed(
        &["./prog"],
        dir,
        &Limits {
            wall: Duration::from_secs(10),
            cpu_secs: 10,
            mem_bytes: 1 << 30,
        },
    )?;
    if run.timed_out {
        return Ok((Verdict::Timeout, run.output));
    }
    if run.status != Some(0) {
        return Ok((Verdict::TestFailure, run.output));
    }
    Ok((Verdict::Pass, run.output))
}

/// The completion up to the first stop sequence (the stop excluded), as the
/// benchmark's harness cuts it.
pub fn cut_at_stop<'a>(text: &'a str, stops: &[String]) -> &'a str {
    let end = stops
        .iter()
        .filter_map(|s| text.find(s.as_str()))
        .min()
        .unwrap_or(text.len());
    &text[..end]
}

/// One task by the benchmark's protocol: the raw prompt, greedy, until a
/// stop sequence, the end of generation or `max_tokens`.
fn complete_raw(
    llm: &mut Llm,
    prompt: &str,
    stops: &[String],
    max_tokens: usize,
) -> Result<(String, usize)> {
    let tokens = llm.tokenize(prompt, false)?;
    llm.clear();
    let cap = llm.batch_cap();
    let n = tokens.chunks(cap).count();
    let mut row = 0;
    let mut pos = 0i32;
    for (i, c) in tokens.chunks(cap).enumerate() {
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: c,
            pos0: pos,
            logits: i + 1 == n,
        }])?;
        if let Some(&r) = rows.first() {
            row = r;
        }
        pos += c.len() as i32;
    }
    let mut out: Vec<i32> = Vec::new();
    let mut text = String::new();
    for _ in 0..max_tokens {
        let t = llm.greedy(row, false)?;
        if llm.is_eog(t) {
            break;
        }
        out.push(t);
        text = llm.text_plain(&out);
        if stops.iter().any(|s| text.contains(s.as_str())) {
            break;
        }
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: &[t],
            pos0: pos,
            logits: true,
        }])?;
        row = rows[0];
        pos += 1;
    }
    Ok((text, out.len()))
}

/// What a run gave, task by task.
pub struct Outcome {
    pub name: String,
    pub verdict: Verdict,
    pub tokens: usize,
    pub secs: f64,
}

pub fn summary(outcomes: &[Outcome]) -> String {
    let n = outcomes.len().max(1);
    let count = |v: Verdict| outcomes.iter().filter(|o| o.verdict == v).count();
    let mut s = format!(
        "{} tasks: pass {} ({:.1} %)",
        outcomes.len(),
        count(Verdict::Pass),
        100.0 * count(Verdict::Pass) as f64 / n as f64
    );
    for v in [
        Verdict::TestFailure,
        Verdict::CompileError,
        Verdict::Timeout,
        Verdict::NoCode,
    ] {
        s.push_str(&format!(", {} {}", v.name(), count(v)));
    }
    let toks: usize = outcomes.iter().map(|o| o.tokens).sum();
    let secs: f64 = outcomes.iter().map(|o| o.secs).sum();
    s.push_str(&format!(
        "; {:.0} tokens a task on average, {:.1} s a task",
        toks as f64 / n as f64,
        secs / n as f64
    ));
    for v in [
        Verdict::CompileError,
        Verdict::TestFailure,
        Verdict::Timeout,
        Verdict::NoCode,
    ] {
        let names: Vec<&str> = outcomes
            .iter()
            .filter(|o| o.verdict == v)
            .map(|o| o.name.as_str())
            .collect();
        if !names.is_empty() {
            s.push_str(&format!("\n{}: {}", v.name(), names.join(" ")));
        }
    }
    s
}

/// A run directory on disk: `~/.cache/phi-stream/code/<label>-<time>`.
pub fn run_dir(label: &str) -> Result<PathBuf> {
    let base = PathBuf::from(crate::expand_home("~/.cache/phi-stream/code"));
    let d = base.join(format!("{label}-{}", crate::clock::now_us() / 1_000_000));
    fs::create_dir_all(&d)?;
    fs::write(d.join("EVALUATION-ONLY.txt"), EVALUATION_ONLY)?;
    Ok(d)
}

/// Marked into every run directory: MultiPL-E's licence (clause 4) forbids
/// its contents as training data, and training on an evaluation set voids
/// the evaluation.
pub const EVALUATION_ONLY: &str =
    "EVALUATION ONLY. Everything here (the tasks, the model's completions, any\n\
reflection episodes) comes from MultiPL-E's HumanEval-rs, whose licence (BSD 3-Clause\n\
with a machine learning restriction, clause 4) forbids its use as training data for any\n\
machine learning model. Never put any of it into a fine-tune: it would break the licence\n\
and void the measurement.\n";

/// `code anchor`: the benchmark's protocol over `tasks`.
pub fn anchor(
    llm: &mut Llm,
    tasks: &[Task],
    max_tokens: usize,
    dir: &Path,
) -> Result<Vec<Outcome>> {
    let mut log = fs::File::create(dir.join("results.jsonl"))?;
    let mut outcomes = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let t0 = Instant::now();
        let (text, n) = complete_raw(llm, &t.prompt, &t.stops, max_tokens)?;
        let body = cut_at_stop(&text, &t.stops);
        let (verdict, output) = if body.trim().is_empty() {
            (Verdict::NoCode, String::new())
        } else {
            let source = format!("{}{}{}", t.prompt, body, t.tests);
            judge(&source, &dir.join(&t.name))?
        };
        let secs = t0.elapsed().as_secs_f64();
        writeln!(
            log,
            "{}",
            serde_json::json!({"name": t.name, "verdict": verdict.name(), "tokens": n, "secs": secs, "completion": body, "output": output})
        )?;
        println!(
            "{:>3}/{} {:<44} {:<13} {:>4} tokens {:>5.1} s",
            i + 1,
            tasks.len(),
            t.name,
            verdict.name(),
            n,
            secs
        );
        outcomes.push(Outcome {
            name: t.name.clone(),
            verdict,
            tokens: n,
            secs,
        });
    }
    Ok(outcomes)
}

/// The tasks to run: all, the first `n`, or the named ones.
pub fn select(tasks: Vec<Task>, first: Option<usize>, only: &[String]) -> Result<Vec<Task>> {
    if !only.is_empty() {
        let v: Vec<Task> = tasks
            .into_iter()
            .filter(|t| only.contains(&t.name))
            .collect();
        if v.len() != only.len() {
            bail!("{} of the {} named tasks found", v.len(), only.len());
        }
        return Ok(v);
    }
    Ok(match first {
        Some(n) => tasks.into_iter().take(n).collect(),
        None => tasks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_are_cut_at_the_first_stop() {
        let stops = vec!["\n}".to_string()];
        assert_eq!(
            cut_at_stop("    x + 1\n}\nfn other() {}", &stops),
            "    x + 1"
        );
        assert_eq!(cut_at_stop("    x + 1", &stops), "    x + 1");
    }

    #[test]
    fn the_sandbox_judges_programs() {
        let base =
            std::env::temp_dir().join(format!("phi-stream-code-test-{}", std::process::id()));
        let ok = "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\nfn main() { assert_eq!(add(2, 3), 5); }\n";
        assert_eq!(judge(ok, &base.join("ok")).unwrap().0, Verdict::Pass);
        let wrong = "fn add(a: i64, b: i64) -> i64 {\n    a - b\n}\nfn main() { assert_eq!(add(2, 3), 5); }\n";
        assert_eq!(
            judge(wrong, &base.join("wrong")).unwrap().0,
            Verdict::TestFailure
        );
        let broken = "fn add(a: i64, b: i64) -> i64 {\n    a +\n}\nfn main() {}\n";
        assert_eq!(
            judge(broken, &base.join("broken")).unwrap().0,
            Verdict::CompileError
        );
        let net = "fn main() { assert!(std::net::TcpStream::connect(\"1.1.1.1:80\").is_err()); }\n";
        assert_eq!(
            judge(net, &base.join("net")).unwrap().0,
            Verdict::Pass,
            "the sandbox has no network"
        );
        let _ = fs::remove_dir_all(&base);
    }
}

/// How a task is put to the stream (`code stream`).
pub struct StreamOpts {
    /// The persona's base: the user's standing instructions or a neutral one.
    pub base: String,
    pub base_label: String,
    /// Thinking tokens before `</think>` is placed (0: no limit).
    pub think_budget: usize,
    /// The repetition penalty (1: off), recorded with the run.
    pub repeat_penalty: f32,
    /// Read the mind at every token (`mind.md`), when set.
    pub mind: Option<crate::mind::MindConfig>,
    /// Check the tokens it places (`reflect.md`; needs `mind`), when set.
    pub reflect: Option<crate::reflect::ReflectConfig>,
}

/// The user turn of a task.
pub fn task_prompt(t: &Task) -> String {
    format!(
        "Complete the Rust function below. Write the complete function, with its signature, in one ```rust code block. Do not write a main function or tests.\n\n```rust\n{}\n```",
        t.prompt.trim_end()
    )
}

/// The function a task asks for: the name on the prompt's last `fn` line.
pub fn function_name(prompt: &str) -> Option<String> {
    let line = prompt
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with("fn "))?;
    let rest = line.trim_start().strip_prefix("fn ")?;
    let end = rest.find(['(', '<']).unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

/// The code of an answer, by the rule written before the first run
/// (code.md): the last fenced block, which must define the task's
/// function; any `main` the model wrote removed.
pub fn extract(answer: &str, name: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut rest = answer;
    while let Some(i) = rest.find("```") {
        let after = &rest[i + 3..];
        // The fence's language tag runs to the end of its line.
        let body_start = after.find('\n').map(|n| n + 1).unwrap_or(after.len());
        let body = &after[body_start..];
        match body.find("```") {
            Some(j) => {
                blocks.push(body[..j].to_string());
                rest = &body[j + 3..];
            }
            None => break,
        }
    }
    let block = blocks.pop()?;
    if !block.contains(&format!("fn {name}")) {
        return None;
    }
    Some(remove_main(&block))
}

/// `block` without a `fn main` and its body (braces matched; braces inside
/// string literals of that body are not told apart, a stated limit).
pub fn remove_main(block: &str) -> String {
    let Some(start) = block.find("fn main(") else {
        return block.to_string();
    };
    let Some(open) = block[start..].find('{').map(|o| start + o) else {
        return block.to_string();
    };
    let mut depth = 0i32;
    for (k, c) in block[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = open + k + 1;
                    return format!("{}{}", &block[..start], &block[end..]);
                }
            }
            _ => {}
        }
    }
    block.to_string()
}

/// `code stream`: every task through the engine in its task mode (the chat
/// frame, thinking, greedy, nothing put in on the engine's account, a
/// throwaway workspace), the answer's code extracted and judged. The model
/// comes back for the caller.
pub fn stream(
    mut llm: Llm,
    tasks: &[Task],
    o: &StreamOpts,
    dir: &Path,
) -> Result<(Vec<Outcome>, Llm)> {
    use crate::engine::{compose_task, Command, Config, Engine, Event, Frame, Kind};
    use std::sync::mpsc;
    let mut log = fs::File::create(dir.join("results.jsonl"))?;
    let mut outcomes = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let t0 = Instant::now();
        llm.clear();
        let tdir = dir.join(&t.name);
        fs::create_dir_all(&tdir)?;
        let cfg = Config {
            frame: Frame::Chat,
            system: compose_task(&o.base),
            seed: task_prompt(t),
            first_words: String::new(),
            direct_max: 48,
            chunk: 0,
            rollover_at: 10.0,
            rollover_tokens: None,
            summary_max: 1024,
            sampling: crate::llm::Sampling {
                temp: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                dry_multiplier: 0.0,
                dry_base: 1.75,
                dry_allowed_length: 2,
                dry_last_n: -1,
                seed: 0,
                repeat_penalty: o.repeat_penalty,
                repeat_last_n: 256,
                ban_dashes: true,
            },
            status_every: 1_000_000,
            time_every_us: 0,
            nudge_every_us: i64::MAX,
            horizon_us: 0,
            task: true,
            think_budget: o.think_budget,
            workspace: tdir.join("ws"),
            mind: o.mind.clone(),
            reflect: o.reflect.clone(),
            dev: None,
            terminal: false,
            // A task's objective is the task.
            gate_output: false,
            summary_on_quit: false,
            second_chain: false,
        };
        let (etx, erx) = mpsc::channel();
        let (ctx, crx) = mpsc::channel::<Command>();
        let engine = Engine::new(llm, cfg, etx, crx)?;
        llm = engine.run()?;
        drop(ctx);
        let (mut answer, mut think_tokens, mut capped, mut thoughts) =
            (String::new(), 0usize, false, String::new());
        let mut episodes: Vec<crate::reflect::Episode> = Vec::new();
        for ev in erx.try_iter() {
            match ev {
                Event::Text(s, Kind::Speak, _) => answer.push_str(&s),
                Event::Text(s, Kind::Think, _) => thoughts.push_str(&s),
                Event::Done {
                    think_tokens: n,
                    capped: c,
                } => {
                    think_tokens = n;
                    capped = c;
                }
                Event::Reflect(e) => episodes.push(e),
                _ => {}
            }
        }
        let name = function_name(&t.prompt).unwrap_or_default();
        let (verdict, output, code) = match extract(&answer, &name) {
            None => (Verdict::NoCode, String::new(), String::new()),
            Some(code) => {
                let tests = t.tests.trim_start().strip_prefix('}').unwrap_or(&t.tests);
                let source = format!("{code}\n{tests}");
                let (v, out) = judge(&source, &tdir)?;
                (v, out, code)
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        let answer_tokens = llm.tokenize(&answer, false)?.len();
        writeln!(
            log,
            "{}",
            serde_json::json!({"name": t.name, "verdict": verdict.name(), "think_tokens": think_tokens, "think_capped": capped,
                "answer_tokens": answer_tokens, "secs": secs, "code": code, "answer": answer, "output": output, "thoughts": thoughts,
                "episodes": episodes.iter().map(crate::reflect::line).collect::<Vec<_>>()})
        )?;
        let checks = if o.reflect.is_some() {
            let changed = episodes
                .iter()
                .filter(|e| e.outcome == crate::reflect::Outcome::Changed)
                .count();
            format!(" checks {:>2} changed {changed}", episodes.len())
        } else {
            String::new()
        };
        println!(
            "{:>3}/{} {:<44} {:<13} think {:>5}{} answer {:>4} {:>6.1} s{checks}",
            i + 1,
            tasks.len(),
            t.name,
            verdict.name(),
            think_tokens,
            if capped { " (capped)" } else { "" },
            answer_tokens,
            secs
        );
        outcomes.push(Outcome {
            name: t.name.clone(),
            verdict,
            tokens: think_tokens + answer_tokens,
            secs,
        });
    }
    Ok((outcomes, llm))
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn the_function_and_its_code_are_found() {
        let prompt = "/// Add.\nfn add_two<T>(a: T) -> T {\n";
        assert_eq!(function_name(prompt).as_deref(), Some("add_two"));
        let answer = "Here:\n```rust\nfn helper() {}\n```\nand\n```rust\nfn add_two(a: i64) -> i64 {\n    a + 2\n}\n\nfn main() {\n    println!(\"{}\", add_two(1));\n}\n```\n";
        let code = extract(answer, "add_two").unwrap();
        assert!(code.contains("fn add_two(a: i64)"));
        assert!(!code.contains("fn main"));
        assert!(extract("```rust\nfn other() {}\n```", "add_two").is_none());
        assert!(extract("no code here", "add_two").is_none());
    }
}
