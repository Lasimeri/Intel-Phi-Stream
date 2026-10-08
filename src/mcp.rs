//! `phi-stream mcp`: the management interface as a Model Context Protocol
//! server on stdin and stdout, so an agent (Claude Code) uses it the way
//! a person does: the real terminal interface (`phi-stream tui --follow`)
//! runs in a tmux session, the agent reads its screen and types into it;
//! and the stream's own channel to Claude (`ask`, `inbox`). The person can
//! watch the same screen: `tmux attach -t phi-stream-mcp`. See mcp.md.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command as Proc;
use std::time::Duration;

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::client::{self, Client};

/// The tmux session the interface runs in, and its size.
const SESSION: &str = "phi-stream-mcp";
const COLS: u32 = 200;
const ROWS: u32 = 60;
/// How long after input the screen is read (the interface redraws at
/// its next tick).
const SETTLE_MS: u64 = 300;

/// The protocol versions it speaks; the client's own is answered when it
/// is one of them.
const VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

pub fn run(socket: &Path) -> Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                reply(&mut out, &rpc_error(Value::Null, -32700, &e.to_string()))?;
                continue;
            }
        };
        // A notification (no id) is answered with nothing.
        let Some(id) = msg.get("id").cloned() else {
            continue;
        };
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let answer = match method {
            "initialize" => {
                let asked = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or(VERSIONS[0]);
                let version = if VERSIONS.contains(&asked) {
                    asked
                } else {
                    VERSIONS[0]
                };
                json!({"jsonrpc": "2.0", "id": id, "result": {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "phi-stream", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": INSTRUCTIONS,
                }})
            }
            "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
            "tools/list" => json!({"jsonrpc": "2.0", "id": id, "result": {"tools": tools()}}),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let (text, is_error) = match call(socket, name, &args) {
                    Ok(t) => (t, false),
                    Err(e) => (format!("{e:#}"), true),
                };
                json!({"jsonrpc": "2.0", "id": id, "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": is_error,
                }})
            }
            other => rpc_error(id, -32601, &format!("no method {other}")),
        };
        reply(&mut out, &answer)?;
    }
    Ok(())
}

const INSTRUCTIONS: &str = "The management interface of the running phi-stream service (one model reasoning without pause; its terminal interface, the same one a person uses). screen reads it; type and keys type into it (a line typed and Enter is sent to the stream as the person's words, a line starting with / is a command: /objective TEXT, /chain on|off|against|audit, /goal on|off, /inject on|off, /help; Tab moves between panes, PageUp and PageDown scroll). ask sends the stream a message from Claude and waits for its answer (its tell_claude tool); inbox reads what it sent Claude on its own. The person can watch the same screen with: tmux attach -t phi-stream-mcp";

fn tools() -> Value {
    json!([
        {
            "name": "screen",
            "description": "Read the management interface's screen as a person sees it (the real terminal interface, in a tmux session started on the first call). Optionally wait first, to see it move.",
            "inputSchema": {"type": "object", "properties": {
                "wait_ms": {"type": "integer", "description": "Milliseconds to wait before reading (at most 60000)."}
            }}
        },
        {
            "name": "type",
            "description": "Type text into the interface's input line, then Enter (unless enter is false); the screen after. A line is said to the stream as the person; a line starting with / is a command (/objective TEXT, /chain on|off|against|audit, /goal on|off, /inject on|off, /help).",
            "inputSchema": {"type": "object", "properties": {
                "text": {"type": "string"},
                "enter": {"type": "boolean", "description": "Press Enter after (default true)."}
            }, "required": ["text"]}
        },
        {
            "name": "keys",
            "description": "Press keys in the interface, in tmux names (Tab, BTab, Enter, Escape, PageUp, PageDown, End, Up, Down, C-c, or a character); the screen after.",
            "inputSchema": {"type": "object", "properties": {
                "keys": {"type": "array", "items": {"type": "string"}}
            }, "required": ["keys"]}
        },
        {
            "name": "ask",
            "description": "Send the stream a message from Claude and wait for its answer (its tell_claude naming the message), which comes at its next turn; the answer's text, its id and the wait.",
            "inputSchema": {"type": "object", "properties": {
                "text": {"type": "string"},
                "timeout_s": {"type": "integer", "description": "Seconds to wait (default 180, at most 570)."}
            }, "required": ["text"]}
        },
        {
            "name": "say",
            "description": "Say something to the stream as Claude, without waiting for an answer.",
            "inputSchema": {"type": "object", "properties": {
                "text": {"type": "string"}
            }, "required": ["text"]}
        },
        {
            "name": "inbox",
            "description": "The messages the stream sent Claude (tell_claude), from its workspace's to-claude.md: those after message number since (default: the last 10).",
            "inputSchema": {"type": "object", "properties": {
                "since": {"type": "integer"}
            }}
        },
        {
            "name": "status",
            "description": "The service's status line (state, rates, context cells, checks).",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ])
}

fn call(socket: &Path, name: &str, args: &Value) -> Result<String> {
    let int = |k: &str| args.get(k).and_then(Value::as_u64);
    let text = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    match name {
        "screen" => {
            ensure_session(socket)?;
            std::thread::sleep(Duration::from_millis(
                int("wait_ms").unwrap_or(0).min(60_000),
            ));
            capture()
        }
        "type" => {
            ensure_session(socket)?;
            let t = text("text");
            if !t.is_empty() {
                tmux(&["send-keys", "-t", SESSION, "-l", &t])?;
            }
            if args.get("enter").and_then(Value::as_bool).unwrap_or(true) {
                tmux(&["send-keys", "-t", SESSION, "Enter"])?;
            }
            std::thread::sleep(Duration::from_millis(SETTLE_MS));
            capture()
        }
        "keys" => {
            ensure_session(socket)?;
            let keys: Vec<String> = args
                .get("keys")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            if keys.is_empty() {
                bail!("keys needs at least one key");
            }
            let mut argv = vec!["send-keys", "-t", SESSION];
            argv.extend(keys.iter().map(String::as_str));
            tmux(&argv)?;
            std::thread::sleep(Duration::from_millis(SETTLE_MS));
            capture()
        }
        "ask" => {
            let t = text("text");
            if t.trim().is_empty() {
                bail!("ask needs a text");
            }
            let secs = int("timeout_s").unwrap_or(180).clamp(1, 570);
            let t0 = std::time::Instant::now();
            let (id, answer) = client::ask_claude(socket, &t, Duration::from_secs(secs))?;
            Ok(match answer {
                Some(m) => format!(
                    "{id} answered by m{} after {:.1} s:\n{}",
                    m.id,
                    t0.elapsed().as_secs_f32(),
                    m.text
                ),
                None => format!(
                    "no answer to {id} within {secs} s (it answers at its next turn; inbox shows it when it comes)"
                ),
            })
        }
        "say" => {
            let t = text("text");
            if t.trim().is_empty() {
                bail!("say needs a text");
            }
            Client::connect(socket)?.ask(&format!("say-as Claude {}", client::escape(&t)))
        }
        "inbox" => inbox(socket, int("since")),
        "status" => {
            let mut c = Client::connect(socket)?;
            c.send("status")?;
            // The status line comes before the ok.
            let mut last = String::new();
            while let Some(l) = c.line()? {
                if l.starts_with("ok ") || l.starts_with("err ") {
                    break;
                }
                if !l.starts_with("info ") {
                    last = l;
                }
            }
            Ok(last)
        }
        other => bail!("no tool {other}"),
    }
}

/// The interface in its tmux session, started when it is not running:
/// this very binary's `tui --follow` (it reloads onto each new build),
/// on this socket.
fn ensure_session(socket: &Path) -> Result<()> {
    if tmux(&["has-session", "-t", SESSION]).is_ok() {
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    let lang = std::env::var("LANG").unwrap_or_else(|_| "C.UTF-8".into());
    let cmd = format!("{} --socket {} tui --follow", quote(&exe), quote(socket));
    tmux(&[
        "new-session",
        "-d",
        "-s",
        SESSION,
        "-x",
        &COLS.to_string(),
        "-y",
        &ROWS.to_string(),
        "-e",
        &format!("LANG={lang}"),
        &cmd,
    ])?;
    // Its first frame.
    std::thread::sleep(Duration::from_millis(800));
    Ok(())
}

fn quote(p: &Path) -> String {
    format!("'{}'", p.display().to_string().replace('\'', "'\\''"))
}

fn capture() -> Result<String> {
    let s = tmux(&["capture-pane", "-p", "-t", SESSION])?;
    Ok(s.trim_end().to_string())
}

fn tmux(args: &[&str]) -> Result<String> {
    let out = Proc::new("tmux").args(args).output()?;
    if !out.status.success() {
        bail!(
            "tmux {}: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// What it sent Claude, from `to-claude.md` in the service's workspace
/// (named in its `info` line): the messages after number `since`, or the
/// last ten.
fn inbox(socket: &Path, since: Option<u64>) -> Result<String> {
    let mut c = Client::connect(socket)?;
    let workspace = match c.line()?.map(|l| client::parse(&l)) {
        Some(client::Msg::Info(i)) => PathBuf::from(i.workspace),
        _ => bail!("the service did not say where its workspace is"),
    };
    let text = std::fs::read_to_string(workspace.join("to-claude.md")).unwrap_or_default();
    let mut msgs: Vec<(u64, String)> = Vec::new();
    for block in text.split("\n## ").map(|b| b.trim_start_matches("## ")) {
        let id = block
            .strip_prefix('m')
            .and_then(|r| r.split(|ch: char| !ch.is_ascii_digit()).next())
            .and_then(|n| n.parse().ok());
        if let Some(id) = id {
            msgs.push((id, format!("## {}", block.trim())));
        }
    }
    let shown: Vec<&String> = match since {
        Some(s) => msgs
            .iter()
            .filter(|(i, _)| *i > s)
            .map(|(_, b)| b)
            .collect(),
        None => msgs.iter().rev().take(10).rev().map(|(_, b)| b).collect(),
    };
    Ok(if shown.is_empty() {
        "no messages".to_string()
    } else {
        shown.into_iter().cloned().collect::<Vec<_>>().join("\n\n")
    })
}

fn reply(out: &mut impl Write, v: &Value) -> Result<()> {
    writeln!(out, "{v}")?;
    out.flush()?;
    Ok(())
}

fn rpc_error(id: Value, code: i32, msg: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_a_schema() {
        let t = tools();
        let names: Vec<&str> = t
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["screen", "type", "keys", "ask", "say", "inbox", "status"]
        );
        for x in t.as_array().unwrap() {
            assert_eq!(x["inputSchema"]["type"], "object");
        }
    }
}
