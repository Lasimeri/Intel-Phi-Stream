# mcp.rs

`phi-stream mcp`: the management interface as a Model Context Protocol
server (JSON-RPC 2.0, one message a line on stdin and stdout), so an agent
(Claude Code) uses the interface the way a person does, and so what it
develops for the interface it also sees.

- The interface is the real one: `phi-stream tui --follow` (this very
  binary, on the same socket) in a tmux session named `phi-stream-mcp`,
  200 by 60, started at the first tool call that needs it and left
  running. The person can watch what the agent sees and types:
  `tmux attach -t phi-stream-mcp` (detach with C-b d). `--follow`
  reloads it onto each new build, as on the desktop.
- Tools:

| tool | what |
| --- | --- |
| `screen` | the screen as text (`tmux capture-pane -p`), after `wait_ms` |
| `type` | text into the input line, then Enter (`enter: false` holds it); the screen after 300 ms |
| `keys` | keys by tmux name (Tab, BTab, PageUp, Escape, C-c, ...); the screen after |
| `ask` | a message from Claude that waits for its answer (`client::ask_claude`): the stream's `tell_claude` naming its id, the wait in seconds |
| `say` | a line as Claude (`say-as Claude`), no wait |
| `inbox` | what it sent Claude, from `to-claude.md` in the workspace its `info` line names: after number `since`, or the last ten |
| `status` | the service's status line |

- Register it once: `claude mcp add --scope user phi-stream --
  "<checkout>/target/release/phi-stream" mcp`. A session started before
  the registration sees it after `/mcp` reconnects.
- The protocol version answered is the client's own when it is one of
  2025-06-18, 2025-03-26, 2024-11-05, else the newest; notifications get no
  answer; an unknown method is `-32601`; a tool's failure is a result with
  `isError`, never a protocol error.

Tests: every tool is listed with an object schema, in order.
