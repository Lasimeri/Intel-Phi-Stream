# phi-stream-mcp.sh

The management interface (`phi-stream mcp`, `src/mcp.md`) for Claude Code,
wherever the service runs. With the harness on another machine (the GPU
rack, 2026-10-07) the interface has to run there too: its `status`, `say`
and `ask` go to the service's socket, `inbox` reads `to-claude.md` in the
service's workspace, and `screen`, `type` and `keys` drive a terminal
interface in a tmux session, all of them local to the service.

- The service on this machine first: when its socket
  (`PHI_STREAM_SOCKET`, else `$XDG_RUNTIME_DIR/phi-stream.sock`) is there,
  or no remote is named, this checkout's `phi-stream mcp` runs.
- Else `PHI_STREAM_MCP_HOST`, a command that runs one remote command with
  stdin and stdout passed through (`ssh HOST`, or a wrapper such as the
  person's `rack`), runs `phi-stream mcp` from the checkout at
  `PHI_STREAM_MCP_DIR` (default: this checkout's path) on that machine. The
  remote command is one string with the path in single quotes, which both
  POSIX shells and fish read.
- The choice is made when Claude Code starts the server: a service that
  moves to another machine needs `/mcp` to reconnect.
- On the remote machine the person watches the interface with `tmux attach
  -t phi-stream-mcp` there (over `ssh -t`).

Registration, once:

```sh
claude mcp add --scope user phi-stream \
    -e PHI_STREAM_MCP_HOST="ssh HOST" -e PHI_STREAM_MCP_DIR="/path/to/checkout" \
    -- "<checkout>/scripts/phi-stream-mcp.sh"
```

Checked 2026-10-07 by hand against the rack's running service: `initialize`,
then `status` (its status line) and `inbox` (its `to-claude.md`) over the
ssh connection.
