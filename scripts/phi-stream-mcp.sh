#!/bin/bash
# phi-stream-mcp.sh: the management interface (phi-stream mcp) for Claude
# Code, wherever the service runs. See phi-stream-mcp.md.
#   phi-stream-mcp.sh        an MCP server on stdin and stdout
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
sock=${PHI_STREAM_SOCKET:-${XDG_RUNTIME_DIR:-/tmp}/phi-stream.sock}
# The service on this machine first; else the one PHI_STREAM_MCP_HOST
# reaches (a command that runs one remote command with stdin and stdout
# passed through, such as "ssh HOST"), its checkout at PHI_STREAM_MCP_DIR.
if [ -S "$sock" ] || [ -z "${PHI_STREAM_MCP_HOST:-}" ]; then
    exec "$here/target/release/phi-stream" mcp
fi
dir=${PHI_STREAM_MCP_DIR:-$here}
# shellcheck disable=SC2086 # the host command is words by design
exec $PHI_STREAM_MCP_HOST "'$dir/target/release/phi-stream' mcp"
