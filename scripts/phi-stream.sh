#!/usr/bin/env bash
# phi-stream.sh: run phi-stream with the cards when the co-processor
# repository is found (its phi-ggml.sh starts the workers and names the
# backend), on the GPU and the host alone otherwise. The binary is this
# repository's release build. See phi-stream.md.
#
#   scripts/phi-stream.sh tui                 # the terminal
#   scripts/phi-stream.sh run < lines.txt     # stdout and stdin
#   scripts/phi-stream.sh probe               # the rates on this machine
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
bin="$root/target/release/phi-stream"
[ -x "$bin" ] || { echo "$0: $bin not built; run make build" >&2; exit 1; }
. "$here/avx512.sh"
if [ -n "${PHI_AVX512_ROOT:-}" ]; then
    # The cards' rows leave host memory after the upload (PHI_GGML_OFFLOAD),
    # unless the caller says otherwise.
    export PHI_GGML_OFFLOAD="${PHI_GGML_OFFLOAD:-1}"
    exec "$PHI_AVX512_ROOT/scripts/phi-ggml.sh" "$bin" "$@"
fi
echo "$0: Intel-Phi-AVX512 not found; running on the GPU and the host alone (scripts/avx512.md)" >&2
exec "$bin" "$@"
