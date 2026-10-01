#!/usr/bin/env bash
# phi-stream.sh: the service and its clients. `start` runs the service in
# a tmux session (with the cards when the co-processor repository is
# found: its phi-ggml.sh starts the workers and names the backend; on the
# GPU and the host alone otherwise); `attach` opens the terminal to it;
# everything else passes through to the binary (`say`, `feed`, `tail`,
# `status`, `persona`, `chunk`, `temp`, `pause`, `resume`, `quit`, `probe`,
# `gate`, `run`, `serve` in the foreground). See phi-stream.md.
#
#   scripts/phi-stream.sh start [serve options]   # the service, in tmux session phi-stream
#   scripts/phi-stream.sh attach                  # the terminal (Ctrl-C leaves it running)
#   scripts/phi-stream.sh say "hello"             # a line to it
#   scripts/phi-stream.sh tail                    # the stream on stdout
#   scripts/phi-stream.sh stop                    # quit the service and end the session
#   scripts/phi-stream.sh lens check              # the lens gates (with the cards)
#
# Model options (`--gpu-blocks`, `-c`, ...) may come before or after the
# subcommand.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
# PHI_STREAM_BIN names another build (a measurement pinned to a frozen
# binary while the tree is rebuilt).
bin="${PHI_STREAM_BIN:-$root/target/release/phi-stream}"
session=${PHI_STREAM_SESSION:-phi-stream}
[ -x "$bin" ] || { echo "$0: $bin not built; run make build" >&2; exit 1; }
. "$here/avx512.sh"

# The binary, with the cards when the co-processor repository is found.
launch() {
    if [ -n "${PHI_AVX512_ROOT:-}" ]; then
        export PHI_GGML_OFFLOAD="${PHI_GGML_OFFLOAD:-1}"
        exec "$PHI_AVX512_ROOT/scripts/phi-ggml.sh" "$bin" "$@"
    fi
    echo "$0: Intel-Phi-AVX512 not found; running on the GPU and the host alone (scripts/avx512.md)" >&2
    exec "$bin" "$@"
}

# The subcommand: the first word that is neither an option nor an option's
# value (the model options may come before it: they are global).
sub=""
skip=0
for a in "$@"; do
    if [ "$skip" = 1 ]; then
        skip=0
        continue
    fi
    case "$a" in
        -m|--model|--backend-dir|-c|--ctx|--batch|--gpu-blocks|-t|--threads|--n-seq|--temp|--top-k|--top-p|--seed|--repeat-penalty|--repeat-last-n|--socket) skip=1 ;;
        -*) ;;
        *) sub=$a; break ;;
    esac
done

case "$sub" in
    start)
        # Everything but the word `start`, given to `serve`.
        args=()
        dropped=0
        for a in "$@"; do
            if [ "$dropped" = 0 ] && [ "$a" = start ]; then
                dropped=1
                continue
            fi
            args+=("$a")
        done
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "$0: the service is already running in tmux session $session (attach, or stop)" >&2
            exit 1
        fi
        # The session runs this script's own launch path, so the cards are found
        # the same way; every word quoted, since the checkout's path may hold spaces.
        cmd=$(printf '%q ' "$here/phi-stream.sh" serve "${args[@]}")
        tmux new-session -d -s "$session" "$cmd"
        echo "started the service in tmux session $session; log: tmux attach -t $session; the terminal: $0 attach"
        ;;
    stop)
        "$bin" quit 2>/dev/null || true
        sleep 2
        tmux kill-session -t "$session" 2>/dev/null || true
        echo "stopped"
        ;;
    attach)
        exec "$bin" tui
        ;;
    serve|probe|gate|run|lens|code)
        launch "$@"
        ;;
    "")
        echo "usage: $0 start|stop|attach|say|feed|tail|status|persona|chunk|temp|pause|resume|quit|serve|probe|gate|run|lens|code ..." >&2
        exit 2
        ;;
    *)
        exec "$bin" "$@"
        ;;
esac
