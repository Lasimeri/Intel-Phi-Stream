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
#   scripts/phi-stream.sh dev [serve options]     # the service developing this repo with Claude
#   scripts/phi-stream.sh attach [--follow]       # the terminal (Ctrl-C leaves it running; --follow reloads on each build)
#   scripts/phi-stream.sh window                  # the terminal in a desktop window (start opens one when the model runs)
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
# PHI_STREAM_INSTANCE=NAME: a second (third...) service beside the first,
# with its own session, socket, workspace and window; PHI_STREAM_CARD=N
# gives it one Phi card (each instance its own: one process per card).
inst=${PHI_STREAM_INSTANCE:-}
suffix=${inst:+-$inst}
session=${PHI_STREAM_SESSION:-phi-stream$suffix}
if [ -n "$inst" ]; then
    export PHI_STREAM_SOCKET="${PHI_STREAM_SOCKET:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/phi-stream$suffix.sock}"
fi
if [ -n "${PHI_STREAM_CARD:-}" ]; then
    export PHI_GGML_CARDS="$PHI_STREAM_CARD"
fi
[ -x "$bin" ] || { echo "$0: $bin not built; run make build" >&2; exit 1; }
. "$here/avx512.sh"

# The binary, with the cards when the co-processor repository is found.
launch() {
    # PHI_STREAM_CARDS=0: this instance leaves the cards alone (one process
    # holds them; a second model runs on the GPU and the host).
    if [ -n "${PHI_AVX512_ROOT:-}" ] && [ "${PHI_STREAM_CARDS:-1}" != 0 ]; then
        export PHI_GGML_OFFLOAD="${PHI_GGML_OFFLOAD:-1}"
        exec "$PHI_AVX512_ROOT/scripts/phi-ggml.sh" "$bin" "$@"
    fi
    echo "$0: Intel-Phi-AVX512 not found; running on the GPU and the host alone (scripts/avx512.md)" >&2
    exec "$bin" "$@"
}

# The desktop window (`window`): a terminal emulator running `attach
# --follow`. Its pid is kept, so one is open at a time; the keeper's pid
# too, so one keeper runs at a time.
rt=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
winpid="$rt/phi-stream$suffix-window.pid"
keeppid="$rt/phi-stream$suffix-window-keeper.pid"
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2> /dev/null; }
# Loaded and running: `status` prints a line (while it loads, none).
running() { [ -n "$("$bin" status 2> /dev/null || true)" ]; }
# The service process listening on this instance's socket, however it was
# started (in tmux by `start`, or outside it by a watchdog or a login).
sock=${PHI_STREAM_SOCKET:-$rt/phi-stream.sock}
serve_pid() { ss -xlpn 2> /dev/null | grep -F " $sock " | grep -oP 'pid=\K[0-9]+' | head -n 1; }
# The service asked to quit (it writes its summary first, src/engine.md)
# and waited for: its process gone and its tmux session ended, up to two
# minutes, then ended anyway. Waiting on the tmux session alone, a restart
# of a service started outside tmux began the next one beside it while it
# still wrote its summary, and the next ran out of GPU memory loading
# (2026-10-03).
quit_and_wait() {
    local p
    p=$(serve_pid)
    "$bin" quit 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! tmux has-session -t "=$session" 2>/dev/null && { [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; }; then
            return 0
        fi
        sleep 1
    done
    tmux kill-session -t "=$session" 2>/dev/null || true
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
        kill "$p" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$p" 2>/dev/null || return 0
            sleep 0.5
        done
        kill -9 "$p" 2>/dev/null || true
    fi
}
open_window() {
    alive "$winpid" && return 0
    # The desktop: this session's, or the user's own Wayland display when
    # started over SSH.
    if [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${DISPLAY:-}" ]; then
        if [ -S "$rt/wayland-0" ]; then
            export WAYLAND_DISPLAY=wayland-0
        else
            echo "$0: no desktop to open a window on; over SSH: $0 attach --follow" >&2
            return 1
        fi
    fi
    local term=${PHI_STREAM_TERMINAL:-} t
    if [ -z "$term" ]; then
        for t in konsole alacritty kitty foot xterm; do
            if command -v "$t" > /dev/null; then
                term=$t
                break
            fi
        done
    fi
    [ -n "$term" ] || { echo "$0: no terminal emulator found (set PHI_STREAM_TERMINAL)" >&2; return 1; }
    # konsole joins a running konsole process unless told not to; its own
    # process is the window, so the pid says whether it is open.
    local targs=(-e)
    [ "$(basename "$term")" = konsole ] && targs=(--separate -e)
    setsid "$term" "${targs[@]}" "$here/phi-stream.sh" attach --follow < /dev/null > /dev/null 2>&1 &
    echo $! > "$winpid"
    echo "opened the window ($term, pid $!)"
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
        -m|--model|--backend-dir|-c|--ctx|--batch|--gpu-blocks|-t|--threads|--n-seq|--temp|--top-k|--top-p|--min-p|--dry-multiplier|--dry-base|--dry-allowed-length|--dry-last-n|--seed|--repeat-penalty|--repeat-last-n|--socket) skip=1 ;;
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
        if tmux has-session -t "=$session" 2>/dev/null; then
            echo "$0: the service is already running in tmux session $session (attach, or stop)" >&2
            exit 1
        fi
        # The session runs this script's own launch path, so the cards are found
        # the same way; every word quoted, since the checkout's path may hold spaces.
        # Its output is also kept on disk (a failure's message outlives the
        # session; never tmpfs).
        log=${PHI_STREAM_LOG:-$HOME/.local/share/phi-stream/serve.log}
        mkdir -p "$(dirname "$log")"
        # The binary chosen here (PHI_STREAM_BIN) goes with it: the session
        # takes the tmux server's environment, not this shell's. So does
        # PHI_STREAM_PRELOAD, put in the service's LD_PRELOAD alone (a
        # watchdog's shim that lets a debugger take a stuck service's
        # stacks, without loosening every program the tmux server runs).
        pre=""
        [ -n "${PHI_STREAM_PRELOAD:-}" ] && pre="LD_PRELOAD=$(printf '%q' "$PHI_STREAM_PRELOAD")"
        cmd="env $pre PHI_STREAM_BIN=$(printf '%q' "$bin") $(printf '%q ' "$here/phi-stream.sh" serve "${args[@]}") 2>&1 | tee -a $(printf '%q' "$log")"
        tmux new-session -d -s "$session" "$cmd"
        echo "started the service in tmux session $session; log: $log (and tmux attach -t $session); the terminal: $0 attach"
        # The window: on the desktop whenever the model is loaded and
        # running, for as long as this session lasts (a keeper detached from
        # this shell; PHI_STREAM_WINDOW=0: none).
        if [ "${PHI_STREAM_WINDOW:-1}" != 0 ]; then
            setsid "$0" window --keep < /dev/null > /dev/null 2>&1 &
            echo "a window with the terminal opens on the desktop when the model is running ($0 window)"
        fi
        ;;
    window)
        # A terminal window on the desktop running `attach --follow`: the
        # diagnostics and the input, reconnecting across restarts and
        # reloading onto each build. Alone: open it now unless it is open.
        # `--keep` (what `start` runs): while the service's session lasts,
        # whenever the model is running and no window is open (not yet, or
        # closed), open one; checked every 3 s.
        keep=0
        for a in "$@"; do
            [ "$a" = --keep ] && keep=1
        done
        if [ "$keep" = 0 ]; then
            if alive "$winpid"; then
                echo "the window is already open (pid $(cat "$winpid"))"
                exit 0
            fi
            open_window
            exit
        fi
        alive "$keeppid" && exit 0
        echo $$ > "$keeppid"
        while tmux has-session -t "=$session" 2> /dev/null; do
            if ! alive "$winpid" && running; then
                open_window || true
            fi
            sleep 3
        done
        rm -f "$keeppid"
        ;;
    accept)
        # Files of the dev stream's working copy that were brought into the
        # repository, taken out of its copy (kept in its workspace under
        # accepted-DATE/), so it sees the repository's versions again: its
        # own copies shadowed them, and it reviewed Claude's fixes to its
        # tool backwards (src/term.md). Paths relative to the repository.
        ws=${PHI_STREAM_DEV_WORKSPACE:-$HOME/.local/share/phi-stream/dev$suffix}
        upper="$ws-copy/upper"
        keep="$ws/accepted-$(date +%Y-%m-%d)"
        shift_done=0
        for a in "$@"; do
            if [ "$shift_done" = 0 ] && [ "$a" = accept ]; then
                shift_done=1
                continue
            fi
            if [ -e "$upper/$a" ]; then
                mkdir -p "$keep/$(dirname "$a")"
                mv "$upper/$a" "$keep/$a"
                # Its directories left empty go too, never the layer itself.
                d=$(dirname "$upper/$a")
                while [ "$d" != "$upper" ] && rmdir "$d" 2>/dev/null; do
                    d=$(dirname "$d")
                done
                echo "accepted $a (its copy kept in $keep)"
            else
                echo "$a: not in the working copy ($upper)" >&2
            fi
        done
        ;;
    restart)
        # The service stopped (its summary written) and started again with
        # the words after `restart` (`dev` and its options, by default
        # `dev`), its window kept open: the terminal in it reconnects, so the
        # interface stays on the desktop through an update (`stop` then
        # `start` closed it for the whole load).
        rest=()
        dropped=0
        for a in "$@"; do
            if [ "$dropped" = 0 ] && [ "$a" = restart ]; then
                dropped=1
                continue
            fi
            rest+=("$a")
        done
        [ "${#rest[@]}" -gt 0 ] || rest=(dev)
        if alive "$keeppid"; then
            kill "$(cat "$keeppid")" 2> /dev/null || true
        fi
        rm -f "$keeppid"
        quit_and_wait
        echo "stopped; starting again, the window kept"
        exec "$0" "${rest[@]}"
        ;;
    stop)
        # The model stops, and its window with it (a restart by `quit` and
        # `start` keeps the window, which reconnects): the keeper first, so
        # it cannot reopen it while the service is going.
        if alive "$keeppid"; then
            kill "$(cat "$keeppid")" 2> /dev/null || true
        fi
        rm -f "$keeppid"
        # The service writes its summary before it stops (src/engine.md): up
        # to two minutes, then the session ends anyway.
        quit_and_wait
        if alive "$winpid"; then
            kill "$(cat "$winpid")" 2> /dev/null || true
        fi
        rm -f "$winpid"
        echo "stopped"
        ;;
    dev)
        # The service developing this repository with Claude (docs/dev.md):
        # its persona says so, its reads resolve here, its notes and
        # preferences live in their own workspace; the mind read and its
        # words checked. Everything but the word `dev` goes to `start`.
        rest=()
        dropped=0
        for a in "$@"; do
            if [ "$dropped" = 0 ] && [ "$a" = dev ]; then
                dropped=1
                continue
            fi
            rest+=("$a")
        done
        exec "$0" start --dev "$root" --mind --reflect --terminal --rollover-tokens 150000 --second-chain \
            --workspace "${PHI_STREAM_DEV_WORKSPACE:-$HOME/.local/share/phi-stream/dev$suffix}" "${rest[@]}"
        ;;
    attach)
        follow=0
        for a in "$@"; do
            [ "$a" = --follow ] && follow=1
        done
        [ "$follow" = 1 ] || exec "$bin" tui
        # `--follow`: the terminal reloads itself onto each new build
        # (src/tui.md). A build that dies is reported here, the terminal
        # reset (a crash can leave raw mode and the alternate screen), and
        # the next build run. Leaving it (Ctrl-C, /quit) ends this too.
        while :; do
            rc=0
            "$bin" tui --follow || rc=$?
            [ "$rc" = 0 ] && exit 0
            printf '\033[?1049l\033[0m\033[?25h'
            stty sane 2> /dev/null || true
            echo "$0: the terminal stopped (exit $rc); waiting for the next build of $bin (Ctrl-C leaves)" >&2
            id=$(stat -Lc '%i %s %Y' "$bin" 2> /dev/null || true)
            while [ "$(stat -Lc '%i %s %Y' "$bin" 2> /dev/null || true)" = "$id" ]; do
                sleep 1
            done
            sleep 1
        done
        ;;
    serve|probe|gate|run|lens|code)
        launch "$@"
        ;;
    "")
        echo "usage: $0 start|dev|restart|accept|stop|attach|window|say|feed|tail|status|persona|chunk|temp|pause|resume|quit|serve|probe|gate|run|lens|code ..." >&2
        exit 2
        ;;
    *)
        exec "$bin" "$@"
        ;;
esac
