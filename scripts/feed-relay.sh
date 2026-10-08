#!/bin/bash
# feed-relay.sh: the camera and microphone feeds written on this machine kept
# in the workspace of the service on another (the GPU rack). See feed-relay.md.
#
#   scripts/feed-relay.sh run       # the relay, in the foreground (what the unit runs)
#   scripts/feed-relay.sh install   # a user unit here that keeps it running, started now
#   scripts/feed-relay.sh remove    # that unit stopped and removed
#   scripts/feed-relay.sh status    # the unit, and each feed's age on both sides
#
# PHI_STREAM_FEEDS_HOST: a command that runs one command on the service's
# machine with stdin passed through (default PHI_STREAM_HOST, conf.md).
# PHI_STREAM_FEEDS_DIR: the checkout there (default PHI_STREAM_HOST_DIR, else the same path as here).
# PHI_STREAM_FEEDS_FROM and PHI_STREAM_FEEDS_TO: the
# feeds folders, here and there (there relative to its home), both by default
# the dev workspace's .local/share/phi-stream/dev/feeds.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
. "$here/conf.sh"
phi_stream_conf "$root"
# The service's machine as phi-stream.sh reaches it, unless set apart.
host=${PHI_STREAM_FEEDS_HOST:-${PHI_STREAM_HOST:-}}
rdir=${PHI_STREAM_FEEDS_DIR:-${PHI_STREAM_HOST_DIR:-$root}}
from=${PHI_STREAM_FEEDS_FROM:-$HOME/.local/share/phi-stream/dev/feeds}
to=${PHI_STREAM_FEEDS_TO:-.local/share/phi-stream/dev/feeds}
unit=phi-stream-feeds.service
unitfile=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$unit

need_host() {
    [ -n "$host" ] || { echo "$0: set PHI_STREAM_FEEDS_HOST (the rack: \"env RACK_NOMUX=1 \$HOME/.local/bin/rack\")" >&2; exit 2; }
}

case "${1:-}" in
    run)
        need_host
        command -v tcc > /dev/null || { echo "$0: tcc not found (pacman -S tcc)" >&2; exit 1; }
        [ -d "$from" ] || { echo "$0: no feeds here at $from" >&2; exit 1; }
        # Both halves compiled by tcc as they start: nothing to install on
        # either side but the checkout. The far side's words go through its
        # login shell (fish on the rack), so the path is quoted once.
        # shellcheck disable=SC2086 # the host command is words by design
        tcc -run "$root/tools/feed-relay.c" send "$from" |
            $host "tcc -run '$rdir/tools/feed-relay.c' recv '$to'"
        ;;
    install)
        need_host
        mkdir -p "$(dirname "$unitfile")"
        cat > "$unitfile" << EOF
[Unit]
Description=Phi Stream feeds: this machine's camera and microphone feeds kept in the service's workspace on another (scripts/feed-relay.md)
After=network-online.target

[Service]
Environment="PHI_STREAM_FEEDS_HOST=$host"
Environment="PHI_STREAM_FEEDS_DIR=$rdir"
Environment="PHI_STREAM_FEEDS_FROM=$from"
Environment="PHI_STREAM_FEEDS_TO=$to"
ExecStart="$here/feed-relay.sh" run
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now "$unit"
        echo "installed and started $unit ($unitfile)"
        ;;
    remove)
        systemctl --user disable --now "$unit" 2> /dev/null || true
        rm -f "$unitfile"
        systemctl --user daemon-reload
        echo "removed $unit"
        ;;
    status)
        systemctl --user --no-pager status "$unit" 2> /dev/null | head -n 3 || echo "$unit: not installed"
        now=$(date +%s)
        echo "here ($from):"
        for f in "$from"/*.status; do
            [ -e "$f" ] || continue
            printf '  %-28s %4d s old\n' "$(basename "$f")" $((now - $(stat -c %Y "$f")))
        done
        if [ -n "$host" ]; then
            echo "there ($to):"
            # A bash script on stdin, whatever the far login shell is.
            # shellcheck disable=SC2086
            $host "bash -s" 2> /dev/null << EOF || echo "  (not reachable)"
cd '$to' || exit 1
now=\$(date +%s)
for f in *.status; do
    [ -e "\$f" ] && printf '  %-28s %4d s old\n' "\$f" \$((now - \$(stat -c %Y "\$f")))
done
EOF
        fi
        ;;
    *)
        echo "usage: $0 run|install|remove|status (feed-relay.md)" >&2
        exit 2
        ;;
esac
