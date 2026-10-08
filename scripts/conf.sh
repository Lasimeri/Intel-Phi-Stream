# conf.sh: this machine's settings for the scripts, sourced by them (not run).
# Read in order, the first to set a key winning, as the family's xks does:
# the environment, $PHI_STREAM_CONFIG, phi-stream.local.conf in the checkout
# (not tracked: this machine's own), ~/.config/phi-stream/phi-stream.conf,
# phi-stream.conf in the checkout (tracked: the defaults). See conf.md.
# shellcheck shell=bash

# One file's KEY=VALUE lines exported, each only when the key is not set
# yet (by the environment or an earlier file); a value as a shell reads it
# (quotes, $HOME), since the files are shell-sourceable by design.
phi_stream_conf_file() {
    local f=$1 line k v
    [ -f "$f" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line#"${line%%[![:space:]]*}"}
        case "$line" in '' | \#*) continue ;; esac
        k=${line%%=*}
        [[ "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
        [ -n "${!k+x}" ] && continue
        v=$(eval "printf '%s' ${line#*=}") || continue
        export "$k=$v"
    done < "$f"
}

phi_stream_conf() {
    local root=$1
    [ -n "${PHI_STREAM_CONFIG:-}" ] && phi_stream_conf_file "$PHI_STREAM_CONFIG"
    phi_stream_conf_file "$root/phi-stream.local.conf"
    phi_stream_conf_file "${XDG_CONFIG_HOME:-$HOME/.config}/phi-stream/phi-stream.conf"
    phi_stream_conf_file "$root/phi-stream.conf"
}

# A word quoted for the far login shell, whichever it is: single quotes,
# an inner quote as '\'' (read the same by bash, sh and fish).
phi_stream_q() {
    local s=$1
    printf "'%s'" "${s//\'/\'\\\'\'}"
}
