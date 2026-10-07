#!/usr/bin/env bash
# remote-vocab.sh: the vocabulary of the model a llama-server serves, for
# `phi-stream --remote` (src/remote.md): the head of its GGUF file (the
# metadata, where the tokens are; never the weights) copied from the
# server's machine to ~/.local/share/phi-stream/remote/STEM.vocab.gguf.
#
#   scripts/remote-vocab.sh http://192.168.0.39:8001
#   PHI_STREAM_REMOTE_SSH="rack sh" scripts/remote-vocab.sh http://192.168.0.39:8001
#
# PHI_STREAM_REMOTE_SSH is a command that runs a bash script from stdin on
# the server's machine (default: ssh HOST bash -s); the GPU rack's `rack sh`
# brings its password. PHI_STREAM_REMOTE_HEAD: bytes copied (default 64 MiB;
# Qwen3.8 Flash Next's metadata is 10.4 MiB).
set -euo pipefail
url=${1:?usage: $0 http://HOST:PORT}
host=${url#http://}
host=${host%%[:/]*}
path=$(curl -sf --max-time 10 "$url/props" | sed -n 's/.*"model_path":"\([^"]*\)".*/\1/p')
[ -n "$path" ] || { echo "$0: $url/props names no model_path (is a llama-server there?)" >&2; exit 1; }
stem=$(basename "$path" .gguf)
dir="$HOME/.local/share/phi-stream/remote"
out="$dir/$stem.vocab.gguf"
mkdir -p "$dir"
head=${PHI_STREAM_REMOTE_HEAD:-67108864}
read -r -a shell <<< "${PHI_STREAM_REMOTE_SSH:-ssh $host bash -s}"
printf 'head -c %d %q\n' "$head" "$path" | "${shell[@]}" > "$out.new"
[ -s "$out.new" ] || { rm -f "$out.new"; echo "$0: nothing came from $host:$path" >&2; exit 1; }
mv "$out.new" "$out"
echo "$out: $(stat -c %s "$out") bytes, the head of $host:$path"
echo "checked against the server at the next start: phi-stream --remote $url ..."
