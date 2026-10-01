#!/usr/bin/env bash
# fetch-lens.sh: the pre-fitted Jacobian lens for Qwen3.6-35B-A3B (the base
# of Qwen3.8-35B-A3B), fetched at a pinned revision and checked against its
# pinned sha256, then converted to this program's format. Idempotent: a
# file that is already there and checks out is kept. See fetch-lens.md.
#
#   scripts/fetch-lens.sh [DIR]     # default ~/models/jlens/qwen3.6-35B-A3B
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
dir=${1:-${PHI_STREAM_LENS_DIR:-$HOME/models/jlens/qwen3.6-35B-A3B}}
repo=stanleytheli/qwen3.6-35B-A3B-jlens
rev=7a5dc7a6c770c272226a321409b30d7e6d773bba
file=lens.pt
sha=2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1
size=327166510
mkdir -p "$dir"
dst="$dir/$file"
ok() { [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ] && echo "$sha  $dst" | sha256sum -c --status; }
if ok; then
    echo "$dst: present, sha256 checks"
else
    echo "fetching $repo $file at $rev ($size bytes)"
    curl -fL --retry 3 -o "$dst.part" "https://huggingface.co/$repo/resolve/$rev/$file"
    mv "$dst.part" "$dst"
    if ! ok; then
        echo "$0: $dst does not match the pinned size and sha256; removed" >&2
        rm -f "$dst"
        exit 1
    fi
    echo "$dst: fetched, sha256 checks"
fi
# The model card, for its licence and provenance.
curl -fsSL -o "$dir/README.md" "https://huggingface.co/$repo/resolve/$rev/README.md" || true
# The reference implementation's evaluation sets (Apache 2.0, Anthropic),
# pinned to a commit and checked, for `phi-stream lens eval`.
eval_dir=${PHI_STREAM_EVAL_DIR:-$(dirname "$dir")/eval}
mkdir -p "$eval_dir"
ref=581d398613e5602a5af361e1c34d3a92ea82ba8e
while read -r esha name; do
    f="$eval_dir/lens-eval-$name.json"
    if ! echo "$esha  $f" | sha256sum -c --status 2>/dev/null; then
        curl -fsSL -o "$f.part" "https://raw.githubusercontent.com/anthropics/jacobian-lens/$ref/data/evaluations/lens-eval-$name.json"
        mv "$f.part" "$f"
        echo "$esha  $f" | sha256sum -c --status || { echo "$0: $f does not match its pinned sha256; removed" >&2; rm -f "$f"; exit 1; }
    fi
done <<SETS
d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c association
50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416 multihop
fa70b9bd89416a6d8d985a80dc628b109ae6fd3b25b9275c0fc5065d7ff4a0ef multilingual
b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560 order-ops
6aeb3415c5a5c3f3827c9efe63f006de02f5ef39a816bbac68e15e733aba60cc poetry
9d05e16b7234a57d0773d120a4e1c4e94fd3bc2235a8125d4200a70e60ab17aa typo
SETS
curl -fsSL -o "$eval_dir/LICENSE" "https://raw.githubusercontent.com/anthropics/jacobian-lens/$ref/LICENSE" || true
echo "$eval_dir: six evaluation sets, sha256 checked"
bin="${PHI_STREAM_BIN:-$root/target/release/phi-stream}"
[ -x "$bin" ] || { echo "$0: $bin not built; run make build, then: $bin lens convert $dst $dir/lens.jlens" >&2; exit 1; }
"$bin" lens convert "$dst" "$dir/lens.jlens"
