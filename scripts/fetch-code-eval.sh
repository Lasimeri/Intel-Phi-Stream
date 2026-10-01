#!/usr/bin/env bash
# fetch-code-eval.sh: MultiPL-E's HumanEval in Rust (156 tasks), fetched at
# a pinned dataset revision and checked against its pinned sha256, then
# converted from parquet to JSON lines by tools/parquet-jsonl (no Python).
# Idempotent. See fetch-code-eval.md.
#
#   scripts/fetch-code-eval.sh [DIR]    # default ~/models/code-eval (PHI_STREAM_CODE_DIR)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
dir=${1:-${PHI_STREAM_CODE_DIR:-$HOME/models/code-eval}}
rev=28441b6024e71d4a1c1c0f6bf171c935cd5a43f2
sha=92dba15e4da4deb4dbae5e15ce1712c95ec5a63c3ead69886271470309d9d314
size=75300
mkdir -p "$dir"
pq="$dir/humaneval-rs.parquet"
ok() { [ -f "$pq" ] && [ "$(stat -c %s "$pq")" = "$size" ] && echo "$sha  $pq" | sha256sum -c --status; }
if ! ok; then
    curl -fL --retry 3 -o "$pq.part" "https://huggingface.co/datasets/nuprl/MultiPL-E/resolve/$rev/humaneval-rs/test-00000-of-00001.parquet"
    mv "$pq.part" "$pq"
    ok || { echo "$0: $pq does not match its pinned size and sha256; removed" >&2; rm -f "$pq"; exit 1; }
fi
echo "$pq: sha256 checks"
curl -fsSL -o "$dir/MultiPL-E-dataset-card.md" "https://huggingface.co/datasets/nuprl/MultiPL-E/resolve/$rev/README.md" || true
# The converter: built from the local cargo cache when it can be, else fetched.
tool="$root/tools/parquet-jsonl/target/release/parquet-jsonl"
if [ ! -x "$tool" ]; then
    cargo build --release --offline --manifest-path "$root/tools/parquet-jsonl/Cargo.toml" ||
        cargo build --release --manifest-path "$root/tools/parquet-jsonl/Cargo.toml"
fi
"$tool" "$pq" > "$dir/humaneval-rs.jsonl.part"
n=$(wc -l < "$dir/humaneval-rs.jsonl.part")
[ "$n" = 156 ] || { echo "$0: $n tasks, expected 156" >&2; exit 1; }
mv "$dir/humaneval-rs.jsonl.part" "$dir/humaneval-rs.jsonl"
echo "$dir/humaneval-rs.jsonl: 156 tasks"
