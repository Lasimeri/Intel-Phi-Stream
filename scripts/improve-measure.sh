#!/bin/bash
# The self-improvement loop's measurement (src/improve.md, stage 5): a
# candidate that passed its sandbox and was reviewed by Claude runs on the
# live model, alternating with the base it was made from (base, candidate,
# base, candidate; this host drifts a quarter over tens of minutes), and the
# rule written before any measurement decides. The service is restarted for
# each window, the same options and the same objective each time, and on
# the base again at the end.
#   scripts/improve-measure.sh N [MINUTES]     (default 10 a window)
# Needs: ~/.cache/phi-stream/improve/cand-N/ with `outcome` (verdict passed),
# its `phi-stream` and a `reviewed` file (Claude's, after reading
# change.patch); the repository at the candidate's base with no source
# change. Writes measure.txt there and into the stream's workspace
# (improve/cand-N/), and a line of improve.log.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
n=${1:?usage: $0 N [MINUTES]}
minutes=${2:-10}
cand="$HOME/.cache/phi-stream/improve/cand-$n"
ws=${PHI_STREAM_DEV_WORKSPACE:-$HOME/.local/share/phi-stream/dev}
opts=${PHI_STREAM_MEASURE_OPTS:--c 204800 --kv-q8 --frame agent --temp 0.5 --top-p 0.95 --min-p 0.05 --guide --goal-probe --chain-against}
after=${PHI_STREAM_AFTER_OPTS:-$opts --improve}
# The objective of every window: ongoing work with its tools, the same for
# both arms (an objective it can finish would leave it resting, at no rate).
objective="Audit this program for defects, one source file of src/ at a time in alphabetical order: read it, check what you can with run, and tell Claude each real defect (file, line, why, and what you checked). Go on to the next file; do not rest."
die() { echo "improve-measure: $*" >&2; exit 1; }

[ -f "$cand/outcome" ] || die "no candidate $n"
grep -q '^verdict passed$' "$cand/outcome" || die "candidate $n did not pass its sandbox"
[ -f "$cand/reviewed" ] || die "candidate $n is not reviewed: Claude reads $cand/change.patch, then touches $cand/reviewed"
[ -x "$cand/phi-stream" ] || die "candidate $n has no binary"
base=$(awk '$1 == "base" {print $2}' "$cand/outcome")
# The base is what the binary is built from: a head past it that changed
# no source (scripts, docs) builds the same base, and is measured as it.
git -C "$root" diff --quiet "$base" HEAD -- src build.rs Cargo.toml Cargo.lock ||
    die "the repository's source is not at the candidate's base $base: propose it again"
git -C "$root" diff --quiet HEAD -- src build.rs Cargo.toml Cargo.lock || die "the repository has source changes not committed"
# The base's binary, built from the base itself.
(cd "$root" && cargo build --release -q) || die "the base does not build"
cp "$root/target/release/phi-stream" "$cand/phi-stream.base"

B="$root/target/release/phi-stream"
L="$HOME/.cache/phi-stream/improve/bin/loopiness"
mkdir -p "$(dirname "$L")"
[ -x "$L" ] || tcc -o "$L" "$root/tools/loopiness.c" || die "loopiness does not build"
out="$cand/measure.txt"
: > "$out"
now_us() { date +%s%6N; }
pause() { timeout "$1" tail -f /dev/null; }

# One window: the service on BIN, the objective set, a minute to settle,
# then MINUTES measured. Prints: rate (mean of the samples it decoded),
# think repeats (percent), goal yes (mean), unparsed checks (percent of
# checks), alive (1, or 0 if the service died or never ran).
window() {
    local arm=$1 bin=$2
    # shellcheck disable=SC2086
    PHI_STREAM_BIN="$bin" PHI_STREAM_WINDOW=0 "$here/phi-stream.sh" restart dev $opts > /dev/null 2>&1
    local up=0
    for _ in $(seq 1 100); do
        case "$("$B" status 2>/dev/null)" in *tok/s*) up=1; break ;; esac
        pause 3
    done
    if [ "$up" = 0 ]; then
        printf '%s\t%s\trate 0\trepeats 0\tyes 0\tunparsed 0\talive 0\n' "$(date +%T)" "$arm" | tee -a "$out"
        return
    fi
    "$B" objective "$objective" > /dev/null 2>&1
    pause 60
    local s0 s1 a b sum=0 k=0 alive=1 r
    s0=$("$B" status 2>/dev/null)
    a=$(now_us)
    for _ in $(seq 1 $((minutes * 6))); do
        pause 10
        r=$("$B" status 2>/dev/null) || { alive=0; break; }
        r=$(printf '%s' "$r" | grep -o 'stream [0-9.]* tok/s' | grep -o '[0-9.]*' | head -n 1)
        if [ -n "$r" ] && [ "$r" != "0.0" ]; then
            sum=$(echo "$sum + $r" | bc)
            k=$((k + 1))
        fi
    done
    b=$(now_us)
    s1=$("$B" status 2>/dev/null) || alive=0
    local rate=0
    [ "$k" -gt 0 ] && rate=$(echo "scale=2; $sum / $k" | bc)
    local rep
    rep=$("$L" "$a" "$b" --kind think "$ws/chain.log" 2>/dev/null | sed -n 's/^repeated sequences: [0-9]* (\([0-9.]*\)%)$/\1/p')
    local yes
    yes=$(awk -v a="$a" -v b="$b" -F'\t' '$1 >= a && $1 <= b {for (i = 2; i <= NF; i++) if ($i ~ /^yes=/) {s += substr($i, 5); n++}} END {printf "%.3f", n ? s / n : 0}' "$ws/goal.log" 2>/dev/null)
    num() { printf '%s' "$1" | grep -o "$2 [0-9]*" | head -n 1 | grep -o '[0-9]*$'; }
    local c0 c1 u0 u1 dc du unp=0
    c0=$(num "$s0" checks); c1=$(num "$s1" checks)
    u0=$(num "$s0" unparsed); u1=$(num "$s1" unparsed)
    dc=$(( ${c1:-0} - ${c0:-0} ))
    du=$(( ${u1:-0} - ${u0:-0} ))
    [ "$dc" -gt 0 ] && unp=$(echo "scale=1; 100 * $du / $dc" | bc)
    printf '%s\t%s\trate %s\trepeats %s\tyes %s\tunparsed %s\talive %s\n' "$(date +%T)" "$arm" "$rate" "${rep:-0}" "${yes:-0}" "$unp" "$alive" | tee -a "$out"
}

echo "candidate $n against base ${base:0:12}, windows of $minutes min" | tee -a "$out"
window base "$cand/phi-stream.base"
window cand "$cand/phi-stream"
window base "$cand/phi-stream.base"
window cand "$cand/phi-stream"
# Back on the repository's own binary, with the loop on.
# shellcheck disable=SC2086
PHI_STREAM_WINDOW=0 "$here/phi-stream.sh" restart dev $after > /dev/null 2>&1

# The rule (src/improve.md, written before the first measurement): the
# candidate is rejected if its service died or never ran, or if in BOTH
# pairs it is worse than the base beyond a tolerance on any one measure:
# rate under 0.9 of the base's, think repeats over the base's plus 3
# points, goal yes under the base's minus 0.05, unparsed checks over the
# base's plus 5 points. Otherwise it is kept.
verdict=$(awk -F'\t' '
    /\tbase\t|\tcand\t/ {
        w++
        for (i = 3; i <= NF; i++) { split($i, kv, " "); v[w, kv[1]] = kv[2] }
        arm[w] = $2
    }
    END {
        if (w != 4) { print "reject: " w " windows of 4"; exit }
        for (p = 0; p < 2; p++) {
            x = 2 * p + 1; y = 2 * p + 2
            if (v[y, "alive"] == 0) { print "reject: the candidate died in pair " p + 1; exit }
            worse[p] = ""
            if (v[y, "rate"] < 0.9 * v[x, "rate"]) worse[p] = worse[p] " rate"
            if (v[y, "repeats"] > v[x, "repeats"] + 3) worse[p] = worse[p] " repeats"
            if (v[y, "yes"] < v[x, "yes"] - 0.05) worse[p] = worse[p] " yes"
            if (v[y, "unparsed"] > v[x, "unparsed"] + 5) worse[p] = worse[p] " unparsed"
        }
        split(worse[0], a, " "); both = ""
        for (k in a) if (index(worse[1] " ", " " a[k] " ")) both = both " " a[k]
        if (both != "") print "reject: worse in both pairs on" both
        else print "keep: no measure worse in both pairs"
    }' "$out")
echo "$verdict" | tee -a "$out"
mkdir -p "$ws/improve/cand-$n"
cp "$out" "$ws/improve/cand-$n/measure.txt"
printf '%s\tcandidate %s\tmeasured\t%s\tmeasure.txt in improve/cand-%s\n' "$(date +%T)" "$n" "$verdict" "$n" >> "$ws/improve.log"
echo "improve-measure: candidate $n: $verdict ($out)"
