# improve-measure.sh

The self-improvement loop's measurement (`src/improve.md`, stage 5):
`scripts/improve-measure.sh N [MINUTES]` runs candidate N on the live
model, alternating with its base (base, candidate, base, candidate; this
host drifts a quarter over tens of minutes), and applies the rule written
in `src/improve.md` before any candidate was measured.

It refuses to start unless the candidate passed its sandbox (`outcome`),
has its binary, and carries Claude's `reviewed` mark (written after
reading its `change.patch`: nothing unreviewed runs outside the sandbox),
and unless the repository's source (`src`, `build.rs`, `Cargo.*`) is the
candidate's base with no change (a head past the base that changed only
scripts or docs builds the same binary, so it is measured as the base);
it builds the base's binary from the base itself.

Each window restarts the service (`phi-stream.sh restart dev`, the same
options for both arms, `PHI_STREAM_MEASURE_OPTS`; the binary through
`PHI_STREAM_BIN`, which `start` carries into its tmux session), sets one
objective (an audit of `src/`, ongoing work with its tools, so neither arm
rests), lets it settle a minute, then measures MINUTES (default 10):
- rate: the mean of the status samples (every 10 s) in which it decoded;
- repeats: the share of its thinking's 8-grams that repeat one seen
  earlier in the window (`tools/loopiness.c --kind think` over
  `chain.log`);
- yes: the goal probe's mean P(yes) in the window (`goal.log`);
- unparsed: unparsed checks, percent of the window's checks;
- alive: 0 when the service died or never ran.

The verdict and the four windows go to the candidate's `measure.txt`, a
copy into the stream's workspace (`improve/cand-N/`), and a line of its
`improve.log`; the service is restarted on the repository's own binary
with the loop on (`PHI_STREAM_AFTER_OPTS`). The rule's arithmetic was run
on example lines (a candidate worse in one pair only is kept; one slower
in both is rejected on rate; one that died is rejected).
