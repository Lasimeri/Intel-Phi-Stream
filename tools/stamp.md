# stamp.c

`tcc -run tools/stamp.c [SKIP] < PIPE > COPY`: how evenly text arrives
on a pipe, the way a person watching it sees it. Every read of stdin is
copied to stdout and stamped with `CLOCK_MONOTONIC`; at end of input it
prints on stderr the number of reads and the gaps between them: median,
90th and 99th percentile, the largest, and how many reached 100, 250 and
500 ms. Reads in the first `SKIP` seconds after the first one (the
opening, which arrives at once) are left out.

Used on `phi-stream run`'s stdout to measure the playout (`src/playout.md`):
the stream's text arrives in one write per piece, so the gaps are the
gaps a reader sees. A pipe can merge writes that come close together, so
the count of reads is a lower bound on the pieces; the gaps that matter
(stalls) are not merged away.

C, built by tcc at run time (no Python, nothing to install).
