# tools

Helpers with their own builds, kept out of the stream's binary.

| path | what |
| --- | --- |
| `parquet-jsonl` | one parquet file to JSON lines (MultiPL-E's coding evaluation is distributed only as parquet); `scripts/fetch-code-eval.sh` builds and runs it |
| `feed-relay.c` | the feeds written on one machine (cameras, microphones, the Glass) kept in the workspace of the service on another: `send` here, `recv` there, joined by ssh; the harness on the GPU rack with its feeds written on the desktop. `scripts/feed-relay.sh install` |
| `stamp.c` | how evenly text arrives on a pipe: gaps between reads (median, percentiles, largest); measures the playout on `phi-stream run`. `tcc -run tools/stamp.c` |
| `reflect-analyze.c` | the reflection loop's episodes summarized: changed and kept, the share below the write threshold, a histogram of the deliberation's first-word certainty; written by the dev stream itself. `tcc -run tools/reflect-analyze.c < reflect.log` |
| `loopiness.c` | the share of 8-token sequences that repeat an earlier one in a time window of `chain.log`, with `--kind think\|speak\|all` to filter which tokens feed the sequence analysis (written by the dev stream). `tcc -o loopiness tools/loopiness.c && ./loopiness START_US END_US --kind all chain.log` |
| `guide-analyze.c` | what the guide lane measured in a time window of `guide.log`: KL mean, median and quartiles, the share of changed likeliest tokens and the commonest changes, the experts shared (written by the dev stream). `tcc -o /tmp/guide-analyze tools/guide-analyze.c -lm && /tmp/guide-analyze START_US END_US` |
