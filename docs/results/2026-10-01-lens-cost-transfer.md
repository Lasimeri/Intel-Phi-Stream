# 2026-10-01: the lens readout, Gates B and C: what it costs the stream, and whether the lens reads this model

Same host, model, split and llama.cpp as
[`2026-10-01-lens-capture.md`](2026-10-01-lens-capture.md) (Gate A),
everything through the cards (`scripts/phi-stream.sh`, the launcher
fixed the same afternoon). Measurements pinned to frozen binaries
(`PHI_STREAM_BIN`) while the tree was being rebuilt; logs under
`~/.cache/phi-asm-test/stream/gate-b/` and `gate-c/`.

## Gate B: the cost in the stream

`phi-stream --gpu-blocks 27 probe --gen 64 --chunks 16` with nothing,
with the eval callback installed but asking for nothing
(`--capture-idle`), and with the per-token readout of 3 and 6 blocks
after every decode that asked for a token (`--mind-layers`, synthetic
transports of the real size: the cost, not a lens; ranking on the GPU,
`src/readout.md`). The split fixed at 27 GPU blocks so only the callback
and the readout differ. Three rounds, interleaved
(`gate-b/run.sh`, `gate-b/gate-b.log`).

Generation alone, ms per token (the readout's time included):

| condition | round 1 | round 2 | round 3 | mean |
| --- | --- | --- | --- | --- |
| no callback | 24.7 | 33.1 | 30.4 | 29.4 |
| callback, asking nothing | 25.5 | 26.1 | 34.6 | 28.7 |
| readout of blocks 20, 26, 32 | 33.5 | 27.3 | 34.9 | 31.9 |
| readout of blocks 16, 20, 24, 28, 32, 36 | 32.8 | 28.5 | 25.8 | 29.0 |

The readout itself, timed on its own in the same runs: 0.89 to 0.96 ms
a token for 3 blocks, 1.11 to 1.35 ms for 6 (capture copy, transport,
norm, unembedding, softmax and top-64 on the GPU, 64 indices and
probabilities back). Reading beside the stream with chunks of 16, ms a
cycle: 147 to 161 with nothing, 145 to 221 idle, 145 to 177 with 3
blocks, 147 to 166 with 6.

Conclusions:

- The readout costs about 1 ms a token, 3 to 4 percent of a cycle, and
  grows by about 0.1 ms a block.
- The callback's own cost and the capture's (the scheduler synchronizes
  after every split once a callback is installed, and splits at each
  wanted node) do not show above this host's drift: the same condition
  ranges over 9 ms between rounds. Anything they cost is below that.
- A sweep of one block at a time (blocks 8, 16, 20, 24, 26, 28, 32, 36,
  39) first suggested that reading a block whose experts run on the host
  and the cards cost 10 to 15 ms (blocks 28 and 32); blocks 36 and 39,
  also host blocks, cost nothing measurable, and the interleaved rounds
  show no such cost: drift. The planner can still keep chosen blocks'
  experts on the GPU first (`--read-blocks-on-gpu`, `src/split.md`); it
  is not on by default, since nothing here asks for it.
- The readout's GPU memory (its backend's buffers and the transports,
  8 MiB a block) is set aside in the split: with 3 blocks read, the
  planner gives the GPU the experts of 27 blocks instead of 28.

Before the ranking moved to the GPU (host log-sum-exp and selection over
248320 logits), the readout cost about 2.1 ms a block; that version's
partial rounds are kept in `gate-b/prelim/`.

## Gate C: does the lens fitted on the base model read this model

`scripts/phi-stream.sh lens eval` (`src/eval.md`): the lens of
Qwen3.6-35B-A3B (fetched pinned and checksummed, converted without
Python; `scripts/fetch-lens.md`, `src/lens.md`) on Qwen3.8-35B-A3B Q6_K,
over the reference implementation's six evaluation sets (pinned to its
commit `581d3986`, checksummed), scored as the paper scores lens
quality: at each item's readout position, every fitted block (0 to 38)
read through the Jacobian lens and through the plain logit lens; an
intermediate recovered at `k` when one of its single-token forms is in
the top `k` at any block; the area under pass@k against log k for k up
to 1000, normalized so that always-first scores 1
(`gate-c/eval.out`, 2 min 40 s for all six on the cards).

| set | items | scored | Jacobian pass@10 | logit pass@10 | Jacobian AUC | logit AUC | model's own output, AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| association | 102 | 102 | 0.098 | 0.029 | **0.291** | 0.121 | 0.009 |
| multihop | 93 | 94 | 0.543 | 0.436 | **0.706** | 0.526 | 0.482 |
| multilingual | 107 | 426 | 0.484 | 0.383 | **0.592** | 0.466 | 0.208 |
| order-ops | 55 | 109 | 0.798 | 0.486 | **0.787** | 0.599 | 0.517 |
| poetry | 98 | 98 | 0.020 | 0.010 | **0.112** | 0.081 | 0.029 |
| typo | 96 | 96 | 0.677 | 0.708 | **0.760** | 0.742 | 0.262 |

**PASS: the lens fitted on the base model reads this fine-tune, better
than the logit lens on every set** (the paper's ordering for Claude: the
Jacobian lens above the logit lens on all six). The margin here is
largest on association (2.4 times) and order-ops, small on typo and
poetry; the paper found it modest on multihop and association and
substantial on the other four, so the per-set margins differ from the
paper's while the ordering holds. Poetry is weak for both lenses at the
newline: the planned rhyme is rarely in the top 10 of any block of this
model at that position. On typo the logit lens recovers the correct
spelling at blocks 1 to 7 (the input fragments' own embeddings) and the
Jacobian lens at blocks 20 to 33, the band.

Skipped intermediates have no single-token form in this vocabulary:
Qwen's tokenizer splits multi-digit numbers into digits, so "26", "52",
"50" and the like are never one token (9 in multihop, 1 in order-ops),
and "Estonian", "Bengali" are several tokens. Order-ops rests on this
program's synonym table (`src/eval.md`), which the sets' README
describes but does not publish.

**The band.** Per block, the share of intermediates recovered at 10 by
the Jacobian lens, averaged over the five sets that recover anything
(poetry excluded):

| block | 20 | 22 | 24 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 35 | 36 | 38 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| share | 0.10 | 0.13 | 0.17 | 0.19 | **0.27** | **0.25** | **0.25** | **0.24** | **0.26** | **0.25** | 0.18 | 0.20 | 0.16 | 0.15 |

Nothing before block 17 except typo's early surface matches; the curve
rises from 19, holds at 0.24 to 0.27 over blocks 27 to 32, and falls
after 33: the workspace band of this model, two thirds of the way up its
40 blocks (the paper: coherent content only after an initial band,
abstract concepts giving way to next-token content at the end). The
mind's default blocks are therefore **27, 29 and 31**.

## The mind live in the stream

`scripts/phi-stream.sh run --mind --max-tokens 160` (journal frame, the
persona's base the user's `CLAUDE.md`, the seed "a friend asks what you
think happens to a promise when the person you made it to dies";
`mind-live/run.err`): 162 readings, one per token placed, 1.0 to 1.5 ms
each. The readings run ahead of the text: at "it was always about the"
blocks 27 and 29 hold *commitment* before the word is written; at "the
person you" they hold *commitment* and 承诺 ("promise" in Chinese) before
"committed your own agency"; at "as a" *memorial, legacy*, and at block
31 *recipient*.

The stream itself, mind off against on, greedy, the same seed, 200
thoughts, interleaved (`mind-live/ab.sh`): mean cycle 23.9 and 27.6 ms
off, 29.5 and 27.3 ms on: about 2.6 ms a token (the readout's 1 ms, the
capture, one expert block fewer on the GPU), inside this host's drift.

## What is built and how to use it

```
scripts/fetch-lens.sh                     # the lens and the evaluation sets, pinned, checked, converted
scripts/phi-stream.sh lens check          # Gate A
scripts/phi-stream.sh lens eval           # Gate C
scripts/phi-stream.sh start --mind        # the service, reading its mind at every token
scripts/phi-stream.sh attach              # the terminal: the mind strip; /mind for the readings token by token
phi-stream tail --mind                    # the readings as lines
```

Next (the plan of record): the reflection loop, the stream reasoning on
what is on its mind at the token it is placing, beside the live token
and within a commit horizon so shown text never changes; real time as
the reference frame for every token, reading and control; and coding
tasks with tests as the measure of whether any of it helps.
