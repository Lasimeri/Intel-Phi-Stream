# 2026-10-01: the lens readout, Gate A: the capture reproduces the model's own logits

Host: Ryzen 7 5800X, 31 GiB, RTX 3090 Ti, both cards up; kernel
7.2.6-1-cachyos. llama.cpp f5b9bd3 unchanged (`build/bin`). Model:
Qwen3.8-35B-A3B Q6_K, split as the stream runs it (`src/split.md`).
Plan of record: the Obsidian note "Intel Phi Stream - J-space
metacognition plan (2026-10-01)", steps 1 and 2.

## What is being built

A per-token readout of what the model holds in mind: the residual after
chosen blocks, at the position of the token being placed, read through
a Jacobian lens (`unembed(J_l h)`, the reference implementation's
formula). Before any lens is trusted, three gates:

- **A, capture correctness** (this record): the residual captured
  live, decoded with no transport, must be the model's own next-token
  distribution.
- **B, cost**: what installing the eval callback and asking for blocks
  costs the stream.
- **C, transfer**: whether the lens fitted on the base model
  (Qwen3.6-35B-A3B) reads our fine-tune better than the plain logit
  lens, on the reference's own evaluation sets.

## How it reads the model without changing llama.cpp

llama.cpp's public eval callback (`cb_eval`): the scheduler asks it,
node by node, which tensors it wants, computes up to each wanted node,
synchronizes the backend and hands the tensor over
(`ggml_backend_sched_compute_splits`). The capture (`src/capture.md`)
wants `l_out-L` (the residual after block `L`), `result_norm` (here the
`get_rows` of the output rows: it says which rows of the micro-batch
asked for logits) and `result_output` (the logits; its first source is
the model's unembedding on the GPU). The readout (`src/readout.md`) uses
the model's own unembedding and output norm in place: the unembedding
is 398 MiB (248320 tokens by 2048 at Q6_K) and is not copied.

## Gate A

`scripts/phi-stream.sh lens check` (`src/check.md`): 18 tokens over
the three kinds of cycle the engine makes.

| cycle | tokens | captured logits row vs llama's | readout top 10 | readout vs llama, largest difference | host norm vs graph's |
| --- | --- | --- | --- | --- | --- |
| end of a prompt read in chunks of 129 | 1 | 0 | same | 0.000000 | 0.000004 |
| decoded alone | 8 | 0 | same | 0.000000 | up to 0.000004 |
| beside another sequence's chunk of 16 | 8 | 0 | same | 0.000000 | up to 0.000004 |
| end of a 41-token injection | 1 | 0 | same | 0.000000 | 0.000002 |

PASS: the captured final residual of the token being placed, decoded by
the readout, is the model's own next-token distribution, in every kind
of cycle, including a token that is one row of a micro-batch it shares
with another sequence.

## Defects met on the way, each found by the gate

1. The output row indices are an unnamed input in this llama.cpp
   (`leaf_820` in the trace), so matching them by name found nothing:
   every decode yielded no output row. The selection is now taken from
   the node llama.cpp names `result_norm`, which here is the `get_rows`
   over those indices; any other placement stops the capture with an
   error instead of a guess.
2. The readout first differed from llama's logits by 12 to 17. Taking
   it apart (`PHI_STREAM_CAPTURE_EXTRA`, `PHI_STREAM_CAPTURE_DEBUG`): the
   unembedding alone of the graph's normed row reproduced llama's logits
   exactly, and the graph's own `l_out-39` copy was exactly
   `attn_residual-39 + ffn_out-39`, but the row the check used was
   block 38's. `llama_model_n_layer` already leaves out the extra
   prediction block (40 for this model, whose file holds 41): the
   program subtracted it a second time. Fixed in `llm.rs`.

## What this enables

Every number the lens will read rests on this: the residual of the
token being placed, at any block, exactly as the model computed it, in
the live stream's own cycles. Next: Gate B (the cost of the callback in
the stream), then the lens file (fetched pinned, converted without
Python) and Gate C.
