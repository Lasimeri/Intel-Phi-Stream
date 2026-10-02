# capture.rs

The residual stream of the token being placed, read through llama.cpp's
public eval callback (`llama_context_params.cb_eval`, `llama.h`). Nothing
of llama.cpp is changed.

**How the scheduler calls it** (`ggml_backend_sched_compute_splits`,
`ggml/src/ggml-backend.cpp`): for every split, node by node, the
callback is asked whether it wants the node (`ask = true`); the nodes
up to the first one it wants are computed as one view of the split's
graph, the split's backend is synchronized, and the callback is handed
the computed node (`ask = false`). A node is therefore read after it is
computed and before anything later in the graph runs. The scheduler
synchronizes after every view whenever a callback is installed, wanted
or not: installing it has a cost even when it asks for nothing (Gate B,
`docs/results/`).

**What it asks for** (names from `src/models/qwen35moe.cpp`):

- `l_out-L` for the blocks in `CaptureConfig::layers`: the residual
  after block `L`, `[n_embd, n_tokens]` f32, every row of the
  micro-batch.
- `result_norm`: in this llama.cpp a `get_rows` of the normed final
  residual over the output row indices (an unnamed input, `leaf_N` in the
  graph). Its `src[1]` gives which rows of the micro-batch asked for
  logits; the rows of the captured blocks are selected with it. The other
  placement llama.cpp has (the selection inside the last block, with
  masked next-token embeddings, which this program never enables) makes
  `result_norm` something else; the capture then stops with an error
  rather than guess.
- `result_output`: the logits, a matrix multiply whose `src[0]` is the
  model's unembedding (`output.weight`) and whose `src[1]` leads to the
  output norm's weight and its `rms_norm` (the epsilon from its
  parameters). The readout (`readout.md`) uses these two tensors in
  place, so the 398 MiB unembedding is never copied. The pointer is
  checked to stay the same across decodes. With `keep_logits` the logits
  rows and the graph's normed rows are copied too (the check).

**What comes out**, per output row of each micro-batch, in order:
`OutputRow { micro_batch, row, layers: [(L, residual)], logits, normed }`.
The engine asks for at most one logits row per decode, so a decode
yields at most one output row: the token whose next token is being
sampled. `all_rows` keeps every row of every micro-batch per block
instead (`BlockRows`), for reading a whole prompt.

**Safety.** `eval_callback` never unwinds into C: a panic is caught,
recorded in `error`, and the callback stops asking. Every read is
`ggml_backend_tensor_get` within the tensor, after the scheduler's
synchronization. The state is boxed in the model (`llm.rs`), so its
address, which llama.cpp keeps, is stable for the context's life.

**Diagnostics.** `PHI_STREAM_CAPTURE_TRACE=1` prints every node the
scheduler asks about (name, op, shape, sources); `PHI_STREAM_CAPTURE_DEBUG=1`
prints the row bookkeeping; `PHI_STREAM_CAPTURE_EXTRA=a,b` copies the
named nodes whole (the last micro-batch's, in `extra`).

**Experts** (`experts_on`, live: `phi-stream experts on|off`, off by
default). With it on, the capture also asks for `ffn_moe_topk-L` at the
same blocks (llama.cpp names the top-k selection so in `build_moe_ffn`,
`llama-graph.cpp`): i32, the experts each row was routed to; the linked llama.cpp builds it as a
view of an argsort, read row by row at its stride (the live service
stopped on "not contiguous" before). They are kept
until the output rows are known, as a residual is, and recorded per output
row with its batch row (`row_experts`, cleared by `take`). Each asked node
is one more synchronization of the scheduler, so it costs a little at
every cycle: measured before it stays on. The guide lane reads them to
compare the experts the guided and the live token were routed to
(`engine.md`). A failure to read them (a llama.cpp whose top-k is a
view, say) turns them off and is noted (`experts_error`); it never stops
the capture the mind reads.

Tests: `cargo test` covers the row selection and the layer names;
`phi-stream lens check` (`check.md`) is the end-to-end gate.
