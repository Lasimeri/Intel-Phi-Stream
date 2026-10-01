# readout.rs

The lens readout on the GPU. A residual vector `h` captured at block `l`
(`capture.md`) is transported into the final block's basis, `J_l h`
(none for the final block itself), then decoded the way the model
decodes its own last residual: `rms_norm` with the graph's epsilon,
times the output norm weight, then the unembedding. This is the
reference implementation's `lens_l(h) = unembed(J_l @ h)` with
`unembed = lm_head(final_norm(.))`
([`jlens/lens.py`, `jlens/hf.py` of anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens)),
with the GGUF's norm weight, which the converter already stores in the
form llama.cpp multiplies by.

- The norm weight and the unembedding are the model's own tensors on the
  GPU, taken from the graph by the capture; nothing is copied (the
  unembedding is 398 MiB: 248320 tokens by 2048 at Q6_K).
- `Readout::new` starts a CUDA backend instance of this program's own on
  CUDA0 (the model's tensors live on the same device) and an allocator
  that reuses one compute buffer across calls.
- `load_transports` puts `J_l` matrices (`d x d` float16, row-major, row
  `i` the coefficients of output coordinate `i`) in GPU memory: 8 MiB
  each for this model.
- `logits(groups)` builds a small graph per call: for each group of
  columns the transport (or none), concatenated, then the norm, the
  weight and the unembedding; returns `n_vocab` floats per column. A
  group may be `normed` (already the normed final residual: unembedding
  only), a diagnostic of the unembedding alone.
- `rank` turns logits into the top `k` tokens with log-probabilities;
  `rank_of` is a token's rank (0 best, ties to the lower id); `compare`
  says whether two logit vectors agree in their top `k` and by how much
  they differ at most.

The GPU memory the readout's own buffers need is kept free by the split
planner (`READOUT_RESERVE` in `main.rs`, `extra_reserve` in `llm.rs`).
Tests: `cargo test` covers `rank`, `rank_of` and `compare`;
`phi-stream lens check` is the end-to-end gate.
