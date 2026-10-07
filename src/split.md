# split.rs

Where the model's bytes go, decided from the file and the GPU's free
memory, never from a table.

- `sizes` reads the tensor table with gguf (`no_alloc`: metadata only, a
  fraction of a second): bytes per block (`blk.N.*`), the expert tensors
  of each (`*_exps.*`), everything outside the blocks (embeddings, output,
  norms), and from the header the attention K and V bytes per token at
  float16 (`head_count_kv` x (`key_length` + `value_length`) x 2 x the
  attention layers: every `full_attention_interval`-th block of a hybrid
  model, all of a plain one), and the recurrent state one sequence slot
  holds (for each recurrent block of the main pass, llama.cpp's f32
  convolution state, `(d_conv - 1) x (d_inner + 2 n_group d_state)`, and
  state matrix, `d_state^2 x dt_rank`: 62.8 MiB for this model's 30
  recurrent blocks). For Qwen3.8-35B-A3B at Q6_K: 41 blocks of
  661 MiB, 630 MiB of experts each, 0.78 GiB outside, 20 KiB of K and V
  per token (10 of 40 blocks attend; the 41st is the extra prediction
  block).
- `plan` puts on the GPU everything that is not an expert (the attention
  and recurrent layers, norms, output: about 2 GiB), then the experts of
  blocks `0..k` for the largest `k` whose bytes fit the budget (the GPU's
  free memory less the context's needs, `llm.rs`), the last block always
  on the host (its extra prediction layer is unused). `--gpu-blocks N`
  takes `k = N` as given. The host blocks' experts are named to llama.cpp
  by one tensor override, the regular expression
  `blk\.(27|28|...)\.ffn_(up|gate|down)_exps\.weight` to the CPU buffer
  type, which is where `libggml_phi.so` finds the weights it takes the
  cards' share of (its device shares the CPU's host buffers).

Everything the GPU holds for the blocks it does not keep experts of is
the small part (31 MiB a block): the split moves the experts only, which
are 95 percent of the file.

Added 2026-10-06 for Qwen3.8 Flash Next (`qwen4exp`, 48 blocks, 512
experts a block, 157 GiB at UD-Q6_K_XL) on the four-GPU rack:

- A model in several files (`split.count` in the header, a u16 from
  llama.cpp's gguf-split) keeps its header in the first and its tensors
  spread over all of them: `sizes` reads every file's table
  (`shard_paths`, through `llama_split_prefix` and `llama_split_path`;
  `tally` adds one file's tensors). The first part must be the one
  named; the Q6_K_XL's part 1 holds no tensor at all.
- `host_only`: tensors kept in host memory whatever the plan, today the
  per-layer token embeddings (`per_layer_token_embd`, 50.7 GiB in that
  model, a lookup table read a row per token). `llm.rs` names them to
  llama.cpp by a second override, so they stay memory-mapped on disk and
  are never counted against a GPU.
- `cache_kind` says per block whether it keeps K and V (1), a recurrent
  state (2) or nothing (0, an extra prediction block); `block_cache`
  turns that into each block's cache bytes for the context's cells and
  sequences, since llama.cpp allocates a block's cache on the GPU that
  holds the block.
- `layer_split` chooses how many blocks each GPU takes for llama.cpp's
  layer split (`tensor_split` given as counts): what each block leaves
  on a GPU (all of it when its experts stay, else all but its experts)
  plus its cache is laid end to end and cut in proportion to each GPU's
  free memory; a block goes to the GPU its middle falls in; the output
  layer, which llama.cpp places after the last block, is counted on the
  last GPU. One GPU takes every block. Without the cache term the last
  GPU, which held the most expert blocks and all the attention blocks'
  K and V, ran out of memory at the first cycle (296 MiB short, the rack,
  2026-10-06).
