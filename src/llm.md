# llm.rs

One model split three ways and one context over llama.cpp's C API.

- `Llm::load` registers the backends the way llama.cpp's programs do
  (`ggml_backend_load_all_from_path`: CUDA and the CPU variants of the
  build, then `GGML_BACKEND_PATH`, the cards' `libggml_phi.so`), asks the
  CUDA device its free memory, plans the split (`split.rs`: the budget is
  the free memory less the cells' K and V, every sequence slot's
  recurrent state, the cycle's compute buffers and a margin), loads the
  model with the device list of every CUDA device (`CUDA0`, `CUDA1`, ...;
  `PHI_STREAM_GPUS=N` keeps the first N; the budget is their free memory
  summed, the margin counted once per GPU, or `--gpu-headroom GIB` each)
  and up to two tensor overrides (the host blocks' experts; the host-only
  tables, `split.md`), the blocks shared out by `tensor_split` from
  `split::layer_split` when there is more than one GPU, repacking off (a
  repacked weight is never offered to the cards), and makes the context:
  a unified KV cache (`kv_unified`, so a range of one sequence's cells is
  given to another by metadata alone), `n_seq` sequences, flash attention
  on, K and V float16 or 8-bit (`--kv-q8`), on the GPUs or in host memory
  (`--kv-host`: llama.cpp's `offload_kqv` off, nothing of the cache
  reserved on a GPU, a longer context), a batch over host weights copied
  to a GPU or computed where the weights are (`--no-op-offload`: the
  cards' backend then takes the prompts), no recurrent snapshots
  (`n_rs_seq 0`), `batch` tokens a cycle. The device list, the overrides
  and the split are kept in the struct: the model keeps the pointers.
- `make_sampler` follows the llama.cpp it is built against: since
  a6aa6f545 (2026-08-04) the penalties sampler takes the vocabulary size,
  DRY takes no context length, and a negative window turns a sampler
  off, so `-1` is resolved to the trained context here; `build.rs` sets
  `llama_samplers_v2` from the header, and both forms are kept.
- `decode` runs one `llama_decode` over lanes: each lane is one sequence's
  tokens at their positions, the last one's logits kept on request. A
  cycle of the engine is one live token (its sequence) and a chunk of a
  sequence being read or caught up; llama.cpp's hybrid memory splits such
  a batch into equal-share micro-batches by itself. `logits`, `sample`
  (the chain: top-k, top-p, temperature and the seeded draw, or greedy at
  temperature 0; the model's header recommends 20, 0.95 and 1) and
  `greedy` read a row of the last decode; `sample_logits` puts logits the
  engine made (the guide's mix) through the same chain, accepted once, as
  `llama_sampler_sample` does with a row; `n_seq` is the context's count
  of sequences. With `ban_dashes` (the
  default) the chain starts with a logit bias of minus infinity on every
  vocabulary token whose text carries an em or en dash, found by one
  scan of the vocabulary at load, so the no-dash rule holds in the
  sampler and not only in the persona. `ban_tokens` adds a frame's control
  tokens to the same bias (the journal frame bans the template's).
- `seq_cp` gives one sequence another's cells in a range and the source's
  recurrent state (llama.cpp shares the state's cell and copies it on the
  next write); `seq_rm` drops a range (a whole sequence always goes, a
  partial range of a recurrent sequence does not); `clear`.
- `tokenize` (control tokens parsed on request), `special` (the one token
  a control string is), `piece` and `text` (control tokens shown),
  `is_eog`, `eot`.

What llama.cpp's own CPU backend computes with `threads` is small (the
token embedding lookups): the experts in host memory are multiplied by
`libggml_phi.so` with its own pool (`PHI_GGML_HOST_THREADS`, 12) and the
cards. llama.cpp is used as a library and never changed.

Sampling also takes `min_p` (tokens under that share of the likeliest
one's probability dropped; 0 off) and DRY, llama.cpp's sampler against
repeated sequences (`dry_multiplier`, 0 off; `dry_base`,
`dry_allowed_length`, `dry_last_n`; breakers newline, colon, quote,
asterisk), in llama.cpp's order: penalties, DRY, top-k, top-p, min-p,
temperature.

`--cpu` loads the model with no GPU (no device listed, no layer
offloaded, a split of zero blocks): on the host, and on the cards when
their backend is loaded. A second model then has the GPU to itself.

## A remote model (`Llm::remote`, `--remote`)

With `--remote URL` nothing is loaded here: `Llm` holds a
[`Remote`](remote.md) instead of a model, a context and a sampler, and
every method goes to it. `decode` keeps the lanes' tokens in order (only
sequence 0) and returns row 0 for the last; `sample` asks the server for
the next token; `tokenize` is the server's; `piece`, `is_eog`, `eot`,
`tokens_containing` and the dash scan come from the vocabulary file;
`seq_rm` and `seq_pos_max` act on the one sequence, `seq_cp` and `accept`
do nothing, `clear` empties it; `set_sampling` sends the settings and the
banned tokens with the next request. `logits`, `greedy` and
`sample_logits` return an error: no logits cross the network. `n_ctx` is
the server's slot, `n_seq` 1, `batch_cap` the whole context (a sequence
goes to the server as one prompt). `forks()` says whether sequences beside
the live one exist (false here); `remote_info()` gives the server, the
model's name and the streams and tokens so far. `sample` and
`sample_logits` return a `Result` in both modes (a network can fail).
