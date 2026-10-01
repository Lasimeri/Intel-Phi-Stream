# Intel-Phi-Stream: notes for an agent working here

Read `CONTRIBUTING.md` first; it is the authority. The non-obvious rules:

- What this is: one model split over the GPU, the Xeon Phi cards and
  host memory, one llama.cpp context, a live sequence that never stops
  while readings run beside it in the same decode cycles
  (`src/engine.md`). llama.cpp is linked, never changed. The cards come
  from the sibling Intel-Phi-AVX512 (`scripts/avx512.md`: `PHI_AVX512_ROOT`,
  a checkout next to this one or in `$HOME`, under either name); never
  copy anything of it here.
- Why one context: the backend's lock is per `begin` and `end`, so two
  contexts in two threads would interleave inside a multiply. Do not add
  a second context or thread on the model.
- Rust only; shell for `scripts/`. No Python or JavaScript, ever.
- Every code file gets a sibling `.md` with the same stem, written in the
  same change. No em or en dashes anywhere. A sibling repository's file is
  a GitHub link, not a bare path.
- Every hardware claim cites a source or a measurement with its command;
  logs on disk, never `/tmp` (tmpfs). This host drifts a quarter over
  tens of minutes: interleave.
- `make check` before committing, push after.
- Traps: one process at a time may hold the cards; M-RoPE refuses a
  token decoded again at a position already in the cache (take logits
  from the chunk that fed the token); `-t 8`, since llama.cpp's own CPU
  work between the backend's multiplies costs 10 ms a token at `-t 2`.
