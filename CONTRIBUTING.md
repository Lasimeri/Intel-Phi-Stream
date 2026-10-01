# Contributing

These rules exist so that someone with the same GPU and cards and a fresh
Arch Linux install can reproduce every result here without asking anyone.
The sections are the same in every repository of the family (see
[The family](#the-family)); what differs is said where it applies.

## Languages

- **Rust** for everything here: the program is one crate over llama.cpp's
  C API, bound by bindgen at build time ([`build.md`](build.md)).
- **Shell** (`bash`) for the launcher and the checks under `scripts/`.
- **Never Python or JavaScript** for anything here.
- llama.cpp is a dependency, read and linked, never changed: no patch, no
  fork, no copied source. What the stream needs of it that llama.cpp
  does not offer is done above the API (several sequences in one decode,
  the composition by `seq_cp`), never inside.

## Documentation

- Every code file (`.rs`, `.sh`) has a sibling `.md` with the same stem in
  the same directory: purpose, the facts the code depends on (with the
  source named), invariants, how to test it. Obvious code is not
  re-narrated. A change to behaviour changes its `.md` in the same commit.
- Every public Rust item has a doc comment. Comments explain intent and
  contract, not syntax.
- No em or en dash characters anywhere, commit messages included. Use
  commas, colons, parentheses, or `--`.
- Relative links between Markdown files must resolve. A file that lives in
  a sibling repository is linked on GitHub, never named as if it were here.
- `scripts/check-docs.sh` enforces the sibling, dash and link rules
  (`make docs-check`, the first step of `make check`).

## Measurements

- Every hardware claim names its source: a file and function in a named
  source tree (llama.cpp's, the backend's), or a measurement made on this
  machine with the command shown.
- Results are recorded under `docs/results/` with the date, the host, the
  llama.cpp commit and the exact command. Timings closer than this host's
  drift (a quarter over tens of minutes) need interleaved runs. Logs go on
  disk, never under `/tmp` (a tmpfs on this host).
- `cargo test` needs no model. What needs one is run by hand: `phi-stream
  probe` (the rates), `phi-stream gate` (the composition against a
  straight sequence), and the terminal under tmux (the record's scripts).
- One process at a time holds the cards: the backend frees every card's
  uploads when it opens.

## The family

| repository | what | finds its dependency by |
| --- | --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack: daemon, kernel, boot, storage, the `phi` CLI | (none) |
| [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) | the cards as an AVX-512 co-processor: phi512, the card worker, the `libggml_phi.so` backend | `PHI_STACK_ROOT`, `phi` on PATH, a checkout next to it, `$HOME` |
| Intel-Phi-Stream (this one) | the thought stream over the GPU, the cards and host memory | `PHI_AVX512_ROOT`, a checkout next to this one, `$HOME` ([`scripts/avx512.md`](scripts/avx512.md)) |
| [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev) | `xks`, a local Jev (System One) whose subject runs on the host and the cards | `PHI_AVX512_ROOT`, a checkout next to it, `$HOME` |
| [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | `mjev`, the asking side of Jev | `MJEV_XKS`, `xks` on PATH, a checkout next to it, `$HOME` |

- A dependency is found in that order, as a checkout under its GitHub
  clone's name (`Intel-Phi-AVX512`) or the spaced one (`Intel Phi
  AVX-512`). Nothing of a sibling is copied into another.
- What this repository consumes from Intel-Phi-AVX512 (`scripts/phi-ggml.sh`,
  `host/target/release/libggml_phi.so`, the `PHI_GGML_*` variables) is
  kept working there across changes: add, do not rename.
- Without the co-processor repository the stream runs on the GPU and the
  host; the cards are an addition, not a requirement.

## Git

- One subject line that says what changed (a leading `Area:` is fine), then
  the why. `make check` before every commit, push after.
- MIT license ([`LICENSE-MIT`](LICENSE-MIT)).
