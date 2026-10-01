# phi-stream

A model that thinks without pause, and reads what you give it without
stopping to. One model, Qwen3.8-35B-A3B at Q6_K, split three ways: the
GPU (an RTX 3090 Ti), two Intel Xeon Phi cards, and host memory. One
context over it, several sequences in one decode cycle: a live sequence
that never stops generating, and beside it, chunk by chunk, the reading
of whatever it is handed, joined into the stream afterwards. A terminal
to talk to it, drawn as something alive. llama.cpp is used as a library
and never changed.

```
make build                                   # needs ~/llama.cpp with its CUDA build (build.md)
scripts/phi-stream.sh tui                    # the terminal; with the cards when Intel-Phi-AVX512 is found
scripts/phi-stream.sh probe                  # the rates on this machine
```

In the terminal, type and press Enter: a short line is heard in the next
cycle and appears in the stream where it was; `/feed FILE` hands a file
over, read beside the thoughts (the strip shows `reading N/M`, the two
rates and how full the context is), then joined. `/chunk N`, `/temp T`,
`/pause`, `/resume`, PgUp and PgDn, Ctrl-C to leave
([`src/tui.md`](src/tui.md)).

## What it does

- **The split** ([`src/split.md`](src/split.md)): read from the file's
  own tensor table, everything that is not an expert goes to the GPU with
  the experts of as many blocks as its free memory allows once the
  context is set aside; the experts of the remaining blocks go to host
  memory by one tensor override, where
  [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512)'s
  `libggml_phi.so` takes the cards' share of them as of any host weight.
  On the machine of record: 28 of 41 blocks on the GPU (19.2 GiB), 8.0
  GiB of experts in host memory, 2.94 GB of it on the cards.
- **The stream** ([`src/engine.md`](src/engine.md)): the live sequence
  decodes its pending token every cycle. Up to 48 tokens, what is said is
  decoded straight in: heard in one cycle. Longer, a free sequence is
  given the live one's cells and recurrent state (`llama_memory_seq_cp`
  under a unified KV cache) and takes a chunk each cycle beside the live
  token; when the reading ends a third sequence is composed from the
  prefix, the read cells and state, and a chase of the thoughts produced
  meanwhile, and becomes the live one. The model's view: thoughts, what
  it was given, the thoughts it had while reading. A full context is
  rolled over through a summary the stream writes (this model's cells
  cannot be shifted: M-RoPE).
- **Why one context**: the backend's lock is taken and dropped inside
  `phi_ggml_begin` and `phi_ggml_end` separately, so two contexts in two
  threads would interleave inside a multiply. One thread, one context,
  several sequences in one `llama_decode` gives the overlap without
  touching the backend; llama.cpp's hybrid memory splits such a batch by
  itself.

## Measured (2026-10-01, [`docs/results/2026-10-01-phi-stream.md`](docs/results/2026-10-01-phi-stream.md))

| | the stream | the reading |
| --- | --- | --- |
| alone | 46 tok/s | 295 tok/s |
| reading beside the stream, chunks of 8 | 7.9 | 63 |
| chunks of 16 | 7.5 | 120 |
| chunks of 32 | 5.1 | 163 |
| chunks of 64 | 3.5 | 220 |

The chunk adapts to what remains, or `/chunk N`. The host blocks'
multiplies of the two sequences run one cycle at a time on the cards and
the host pool, which is the trade; the backend's fixed cost a multiply
at small batches (8 tokens cost 96 ms where 129 cost 450) is the lever,
and it is the backend's. The composition is gated against a straight
sequence: the same next token, a largest logit difference of 0.70 against
0.65 for the straight sequence fed in the same chunks, so the join is
exact to the kernels' rounding ([`src/gate.md`](src/gate.md)).

## What you need

- A llama.cpp checkout with its headers and a CUDA build with shared
  libraries (`LLAMA_CPP_DIR`, default `~/llama.cpp`; the libraries from
  `build/bin`, or `PHI_STREAM_LLAMA_BUILD_DIR`); libclang for bindgen
  ([`build.md`](build.md)). llama.cpp f5b9bd3 is what this was built
  and measured against.
- A GPU with CUDA and about 22 GiB free for the Q6_K split, or any size
  with `--gpu-blocks N` and a smaller model.
- For the cards: Intel-Phi-AVX512 cloned next to this repository or in
  `$HOME` (or `PHI_AVX512_ROOT`), built, with its card workers deployable
  and the cards' stack
  [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) up
  ([`scripts/avx512.md`](scripts/avx512.md)). Without them the stream
  runs on the GPU and the host.
- `make check` before committing ([`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Layout

| path | what |
| --- | --- |
| `src/main.rs` | the command line: `probe`, `gate`, `run`, `tui` |
| `src/split.rs` | the placement, from the file's tensor table and the GPU's free memory |
| `src/llm.rs` | the model and the context over llama.cpp's C API; lanes of a cycle; the sampler |
| `src/engine.rs` | the stream: hearing, reading beside, the join, the rollover, the nudges |
| `src/tui.rs` | the terminal |
| `src/probe.rs`, `src/gate.rs` | the rates on this machine; the composition against a straight sequence |
| `scripts/phi-stream.sh` | the launcher, with the cards when the co-processor repository is found |
| `docs/results/` | measurements, with their commands |

## The repositories

| repository | what |
| --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack |
| [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) | the cards as an AVX-512 co-processor; `libggml_phi.so` and `scripts/phi-ggml.sh`, which this repository consumes |
| Intel-Phi-Stream (this one) | the thought stream |
| [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev), [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | a local Jev over the same cards, and its asking side |

MIT ([`LICENSE-MIT`](LICENSE-MIT)).
