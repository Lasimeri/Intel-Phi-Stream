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
make build                        # needs ~/llama.cpp with its CUDA build (build.md)
scripts/phi-stream.sh start       # the service, in tmux; with the cards when Intel-Phi-AVX512 is found
scripts/phi-stream.sh attach      # the terminal; Ctrl-C leaves the stream running
scripts/phi-stream.sh say "hello" # from any shell, script or agent: say, feed, tail, status, persona
```

The model is owned by a service that loads it once and listens on a Unix
socket; the terminal and the one-shot commands are its clients, so a
person, a script and an agent can talk to the same running mind, and
the persona, the notes and the policy change without a reload
([`src/serve.md`](src/serve.md), [`src/client.md`](src/client.md)). In
the terminal, type and press Enter: a short line is heard in the next
cycle and appears in the stream where it was; `/feed FILE` hands a file
over, read beside the thoughts (the strip shows `reading N/M`, the two
rates and how full the context is), then joined; `/persona FILE` gives
it a new persona, which it rolls its context over onto after a summary;
`/chunk N`, `/temp T`, `/pause`, `/resume`, PgUp and PgDn
([`src/tui.md`](src/tui.md)).

The persona's base is your own `~/CLAUDE.md` when it exists (or
`--personality FILE`): its manner becomes the mind's manner, with a
preamble that reads its talk of tools and memory files as another
harness's, and the frame's mechanics after it; and the sampler never
draws a token carrying an em or en dash. The text is a journal by
default: one continuous first-person text with
no turns, the mind's own threads kept and returned to, what comes from
outside as `«` lines, what it says aloud as `»` lines; and two things it
does by itself inside the text, `[note: ...]` (kept in the workspace and
shown to it again at every rollover) and `[read: PATH]` (brings a file
in). `--frame chat` keeps the model's own template instead
([`src/engine.md`](src/engine.md)).

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

## Reading what is on its mind (in progress)

The next stage gives the stream a view of its own workspace: per token,
the residual stream at chosen blocks read through a Jacobian lens (the
technique of Anthropic's "Verbalizable Representations Form a Global
Workspace in Language Models"), then fed back so the mind can reason
about the concepts it is holding at the token it is placing. Built in
gated steps, each usable and recorded:

- **Done, Gate A**: the capture and the readout. `scripts/phi-stream.sh
  lens check` proves that the residual captured live, decoded with the
  model's own norm and unembedding, is exactly the model's next-token
  distribution in every kind of cycle
  ([`docs/results/2026-10-01-lens-capture.md`](docs/results/2026-10-01-lens-capture.md)).
- **Done, Gates B and C**: the readout costs about 1 ms a token; the
  lens fitted on the base model (fetched pinned and converted without
  Python, `scripts/fetch-lens.sh`) reads this fine-tune better than the
  plain logit lens on all six of the reference's evaluation sets
  (`scripts/phi-stream.sh lens eval`), and places the workspace band at
  blocks 27 to 32
  ([`docs/results/2026-10-01-lens-cost-transfer.md`](docs/results/2026-10-01-lens-cost-transfer.md)).
- **Usable now**: `scripts/phi-stream.sh start --mind` reads what is on
  its mind at every token it places (blocks 27, 29, 31; about 2.6 ms a
  token in the stream); the terminal shows it in a strip and token by
  token with `/mind`; `phi-stream tail --mind` prints it
  ([`src/mind.md`](src/mind.md)).
- **On the real-time clock**: every piece of the stream, every reading
  and every status is stamped to the microsecond; the chain carries the
  time of what it hears and of its silences; the engine's intervals are
  real time, not cycles
  ([`docs/results/2026-10-01-clock.md`](docs/results/2026-10-01-clock.md)).
- **Writes working code**: by MultiPL-E's HumanEval in Rust (156 tasks,
  compiled and tested in a sandbox), the model through the split passes
  74.4 percent greedy by the benchmark's protocol
  (`scripts/phi-stream.sh code anchor`), and the stream itself is
  measured the same way (`code stream`)
  ([`docs/results/2026-10-01-code.md`](docs/results/2026-10-01-code.md)).
- Next: the reflection loop (the stream reasoning on what is on its mind
  at the token it is placing, beside the live token, within a commit
  horizon so shown text never changes), real time as the reference
  frame for every token and control, and coding tasks with tests as the
  measure.

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
| `src/serve.rs`, `src/client.rs` | the service that owns the model and its socket; the wire and the client side |
| `src/tui.rs` | the terminal, a client of the service |
| `src/probe.rs`, `src/gate.rs` | the rates on this machine; the composition against a straight sequence |
| `src/capture.rs`, `src/readout.rs`, `src/check.rs` | the residual of the token being placed, read through llama.cpp's eval callback; the lens readout on the GPU; the gate that both reproduce the model's own logits |
| `src/clock.rs` | the wall clock the stream is kept against, to the microsecond |
| `src/torch.rs`, `src/lens.rs`, `src/eval.rs`, `src/mind.rs` | the reference's lens file read without Python; the `.jlens` format; the lens against the logit lens on the reference's sets; what is on its mind at every token |
| `scripts/fetch-lens.sh` | the lens and the evaluation sets, pinned, checksummed, converted |
| `src/code.rs`, `scripts/fetch-code-eval.sh`, `tools/parquet-jsonl` | whether it writes working code: MultiPL-E's HumanEval in Rust, fetched pinned, compiled and tested in a sandbox |
| `scripts/phi-stream.sh` | the launcher: the service in tmux, the terminal, the clients; with the cards when the co-processor repository is found |
| `docs/results/` | measurements, with their commands |

## The repositories

| repository | what |
| --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack |
| [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) | the cards as an AVX-512 co-processor; `libggml_phi.so` and `scripts/phi-ggml.sh`, which this repository consumes |
| Intel-Phi-Stream (this one) | the thought stream |
| [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev), [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | a local Jev over the same cards, and its asking side |

MIT ([`LICENSE-MIT`](LICENSE-MIT)).
