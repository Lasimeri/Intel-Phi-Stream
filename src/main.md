# main.rs

`phi-stream`: one model split over the GPU, the two cards and host
memory, thinking without pause while it reads what it is given, over
llama.cpp used as a library and never changed. The model is owned by a
service; everything else is a client of it.

```
scripts/phi-stream.sh start           # the service, in tmux (scripts/phi-stream.md)
scripts/phi-stream.sh attach          # the terminal (src/tui.md); Ctrl-C leaves it running
phi-stream say "what are you working on?"
phi-stream feed notes/plan.md         # read beside the stream, then joined
phi-stream tail [--status]            # the stream on stdout as it happens
phi-stream status
phi-stream persona my-persona.md      # the context rolls over onto a new persona
phi-stream chunk 16 | temp 0.8 | pause | resume | quit
```

Subcommands that own the model (no service): `serve` (the service
itself, `src/serve.md`), `probe` (`src/probe.md`), `gate` (`src/gate.md`),
`run` (the stream on stdout, stdin lines said to it, `/feed FILE`,
`/chunk N`, `/quit`; `--max-tokens` to stop; for scripted tests). The
clients take `--socket PATH` (or `PHI_STREAM_SOCKET`; `src/client.md`).

Model options, before the subcommand: `-m` (or `PHI_STREAM_MODEL`,
default the 35B-A3B Q6_K), `--backend-dir` (or `PHI_STREAM_BACKEND_DIR`:
the llama.cpp build whose backends are loaded, default the one linked at
build time), `-c` (32768 cells, shared by every sequence), `--batch` (129:
the live token and a chunk of 128 at most), `--gpu-blocks` (default: as
many as fit, `src/split.md`), `--kv-q8`, `-t` (8: llama.cpp's own CPU work
between the backend's multiplies, the expert gating and activations of
the host blocks; 2 costs 10 ms a token), the sampling (`--temp` 1,
`--top-k` 20, `--top-p` 0.95: the model header's recommendation; `--seed`,
from the clock unless given, so each start is its own;
`--repeat-penalty` 1.05 over `--repeat-last-n` 256, against the loops a
quiet stream falls into), `-v`.

Stream options, after `serve` or `run`: `--frame journal|chat` (journal:
one continuous first-person text, `«` from outside, `»` said aloud;
chat: the model's template, `src/engine.md`), `--workspace DIR` (or
`PHI_STREAM_WORKSPACE`, default `~/.local/share/phi-stream`: `persona.md`,
`notes.md`, `stream.log`), `--system FILE` (the persona; default the
workspace's `persona.md` when it exists, else the frame's own, so the
persona is edited in place and reloaded with `persona`), `--seed-text`
(the first thing from outside), `--direct-max` 48, `--chunk` 0
(adapting), `--rollover-at` 0.6, `--feed FILE` at the start.

The cards' backend is named by `GGML_BACKEND_PATH` (the sibling
repository's `scripts/phi-ggml.sh` sets it and starts the workers, which
`scripts/phi-stream.sh` arranges); unset, a `libggml_phi.so` beside the
binary is used when there is one, else the GPU and the host alone.
