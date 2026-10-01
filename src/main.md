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
`lens check` (the capture and the readout reproduce the model's own
logits, `src/check.md`), `lens convert` and `lens info` (the lens file,
`src/lens.md`; `scripts/fetch-lens.sh` fetches and converts), `lens eval`
(the lens against the logit lens on the reference's sets, `src/eval.md`),
`code anchor` and `code stream` (whether the model, and the stream,
write working code: MultiPL-E's HumanEval in Rust, compiled and tested
in a sandbox, `src/code.md`; `scripts/fetch-code-eval.sh` fetches the
tasks),
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
`PHI_STREAM_WORKSPACE`, default `~/.local/share/phi-stream`: `persona.md`
as composed, `notes.md`, `stream.log`), `--personality FILE` (or
`PHI_STREAM_PERSONALITY`: the base of the persona, a person's standing
instructions; default `~/CLAUDE.md` when it exists; `src/engine.md` has
the composition), `--system FILE` (the whole persona verbatim, an
experiment's override), `--first-words` (the journal's first words in its
own voice after the seed, "Where was I. "), `--seed-text`
(the first thing from outside), `--direct-max` 48, `--chunk` 0
(adapting), `--rollover-at` 0.6, `--time-every` 60 (seconds of quiet before the
clock is put into the chain; 0 never), `--nudge-every` 60 (seconds
between nudges), `--feed FILE` at the start; `--mind` (read what is
on its mind at every token, `src/mind.md`) with `--lens`,
`--mind-layers`, `--mind-k`; `--reflect` (needs `--mind`: check the
tokens it places, `src/reflect.md`) or `--reflect-dry` (deliberate, never
change), `--reflect-keep-at P` (keep's share of the choice at which a
check keeps, 0.45, the stream's own choice; above 1 every check writes, a test of the rewind); `keep-at P` sets it
while the service runs; `--horizon SECS` (show the text that far behind its placement at
an even pace, `src/playout.md`; 0 by default, 1 with `--reflect`). The mind's and the loop's options are
one group (`MindArgs`), the same for `serve`, `run` and `code stream`, so
a measurement runs what the stream runs; the configuration is checked
before the model is loaded. `tail --mind` prints the readings too;
`tail` and `run` print every check's episode on stderr. `--dev REPO`
(`docs/dev.md`): develop that repository with Claude: the persona says
so, `[read: PATH]` resolves there, `[prefer: ...]` lines are kept.
`say --as NAME` names the speaker; `ask` says and waits for the next
spoken line; `listen` prints spoken lines, notes, preferences, what was
heard and changed words, one line each, for a monitor.

The cards' backend is named by `GGML_BACKEND_PATH` (the sibling
repository's `scripts/phi-ggml.sh` sets it and starts the workers, which
`scripts/phi-stream.sh` arranges); unset, a `libggml_phi.so` beside the
binary is used when there is one, else the GPU and the host alone.
