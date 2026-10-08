# phi-stream.sh

The launcher: the service and its clients.

```
scripts/phi-stream.sh start [serve options]   # the service in tmux session phi-stream (PHI_STREAM_SESSION)
scripts/phi-stream.sh dev [serve options]     # the same, developing this repository with Claude: start --dev <repo> --mind --reflect --workspace ~/.local/share/phi-stream/dev (docs/dev.md)
scripts/phi-stream.sh attach [--follow]       # the terminal (src/tui.md); Ctrl-C leaves it running; --follow reloads it onto each build
scripts/phi-stream.sh window [--keep]         # the terminal in a desktop window; start runs --keep
scripts/phi-stream.sh say "hello"             # and feed, tail, status, persona, chunk, temp, pause, resume, quit
scripts/phi-stream.sh stop                    # quit the service, end the session, close the window
scripts/phi-stream.sh probe | gate | run | lens ...  # the subcommands that own the model, without a service
scripts/phi-stream.sh serve ...               # the service in the foreground
scripts/phi-stream.sh doctor                  # what stands between this machine and a working service, and the fix for each
```

`start`, `serve`, `probe`, `gate`, `run` and `lens` go through the cards when the
co-processor repository is found (`avx512.md`): that repository's
`scripts/phi-ggml.sh` starts a worker on every card that is up and names
`libggml_phi.so` to ggml through `GGML_BACKEND_PATH`. Since 2026-10-06 (the
four-card rack, 251 GB of host memory) the host keeps its copy of the
cards' rows (`PHI_GGML_OFFLOAD=0` unless the caller sets it) and the
cards' work is chosen by `PHI_GGML_PP_ONLY`: with no placement file the
cards take the prompts alone and the GPUs and the host generate
(`PHI_GGML_PP_ONLY=1`; measured on Flash-Next Q6_K_XL, twice
interleaved: prompts 88.0 against 85.6 tok/s, generation 7.9 against
7.6); with `PHI_GGML_EXPERTS=<placement file>` (the experts a routing
calibration found most used, `tools/expert-placement.c` there) the
cards hold those experts whole and work at the generation steps beside
the host (`PHI_GGML_PP_ONLY=0`) while the prompts go to the GPUs through
llama.cpp's op offload (the stream's default; `--no-op-offload` sends
them to the host and the cards instead). `PHI_GGML_OFFLOAD=1` is for a
host short of memory: the cards' rows leave host memory after the
upload. Without the repository, the binary runs on the GPUs and the host.
The tmux session runs this script's own `serve`, so the cards are found
the same way; `tmux attach -t phi-stream` shows the service's log. Every
other verb is a client and passes through to the binary (`src/main.md`),
which finds the service by its socket (`src/client.md`). The model
options (`--gpu-blocks`, `-c`, `-t`, ...) are global: they may come
before or after the subcommand, `start` included; the script finds the
subcommand past them. The binary is
`target/release/phi-stream` (`make build`), or `PHI_STREAM_BIN` (another
build: a measurement pinned to a frozen binary while the tree is
rebuilt); the script stops with a message when it is not built.
`start` carries `PHI_STREAM_BIN` into its tmux session (the session takes
the tmux server's environment, not the calling shell's; the improvement
loop's measurement runs a candidate's binary so, `improve-measure.md`), and
names its session exactly (`-t =phi-stream`: a prefix matched
`phi-stream-mcp`, and `start` refused, seeing the MCP terminal's session
as the service).

`start` (and `dev`) also keeps the service's output on disk, appended to
`~/.local/share/phi-stream/serve.log` (`PHI_STREAM_LOG`; never tmpfs):
a failure's message outlives the tmux session.

## The window

While the model is loaded and running, a terminal window on the desktop
shows it: the diagnostics (the placement, the rates, the context, the
mind strip, the checks) and the input line (`src/tui.md`).
- `dev` adds `--terminal` (the stream's sandboxed terminal, `src/term.md`)
and `--rollover-tokens 150000` (it is told its memory is nearly full only
past 150 thousand cells) and `--second-chain`: measured in the agent frame
(2026-10-02, two interleaved pairs of 5 minutes) it cut the repeated
8-grams from 19.8 to 11.3 percent for 13 percent of the stream's rate.
`phi-stream chain off` (or `/chain off` in the terminal) turns it off live.
`stop` waits for the service to write its summary and end
(`src/engine.md`) before it ends the session, as long as the service
answers: the service bounds its own wait (`PHI_STREAM_QUIT_WAIT` seconds,
default 120, in which its summary gains no token), and the script ends one
that stops answering for 30 s or is still there after
`PHI_STREAM_QUIT_HARD` seconds (default 3600), saying each minute what it
waits on (a fixed two minutes, then 480 s, cut off a summary on the rack on
2026-10-08 while its remote server re-read 99k tokens for twelve minutes); it waits for the
process listening on the socket (`quit_and_wait`), not only the tmux
session, so a service started outside tmux (by a watchdog, at a login) is
stopped too, and killed past the two minutes (a restart had started the
next service beside one still writing its summary, and the next ran out
of GPU memory loading, 2026-10-03). `PHI_STREAM_PRELOAD` goes into the
service's `LD_PRELOAD` alone (a watchdog's debugger shim). `restart [dev OPTIONS]`
does the same and starts it again (by default `dev`), its window kept open:
the terminal in it reconnects, so the interface stays on the desktop
through an update (`stop` and `start` closed it for the whole load).
`accept PATH...` takes files the dev stream wrote out of its working copy
once they are in the repository (kept in its workspace, `accepted-DATE/`),
so it sees the repository's versions (`src/term.md`).

`start` (and so `dev`) runs `window --keep`, detached: for as long as
  the service's tmux session lasts, every 3 s, when no window is open and
  the model is running (`status` prints a line; while it loads it prints
  none), it opens one. A window closed by hand opens again while the
  model runs. One keeper and one window at a time: their pids are kept
  in `$XDG_RUNTIME_DIR/phi-stream-window-keeper.pid` and
  `phi-stream-window.pid`.
- The window runs `attach --follow`: it reconnects across restarts of
  the service (showing `NO SERVICE` meanwhile) and reloads onto each new
  build of the binary. In `--follow`, a terminal that dies (a crash of a
  build) is reported, the terminal reset, and the next build run.
- A restart by `quit` and `start` keeps the window, which reconnects;
  `stop` closes it (the keeper first, so it does not reopen it).
- The desktop: the session's own `WAYLAND_DISPLAY` or `DISPLAY`, or the
  user's `wayland-0` when started over SSH. The terminal emulator:
  `PHI_STREAM_TERMINAL`, or the first of konsole (in its own process,
  `--separate`, so its pid is the window), alacritty, kitty, foot,
  xterm. `PHI_STREAM_WINDOW=0` opens none (a host with no desktop).
- Over SSH, the same terminal without a window:
  `ssh -t HOST '"$HOME/Intel Phi Stream/scripts/phi-stream.sh" attach --follow'`.

## Instances

`PHI_STREAM_INSTANCE=NAME` runs a service beside the first: its own tmux
session (`phi-stream-NAME`), socket (`phi-stream-NAME.sock`, exported as
`PHI_STREAM_SOCKET`, so the clients in the same shell reach it), workspace
(`dev-NAME`) and window; `PHI_STREAM_CARD=N` gives it one card
(`PHI_GGML_CARDS`), `PHI_STREAM_CARDS=0` none. Measured 2026-10-01: a
second instance on card 1 next to one on card 0 did not come up, and the
first's backend then reported "could not clear card 0: no answer within
30s"; the backend frees the cards when it opens (one backend process at a
time, Intel-Phi-AVX512), so two instances each with a card are not
supported yet. A second instance with `PHI_STREAM_CARDS=0` and `--cpu`, or
on the GPU, does not touch the cards.

## A remote model

With `--remote URL` among the options (or `PHI_STREAM_REMOTE` set),
`launch` runs the binary directly: the model is served elsewhere
([`src/remote.md`](../src/remote.md)), so `phi-ggml.sh`, the cards and the
GPU are left alone. The option and `--remote-vocab`/`--remote-slot` take a
value, so the subcommand is found after them. The development service on
the GPU rack's Flash Next:

    scripts/phi-stream.sh dev --remote http://192.168.0.39:8001 --frame agent --improve --temp 0.5 --top-p 0.95 --min-p 0.05

(`dev` adds `--mind --reflect --second-chain`; the binary turns them off
for a remote model and says so.) The vocabulary comes first, once:
`PHI_STREAM_REMOTE_SSH="rack sh" scripts/remote-vocab.sh http://192.168.0.39:8001`.

## On the GPU rack

`start` passes every `PHI_STREAM_*` and `PHI_GGML_*` variable of the
calling shell into the tmux session (which otherwise takes the tmux
server's environment). The development service runs on the rack beside
llama.phi's server, its model, with the build sandbox and the terminal
on cores the server does not use (measured placement:
`docs/results/` of 2026-10-07; the server's threads on three cores of
each of the 16 L3 groups, the card daemons on 51, 55, 59, 63):

    cd "/mnt/raid5/phi/Intel Phi Stream"
    PHI_STREAM_WINDOW=0 PHI_STREAM_QUIT_WAIT=480 PHI_STREAM_BUILD_CPUS=3,7,11,15,19,23,27,31 PHI_STREAM_BUILD_JOBS=8 PHI_STREAM_TERM_CPU=35 \
        scripts/phi-stream.sh dev --remote http://127.0.0.1:8001 --frame agent --improve --temp 0.5 --top-p 0.95 --min-p 0.05

The vocabulary there is the model's own first file (its metadata):
`~/.local/share/phi-stream/remote/STEM.vocab.gguf` a link to it. Its
terminal on the desktop: a Konsole running `rack -t` with
`scripts/phi-stream.sh attach --follow` in the rack's checkout.

## Settings, another machine, the doctor

The settings come from the environment, then `phi-stream.local.conf` in
the checkout (this machine's own, not tracked), then the tracked defaults
in [`phi-stream.conf`](../phi-stream.conf) ([`conf.md`](conf.md)).
`dev` with no options after it adds `PHI_STREAM_DEV_ARGS` (the GPU rack:
`--remote http://127.0.0.1:8001 --frame agent --improve` and its
sampling), so `restart` alone brings the harness back as it was.

With `PHI_STREAM_HOST` set (a command that runs one command on the
service's machine, `PHI_STREAM_HOST_TTY` the same with a terminal,
`PHI_STREAM_HOST_DIR` the checkout there) and no service answering on
this machine's socket, every verb runs there in that checkout: on the
desktop, `status`, `say`, `ask`, `tail`, `listen`, `objective`, `chain`,
`restart`, `stop`, `attach` (with a terminal) all reach the harness on
the rack. `window` opens the window here, running `attach --follow`,
which goes there. The management interface for an agent does the same
through [`phi-stream-mcp.sh`](phi-stream-mcp.md), and the feeds written
here reach the workspace there through [`feed-relay.sh`](feed-relay.md).

`doctor` prints one line per check, `ok` or what is wrong and the fix,
and exits 0 when nothing is. On the service's machine: the binary built
and newer than the sources, a service on the socket and its status, the
service running the current build (else `restart`), and for a remote
model its server's `/health`, enough slots for the stream's pinned slot,
the vocabulary file for the served model, and a feed fresher than 15 s in
the workspace. On a machine with `PHI_STREAM_HOST`: the feed relay's unit
and Claude Code's `phi-stream` MCP registration (the remote launcher),
then the same verb there.
