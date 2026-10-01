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
```

`start`, `serve`, `probe`, `gate`, `run` and `lens` go through the cards when the
co-processor repository is found (`avx512.md`): that repository's
`scripts/phi-ggml.sh` starts a worker on every card that is up and names
`libggml_phi.so` to ggml through `GGML_BACKEND_PATH`; `PHI_GGML_OFFLOAD=1`
is set unless the caller sets it, so the cards' rows leave host memory
after the upload. Without it, the binary runs on the GPU and the host.
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

`start` (and `dev`) also keeps the service's output on disk, appended to
`~/.local/share/phi-stream/serve.log` (`PHI_STREAM_LOG`; never tmpfs):
a failure's message outlives the tmux session.

## The window

While the model is loaded and running, a terminal window on the desktop
shows it: the diagnostics (the placement, the rates, the context, the
mind strip, the checks) and the input line (`src/tui.md`).
- `start` (and so `dev`) runs `window --keep`, detached: for as long as
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
