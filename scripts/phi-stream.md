# phi-stream.sh

The launcher: the service and its clients.

```
scripts/phi-stream.sh start [serve options]   # the service in tmux session phi-stream (PHI_STREAM_SESSION)
scripts/phi-stream.sh attach                  # the terminal (src/tui.md); Ctrl-C leaves it running
scripts/phi-stream.sh say "hello"             # and feed, tail, status, persona, chunk, temp, pause, resume, quit
scripts/phi-stream.sh stop                    # quit the service, end the session
scripts/phi-stream.sh probe | gate | run ...  # the subcommands that own the model, without a service
scripts/phi-stream.sh serve ...               # the service in the foreground
```

`start`, `serve`, `probe`, `gate` and `run` go through the cards when the
co-processor repository is found (`avx512.md`): that repository's
`scripts/phi-ggml.sh` starts a worker on every card that is up and names
`libggml_phi.so` to ggml through `GGML_BACKEND_PATH`; `PHI_GGML_OFFLOAD=1`
is set unless the caller sets it, so the cards' rows leave host memory
after the upload. Without it, the binary runs on the GPU and the host.
The tmux session runs this script's own `serve`, so the cards are found
the same way; `tmux attach -t phi-stream` shows the service's log. Every
other verb is a client and passes through to the binary (`src/main.md`),
which finds the service by its socket (`src/client.md`). The binary is
`target/release/phi-stream` (`make build`); the script stops with a
message when it is not built.
