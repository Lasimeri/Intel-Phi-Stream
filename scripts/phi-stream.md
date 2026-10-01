# phi-stream.sh

The launcher. With the co-processor repository found (`avx512.md`), it
runs `phi-stream` through that repository's `scripts/phi-ggml.sh`, which
starts a worker on every card that is up and names `libggml_phi.so` to
ggml through `GGML_BACKEND_PATH`; `PHI_GGML_OFFLOAD=1` is set unless the
caller sets it, so the cards' rows leave host memory after the upload.
Without it, the binary runs as it is: the GPU and the host.

```
scripts/phi-stream.sh tui                 # the terminal (src/tui.md)
scripts/phi-stream.sh run < lines.txt     # stdout and stdin (src/main.md)
scripts/phi-stream.sh probe               # the rates on this machine (src/probe.md)
scripts/phi-stream.sh gate                # the composition check (src/gate.md)
```

Every option of `phi-stream` passes through (`src/main.md`). The binary
is `target/release/phi-stream` (`make build`); the script stops with a
message when it is not built.
