# remote-vocab.sh

Copies the vocabulary of the model a llama-server serves to this machine,
for `phi-stream --remote` ([`src/remote.md`](../src/remote.md)): it asks
the server's `/props` for its `model_path`, runs `head -c` of that file on
the server's machine and writes
`~/.local/share/phi-stream/remote/STEM.vocab.gguf`. The head holds the
GGUF metadata (the tokens, their types, the chat template); the weights
are never read. Qwen3.8 Flash Next's first file is its metadata alone
(10,946,624 bytes, measured 2026-10-07), so the default 64 MiB copies it
whole.

- `PHI_STREAM_REMOTE_SSH`: a command that runs a bash script from stdin on
  the server's machine; default `ssh HOST bash -s`. The GPU rack takes a
  password, so there: `PHI_STREAM_REMOTE_SSH="rack sh"` (the desktop's
  `~/.local/bin/rack`).
- `PHI_STREAM_REMOTE_HEAD`: bytes copied (default 67108864).

Run it once per model the server loads; the stream checks the file
against the server at every start and names this script when they
differ.
