# feed-relay.sh

The feeds written on this machine kept in the workspace of the service on
another ([`tools/feed-relay.md`](../tools/feed-relay.md)): on the desktop,
for the harness on the GPU rack. The far machine comes from this machine's
`phi-stream.local.conf` ([`conf.md`](conf.md)), as for `phi-stream.sh`.

```sh
scripts/feed-relay.sh install   # a user unit (phi-stream-feeds) that keeps it running, started now
scripts/feed-relay.sh status    # the unit, and each feed's age here and there
scripts/feed-relay.sh run       # the relay in the foreground (what the unit runs)
scripts/feed-relay.sh remove    # the unit stopped and removed
```

- `PHI_STREAM_FEEDS_HOST` (default `PHI_STREAM_HOST`, [`conf.md`](conf.md)): a command that runs one command on the
  service's machine with stdin passed through (`ssh HOST`). The rack's
  wrapper with `RACK_NOMUX=1` gives the relay a connection of its own, so
  restarting the shared one does not cut it.
- `PHI_STREAM_FEEDS_DIR`: the checkout there (default `PHI_STREAM_HOST_DIR`, else the same path as
  here); its `tools/feed-relay.c` is the receiving half.
- `PHI_STREAM_FEEDS_FROM`, `PHI_STREAM_FEEDS_TO`: the folders, here and
  there (there relative to its home); by default both are the dev
  workspace's `.local/share/phi-stream/dev/feeds`.
- `install` writes `~/.config/systemd/user/phi-stream-feeds.service` with
  the four settings as they are at that moment, `Restart=always` every
  5 s (the far end restarting, the link dropping), and enables it.
- Both halves are compiled by tcc as they start, so both machines need
  `tcc` and nothing else.
