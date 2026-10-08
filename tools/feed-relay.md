# feed-relay.c

`feed-relay send DIR | ssh HOST feed-relay recv DIR`: the feeds written
on one machine kept in the workspace of a service on another. Built for
the GPU rack (2026-10-08): the harness runs there, while the cameras'
face trackers, the microphones' watchers and the Glass write their
`feeds/NAME.status` and `feeds/NAME.log` on the desktop
([`src/feeds.md`](../src/feeds.md)), so the rack's copy of the folder had
stood still since the move and the harness saw every camera offline.

- **`send DIR`** looks once a second at every `NAME` with a
  `NAME.status` in `DIR`. The status file is sent whole when its time or
  size changes (`S` frame). The log goes as whole lines appended since
  the last look (`A`); a line still being written waits for the next
  look. A log seen for the first time starts from its last 64 KiB, cut at
  a line's start (the harness reads the last five seconds); one that
  shrank or was replaced (a rotation to `NAME.log.1`) starts again from
  its beginning after a reset (`T`). Files with no status beside them
  (`events.log`, written by the service itself) are left alone.
- **`recv DIR`** applies the frames: a status through a temporary file and
  a rename (a reader never sees half of one), a log by appending, a reset
  by truncating. It refuses names with a slash, a space or a leading dot,
  anything but `.status` and `.log`, and `events.log`.
- **Frames:** a header line `KIND FILE LEN`, then `LEN` bytes (at most
  256 KiB). A failed write on the sending side, or a pipe closed inside a
  frame on the receiving side, ends the process, so whatever runs it
  starts it again ([`scripts/feed-relay.md`](../scripts/feed-relay.md):
  a user unit restarting it every 5 s).

The writers are untouched: they keep writing on their own machine, which
keeps the whole logs; the far copy holds what was sent since the relay
began.

Measured 2026-10-08 (desktop to rack, eight seconds into a scratch
folder): eleven feeds, the rack's last `mic-desk` line the desktop's own
within the second.

C, compiled by tcc as it starts (`tcc -run`), nothing installed.
