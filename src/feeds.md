# feeds.rs: Camera-feed monitoring

Parses `feeds/NAME.status` key=value lines, compares snapshots, emits events to `feeds/events.log`.

## Status line format

Each camera writes one line every second:

```
t=1791020403094649 cam=bedroom faces=0 locked=0 box=0,0,0,0 motion=0.007 light=36 fps=10
```

Fields: `t` (microseconds), `cam`, `faces`, `locked`, `box=X,Y,W,H`, `motion`, `light`, `fps`.

## Events

- **OFFLINE**: camera in previous snapshot but absent or stale (>10s)
- **FACE_APPEARED**: faces went from 0 to N
- **FACE_LEFT**: faces went from N to 0
- **MOTION_NO_FACE**: motion > 0.15 with no face detected
- **DARK**: light < 20
- **ONLINE_FACE**: new camera with faces > 0 (first seen)

Events are appended to `feeds/events.log` as `[t=US] EVENT cam=NAME details`.

## Engine integration

`Engine::poll_feeds()` is called after each `poll_term()` in the main loop (after, awaiting, rest). It reads all `.status` files, compares with the previous snapshot stored in `feeds_prev`, appends events, and pushes event lines to `self.waiting`.
