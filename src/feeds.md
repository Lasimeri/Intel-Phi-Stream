# feeds.rs: camera-feed monitoring

The person's objective for the stream (2026-10-03): a monitoring system
for their camera feeds. The cameras report in text, the stream reads no
pictures: each camera's face tracker (`facetrack`, in the
[optical-079](https://github.com/Lasimeri/optical-079) repository) writes,
once a second, one line to `feeds/NAME.status` in the stream's workspace
(rewritten) and to `feeds/NAME.log` (appended):

```
t=1791020403094649 cam=bedroom faces=0 locked=0 box=0,0,0,0 motion=0.007 light=36 fps=10
```

`t` microseconds of real time, `faces` the most seen in one frame that
second, `locked` the tracker's reticle on a face, `box` X,Y,W,H, `motion`
the mean frame difference (0 to 1), `light` the mean grey (0 to 255).

## Polls and events

`Engine::poll_feeds` reads every `.status` at most every 5 s
(`FEEDS_EVERY_US`; it is called where `poll_term` is, after a cycle, in
the agent-wait loop and in the rest loop). A camera's face count is the
most in any second of its log's last five (`faces_in_window`): the
detector loses a face for a frame or a second. `compare_snapshots` takes
each camera's state at the previous poll and at this one (fresh: its
newest line within 10 s, `OFFLINE_US`) and tells each change once, when
it starts:

- **offline**: fresh, then stale or gone; **online**: first seen, or back;
- **a face appeared**: none in the window, then one; **no face for 5 s**;
- **motion with no face**: above 0.15 with no face, starting;
- **dark**: light under 20, starting.

While a condition holds nothing more is said: the stream's first version
(candidate 21) pushed the same event every poll, many times a second, and
its second (22) re-told a stale camera as offline every poll and set its
throttle at 5 000 000 000 microseconds (83 minutes); both fixed in
Claude's review, credited to the stream. Each event is appended to
`feeds/events.log` and given to the stream as one line at its next user
turn (`self.waiting`), `[HH:MM:SS] camera NAME: ...`.

Tests: a status line parses; a camera going stale is offline once (and
not again on the next polls); coming back is online; a face, motion and
dark are told when they start and not while they last; a face counts over
the last five seconds of the log.
