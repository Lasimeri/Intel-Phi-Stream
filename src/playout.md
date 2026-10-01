# playout.rs

The display's own clock. The engine places pieces in jumps:
- a token every 25 ms alone;
- every 35 ms with a check's lane beside it;
- nothing for about a third of a second while a reading's or a question's
  chunk is decoded (the engine's thread is inside `llama_decode` then);
- a burst when a check lets its held token go.

Shown as placed, every jump shows. Pacing on the engine's own thread
cannot help, since that thread is blocked in the decode.

The design comes from BF++'s scene oracle (`runtime/bfpp_rt_3d_oracle.c`
in [Lasimeri/bfpp](https://github.com/Lasimeri/bfpp)). There a lock-free
triple buffer decouples the simulation's rate from the renderer's: the
consumer, on its own thread, takes the latest snapshot and extrapolates;
its confidence degrades as the snapshot ages; the lookahead is clamped
against runaway prediction. Here the consumer is a thread between the
engine and the clients:

- **Text** is held and goes out at the pace it has been arriving: an
  exponential average of the intervals between arrivals, each interval
  clamped to a third to three times the pace so far (a stall or a burst
  is the jitter to absorb, not the pace).
- **The step** is the pace scaled by `horizon / lag`, where lag is how
  long the oldest held piece has waited, clamped to 0.5 to 2. While the
  buffer is low (lag below the horizon) the text slows; while it is full
  it speeds up. The lag settles near the horizon. This is the oracle's
  graceful degradation: as the buffer drains the display slows, rather
  than stopping and then bursting.
- **Bounds**: nothing waits longer than twice the horizon (the oracle's
  clamp). There is no floor: text that arrives late, after a check held
  it longer than the buffer lasted, goes out at once and then at the
  pace, since the display was already starving. A first version held
  every piece at least a quarter of the horizon, which added that
  quarter to every such stall (`docs/results/2026-10-01-reflect.md`).
  Until four intervals have been seen, a piece goes out at the horizon.
- **Other events** (status, notes, readings of the mind, checks) pass at
  once, so the mind still runs ahead of the text it explains.
  - `Stopped` and `Done` send everything held first.
  - `Flush` (a pause) sends everything held.
  - `finish` (the engine done) drains and joins the thread, so a caller
    that collects events after `Engine::run` returns has all of them.

`--horizon SECS` is the target lag: 0 by default (as placed), 1 s with
`--reflect`. Taking back is not the playout's job: a check holds its
token and everything after it in the engine (`engine.md`), so nothing it
might change reaches the playout before the check ends. A stall shorter
than what the buffer holds does not show; a check that outlasts it shows
as a pause at its token, never as text taken back.

Tests (a synthetic clock):
- a zero horizon passes everything at once;
- a 344 ms stall in a stream of 25 ms tokens shows no gap over 100 ms;
- a check holding its token 1.25 s (longer than the horizon) shows no
  gap of half a second;
- no lag exceeds twice the horizon;
- the lag settles near the horizon;
- the end and a flush send everything, in order.
