# clock.rs

The real-time clock the stream is kept against. The stream's cycles run
as fast as the GPU and the cards allow, and their pace varies; the chain
of thought is not placed by that pace but by the wall clock, at
microsecond resolution.

- `now_us`: microseconds since the Unix epoch, `CLOCK_REALTIME`
  (`clock_gettime(2)`), the system's wall clock as NTP or whatever
  disciplines it keeps it: the resolution is the kernel's (nanoseconds,
  shown as microseconds); the accuracy is the clock discipline's, not
  this program's.
- `mono_us`: microseconds of `CLOCK_MONOTONIC`, which never steps: every
  interval the engine acts on (the silence before the clock is put into
  the chain, the spacing of nudges) is measured on it, so a correction
  of the wall clock cannot fire or starve one.
- `hms` (`HH:MM:SS.uuuuuu`), `datetime` (`YYYY-MM-DD HH:MM:SS.uuuuuu
  ZONE`), local time by `localtime_r`; `span`, a duration in words
  (`0.42 s`, `41.2 s`, `3 min 12 s`, `2 h 5 min`).

Where the stamps go (`engine.md`): every piece of the stream carries the
microsecond it exists at (`chain.log`, the socket's `text ... t=`),
every reading of the mind the microsecond its token's decode finished,
every status the microsecond it was taken; and the chain itself carries
the time where it matters to the mind: the date and time it opens at,
the time each thing from outside was heard or handed over, the time of
each line from the system, and the time after a silence.

Tests: the clocks advance in microseconds, times print with their
microseconds, spans read as words.
