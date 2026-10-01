# tui.rs

`phi-stream tui` (`scripts/phi-stream.sh attach`): the stream as
something alive in the terminal, a client of the service (`client.md`),
crossterm only, drawn in seaof.glass's palette as Mechanical Jev's tui
is. Closing the terminal leaves the stream running.

- The middle is the stream, wrapped to the width by words (PgUp and PgDn
  scroll, End follows again): thoughts in the warm text colour, speech
  (a `»` line in the journal frame, after `</think>` in the chat frame)
  bold and lifted, what was given (what you said, what you handed over,
  the engine's marks, the mind's own `[note: ...]` lines are shown as
  thoughts) italic and cooler on the surface colour, placed where the
  stream was when it arrived, which is where the model has it too
  (`engine.md`). On connect the last 12k characters are shown first.
- The strip above the input says what it is doing with a slow pulse:
  thinking, speaking, reading what you gave it (with a bar), taking it in
  (the chase, with a bar), gathering its thoughts (the summary before a
  rollover or a new persona), paused; then the rates (live tokens a
  second, tokens read or caught up a second, the cycle), how full its
  context is, the queue, the chunk, the rollovers.
- When the service reads its mind (`--mind`, `mind.md`), a mind strip
  above the status shows the last token placed and the words on its mind
  at each block, with the readout's time; `/mind` switches the main area
  to the readings token by token (PgUp and PgDn scroll them), `/mind`
  again back to the stream.
- When it reflects (`--reflect`, `reflect.md`), the newest check shows
  at the head of the mind strip for eight seconds of the stream's clock
  (why, the token, its probability, what became of it, how long it
  took), the status counts the checks and changes (and says when one is
  in flight), and in `/mind` the reading a check was asked from carries
  the check beside it.- The input line: Enter says the line to the stream (heard at once when
  short, read beside the thoughts when long); `/feed FILE` hands a file
  over; `/persona FILE` gives it a new persona (the context rolls over
  onto it after a summary); `/pause`, `/resume`; `/chunk N` (0 adapts);
  `/temp T`; `/quit` stops the service; Esc clears; Ctrl-C leaves the
  terminal with the service running. The last line shows the keys, the
  workspace and the engine's last note.
- The title: the model, the frame, the placement (blocks on the GPU, the
  bytes the cards and the host hold), the cells, the time up, things
  heard, and the stream's clock: the real time of its newest piece, to
  the microsecond (`clock.md`). The mind view shows each reading's time
  the same way.

Nothing is drawn with ratatui or any widget library: each frame is drawn
into a fresh screen of styled cells, and only the cells that differ from
the frame the terminal shows are written (`screen.md`, after BF++'s
double-buffered TUI runtime). A frame is drawn when something changed or
every quarter second for the pulse, and in full after a resize. The service's lines are read on a
thread and handed to the drawing loop through a channel.

## It outlives the service, and follows the build

- **No service is a state, not an exit.** When the service goes (a
  restart, a crash) the strip says `NO SERVICE`, when it went and the
  socket, and the terminal looks for it every 3 s; when it answers, the
  stream's tail replays and the title and strip fill again. Started with
  no service yet, it says `CONNECTING`; while the model loads, `waking`.
  `/quit` stops the service and ends the terminal with it; with no
  service, `/quit` just leaves. A line typed with no service is not sent,
  and the last line says so.
- **`--follow`** (`scripts/phi-stream.sh attach --follow`, what the
  desktop window runs): the terminal looks at its own binary every
  500 ms. A build replaces the file at that path with a new one (a new
  inode); seen unchanged at two looks in a row (the build is done
  writing it), the new build is `exec`ed in this process's place, with
  the same arguments and the view handed over in `PHI_STREAM_TUI_STATE`:
  the view (`/mind`), the scroll, the counts, the time up, and the line
  being typed. The alternate screen stays, so nothing flashes; raw mode
  is left just before, so the new process records the terminal's own
  mode to restore when it ends. The last line counts the reloads. If
  the exec fails, the old build goes on and the next finished build is
  tried. The service is not touched: a client-only change reaches the
  terminal within a second of the build, while the stream runs on.
- **Version skew.** A reloaded terminal is often newer than the running
  service, which restarts only at a natural break (its restart costs the
  stream its context). So every feature of the terminal degrades against
  an older service: a command it does not know comes back as one `err`
  line on the last line, and a line it does not send yet is shown as not
  offered by this service, never as an empty or invented value.
- **A panic** restores the terminal before its message is printed; the
  launcher's `--follow` loop then resets the terminal (a crash by signal
  leaves raw mode and the alternate screen), says why it stopped, and
  runs the next build when there is one.
- Verified on the live service (2026-10-01) with one client in a tmux
  pane: a half-typed line survived a rebuild (same pid, the new inode
  running, `(1 reloaded)`); after the client was aborted by pid the loop
  reset the pane, waited, and ran the next build.

## Compartments

The screen is divided into framed compartments, each with an upper-case
label in its top edge (the management plan's layout, 2.6):

| at | compartment | holds |
| --- | --- | --- |
| row 0 | title | the placement, the cells, the time up, the stream's clock |
| 80 to 119 columns | the view (rows 1 to 13 at 24 rows), then ASSESSMENT (4 lines) | |
| 120 columns and more | the view on the left; a 40-column side column with ASSESSMENT (12 rows) above LOG | |
| row h-4 | MIND strip | the last reading of its mind |
| row h-3 | status strip | what it is doing, the rates, the checks |
| row h-2 | input | |
| row h-1 | keys and the last note | |

- **The view** is one of FEED (the stream), MIND (the readings token by
  token) and LOG; `Tab` cycles them, `/feed`, `/mind`, `/log` name one
  (`/feed FILE` still hands a file over; `/mind` again goes back to the
  stream). Its label names it and the next one.
- **ASSESSMENT** is the last check that ended, with its numbers exactly
  as the `reflect` line has them (`reflect.md`): the outcome word, KEEP
  and WRITE (keep's share of the two and the rest), ANSWERED (`fmt`), the
  model's probability of the token and the trigger, the three likeliest
  tokens at `Decision:` with their percent, the words on its mind, the
  rule that decided, the time, position and duration. In the side
  column the token stands in a frame of 5 rows by 13 columns with a tick
  at each side's midpoint (the plan's 2.4): light and white for a check
  that kept it; heavy and red for one that changed it; a white outline
  with heavy red ticks for one that would have (`--reflect-dry`). The
  heavy glyphs (or `#` in ASCII) carry the state without colour, and red
  is never text (4.44:1). The label says `ASSESSING NOW` while the
  status says a check is in flight; the token of a check in flight is
  not sent by this service, so it is not shown.
- **LOG** holds the checks and the engine's notes, oldest first, each with
  its time: a check's own time, a note's arrival (notes carry none),
  wrapped to the compartment, the last 500.
- With nothing yet, a compartment says what it knows: no status yet, no
  check since this terminal connected (with the service's count), no
  reading yet, or that the service does not read its mind (its status
  reports a reading time of 0).
- Under 80x24 one line says the size it needs, and nothing else is drawn.
- Outlines are box drawing when the locale (`LC_ALL`, `LC_CTYPE`, `LANG`,
  the first set) is UTF-8, ASCII otherwise.
- Verified on the live service (2026-10-01): captured at 80x24 and
  150x36, the panel's numbers equal to two decimals the `reflect.log`
  line of the same check (keep 0.8900, fmt 0.2563, top 0.2196, 0.2053,
  0.1855).

Tests: the compartments fit and never overlap from 80x24 to 200x60, each
with room for the token frame, the side column from 120 columns;
wrapping keeps to the width, CJK counted two; a reload hands the view,
scroll, counts and the typed line over, and reads an older build's
`mind=1`.
