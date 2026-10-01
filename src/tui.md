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
- The input line: Enter says the line to the stream (heard at once when
  short, read beside the thoughts when long); `/feed FILE` hands a file
  over; `/persona FILE` gives it a new persona (the context rolls over
  onto it after a summary); `/pause`, `/resume`; `/chunk N` (0 adapts);
  `/temp T`; `/quit` stops the service; Esc clears; Ctrl-C leaves the
  terminal with the service running. The last line shows the keys, the
  workspace and the engine's last note.
- The title: the model, the frame, the placement (blocks on the GPU, the
  bytes the cards and the host hold), the cells, the time up, things
  heard.

Nothing is drawn with ratatui or any widget library: rows of styled runs
are queued and flushed, the screen redrawn when something changed or
every quarter second for the pulse. The service's lines are read on a
thread and handed to the drawing loop through a channel.
