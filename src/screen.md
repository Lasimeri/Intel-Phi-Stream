# screen.rs

The terminal's screen, double-buffered. The design is after BF++'s TUI
runtime ([Lasimeri/bfpp](https://github.com/Lasimeri/bfpp): `__tui_begin`
clears a back buffer, `__tui_end` diffs it against the front one and
emits only the cells that changed).

A `Screen` is a grid of cells, each one character with a style:
- `fg` and `bg` colours;
- a weight (plain, bold, italic).

The terminal (`tui.md`) draws every frame into a fresh screen:
- `put` writes text at a position, clipped at the right edge, with
  control characters shown as spaces;
- `fill` pads to the end of a row;
- `line` writes a whole row.

`diff` then writes what turns the screen the terminal shows (the last
frame) into this one:
- runs of changed cells, each after one cursor move;
- the style set only when it changes;
- nothing at all for an unchanged frame;
- everything when there is no last frame or its size differs (the
  first frame, a resize).

A frame in which one token arrived writes that token and whatever else
changed with it (the status line's rates, the clock in the title). It
never writes the whole screen, which the terminal did before at every
token: about 2 KB a frame at 80x24, more on a large terminal.

Columns are counted in characters, as everywhere in the terminal; a
character two columns wide would shift its row.

Tests:
- an unchanged frame writes nothing;
- one changed word writes under 80 bytes where the whole screen is over
  1900;
- another size, or no last frame, redraws everything;
- text is clipped at the edge and newlines are blanked.
