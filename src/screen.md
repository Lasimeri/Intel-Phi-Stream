# screen.rs

The terminal's screen, double-buffered. The design is after BF++'s TUI
runtime ([Lasimeri/bfpp](https://github.com/Lasimeri/bfpp): `__tui_begin`
clears a back buffer, `__tui_end` diffs it against the front one and
emits only the cells that changed).

A `Screen` is a grid of cells, each one character with a style:
- `fg` and `bg` colours;
- a weight (plain, bold, italic, struck: crossed out, for a word a check
  wrote over).

The terminal (`tui.md`) draws every frame into a fresh screen:
- `put` writes text at a position, clipped at the right edge, with
  control characters shown as spaces;
- `fill` pads to the end of a row;
- `line` writes a whole row.
- `put_to` writes clipped at a given column, for text inside a
  compartment;
- `frame` draws the outline of a `Rect` with a label in its top edge, in
  `LIGHT` or `HEAVY` box drawing or `ASCII` (`utf8_locale` decides:
  the first of `LC_ALL`, `LC_CTYPE`, `LANG` that is set names UTF-8).

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

Columns are terminal columns (`columns`, from `unicode-width`): a wide
character (CJK, which the stream's tokens hold) takes two cells, itself
and a second-column cell that is never written (the terminal draws the
character over both); a zero-width character (a combining mark) takes
none; a wide character that does not fit the last column is a space. A
wide character half overwritten leaves a space in its other half, and
one whose second column changed is written again whole. Counting
characters, as before, put a mind strip of CJK tokens past the right
edge onto the next row (seen on the live service, 2026-10-01). The
terminal wraps the stream by the same widths.

Tests:
- an unchanged frame writes nothing;
- one changed word writes under 80 bytes where the whole screen is over
  1900;
- another size, or no last frame, redraws everything;
- text is clipped at the edge and newlines are blanked;
- wide characters take two columns and stay in their row, and the bytes
  written hold no second-column cell;
- a wide character that does not fit is a space; half overwritten ones
  leave spaces; a changed second column rewrites its character;
- a combining mark takes no column.
