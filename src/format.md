# format.rs

The stream's text set for the terminal (`tui.md`): pure functions from
the stream's lines (characters with the kind of piece each came in:
thought, speech, given) to rows of at most a given width, each character
with a class that decides its style.

**Code blocks.** A fence line (```` ``` ```` or `~~~` after up to three
spaces) opens a block, the next one closes it; the info string names
the language. Inside:
- lines are kept as written: their indentation, tabs as four spaces,
  never reflowed. A line longer than the row is cut, and the rest
  continues two columns past the line's own indentation;
- highlighting by language: keywords, strings (with escapes), numbers,
  comments (`//` and `/* */` across lines, or `#` for shell and Python).
  Languages: Rust (`rust`, `rs`; a lifetime `'a` is not a string), C and
  C++ (`c`, `h`, `cpp`, ...; the preprocessor's directives as keywords),
  TypeScript and JavaScript (`typescript`, `ts`, `tsx`, `javascript`,
  `js`, ...; template strings in backticks), shell (`sh`, `bash`, `zsh`,
  `fish`, `console`), Python; any other or none as C;
- a block still open (the stream is writing it) is code to the end.

**Prose.** Wrapped by words to the width, its leading spaces kept, the
lines after the first under its hanging indent (its leading spaces and a
list marker: `- `, `* `, `1. `). The light Markdown it writes shows as
a style instead of its marks: a `#` heading (bold, lifted), `**bold**`,
`` `code` `` (the code palette). A mark is taken out only when it closes
on its line, so one being written shows as it is until it closes.
Before this, every line was reflowed by words and lost its leading
spaces, so code came out flat (seen on the live service, 2026-10-01).

Widths are terminal columns (`screen.md`: a CJK character takes two).

The terminal's styles (`tui.rs` `style_of`): prose by its kind as
before; code on its own background (`#16161e`) with keywords in the
management plan's yellow, strings in its green, numbers white, comments
italic in the given colour. Every text style is at least 4.5:1 (a test
computes WCAG contrast for each kind and class).

Tests: code keeps its indentation and is never reflowed; TypeScript is
highlighted (keywords, a template string, a comment, a type name left
plain); C comments span lines and strings hold escapes; Rust lifetimes
are not strings; long code lines are cut under their indent; prose wraps
with its hanging indent; Markdown marks become styles only when closed;
an open block stays code; CJK counts two columns.
