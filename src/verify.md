# verify.rs

A note of the stream's, checked against the code (development only,
`docs/dev.md`).

In the first dev sessions the stream noted code that does not exist and
attributed some of it to Claude:
- a `mod mirror` in `reflect.rs`;
- `mirror.check_line(text)` at `engine.rs` line 449;
- a `CHECK_THRESHOLD = 0.55` in `check.rs`;
- `self.mirror.reflect(line_idx)`.

Its notes come back at every rollover and start, so one false note
re-seeded the same confabulation after each correction, even right after
it read a file of the real lines.

**The check:**
- **Load.** `Repo::load` reads the repository's text files (`src/`,
  `scripts/`, `docs/`, `tools/` and the top level; `.rs .md .sh .c .h
  .toml .S`; never `target/` or `.git/`).
- **Names.** `code_names` takes the code names a note uses:
  - what is inside backticks;
  - every identifier with an underscore;
  - a name that is called (`name(`);
  - every part of a `::` path;
  - the name after `mod`, `fn`, `struct`, `enum`, `impl`, `trait` or
    `const`.

  Plain words are not code names.
- **File references.** `file_refs` takes the files a note names
  (`src/engine.rs`, `engine.rs:449`, `engine.rs line 449`).
- **Results.** `Repo::check` reports:
  - each name no file holds as a whole word;
  - each file that does not exist, and each line past a file's end;
  - for each `file:line`, the real line.

Nothing is inferred: a name is missing only when no file holds it.

**Summaries too.** A summary it carries forward (into the base of a
rollover, or into the opening after a restart) is checked the same way,
and a line `[checked against the code: ...]` follows it when it names
what the repository does not hold. In the dev session its summary
carried "two gates" and a "mirror gate" through a rollover.

**In the engine** (`add_note`):
- **A note with a missing name or a bad reference is kept**, marked
  `[unverified: ...]`, and the stream is told (a system line through the
  queue) what the repository holds.
- **Every `file:line` it names** is quoted back to it.
- **Notes already kept are checked at load**, so a false one carries its
  mark into the opening.
- **`[unnote: TEXT]`** removes its notes containing TEXT, from memory and
  from `notes.md`.

Tests:
- the names of a real false note are found (`reflection_check`,
  `check_line`, `keep_at`, `mirror`) and plain words are not;
- a true note passes and a false one is caught, its `file:line` quoted;
- a line past a file's end and a missing file are reported;
- whole words only.
