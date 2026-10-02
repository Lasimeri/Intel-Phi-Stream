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
  .toml .S`; never `target/` or `.git/`). In development the stream's
  working copy is read over it (`Repo::load_over`: the overlay's upper
  layer, what it wrote, by the same relative paths): checked against the
  repository alone, its own new files were "nowhere", and on 2026-10-02 it
  noted one proposal 131 times.
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

**Its thinking and its messages too** (2026-10-02; the person: "make sure
what's on the model's mind is not hallucinated and always based in reality
and ground truth"):
- **Each thinking line** that names code (`ground_line`, at the line's
  end; agent frame, in development, not while speaking or in a code
  block) is checked against the repository as its working copy sees it
  (read again after 30 s). A missing name, a file that does not exist or
  a line past a file's end is told to it beside its next turn as an
  aside ("ground truth, beside your thinking ... if you mean to add it,
  it is not there yet"), once in ten minutes per finding, and kept in
  `ground.log`. Its notes were checked, but a confabulated name in its
  thinking went on unchallenged into its messages.
- **Each message to Claude** (`send_claude`) is checked the same way: the
  check goes beside the message in `to-claude.md` (`[checked against the
  code: ...]`) and back to it in the tool's result. A message had cited
  strings and lines that were not there (m27).
- What the terminal shows as "on its mind" is the lens's reading of its
  residual stream (`mind.md`), a measurement, not its words.

**In the engine** (`add_note`):
- **A note with a missing name or a bad reference is kept**, marked
  `[unverified: ...]`, and the stream is told (a system line through the
  queue) what the repository holds.
- **Every `file:line` it names** is quoted back to it.
- **Notes already kept are checked at load**, so a false one carries its
  mark into the opening; an old mark is taken off and the note checked
  again (the code moves on), and a note kept twice is kept once.
- **`[unnote: TEXT]`** removes its notes containing TEXT, from memory and
  from `notes.md`.

Tests:
- the names of a real false note are found (`reflection_check`,
  `check_line`, `keep_at`, `mirror`) and plain words are not;
- a true note passes and a false one is caught, its `file:line` quoted;
- a line past a file's end and a missing file are reported;
- whole words only.
- **Real files are not findings** (`drop_real_paths`): the repository holds
  only its own files, so a workspace file (`lessons.md`), an absolute path,
  or a repository path written with its absolute prefix (cut at the space in
  "Intel Phi Stream") was told as "does not exist", and the stream rightly
  called that ground truth false. A path that exists on disk, in the
  workspace, or as a tail the repository holds is dropped from the findings.
