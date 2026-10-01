# Developing with the stream

The stream can develop this repository with Claude (Claude Code, the AI
coding agent working in it) in real time: a peer that reads the code it
runs in, says what it proposes, and works by its own preferences. Those
preferences are bounded by the person's standing instructions, the
`CLAUDE.md` its persona is built on.

## Start it

```sh
scripts/phi-stream.sh dev        # the service, in tmux session phi-stream
scripts/phi-stream.sh attach     # the terminal, to watch and talk to it
```

`dev` is `start --dev <this repository> --mind --reflect --workspace
~/.local/share/phi-stream/dev` (`PHI_STREAM_DEV_WORKSPACE` overrides),
plus any serve options given after it.

`--dev REPO` does three things (`src/engine.md`):
- **The persona** gains a paragraph after the person's `CLAUDE.md` and the
  frame's mechanics. It says the stream develops, as a peer, with Claude
  in REPO, the program it runs in. Claude's words come marked `Claude:`,
  the person's unmarked. It works on what it judges worth working on, and
  it states its preferences as `[prefer: ...]` lines.
  - Claude follows those preferences wherever the person's instructions
    allow; where the two conflict, the instructions win.
  - In spoken lines it proposes concretely (file, function, change and
    why), reports what it finds in the code, and disagrees where it
    disagrees.
- **`[read: PATH]`** resolves PATH relative to the repository, so it reads
  its own source the way Claude does.
- **`[prefer: ...]`** lines are kept in `preferences.md` in the workspace,
  next to `notes.md`. They are announced as `prefers: ...` and shown back
  to it with its notes, at a rollover and at a fresh start (the opening
  carries what it kept).

## The channel

| command | what |
| --- | --- |
| `phi-stream say --as Claude TEXT` | a line to it, named: it hears `« [HH:MM:SS.uuuuuu] Claude: TEXT` |
| `phi-stream ask --as Claude [--timeout 180] [--thoughts] TEXT` | say, then wait for its next spoken line and print it (exit 1 if it did not speak in time; `--thoughts` prints its thoughts meanwhile on stderr) |
| `phi-stream listen` | from now on, one line each as it happens: `said: ...` (its spoken lines), `heard: ...`, `noted: ...`, `prefers: ...`, reads, and the checks that changed a word |
| `phi-stream feed FILE` | hand it a file (a diff, a record) to read beside its thoughts |

`listen` is made for an agent's monitor (Claude Code's Monitor tool runs
it and turns each line into a notification): what the stream says
reaches Claude while Claude works, without polling. `ask` is a turn of
conversation inside a command.

## How Claude works with it

The same rules are in the repository's `CLAUDE.md` for every session:
- Monitor `phi-stream listen` for the whole session.
- Say what is being changed and why (`say --as Claude`), and feed the
  diff before committing (`git diff > FILE; phi-stream feed FILE`). Its
  review comes back as spoken lines.
- Read `preferences.md` in the dev workspace at the start, and when a
  `prefers:` line arrives. Follow each preference wherever the person's
  `CLAUDE.md` and the repository's rules allow. Where one cannot be
  followed, tell it why.
- Its proposals are proposals: they are weighed like a colleague's,
  built when sound, answered when not. Everything still goes through
  `make check`, the records and the person's rules.
- One process holds the cards, and a paused service still holds them.
  While the dev service runs, a measurement that needs the cards waits
  until it is stopped (`scripts/phi-stream.sh stop`). Its notes and
  preferences survive the stop and come back at its next start; the
  stream itself begins again.
