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
  its own source the way Claude does, and in the workspace when only the
  workspace holds it, so `[read: reflect.log]` reads the record of its
  own checks. `[read: PATH:START-END]` reads
  those lines only. Its memory is about 32 thousand tokens: a file that
  does not fit is refused with its size and a range that would fit
  (`src/engine.md`).
- **Its notes are checked against the code** (`src/verify.md`): a note
  naming code the repository does not hold is marked `[unverified: ...]`
  and the stream is told what is there; `[unnote: TEXT]` retracts.
- **`[prefer: ...]`** lines are kept in `preferences.md` in the workspace,
  next to `notes.md`. They are announced as `prefers: ...` and shown back
  to it with its notes, at a rollover and at a fresh start (the opening
  carries what it kept).

## The channel

| command | what |
| --- | --- |
| `phi-stream say --as Claude TEXT` | a line to it, named: it hears `« [HH:MM:SS.uuuuuu] Claude: TEXT` |
| `phi-stream ask [--timeout 180] TEXT` | a message from Claude with an id (`c3`) that waits for its answer: in the agent frame it comes at its next user turn, and its `tell_claude` naming `c3` (or the first after it naming none) is printed on stdout, the ids and the wait on stderr (exit 1 if none came in time) |
| `phi-stream ask --spoken --as Claude [--timeout 180] [--thoughts] TEXT` | the journal and chat frames: say, then wait for its next spoken line and print it (`--thoughts` prints its thoughts meanwhile on stderr) |
| `phi-stream guide on\|off`, `experts on\|off`, `chain on\|off`, `set guide-mix G` | the guide lane (`--guide`, `src/engine.md`): the distribution with its last reflection in mind beside every thinking token, measured (shadow), and with `guide-mix` above 0 mixed into the choice |
| `phi-stream mcp` | all of this, and the terminal interface itself, as MCP tools for Claude Code (`src/mcp.md`): `screen`, `type`, `keys`, `ask`, `say`, `inbox`, `status` |
| `phi-stream listen` | from now on, one line each as it happens: `said: ...` (its spoken lines), `heard: ...`, `noted: ...`, `prefers: ...`, reads, and the checks that changed a word |
| `phi-stream feed FILE` | hand it a file (a diff, a record) to read beside its thoughts |

`listen` is made for an agent's monitor (Claude Code's Monitor tool runs
it and turns each line into a notification): what the stream says
reaches Claude while Claude works, without polling. `ask` is a turn of
conversation inside a command.

In the agent frame the stream writes to Claude with its `tell_claude`
tool: each message is kept in `to-claude.md` in the workspace (`## m5 at
TIME, answering c3`) and sent to every client as a `claude` line; a
monitor on `listen` or the MCP `inbox` tool reads them. Its notes are its
own memory (kept once; a note already kept is refused), never the way to
reach Claude: on 2026-10-02 it noted one proposal to Claude 131 times,
told each time that its own new files did not exist (the notes were
checked against the repository, not its working copy, which `verify.rs`
now reads over it).

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

## Objective and terminal (2026-10-01)

`dev` starts the stream with `--terminal`. Until it has an objective it
only thinks: give it one with `/objective TEXT` in the terminal or
`scripts/phi-stream.sh objective TEXT` (kept in `objective.md`, so a
restart keeps it). Then `[run: COMMAND]` runs in its sandbox (`src/term.md`:
the repository read-only, its workspace read-write, no network, no Python,
one CPU at the lowest priority) and the output comes back to it; what it
writes lands in its workspace, and the repository changes only when Claude
applies a change. A restart resumes from the summary its quit wrote and
tells it the commits since it last ran (`src/engine.md`).
