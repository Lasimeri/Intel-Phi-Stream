# term.rs

The stream's terminal (`--terminal`, on in `scripts/phi-stream.sh dev`):
a line of the exact form `[run: COMMAND]` runs `COMMAND` with `sh -c`,
and the output comes back into the stream as a document when the command
ends (`engine.md`). The stream keeps thinking meanwhile: commands run on
a thread of their own, in order, one at a time (at most 4 waiting).

The sandbox (bubblewrap), so a wrong or invented command can do little:
- it sees `/usr` (and the `/bin`, `/lib` links into it), the repository
  being developed read-only, and its own workspace read-write, where the
  command starts; nothing else of the home directory, so private files
  never enter its context;
- no network (`--unshare-all`: only `lo` exists), a fresh `/tmp`, its own
  process tree, a cleared environment (`PATH=/usr/bin`, `HOME` the
  workspace);
- Python masked (each interpreter's file bound over by `/dev/null`): the
  person's standing instructions forbid it, and they are its manner too;
- at the lowest priority (`nice -n 19`) on the last CPU (`taskset`): the
  stream's own threads start from the first, and a second process on
  them cost the stream about ten times (xks measured 2026-10-01);
- stopped after 60 s, the output (stdout and stderr) cut at 16 KiB.

`[run: ...]` lines wait for an objective like its other tool lines
(`engine.md`). Each command and its end are sent to the terminals
(`term start`, `term end` lines, `client.md`) and shown in TERMINAL
(`tui.md`).

Tests, on the real sandbox (skipped where bwrap is missing): the
repository reads and does not write; the workspace writes and is where a
command starts; the home directory's private files are not there; Python
does not run; only `lo` exists; the time limit stops a command; the cap
cuts the output.
