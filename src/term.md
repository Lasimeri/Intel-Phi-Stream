# term.rs

The stream's terminal (`--terminal`, on in `scripts/phi-stream.sh dev`):
a line of the exact form `[run: COMMAND]` runs `COMMAND` with `sh -c`,
and the output comes back into the stream as a document when the command
ends (`engine.md`). The stream keeps thinking meanwhile: commands run on
a thread of their own, in order, one at a time (at most 4 waiting).

A command starts in the repository being developed (its paths are what the
stream reads by), else in its workspace. In development the repository is
mounted as an overlay: the repository as it is, with the stream's writes
in an upper layer kept beside the workspace (`dev-copy/upper`, outside
the sandbox's view), so it can edit, compile and test in place while the
repository itself never changes; its changes are files in that layer, for
Claude to review and apply. Measured on the live service: with commands
starting in the workspace, its `cat src/engine.md` failed (exit 1), and
`cat src/engine.rs | head` reported exit 0 although `cat` had failed.

The sandbox (bubblewrap), so a wrong or invented command can do little:
- it sees `/usr` (and the `/bin`, `/lib` links into it), the repository
  being developed (through its working copy, or read-only), and its own
  workspace read-write; nothing else of the home directory, so private files
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
repository reads and does not write (no working copy); the workspace
writes, and a command starts in the repository; with a working copy, a
write lands in its upper layer, a later command sees it, and the
repository has none; the home directory's private files are not there; Python
does not run; only `lo` exists; the time limit stops a command; the cap
cuts the output.
