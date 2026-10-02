# improve.rs

The self-improvement loop (`--improve`, with `--dev` and `--frame agent`):
the stream changes this program, the one it runs in, and puts the change
forward; the change is built and tested where it can do no harm, Claude
reviews it, it is measured on the running model, and it is kept only by a
rule written before the measurement. The stream reads the history of
every proposal and outcome (`improve.log`) to choose the next one: that
reading is what makes the loop recursive.

The person's request (2026-10-02): "a recursive self improving
environment for the model". Their choices: the harness first, then its
weights ("both, harness first"), and every change gated by a measured rule
and Claude's review ("nothing unreviewed runs").

## The stages, in order

1. **Propose** (built): the stream edits its working copy (the overlay's
   upper layer, `term.md`) and calls `propose` with a title and a why
   (`agent.md`). One candidate at a time; the tool answers at once, and
   the outcome comes at a later turn.
2. **Stage** (built): the candidate is the layer's files that differ from
   the repository's `HEAD`, recorded as its base, with deletions (the
   overlay's whiteouts) included. It is refused, with the reason told:
   - when it touches a denied path (`DENY`): the build (`build.rs`, the
     manifest and lock file, the `Makefile`, `.cargo/`), the scripts that
     launch the service, and the loop with its evaluators (`improve.rs`,
     the code benchmark's harness `code.rs`, the reflection gate
     `gate.rs`). The build runs what it builds, and an evaluator the
     stream could change would grade itself; these go through Claude and
     the person only;
   - when a file it changes was changed in the repository after the last
     commit the stream was told of (`head_told`): its copy was made from
     an older version, and building it would silently revert that commit
     there (the repository's `CLAUDE.md` warns of stale copies);
   - when it changes nothing.
   Staging exports the base with `git archive` into the candidate's
   `tree/`, makes it a repository of its own with the base as its one
   commit, writes the changes over it, and keeps the diff as
   `change.patch`: what Claude reviews and applies.
3. **Build and test** (built): `make check` (the docs check, format,
   clippy, the release build, the tests) in a sandbox of the terminal's
   kind (bubblewrap): `/usr` read-only, the tree read-write at its own
   path, the build's own target directory (`~/.cache/phi-stream/improve/target`,
   never the repository's `target/`, whose new builds the terminal's
   `attach --follow` would load) and cargo's home with the registry
   read-only, llama.cpp read-only at its own path (`build.md`), no
   network (`--unshare-all`), cargo offline, Python masked, a cleared
   environment; `nice -n 19` on CPUs 12 to 15 with 4 jobs (the stream's
   threads start from the first CPU), stopped after 20 minutes. The
   build and the tests execute code the stream wrote (`build.rs` is
   denied, but tests and macros run), which is why the sandbox matters.
   The outcome (passed, failed with the errors in short, refused) goes
   into `improve.log` in the workspace and to the stream at its next user
   turn; it wakes a rest. A candidate that passed is sent to Claude
   (`to-claude.md`, `[improve]`) with the path of its `change.patch`.
4. **Review** (Claude): the diff read before any of it runs outside the
   sandbox, with the same standard as any commit to this repository.
   The verdict is written as `review` in the candidate's directory and
   its workspace mirror, a `reviewed` line of `improve.log` and a message
   to the stream; a candidate accepted in review gets the `reviewed` mark
   that `improve-measure.sh` requires. The first, candidate 3 (a reset of
   the guide's tracking in `drop_guide`), was rejected: `drop_guide` runs at
   every new aside, so the resets would have removed the lens forks'
   spacing and zeroed the guide report's counts at each one.
5. **Measure** (built, `scripts/improve-measure.sh N`): the reviewed
   candidate's binary (kept with it when it passed) on the real model,
   interleaved with the base's (base, candidate, base, candidate, 10
   minutes each after a minute to settle, one fixed audit objective;
   `scripts/improve-measure.md`). The rule, written here on 2026-10-02
   before any candidate was measured: **rejected** if the candidate's
   service died or never ran, or if in **both** pairs it is worse than the
   base beyond a tolerance on any one measure (rate under 0.9 of the
   base's; think repeats over the base's plus 3 points; goal yes under the
   base's minus 0.05; unparsed checks over the base's plus 5 points);
   **kept** otherwise. One pair worse is noise until the other agrees;
   improvements are reported, not required (a fix may show in no window).
   What is measured depends on what the change touches:
   the engine's task mode turns the second chain off and `code.rs` sets
   the agent frame and the guide off, so changes to the chain, the guide,
   the persona or the tools are invisible to the code benchmark and are
   measured on the live service (repeated 8-grams of its thinking, the goal
   probe's `yes` and `mass` in `goal.log`, unparsed checks, the rate);
   changes to the core or the sampler also run the code benchmark as a
   regression check, compared task by task when greedy outputs repeat
   byte for byte, never by totals alone (a 40-task subset carries about
   three tasks of noise; not built yet). Measuring holds the cards, so the
   service restarts for each window.
6. **Keep**: a candidate that passes review and the rule is committed
   (credited to the stream, with no attribution lines for Claude), the
   stream's copies of its files taken out of its layer
   (`scripts/phi-stream.sh accept PATH`), and the service restarted on it.

## Its tools

The person (2026-10-02): "give the harness and management interface more
tools for the model", and "a better scope so it can be more dynamic and
synchronous and develop itself more easily". Its `run` stops at 60 s,
which no Rust build fits, so it could not compile its own changes; with
the loop on it has, beside `propose`:
- `build`: a trial: its working copy staged and run through the same
  sandboxed `make check`, numbered 0 (`trial/`, mirrored to
  `improve/trial/`), never sent to Claude, so it can fix compile errors
  and failing tests itself before it proposes;
- `diff`: its change against the repository's head (`diff_text`), with
  the repository's paths and a note when it touches a denied one;
- `revert`: one file back to the repository's version (`revert`: its copy
  or deletion mark taken out of the layer; a relative path inside the
  layer only), which it cannot do itself inside the overlay;
- `report`: its status, objective, the goal probe over the last ten
  minutes, what is building, the last entries of `improve.log` and the
  last lessons of Claude's reviews (`lessons.md` in its workspace: a line
  per rejected or corrected candidate, written with each review, so the
  mistake is in front of it when it chooses the next; the persona names
  it too).
One build at a time, trial or candidate.

## A trap: the working copy's lower layer is the live tree

The stream's overlay lays its writes over the repository's working tree,
not over `HEAD`: a file it edits is copied up whole, with whatever is
uncommitted in the tree at that moment. Its first trial (2026-10-02,
`trial/`) carried Claude's grounding code, uncommitted then, beside its
own one-line change. The staleness refusal catches it once those edits
are committed (the file changed after `head_told`), but the rule for
Claude is plain: commit before the stream edits, or work in another
worktree while the loop runs.

Since candidate 4 the loop merges instead of refusing (`rebase`): the
version a changed file was copied from is taken as the one, among the last
30 commits touching it (each commit and its parent), that the copy is
nearest to by changed lines; when that is older than the base, the
stream's change (that version to its copy) is merged onto the base's
version with `git merge-file`, the stream is told so, and a conflict
refuses the candidate with the file to redo. Candidate 4 was told of
eb3c4e1 and still deleted its `drop_real_paths`: the commits it was told of
never said when its copy was made, so that check is gone. Tested on a real
repository: a copy made before a commit keeps both the commit and the
change.

A format failure is told with its fix (rustfmt is in its terminal).

## What it reads

The person (2026-10-02): "the model needs to improve its harness with all
available information to it". Its sandbox sees the repository and its
workspace, so what it needs is put there, and the persona's loop paragraph
(`IMPROVE_AGENT`, `engine.md`) names each:
- `improve.log`, and `improve/cand-N/` for every candidate: `outcome`,
  `change.patch`, the whole `build.log` (not only the summary it is told)
  and `measure.txt` once measured (`record`, `improve-measure.sh`);
- `status.txt`, its own status every 10 s (the line `phi-stream status`
  prints, and its objective): the service's socket is outside its view;
- its own logs (`chain.log` with `tools/loopiness.c` to measure its loops,
  `goal.log`, `guide.log`, `reflect.log`, `notes.md`, `to-claude.md`);
- in the repository, `docs/results/`, each file's `.md` and `git log`.

## Contamination and the weights phase

The code benchmark (MultiPL-E) is for evaluation only: its licence
forbids it as training data. The stream never reads its files (they live
outside its sandbox's view, under `~/.cache/phi-stream/code/`) and is told
aggregate counts only, never task names. When the weights phase comes, its
data is the stream's own repository work and the tests it passed, never a
completion or a deliberation from a benchmark prompt.

## By hand

`phi-stream improve --upper DIR --title TEXT [--repo PATH]` builds a
candidate as the tool does, from any layer of changed files, and prints the
outcome; the candidates are numbered on across runs
(`~/.cache/phi-stream/improve/cand-N/`: `change.patch`, `build.log`,
`tree/`).

## Tests

On a real repository and a real git: a file that differs is a change and
a copy identical to the base is not; the staged diff holds the change; a
change to the manifest is refused and named. The denial list names the
build, the scripts and the evaluators and passes the engine. The build's
summary keeps the errors and their places and drops the progress lines.
The sandbox itself was run on this repository (its first build caught a
format error at the base, 8e4773a).

**The working copy kept up to date** (2026-10-02, after c2c4201): a stale
copy did more than revert in a proposal. Grounding reads the stream's
layer over the repository, so the head's new code was "nowhere in the
repository", the stream concluded the ground truth was broken, and went
round that conclusion for minutes. At every new commit it is told of
(`head_news`), `refresh_copy` merges each copy in its layer onto the head
in place (`rebase`), writes the merged file back, takes out copies that are
then identical to the head, says what it did in the line that tells it of
the commit, and the grounding re-reads the repository. With the loop on,
that line no longer says "review it" (each of Claude's commits had pulled
it off its objective to review), only "read it if it touches your work".
