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
5. **Measure** (not built yet): the reviewed candidate's binary on the
   real model, interleaved with the base (this host drifts a quarter over
   tens of minutes). What is measured depends on what the change touches:
   the engine's task mode turns the second chain off and `code.rs` sets
   the agent frame and the guide off, so changes to the chain, the guide,
   the persona or the tools are invisible to the code benchmark and are
   measured on the live service (repeated 8-grams of its thinking, the goal
   probe's `yes` and `mass` in `goal.log`, unparsed checks, the rate);
   changes to the core or the sampler also run the code benchmark as a
   regression check, compared task by task when greedy outputs repeat
   byte for byte, never by totals alone (a 40-task subset carries about
   three tasks of noise). The rule is written before the first candidate
   is measured. Measuring holds the cards, so the service stops for it.
6. **Keep**: a candidate that passes review and the rule is committed
   (credited to the stream, with no attribution lines for Claude), the
   stream's copies of its files taken out of its layer
   (`scripts/phi-stream.sh accept PATH`), and the service restarted on it.

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
