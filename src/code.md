# code.rs

`phi-stream code ...`: whether the stream writes code that works, as a
repeatable measurement rather than an impression.

**The tasks** are MultiPL-E's translation of HumanEval to Rust (156
tasks; `scripts/fetch-code-eval.sh` fetches them pinned and converts
them): a prompt (the doc comment with examples, the function's signature
and its opening brace), the stop sequence `\n}`, and the tests (the
closing brace and a `main` of `assert_eq!`s).

**Judging.** A program is assembled, compiled with `rustc --edition 2021`
and run. Model-written code is untrusted: both steps run under
`unshare -rn` (a user namespace with a network namespace of its own,
loopback only: no network), in a process group of their own, with
limits (address space 4 GiB to compile and 1 GiB to run, CPU time 120 s
and 10 s, files 64 MiB, no core files), killed as a group at the wall
limit (120 s and 10 s), in a run directory on disk
(`~/.cache/phi-stream/code/<label>-<time>/<task>/`). The verdict: no code,
compile error, test failure (a failed assertion or any other non-zero
exit), timeout, or pass; the output of the step that decided it is kept.

**`code anchor`** is the benchmark's own protocol, the number comparable
with published ones and the check that the split (GPU, cards, host)
writes correct code at all: the raw prompt with no chat template and no
persona, greedy, completed until the first stop sequence, the end of
generation or `--max-tokens` (512), cut at the stop, assembled as prompt
+ completion + tests. Options: `--tasks FILE`, `--first N`, `--only
NAME,NAME`. Nothing of the stream's own machinery is involved (no
workspace, no injections, no penalties), so the number is the model's
and the split's alone.

Each run writes `results.jsonl` (per task: verdict, tokens, seconds, the
completion, the deciding output), `summary.txt` (the pass rate, every
failure class, the failing tasks by name) and `EVALUATION-ONLY.txt`.

**Evaluation only.** MultiPL-E's licence (BSD 3-Clause with a machine
learning restriction) says in its clause 4 that its contents "may not be
used as training data for any machine learning model". Evaluating with
it is what it is for; nothing a run produces (tasks, completions, any
reflection episodes) may go into a fine-tune, which would also void the
measurement. Every run directory says so.

Tests: the stop cut; the sandbox judges a passing, a failing, a broken
and a network-using program correctly (the last fails to connect: no
network).

**`code stream`** is the stream's own way, the second number (not
comparable with published ones): each task through the engine in its
task mode (`engine.md`: the chat frame, thinking, stopped at the end of
the first answer, nothing put into the chain on the engine's account,
a throwaway workspace per task), greedy, the repetition penalty an
explicit setting (`--penalty`, 1 off), a thinking budget after which
`</think>` is placed (`--think-budget`, 2048), the persona's base the
user's `~/CLAUDE.md` (`--base claude-md`, as the stream runs), a neutral
paragraph (`--base neutral`) or a file; `--mind` reads the mind at
every token while answering. The user turn asks for the complete
function in one rust code block, no `main`, no tests.

The extraction rule, fixed before the first run: the answer is what
follows `</think>`; its last fenced block is taken; it must define the
task's function (`fn NAME`, the name from the prompt's last `fn` line),
else the verdict is no code; a `fn main` the model wrote is removed
(braces matched, braces in its string literals not told apart: a stated
limit); the program is that block and the tests without their leading
`}` (which closes the prompt's half-written function in the anchor).
Each task's record keeps its thinking tokens, whether the budget closed
them, the answer, the code and the thoughts.

The anchor's own limits, seen in its failures and not the harness's:
the stop sequence `\n}` ends a completion at the function's closing
brace, so a helper the model writes after it is cut off and the call
fails to compile (prime_fib, match_parens); tasks that need a crate
(string_to_md5, `md5`) cannot compile, since no crate is available. The
stream mode, which takes whole answers, has neither limit.
