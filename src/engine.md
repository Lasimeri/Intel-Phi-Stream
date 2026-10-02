# engine.rs

The stream. One context (`llm.rs`), four sequences, one thread.

**The live sequence** generates without pause: every cycle decodes the
pending token at its position, samples the next (the chain of `llm.rs`),
and emits its text as `Think`, or `Speak` after the model closes its
thoughts with `</think>`. When the turn ends (`<|im_end|>`) a silent user
turn is decoded straight in and the thoughts are reopened. `history` is
the live sequence's tokens, index equal to position.

**Hearing and reading.** What is said (`Command::Say`) or handed over
(`Feed`) is framed in square brackets and queued. Up to `direct_max`
tokens it is heard at once: decoded into the live sequence in one cycle
after the pending token (a few hundred milliseconds). Longer, it is read
beside the stream: a free sequence is given the live one's cells up to
the current position and its recurrent state (`seq_cp`), the first chunk
is decoded alone (the two share the state's cell until the new sequence
writes its own), then every cycle carries the live token and the next
chunk (`chunk` tokens; 0 adapts to what remains: 8 up to 64 tokens, 16
up to 512, 32 up to 2048, else 64). The live token keeps coming, slower:
the host blocks' multiplies of both run one cycle at a time on the cards
and the host pool (`probe.md` has the curve).

**Joining.** When the reading ends, a third sequence is composed: the
live prefix (`seq_cp` of the cells before the fork), then the read cells
and the recurrent state after the reading (`seq_cp` from the reading
sequence, which also makes that state the new sequence's), the reading
sequence dropped. Then the chase: the thoughts produced since the fork are
decoded into the composed sequence in chunks beside the live token, and
whatever the live sequence adds meanwhile joins the chase. The cycle that
can take the rest (the thoughts left and the pending token) takes it
alone with logits: the next live token is sampled from the composed
sequence, which becomes the live one; the old live sequence is dropped
(its cells past the fork return to the pool; the prefix stays shared).
The model's view is then `[...thoughts][what was read][thoughts produced
while reading][...]`: what arrived is placed where the stream was when
it arrived, and the text already shown is unchanged. A `Given` mark names
the join.

**The summary at a rollover** is asked for with its first words already
written in its own voice ("What I was working on: "), and its end mark
(`---`, or the end of its turn) counts only after 96 tokens. In the dev
session it once restated the ask one word a line and ended it with `---`
after 34 tokens, losing its thread.

**Rollover.** Past `rollover_at` of the pool (0.6) with nothing in
flight, the engine decodes a request for a summary straight in and
collects what the stream writes until its `---` line (or `summary_max`
tokens, the turn's end, or `</think>`). The new base (the system prompt,
the summary as the first user turn, the thoughts reopened) is read as a
reading whose prefix is empty (positions from 0, a fresh recurrent
state), joined the same way with a chase of everything produced since;
the old sequence goes and the pool is mostly free again. The recurrent
memory of the old stream survives only through the summary, which is
what the architecture allows: llama.cpp cannot shift this model's cells
(`get_can_shift` is false for M-RoPE) and a recurrent state cannot be
cut.

**The mark `«` is never sampled** in the journal frame (every token
carrying it is banned, as the dash-carrying ones are): lines beginning
with `«` come only from outside. In the dev session the stream wrote
hundreds of bare `«` lines, a loop no nudge broke.

**The journal frame** (the default; `--frame chat` keeps the model's
template). The text is one continuous first-person journal with no
turns: the persona is a paragraph at the top, what comes from outside is
a line beginning with `«`, what the mind says aloud is a line beginning
with `»` (shown as speech), everything else is thought. An end-of-text
token sampled in this frame is replaced by a newline (a journal has no
end); three in a row bring a word from the system. A `«` line the mind writes
itself (a line in someone else's voice, the frame's one known leak) is
shown as thought and counted in the status. In the chat frame the
end of a turn is followed by the assistant's turn reopened at once (no
silent user turn), so the floor stays the mind's. Both frames carry two
lines the mind writes for itself: `[note: ...]` is kept in the workspace's
`notes.md` and shown to it again in every new base (a rollover, a new
persona), and `[read: PATH]` brings the file (a regular file under 1 MiB,
relative to the workspace or absolute) in as a reading, or tells it why
not: each failing path goes into the chain once, with what its nearest
directory holds, and the same path asked again within five minutes is
dropped quietly (a note outside the chain, counted in the status as
`reads_quiet`). A first version let one failure line a minute in,
whatever the path; it hid most failures and their listings, and the dev
stream asked for the same missing file every few seconds. The workspace (`--workspace`) holds `persona.md` (written at start,
reloaded by `persona`, edited in place between runs), `notes.md` and
`stream.log` (everything shown, appended).

**The persona is composed** (`compose`): the frame's preamble, then the
base quoted line by line (every line prefixed `> `, so it reads as a
document the mind has read, never as its own voice; without the quoting
the journal copied the base back as its first text), then the frame's
mechanics. The base is a person's
standing instructions, their `CLAUDE.md` (`--personality FILE`, else
`~/CLAUDE.md` when it exists, else `DEFAULT_BASE`), taken verbatim; the
preamble tells the mind to take the manner it prescribes (dense,
analytical, neutral, no hedging, mechanism over judgment, no em or en
dash) as its own, and to read its talk of responses, delimiters, tools
and memory files as another harness's, memory being the `[note: ...]`
line here. `--system FILE` replaces the whole composition verbatim. The journal's
opening ends with a rule line, the seed, and the journal's first words in
its own voice (`--first-words`, "Where was I. "): without them the model
analysed the base as a task; with them it goes on as itself. In the
journal frame the template's control tokens (`<think>`, `</think>`,
`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) are never sampled (a logit
bias of minus infinity, `llm.md`), since the journal has no template. The
sampler backs the dash rule: every vocabulary token whose text carries
U+2014 or U+2013 gets a logit bias of minus infinity (`llm.md`), unless
`--allow-dashes`.

**A new persona** (`Command::Persona`, the text of a file) becomes the
new base, recomposed with the same frame, and takes effect the way a
rollover does: the summary is asked for with nothing in flight, and the
new base is the new persona, the summary and the notes. The workspace's
`persona.md` is the composed text as last written, a record to read,
not read back.

**Why one context.** The backend's lock is taken and dropped inside
`phi_ggml_begin` and `phi_ggml_end` separately (`host/asm/common/lock.md`:
one caller at a time in practice), with one request in flight per card
and static argument blocks between the two: two contexts in two threads
would interleave inside a multiply. One thread, one context, several
sequences in one batch gives the same overlap without touching the
backend, and llama.cpp's hybrid memory splits such a batch by itself.

**Keeping it moving.** The sampler carries a repetition penalty (1.05
over the last 256 tokens by default, llama.cpp's penalties sampler),
and the engine watches the last 192 live tokens: a 6-gram seen five
times is circling, and a nudge (a bracketed line asking the thoughts to
move on) is decoded straight in, at most once in 256 tokens. A document
is framed with an opening and a closing line so the join reads as its
end.

**Real time** (`clock.md`). The chain is placed on the wall clock, not
the cycles: the opening carries its date and time to the microsecond,
every line from outside the time it was heard or handed over (`« [HH:MM:SS.uuuuuu] ...`),
every line from the system the time it was written, a rollover base the
date and time; after `--time-every` seconds (60) with nothing from
outside, the time and the length of the quiet are put in (`« [HH:MM:SS.uuuuuu]
(nothing from outside for 2 min 0 s)`), so the mind can reason about
when, not only what. Every piece of the stream goes out stamped with the
microsecond it exists at (`chain.log`: `t_us<TAB>kind<TAB>text`, and
the socket); circling thoughts are nudged at most once per
`--nudge-every` seconds (60). Intervals are taken on the monotonic
clock.

**A task** (`Config::task`, `code.md`): the same engine stopped at the end
of its first answer (`Event::Done` with its thinking tokens, then
`Stopped`), with nothing put into the chain on the engine's account (no
clock lines, nudges, reads, rollover), no wall clock in its opening or in a
check's question (a task is a measurement: greedy on a deterministic
backend it gives the same answer on every run, which a time to the
microsecond in the prompt prevented), and a thinking budget after which
`</think>` is placed; `run` hands the model back, so one load serves a
whole set of tasks. The run's sampling is set explicitly when the engine
starts.

**Reads it asks for** (`[read: PATH]`, `[read: PATH:START-END]` for
those lines, 1-based and inclusive, `END` or `end` for the last line):
- **Must fit.** A file is read beside the live sequence, then joined. The
  reading's cells, the thoughts placed meanwhile and the chase all come
  out of the cells the live sequence leaves, so a read must fit what is
  left (`read_room`): the free cells less 2048, less the thoughts, and
  never more than an eighth of the context (about 4k tokens). A read
  that does not fit is refused into the chain with its size, its line
  count and a range of about 2k tokens. With a third as the cap, the
  stream took the largest range offered (10.9k tokens of `main.rs`) and
  went straight to a rollover.
  - In the first dev session, one 21026-token file read at position
    18000 filled the 32768 cells. The decode failed and the service
    stopped.
- **A directory reads as its listing** (sorted, directories with a
  trailing `/`): how it learns the tree.
- **Asked twice, read once.** A path asked for twice is read once.
- **Inside, in development.** With `--dev`, a read must lie in the
  repository or the workspace, with `..` and symbolic links resolved
  first. Its reads are logged and shown, so nothing outside (a key, a
  private file) is brought in. Closed by default, as in the Machine note
  (2.9).
- **Dropped at a rollover.** Reads still waiting when a rollover starts
  are dropped, with a note: the summary carries what they were for, and
  stale whole files would refill the new context at once.

**Development** (`--dev REPO`, `docs/dev.md`). Its notes are checked against the code
(`verify.md`): a name the repository does not hold marks the note
`[unverified: ...]` and the stream is told; `[unnote: TEXT]` retracts. The persona gains a
paragraph after the person's instructions and the frame's mechanics: the
stream develops REPO (the program it runs in) as a peer with Claude.
- The size of its memory in the persona is the context's (`with_ctx`
  fills `{ctx}` with the cells in thousands at start and after a new
  persona): it was a fixed 32 thousand, false at any other `-c`.
- `[read: PATH]` resolves PATH in REPO, and in the workspace when only
  the workspace holds it (`dev_path`): its own records (`reflect.log`,
  `notes.md`, `preferences.md`) are read by their bare names. The persona
  says so.
- `[prefer: ...]` lines are its preferences: kept in `preferences.md`,
  announced as `prefers: ...`, and shown back with its notes. Claude
  follows them within the person's instructions.
- What is said can name its speaker (`Command::SayAs`): `« [time] Claude:
  ...`.

Its notes and preferences, and its last rollover summary (kept in
`summary.md`), are shown to it in the opening, so a restarted stream
resumes from what it kept, as after a rollover (not in a task).

**The mind** (`--mind`, `mind.md`): after every decode that asked for a
token, the residual of that token at the chosen blocks is read through
the Jacobian lens, synchronously, and sent out as `Event::Mind`; the
readout's time is `mind_ms` in the status.

**The hold and the playout.** Every piece of the stream (a token's
text, a line from outside, a mark) goes into a hold stamped with the
real time it came to exist. It goes out at the end of the cycle unless a
check holds it (`release`): a check holds its token and everything
after it, which can then be taken back without anyone having seen it. A
token's side effects happen when it goes out, not when it is placed:
- its text in `stream.log` and `chain.log`;
- its kind;
- the lines it completes (notes, reads);
- the leak count.

The events then pass through the playout (`playout.md`, `--horizon
SECS`, `Config::horizon_us`). That is the display's own clock on its
own thread, which shows the text about `horizon` behind its placement
at an even pace, absorbing the placement's jumps. The default is 0 (as
placed), or 1 s with `--reflect`. Pausing flushes it, and the end of a
run drains it before the model goes back.

**Checks** (`--reflect`, `reflect.md`). The loop starts after the
sampler chooses a live token and before that token is decoded:

1. The token's signals are read from the reading at the position
   before it.
2. If a trigger fires with nothing else beside the live sequence, two
   sequences are copied from the live one: the snapshot S (the state
   before the token, never decoded unless the token changes) and the
   deliberation D. A reading or a chase counts as beside it in the very
   cycle that feeds it: each is back in place before the live token
   advances (a check that started while one was out of place took the
   last two sequences, and the reading's composition then found none:
   "no free sequence for the composition" stopped the service on
   2026-10-01).
3. The token is placed and held as usual. A check holds that piece, and
   every piece after it, until the check ends.
4. The next cycle decodes D's first question token alone. D shares the
   live recurrent state until it writes its own, so this costs one
   cycle without a live token, about one token's time.
5. Every following cycle decodes the live token and D's lane in one
   batch, the live lane first, so the captured row the mind reads is
   the first one (`gate --reflect` proves this). D's lane is the rest
   of the question as one chunk, ending on `Decision:`. D's row there is
   read as the choice between keep and write (summed probabilities, not
   a sample, `reflect.md`). A keep ends the check. A write feeds
   ` write:`, and the word follows one token a cycle, greedy and outside
   the live sampler's history, to its first break or 8 tokens.
6. Then the check ends:
   - **Kept**: D and S are dropped, and the hold goes on.
   - **Changed**: the live sequence rewinds onto S. The held pieces
     from the token on are dropped unseen and the history is cut at the
     token. The placement state is restored (speaking, the thinking
     count and budget, the circling window, a task's end). The old live
     sequence and D are dropped, and S becomes live. The first token of
     the answer's word comes next; its other tokens are forced (accepted
     into the sampler's history, never sampled). The sampler's chain is
     rebuilt from the live history as it now stands, its seed advanced
     per change.

While a check is in flight:
- Nothing from outside goes into the live sequence. Said things,
  documents, reads, the clock line, nudges and rollovers wait for it,
  about half a second.
- A task's end waits for it, with D running alone and the live sequence
  stopped.
- Pausing or stopping abandons it.

When no check is in flight, nothing of this runs. During one, each
cycle carries the live token beside D's lane, plus the one solo cycle.
Every check is an `Event::Reflect` (an episode: `reflect.md`), a line in
`reflect.log`, and counts in the status (`checks`, `changes`,
`unparsed`, `checking`).

**Its code is left alone**: inside a ``` block of its own text (tracked
across tokens, restored by a rewind) no check fires and no circling
nudge comes and the clock line waits until the block closes (in the
dev session it cut two of the stream's code blocks in half), since code repeats by nature.

**Logs rotate** (`rotlog.md`): `stream.log`, `chain.log`, `mind.log` and
`reflect.log` each move to `NAME.1` past 64 MiB, the dev stream's own
audit finding.

Rates: exponential averages over recent cycles (`stream_tps` over cycles
that carried a live token, `side_tps` over those that read or caught up,
`cycle_ms` over all), in `Event::Status` every `status_every` cycles.
Commands: `Say`, `Feed`, `Pause` (the engine waits for the next command),
`Resume`, `Chunk`, `Temp` (the temperature alone; the rest of the
sampling stays), `Quit`.

## What it is, its objective, its tools, its restarts

- **The self-model.** After the persona, the opening and every rollover
  base carry "what this mind is" (`about`), from the run's own facts: the
  model's file, its blocks on the GPU and where the rest are (the cards
  when `GGML_BACKEND_PATH` names `ggml_phi`), its memory (the context in
  thousands), what it perceives (its context: its text, what it is told
  and handed with the time it arrived, what its tools return; no screen,
  no sound; it does not claim what it has not seen), how it goes on (the
  summary at a rollover, notes and preferences on disk, a restart resumes
  from the summary and is told what changed), its tools, and its
  objective. The preamble no longer says it has no tools, nor the
  mechanics that it never speaks of a system: it knows what it is and
  does not dwell on it. None of it in a task (a measurement's text does
  not move).
- **The objective, and output idle without one** (on by default;
  `--no-objective-gate` turns it off; never in a task). Kept in
  `objective.md` in the workspace, set by `objective TEXT` on the socket
  (`phi-stream objective TEXT`, `/objective TEXT` in the terminal; `-` or
  empty clears it). Without one it keeps thinking, and its output idles:
  the tokens carrying `»` (in chat, `</think>`) join the banned ones (the
  sampler rebuilt with its history, so its penalties stand), and its tool
  lines do nothing (it is told so, at most every 5 minutes). A change is
  told to it as a line from the system and sent as an `objective` line.
  After the Machine of Person of Interest, which acts only when asked
  (the note "The Machine learning from Finch" in the person's vault).
- **Its terminal** (`term.md`): `[run: COMMAND]`; the output comes back as
  a document, `the command ... ended (exit 0, 12 ms); its output`.
- **Quit writes the summary first** (the service): `quit` asks for the
  summary at the next point with nothing in flight, keeps it, and stops
  (by two minutes at the latest; a second `quit` stops at once). Every
  summary is also kept by its time in `summaries/`, so none is lost to
  the next.
- **What changed** (in development): the commit it runs is kept in
  `last_run`; at a start on another commit it is told the commits between,
  newest first, at most 30.

## The second chain (`--second-chain`, on in `dev`)

Reflection at every token, not on a command: beside the live token runs
a second chain (`Chain`), which reflects on each line the journal ends.
- At a line's end (the live token placed holds a newline), when nothing
  else is in flight (no check, reading, chase or summary, not inside a
  code fence), at most once a second, and with three sequences free (two
  stay for a check or a reading), a sequence is copied from the live one:
  the whole context, nothing read again.
- It is opened with a marker, not a question: the line's J-space words
  (each word's lens probability summed over the line's tokens and blocks,
  the six likeliest), `« [beside the journal; on its mind in the line
  above: ...]`, then its first words in its own voice, `On reflection,`
  (as a summary begins "What I was working on: "): without them, on the
  live service, the copy went on with the journal's structure and echoed
  the marker or a « line back into the journal. Its tokens are the
  likeliest that the live stream may write (no control token, no « or »
  mark, no end of text), from the 16 likeliest; a reflection that only
  repeats the frame, or says nothing, is dropped; so is one whose opening
  (60 characters) repeats one of the last 8 inserted. Its tokens are
  sampled (temperature 0.8 over those 16, its own random state; the live
  sampler's penalty history is the live chain's): greedy, on the live
  service, every fork of nearly the same context wrote the same reflection,
  five times in 30 s, each one making the next likelier. `chain off`
  also drops a reflection waiting for the journal. The marker's first
  token is decoded alone (the copy
  shares the live sequence's recurrent state until it writes its own, as
  a check's deliberation does), the rest and then each of its own tokens
  (greedy) in the live token's batch, until a newline after some text,
  64 tokens, or an end of text. It pauses while a check, reading or
  chase runs (their cycles are their own) and resumes after.
- Its reflection joins the journal at a later line's end, when nothing
  is in flight, as `« [HH:MM:SS] [beside the journal: ...]`: the J-space
  generalizations come back into the reasoning chain's own context. A
  rollover (the live sequence replaced) or a word written over by a check
  drops a reflection in flight.
- It runs whether or not there is an objective: it is reasoning, not
  output. Its text goes to the terminals as `delib` lines (start: the
  words it was given; piece: each token; end: what became of it), shown
  in DELIBERATION; the service replays the last 400 to a new tail.
- `chain on|off` on the socket turns it on or off live, so its cost is
  measured interleaved on one service (`docs/results`).

## What each thing weighs on the main chain

- **A reflection** that joins the journal is weighed when it does: a copy
  of the live sequence decodes the pending token alone (the journal
  without the reflection), the live one decodes the reflection, and the
  divergence of the first next-token distribution from the second is its
  weight, in nats (`weigh`), with the likeliest next token of each. It is
  sent as a note (LOG) and as the end of its `delib` lines (DELIBERATION).
  One single-token decode per reflection.
- **A check's choice** (`reflect.md`): the deliberation's likeliest token
  at `Decision:` was a quote, 37 to 56 percent, on the live service at
  200K (it answers `"keep"`), and keep and write together under a fifth:
  30 percent of the checks went unread. The format token it wants first,
  a newline or a quote (whichever is likelier, when it outweighs keep and
  write), is fed once, and the choice read after it.
- A reflection repeats a recent one when their words overlap (Jaccard) by
  half or more: rewordings of one reflection were inserted again and again
  under the earlier test of their first 60 characters.

## Where and when (the opening and every rollover)

Before the self-model, the context opens with where and when it runs
(`situation.md`): the date and time to the microsecond with its zone and
UTC offset, the computer as read from the system (name, system, kernel,
processor, memory, GPUs, the Xeon Phi cards, uptime), the model as loaded
(its blocks on the GPU, where the rest are, the context's cells), its
workspace and repository, and who is present. All of it is read when the
text is written, so a rollover carries the time of the rollover. A
reflection's weight is measured against a placebo: a copy given the same
frame with nothing in it, so the weight is what the reflection says, not
that a line came (the frame alone moved the next token, measured at first
as 7 to 14 nats).

## Tools first (2026-10-01)

- Its tools work with or without an objective: their results are real, so
  they ground it. Without an objective only speech waits (the `»` tokens
  stay banned). With every tool gated it only reasoned, and invented what
  it had done (a file it said it wrote; a file it said did not exist,
  after reading a URL-style path).
- Every tool use goes to the terminals as an `act` line (`act start`: the
  tool and what it was given; `act end`: whether it went through and what
  it came to, in short: lines and tokens read, a command's exit and first
  line, a note kept or marked unverified). OUTPUT shows them.
- No tool for 90 s: at a line's end, with nothing in flight, it is told
  to check one real thing with a tool and read what it returns (at most
  once per 90 s).
- The self-model says it works with its tools, not in its head: what a
  file says it reads, whether something works it runs, what it has done a
  tool's output shows, and what it has not checked it does not claim.
- `%XX` in a read's path is decoded (`Intel%20Phi%20Stream`).

## Loops (2026-10-01)

On the live service at 200K, temperature 0.5 and DRY 0.8, the stream fell
into writing "```" on every line: 40 to 89 percent of its 8-token
sequences repeated within 150 s windows (`loopiness`, from chain.log).
Two reasons: DRY took the newline as a sequence breaker (llama.cpp's
usual set), so a repeated line was never a repeat to it, and each fence
flipped the code state, inside which the circling nudge and checks stand
down. Now DRY breaks only at a colon, a quote and an asterisk, and a line
written 3 times running has its first token held back 30 s (added to the
banned tokens, then released), the fence state is forgotten when that
line is a fence, and the stream is told.

A summary is asked for with its real reason: the context nearly full
(with how many of its cells are used), a restart (`quit`), or a new
persona. Every quit had asked "your memory is nearly full" at a few
thousand of 205 thousand cells. `--rollover-tokens` sets the rollover
point in cells instead of a share (`dev`: 150000).

## Stability: refused summaries, live knobs (2026-10-01 evening)

The fixes for the loops compounded: each restart resumed from a summary a
degenerating stream had written (one held 26 fragments of the person's
delimiters), and the breaker's holding back cascaded. Now:
- a summary is refused when it is degenerate (`degenerate`: the
  delimiters, a third of its lines repeating earlier ones, or two fifths
  of its word 4-grams repeating); the last good one stays, and the refused
  one is kept apart in `summaries/` as `...-refused.md`;
- tokens carrying `•` are never sampled in the journal (the person's
  response delimiters belong to another harness);
- the breaker is off by default, `breaker on|off` and `nudges on|off`
  turn the harness's own interventions on and off live, and `set KEY
  VALUE` changes temp, top-k, top-p, min-p, dry or repeat-penalty live
  (the sampler rebuilt with its history), so a change is measured on one
  service without a restart.

## The agent frame between turns, and the channel to Claude (2026-10-02)

On the live service the agent frame looped: the stream noted one proposal
to Claude 131 times in 45 minutes. Each note was checked against the
repository, which did not hold the files it had just written to its working
copy, so each was marked unverified and the same was put inside its next
turn's thinking; a clock line put mid-turn cut a tool call in two; and its
thinking came to hold only imitations of those lines, with times it made up.
The quit's summary, asked inside a turn, came back as the template's marks
and two tool calls. Now:
- nothing is put inside an agent turn: `tell` holds a line from the system
  for the next user turn, and the queue (what was said, handed over, or
  asked by Claude) is taken there too (`take_waiting`), each within the
  room the context has (`read_room`; one too big is refused with its size);
  the clock and the tool reminder are not put in at all (every user turn
  carries the time and the objective);
- a turn that runs past `AGENT_TURN_MAX_US` (180 s, at a line start; 15 s
  when quitting)
  while something waits for it is closed, its unfinished calls not run
  (`agent_stalled`);
- a summary is due at the end of the turn it is in (`summary_due`), and
  asked in a user turn of its own with the thinking closed
  (`open_summary`, `agent::summary_turn`); a summary holding the chat
  template's marks is refused (`degenerate`);
- a note already kept is not kept again, notes are checked against the
  working copy (`repo_view`, `verify::Repo::load_over`), and in the agent
  frame the check comes back in the tool's result only;
- the queue's pop checks the room for whatever brought the text (the
  stream's own proposal: a feed past `direct_max` went to a reading with no
  check);
- `tell_claude` (`send_claude`) writes `to-claude.md` and sends a `claude`
  line (a message whose words are 60 percent or more in its last one, and
  that answers nothing, is not sent: each review was followed by a
  "final" one restating it, at 0.64 and 0.71, while new messages held 0.09
  to 0.29 of the last one's words); an answer's `re` must name a message Claude sent
  (`asked`): an id Claude never sent goes as a message of its own, under
  the repeat check, and it is told so (it named c3 while none had been
  asked, which also passed the check); and each is answered once (`answered`: it
  kept naming c1 after m13 had answered it, in messages of its own); `Command::Ask` (from `phi-stream ask` or the MCP `ask` tool) is a
  message from Claude with an id, which it answers with `re`.

## The guide lane (`--guide`, shadow, 2026-10-02)

Every thinking token reasoned against: a lane that holds the live
sequence with the second chain's last reflection placed in it as an aside
in its own voice (`\n(On reflection, ...)\n`), fed every live token after
it in the live token's batch (`Guide`, `guide_take`, `guide_fed`). At each
thinking token its row and the live row are read side by side
(`guide_measure`, `kl_and_tops`): the KL of the guided next-token
distribution from the live one, in nats, and whether the likeliest token
differs. One line each in `guide.log` in the workspace; every 128 thinking
tokens a report (a note, and a line in DELIBERATION).

- Shadow by default: nothing is chosen from it, and the live text never
  holds the aside (nothing is put inside a turn). `set guide-mix G` (live,
  0 to 2, 0 by default) mixes it into the choice of each thinking token:
  the live logits moved toward the guided ones by G (`l + G(g - l)`; at 1,
  as if the aside had been written in), sampled through the same chain
  (`Llm::sample_logits`: its bans, penalties and random state, accepted
  once). Top-down on every token: the reflection weighs on each choice
  while the text stays the stream's own. Measured before it stays on.
- Its own sequence, the last of five (`--guide` implies `--n-seq 5`; 62.8
  MiB of recurrent state), so checks, the chain and readings keep theirs.
- A new reflection forks it again from the live sequence (a hybrid
  sequence cannot be cut in part), the aside's first token decoded alone
  (the copy shares the live recurrent state until it writes its own). It
  is dropped when the live sequence is replaced (a reading's swap, a
  rollover) or cut (a check's write), and after 4096 live tokens.
- Paused while a check, reading or chase has the cycle; it catches up in
  the room a batch leaves, with no row until it holds the pending token.
- Inside thinking only: not in a tool call, not in code.
- `guide on|off` turns it live (when started with `--guide`).
- **Its aside's source** (`GuideSrc`, `guide chain|lens|placebo`, live;
  2026-10-02, the person: weave the J-lens into the reasoning):
  - `chain` (as before): the second chain's reflection;
  - `lens`: at the end of a thinking line, the J-lens words strong on its
    mind over the line that the line does not say (`mind::unsaid`, the
    terminal's row's own rule: its strongest at `LENS_ASIDE_MIN`, 10
    percent, and the others of at least half its weight), as
    `(on my mind: again, timestamp)` in its own voice (`lens_aside`; the
    line's text from `history`, its words from its readings counted per
    block, `lens_sums`, `lens_readings`);
  - `placebo`: at the same moments, the same frame holding as many of the
    line's lens words it does say (`mind::said`): any aside moves the next
    token (the reflection's frame alone moved it 7 to 14 nats), so lens
    against placebo is what says whether its J-space content does.
  - A lens or placebo aside at most every `LENS_EVERY_US` (10 s) and never
    the same words twice running: each fork copies 62.8 MiB of recurrent
    state, and 22.7 percent of thinking lines pass the bar. A change of
    source drops the lane and starts the report's counts afresh, so no
    window mixes two; each `guide.log` line and `guide` line names its
    source (`src=`), the report too.
  - Decided before measuring: over interleaved windows of its active work,
    1500 thinking tokens or more measured per arm, lens against placebo.
    If lens moves a token by at least 1.5 times placebo's mean KL and
    changes the likeliest one at least 3 points more often, its J-space
    content carries something the frame does not, and a mixing test may
    follow (`guide-mix` at most 0.3, stopped when the thinking's repeated
    8-grams pass twice the window before). Otherwise the lens stays in
    the terminal (`tui.md`) and is not mixed. Each arm's rate is recorded:
    a lens arm 10 percent slower than the chain arm counts against it.
- With `experts on` (`capture.md`), each thinking token also compares
  the experts the guided token and the live one were routed to at the
  mind's blocks: the share they hold in common, block by block
  (`experts_shared`), in `guide.log` and the report. Every expert of every
  thinking token, reasoned against the reflection's.

## Rest (`wait`, 2026-10-02)

A finished objective had no end in the agent frame: each turn without a
call was answered "go on, act with a tool", the circling nudge said "move
on", and the stream went round saying "Done" (its thinking's 8-grams
repeated at 70 percent over five minutes). Its `wait` tool rests it
(`Rest`): the turn's results are held, nothing is decoded, and the run
loop looks every 200 ms (`rest_look`) for what wakes it: something waiting
for it (a message, a line from the system, an objective), a new commit in
the repository (`head_of`, read from `.git` every 2 s, against the head it
was last told, `head_told`: a rest that took the head at its start missed
564ad0b, made during a turn 5 s after its `git log`; a commit made during a
turn is told at its next user turn instead, `take_waiting`; kept in the
workspace, `head-told`, so a commit made while the service was down is told
too), a quit (straight
to the summary), or the minutes it gave (15 by default, 60 at most). Its
next turn opens with the rested turn's results and what woke it. The
status says `resting`, with rates of 0 (nothing is decoded). The second chain's reflections wait apart (`asides`, the last
two): they ride with the next turn but neither stop a rest nor wake one (at
first each turn left one waiting, and every rest was refused); a rest
refused because something came says so. The continue line and the circling nudge name
`wait` as the way to stop when the objective is met.

## Reads of the agent frame, and a long prefill (2026-10-02)

A tool response goes into the live sequence as one prefill, a batch at a
time (`feed_live`), at about 220 tokens a second on the live service
(`mind.log`: every jump of 2000 tokens or more ran at 218 to 229 tokens a
second), with the run loop waiting on it. The stream read the whole of
`reflect.rs` (9311 tokens, 42 s) and a grep of 9732 tokens (45 s), and the
context went from 30k to 40k cells in one turn. Now:
- one `read` gives at most `READ_MAX_TOKENS` (4096, about 19 s): past that,
  the leading lines that fit (`lines_within`, counted line by line) and a
  header saying which lines are not shown and the `start` that reads them;
  a single line past it is refused with its size. The room check
  (`read_room`) stays the outer bound, and near a rollover the message
  names the room (`past`);
- a command's result too (`fit_output`): the grep above came back through
  `run`, whose byte cut (16 KiB, about 8k tokens of log text) let it
  through, and a turn may run four commands. Past the budget, its leading
  lines and a line saying how much was cut and to narrow the command;
- past one batch, the status says reading, how many of how many tokens,
  after each batch (`prefill`): it said speaking for those 42 s, on the
  terminal and the MCP screen.

What it is not: the doubt checks. The stream proposed turning them off
(m21: 95 percent kept, about 0.55 s each). They run beside the live token
in the same decode: the live rate in a check's window was 14.7 tokens a
second against 16.4 outside it (499 checks, 53k tokens, `reflect.log`
windows against `mind.log`), under 1 percent of the throughput; they stay.

## A token's text carries its position (2026-10-02)

A placed token is held with the position it is decoded at (`Piece::Token`
`pos`: `pos()` when it is placed, a check's replacement at the checked
position) and goes out with it (`out_at`, `Event::Text`'s fourth field;
other pieces carry none). Its reading (`Event::Mind`) has the same
position and comes first: the token is read when it is decoded, a cycle
after it is placed, and the hold releases it later. The terminal joins
the two by it to weave the J-lens into the reasoning (`tui.md`).
