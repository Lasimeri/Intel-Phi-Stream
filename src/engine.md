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
