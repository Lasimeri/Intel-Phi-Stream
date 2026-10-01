# reflect.rs

The reflection loop: per token, before a token is shown, the stream may
take a second look at it. Where a trigger fires, a deliberation runs
beside the live token (never as a pause) and answers `keep` or
`write: X`; a change rewinds the live sequence onto a copy made before
the token and places X, before anything after the token went out. The
plan of record is the J-space note's section 7 and its balance section 8
(Obsidian, "Intel Phi Stream - J-space metacognition plan"). This file
holds the logic apart from the engine: the triggers, the controls, the
question, the answer's reading, and the episode record. The engine's
part is in `engine.md` (the check's lanes, the hold, the rewind).

## Triggers (`Reflector::signals`, `should_check`)

Read at every live token from the reading at the position before it
(`mind.md`: the band's blocks through the lens, and the final block as
it is, the model's own next-token distribution, `model_top`):

| trigger | fires when | on |
| --- | --- | --- |
| doubt | the model's own probability of the chosen token is below `doubt_p` (0.30) and the token is not among the top five words of any band block | word-like tokens only (`mind::is_wordlike`) |
| flag | the probability mass of error words (`FLAG_WORDS`: error, wrong, bug, wait, actually, ...) at the band's blocks, averaged over the blocks and smoothed over about `flag_tokens` (6) readings, rises above `flag_hi` (0.20) | any token that is not blank and not a control token |

A token whose probability is below the model's 64th is given the 64th's
(an upper bound); without the final block's reading the doubt trigger
never fires.

## Controls (section 8: neither runaway nor lock-up)

- **Smoothing**: the flag score is an exponential average; one lit
  reading is noise, a run of them is a signal. The doubt trigger is per
  token by design (the decision is about this token).
- **Hysteresis**: after the flag trigger fires it re-arms only once the
  smoothed score has fallen below `flag_lo` (0.08).
- **Habituation**: a word lit for more than `habituate_after` (24)
  readings in a row stops counting toward the flag score and is left out
  of the question's words, until it goes out for a reading.
- **Refractory periods**: no check within 2 s of the last one, nor
  within 10 s of a change (monotonic clock).
- **Budget**: at most 12 checks and 4 changes in any minute; when spent,
  the loop is read-only (readings go on) and a note says so once.
- One check at a time, and none while a reading, a chase, a summary or a
  rollover is in flight, or while a changed answer's tokens are still
  being placed.

## The question and the answer

`question` puts the following, in the frame's own style, at exactly the
position of the token in question:
- the real time to the microsecond;
- the token;
- up to `words` (8) words on the mind there: the band's, best first,
  each once, habituated ones left out.

It ends on a `Decision:` line:

```text
[at 14:02:03.123456, a check on the next word: you were about to write "i32" here; on your mind here: usize, index, length. Is "i32" right at this place? Keep it, or write the word to use instead.]
Decision:
```

The journal frame puts `« ` before the bracket.

**The choice is read, not sampled.** The deliberation's next-token
distribution after `Decision:` is summed over the one-token forms of
each word (`KEEP_FORMS`: ` keep`, ` Keep`, `keep`, ...; `WRITE_FORMS`
likewise). Keep's share of the two (`keep` in the episode) decides it,
and a keep ends the check right there. The episode also records the
share of the whole distribution the two classes hold (`fmt`), which
shows how far the model was answering the question at all.

A write feeds ` write:` to the deliberation, and the word follows
greedily, to its first break after something or `answer_tokens` (8)
tokens. `parse_answer` reads the word: the first word out of its quotes,
with a `]` at its end dropped only when unmatched inside it (`v[i]`
survives) and a sentence's `.`, `,`, `;` or `:` after a word dropped. An
empty word is `Unparsed`, kept and counted apart. `replacement_text`
keeps the chosen token's leading space.

**A first version asked for one free line**, "keep, or write: TOKEN, and
why". In the live journal the model answered "TOKEN, because ..." twelve
times out of twelve: it reasoned about the word ("\"determined\" is the
wrong word here") but copied the placeholder. Every answer was
unparsed, and every check ran to its 24-token cap, about 1.2 s
(`docs/results/2026-10-01-reflect.md`). Counting unparsed apart from
kept is what showed it. The choice costs one row of the deliberation
for a keep, so a check takes about 0.4 s, nearly all of it the
question's one cycle.

## Episodes

Every check ends as an `Episode`: when the token was chosen (real time,
microseconds), its position, why, the token and its probability, the
smoothed flag score, the words shown, the choice (`keep`, `fmt`), the
answer as written, the
outcome, what was placed instead, how long the check took, the live
tokens placed meanwhile, and on a rewind the positions taken back
(`back=A-B`: readings of those positions were shown and belong to text
that never went out). Outcomes: `kept`, `changed`, `same` (wrote the
token it was about to write), `unparsed`, `dry` (read-only: would have
changed), `abandoned` (the stream paused or stopped first). One line
each (`line`, `parse_line`; text escaped, spaces as `\s`):

```text
t=1790881234123456 pos=812 why=doubt outcome=changed p=0.2134 flag=0.0312 keep=0.1875 fmt=0.9312 ms=412.5 placed=14 back=812-826 chosen=\si32 to=\susize words=usize,index answer=write:\susize,
```

It goes to `reflect.log` in the workspace, to the socket as a `reflect`
line, to `tail` and `run` on stderr, and to the terminal (`tui.md`).

## Options

`--reflect` (needs `--mind`; `serve`, `run`, `code stream`) and
`--reflect-dry` (every deliberation runs, nothing changes: the third arm
of a measurement, which isolates what the extra lanes alone do to the
greedy text). With either, `--horizon` defaults to 1 s: the playout (`playout.md`)
shows the text a second behind its placement at an even pace, so a
check that ends within that second, kept or changed, never shows as a
stall. Whatever the horizon, a check holds its token and everything
after it until it ends, so a change is always taken back before anyone
saw it. The check is a result block in BF++'s sense (`R{...}K{...}` in
Lasimeri/bfpp, whose error register is saved before the block): the
state saved before the token, the deliberation's `write` the catch that
restores it and places another. The defaults above are the starting point of the
sweep section 8 asks for, not its result.

## Episodes on the code benchmark

Episodes recorded on MultiPL-E tasks (`code stream --reflect`, kept in
`results.jsonl`) are evaluation output under that dataset's licence,
whose clause 4 forbids using it as training data: they never enter a
fine-tune (`code.md`).

Tests: answers read (keep, write, unparsed, code with brackets and
punctuation), the question's content, doubt only on words and the
refractory period, the flag's smoothing and hysteresis, habituation
(score and question), the budget, the episode line's round trip.
