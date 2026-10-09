# agent.rs

The agent frame (`--frame agent`): the chat template with the model's own
tool calls, as its chat template (read from the GGUF's
`tokenizer.chat_template`) writes them.

- The system turn opens with the template's tools section, word for word:
  `# Tools`, the functions in a `<tools>` block, one JSON object each in the
  form the template's `tojson` gives (keys in order, `", "` and `": "`
  between them), then the template's own instructions for the call format.
  Qwen3.8 Flash-Next's template (the rack's, read from llama-server's
  `/props` on 2026-10-09) puts a line before `# Tools` for a reasoning
  effort of xhigh (its default) or low, and none for medium; the harness
  writes none, so its system turn is the template's at medium.
- One table (`Spec`: name, description, parameters with their JSON
  types, the required ones) gives both the tools section and `check`, so
  what the model is told and what is checked cannot drift apart.
  Seven functions: `run` (a command in the sandboxed terminal, `term.md`; its
  output at most about 4096 tokens, past that the leading lines and what
  was cut, `engine.rs` `fit_output`; it
  starts in the repository, so no cd (a command that does cd there is
  told so once, `cds_into`: 169 of 243 commands of the live service
  began with it), and `/tmp` is kept between commands; an exit whose
  meaning is known is explained beside it (`term_hint`: grep's no-match,
  a git write into the read-only `.git` with what to use instead, a
  crash or kill by its signal), since bare exits 1 and 128 were read as
  failures and retried; a
  command stopped at its 60 s limit comes back as not finished, its result
  unknown, and the description says a Rust build does not fit there: three
  cargo checks stopped at the limit had become "compilation passes" in a
  review),
  `read` (a file whole or by lines, or a directory; at most about 4096
  tokens a read: past that, the leading lines and where the rest begins,
  `engine.rs` `READ_MAX_TOKENS`), `edit` (one exact
  replacement, `old` occurring once; for every change to an existing file:
  without it, it rewrote a whole file for each fix), `write` (a new file, or
  one replaced entirely), `note` (its own memory, kept once), `wait` (rest
  until something new comes: a message from Claude, a new commit, a new
  objective, one of its terminal commands ending, a candidate's outcome,
  a quit, or the minutes it gave; nothing is decoded meanwhile, while
  `diag.md` and `status.txt` stay fresh (a rest had left them stale, and
  the watchdog restarted every rest as a wedge)
  and its next turn opens with the rested turn's results and what woke it;
  without it a finished objective was answered every turn with "go on, act
  with a tool", and its thinking went round saying "Done", 70 percent of
  its 8-grams repeated), `tell_claude` (a message to Claude, `re` naming
  the message of Claude's it answers). With `--improve`, `propose` too (a title
  and a why): its working copy's change built and tested in a sandbox,
  the outcome at a later turn and in `improve.log` (`improve.md`).
  And with it `build` (a trial of the same build, never sent, to fix
  errors before proposing), `diff` (its change against the repository's
  head), `revert` (one file back to the repository's version, which it
  cannot do inside its overlay) and `report` (its status, objective, the
  goal probe over ten minutes, what is building, the last of
  `improve.log`).
  The persona then gains a paragraph naming the loop and its log
  (`IMPROVE_AGENT`); it asks it to keep working on its next change while
  a candidate builds, waits for Claude's review or is measured (it had
  rested through all three, and the person saw the loop stall,
  2026-10-03).
  In development the persona's text names these tools
  (`engine::agent_persona`), not the chat frame's bracketed lines
  (`[read: PATH]`, `[prefer: ...]`), which it still taught beside them. The persona follows,
  with the agent's mechanics in place of the chat's.
- A turn's first token, sampled right after the given text, is shown and
  counted like every other (`engine.rs` `emit_pending`, after the text and
  with the turn's thinking or speech set): it had been placed but never
  shown, so every turn's first word was missing from the screen and
  `stream.log` ("'m noticing", " me organize"), and a `!` from the progress
  events had stood in its place from 10-08 to 10-09. The same holds after
  a swap, a resume and a splice.
- Each turn opens `<|im_start|>assistant\n<think>\n`: it reasons first,
  then closes its thoughts and acts. A call is
  `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
  (`parse_blocks`: each block in the order written, as its call or why it
  does not parse). Calls
  written inside its thinking, in a turn that never closed it, run too, and
  it is told to close its thoughts first, in the user turn after the
  responses (3 percent of its calls were
  dropped so on the live service, without a word). The mechanics ask for
  calls that do not depend on one another in one turn (several reads,
  searches or commands at once): on the live service it made one call a
  turn, each turn about 200 thinking tokens and 40 to 60 s (2026-10-03).
- When its turn ends, each call is checked against its tool's declaration
  (`check`): an unknown function, a parameter the tool does not have, a
  required one missing, one given twice, or an integer that is not one.
  A refused call is not run; its answer says what is wrong and the tool's
  parameters (`read: it has no parameter "start_line"; read takes path
  (required), start, end`). Before this a `read` with start_line and
  end_line read the whole file without a word (8 calls of 70 between
  2026-10-07 and 10-09, `stream.log`), and an integer that was no number
  fell back to a default. An unknown parameter left empty says nothing
  and is let pass (a read with an empty end_path beside start and end
  had read just that range).
- Then the calls run (`engine.rs`, `agent_turn_end`): read,
  write and note at once, a command in the terminal. Nothing is decoded
  while a command runs; its result is waited for, so it never invents one.
  The results go back in one user turn, exactly one `<tool_response>` a
  block in the order written, a block that did not parse or was refused
  answered in its place with why (`responses_turn`; each trimmed at both
  ends, as the template trims a message), and its next turn opens. The
  harness's own lines (calls written inside thinking) come in the user
  turn after the responses, never as responses of their own: they had
  made more responses than calls (4 turns, 10-07 to 10-09). A turn without a call is
  answered with the time and its objective (`continue_turn`).
- Every call is a JSON line of `calls.log` in the workspace (`CallRec`:
  name, parameter names, why it was refused, inside thinking or not, the
  result's characters, the milliseconds until the result), and the
  counts since the start are in `diag.md` (`tool calls`). `stream.log`
  shows no control token the model wrote (`</think>`, `<|im_end|>`), so
  a turn's end cannot be read from it; the ledger says how each call went.
- Nothing is put inside a turn. Lines from the system, what was said or
  handed over, and Claude's messages wait for the next user turn, after
  the tool responses in a user turn of their own (`extra`): put inside
  its thinking, they cut its tool calls in two, and it took to writing
  such lines itself with times it made up (2026-10-02, the live service).
  A turn that runs past 180 s while something waits for it is closed at
  its next line start (anywhere past 360 s)
  (`engine.rs`, `agent_stalled`), but never while it holds a call, open or
  written (the pending token counted), up to 180 s more
  (`AGENT_CALL_GRACE_US`): closing never runs a turn's calls, so a turn
  cut inside one left a call without its end in the context, one cut
  between two calls dropped the first unanswered, and notes written as a
  quit came (15 s then 30 s) were lost so.
  - **The one exception: a splice** (`inject on`, the model remote,
    2026-10-08; `engine.md`). An input past `--direct-max` tokens is
    read on the server's prefill engine while the turn goes on. Once
    read, it goes in where the stream was when it arrived, as a user turn
    of its own (`<|im_end|>`, the user turn, then the assistant's turn and
    thinking opened again, the form `responses_turn` gives it), with the
    thoughts written since carried after it. Guards, against the two
    failures above:
    - The fork point is a line start in the thinking, outside a code
      block, in a turn that has not closed its thinking or begun a tool
      call. Elsewhere, the input waits for the turn's end.
    - The join waits while a tool call is open.
    - The marks are control tokens, and the sampler never draws
      `<|im_start|>` in this frame, so the model cannot write such a turn
      itself. Bracketed text was what it imitated.
    - Its calls are read from the new turn on (`turn_start` after the
      input).

    A short input, or a splice given up (its turn ended first, the server
    failed it or lacks the call), keeps the path above.
- A summary (a rollover, a restart) is asked in a user turn of its own at
  the end of the turn it is in, its answer opened with no thinking and
  the summary's first words (`summary_turn`); asked inside a turn, it
  called tools, and the summary kept the template's marks. The turn's
  `note` and `tell_claude` calls still go first (its memory and its
  messages); nothing else of it runs, and every call of it is in the
  ledger (the rest with "the summary came first").
- Paths: relative to the repository, through its working copy (a file it
  wrote reads from there, a write goes there; the repository never
  changes), or in its workspace; `/tmp` is its workspace's `tmp/` as in
  its terminal (a file a command wrote to /tmp had come back "outside"
  to `read`); nothing else. A file missing under one of the two roots
  that the other holds (the same path, or the same name at its top) is
  named in the error (`elsewhere`: improve.log and lessons.md were asked
  for in the repository, tools/loopiness.c in the workspace).

Tests: calls parse as the template writes them (multi-line values,
several calls); broken blocks are counted and not run, each keeping its
place with why it failed; calls are checked against their declarations
(an unknown parameter, a non-integer, one given twice, one missing, a
tool of the loop without it); the tools section
is the template's, and the responses turn its tool form; what waited
comes in a user turn of its own after the responses, or before the
objective in a continuing turn; the summary turn opens on its first
words with the thinking closed.
