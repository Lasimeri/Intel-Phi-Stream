# agent.rs

The agent frame (`--frame agent`): the chat template with the model's own
tool calls, as its chat template (read from the GGUF's
`tokenizer.chat_template`) writes them.

- The system turn opens with the template's tools section, word for word:
  `# Tools`, the functions in a `<tools>` block, one JSON object each in the
  form the template's `tojson` gives (keys in order, `", "` and `": "`
  between them), then the template's own instructions for the call format.
  Five functions: `run` (a command in the sandboxed terminal, `term.md`),
  `read` (a file whole or by lines, or a directory), `write` (a whole file
  into the working copy or the workspace), `note` (its own memory, kept
  once), `tell_claude` (a message to Claude, `re` naming the message of
  Claude's it answers). The persona follows,
  with the agent's mechanics in place of the chat's.
- Each turn opens `<|im_start|>assistant\n<think>\n`: it reasons first,
  then closes its thoughts and acts. A call is
  `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
  (`parse_calls`; a block that does not parse is counted and told).
- When its turn ends, the calls run (`engine.rs`, `agent_turn_end`): read,
  write and note at once, a command in the terminal. Nothing is decoded
  while a command runs; its result is waited for, so it never invents one.
  The results go back in one user turn, one `<tool_response>` each
  (`responses_turn`), and its next turn opens. A turn without a call is
  answered with the time and its objective (`continue_turn`).
- Nothing is put inside a turn. Lines from the system, what was said or
  handed over, and Claude's messages wait for the next user turn, after
  the tool responses in a user turn of their own (`extra`): put inside
  its thinking, they cut its tool calls in two, and it took to writing
  such lines itself with times it made up (2026-10-02, the live service).
  A turn that runs past 90 s while something waits for it is closed
  (`engine.rs`, `agent_stalled`).
- A summary (a rollover, a restart) is asked in a user turn of its own at
  the end of the turn it is in, its answer opened with no thinking and
  the summary's first words (`summary_turn`); asked inside a turn, it
  called tools, and the summary kept the template's marks.
- Paths: relative to the repository, through its working copy (a file it
  wrote reads from there, a write goes there; the repository never
  changes), or in its workspace; nothing else.

Tests: calls parse as the template writes them (multi-line values,
several calls); broken blocks are counted and not run; the tools section
is the template's, and the responses turn its tool form; what waited
comes in a user turn of its own after the responses, or before the
objective in a continuing turn; the summary turn opens on its first
words with the thinking closed.
