# agent.rs

The agent frame (`--frame agent`): the chat template with the model's own
tool calls, as its chat template (read from the GGUF's
`tokenizer.chat_template`) writes them.

- The system turn opens with the template's tools section, word for word:
  `# Tools`, the functions in a `<tools>` block, one JSON object each in the
  form the template's `tojson` gives (keys in order, `", "` and `": "`
  between them), then the template's own instructions for the call format.
  Four functions: `run` (a command in the sandboxed terminal, `term.md`),
  `read` (a file whole or by lines, or a directory), `write` (a whole file
  into the working copy or the workspace), `note`. The persona follows,
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
- Paths: relative to the repository, through its working copy (a file it
  wrote reads from there, a write goes there; the repository never
  changes), or in its workspace; nothing else.

Tests: calls parse as the template writes them (multi-line values,
several calls); broken blocks are counted and not run; the tools section
is the template's, and the responses turn its tool form.
