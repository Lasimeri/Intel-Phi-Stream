# remote.rs

The model served by another machine (`--remote http://HOST:PORT`): a
llama-server holds the live sequence in one of its slots and samples it;
this process keeps the token history, the vocabulary and the whole engine
(tools, terminal, the self-improvement loop, the service and its window).
Built for the GPU rack, whose llama.phi serves Qwen3.8 Flash Next on
`http://192.168.0.39:8001` (prompts on its four RTX 3080s, generation on
its CPU and four Xeon Phi cards; the rack runs no phi-stream, only the
server). Nothing here loads a model, a GPU backend or the cards.

## What the server does, what this side does

- **One sequence, one slot.** The engine's lanes (`llm.rs` `decode`) for
  sequence 0 are kept in order (`place`); every other sequence is refused,
  so readings beside the stream, forks, the checks, the second chain, the
  guide and the goal probe cannot run. `main.rs` turns them off at start
  (`remote_off`, said on stderr), and the engine has no free sequence for
  them (`Engine::new`: the free sequences come from `n_seq`, 1 here). A
  reading beside the stream has a remote form of its own: the server reads
  it on its prefill engine (the prefetch, below), and the engine splices
  it in once read.
- **Sampling on the server.** `sample_pumped` sends the whole sequence as a token
  array to `POST /completion` with `stream`, `return_tokens` and
  `cache_prompt`, pinned to `--remote-slot` (default 1, so a client that
  does not pin takes slot 0 and does not evict this cache), with the
  stream's sampling (`temperature`, `top_k`, `top_p`, `min_p`, the
  penalties, DRY with `llm.rs`'s breakers, a seed that changes per
  request) and every banned token (the frame's control tokens, the
  dash-carrying ones) as a `logit_bias` of `false`. A window of -1 (the
  whole context to `llm.rs`) is sent as the slot's context: llama-server
  refuses a negative `dry_penalty_last_n` (HTTP 400, seen 2026-10-07).
- **A stream goes on while the engine places exactly what it gave**
  (`Stream::continues`): the engine decodes the token it was handed, asks
  for the next, and gets the next event's token. Anything else placed (a
  forced token, a tool's results, a rollover) drops the connection, which
  stops the server's generation, and the next `sample_pumped` opens a new request
  from the whole sequence; the server reuses its slot's cache up to where
  the two differ. Measured on the rack (2026-10-07, Flash-Next Q6_K_XL,
  llama.phi b11462-d02556691): a prompt of 15966 tokens 31.6 s cold, the
  same plus 12 tokens 0.98 s; the stream itself 5.8 to 6.6 tokens a second
  (`phi-stream run --remote ... --frame chat`, 1925-token opening plus 120
  tokens in 27 s).
- **Ends of turns.** The server's stream carries the end-of-generation
  token it sampled (248046, `<|im_end|>`) before its `stop` event
  (`stop_type` `eos`), so the engine sees a turn end exactly as it does in
  process. A stop at the length limit opens a new request; a fresh stream
  that stops without a token is an error, never a loop.
- **Retries.** No connection, HTTP 5xx or 429: asked again every 5 s for
  up to 15 minutes (a server restarting loads its model in about five),
  said on stderr once a minute. HTTP 4xx: an error at once, with the
  server's message. A stream that opens and breaks before its first
  token is asked again the same way, every 5 s within the same 15
  minutes, then an error: asking again at once could go on forever, and a
  count of five breaks would have ended the service at the rack's model
  switch of 2026-10-08, when a restarting server broke more streams than
  that. Waits: the first token up to 30 minutes (a whole 131072-token
  context read at the rack's 350 tokens a second is six), then 5 minutes
  between tokens.
- **During a wait.** A held `sample` is not deaf: its pump
  (`sample_pumped`, driven by the engine's `sample_live`) takes the
  terminal's commands at every round and every `POLL` while bytes are
  awaited. A Quit cancels the stream with an `Aborted` error and the
  service ends there, without a summary (the model is busy reading, and
  the half-done step's pending token is spent, so nothing decodes on);
  other commands wait their turn in `cmd_pending`, ahead of the
  terminal's next. `status` never waits: the service answers it with the
  last line it has, which the engine's heartbeat keeps marked with how
  long the step has held it (`engine.md`).

## Reading beside the stream: the prefetch (2026-10-08)

The remote form of a reading beside the live sequence (`engine.md`,
splicing). llama.phi's decode server reads a prompt on its prefill
engine (the GPUs) without taking a slot, and the finished state lands in
its prompt cache, so the slot's stream keeps going meanwhile. The two
calls live here, and nowhere else in the program:

- `prefetch(tokens)`: `POST /phi/prefetch {"tokens": [...]}`, answered at
  once with `{"id": N}` (`PrefetchStart::Id`). HTTP 404 on the path is a
  server without it (`Unsupported`: llama.cpp's own server, or llama.phi
  before it had one). Any other refusal is an error with the server's
  words.
- `prefetch_state(id)`: `GET /phi/prefetch?id=N`. The state is read from
  a string field `state` (else `status`), or from a body that is itself a
  string. `waiting`, `queued`, `pending`, `running` and `reading` are
  `Waiting`. `ready` and `done` are `Ready`: both are taken as read until
  the server's final format says otherwise. `failed` and `error` are
  `Failed`, with its `error` or `message`. Anything else is `Unknown`,
  waited on as `Waiting` and named once by the engine. HTTP 5xx is an
  error (asked again at the next poll); another failure status (an id
  the server does not know) is `Failed`.
- Both wait at most `PREFETCH_ASK` (2 s), so a busy server never holds
  the engine's cycle for longer. The mapping of answers is two pure
  functions (`prefetch_started`, `prefetch_state`), tested on the bytes
  a server sends.

The splice then needs nothing new of `Remote`. The engine places the
composed sequence as one decode at position 0 (`place` replaces `hist`),
the open stream no longer `continues`, so it drops, and the next request
sends the whole sequence. A server whose cache holds the prefetched
prefix (the history up to the fork and the input) starts from that state
and reads only the rest. That needs llama.phi to load, for a pinned slot
that is not empty, a cached state sharing more of the new prompt than the
slot itself does. This is required, not an optimization. The slot shares
only the history up to the fork with the new prompt, and this model's
recurrent state cannot be cut back the tokens generated since (`engine.md`:
neither shifted nor cut). Without the load, the server goes back to a
checkpoint at or before the fork, or to zero, and reads from there, which
could be minutes deep in a context. Today's path never rolls a slot back
more than the few tokens a dropped stream sampled ahead.

## The vocabulary

Tokenizing goes to the server (`POST /tokenize`, `parse_special` as
asked, `add_special` off), so the ids are the server's own. Turning ids
back into text happens here, token by token, from the model's GGUF
metadata (`Vocab`): `tokenizer.ggml.tokens` and `token_type` read by a
small reader of GGUF's key-value section that stops before the tensor
table (so the head of a file is enough), each piece as llama.cpp's
`token_to_piece` gives it with lstrip 0 (llama-vocab.cpp: a control or
unknown token only with `special`, a user-defined one always, a normal
one through GPT-2's byte-to-character mapping reversed,
`unicode_utf8_to_byte`; an unused or byte token nothing), and the
end-of-generation set as llama.cpp finds it (the end-of-sequence and
end-of-turn ids and the tokens named `<|im_end|>`, `<|endoftext|>`,
`<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>` and the other templates'
end markers, each made a control token). Only byte-level BPE (`gpt2`)
vocabularies are read; anything else is refused at start.

The file lives at `~/.local/share/phi-stream/remote/STEM.vocab.gguf`, STEM
the server's model file name (`/props` `model_path`), or `--remote-vocab`.
[`scripts/remote-vocab.sh`](../scripts/remote-vocab.md) copies the head
of the model file from the server's machine (Flash Next's first file is
its metadata alone, 10.9 MB). At every start `check_vocab` compares it
with the server: the count against `/v1/models` `meta.n_vocab` when given,
and every token of a fixed text (control tokens, the tool call's tags,
several scripts, an emoji split into byte tokens, runs of spaces, code)
against `/tokenize` with its pieces; any difference stops the start with
the command that fetches the right file.

## Limits

- A long prompt blocks the engine's loop while the server reads it (the
  service answers no command and writes no status until the first token):
  the opening, a restart from a summary, a rollover. Tool results are a
  few thousand tokens at most (`READ_MAX_TOKENS`, `fit_output`).
- llama.phi's prefill server has one slot: two clients of the rack with
  long prompts make it read each one's whole prompt again.
- No logits here: `llm.rs` `logits`, `greedy` and `sample_logits` return
  an error, so `gate`, `code`, `eval`, `probe` and `lens` need a model in
  this process.

## Tests

`cargo test remote` (no network: the improvement sandbox has none): the
vocabulary of a small hand-made GGUF head (pieces, control and
user-defined tokens, the end set, a skipped merges array), a cut head as
an error, addresses, the chunked event stream read through a source that
gives seven bytes at a time with timeouts between them (tokens queued,
`continues` after each, the stop), a sized answer, a place past the
server's context refused before the sequence is touched, and the
prefetch's answers read off such a connection (an id given as a number
or as a string, llama-server's 404 as `Unsupported`, a refusal, an
answer with no id, each state including an unknown one, an unknown id,
and a 5xx).
