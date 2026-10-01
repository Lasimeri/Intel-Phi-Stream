# serve.rs

`phi-stream serve`: the service that owns the model, so that the
terminal, one-shot commands, scripts and an agent all talk to the same
running mind and the 40-second load happens once.

- The engine (`engine.md`) runs on its thread; this side keeps the
  recent text (the last 400k characters) and the last status line,
  listens on a Unix socket (`client.md`: where it is), and serves every
  connection on a thread of its own: the `info` line first, then a reply
  (`ok` or `err`) to each command line.
- `tail` subscribes the connection: it gets the last 12k characters of
  text as `text` lines and the last status, then every `text`, `status`,
  `note`, `mind` and `reflect` line as it happens, written by a thread per subscriber, until the
  client goes away or the engine stops (`bye`).
- `say TEXT` and `feed PATH` queue what is said or handed over (the file
  is read here, in the service's own file system); `persona PATH` reads
  the file and hands the new persona to the engine, which rolls its
  context over onto it after a summary; `chunk N`, `temp T`, `pause`,
  `resume`, `status` (the last status line, and a fresh one asked for),
  `recent` (the replay without subscribing), `quit` (the engine stops;
  the service removes its socket and returns).
- A socket file nobody answers on is stale and replaced; one that
  answers means another service is running, and this one stops with that
  message. One process at a time holds the cards, so one service at a
  time is the rule anyway.

Nothing here touches the model: every command becomes an engine
`Command` through the channel, and every event is relayed as it comes.
