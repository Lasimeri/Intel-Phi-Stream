# client.rs

The wire between the service and its clients, and the client side of it.

**The socket**: `PHI_STREAM_SOCKET`, else `$XDG_RUNTIME_DIR/phi-stream.sock`,
else `/tmp/phi-stream-<uid>.sock` (`--socket PATH` on any command
overrides).

**Lines**, UTF-8, one per message; a text's backslashes, newlines and
returns are escaped (`\\`, `\n`, `\r`) so that one piece is one line.

From the client: `say TEXT`, `say-as NAME TEXT` (named: the stream
hears `NAME: TEXT`, `docs/dev.md`), `feed PATH`, `persona PATH`, `chunk N`,
`temp T`, `keep-at P` (the checks' keep threshold, live), `pause`, `resume`, `status`, `recent`, `tail`, `quit`.

From the service: `info model=... gpu_blocks=N n_blocks=N gpu_gib=X
host_gib=X n_ctx=N frame=journal|chat workspace=PATH` once on connect;
`ok MESSAGE` or `err MESSAGE` for each command; to a subscriber `text
think|speak|given t=MICROSECONDS TEXT` (the real time the piece exists
at, `clock.md`), `status mode=... stream=... beside=... cycle=...
pos=... ctx=... queued=... chunk=... rollovers=... notes=... frame=... leaks=... mind_ms=... t=MICROSECONDS reads_quiet=N checks=N changes=N unparsed=N checking=0|1`
(mode is `thinking`, `speaking`, `reading:DONE/TOTAL`,
`catching:DONE/TOTAL`, `summarizing:N` or `paused`), `note TEXT`, and
`bye` when the service stops; `mind pos=... ms=... tok=... lN=w:logp,...`
when the service reads its mind (`mind.md`); the last 256 are shown to
a new `tail`. `reflect t=... pos=... why=... outcome=...` for every
check of a token when it reflects (`reflect.md`: the episode line). `delib
start|piece|end t=US pos=P TEXT`: the deliberation's own text for one
check (its question, each piece of its reasoning, its outcome; `Delib`,
`delib_line`), and `objective t=US TEXT`: what the stream is working
toward, when it changes. Both are additive: a terminal from before them
notes each kind once and drops it. `term start t=US id=N COMMAND` and `term end
t=US id=N code=C ms=M cut=0|1 timeout=0|1 OUTPUT` carry its terminal
(`term_end_line`, `TermLine`). `leaks` in the status counts the lines
beginning with `«` that the mind wrote itself (the journal frame's one
known leak: a line in someone else's voice).

`Client` connects, sends a line, reads a line, or `ask`s (sends and waits
for the `ok` or `err`, skipping what comes between); `parse` turns a
service line into a `Msg`; `status_line` and `kind_name` are the service's
side of the same format. A script needs nothing more than `nc -U` or a
socket in any language to do what the terminal does.
