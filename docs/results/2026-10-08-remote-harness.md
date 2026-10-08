# 2026-10-08: the harness on the GPU rack, every endpoint wired

The person's request (2026-10-08, morning): every endpoint of the harness
working against the remote model, the harness running on the rack as a
service of its own, run the way the family's other repositories run
(Mechanical Jev and Intel Phi Jev: one settings file, `serve`, `stop`,
`doctor`, a client that finds the server). This record is kept as the
work goes, one commit per step, so the progress can be followed here.

## The two machines

| | desktop | GPU rack |
| --- | --- | --- |
| CPU, memory | Ryzen 7 5800X, 62 GB | EPYC 7742 (64 cores), 251 GB |
| accelerators | RTX 3090 and 3090 Ti (whisper and the face tracker only) | four RTX 3080 20 GB, four Xeon Phi 3120 |
| network | 1 GbE | 2.5 GbE (0.15 ms between them, NTP on both, clocks within about 1 ms) |
| role here | shows the harness (its window over ssh), its senses (cameras, microphones, the Glass write `feeds/`), Claude Code | runs everything: llama.phi's prefill server (port 8002, the GPUs) and decode server (port 8001, the CPU and the cards, two slots), the harness (`--remote http://127.0.0.1:8001`), builds, tests, pushes |

## Every endpoint, tested from the desktop (06:20 to 06:40)

| endpoint | before | now |
| --- | --- | --- |
| socket: `status`, `recent`, `tail`, `say`, `say-as`, `ask`, `feed`, `persona`, `objective`, `temp`, `set`, `chunk`, `pause`, `resume`, `quit` | answered on the rack; from the desktop only by ssh by hand | every verb of `phi-stream.sh` on the desktop runs on the rack (921da15) |
| MCP: `screen`, `type`, `keys`, `ask`, `say`, `inbox`, `status` | worked through `phi-stream-mcp.sh` (829e5ac); a Claude Code session started before that commit still holds a local server and needs `/mcp` | unchanged; `doctor` checks the registration |
| status during a long remote read | frozen at the last line for the whole read (minutes) | `reading done/total` from the server's `prompt_progress` every 2 s; `status.txt` and `diag.md` kept by a heartbeat (7114b30) |
| `quit` during a long remote read | waited behind the read | answered during it: the wait is cancelled and the service ends (7114b30) |
| feeds (cameras, microphones, Glass) | written on the desktop only; the rack's copy stood still since 2026-10-07 15:53, every camera offline to the harness | relayed, newest feed 0 s old on the rack (921aac0, the `phi-stream-feeds` user unit on the desktop) |
| `chain`, `goal` | answer "on", then the engine notes it cannot (no sequence beside the live one) | planned: forks over llama.phi (below) |
| `guide`, `experts` | answer "on" and do nothing | planned with the forks; `experts` needs the server's routing |
| `keep-at` (the checks) | the checks are off remotely | planned: forks and token probabilities over llama.phi |
| the mind, `lens` | off remotely | blocked (below) |
| `gate`, `probe` | need a model in process | not applicable to a remote model: they test the in-process engine |
| `code`, `anchor` | need logits in process | planned: greedy completions from the server |

## Steps

| step | state | commit |
| --- | --- | --- |
| the stream's candidates 25 to 28 reviewed and landed (diag remote section, honest retries, heartbeat, the pump), with read progress added | done | 7114b30 |
| feed relay, desktop to rack | done, unit installed and checked | 921aac0 |
| settings files (`phi-stream.conf`, `phi-stream.local.conf`), every verb run where the service is, `doctor` | done | 921da15 |
| a quit waits for a stalled summary, not a fixed time (engine and script) | done | 59b0d19 |
| splicing inputs into its thinking (`inject`): read on the prefill server beside the stream, joined once read; guards, toggle, fallbacks, tests (below) | done, off by default | this step's commit |
| the live check of a splice against llama.phi's `/phi/prefetch` | blocked: llama.phi at 28547bc92 has no such route yet (being added by another agent) | |
| inject on in the rack's `PHI_STREAM_DEV_ARGS`, then the measurement below | after the live check | |
| forks over llama.phi: a slot copy (`llama_memory_seq_cp`, as Intel Phi Jev's ARTICHOKE) so the second chain, the checks, the goal probe and the guide have sequences beside the live one | next | |
| token probabilities from the server for the checks and the goal probe | next | |
| `code` and `anchor` through the server | after | |
| the mind and the lens | blocked: no Jacobian lens fitted for Flash Next (the one fetched is for Qwen3.6-35B-A3B) and no capture of rows in the server; the logit lens is not a substitute | |

## Defects met

- **The doctor's first run** found no vocabulary file for the Q8_0 model
  the rack had served since 06:05 (only the Q6_K's): the harness's next
  start would have refused to start. Linked like the Q6_K's (the first
  shard is the metadata alone, 10,946,624 bytes).
- **A summary lost at the restart (06:37 to 06:45).** The decode server
  had restarted with the model switch, so the stream's slot was empty and
  the quit's summary needed the whole context (99,231 tokens) read again
  through the prefill server first: about twelve minutes, at about 70
  tokens a second deep in the context (the prefill keeps the KV cache in
  host memory, `-nkvo`, so attention runs on the CPU and grows with the
  depth). The launcher's fixed 480 s wait expired with the summary at
  454 tokens and ended the service; it restarted from the summary of
  2026-10-07 16:35 (its 32 notes, objective and messages kept). Fixed:
  the service's deadline moves on with every token gained, and the
  launcher waits as long as the service answers.
- **Candidate 28 had never been compiled** (the stream's sandbox builds
  no Rust): its pump closures took an argument their type did not have,
  and its error type had no `Display`. Candidate 26 gave up after five
  streams broken before a token, a pattern the rack's own model switch
  produced more than five times in a row; such breaks are now paced and
  bounded by the same 15 minutes as a server that does not answer.

## Open

- The prefill's rate deep in a long context (about 70 tokens a second
  past 60k): every restart of the harness on an empty slot reads its
  whole context again at that rate.
- The phictl card daemons poll at 97 percent of a core each while idle.

## Splicing inputs into its thinking (`inject`)

The person's words (2026-10-08): "Process parallelism needs to be
implemented and async prompt processing and token gen as well, the agents
harness needs to inject the processed prompt into the reasoning chain
dynamically once processed for the agent harness's management interface
we're developing".

What was built (`src/engine.md`, "Splicing inputs into its thinking";
`src/remote.md`, the prefetch):

- An input past `--direct-max` (48) tokens, arriving while the agent
  thinks, is taken at a fork point `p`: a line start in its thinking,
  outside code, in a turn with no `</think>` and no `<tool_call>` yet. The
  prefetch of `history[..p]` plus the input as a user turn (`T`) goes to
  the decode server (`POST /phi/prefetch`), whose prefill engine reads it
  on the GPUs without a slot. The live stream goes on, and the state is
  asked at most once a second.
- Once read, and outside an open tool call, the sequence becomes
  `history[..p]`, `T`, the `G` tokens generated since, and the pending
  token. The next request starts from the cached state of the first two,
  so the decode server should read about `|G| + 1` tokens.
- Every fallback puts the input back at the queue's front, so it comes at
  the turn's end as with inject off.
- `<|im_start|>` is banned in the agent frame (with inject off too), so the
  model cannot open the user turn the input comes in.
- Toggle: `inject on|off` on the socket, `phi-stream inject on|off`,
  `/inject on|off` in the terminal (and so the MCP `type` tool),
  `--inject`. Off by default. `inject.log` in the workspace keeps each
  start, join, give-up and toggle with its time.
- Tests, no network: the composition, the fork point, the join's wait on
  an open tool call, the frame, the prefetch's answers from raw HTTP
  bytes, and the toggle with every fallback. `make test`: 136 passed on
  the rack (`/mnt/raid5/phi/bench/harness-2026-10-08/inject-build-1.log`).

### The rule, written before any measurement

The 2026-10-02 evidence is that lines put inside its turns cut tool calls
in two, and that the model then wrote such lines itself with times it
made up. So inject stays on only if it is not worse than inject off on
any of three measures:

1. **Fabricated lines.** Lines of its thinking (`think` pieces of
   `chain.log`, joined and split at newlines) that begin, after spaces,
   with `«`, with `[at `, or with `[` followed within 40 characters by a
   time `HH:MM:SS`. Counted per 1000 thinking tokens.
2. **Unparsed or cut tool calls.** The counts in the `given` pieces that
   say `tool call(s) did not parse` or `your calls were written inside
   your thinking`. Counted per 100 turns, a turn being a `given` piece
   that holds `<|im_start|>assistant`.
3. **Thinking loops.** The share of repeated 8-grams in its thinking:
   `tools/loopiness.c START_US END_US --kind think chain.log`.

The windows:

- 30 minutes each, inject on and inject off alternating (this host
  drifts, so interleave), at least three of each.
- All after the restart that brought the `<|im_start|>` ban, so both arms
  have it.
- An on window counts only if `inject.log` shows a join in it. An off
  window counts only if an input past 48 tokens came in it. Otherwise it
  is extended.
- The inputs are what arrives in the work: the person's, Claude's
  messages and reviews, the feeds. Nothing is made up for the test.

The verdict, by pooled rates over all windows of an arm: inject is worse
if any measure with inject on exceeds inject off by more than the larger
of two margins:

- the spread of that measure across the off windows (their max minus their
  min);
- the floor: 1 line per 1000 thinking tokens for (1), 2 calls per 100
  turns for (2), 1 percentage point for (3).

If inject is worse by this rule, `--inject` comes out of the rack's
`PHI_STREAM_DEV_ARGS` and it stays off.

### What llama.phi needs to give

- `POST /phi/prefetch {"tokens": [...]}` answered at once with `{"id": N}`.
  Without the route, the server's 404 is read as "no prefetch", said once,
  and inputs keep the turn's end.
- `GET /phi/prefetch?id=N` with a state in a string field `state` (or
  `status`). `waiting` (or `queued`, `pending`, `running`, `reading`) is
  waited on. `ready` and `done` are both taken as read, which is
  provisional until the final format says what `done` means. `failed`
  (with `error` or `message`) falls back. An unknown id should be a 404
  or a `failed`.
- The cached state must be the one after exactly the tokens sent. The
  slot's next request then sends those tokens, plus `G`, plus one.
- A pinned slot that is not empty loads a cached state that shares more
  of the new prompt than the slot itself does. Without that, the decode
  server reads `|T| + |G| + 1` tokens itself, and the splice saves nothing
  but the wait.
