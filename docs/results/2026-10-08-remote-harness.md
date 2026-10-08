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
| a quit waits for a stalled summary, not a fixed time (engine and script) | done | this record's commit |
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
