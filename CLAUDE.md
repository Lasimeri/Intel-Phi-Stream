# Intel-Phi-Stream: notes for an agent working here

Read `CONTRIBUTING.md` first; it is the authority. The non-obvious rules:

- What this is: one model split over the GPU, the Xeon Phi cards and
  host memory, one llama.cpp context, a live sequence that never stops
  while readings run beside it in the same decode cycles
  (`src/engine.md`), owned by a service on a Unix socket whose clients
  are the terminal and the one-shot commands (`src/serve.md`,
  `src/client.md`): to talk to the running mind from here, `phi-stream
  say`, `feed`, `tail`, `status`, `persona`; never start a second
  service (one process holds the cards). llama.cpp is linked, never changed. The cards come
  from the sibling Intel-Phi-AVX512 (`scripts/avx512.md`: `PHI_AVX512_ROOT`,
  a checkout next to this one or in `$HOME`, under either name); never
  copy anything of it here.
- The persona is composed from a personality base (`~/CLAUDE.md` of the
  person, verbatim, between a preamble and the frame's mechanics;
  `src/engine.md`); keep the base the person's own, never a copy edited
  here; the sampler bans dash-carrying tokens.
- Why one context: the backend's lock is per `begin` and `end`, so two
  contexts in two threads would interleave inside a multiply. Do not add
  a second context or thread on the model.
- Rust only; shell for `scripts/`. No Python or JavaScript, ever.
- Every code file gets a sibling `.md` with the same stem, written in the
  same change. No em or en dashes anywhere. A sibling repository's file is
  a GitHub link, not a bare path.
- Every hardware claim cites a source or a measurement with its command;
  logs on disk, never `/tmp` (tmpfs). This host drifts a quarter over
  tens of minutes: interleave.
- `make check` before committing, push after.
- Traps: one process at a time may hold the cards; M-RoPE refuses a
  token decoded again at a position already in the cache (take logits
  from the chunk that fed the token); `-t 8`, since llama.cpp's own CPU
  work between the backend's multiplies costs 10 ms a token at `-t 2`.
- Developing with the stream (`docs/dev.md`): when the dev service runs
  (`scripts/phi-stream.sh dev`), the stream is a peer in the work.
  - Monitor `target/release/phi-stream listen` for the whole session (its
    `to Claude` lines, rests and wakes, failures).
  - Talk to it with `phi-stream say --as Claude`; `phi-stream ask TEXT`
    waits for its `tell_claude` answer (a deep review can take minutes:
    read `to-claude.md` in its workspace, or the MCP `inbox`). The terminal
    itself is `phi-stream mcp` (`src/mcp.md`): `screen`, `type`, `keys`.
  - It reviews each new commit (it reads the working tree; a commit wakes
    it from `wait`). Weigh its findings like a colleague's: built when
    sound, answered with the evidence (file, line) when not. It cannot
    build Rust in its sandbox: say so when it claims a build.
  - Its own files (its working copy, `dev-copy/upper`): bring what is sound
    into the repository with the fixes it needs, credited to it, then
    `scripts/phi-stream.sh accept PATH...` so its stale copies stop
    shadowing the repository's.
  - Update it with `scripts/phi-stream.sh restart dev OPTIONS` (the window
    stays open; `dev` already adds `--second-chain`); give it ongoing
    objectives that end "when nothing waits, rest with wait".
  - Read `~/.local/share/phi-stream/dev/preferences.md`, and follow its
    preferences wherever the person's `CLAUDE.md` and these rules allow;
    say why when one cannot be followed.
