# mind.rs

What is on the stream's mind, token by token. For every token the live
sequence places, the residual after a few chosen blocks at that token's
own position (`capture.md`) is read through the Jacobian lens
(`lens.md`): `unembed(J_l h)`, ranked on the GPU (`readout.md`). A
reading belongs to exactly one position; nothing is averaged over
tokens (the averaging that makes a lens is the fit's, over contexts,
done once).

- `Mind::new` runs after the stream's first decode (the capture has
  then seen the model's unembedding and output norm): it loads the
  lens's transports for the chosen blocks onto the GPU and computes the
  vocabulary's display mask.
- `Mind::read` takes the capture's output row of the decode that just
  happened (the token placed at `pos`), transports, norms, unembeds and
  takes the top 64 per block on the GPU, then keeps the first `k`
  word-like ones for display. Ranks are over the whole vocabulary; only
  the display is filtered, by the reference's rule (`is_wordlike`,
  `_meaningful_token_mask` in jlens/vis.py): the token's text, stripped,
  is non-empty, not a special token, and alphanumeric with apostrophes or
  hyphens only inside.
- Each reading goes to `mind.log` in the workspace and out as
  `Event::Mind`: one line, `pos=P ms=M tok=TEXT l20=w:logp,... l26=...`
  (`line`, `parse_line`; the token escaped, spaces as `\s`).

**In the engine** (`engine.md`): `mind_step` runs right after every
decode that asked for a token (the opening, every live token, the last
token of an injection, the swap at the end of a chase), synchronously,
before the next cycle: the stream waits for it, which is why its cost is
measured (Gate B). With `--mind` off no callback is installed and
nothing of this runs.

**Options** (`serve`, `run`): `--mind`, `--lens FILE` (default
`~/models/jlens/qwen3.6-35B-A3B/lens.jlens`, made by
`scripts/fetch-lens.sh`), `--mind-layers` (the blocks read), `--mind-k`
(words shown per block). The blocks must be ones the lens fitted.

**Where it shows**: `phi-stream tail --mind`; the terminal's mind strip
(the last token and its words per block) and `/mind` (the readings token
by token in place of the stream); `mind.log`; the status line's
`mind_ms`.

Tests: the display rule, the line format's round trip.
