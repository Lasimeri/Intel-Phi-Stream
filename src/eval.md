# eval.rs

`phi-stream lens eval [--lens FILE] [SET.json ...]`: Gate C, whether the
Jacobian lens fitted on the base model reads this model, scored the way
the paper scores lens quality.

**The sets** are the reference implementation's six
(`data/evaluations` of anthropics/jacobian-lens at
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, Apache 2.0, fetched and
checksummed by `scripts/fetch-lens.sh` into `~/models/jlens/eval`):
multihop, multilingual, order-ops, poetry, association, typo. Each item
is a prompt and the intermediate concepts the model should be computing
at one position.

**The readout position** follows the sets' README: the last prompt token
(the token before the target for multihop, multilingual and order-ops;
the closing period for association; the last fragment of the misspelling
for typo), and for poetry the last newline token (the end of the
couplet's first line, where the rhyme is planned). Prompts are tokenized
with no BOS, as the reference does for this tokenizer (Qwen3.6's
`tokenizer_config.json`: `bos_token` null, `add_bos_token` false).

**Scoring** (the paper's methods comparison): at the readout position
the residual of every fitted block (0..38 here) is read through the
lens, `unembed(J_l h)`, and through the plain logit lens, `unembed(h)`;
an intermediate's rank is the best over its single-token forms (the word
with and without a leading space, each kept only if it is one token) and
over the blocks. Recovered at `k` when the rank is below `k`. Reported:
pass@1, @10, @100, and the area under pass@k against log k for k from 1
to 1000, normalized so that always-first scores 1 (exact for integer
ranks: a rank `r` contributes `(ln 1000 - ln (r + 1)) / ln 1000`). Also
the model's own final distribution (block 39 decoded, the same for both
lenses), and per block the fraction of intermediates recovered at 10 for
each lens: where that curve rises and falls is the workspace band.

**Order-ops** uses a synonym expansion the README describes ("numbers:
digit and word forms; operations: symbol and word forms") but does not
publish; `synonyms` holds this program's table (numbers 0 to 20 with
their English words; addition, subtraction, multiplication, division,
mod and squared with their words and symbols). It is stated so it can be
checked, and the record says that set's numbers rest on it.

Intermediates with no single-token form in this vocabulary are skipped
and listed. The readout of one item is one decode of its prompt (with
the capture of every block, `capture.md`) and one readout graph of 79
columns (`readout.md`); the transports of all 39 blocks are resident on
the GPU (312 MiB), set aside in the split.

Tests: the area's normalization and pass@k; the synonym table.
