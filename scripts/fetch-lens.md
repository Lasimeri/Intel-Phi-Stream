# fetch-lens.sh

```
scripts/fetch-lens.sh [DIR]     # default ~/models/jlens/qwen3.6-35B-A3B (PHI_STREAM_LENS_DIR)
```

Fetches, checks and converts everything the lens commands read:

- The pre-fitted Jacobian lens for Qwen3.6-35B-A3B, the base model of
  Qwen3.8-35B-A3B: `lens.pt` of
  [stanleytheli/qwen3.6-35B-A3B-jlens](https://huggingface.co/stanleytheli/qwen3.6-35B-A3B-jlens)
  at revision `7a5dc7a6c770c272226a321409b30d7e6d773bba`, 327166510
  bytes, sha256 `2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1`
  (fitted with the reference implementation over 1000 WikiText-103
  prompts; blocks 0 to 38, target block 39; float16; Apache 2.0), and its
  model card.
- The reference implementation's six evaluation sets
  ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
  at `581d398613e5602a5af361e1c34d3a92ea82ba8e`, `data/evaluations`,
  Apache 2.0) into `eval/` beside the lens directory
  (`PHI_STREAM_EVAL_DIR`), each checked against its sha256, with the
  licence.
- Then `phi-stream lens convert lens.pt lens.jlens` (`src/lens.md`).

Idempotent: a file already present that checks out is kept; a download
that does not match its pinned size and sha256 is removed and the script
stops. Needs `curl` and `sha256sum`; the conversion needs the built
binary (`make build`, or `PHI_STREAM_BIN`).
