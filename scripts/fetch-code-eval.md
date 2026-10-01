# fetch-code-eval.sh

```
scripts/fetch-code-eval.sh [DIR]     # default ~/models/code-eval (PHI_STREAM_CODE_DIR)
```

Fetches the coding evaluation the stream is measured with: MultiPL-E's
translation of HumanEval to Rust (`humaneval-rs`, 156 of HumanEval's 164
tasks; the "reworded" prompts), from the Hugging Face dataset
[nuprl/MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E) at
revision `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`, the file
`humaneval-rs/test-00000-of-00001.parquet`, 75300 bytes, sha256
`92dba15e4da4deb4dbae5e15ce1712c95ec5a63c3ead69886271470309d9d314`,
with the dataset card. Licence: the dataset card says MIT; the
[MultiPL-E repository](https://github.com/nuprl/MultiPL-E) the tasks
come from carries the BSD 3-Clause licence with a fourth clause: its
contents "may not be used as training data for any machine learning
model, including but not limited to neural networks". The stricter one
applies: this program only evaluates with the tasks, and nothing a run
produces may go into a fine-tune (`src/code.md`).

The set is distributed only as parquet; `tools/parquet-jsonl` (a helper
of its own, built from the local cargo cache when it can be) converts
it to `humaneval-rs.jsonl`, one task a line: `name`, `prompt` (a doc
comment and the function's signature, ending `{`), `stop_tokens`
(`\n}`), `tests` (the closing brace and a `main` of `assert_eq!`s). The
script checks the count, 156. Idempotent; a download that does not
match its pinned size and sha256 is removed.
