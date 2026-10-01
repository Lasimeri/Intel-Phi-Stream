# main.rs (tools/parquet-jsonl)

`parquet-jsonl IN.parquet > OUT.jsonl`: every row of a parquet file as
one JSON object a line, columns by name (the parquet crate's record
reader and its `to_json_value`), the row count on stderr. It exists so
that evaluation sets distributed only as parquet (MultiPL-E's
HumanEval in Rust, `scripts/fetch-code-eval.md`) are converted once,
without Python, and the stream's own binary carries no parquet reader.
Snappy and zstd pages are read (the codecs the Hugging Face hub writes).

```
cargo build --release --manifest-path tools/parquet-jsonl/Cargo.toml
tools/parquet-jsonl/target/release/parquet-jsonl test.parquet > test.jsonl
```
