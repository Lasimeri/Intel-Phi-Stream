# Documentation

| record | what |
| --- | --- |
| [2026-10-01-phi-stream.md](results/2026-10-01-phi-stream.md) | the founding record, measured in Intel-Phi-AVX512 the day the crate was written there: the split (28 of 41 blocks on the 3090 Ti, 2.94 GB on the cards, 8.0 GiB in host memory), why one context, the rates alone and reading beside the stream by chunk, the composition gated against a straight sequence with a batch-size control, the stream in use and the terminal under tmux |
| [2026-10-01-lens-capture.md](results/2026-10-01-lens-capture.md) | the lens readout, Gate A: the residual of the token being placed, captured live through llama.cpp's public eval callback and decoded with the model's own output norm and unembedding (used in place, not copied), reproduces llama's logits exactly in every kind of cycle (alone, beside a reading, end of an injection); the two defects the gate found |
