# lens.rs

The Jacobian lens as this program keeps it: a `.jlens` file converted
once from the reference implementation's `lens.pt` (`torch.rs`), so that
nothing at run time parses a pickle.

**The reference's file** (`JacobianLens.save`,
[`jlens/lens.py`](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py)):
`{"J": {layer: float16 [d_model, d_model]}, "n_prompts", "source_layers",
"d_model"}`. `J_l` maps the residual after block `l` into the final
block's basis; the reference applies it as `residual @ J.T`, so row `i`
of `J_l` holds the coefficients of output coordinate `i`.

**`convert`** checks every tensor (float16, `d x d`, contiguous, one per
listed block, all values finite) and writes:

| bytes | what |
| --- | --- |
| 0..8 | `PHIJLENS` |
| 8..12 | version, 1 (u32 little-endian, as every number here) |
| 12..16 | `d_model` |
| 16..20 | number of blocks `n` |
| 20..24 | `n_prompts` |
| 24..88 | the sha256 of the source file, hex |
| 88.. | `n` block numbers (i32), then `n` values of `||J_l||_F / sqrt(d)` (f32) |
| ..4096 | zeros |
| 4096.. | per block, ascending: `d * d` float16, row-major, little-endian |

The matrices start page-aligned. The conversion writes to `OUT.part` and
renames it, so a failed conversion leaves no half file; it is
deterministic (the same `lens.pt` gives the same bytes).

**`Lens::open`** checks the magic, the version and that the file's size
is exactly the header's promise; `matrix(l)` reads one block's matrix.
`f16_to_f32` converts exactly (subnormals, infinities, NaNs).

The norms are the reference's own convergence measure (its fit logs the
largest `||J||/sqrt(d)`); for the Qwen3.6-35B-A3B lens they rise from
0.55 (block 0) to 1.28 (block 34) and fall to 1.02 (block 38).

Tests: exact half-float conversion, the header round trip.
