# gate.rs

`phi-stream gate [--thoughts 48] [--chunk 16] [--compare 32]`: does the
engine's composition (`engine.md`) give what a straight sequence gives?

A (the opening turns) is read into sequence 0 and `thoughts` tokens are
generated greedily while B (a bracketed document, six paragraphs) is
read beside them the engine's way: sequence 1 given A's cells and state,
the first chunk alone, then a chunk a cycle with the live token. Sequence
2 is composed (A's cells, B's cells and state, then the thoughts and the
pending token caught up in chunks) and sequence 3 is fed A, B, the
thoughts and the pending token in one go. Compared: the next token's
logits (its identity, the largest absolute difference across the
vocabulary, the straight sequence's top-2 margin), and `compare` greedy
tokens after each, counted up to the first difference. Greedy throughout,
so a difference is either the composition's or the kernels' rounding
across batch sizes. The control separates the two: the straight sequence
fed again in chunks of `chunk` (the composed one's sizes) against itself
at the full batch gives the rounding's own size, and against the
composed sequence what is left for the composition. The verdict line
says whether the next token is the same.

**`gate --snapshot`**: whether a sequence copied from the live one keeps
the state at the copy while the live one goes on writing (what taking a
token back without anyone seeing it rests on). The live sequence (0)
reads a prompt and generates 24 tokens greedily; the snapshot (1) is
copied from it (`seq_cp` of every cell and of the recurrent state,
which llama.cpp shares until one of the two writes); the live sequence
places its choice and 24 more tokens; the snapshot then places the
second-best token where the live one placed its choice. Each is
compared with a control that decodes exactly the same tokens in exactly
the same chunks without any snapshot: the snapshot's next-token logits,
and the live sequence's after the copy. The largest difference in each
is printed (an exact zero means the computation was the same; a small
one, the cache's cell layout, which the controls do not share); a
different next token, or a difference above 0.5, fails.

**`gate --reflect [--episodes 8]`**: whether the reflection loop's lanes
(`reflect.md`, `engine.md`: a snapshot S and a deliberation D copied
from the live sequence before a token, D's lane decoded beside the live
token, then kept or rewound onto S) keep every sequence's state, and
whether every captured row is its own lane's. The capture is installed
as the engine has it when it reflects (blocks 27, 29, 31 and the final
one).

Each episode starts from a fresh prompt, and the gap grows by five
tokens per episode. The sequences and cells one episode frees are taken
by the next, so the layout drifts. Kept and rewound episodes alternate.
An episode is run three ways:

- **The lanes as the engine runs them.** The live sequence reads the
  prompt and goes on greedily, one token at a time. S and D are copied
  before the token it chose, and D's first question token goes alone.
  Then 16 cycles each decode the live token and D's lane in one batch,
  the live lane first: the question's rest as one chunk, then D's greedy
  answer one token at a time. Kept: S and D are dropped and the live
  sequence goes on alone for 3 tokens. Rewound: the live sequence and D
  are dropped, and S places the second-best token and goes on.
- **The same without the snapshot, twice.** The same tokens go through
  the same decodes in the same cells, with D copied from the live
  sequence as in the run, so the prefix's cells are shared the same way.
  The live sequence then owns its recurrent state, where in the run it
  shares that state with S until its first write inside a two-lane
  batch. That copy-on-write is the path under test. On the rewound path,
  S's part is played by a sequence prefilled as the live one was, into
  the same cells.
- **The resolution check (rewound episodes only).** The control runs
  again with the token in question placed instead of the second best.

Checked in every decode: each captured output row, read as it is (the
final block, no transport), must give llama's logits for the lane at
its index.

Compared, per lane, between run and control: the largest |delta logit|
and KL of every row, for the live sequence, D, and the tokens after,
plus whether each row's next token agrees. The control against itself
is the run-to-run floor. The copies pass if they are within twice the
floor (at least 0.01) with the same next tokens. The gate fails if the
control does not reproduce itself, or if the one-token difference is not
far above the limit (four times): in either case it would say nothing.

Two controls tried first did not measure the copies:

- **One token at a time against pairs.** This floor is the kernels'
  rounding across batch sizes (KL up to 0.012). A one-token difference
  three positions back reached only 0.008, so it could not resolve the
  smallest mix-up.
- **D prefilled separately instead of copied.** Its prefix sits in other
  cells, so the attention runs over a different number of cells and
  reduces in a different order. That moves logits by up to 1.65 even
  though the snapshot's own path matched exactly.

Both runs are kept in the record.
