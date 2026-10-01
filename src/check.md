# check.rs

`phi-stream lens check`: the gate that the capture and the readout give
the model's own next-token distribution, before any lens is trusted.

The model is loaded with a capture of the final block (`l_out-39` here,
`capture.md`) and the logits; for each compared token the captured
final residual is decoded by the readout with no transport
(`readout.md`) and compared with `llama_get_logits_ith` for the same
token. Also compared: the logits row the capture copied from the graph
against llama's (bit for bit: the row bookkeeping is right), the host's
norm of the captured residual against the graph's normed row (the
capture holds the residual the model normalized), and the unembedding
alone of the graph's normed row against llama's logits (the model's own
unembedding, used in place, is the one llama.cpp used).

The tokens are the three kinds of cycle the engine makes: the end of a
prompt read in chunks, eight tokens decoded alone, eight decoded beside
another sequence's chunk of 16 (the live token is one row of a shared
micro-batch), and the last token of a 41-token injection. A token
passes when the graph's row equals llama's exactly, the top 10 agree in
order, and the readout differs by at most 0.01 anywhere. The table gives
each token's micro-batch and row (`mb:row`). Exit status 1 on any
failure.

```
scripts/phi-stream.sh lens check
```

Needs no lens file: it checks everything the lens readout rests on.
