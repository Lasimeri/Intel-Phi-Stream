# reflect-analyze.c

`tcc -run tools/reflect-analyze.c < ~/.local/share/phi-stream/dev/reflect.log`:
a summary of the reflection loop's episodes (`src/reflect.md`).

It prints:
- the lines read;
- the checks changed and the checks kept;
- how many choices fell below the write threshold (`keep` under 0.45: a
  check that went on to write);
- a histogram, in 5 percent buckets, of `top1p`, the probability of the
  deliberation's likeliest next token at `Decision:`. That is how sure the
  deliberation was of its first word, not the decision itself.

**Written by the dev stream itself** in the first dev session
(`docs/dev.md`), as its first piece of code, to audit its own checks. It
could not write files, so it wrote the program in its journal and asked
Claude to place it. Claude built and ran each version on its real log and
handed back the output, and it fixed its own bugs:
- `val` was not ended at its space, so no outcome ever matched;
- after that fix, the loop stopped after each line's first field.
  It diagnosed this one itself and named the fix.

Claude's changes, marked in the source:
- `kept` counts only `outcome=kept` (it had counted every outcome but
  `changed`);
- the three closing lines, where its block was cut by its own fence.

On the 877 episodes of that session it printed changed 11, kept 746, and
51 below the threshold (5.8 percent), each matching the file itself.

C, built by tcc at run time.
