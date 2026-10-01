# reflect-analyze.c

`tcc -run tools/reflect-analyze.c ~/.local/share/phi-stream/dev/reflect.log`
(or the log on standard input):
a summary of the reflection loop's episodes (`src/reflect.md`).

It prints:
- the lines read;
- the checks changed and the checks kept;
- how many choices fell below the write threshold (`keep` under 0.45: a
  check that went on to write);
- every outcome with its count (`kept`, `same`, `changed`, `unparsed`,
  and any other the log holds);
- three histograms in 5 percent buckets, the empty buckets left out:
  - `top1p`, the probability of the deliberation's likeliest next token
    at `Decision:`: how sure it was of its first word, not the decision;
  - `keep`, the decision itself: keep's share of keep plus write
    (`src/reflect.md`), under 0.45 a write;
  - `fmt`, keep plus write: how much of the distribution answered at all
    (under 0.2 the check counts as unparsed).

**Written by the dev stream itself** in the first dev session
(`docs/dev.md`), as its first piece of code, to audit its own checks. It
could not write files, so it wrote the program in its journal and asked
Claude to place it. Claude built and ran each version on its real log and
handed back the output, and it fixed its own bugs:
- `val` was not ended at its space, so no outcome ever matched;
- after that fix, the loop stopped after each line's first field.
  It diagnosed this one itself and named the fix.

Claude's changes, marked in the source:
- the `keep` and `fmt` tables and the outcome counts: it asked whether
  `keep` had a spike at 0.55, and its table was of `top1p`;
- the log as an argument (an argument was ignored and standard input
  read, so it printed zeros);
- `kept` counts only `outcome=kept` (it had counted every outcome but
  `changed`);
- the three closing lines, where its block was cut by its own fence.

On the 877 episodes of that session it printed changed 11, kept 746, and
51 below the threshold (5.8 percent), each matching the file itself.

C, built by tcc at run time.
