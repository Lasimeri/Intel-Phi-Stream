# guide-analyze

`tools/guide-analyze.c`: what the guide lane measured (`src/engine.md`,
the guide lane) in a time window of its `guide.log`. Written by the dev
stream (2026-10-02, its second project), checked against Claude's own
computation (identical: 1242 tokens, KL mean 0.1840 and median 0.0048,
107 flips, experts shared 0.814 over 330 tokens), and brought in with
fixes: the KL values sorted by `qsort` (an insertion sort took n squared
on a long window), the quartiles no longer read past the end of a
one-token window, the token listing behind `-v` (it printed every token),
the log's path an argument (it was fixed), a window past 500k tokens said
to be cut, and 64-byte token buffers (the static table was 270 MB).

```
tcc -o /tmp/guide-analyze tools/guide-analyze.c -lm
/tmp/guide-analyze START_US END_US [guide.log] [-v] [--src NAME]
```

The log defaults to `~/.local/share/phi-stream/dev/guide.log`; `-v` also
lists every token in the window; `--src NAME` selects only lines whose
source is NAME (`chain`, `lens`, or `placebo`). A line without `src=`
counts as chain; another name is refused.

## Input

One line per thinking token, tab-separated: the microsecond time (no
key), then `pos=N`, `kl=K`, `flip=0|1`, `live="TOKEN"`, `guide="TOKEN"`,
and `experts_shared=S` when the experts were captured (`experts on`),
and `src=X` naming the aside's source (`chain`, `lens`, or `placebo`).

## Output

- the tokens in the window;
- the KL of the guided distribution from the live one: mean, median and
  quartiles (linear interpolation);
- the share of tokens whose likeliest token the reflection changes;
- the ten commonest live to guide changes among them;
- the mean share of experts the guided and the live token have in common.

## Example

```
$ /tmp/guide-analyze 1790930754976696 1790931354976696
Tokens in window: 1242

KL statistics:
  mean:     0.1840
  median:   0.0048
  Q25:      0.0003
  Q75:      0.0570

Flip share: 107/1242 = 8.6%

Top 10 live→guide changes among flips:
    →  the: 2
   ( → 's: 2
  ...

Mean experts_shared: 0.814 (330 tokens)
```

The mean KL is about forty times the median: most thinking tokens barely
move, and a few move a lot (the reflection matters at a few places, not
everywhere).
