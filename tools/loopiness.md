# loopiness

`tools/loopiness.c`: read a phi-stream `chain.log` and report how loopy the
stream was in a time window. Written by the dev stream itself (its first
project in the agent frame, 2026-10-02, in its working copy), checked
against Claude's own meter (identical counts on the same windows) and
brought in with two fixes: the hash table is sized to the window (a fixed
2^20 slots would probe forever past a million distinct 8-grams), and think
and speak tokens are counted apart. Its third project added `--kind think|speak|all`
(made with its `edit` tool): tool calls and code repeat by their structure
(46 percent of speech 8-grams over ten minutes, against 1.7 percent of
thinking), so a loop in its thinking shows only with `--kind think`. Brought
in with a misspelled kind refused (it counted nothing, silently) and the
README's pipes escaped (they broke its table).

## What it prints

- The window, in seconds and microseconds.
- Think and speak tokens in it (and each apart).
- 8-token sequences, the distinct ones, and the share that repeats one
  seen earlier in the window.
- Given lines, and how many contain 'from the system' (the harness's own
  interventions in the journal frame, candidates for setting off a loop).

## Usage

```
tcc -o loopiness tools/loopiness.c
./loopiness START_US END_US [--kind think|speak|all] [chain.log]
```

The two arguments are microsecond times as in the log's first column; the
log defaults to `chain.log` in the current directory. `--kind` filters which
token kind is collected for sequence analysis (default: all). This lets you
see how loopy the thinking tokens are separately from speech, because tool
calls and code repeat by their structure and hide the loops in thinking.

## Example

```
$ ./loopiness 1790928183845391 1790928483845391 ~/.local/share/phi-stream/dev/chain.log --kind think
window: 300.0 s (1790928183845391..1790928483845391 us)
tokens (think): 2659 (think 2659, speak 0)
8-token sequences: 2652
unique sequences: 2341
repeated sequences: 147 (5.54%)
given lines: 25, from the system: 2
```

Without `--kind`, all tokens are collected and sequences span both kinds.
The whole dev log (415k tokens) takes 0.15 s.

## How it works

It keeps the think and speak tokens in the window, slides 8 tokens over
them, and hashes each window's texts with FNV-1a (the texts concatenated:
the same text split differently counts as the same, which is what a
repeat of text means here). A table at least twice the number of windows,
probed linearly, finds the windows seen before.
