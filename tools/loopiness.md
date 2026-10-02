# loopiness

`tools/loopiness.c`: read a phi-stream `chain.log` and report how loopy the
stream was in a time window. Written by the dev stream itself (its first
project in the agent frame, 2026-10-02, in its working copy), checked
against Claude's own meter (identical counts on the same windows) and
brought in with two fixes: the hash table is sized to the window (a fixed
2^20 slots would probe forever past a million distinct 8-grams), and think
and speak tokens are counted apart.

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
./loopiness START_US END_US [chain.log]
```

The two arguments are microsecond times as in the log's first column; the
log defaults to `chain.log` in the current directory.

## Example

```
$ ./loopiness 1790928183845391 1790928483845391 ~/.local/share/phi-stream/dev/chain.log
window: 300.0 s (1790928183845391..1790928483845391 us)
think and speak tokens: 4546 (think 2659, speak 1887)
8-token sequences: 4539
unique sequences: 3709
repeated sequences: 830 (18.29%)
given lines: 25, from the system: 2
```

The whole dev log (415k tokens) takes 0.15 s.

## How it works

It keeps the think and speak tokens in the window, slides 8 tokens over
them, and hashes each window's texts with FNV-1a (the texts concatenated:
the same text split differently counts as the same, which is what a
repeat of text means here). A table at least twice the number of windows,
probed linearly, finds the windows seen before.
