# conf.sh

This machine's settings for the scripts, sourced by `phi-stream.sh` and
`feed-relay.sh` (not run). The same order as the family's `xks`
([Intel-Phi-Jev `src/config.md`](https://github.com/Lasimeri/Intel-Phi-Jev/blob/main/src/config.md)),
the first to set a key winning:

1. the environment (anything already set, even empty, is never overridden);
2. `$PHI_STREAM_CONFIG`, when set;
3. `phi-stream.local.conf` in the checkout (not tracked: this machine's own);
4. `~/.config/phi-stream/phi-stream.conf`;
5. [`phi-stream.conf`](../phi-stream.conf) in the checkout (tracked: the
   defaults, every key described there).

Each file is `KEY=VALUE` lines and `#` comments, shell-sourceable: a value
is read as a shell reads it (quotes, `$HOME`), so a value with spaces is
quoted. Keys are upper-case names; anything else is skipped.

`phi_stream_q` quotes a word for the far login shell of
`PHI_STREAM_HOST`, whichever it is: single quotes, an inner quote as
`'\''`, read the same by bash, sh and fish (the GPU rack's login shell is
fish).

## The two machines (2026-10-08)

The desktop shows the service and runs its senses; the GPU rack runs it:

```sh
# the desktop: ~/Intel Phi Stream/phi-stream.local.conf
PHI_STREAM_HOST="env RACK_NOMUX=1 $HOME/.local/bin/rack"
PHI_STREAM_HOST_TTY="env RACK_NOMUX=1 $HOME/.local/bin/rack -t"
PHI_STREAM_HOST_DIR="/mnt/raid5/phi/Intel Phi Stream"

# the rack: /mnt/raid5/phi/Intel Phi Stream/phi-stream.local.conf
PHI_STREAM_DEV_ARGS="--remote http://127.0.0.1:8001 --frame agent --improve --temp 0.5 --top-p 0.95 --min-p 0.05"
PHI_STREAM_WINDOW=0
PHI_STREAM_QUIT_WAIT=480
PHI_STREAM_BUILD_CPUS=3,7,11,15,19,23,27,31
PHI_STREAM_BUILD_JOBS=8
PHI_STREAM_TERM_CPU=35
```

Then on the desktop `scripts/phi-stream.sh status` (and `say`, `ask`,
`tail`, `listen`, `restart`, every verb) runs on the rack,
`scripts/phi-stream.sh window` opens the harness's window on the desktop,
`scripts/phi-stream.sh doctor` checks both sides, and
`scripts/feed-relay.sh install` keeps the desktop's feeds in the rack's
workspace. On the rack, `scripts/phi-stream.sh restart` starts the harness
with `PHI_STREAM_DEV_ARGS`.
