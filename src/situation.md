# situation.rs

Where and when the stream runs, read from the running system each time it
is asked (the opening, and every rollover's base, `engine.md`), never
written in by hand. A fact that cannot be read is left out, not guessed.

| fact | read from |
| --- | --- |
| the date and time, to the microsecond, its zone and offset from UTC | the real-time clock (`clock.rs`), the C library's local time (`TZ`, else `/etc/localtime`), the zone's name from `TZ` or the link's target |
| the host's name, system, kernel | `/proc/sys/kernel/hostname`, `/etc/os-release` (`PRETTY_NAME`), `/proc/sys/kernel/osrelease` |
| the processor and its threads, the memory | `/proc/cpuinfo`, `/proc/meminfo` |
| the GPUs | the NVIDIA driver's `/proc/driver/nvidia/gpus/*/information` |
| the Xeon Phi cards | the stack's cards file (`PHI_CARDS`, else `~/.config/phi/cards`): its lines that are not comments |
| how long the host has been up | `/proc/uptime` |

The engine adds what only it knows (`situation` in `engine.rs`): the model
file as loaded, its blocks on the GPU and where the rest are (the cards
only when this run's backend is the cards', `GGML_BACKEND_PATH`), the
context's cells, its workspace and the repository it develops, and who is
present. The self-model after it (`about`) no longer names the model or
the hardware: it says what the mind is, and the situation where.

Tests: the files are read as they are written (os-release, cpuinfo,
meminfo, a cards file with comments and blank lines); the time carries its
zone; a host is described by what was read, nothing for nothing, and this
machine read live gives a sentence.
