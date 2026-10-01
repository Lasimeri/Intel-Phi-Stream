# rotlog.rs

A log file that rotates by size, for the stream's four logs:
- `stream.log`: the text;
- `chain.log`: each piece with its microsecond;
- `mind.log`: a reading per token;
- `reflect.log`: an episode per check.

The stream never stops, and none of them rotated. The dev stream's own
audit of its checks found it: "it is continuously accumulating with NO
rotation policy ... it could grow unbounded".

Past `MAX` (64 MiB) a log is renamed to `NAME.1`, replacing the previous
one, and a fresh file is begun, so each log holds at most 128 MiB on
disk. A write that would take the file past its size rotates first. A
log reopened at a start counts what the file already holds. A log that
cannot be opened is skipped, as before.

Test: a log rotates past its size, keeps exactly one file before it,
and counts an existing file when reopened.
