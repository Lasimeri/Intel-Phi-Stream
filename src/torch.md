# torch.rs

A reader for the one kind of `torch.save` file this program needs, so
that the reference implementation's lens is read without Python.

**The archive.** `torch.save` (since torch 1.6) writes a zip archive:
`<prefix>/data.pkl` (a pickle), `<prefix>/data/<key>` (the raw bytes of
each storage, stored uncompressed), `<prefix>/byteorder`, `version` and a
few markers. `Archive::open` finds the pickle by its name, refuses an
archive written big-endian, and runs the pickle; `tensor_bytes` returns
a tensor's bytes after checking that it is contiguous and row-major,
within its storage, and that the storage entry is stored and has exactly
`numel x element size` bytes.

**The pickle machine** knows exactly the opcodes torch's protocol-2
pickler emits for dictionaries, lists, tuples, integers, floats, strings
and booleans (`PROTO`, `EMPTY_DICT`, `EMPTY_LIST`, `EMPTY_TUPLE`, `MARK`,
`BININT`, `BININT1`, `BININT2`, `LONG1`, `BINFLOAT`, `BINUNICODE`,
`SHORT_BINUNICODE`, `NONE`, `NEWTRUE`, `NEWFALSE`, `TUPLE`, `TUPLE1..3`,
`APPEND`, `APPENDS`, `SETITEM`, `SETITEMS`, `BINPUT`, `LONG_BINPUT`,
`BINGET`, `LONG_BINGET`, `GLOBAL`, `STOP`) and for tensors: `BINPERSID`
over torch's `('storage', torch.<Type>Storage, key, location, numel)`
and `REDUCE` of `torch._utils._rebuild_tensor_v2(storage, offset, size,
stride, requires_grad, OrderedDict())`, plus `REDUCE` of an empty
`collections.OrderedDict`. Anything else (another opcode, protocol above
2, a `REDUCE` of any other global, a storage class other than Half,
BFloat16 or Float, a tensor with hooks) is refused with its name: the
reader never executes or imports anything, it only builds values.

Tests (`cargo test`): a lens-shaped pickle written out opcode by opcode
reads to the expected dictionary and tensor; an unknown opcode, a
protocol 4 stream and a `REDUCE` of `os.system` are refused; negative
`LONG1` integers read.
