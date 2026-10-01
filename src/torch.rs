//! A reader for the one kind of file `torch.save` writes that this program
//! needs: a zip archive (entries stored, not compressed) holding a pickle
//! (`<prefix>/data.pkl`, protocol 2) and the raw bytes of each tensor's
//! storage (`<prefix>/data/<key>`). The pickle is run on a small machine
//! that knows exactly the opcodes torch's pickler emits for dictionaries,
//! lists, numbers, strings and tensors (`torch._utils._rebuild_tensor_v2`
//! over a persistent storage id), and refuses everything else by name.
//! No Python. See torch.md.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use anyhow::{bail, Context as _, Result};

/// A storage's element type, from its pickled class name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F16,
    BF16,
    F32,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
        }
    }

    fn from_class(name: &str) -> Option<Self> {
        match name {
            "HalfStorage" => Some(Dtype::F16),
            "BFloat16Storage" => Some(Dtype::BF16),
            "FloatStorage" => Some(Dtype::F32),
            _ => None,
        }
    }
}

/// A tensor as the pickle describes it: where its bytes are and their shape.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRef {
    pub key: String,
    pub dtype: Dtype,
    pub storage_numel: i64,
    pub offset: i64,
    pub size: Vec<i64>,
    pub stride: Vec<i64>,
}

/// A value of the pickle.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage {
        key: String,
        dtype: Dtype,
        numel: i64,
    },
    Tensor(TensorRef),
    /// `collections.OrderedDict()`, as torch pickles a tensor's hooks.
    OrderedDict(Vec<(Value, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(d) | Value::OrderedDict(d) => d
                .iter()
                .find(|(k, _)| matches!(k, Value::Str(s) if s == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
}

enum Slot {
    Value(Value),
    Mark,
}

/// Run a pickle; the value left on the stack at STOP.
pub fn unpickle(data: &[u8]) -> Result<Value> {
    let mut stack: Vec<Slot> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        if *i + n > data.len() {
            bail!("the pickle ends inside an operand at byte {}", *i);
        }
        let s = &data[*i..*i + n];
        *i += n;
        Ok(s)
    };
    fn pop(stack: &mut Vec<Slot>) -> Result<Value> {
        match stack.pop() {
            Some(Slot::Value(v)) => Ok(v),
            Some(Slot::Mark) => bail!("a mark where a value was expected"),
            None => bail!("the pickle stack is empty"),
        }
    }
    fn pop_mark(stack: &mut Vec<Slot>) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        loop {
            match stack.pop() {
                Some(Slot::Value(v)) => items.push(v),
                Some(Slot::Mark) => break,
                None => bail!("no mark on the pickle stack"),
            }
        }
        items.reverse();
        Ok(items)
    }
    fn line(data: &[u8], i: &mut usize) -> Result<String> {
        let start = *i;
        while *i < data.len() && data[*i] != b'\n' {
            *i += 1;
        }
        if *i >= data.len() {
            bail!("an unterminated line in the pickle");
        }
        let s = String::from_utf8(data[start..*i].to_vec())?;
        *i += 1;
        Ok(s)
    }
    loop {
        if i >= data.len() {
            bail!("the pickle ends without STOP");
        }
        let op = data[i];
        i += 1;
        match op {
            0x80 => {
                // PROTO
                let p = take(&mut i, 1)?[0];
                if p > 2 {
                    bail!("pickle protocol {p}: this reader knows torch.save's protocol 2");
                }
            }
            b'.' => {
                // STOP
                return pop(&mut stack);
            }
            b'}' => stack.push(Slot::Value(Value::Dict(Vec::new()))),
            b']' => stack.push(Slot::Value(Value::List(Vec::new()))),
            b')' => stack.push(Slot::Value(Value::Tuple(Vec::new()))),
            b'(' => stack.push(Slot::Mark),
            b'N' => stack.push(Slot::Value(Value::None)),
            0x88 => stack.push(Slot::Value(Value::Bool(true))),
            0x89 => stack.push(Slot::Value(Value::Bool(false))),
            b'K' => {
                let v = take(&mut i, 1)?[0] as i64;
                stack.push(Slot::Value(Value::Int(v)));
            }
            b'M' => {
                let b = take(&mut i, 2)?;
                stack.push(Slot::Value(Value::Int(u16::from_le_bytes([b[0], b[1]]) as i64)));
            }
            b'J' => {
                let b = take(&mut i, 4)?;
                stack.push(Slot::Value(Value::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)));
            }
            0x8a => {
                // LONG1: a little-endian two's complement integer of n bytes.
                let n = take(&mut i, 1)?[0] as usize;
                let b = take(&mut i, n)?;
                if n > 8 {
                    bail!("a {n}-byte integer does not fit 64 bits");
                }
                let mut v: i64 = 0;
                for (k, &byte) in b.iter().enumerate() {
                    v |= (byte as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Slot::Value(Value::Int(v)));
            }
            b'G' => {
                let b = take(&mut i, 8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                stack.push(Slot::Value(Value::Float(f64::from_be_bytes(a))));
            }
            b'X' => {
                let b = take(&mut i, 4)?;
                let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                let s = String::from_utf8(take(&mut i, n)?.to_vec())?;
                stack.push(Slot::Value(Value::Str(s)));
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = take(&mut i, 1)?[0] as usize;
                let s = String::from_utf8(take(&mut i, n)?.to_vec())?;
                stack.push(Slot::Value(Value::Str(s)));
            }
            b'c' => {
                let module = line(data, &mut i)?;
                let name = line(data, &mut i)?;
                stack.push(Slot::Value(Value::Global(module, name)));
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                let v = match stack.last() {
                    Some(Slot::Value(v)) => v.clone(),
                    _ => bail!("BINPUT with no value on the stack"),
                };
                memo.insert(k, v);
            }
            b'r' => {
                let b = take(&mut i, 4)?;
                let k = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let v = match stack.last() {
                    Some(Slot::Value(v)) => v.clone(),
                    _ => bail!("LONG_BINPUT with no value on the stack"),
                };
                memo.insert(k, v);
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                stack.push(Slot::Value(memo.get(&k).cloned().with_context(|| format!("BINGET of unset memo {k}"))?));
            }
            b'j' => {
                let b = take(&mut i, 4)?;
                let k = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                stack.push(Slot::Value(memo.get(&k).cloned().with_context(|| format!("LONG_BINGET of unset memo {k}"))?));
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Slot::Value(Value::Tuple(items)));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Slot::Value(Value::Tuple(vec![a])));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Slot::Value(Value::Tuple(vec![a, b])));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Slot::Value(Value::Tuple(vec![a, b, c])));
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Slot::Value(Value::List(l))) => l.push(v),
                    _ => bail!("APPEND to something that is not a list"),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Slot::Value(Value::List(l))) => l.extend(items),
                    _ => bail!("APPENDS to something that is not a list"),
                }
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Slot::Value(Value::Dict(d))) | Some(Slot::Value(Value::OrderedDict(d))) => d.push((k, v)),
                    _ => bail!("SETITEM on something that is not a dictionary"),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if items.len() % 2 != 0 {
                    bail!("SETITEMS with an odd number of items");
                }
                let pairs: Vec<(Value, Value)> = items.chunks(2).map(|p| (p[0].clone(), p[1].clone())).collect();
                match stack.last_mut() {
                    Some(Slot::Value(Value::Dict(d))) | Some(Slot::Value(Value::OrderedDict(d))) => d.extend(pairs),
                    _ => bail!("SETITEMS on something that is not a dictionary"),
                }
            }
            b'Q' => {
                // BINPERSID: torch's ('storage', class, key, location, numel).
                let pid = pop(&mut stack)?;
                stack.push(Slot::Value(persistent_storage(pid)?));
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(Slot::Value(reduce(callable, args)?));
            }
            other => bail!("pickle opcode 0x{other:02x} ({:?}) at byte {}: not one torch.save writes for a lens", other as char, i - 1),
        }
    }
}

fn persistent_storage(pid: Value) -> Result<Value> {
    let Value::Tuple(t) = pid else {
        bail!("a persistent id that is not a tuple");
    };
    match t.as_slice() {
        [Value::Str(kind), Value::Global(module, class), Value::Str(key), Value::Str(_location), Value::Int(numel)]
            if kind == "storage" && module == "torch" =>
        {
            let dtype = Dtype::from_class(class).with_context(|| {
                format!("storage class torch.{class} is not one this reader knows")
            })?;
            Ok(Value::Storage {
                key: key.clone(),
                dtype,
                numel: *numel,
            })
        }
        _ => bail!("a persistent id that is not torch's storage tuple: {t:?}"),
    }
}

fn ints(v: &Value, what: &str) -> Result<Vec<i64>> {
    match v {
        Value::Tuple(t) => t
            .iter()
            .map(|x| {
                x.as_int()
                    .with_context(|| format!("{what}: not an integer"))
            })
            .collect(),
        _ => bail!("{what}: not a tuple"),
    }
}

fn reduce(callable: Value, args: Value) -> Result<Value> {
    let Value::Global(module, name) = &callable else {
        bail!("REDUCE of something that is not a global");
    };
    let Value::Tuple(a) = &args else {
        bail!("REDUCE of {module}.{name} with arguments that are not a tuple");
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") if a.is_empty() => Ok(Value::OrderedDict(Vec::new())),
        ("torch._utils", "_rebuild_tensor_v2") => {
            // (storage, storage_offset, size, stride, requires_grad, backward_hooks)
            if a.len() != 6 {
                bail!("_rebuild_tensor_v2 with {} arguments, not 6", a.len());
            }
            let Value::Storage { key, dtype, numel } = &a[0] else {
                bail!("_rebuild_tensor_v2 over something that is not a storage");
            };
            let offset = a[1]
                .as_int()
                .context("a tensor offset that is not an integer")?;
            let size = ints(&a[2], "a tensor size")?;
            let stride = ints(&a[3], "a tensor stride")?;
            if !matches!(&a[5], Value::OrderedDict(h) if h.is_empty()) {
                bail!("a tensor with backward hooks");
            }
            Ok(Value::Tensor(TensorRef {
                key: key.clone(),
                dtype: *dtype,
                storage_numel: *numel,
                offset,
                size,
                stride,
            }))
        }
        _ => bail!("REDUCE of {module}.{name}: not one torch.save writes for a lens"),
    }
}

/// A `torch.save` archive: its pickle's value and access to its storages.
pub struct Archive {
    zip: zip::ZipArchive<File>,
    prefix: String,
    pub value: Value,
}

impl Archive {
    pub fn open(path: &str) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("opening {path}"))?;
        let mut zip = zip::ZipArchive::new(f)
            .with_context(|| format!("{path} is not a zip archive (a torch.save file?)"))?;
        let pkl = (0..zip.len())
            .filter_map(|i| zip.by_index(i).ok().map(|e| e.name().to_string()))
            .find(|n| n.ends_with("/data.pkl"))
            .with_context(|| format!("{path} holds no data.pkl"))?;
        let prefix = pkl.trim_end_matches("data.pkl").to_string();
        let mut order = String::new();
        if let Ok(mut e) = zip.by_name(&format!("{prefix}byteorder")) {
            e.read_to_string(&mut order)?;
            if order.trim() != "little" {
                bail!("{path} was written {order}-endian; this reader knows little-endian");
            }
        }
        let mut bytes = Vec::new();
        zip.by_name(&pkl)?.read_to_end(&mut bytes)?;
        let value = unpickle(&bytes).with_context(|| format!("reading the pickle of {path}"))?;
        Ok(Self { zip, prefix, value })
    }

    /// The bytes of a tensor, checked to be contiguous, row-major and within
    /// its storage.
    pub fn tensor_bytes(&mut self, t: &TensorRef) -> Result<Vec<u8>> {
        let n: i64 = t.size.iter().product();
        let mut expect = 1i64;
        for d in (0..t.size.len()).rev() {
            if t.stride[d] != expect {
                bail!(
                    "tensor {} is not contiguous row-major: size {:?} stride {:?}",
                    t.key,
                    t.size,
                    t.stride
                );
            }
            expect *= t.size[d];
        }
        if t.offset < 0 || t.offset + n > t.storage_numel {
            bail!("tensor {} reaches outside its storage", t.key);
        }
        let mut e = self
            .zip
            .by_name(&format!("{}data/{}", self.prefix, t.key))
            .with_context(|| format!("storage {} missing", t.key))?;
        if e.compression() != zip::CompressionMethod::Stored {
            bail!(
                "storage {} is compressed; torch stores storages uncompressed",
                t.key
            );
        }
        let es = t.dtype.size() as u64;
        if e.size() != t.storage_numel as u64 * es {
            bail!(
                "storage {} holds {} bytes, not {}",
                t.key,
                e.size(),
                t.storage_numel as u64 * es
            );
        }
        let mut all = Vec::with_capacity(e.size() as usize);
        e.read_to_end(&mut all)?;
        let start = t.offset as usize * es as usize;
        Ok(all[start..start + n as usize * es as usize].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The opcodes torch.save emits for `{"J": {0: tensor}, "n": 5, "l": [0]}`
    /// with a 2x2 float16 tensor, written out by hand.
    fn sample() -> Vec<u8> {
        let mut p = vec![0x80, 2, b'}', b'q', 0, b'('];
        let s = |p: &mut Vec<u8>, x: &str| {
            p.push(b'X');
            p.extend_from_slice(&(x.len() as u32).to_le_bytes());
            p.extend_from_slice(x.as_bytes());
        };
        s(&mut p, "J");
        p.extend_from_slice(&[b'}', b'q', 1, b'(', b'K', 0]);
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.extend_from_slice(&[b'q', 2, b'(', b'(']);
        s(&mut p, "storage");
        p.extend_from_slice(b"ctorch\nHalfStorage\n");
        s(&mut p, "0");
        s(&mut p, "cpu");
        p.extend_from_slice(&[b'K', 4, b't', b'Q']);
        p.extend_from_slice(&[
            b'K', 0, b'K', 2, b'K', 2, 0x86, b'K', 2, b'K', 1, 0x86, 0x89,
        ]);
        p.extend_from_slice(b"ccollections\nOrderedDict\n");
        p.extend_from_slice(b")RtRu");
        s(&mut p, "n");
        p.extend_from_slice(&[b'K', 5]);
        s(&mut p, "l");
        p.extend_from_slice(&[b']', b'(', b'K', 0, b'e', b'u', b'.']);
        p
    }

    #[test]
    fn a_lens_shaped_pickle_reads() {
        let v = unpickle(&sample()).unwrap();
        assert_eq!(v.get("n").and_then(Value::as_int), Some(5));
        assert_eq!(v.get("l"), Some(&Value::List(vec![Value::Int(0)])));
        let Some(Value::Dict(j)) = v.get("J") else {
            panic!("no J");
        };
        assert_eq!(j.len(), 1);
        assert_eq!(j[0].0, Value::Int(0));
        let Value::Tensor(t) = &j[0].1 else {
            panic!("not a tensor");
        };
        assert_eq!(t.dtype, Dtype::F16);
        assert_eq!(t.size, vec![2, 2]);
        assert_eq!(t.stride, vec![2, 1]);
        assert_eq!(t.storage_numel, 4);
    }

    #[test]
    fn unknown_opcodes_and_protocols_are_refused() {
        assert!(unpickle(&[0x80, 4, b'N', b'.']).is_err());
        let e = unpickle(&[0x80, 2, b'b', b'.']).unwrap_err().to_string();
        assert!(e.contains("0x62"), "{e}");
        assert!(unpickle(&[0x80, 2, b'N']).is_err());
    }

    #[test]
    fn negative_long1_integers_read() {
        assert_eq!(unpickle(&[0x8a, 1, 0xff, b'.']).unwrap(), Value::Int(-1));
        assert_eq!(
            unpickle(&[0x8a, 2, 0x00, 0x01, b'.']).unwrap(),
            Value::Int(256)
        );
    }

    #[test]
    fn a_reduce_of_anything_else_is_refused() {
        let mut p = b"cos\nsystem\n".to_vec();
        p.extend_from_slice(&[b'X', 2, 0, 0, 0, b'l', b's', 0x85, b'R', b'.']);
        let e = unpickle(&p).unwrap_err().to_string();
        assert!(e.contains("os.system"), "{e}");
    }
}
