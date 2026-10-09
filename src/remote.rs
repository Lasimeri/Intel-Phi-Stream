//! The model on another machine (`--remote`): a llama-server (llama.cpp's,
//! or llama.phi's on the GPU rack) holds the live sequence in one of its
//! slots and samples it; this process keeps the token history, the
//! vocabulary (read from the model's GGUF metadata, `--remote-vocab`) and
//! the engine. One sequence: no readings beside it, no forks, no logits.
//! HTTP/1.1 over `std::net`, the server-sent events of `/completion` with
//! `return_tokens`. See remote.md.

use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Value};

use crate::llm::Sampling;

/// How long one read of the socket waits before the caller looks at its
/// deadline again.
const POLL: Duration = Duration::from_secs(1);
/// A short answer (`/props`, `/tokenize`): this long at most.
const SHORT: Duration = Duration::from_secs(60);
/// The first token of a stream comes after the server has read the prompt:
/// a whole context of 131072 tokens at the rack's 350 tokens a second is
/// six minutes (llama.phi's docs/phi/results.md), so half an hour.
const FIRST_TOKEN: Duration = Duration::from_secs(1800);
/// Between two tokens of a stream (0.15 s on the rack): five minutes.
const NEXT_TOKEN: Duration = Duration::from_secs(300);
/// A server that does not answer (restarting, loading its model: about
/// five minutes on the rack) is asked again every 5 s for this long.
const RETRY_FOR: Duration = Duration::from_secs(900);
/// DRY's sequence breakers, as `llm.rs` gives them to its own chain (the
/// newline left out).
const DRY_BREAKERS: [&str; 3] = [":", "\"", "*"];

/// Text the vocabulary is checked with against the server's own
/// tokenizer: control tokens, the tool call's tags, words in several
/// scripts, an emoji (bytes split over tokens), runs of spaces, code.
const SAMPLE: &str = "<|im_start|>system\nThe stream's vocabulary check: plain words, caf\u{e9}, na\u{ef}ve, \u{6771}\u{4eac}, \u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac}, an emoji \u{1F642}, tabs\tand  double  spaces, code `fn main() { let x = 0x1F; }` 3.14159\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=run>\n<parameter=command>\nls -la ~/src | grep -c '\\.rs$'\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|endoftext|>";

// ---------------------------------------------------------------- GGUF

/// A cursor over a GGUF file's bytes.
struct Cur<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .context("the file ends inside its metadata (fetch more of its head)")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<&'a [u8]> {
        let n = self.u64()? as usize;
        self.take(n)
    }
}

/// Bytes of one value of a GGUF scalar type (ggml's gguf.h `gguf_type`).
fn scalar_size(ty: u32) -> Option<usize> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// An integer of a GGUF integer type, little endian.
fn int_of(ty: u32, b: &[u8]) -> Option<i64> {
    Some(match ty {
        0 => b[0] as i64,
        1 => b[0] as i8 as i64,
        2 => u16::from_le_bytes([b[0], b[1]]) as i64,
        3 => i16::from_le_bytes([b[0], b[1]]) as i64,
        4 => u32::from_le_bytes(b[..4].try_into().ok()?) as i64,
        5 => i32::from_le_bytes(b[..4].try_into().ok()?) as i64,
        10 => u64::from_le_bytes(b[..8].try_into().ok()?) as i64,
        11 => i64::from_le_bytes(b[..8].try_into().ok()?),
        _ => return None,
    })
}

/// One metadata value as far as this module reads values.
enum Val<'a> {
    Int(i64),
    Str(&'a [u8]),
    Strs(Vec<&'a [u8]>),
    Ints(Vec<i64>),
    Other,
}

/// A value of type `ty`; arrays read only when `want`, skipped otherwise.
fn value<'a>(c: &mut Cur<'a>, ty: u32, want: bool) -> Result<Val<'a>> {
    if let Some(n) = scalar_size(ty) {
        let b = c.take(n)?;
        return Ok(int_of(ty, b).map_or(Val::Other, Val::Int));
    }
    match ty {
        8 => Ok(Val::Str(c.str()?)),
        9 => {
            let et = c.u32()?;
            let n = c.u64()? as usize;
            if et == 8 {
                let mut v = Vec::with_capacity(if want { n } else { 0 });
                for _ in 0..n {
                    let s = c.str()?;
                    if want {
                        v.push(s);
                    }
                }
                return Ok(if want { Val::Strs(v) } else { Val::Other });
            }
            if let Some(sz) = scalar_size(et) {
                let b = c.take(n.checked_mul(sz).context("an array too large")?)?;
                if !want {
                    return Ok(Val::Other);
                }
                return Ok(Val::Ints(
                    b.chunks(sz).map(|x| int_of(et, x).unwrap_or(0)).collect(),
                ));
            }
            for _ in 0..n {
                value(c, et, false)?;
            }
            Ok(Val::Other)
        }
        _ => bail!("an unknown GGUF value type {ty}"),
    }
}

// ---------------------------------------------------------- vocabulary

/// The vocabulary of a byte-level BPE model (`tokenizer.ggml.model` gpt2),
/// read from its GGUF metadata: each token's bytes as llama.cpp's
/// `token_to_piece` gives them (llama-vocab.cpp), its type, the
/// end-of-generation set.
pub struct Vocab {
    /// The token's bytes: decoded from GPT-2's byte-to-character mapping
    /// for a normal token, the text itself for any other.
    pieces: Vec<Vec<u8>>,
    /// The GGUF token type: 1 normal, 2 unknown, 3 control, 4 user
    /// defined, 5 unused, 6 byte.
    kind: Vec<u8>,
    /// The tokens that end a generation (llama.cpp's `special_eog_ids`).
    pub eog: Vec<i32>,
    /// The end of a turn (`<|im_end|>` when the vocabulary has it, else
    /// the end of sequence).
    pub eot: i32,
    /// Decoder blocks (`ARCH.block_count`).
    pub n_layer: i32,
    /// The model's name (`general.name`).
    pub name: String,
}

/// llama.cpp's end-of-generation tokens found by their text (llama-vocab.cpp,
/// `load`: the eot list, the eog list, the fill-in-the-middle pad, repo and
/// separator tokens).
const EOG_TEXT: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
];

/// GPT-2's byte-level mapping, reversed: the character each byte was
/// written as, back to the byte (llama.cpp's unicode.cpp `unicode_utf8_to_byte`).
fn byte_of() -> std::collections::HashMap<char, u8> {
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars = bytes.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    bytes
        .iter()
        .zip(&chars)
        .filter_map(|(&b, &c)| char::from_u32(c).map(|c| (c, b as u8)))
        .collect()
}

/// A normal token's text as bytes (llama.cpp's `llama_decode_text`).
fn decode_text(text: &[u8], map: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    let s = String::from_utf8_lossy(text);
    let mut out = Vec::with_capacity(text.len());
    for ch in s.chars() {
        match map.get(&ch) {
            Some(&b) => out.push(b),
            None => {
                let mut tmp = [0u8; 4];
                out.extend_from_slice(b"[UNK_BYTE_0x");
                for b in ch.encode_utf8(&mut tmp).bytes() {
                    out.extend_from_slice(format!("{b:02x}").as_bytes());
                }
                out.extend_from_slice(text);
                out.push(b']');
            }
        }
    }
    out
}

impl Vocab {
    /// Read the metadata of a GGUF file: the whole model, its metadata
    /// alone, or the head of it (the tensor table after the metadata is
    /// never read).
    pub fn load(path: &Path) -> Result<Self> {
        let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&b).with_context(|| format!("reading the metadata of {}", path.display()))
    }

    fn from_bytes(b: &[u8]) -> Result<Self> {
        let mut c = Cur { b, at: 0 };
        if c.take(4)? != b"GGUF" {
            bail!("not a GGUF file");
        }
        let version = c.u32()?;
        if version < 2 {
            bail!("GGUF version {version}: 2 or later is read");
        }
        let _tensors = c.u64()?;
        let n_kv = c.u64()?;
        let mut arch = String::new();
        let mut name = String::new();
        let mut model = String::new();
        let mut tokens: Vec<&[u8]> = Vec::new();
        let mut kinds: Vec<i64> = Vec::new();
        let mut eos = None;
        let mut eot_id = None;
        let mut blocks: Vec<(String, i64)> = Vec::new();
        for _ in 0..n_kv {
            let key = String::from_utf8_lossy(c.str()?).into_owned();
            let ty = c.u32()?;
            let want = key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.token_type";
            let v = value(&mut c, ty, want)?;
            match (key.as_str(), v) {
                ("general.architecture", Val::Str(s)) => arch = String::from_utf8_lossy(s).into(),
                ("general.name", Val::Str(s)) => name = String::from_utf8_lossy(s).into(),
                ("tokenizer.ggml.model", Val::Str(s)) => model = String::from_utf8_lossy(s).into(),
                ("tokenizer.ggml.tokens", Val::Strs(v)) => tokens = v,
                ("tokenizer.ggml.token_type", Val::Ints(v)) => kinds = v,
                ("tokenizer.ggml.eos_token_id", Val::Int(i)) => eos = Some(i as i32),
                ("tokenizer.ggml.eot_token_id", Val::Int(i)) => eot_id = Some(i as i32),
                (k, Val::Int(i)) if k.ends_with(".block_count") => blocks.push((k.to_string(), i)),
                _ => {}
            }
        }
        if model != "gpt2" {
            bail!("tokenizer.ggml.model is {model:?}: only byte-level BPE (gpt2) vocabularies are read here");
        }
        if tokens.is_empty() || kinds.len() != tokens.len() {
            bail!(
                "no vocabulary ({} tokens, {} token types)",
                tokens.len(),
                kinds.len()
            );
        }
        let map = byte_of();
        let mut kind: Vec<u8> = kinds.iter().map(|&k| k.clamp(0, 255) as u8).collect();
        let pieces: Vec<Vec<u8>> = tokens
            .iter()
            .zip(&kind)
            .map(|(t, &k)| {
                if k == 1 {
                    decode_text(t, &map)
                } else {
                    t.to_vec()
                }
            })
            .collect();
        let id_of = |text: &str| {
            tokens
                .iter()
                .position(|t| *t == text.as_bytes())
                .map(|i| i as i32)
        };
        let mut eog: Vec<i32> = EOG_TEXT.iter().filter_map(|t| id_of(t)).collect();
        // llama.cpp makes each of these a control token (never shown as
        // text without `special`).
        for &t in &eog {
            kind[t as usize] = 3;
        }
        for t in [eos, eot_id].into_iter().flatten() {
            if !eog.contains(&t) && (t as usize) < tokens.len() {
                eog.push(t);
            }
        }
        eog.sort_unstable();
        let eot = eot_id
            .or_else(|| id_of("<|im_end|>"))
            .or(eos)
            .context("no end-of-turn or end-of-sequence token")?;
        let n_layer = blocks
            .iter()
            .find(|(k, _)| *k == format!("{arch}.block_count"))
            .map_or(0, |b| b.1 as i32);
        Ok(Self {
            pieces,
            kind,
            eog,
            eot,
            n_layer,
            name,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// A token's bytes, as llama.cpp's `token_to_piece` with lstrip 0: a
    /// control or unknown token only when `special`, a user-defined one
    /// always, a normal one decoded, an unused or byte token nothing.
    pub fn piece(&self, t: i32, special: bool, out: &mut Vec<u8>) {
        let Some(&k) = self.kind.get(t as usize) else {
            return;
        };
        match k {
            2 | 3 if special => out.extend_from_slice(&self.pieces[t as usize]),
            1 | 4 => out.extend_from_slice(&self.pieces[t as usize]),
            _ => {}
        }
    }

    /// The tokens whose bytes (as `piece` without `special`) hold `needle`.
    pub fn containing(&self, needle: &[u8]) -> Vec<i32> {
        let mut out = Vec::new();
        let mut buf = Vec::new();
        for t in 0..self.len() as i32 {
            buf.clear();
            self.piece(t, false, &mut buf);
            if buf.windows(needle.len()).any(|w| w == needle) {
                out.push(t);
            }
        }
        out
    }
}

// --------------------------------------------------------------- HTTP

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

/// A server's error answer in a line.
fn err_text(v: &Value) -> String {
    v.pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| v.to_string().chars().take(300).collect())
}

/// A response body read off the socket as it comes: raw bytes, then the
/// chunked framing (or the length) taken off. Reads wait `POLL` at most,
/// so a caller keeps its own deadline and loses nothing on a timeout.
struct Body {
    /// The connection (a test gives it bytes of its own).
    src: Box<dyn Read + Send>,
    raw: Vec<u8>,
    chunked: bool,
    /// `Content-Length` still to come.
    left: Option<usize>,
    chunk_left: usize,
    /// A chunk's closing CRLF is due before the next size line.
    after_chunk: bool,
    /// The last chunk (or the whole length) has come.
    done: bool,
    /// The server closed the connection.
    closed: bool,
    /// When bytes last came.
    last: Instant,
}

impl Body {
    /// More bytes from the socket: Ok(false) when none came within `POLL`.
    fn fill(&mut self) -> Result<bool> {
        let mut tmp = [0u8; 16384];
        match self.src.read(&mut tmp) {
            Ok(0) => {
                self.closed = true;
                Ok(true)
            }
            Ok(n) => {
                self.raw.extend_from_slice(&tmp[..n]);
                self.last = Instant::now();
                Ok(true)
            }
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                ) =>
            {
                Ok(false)
            }
            Err(e) => Err(e).context("reading from the server"),
        }
    }

    /// The status line and headers, by `deadline`.
    fn head(&mut self, deadline: Instant) -> Result<u16> {
        loop {
            if let Some(end) = find(&self.raw, b"\r\n\r\n") {
                let text = String::from_utf8_lossy(&self.raw[..end]).into_owned();
                self.raw.drain(..end + 4);
                let mut lines = text.split("\r\n");
                let status = lines
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|c| c.parse().ok())
                    .context("no HTTP status line")?;
                for l in lines {
                    if let Some((k, v)) = l.split_once(':') {
                        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
                        if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                            self.chunked = true;
                        } else if k == "content-length" {
                            self.left = v.parse().ok();
                        }
                    }
                }
                return Ok(status);
            }
            if self.closed {
                bail!("the server closed the connection before answering");
            }
            if !self.fill()? && Instant::now() > deadline {
                bail!("no answer from the server in time");
            }
        }
    }

    /// The body bytes that have come, framing taken off, appended to `out`.
    fn take(&mut self, out: &mut Vec<u8>) -> Result<()> {
        if !self.chunked {
            let n = self.left.map_or(self.raw.len(), |l| l.min(self.raw.len()));
            out.extend(self.raw.drain(..n));
            if let Some(l) = self.left.as_mut() {
                *l -= n;
                if *l == 0 {
                    self.done = true;
                }
            }
            if self.closed && self.raw.is_empty() {
                self.done = true;
            }
            return Ok(());
        }
        loop {
            if self.after_chunk {
                if self.raw.len() < 2 {
                    break;
                }
                if &self.raw[..2] != b"\r\n" {
                    bail!("a chunk not closed by CRLF");
                }
                self.raw.drain(..2);
                self.after_chunk = false;
            }
            if self.chunk_left == 0 {
                if self.done {
                    break;
                }
                let Some(e) = find(&self.raw, b"\r\n") else {
                    break;
                };
                let line = String::from_utf8_lossy(&self.raw[..e]).into_owned();
                self.raw.drain(..e + 2);
                let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
                    .with_context(|| format!("a chunk size {line:?}"))?;
                if size == 0 {
                    self.done = true;
                    break;
                }
                self.chunk_left = size;
            }
            let n = self.chunk_left.min(self.raw.len());
            if n == 0 {
                break;
            }
            out.extend(self.raw.drain(..n));
            self.chunk_left -= n;
            if self.chunk_left == 0 {
                self.after_chunk = true;
            }
        }
        Ok(())
    }

    /// The whole of a short answer, by `deadline`.
    fn read_all(mut self, deadline: Instant) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            self.take(&mut out)?;
            if self.done {
                return Ok(out);
            }
            if self.closed {
                bail!("the server closed the connection inside its answer");
            }
            if !self.fill()? && Instant::now() > deadline {
                bail!("the server's answer did not finish in time");
            }
        }
    }
}

/// A server's address: `http://HOST:PORT`.
#[derive(Clone, Debug)]
pub struct Server {
    host: String,
    port: u16,
}

impl Server {
    pub fn parse(url: &str) -> Result<Self> {
        let rest = url
            .trim()
            .strip_prefix("http://")
            .with_context(|| format!("{url}: an address is http://HOST:PORT"))?
            .trim_end_matches('/');
        if rest.contains('/') {
            bail!("{url}: an address is http://HOST:PORT, with no path");
        }
        let (host, port) = match rest.rsplit_once(':') {
            Some((h, p)) => (h, p.parse().with_context(|| format!("{url}: the port"))?),
            None => (rest, 80),
        };
        if host.is_empty() {
            bail!("{url}: no host");
        }
        Ok(Self {
            host: host.to_string(),
            port,
        })
    }

    pub fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// One request (its own connection); the status and the body to read,
    /// the head awaited until `deadline`.
    fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        deadline: Instant,
    ) -> Result<(u16, Body)> {
        let addr = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .with_context(|| format!("resolving {}", self.host))?
            .next()
            .with_context(|| format!("no address for {}", self.host))?;
        let mut sock = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .with_context(|| format!("connecting to {}", self.url()))?;
        sock.set_nodelay(true).ok();
        let data = body.map(Value::to_string).unwrap_or_default();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nAccept: */*\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            self.host,
            self.port,
            data.len()
        );
        sock.write_all(head.as_bytes())?;
        sock.write_all(data.as_bytes())?;
        sock.set_read_timeout(Some(POLL))?;
        let mut b = Body {
            src: Box::new(sock),
            raw: Vec::new(),
            chunked: false,
            left: None,
            chunk_left: 0,
            after_chunk: false,
            done: false,
            closed: false,
            last: Instant::now(),
        };
        let status = b.head(deadline)?;
        Ok((status, b))
    }

    /// A request answered within `within`: its status and its body's bytes,
    /// whatever the status (the caller reads both).
    fn exchange(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        within: Duration,
    ) -> Result<(u16, Vec<u8>)> {
        let deadline = Instant::now() + within;
        let (status, b) = self.send(method, path, body, deadline)?;
        Ok((status, b.read_all(deadline)?))
    }

    /// A short JSON request and its JSON answer; an error for any status
    /// but 200, with the server's message.
    pub fn json(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
        let deadline = Instant::now() + SHORT;
        let (status, b) = self.send(method, path, body, deadline)?;
        let bytes = b.read_all(deadline)?;
        let v: Value = serde_json::from_slice(&bytes).with_context(|| {
            format!("{path}: an answer that is not JSON ({} bytes)", bytes.len())
        })?;
        if status != 200 {
            bail!("{path}: HTTP {status}: {}", err_text(&v));
        }
        Ok(v)
    }
}

// ----------------------------------------------------------- prefetch

/// How long asking for a prefetch, or for its state, may hold the engine's
/// cycle: the server answers both at once (it queues the prompt, or looks
/// the id up), so a longer wait is a busy server, asked again later.
const PREFETCH_ASK: Duration = Duration::from_secs(2);

/// The server's path for prompts read by its prefill engine without a slot
/// (llama.phi's decode server; llama.cpp's own server has none).
const PREFETCH_PATH: &str = "/phi/prefetch";

/// What asking for a prefetch gave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefetchStart {
    /// Accepted: the id its state is asked by.
    Id(u64),
    /// The server has no prefetch (HTTP 404 on its path): llama.cpp's own
    /// server, or llama.phi before it had one.
    Unsupported,
}

/// A prefetch's state, as the server gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prefetch {
    /// Queued or being read.
    Waiting,
    /// Read: the state is in the decode server's prompt cache. The
    /// contract names two such states, `ready` and `done`; both are taken
    /// as read until the server's final format says otherwise.
    Ready,
    /// The server gave it up, with its words.
    Failed(String),
    /// A state this side does not know (the format may change): waited on
    /// as `Waiting`, and named so the caller can say so.
    Unknown(String),
}

/// The answer to `POST /phi/prefetch` (`{"id": N}`), by its status and
/// body.
pub fn prefetch_started(status: u16, body: &[u8]) -> Result<PrefetchStart> {
    if status == 404 {
        return Ok(PrefetchStart::Unsupported);
    }
    let v: Value = serde_json::from_slice(body).with_context(|| {
        format!(
            "{PREFETCH_PATH}: an answer that is not JSON (HTTP {status}, {} bytes)",
            body.len()
        )
    })?;
    if status != 200 {
        bail!("{PREFETCH_PATH}: HTTP {status}: {}", err_text(&v));
    }
    let id = match v.get("id") {
        Some(Value::Number(n)) => n.as_u64(),
        Some(Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    };
    id.map(PrefetchStart::Id)
        .with_context(|| format!("{PREFETCH_PATH}: no id in {}", err_text(&v)))
}

/// The answer to `GET /phi/prefetch?id=N`, by its status and body: the
/// state as a string field (`state`, else `status`) or the body itself a
/// string. HTTP 5xx is an error (the caller asks again); another failure
/// status is the prefetch failed, with the server's message.
pub fn prefetch_state(status: u16, body: &[u8]) -> Result<Prefetch> {
    let v: Value = serde_json::from_slice(body).with_context(|| {
        format!(
            "{PREFETCH_PATH}: an answer that is not JSON (HTTP {status}, {} bytes)",
            body.len()
        )
    })?;
    if status >= 500 {
        bail!("{PREFETCH_PATH}: HTTP {status}: {}", err_text(&v));
    }
    if status != 200 {
        return Ok(Prefetch::Failed(format!("HTTP {status}: {}", err_text(&v))));
    }
    let state = match &v {
        Value::String(s) => s.clone(),
        _ => v
            .get("state")
            .or_else(|| v.get("status"))
            .and_then(Value::as_str)
            .with_context(|| format!("{PREFETCH_PATH}: no state in {}", err_text(&v)))?
            .to_string(),
    };
    Ok(match state.trim().to_ascii_lowercase().as_str() {
        "waiting" | "queued" | "pending" | "running" | "reading" => Prefetch::Waiting,
        "ready" | "done" => Prefetch::Ready,
        "failed" | "error" => Prefetch::Failed(
            v.get("error")
                .or_else(|| v.get("message"))
                .map(|e| match e {
                    Value::String(s) => s.clone(),
                    other => err_text(&json!({ "error": other })),
                })
                .unwrap_or_else(|| "failed, no reason given".to_string()),
        ),
        _ => Prefetch::Unknown(state),
    })
}

// ------------------------------------------------------------- stream

/// One `/completion` stream: the tokens the server samples after the prompt
/// it was opened with, handed out one at a time while the engine places
/// exactly them.
struct Stream {
    body: Body,
    /// Body bytes not yet split into events.
    sse: Vec<u8>,
    /// Tokens in the prompt it was opened with.
    base: usize,
    /// Tokens handed to the engine.
    given: Vec<i32>,
    /// Tokens received and not yet handed.
    queue: VecDeque<i32>,
    /// The server's `stop_type`, once it stopped.
    stop: Option<String>,
    /// The server's reading of the prompt, (done, total) tokens, from its
    /// `prompt_progress` events (`return_progress`); none before the first.
    progress: Option<(usize, usize)>,
}

impl Stream {
    /// The engine placed exactly what this stream gave since it opened: the
    /// server's next token follows what this process holds.
    fn continues(&self, hist: &[i32]) -> bool {
        hist.len() == self.base + self.given.len() && hist[self.base..] == self.given[..]
    }

    /// The events that have come: their tokens queued, a stop noted.
    fn events(&mut self) -> Result<()> {
        self.body.take(&mut self.sse)?;
        self.sse.retain(|&c| c != b'\r');
        while let Some(e) = find(&self.sse, b"\n\n") {
            let ev: Vec<u8> = self.sse.drain(..e + 2).collect();
            for line in ev.split(|&c| c == b'\n') {
                let Some(d) = line.strip_prefix(b"data:") else {
                    continue;
                };
                let v: Value =
                    serde_json::from_slice(d).context("a stream event that is not JSON")?;
                if v.get("error").is_some() {
                    bail!("the server: {}", err_text(&v));
                }
                // A progress event is a partial result made from an empty
                // token: llama-server sends `"tokens":[0]` with it
                // (`send_partial_response(slot, {}, true)`), and 0 is `!`.
                // Taken as the model's, one `!` a progress event opened every
                // turn's thinking, and the server, which never held them,
                // read each turn again at the next request (10-08 to 10-09).
                let ts = match v.get("prompt_progress") {
                    Some(_) => None,
                    None => v.get("tokens").and_then(Value::as_array),
                };
                for t in ts.into_iter().flatten() {
                    self.queue
                        .push_back(t.as_i64().context("a token id")? as i32);
                }
                if let Some(p) = v.get("prompt_progress") {
                    let n = |k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
                    // `processed` is the slot's whole prompt so far, the
                    // cached part included (`slot.prompt.tokens.size()`).
                    self.progress = Some((n("processed"), n("total")));
                }
                if v.get("stop").and_then(Value::as_bool) == Some(true) {
                    let why = v
                        .get("stop_type")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    self.stop = Some(why.to_string());
                }
            }
        }
        Ok(())
    }

    /// The next token; none once the server stopped and every token it sent
    /// was handed out (the tests' form: the engine always pumps).
    #[cfg(test)]
    fn next(&mut self) -> Result<Option<i32>> {
        self.next_with(&mut |_| Ok(true))
    }

    /// `next` with a pump called on every wait for bytes (once per `POLL`):
    /// Ok(false) cancels with an `Aborted` error, so what waits behind a
    /// slow first token is answered during the wait, not queued after it
    /// (`Remote::sample_pumped`).
    fn next_with(
        &mut self,
        pump: &mut dyn FnMut(Option<(usize, usize)>) -> Result<bool>,
    ) -> Result<Option<i32>> {
        loop {
            if let Some(t) = self.queue.pop_front() {
                self.given.push(t);
                return Ok(Some(t));
            }
            if self.stop.is_some() {
                return Ok(None);
            }
            self.events()?;
            if !self.queue.is_empty() || self.stop.is_some() {
                continue;
            }
            if self.body.done || self.body.closed {
                bail!("the server ended the stream without saying it stopped");
            }
            if !pump(self.progress)? {
                return Err(
                    anyhow::Error::new(Aborted).context("the wait was cancelled by a command")
                );
            }
            let wait = if self.given.is_empty() {
                FIRST_TOKEN
            } else {
                NEXT_TOKEN
            };
            if !self.body.fill()? && self.body.last.elapsed() > wait {
                bail!("no token from the server in {} s", wait.as_secs());
            }
        }
    }
}

/// The engine cancelled a wait (`sample_pumped`'s pump said stop): a mark
/// on the error, so a caller can tell a clean stop from a server failure.
#[derive(Debug)]
pub struct Aborted;

impl std::fmt::Display for Aborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled by a command")
    }
}

impl std::error::Error for Aborted {}

/// Why a stream could not be opened.
enum Open {
    /// Worth asking again (no connection, a server loading or busy).
    Retry(anyhow::Error),
    /// Not (the request itself was refused).
    Fatal(anyhow::Error),
}

// ------------------------------------------------------------- remote

/// The live sequence held by a server's slot.
pub struct Remote {
    server: Server,
    pub vocab: Vocab,
    /// The slot's context, tokens (the server's `n_ctx` per slot).
    pub n_ctx: u32,
    /// The model file the server loaded.
    pub model: String,
    slot: i32,
    /// The sequence: every token placed, as the slot holds it (the last
    /// one waits for the next stream to decode it).
    hist: Vec<i32>,
    stream: Option<Stream>,
    sampling: Sampling,
    /// Tokens the server must never sample (a logit bias of minus infinity).
    never: Vec<i32>,
    /// Streams opened, and tokens they gave.
    pub streams: u64,
    pub tokens: u64,
}

/// Where the vocabulary of a server's model is kept by default:
/// `~/.local/share/phi-stream/remote/STEM.vocab.gguf`, STEM the model
/// file's name without `.gguf`.
pub fn default_vocab(model: &str) -> PathBuf {
    let stem = Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(format!(".local/share/phi-stream/remote/{stem}.vocab.gguf"))
}

impl Remote {
    /// Ask the server what it holds, read the vocabulary (`vocab`, or the
    /// default for its model) and check it against the server's tokenizer.
    pub fn connect(url: &str, vocab: Option<&str>, slot: i32, sampling: &Sampling) -> Result<Self> {
        let server = Server::parse(url)?;
        let props = server
            .json("GET", "/props", None)
            .with_context(|| format!("{}: no llama-server answers there", server.url()))?;
        let model = props
            .get("model_path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let n_ctx = props
            .pointer("/default_generation_settings/n_ctx")
            .and_then(Value::as_u64)
            .context("/props gives no n_ctx")? as u32;
        let slots = props
            .get("total_slots")
            .and_then(Value::as_i64)
            .unwrap_or(1);
        if slot < 0 || slot as i64 >= slots {
            if slots <= 0 {
                bail!("--remote-slot {slot}: the server says it has no slots");
            }
            bail!(
                "--remote-slot {slot}: the server has {slots} slot(s), 0 to {}",
                slots - 1
            );
        }
        let path = vocab
            .map(PathBuf::from)
            .unwrap_or_else(|| default_vocab(&model));
        let vocab = Vocab::load(&path).with_context(|| {
            format!(
                "the vocabulary of {model}: fetch it with scripts/remote-vocab.sh {}",
                server.url()
            )
        })?;
        let r = Self {
            server,
            vocab,
            n_ctx,
            model,
            slot,
            hist: Vec::new(),
            stream: None,
            sampling: sampling.clone(),
            never: Vec::new(),
            streams: 0,
            tokens: 0,
        };
        r.check_vocab(&path)?;
        Ok(r)
    }

    pub fn url(&self) -> String {
        self.server.url()
    }

    /// The vocabulary agrees with the server's: the same count (`/v1/models`
    /// when it says) and, token by token, the same ids and bytes for
    /// `SAMPLE` as the server's `/tokenize` gives with its pieces.
    fn check_vocab(&self, path: &Path) -> Result<()> {
        let where_ = || format!("{} against {}", path.display(), self.model);
        if let Ok(m) = self.server.json("GET", "/v1/models", None) {
            if let Some(n) = m.pointer("/data/0/meta/n_vocab").and_then(Value::as_u64) {
                if n as usize != self.vocab.len() {
                    bail!(
                        "{}: {} tokens here, {n} on the server: fetch the server's (scripts/remote-vocab.sh)",
                        where_(),
                        self.vocab.len()
                    );
                }
            }
        }
        let v = self.server.json(
            "POST",
            "/tokenize",
            Some(&json!({"content": SAMPLE, "add_special": false, "parse_special": true, "with_pieces": true})),
        )?;
        let toks = v
            .get("tokens")
            .and_then(Value::as_array)
            .context("/tokenize gives no tokens")?;
        for (i, t) in toks.iter().enumerate() {
            let id = t
                .get("id")
                .and_then(Value::as_i64)
                .context("/tokenize: a token without its id")? as i32;
            let theirs: Vec<u8> = match t.get("piece") {
                Some(Value::String(s)) => s.as_bytes().to_vec(),
                Some(Value::Array(a)) => a.iter().map(|b| b.as_u64().unwrap_or(0) as u8).collect(),
                _ => bail!("/tokenize: a token without its piece"),
            };
            let mut ours = Vec::new();
            self.vocab.piece(id, true, &mut ours);
            if ours != theirs {
                bail!(
                    "{}: token {i} of the check ({id}) is {:?} here and {:?} on the server: fetch the server's vocabulary (scripts/remote-vocab.sh)",
                    where_(),
                    String::from_utf8_lossy(&ours),
                    String::from_utf8_lossy(&theirs)
                );
            }
        }
        Ok(())
    }

    /// Text to tokens by the server's tokenizer (`special`: the template's
    /// control tokens parsed).
    pub fn tokenize(&self, text: &str, special: bool) -> Result<Vec<i32>> {
        let v = self.server.json(
            "POST",
            "/tokenize",
            Some(&json!({"content": text, "add_special": false, "parse_special": special})),
        )?;
        v.get("tokens")
            .and_then(Value::as_array)
            .context("/tokenize gives no tokens")?
            .iter()
            .map(|t| {
                t.as_i64()
                    .map(|i| i as i32)
                    .context("/tokenize: a token that is not an id")
            })
            .collect()
    }

    /// Ask the server to read `tokens` on its prefill engine, beside the
    /// slot's stream (llama.phi's `POST /phi/prefetch`): the finished state
    /// lands in the decode server's prompt cache, and a later stream whose
    /// prompt begins with `tokens` starts from it. Answered at once.
    pub fn prefetch(&self, tokens: &[i32]) -> Result<PrefetchStart> {
        let (status, body) = self.server.exchange(
            "POST",
            PREFETCH_PATH,
            Some(&json!({ "tokens": tokens })),
            PREFETCH_ASK,
        )?;
        prefetch_started(status, &body)
    }

    /// The state of the prefetch `id` (`GET /phi/prefetch?id=N`).
    pub fn prefetch_state(&self, id: u64) -> Result<Prefetch> {
        let (status, body) = self.server.exchange(
            "GET",
            &format!("{PREFETCH_PATH}?id={id}"),
            None,
            PREFETCH_ASK,
        )?;
        prefetch_state(status, &body)
    }

    /// Place `tokens` at `pos0` of sequence `seq` (only 0 exists here):
    /// what was past `pos0` is dropped first.
    pub fn place(&mut self, seq: i32, tokens: &[i32], pos0: i32) -> Result<()> {
        if seq != 0 {
            bail!("the remote model holds one sequence and sequence {seq} was asked for: readings, checks, the second chain and the guide need the model in this process");
        }
        let p = pos0.max(0) as usize;
        if p > self.hist.len() {
            bail!(
                "a decode at position {p}, past the sequence's end at {}",
                self.hist.len()
            );
        }
        // Room first, then the change: truncating before the check left a
        // destroyed sequence behind when the place was refused.
        if p + tokens.len() >= self.n_ctx as usize {
            bail!(
                "the sequence ({p} + {} tokens) fills the server's context of {}",
                tokens.len(),
                self.n_ctx
            );
        }
        self.hist.truncate(p);
        self.hist.extend_from_slice(tokens);
        Ok(())
    }

    /// Drop `seq`'s positions from `p0` on (`p1` -1); a hole in the middle
    /// is refused, as a recurrent model refuses it.
    pub fn remove(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
        if seq != 0 {
            return true;
        }
        let p0 = p0.max(0) as usize;
        if p1 >= 0 && (p1 as usize) < self.hist.len() && p0 < p1 as usize {
            return false;
        }
        self.hist.truncate(p0.min(self.hist.len()));
        true
    }

    pub fn pos_max(&self, seq: i32) -> i32 {
        if seq == 0 {
            self.hist.len() as i32 - 1
        } else {
            -1
        }
    }

    pub fn clear(&mut self) {
        self.hist.clear();
        self.stream = None;
    }

    /// New settings for the next stream (the open one is dropped).
    pub fn set_sampling(&mut self, s: &Sampling, never: &[i32]) {
        self.sampling = s.clone();
        self.never = never.to_vec();
        self.stream = None;
    }

    /// The token after the sequence, sampled by the server. The open
    /// stream goes on while the engine places what it gave; anything else
    /// placed drops it (the server stops generating when the connection
    /// closes) and a new request continues from the whole sequence, the
    /// server reusing its slot's cache up to where the two differ.
    ///
    /// The pump is called once per round of the retry loop and, through
    /// the stream, once per wait for bytes (`Stream::next_with`), with the
    /// server's reading of the prompt when it has said; Ok(false) cancels
    /// with an `Aborted` error, which is how a command answered during a
    /// long first-token wait stops this (the engine's `sample_live`).
    pub fn sample_pumped(
        &mut self,
        pump: &mut dyn FnMut(Option<(usize, usize)>) -> Result<bool>,
    ) -> Result<i32> {
        if self.hist.is_empty() {
            bail!("nothing to continue: the sequence is empty");
        }
        let give_up = Instant::now() + RETRY_FOR;
        let mut told: Option<Instant> = None;
        let mut fresh = false;
        loop {
            if !pump(None)? {
                return Err(
                    anyhow::Error::new(Aborted).context("the wait was cancelled by a command")
                );
            }
            match self.stream.as_mut() {
                Some(s) if s.continues(&self.hist) => match s.next_with(pump) {
                    Ok(Some(t)) => {
                        self.tokens += 1;
                        return Ok(t);
                    }
                    Ok(None) => {
                        let why = s.stop.clone().unwrap_or_default();
                        self.stream = None;
                        if fresh {
                            bail!("the server stopped ({why}) without a token");
                        }
                    }
                    Err(e) if e.downcast_ref::<Aborted>().is_some() => {
                        // The pump cancelled the wait: a clean stop, not a
                        // broken stream to ask again for.
                        self.stream = None;
                        return Err(e);
                    }
                    Err(e) => {
                        eprintln!("phi-stream: remote: the stream broke ({e:#}); asking again");
                        let empty = s.given.is_empty();
                        self.stream = None;
                        if empty {
                            // Opened, then broken before a token: within the
                            // same window as a server that does not answer
                            // (a restarting server does this several times;
                            // seen at the rack's model switch, 2026-10-08),
                            // and paced like it, never at once and forever.
                            if Instant::now() > give_up {
                                return Err(e.context(format!(
                                    "{} broke every stream before a token for {} minutes",
                                    self.server.url(),
                                    RETRY_FOR.as_secs() / 60
                                )));
                            }
                            std::thread::sleep(Duration::from_secs(5));
                        }
                    }
                },
                _ => self.stream = None,
            }
            match self.open() {
                Ok(s) => {
                    self.stream = Some(s);
                    fresh = true;
                }
                Err(Open::Fatal(e)) => return Err(e),
                Err(Open::Retry(e)) => {
                    if Instant::now() > give_up {
                        return Err(e.context(format!(
                            "{} did not answer for {} minutes",
                            self.server.url(),
                            RETRY_FOR.as_secs() / 60
                        )));
                    }
                    if told.is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
                        eprintln!(
                            "phi-stream: remote: {e:#}; asking again every 5 s for up to {} minutes",
                            RETRY_FOR.as_secs() / 60
                        );
                        told = Some(Instant::now());
                    }
                    std::thread::sleep(Duration::from_secs(5));
                }
            }
        }
    }

    /// The request for a stream from the whole sequence.
    fn request(&self) -> Value {
        let s = &self.sampling;
        let room = (self.n_ctx as usize)
            .saturating_sub(self.hist.len() + 1)
            .max(1);
        let bias: Vec<Value> = self.never.iter().map(|&t| json!([t, false])).collect();
        json!({
            "prompt": self.hist,
            "n_predict": room,
            "stream": true,
            "return_tokens": true,
            "cache_prompt": true,
            // Its reading of a long prompt as it goes, for the status
            // during the wait (`prompt_progress` events, `sample_pumped`).
            "return_progress": true,
            "id_slot": self.slot,
            "temperature": s.temp,
            "top_k": s.top_k,
            "top_p": s.top_p,
            "min_p": s.min_p,
            "repeat_penalty": s.repeat_penalty,
            // -1 is the whole context to llm.rs; llama-server refuses a negative window.
            "repeat_last_n": if s.repeat_last_n < 0 { self.n_ctx as i32 } else { s.repeat_last_n },
            "dry_multiplier": s.dry_multiplier,
            "dry_base": s.dry_base,
            "dry_allowed_length": s.dry_allowed_length,
            "dry_penalty_last_n": if s.dry_last_n < 0 { self.n_ctx as i32 } else { s.dry_last_n },
            "dry_sequence_breakers": DRY_BREAKERS,
            // Each stream its own draws: the same seed again would repeat them.
            "seed": s.seed.wrapping_add(self.streams as u32),
            "logit_bias": bias,
        })
    }

    fn open(&mut self) -> std::result::Result<Stream, Open> {
        if self.hist.len() + 1 >= self.n_ctx as usize {
            return Err(Open::Fatal(anyhow!(
                "the sequence ({} tokens) fills the server's context of {}",
                self.hist.len(),
                self.n_ctx
            )));
        }
        let body = self.request();
        let (status, b) = self
            .server
            .send(
                "POST",
                "/completion",
                Some(&body),
                Instant::now() + FIRST_TOKEN,
            )
            .map_err(Open::Retry)?;
        if status != 200 {
            let text = b
                .read_all(Instant::now() + SHORT)
                .ok()
                .and_then(|v| serde_json::from_slice::<Value>(&v).ok())
                .map(|v| err_text(&v))
                .unwrap_or_default();
            let e = anyhow!("/completion: HTTP {status}: {text}");
            return Err(if status >= 500 || status == 429 {
                Open::Retry(e)
            } else {
                Open::Fatal(e)
            });
        }
        self.streams += 1;
        Ok(Stream {
            body: b,
            sse: Vec::new(),
            base: self.hist.len(),
            given: Vec::new(),
            queue: VecDeque::new(),
            stop: None,
            progress: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_str(out: &mut Vec<u8>, key: &str, val: &str) {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(val.len() as u64).to_le_bytes());
        out.extend_from_slice(val.as_bytes());
    }

    fn kv_u32(out: &mut Vec<u8>, key: &str, val: u32) {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    }

    /// A small GGUF head: five tokens, their types, the keys read, and a
    /// merges array that is skipped.
    fn tiny() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        kv_str(&mut b, "general.architecture", "toy");
        kv_str(&mut b, "general.name", "Toy");
        kv_u32(&mut b, "toy.block_count", 7);
        kv_str(&mut b, "tokenizer.ggml.model", "gpt2");
        let tokens = [
            "\u{120}hi",
            "<|im_end|>",
            "<think>",
            "a\u{10A}",
            "\u{120}\u{2014}",
        ];
        let key = "tokenizer.ggml.tokens";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for t in tokens {
            b.extend_from_slice(&(t.len() as u64).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        }
        let key = "tokenizer.ggml.token_type";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&5u32.to_le_bytes());
        b.extend_from_slice(&5u64.to_le_bytes());
        for k in [1i32, 3, 4, 1, 1] {
            b.extend_from_slice(&k.to_le_bytes());
        }
        let key = "tokenizer.ggml.merges";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        b.extend_from_slice(b"h i");
        kv_u32(&mut b, "tokenizer.ggml.eos_token_id", 1);
        b
    }

    #[test]
    fn vocabulary_pieces_as_llama_cpp_gives_them() {
        let v = Vocab::from_bytes(&tiny()).unwrap();
        assert_eq!(v.len(), 5);
        assert_eq!(v.n_layer, 7);
        assert_eq!(v.name, "Toy");
        assert_eq!(v.eot, 1);
        assert_eq!(v.eog, vec![1]);
        let piece = |t, special| {
            let mut o = Vec::new();
            v.piece(t, special, &mut o);
            o
        };
        // U+0120 is the space, U+010A the newline in GPT-2's mapping.
        assert_eq!(piece(0, false), b" hi");
        assert_eq!(piece(3, false), b"a\n");
        // A control token only with `special`; a user-defined one always.
        assert_eq!(piece(1, false), b"");
        assert_eq!(piece(1, true), b"<|im_end|>");
        assert_eq!(piece(2, false), b"<think>");
        assert_eq!(v.containing("\u{2014}".as_bytes()), vec![4]);
    }

    #[test]
    fn a_truncated_head_is_an_error_not_a_panic() {
        let b = tiny();
        assert!(Vocab::from_bytes(&b[..b.len() - 3]).is_err());
        assert!(Vocab::from_bytes(b"GGUF").is_err());
    }

    #[test]
    fn addresses() {
        let s = Server::parse("http://192.168.0.39:8001/").unwrap();
        assert_eq!(s.url(), "http://192.168.0.39:8001");
        assert!(Server::parse("https://x:1").is_err());
        assert!(Server::parse("http://x:1/v1").is_err());
        assert_eq!(Server::parse("http://rack").unwrap().port, 80);
    }

    /// Bytes handed out seven at a time, a timeout between pieces, as a
    /// slow connection gives them.
    struct Trickle {
        bytes: Vec<u8>,
        at: usize,
        pause: bool,
    }

    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.pause = !self.pause;
            if self.pause {
                return Err(std::io::Error::new(ErrorKind::WouldBlock, "nothing yet"));
            }
            let n = 7.min(buf.len()).min(self.bytes.len() - self.at);
            buf[..n].copy_from_slice(&self.bytes[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    /// A stream's events through the chunked framing, split at awkward
    /// places (no network: the sandbox of `improve.md` has none).
    #[test]
    fn chunked_events_give_their_tokens_and_stop() {
        let events = [
            // A progress event as llama-server sends it: its token is no token.
            "data: {\"index\":0,\"content\":\"\",\"tokens\":[0],\"stop\":false,\"id_slot\":1,\"tokens_predicted\":0,\"tokens_evaluated\":3,\"prompt_progress\":{\"total\":3,\"cache\":1,\"processed\":2,\"time_ms\":5}}\n\n",
            "data: {\"tokens\":[16],\"stop\":false}\n\n",
            "data: {\"tokens\":[198, 17],\"stop\":false}\r\n\r\n",
            "data: {\"tokens\":[],\"stop\":true,\"stop_type\":\"limit\"}\n\n",
        ];
        let mut out = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for e in events {
            out.extend_from_slice(format!("{:x}\r\n{e}\r\n", e.len()).as_bytes());
        }
        out.extend_from_slice(b"0\r\n\r\n");
        let mut body = Body {
            src: Box::new(Trickle {
                bytes: out,
                at: 0,
                pause: false,
            }),
            raw: Vec::new(),
            chunked: false,
            left: None,
            chunk_left: 0,
            after_chunk: false,
            done: false,
            closed: false,
            last: Instant::now(),
        };
        assert_eq!(body.head(Instant::now() + SHORT).unwrap(), 200);
        assert!(body.chunked);
        let mut s = Stream {
            body,
            sse: Vec::new(),
            base: 2,
            given: Vec::new(),
            queue: VecDeque::new(),
            stop: None,
            progress: None,
        };
        assert!(s.continues(&[5, 6]));
        assert_eq!(s.next().unwrap(), Some(16));
        assert!(s.continues(&[5, 6, 16]));
        assert!(!s.continues(&[5, 6, 15]));
        assert_eq!(s.next().unwrap(), Some(198));
        assert_eq!(s.next().unwrap(), Some(17));
        assert_eq!(s.next().unwrap(), None);
        assert_eq!(s.stop.as_deref(), Some("limit"));
        assert_eq!(s.progress, Some((2, 3)));
    }

    /// A short answer by its length, then the connection closed.
    #[test]
    fn a_sized_answer_reads_whole() {
        let body = b"{\"tokens\":[1,2,3]}";
        let mut out =
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(body);
        let mut b = Body {
            src: Box::new(Trickle {
                bytes: out,
                at: 0,
                pause: false,
            }),
            raw: Vec::new(),
            chunked: false,
            left: None,
            chunk_left: 0,
            after_chunk: false,
            done: false,
            closed: false,
            last: Instant::now(),
        };
        assert_eq!(b.head(Instant::now() + SHORT).unwrap(), 200);
        assert_eq!(b.read_all(Instant::now() + SHORT).unwrap(), body);
    }

    /// A `Remote` with no live server: placing and the context's bound are
    /// local bookkeeping, so they can be tested without one.
    fn bare(n_ctx: u32) -> Remote {
        Remote {
            server: Server::parse("http://127.0.0.1:9").unwrap(),
            vocab: Vocab::from_bytes(&tiny()).unwrap(),
            n_ctx,
            model: "toy".into(),
            slot: 0,
            hist: vec![1, 2, 3],
            stream: None,
            sampling: Sampling {
                temp: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                dry_multiplier: 0.0,
                dry_base: 1.0,
                dry_allowed_length: 2,
                dry_last_n: -1,
                seed: 1,
                repeat_penalty: 1.0,
                repeat_last_n: -1,
                ban_dashes: false,
            },
            never: Vec::new(),
            streams: 0,
            tokens: 0,
        }
    }

    /// A place that does not fit is refused before the sequence is touched:
    /// truncating first left a destroyed sequence behind when it bailed.
    #[test]
    fn a_refused_place_leaves_the_sequence_as_it_was() {
        let mut r = bare(8);
        // Six tokens at 2 reaches 8, the whole context: refused, and the
        // old sequence stands whole.
        assert!(r.place(0, &[4, 5, 6, 7, 8, 9], 2).is_err());
        assert_eq!(r.hist, vec![1, 2, 3]);
        // One under the cap goes in.
        r.place(0, &[4, 5], 0).unwrap();
        assert_eq!(r.hist, vec![4, 5]);
    }

    /// A whole HTTP answer read off a trickling connection, as `exchange`
    /// reads one: its status and body.
    fn answer(raw: &str) -> (u16, Vec<u8>) {
        let mut b = Body {
            src: Box::new(Trickle {
                bytes: raw.as_bytes().to_vec(),
                at: 0,
                pause: false,
            }),
            raw: Vec::new(),
            chunked: false,
            left: None,
            chunk_left: 0,
            after_chunk: false,
            done: false,
            closed: false,
            last: Instant::now(),
        };
        let status = b.head(Instant::now() + SHORT).unwrap();
        (status, b.read_all(Instant::now() + SHORT).unwrap())
    }

    fn sized(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
    }

    /// The prefetch's answers, from the bytes a server sends (no network):
    /// an id, a server without the path, a refusal, and every state.
    #[test]
    fn prefetch_answers_map_to_their_states() {
        let (s, b) = answer(&sized("200 OK", "{\"id\":7}"));
        assert_eq!(prefetch_started(s, &b).unwrap(), PrefetchStart::Id(7));
        let (s, b) = answer(&sized("200 OK", "{\"id\":\"12\"}"));
        assert_eq!(prefetch_started(s, &b).unwrap(), PrefetchStart::Id(12));
        // llama-server's answer to a path it does not have.
        let (s, b) = answer(&sized(
            "404 Not Found",
            "{\"error\":{\"message\":\"File Not Found\",\"type\":\"not_found_error\",\"code\":404}}",
        ));
        assert_eq!(prefetch_started(s, &b).unwrap(), PrefetchStart::Unsupported);
        // Refused, and accepted without an id: errors, with the server's words.
        let (s, b) = answer(&sized(
            "400 Bad Request",
            "{\"error\":{\"message\":\"no tokens\"}}",
        ));
        assert!(format!("{:#}", prefetch_started(s, &b).unwrap_err()).contains("no tokens"));
        let (s, b) = answer(&sized("200 OK", "{\"queued\":true}"));
        assert!(prefetch_started(s, &b).is_err());
        assert!(prefetch_started(200, b"not json").is_err());

        let state = |raw: &str| {
            let (s, b) = answer(raw);
            prefetch_state(s, &b)
        };
        assert_eq!(
            state(&sized("200 OK", "{\"id\":7,\"state\":\"waiting\"}")).unwrap(),
            Prefetch::Waiting
        );
        assert_eq!(
            state(&sized("200 OK", "{\"status\":\"ready\"}")).unwrap(),
            Prefetch::Ready
        );
        assert_eq!(
            state(&sized("200 OK", "\"done\"")).unwrap(),
            Prefetch::Ready
        );
        assert_eq!(
            state(&sized(
                "200 OK",
                "{\"state\":\"failed\",\"error\":\"out of memory\"}"
            ))
            .unwrap(),
            Prefetch::Failed("out of memory".into())
        );
        assert_eq!(
            state(&sized("200 OK", "{\"state\":\"evicted\"}")).unwrap(),
            Prefetch::Unknown("evicted".into())
        );
        // An id the server does not know: failed; a server error: asked again.
        assert!(matches!(
            state(&sized("404 Not Found", "{\"error\":{\"message\":\"no such id\"}}")).unwrap(),
            Prefetch::Failed(m) if m.contains("no such id")
        ));
        assert!(state(&sized(
            "503 Service Unavailable",
            "{\"error\":{\"message\":\"loading\"}}"
        ))
        .is_err());
        assert!(state(&sized("200 OK", "{\"id\":7}")).is_err());
    }

    /// A source that never has bytes: every wait stays in the pump.
    struct Stall;
    impl Read for Stall {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(ErrorKind::WouldBlock, "nothing yet"))
        }
    }

    /// The pump runs on every wait for bytes and can cancel one: no
    /// network, no server, nothing slept (no network: `improve.md`).
    #[test]
    fn the_pump_answers_during_a_wait_and_cancels_it() {
        let body = Body {
            src: Box::new(Stall),
            raw: Vec::new(),
            chunked: false,
            left: None,
            chunk_left: 0,
            after_chunk: false,
            done: false,
            closed: false,
            last: Instant::now(),
        };
        let mut s = Stream {
            body,
            sse: Vec::new(),
            base: 0,
            given: Vec::new(),
            queue: VecDeque::new(),
            stop: None,
            progress: None,
        };
        let mut calls = 0usize;
        let r = s.next_with(&mut |_| {
            calls += 1;
            Ok(calls < 3)
        });
        assert_eq!(calls, 3);
        let e = r.unwrap_err();
        assert!(e.downcast_ref::<Aborted>().is_some());
    }
}
