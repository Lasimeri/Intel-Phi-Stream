//! The Jacobian lens as this program keeps it: a `.jlens` file converted
//! once from the reference implementation's `torch.save` file (`torch.rs`),
//! so that nothing at run time parses a pickle. The file is a 4096-byte
//! header and then, per fitted block, its `J_l` as `d x d` float16 bit
//! patterns, row-major (row `i`: the coefficients of output coordinate
//! `i`, as the reference's `residual @ J.T` reads them). See lens.md.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::{bail, Context as _, Result};
use sha2::{Digest, Sha256};

use crate::torch::{Archive, Dtype, Value};

const MAGIC: &[u8; 8] = b"PHIJLENS";
const VERSION: u32 = 1;
const HEADER: usize = 4096;

/// What a `.jlens` file says about itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub d_model: u32,
    /// Prompts the reference averaged over.
    pub n_prompts: u32,
    /// The fitted blocks, ascending.
    pub layers: Vec<i32>,
    /// `||J_l||_F / sqrt(d)` per block (the reference's fit logs the max).
    pub norms: Vec<f32>,
    /// The sha256 of the file it was converted from, hex.
    pub source_sha256: String,
}

impl Header {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut h = Vec::with_capacity(HEADER);
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&self.d_model.to_le_bytes());
        h.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        h.extend_from_slice(&self.n_prompts.to_le_bytes());
        if self.source_sha256.len() != 64 {
            bail!("a source sha256 of {} characters", self.source_sha256.len());
        }
        h.extend_from_slice(self.source_sha256.as_bytes());
        for &l in &self.layers {
            h.extend_from_slice(&l.to_le_bytes());
        }
        for &n in &self.norms {
            h.extend_from_slice(&n.to_le_bytes());
        }
        if h.len() > HEADER {
            bail!("{} blocks do not fit the header", self.layers.len());
        }
        h.resize(HEADER, 0);
        Ok(h)
    }

    fn decode(h: &[u8]) -> Result<Self> {
        if h.len() < HEADER || &h[..8] != MAGIC {
            bail!("not a .jlens file");
        }
        let u = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        if u(8) != VERSION {
            bail!(".jlens version {}, this program reads {VERSION}", u(8));
        }
        let d_model = u(12);
        let n = u(16) as usize;
        let n_prompts = u(20);
        let source_sha256 = String::from_utf8(h[24..88].to_vec())?;
        let mut o = 88;
        if o + n * 8 > HEADER {
            bail!("a header claiming {n} blocks");
        }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(u(o) as i32);
            o += 4;
        }
        let mut norms = Vec::with_capacity(n);
        for _ in 0..n {
            norms.push(f32::from_bits(u(o)));
            o += 4;
        }
        Ok(Self {
            d_model,
            n_prompts,
            layers,
            norms,
            source_sha256,
        })
    }
}

/// An open `.jlens` file; matrices are read one block at a time.
pub struct Lens {
    file: File,
    pub header: Header,
    pub path: String,
}

impl Lens {
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("opening the lens {path} (scripts/fetch-lens.sh makes it)"))?;
        let mut h = vec![0u8; HEADER];
        file.read_exact(&mut h)
            .with_context(|| format!("{path} is too short for a .jlens header"))?;
        let header = Header::decode(&h).with_context(|| format!("reading {path}"))?;
        let d = header.d_model as u64;
        let want = HEADER as u64 + header.layers.len() as u64 * d * d * 2;
        let have = file.metadata()?.len();
        if have != want {
            bail!("{path} is {have} bytes; its header says {want}");
        }
        Ok(Self {
            file,
            header,
            path: path.to_string(),
        })
    }

    /// Block `layer`'s `J_l` as float16 bit patterns, row-major.
    pub fn matrix(&mut self, layer: i32) -> Result<Vec<u16>> {
        let k = self
            .header
            .layers
            .iter()
            .position(|&l| l == layer)
            .with_context(|| {
                format!(
                    "the lens has no block {layer} (it has {:?})",
                    self.header.layers
                )
            })?;
        let d = self.header.d_model as usize;
        self.file
            .seek(SeekFrom::Start((HEADER + k * d * d * 2) as u64))?;
        let mut bytes = vec![0u8; d * d * 2];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }
}

/// IEEE 754 binary16 to binary32, exactly (subnormals, infinities, NaNs).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn sha256_hex(path: &str) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Convert the reference's `lens.pt` to `out` (`.jlens`), checking every
/// tensor: float16, `d x d`, contiguous, one per fitted block.
pub fn convert(src: &str, out: &str) -> Result<Header> {
    let source_sha256 = sha256_hex(src)?;
    let mut ar = Archive::open(src)?;
    let v = ar.value.clone();
    let j = match v.get("J") {
        Some(Value::Dict(j)) => j.clone(),
        _ => bail!("{src} has no J dictionary: not a JacobianLens file (a fit checkpoint?)"),
    };
    let d_model = v
        .get("d_model")
        .and_then(Value::as_int)
        .context("no d_model")? as u32;
    let n_prompts = v
        .get("n_prompts")
        .and_then(Value::as_int)
        .context("no n_prompts")? as u32;
    let listed: Vec<i64> = match v.get("source_layers") {
        Some(Value::List(l)) => l
            .iter()
            .map(|x| x.as_int().context("a source layer that is not an integer"))
            .collect::<Result<_>>()?,
        _ => bail!("no source_layers list"),
    };
    let mut entries = Vec::new();
    for (k, t) in &j {
        let layer = k.as_int().context("a J key that is not a block number")? as i32;
        let Value::Tensor(t) = t else {
            bail!("J[{layer}] is not a tensor");
        };
        if t.dtype != Dtype::F16 {
            bail!("J[{layer}] is {:?}, the reference saves float16", t.dtype);
        }
        if t.size != vec![d_model as i64, d_model as i64] {
            bail!("J[{layer}] is {:?}, not {d_model} x {d_model}", t.size);
        }
        entries.push((layer, t.clone()));
    }
    entries.sort_by_key(|e| e.0);
    let layers: Vec<i32> = entries.iter().map(|e| e.0).collect();
    if layers.iter().map(|&l| l as i64).collect::<Vec<_>>() != listed {
        bail!("J holds blocks {layers:?} but source_layers lists {listed:?}");
    }
    let d = d_model as usize;
    let tmp = format!("{out}.part");
    let mut f = File::create(&tmp).with_context(|| format!("creating {tmp}"))?;
    f.write_all(&vec![0u8; HEADER])?;
    let mut norms = Vec::new();
    for (layer, t) in &entries {
        let bytes = ar.tensor_bytes(t)?;
        if bytes.len() != d * d * 2 {
            bail!("J[{layer}] holds {} bytes", bytes.len());
        }
        let mut ss = 0f64;
        let mut bad = 0usize;
        for b in bytes.chunks_exact(2) {
            let x = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            if !x.is_finite() {
                bad += 1;
            }
            ss += (x as f64) * (x as f64);
        }
        if bad > 0 {
            bail!("J[{layer}] holds {bad} values that are not finite");
        }
        norms.push((ss.sqrt() / (d as f64).sqrt()) as f32);
        f.write_all(&bytes)?;
    }
    let header = Header {
        d_model,
        n_prompts,
        layers,
        norms,
        source_sha256,
    };
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header.encode()?)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, out)?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_convert_exactly() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8_f32);
        assert_eq!(f16_to_f32(0x03ff), 6.097_555e-5);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
    }

    #[test]
    fn headers_round_trip() {
        let h = Header {
            d_model: 2048,
            n_prompts: 1000,
            layers: (0..39).collect(),
            norms: (0..39).map(|i| i as f32 * 0.5).collect(),
            source_sha256: "2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1"
                .into(),
        };
        let e = h.encode().unwrap();
        assert_eq!(e.len(), HEADER);
        assert_eq!(Header::decode(&e).unwrap(), h);
        assert!(Header::decode(&e[..100]).is_err());
    }
}
