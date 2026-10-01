//! Where the model's bytes go: everything that is not an expert (the
//! attention and recurrent layers, norms, the output) on the GPU; the
//! experts of the first blocks on the GPU as far as its free memory allows
//! once the context's own needs are set aside; the experts of the
//! remaining blocks in host memory, where `libggml_phi.so` takes the
//! cards' share. Every size is read from the file's own tensor table and
//! header (gguf), never guessed. See split.md.

use std::ffi::{CStr, CString};

use anyhow::{bail, Context as _, Result};

use crate::sys;

/// What the file holds, in bytes.
pub struct Sizes {
    /// Per block: every tensor named `blk.N.*`.
    pub block: Vec<u64>,
    /// Per block: the expert tensors alone (`*_exps.*`).
    pub experts: Vec<u64>,
    /// Everything outside the blocks (embeddings, output, norms).
    pub other: u64,
    /// Attention K and V bytes per token at float16 (the KV cache's growth).
    pub kv_per_token_f16: u64,
    /// The architecture's name, for the report.
    pub arch: String,
}

unsafe fn key_u32(g: *const sys::gguf_context, key: &str) -> Option<u32> {
    let k = CString::new(key).ok()?;
    let id = sys::gguf_find_key(g, k.as_ptr());
    (id >= 0).then(|| sys::gguf_get_val_u32(g, id))
}

/// Read the tensor table and the header numbers the plan needs.
pub fn sizes(path: &str) -> Result<Sizes> {
    let file = CString::new(path)?;
    // SAFETY: gguf reads the file's metadata only (`no_alloc`); every
    // pointer it returns is valid until `gguf_free`.
    unsafe {
        let params = sys::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        let g = sys::gguf_init_from_file(file.as_ptr(), params);
        if g.is_null() {
            bail!("could not read the tensor table of {path}");
        }
        let arch_key = CString::new("general.architecture")?;
        let aid = sys::gguf_find_key(g, arch_key.as_ptr());
        if aid < 0 {
            sys::gguf_free(g);
            bail!("{path} names no architecture");
        }
        let arch = CStr::from_ptr(sys::gguf_get_val_str(g, aid))
            .to_string_lossy()
            .into_owned();
        let n_block = key_u32(g, &format!("{arch}.block_count")).unwrap_or(0);
        let n_kv = key_u32(g, &format!("{arch}.attention.head_count_kv")).unwrap_or(0) as u64;
        let klen = key_u32(g, &format!("{arch}.attention.key_length")).unwrap_or(0) as u64;
        let vlen =
            key_u32(g, &format!("{arch}.attention.value_length")).unwrap_or(klen as u32) as u64;
        // A hybrid model keeps K and V only in its full-attention layers
        // (every `full_attention_interval`-th block); a plain one in all.
        let interval = key_u32(g, &format!("{arch}.full_attention_interval"));
        let attn_layers = match interval {
            Some(iv) if iv > 1 => (0..n_block).filter(|i| (i + 1) % iv == 0).count() as u64,
            _ => n_block as u64,
        };
        let kv_per_token_f16 = attn_layers * n_kv * (klen + vlen) * 2;

        let n = sys::gguf_get_n_tensors(g);
        let mut s = Sizes {
            block: Vec::new(),
            experts: Vec::new(),
            other: 0,
            kv_per_token_f16,
            arch,
        };
        for i in 0..n {
            let name = CStr::from_ptr(sys::gguf_get_tensor_name(g, i))
                .to_string_lossy()
                .into_owned();
            let bytes = sys::gguf_get_tensor_size(g, i) as u64;
            match name.strip_prefix("blk.") {
                Some(rest) => {
                    let b: usize = rest
                        .split('.')
                        .next()
                        .and_then(|x| x.parse().ok())
                        .with_context(|| format!("tensor {name}: no block number"))?;
                    if s.block.len() <= b {
                        s.block.resize(b + 1, 0);
                        s.experts.resize(b + 1, 0);
                    }
                    s.block[b] += bytes;
                    if name.contains("_exps.") {
                        s.experts[b] += bytes;
                    }
                }
                None => s.other += bytes,
            }
        }
        sys::gguf_free(g);
        Ok(s)
    }
}

/// The decision.
#[derive(Clone, Debug)]
pub struct Split {
    pub n_blocks: usize,
    /// How many blocks keep their experts on the GPU.
    pub gpu_blocks: usize,
    /// Which, ascending (`0..gpu_blocks` unless some were asked to stay).
    pub gpu_set: Vec<usize>,
    /// Weight bytes the GPU holds (the fixed part and those experts).
    pub gpu_bytes: u64,
    /// Expert bytes in host memory (the cards take their share of them).
    pub host_bytes: u64,
    /// The tensor override that sends the host blocks' experts to host
    /// memory, llama.cpp's own regular expression form; none when every
    /// block is on the GPU.
    pub pattern: Option<String>,
}

/// Choose the blocks. `budget`: bytes the GPU may give to weights. With
/// `want` given the count is the user's; else the most that fit, the last
/// block (a model's extra prediction block, unused here) always on the host.
/// `keep` names blocks whose experts go to the GPU first (the blocks the
/// mind reads: reading a block whose experts run elsewhere costs the
/// stream, Gate B); the rest of the count is filled from block 0 upward.
pub fn plan(s: &Sizes, budget: u64, want: Option<usize>, keep: &[usize]) -> Split {
    let n = s.block.len();
    let fixed: u64 = s.other + (0..n).map(|b| s.block[b] - s.experts[b]).sum::<u64>();
    // The order blocks are given to the GPU: the kept ones, then the rest.
    let mut order: Vec<usize> = keep.iter().copied().filter(|&b| b + 1 < n).collect();
    order.sort_unstable();
    order.dedup();
    for b in 0..n {
        if !order.contains(&b) {
            order.push(b);
        }
    }
    let mut used = fixed;
    let mut gpu: Vec<usize> = Vec::new();
    match want {
        Some(g) => {
            for &b in order.iter().take(g.min(n)) {
                used += s.experts[b];
                gpu.push(b);
            }
        }
        None => {
            for &b in &order {
                // The last block stays home.
                if b + 1 == n || used + s.experts[b] > budget {
                    break;
                }
                used += s.experts[b];
                gpu.push(b);
            }
        }
    }
    gpu.sort_unstable();
    let host: Vec<usize> = (0..n).filter(|b| !gpu.contains(b)).collect();
    let host_bytes = host.iter().map(|&b| s.experts[b]).sum();
    let pattern = (!host.is_empty()).then(|| {
        let alts: Vec<String> = host.iter().map(|b| b.to_string()).collect();
        format!(
            r"blk\.({})\.ffn_(up|gate|down)_exps\.weight",
            alts.join("|")
        )
    });
    Split {
        n_blocks: n,
        gpu_blocks: gpu.len(),
        gpu_set: gpu,
        gpu_bytes: used,
        host_bytes,
        pattern,
    }
}

/// A set of blocks as ranges, `0..20, 26, 32` style.
pub fn ranges(set: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < set.len() {
        let mut j = i;
        while j + 1 < set.len() && set[j + 1] == set[j] + 1 {
            j += 1;
        }
        out.push(if j == i {
            set[i].to_string()
        } else {
            format!("{}..{}", set[i], set[j] + 1)
        });
        i = j + 1;
    }
    out.join(", ")
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes() -> Sizes {
        Sizes {
            block: vec![100; 5],
            experts: vec![90; 5],
            other: 50,
            kv_per_token_f16: 0,
            arch: "test".into(),
        }
    }

    #[test]
    fn auto_fills_the_budget_and_keeps_the_last_block_home() {
        // fixed = 50 + 5 * 10 = 100; each block's experts 90
        let p = plan(&sizes(), 100 + 90 * 2 + 10, None, &[]);
        assert_eq!(p.gpu_blocks, 2);
        assert_eq!(p.gpu_bytes, 280);
        assert_eq!(p.host_bytes, 270);
        assert_eq!(
            p.pattern.as_deref(),
            Some(r"blk\.(2|3|4)\.ffn_(up|gate|down)_exps\.weight")
        );
        let all = plan(&sizes(), u64::MAX, None, &[]);
        assert_eq!(all.gpu_blocks, 4);
    }

    #[test]
    fn kept_blocks_go_first_and_the_rest_fills_from_zero() {
        // Room for two blocks of experts: block 3 is kept, then block 0.
        let p = plan(&sizes(), 100 + 90 * 2 + 10, None, &[3]);
        assert_eq!(p.gpu_set, vec![0, 3]);
        assert_eq!(
            p.pattern.as_deref(),
            Some(r"blk\.(1|2|4)\.ffn_(up|gate|down)_exps\.weight")
        );
        // The last block is never kept.
        let p = plan(&sizes(), u64::MAX, None, &[4]);
        assert_eq!(p.gpu_set, vec![0, 1, 2, 3]);
        assert_eq!(ranges(&[0, 1, 2, 5, 7, 8]), "0..3, 5, 7..9");
    }

    #[test]
    fn a_wanted_count_is_taken_as_is() {
        let p = plan(&sizes(), 0, Some(5), &[]);
        assert_eq!(p.gpu_blocks, 5);
        assert!(p.pattern.is_none());
        let p = plan(&sizes(), 0, Some(0), &[]);
        assert_eq!(p.gpu_blocks, 0);
        assert_eq!(p.host_bytes, 450);
    }
}
