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
    /// Everything outside the blocks (embeddings, output, norms) that a GPU
    /// holds.
    pub other: u64,
    /// Tensors kept in host memory whatever the plan: the per-layer token
    /// embeddings (`per_layer_token_embd`, a lookup table of 50.7 GiB in
    /// Qwen3.8 Flash Next), read a row per token. 0 for most models.
    pub host_only: u64,
    /// Attention K and V bytes per token at float16 (the KV cache's growth).
    pub kv_per_token_f16: u64,
    /// The recurrent state one sequence slot holds, f32: for each recurrent
    /// block the convolution state (kernel - 1 times the convolution's
    /// channels) and the state matrix (state size squared times the value
    /// heads). 0 for a model with no recurrent blocks.
    pub recurrent_per_seq: u64,
    /// Per block: 1 when it keeps K and V (an attention block), 2 when it
    /// keeps a recurrent state, 0 for neither (an extra prediction block).
    /// Where a block's cache lives is the GPU that holds the block.
    pub cache_kind: Vec<u8>,
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
        // llama.cpp's recurrent cache for a gated delta net block (the
        // tensors cache_r_l<N> and cache_s_l<N>, f32): conv state
        // (d_conv - 1) x (d_inner + 2 n_group d_state), ssm state
        // d_state x d_state x dt_rank.
        // Only the main pass's blocks keep a recurrent cache (llama.cpp:
        // is_recr_impl is false past hparams.n_layer(), which leaves out the
        // extra prediction blocks).
        let nextn = key_u32(g, &format!("{arch}.nextn_predict_layers")).unwrap_or(0) as u64;
        let recurrent_layers = (n_block as u64)
            .saturating_sub(nextn)
            .saturating_sub(attn_layers);
        let ssm = |k: &str| key_u32(g, &format!("{arch}.ssm.{k}")).unwrap_or(0) as u64;
        let (d_conv, d_inner, d_state, n_group, dt_rank) = (
            ssm("conv_kernel"),
            ssm("inner_size"),
            ssm("state_size"),
            ssm("group_count"),
            ssm("time_step_rank"),
        );
        let recurrent_per_seq = if interval.is_some() && d_state > 0 {
            let conv = d_conv.saturating_sub(1) * (d_inner + 2 * n_group * d_state);
            let state = d_state * d_state * dt_rank;
            recurrent_layers * (conv + state) * 4
        } else {
            0
        };

        let mut s = Sizes {
            block: Vec::new(),
            experts: Vec::new(),
            other: 0,
            host_only: 0,
            kv_per_token_f16,
            recurrent_per_seq,
            cache_kind: (0..n_block as u64)
                .map(|i| {
                    let attn = match interval {
                        Some(iv) if iv > 1 => (i + 1) % iv as u64 == 0,
                        _ => true,
                    };
                    if attn {
                        1
                    } else if i < (n_block as u64).saturating_sub(nextn) && recurrent_per_seq > 0 {
                        2
                    } else {
                        0
                    }
                })
                .collect(),
            arch,
        };
        // A model split over several files (`split.count`, a u16 in
        // llama.cpp's gguf-split) keeps its header in the first and its
        // tensors spread over all of them: every file's table is read.
        let shards = shard_paths(g, path)?;
        let first = tally(g, &mut s);
        sys::gguf_free(g);
        first?;
        for p in shards.iter().skip(1) {
            let f = CString::new(p.as_str())?;
            let gs = sys::gguf_init_from_file(f.as_ptr(), params);
            if gs.is_null() {
                bail!("could not read the tensor table of {p} (part of {path})");
            }
            let t = tally(gs, &mut s);
            sys::gguf_free(gs);
            t?;
        }
        Ok(s)
    }
}

/// The files of a split model, in order (the given one first), or the given
/// file alone when it is not split. The given file must be the first part.
unsafe fn shard_paths(g: *const sys::gguf_context, path: &str) -> Result<Vec<String>> {
    let key = CString::new("split.count")?;
    let id = sys::gguf_find_key(g, key.as_ptr());
    if id < 0 {
        return Ok(vec![path.to_string()]);
    }
    let t = sys::gguf_get_kv_type(g, id);
    let n = if t == sys::gguf_type_GGUF_TYPE_UINT16 {
        sys::gguf_get_val_u16(g, id) as i32
    } else if t == sys::gguf_type_GGUF_TYPE_UINT32 {
        sys::gguf_get_val_u32(g, id) as i32
    } else {
        bail!("{path}: split.count has an unexpected type");
    };
    if n <= 1 {
        return Ok(vec![path.to_string()]);
    }
    let file = CString::new(path)?;
    let mut prefix = vec![0u8; 4096];
    let len = sys::llama_split_prefix(
        prefix.as_mut_ptr() as *mut std::os::raw::c_char,
        prefix.len(),
        file.as_ptr(),
        0,
        n,
    );
    if len <= 0 {
        bail!("{path} is one part of {n}: name the first part (-00001-of-{n:05})");
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut buf = vec![0u8; 4096];
        let k = sys::llama_split_path(
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf.len(),
            prefix.as_ptr() as *const std::os::raw::c_char,
            i,
            n,
        );
        buf.truncate(k.max(0) as usize);
        out.push(String::from_utf8(buf).context("split path")?);
    }
    Ok(out)
}

/// Add one file's tensors to the per-block and other totals.
unsafe fn tally(g: *const sys::gguf_context, s: &mut Sizes) -> Result<()> {
    let n = sys::gguf_get_n_tensors(g);
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
            None if name.starts_with("per_layer_token_embd") => s.host_only += bytes,
            None => s.other += bytes,
        }
    }
    Ok(())
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

/// Blocks per GPU, for llama.cpp's layer split (`tensor_split` given as
/// counts). What each block leaves on a GPU (all of it when its experts
/// stay, else all but its experts) is laid end to end and cut where each
/// GPU's share of the free memory ends; a block goes to the GPU its middle
/// falls in. The last count includes the output layer, which llama.cpp
/// places after the last block (its layer index is the block count), so
/// the counts sum to blocks + 1 and llama.cpp's cut points land exactly
/// between the blocks chosen here. One GPU: every block on it.
///
/// `cache` is each block's own cache on its GPU (`block_cache`), counted
/// with its weights; everything outside the blocks (the output layer and
/// its norms) is counted on the last GPU, where llama.cpp puts it, so the
/// last GPU takes fewer blocks.
pub fn layer_split(s: &Sizes, plan: &Split, free: &[u64], cache: &[u64]) -> Vec<usize> {
    let n = free.len();
    let nb = s.block.len();
    match n {
        0 => return Vec::new(),
        1 => return vec![nb + 1],
        _ => {}
    }
    let on_gpu: Vec<u64> = (0..nb)
        .map(|b| {
            let w = if plan.gpu_set.contains(&b) {
                s.block[b]
            } else {
                s.block[b] - s.experts[b]
            };
            w + cache.get(b).copied().unwrap_or(0)
        })
        .collect();
    let all = (on_gpu.iter().sum::<u64>() + s.other).max(1) as f64;
    let fsum = free.iter().sum::<u64>().max(1) as f64;
    let mut counts = vec![0usize; n];
    let (mut d, mut cum) = (0usize, 0f64);
    let mut bound = all * free[0] as f64 / fsum;
    for &bytes in &on_gpu {
        let mid = cum + bytes as f64 / 2.0;
        while d + 1 < n && mid > bound {
            d += 1;
            bound += all * free[d] as f64 / fsum;
        }
        counts[d] += 1;
        cum += bytes as f64;
    }
    counts[n - 1] += 1;
    counts
}

/// Each block's cache bytes on its GPU: an attention block its share of K
/// and V for `ctx` cells (`kv_per_token`: every attention block's bytes
/// per token, as the context stores them), a recurrent block its share of
/// the recurrent state for `n_seq` sequences.
pub fn block_cache(s: &Sizes, kv_per_token: u64, ctx: u64, n_seq: u64) -> Vec<u64> {
    let attn = s.cache_kind.iter().filter(|&&k| k == 1).count().max(1) as u64;
    let rec = s.cache_kind.iter().filter(|&&k| k == 2).count().max(1) as u64;
    s.cache_kind
        .iter()
        .map(|&k| match k {
            1 => kv_per_token / attn * ctx,
            2 => s.recurrent_per_seq / rec * n_seq,
            _ => 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes() -> Sizes {
        Sizes {
            block: vec![100; 5],
            experts: vec![90; 5],
            other: 50,
            host_only: 0,
            kv_per_token_f16: 0,
            recurrent_per_seq: 0,
            cache_kind: Vec::new(),
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
    fn layer_split_balances_bytes_not_block_counts() {
        // Eight blocks of 100, experts 90 each; blocks 0 and 1 keep their
        // experts (100 each on a GPU), the other six leave 10 each: 260
        // bytes over two equal GPUs, 130 each. Block 0 alone is 100, its
        // middle and block 1's (150) split them; the six small blocks go
        // with block 1.
        let s = Sizes {
            block: vec![100; 8],
            experts: vec![90; 8],
            other: 0,
            host_only: 0,
            kv_per_token_f16: 0,
            recurrent_per_seq: 0,
            cache_kind: Vec::new(),
            arch: "test".into(),
        };
        let p = plan(&s, 200 + 60, Some(2), &[]);
        assert_eq!(p.gpu_set, vec![0, 1]);
        assert_eq!(layer_split(&s, &p, &[1000, 1000], &[]), vec![1, 8]);
        // Four GPUs, the last with twice the room: every block placed once,
        // plus the output layer on the last.
        let c = layer_split(&s, &p, &[1, 1, 1, 2], &[]);
        assert_eq!(c.iter().sum::<usize>(), 9);
        assert_eq!(layer_split(&s, &p, &[5], &[]), vec![9]);
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
