//! The lens readout on the GPU: a residual vector is transported into the
//! final block's basis (`J_l h`, or `h` itself for the final block), then
//! decoded the way the model decodes its own last residual: the output
//! norm (`rms_norm` with the graph's epsilon, times the norm weight) and
//! the unembedding. The norm weight and the unembedding are the model's own
//! tensors, taken from the graph by the capture (`capture.rs`), so nothing
//! of the 398 MiB unembedding is copied. The graph is built per call with
//! ggml on a CUDA backend instance of this program's own; `rank` turns the
//! logits into a ranked readout on the host. See readout.md.

use std::ffi::c_void;

use anyhow::{bail, Context as _, Result};

use crate::sys;

/// A transport matrix resident on the GPU: `d x d`, float16, row `i` holding
/// the coefficients of output coordinate `i` (the reference's `J[i][j]`).
// `layer` and `Readout::transport` are read by the lens readouts (next).
#[allow(dead_code)]
pub struct Transport {
    pub layer: i32,
    tensor: *mut sys::ggml_tensor,
}

/// A group of columns sharing one transport (or none: the final block).
pub struct Group<'a> {
    pub transport: Option<&'a Transport>,
    /// The columns are already the normed final residual: unembed only
    /// (a diagnostic of the unembedding alone).
    pub normed: bool,
    /// `n` residual vectors of `d` floats, one after the other.
    pub columns: &'a [f32],
}

pub struct Readout {
    backend: sys::ggml_backend_t,
    galloc: sys::ggml_gallocr_t,
    /// The transports' own context and buffer.
    weights_ctx: *mut sys::ggml_context,
    weights_buf: sys::ggml_backend_buffer_t,
    pub transports: Vec<Transport>,
    unembed: *mut sys::ggml_tensor,
    norm_w: *mut sys::ggml_tensor,
    eps: f32,
    pub d: usize,
    pub n_vocab: usize,
}

// SAFETY: used from the engine thread only, like the model it reads from.
unsafe impl Send for Readout {}

impl Readout {
    /// A readout over the model's own `unembed` and `norm_w` tensors (both
    /// on `device`), with room for `transports` matrices of `d x d`.
    pub fn new(
        device: sys::ggml_backend_dev_t,
        unembed: *mut sys::ggml_tensor,
        norm_w: *mut sys::ggml_tensor,
        eps: f32,
        layers: &[(i32, &[u16])],
    ) -> Result<Self> {
        // SAFETY: plain ggml calls; every pointer is checked before use, and
        // the model's tensors outlive the readout (the model owns both).
        unsafe {
            if unembed.is_null() || norm_w.is_null() {
                bail!("the capture has not seen the unembedding and the output norm yet (decode once first)");
            }
            let d = (*unembed).ne[0] as usize;
            let n_vocab = (*unembed).ne[1] as usize;
            if (*norm_w).ne[0] as usize != d || (*norm_w).type_ != sys::ggml_type_GGML_TYPE_F32 {
                bail!("the output norm weight is not {d} floats");
            }
            for (name, t) in [("unembedding", unembed), ("output norm", norm_w)] {
                if (*t).buffer.is_null() || sys::ggml_backend_buffer_is_host((*t).buffer) {
                    bail!("the model's {name} is not in GPU memory; the readout needs it there");
                }
            }
            let backend = sys::ggml_backend_dev_init(device, std::ptr::null());
            if backend.is_null() {
                bail!("could not start a CUDA backend for the readout");
            }
            let galloc = sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(backend));
            let mut r = Self {
                backend,
                galloc,
                weights_ctx: std::ptr::null_mut(),
                weights_buf: std::ptr::null_mut(),
                transports: Vec::new(),
                unembed,
                norm_w,
                eps,
                d,
                n_vocab,
            };
            if !layers.is_empty() {
                r.load_transports(layers)?;
            }
            Ok(r)
        }
    }

    /// Replace the resident transports with `layers` (each `d * d` float16
    /// bit patterns, row-major).
    pub fn load_transports(&mut self, layers: &[(i32, &[u16])]) -> Result<()> {
        let d = self.d;
        // SAFETY: a context of our own; tensors are sized before the copy.
        unsafe {
            self.free_transports();
            let params = sys::ggml_init_params {
                mem_size: sys::ggml_tensor_overhead() * (layers.len() + 1),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = sys::ggml_init(params);
            if ctx.is_null() {
                bail!("ggml_init failed");
            }
            let mut ts = Vec::new();
            for (layer, data) in layers {
                if data.len() != d * d {
                    sys::ggml_free(ctx);
                    bail!(
                        "the transport of block {layer} holds {} values, not {}",
                        data.len(),
                        d * d
                    );
                }
                let t =
                    sys::ggml_new_tensor_2d(ctx, sys::ggml_type_GGML_TYPE_F16, d as i64, d as i64);
                ts.push(Transport {
                    layer: *layer,
                    tensor: t,
                });
            }
            let buf = sys::ggml_backend_alloc_ctx_tensors(ctx, self.backend);
            if buf.is_null() {
                sys::ggml_free(ctx);
                bail!(
                    "no GPU memory for {} transports of {} MiB",
                    layers.len(),
                    (d * d * 2) >> 20
                );
            }
            for (t, (_, data)) in ts.iter().zip(layers) {
                sys::ggml_backend_tensor_set(
                    t.tensor,
                    data.as_ptr() as *const c_void,
                    0,
                    d * d * 2,
                );
            }
            self.weights_ctx = ctx;
            self.weights_buf = buf;
            self.transports = ts;
            Ok(())
        }
    }

    fn free_transports(&mut self) {
        // SAFETY: each was made once by `load_transports`.
        unsafe {
            if !self.weights_buf.is_null() {
                sys::ggml_backend_buffer_free(self.weights_buf);
                self.weights_buf = std::ptr::null_mut();
            }
            if !self.weights_ctx.is_null() {
                sys::ggml_free(self.weights_ctx);
                self.weights_ctx = std::ptr::null_mut();
            }
        }
        self.transports.clear();
    }

    /// The model's output norm weight, read back from the GPU.
    pub fn norm_weight(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.d];
        // SAFETY: the tensor is `d` floats (checked in `new`).
        unsafe {
            sys::ggml_backend_tensor_get(self.norm_w, w.as_mut_ptr() as *mut c_void, 0, self.d * 4)
        };
        w
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    #[allow(dead_code)]
    pub fn transport(&self, layer: i32) -> Option<&Transport> {
        self.transports.iter().find(|t| t.layer == layer)
    }

    /// Logits for every column of every group, in order: `n_vocab` floats
    /// per column.
    pub fn logits(&mut self, groups: &[Group]) -> Result<Vec<Vec<f32>>> {
        let d = self.d;
        let n_cols: usize = groups.iter().map(|g| g.columns.len() / d).sum();
        if n_cols == 0 {
            return Ok(Vec::new());
        }
        for g in groups {
            if g.columns.len() % d != 0 {
                bail!("a group's columns are not whole vectors of {d}");
            }
        }
        // SAFETY: a graph of our own over our tensors and the model's two
        // (read only); inputs are set after allocation, the output read
        // after a synchronous compute.
        unsafe {
            let n_nodes = 8 + 4 * groups.len();
            let params = sys::ggml_init_params {
                mem_size: sys::ggml_tensor_overhead() * n_nodes * 2 + sys::ggml_graph_overhead(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = sys::ggml_init(params);
            if ctx.is_null() {
                bail!("ggml_init failed");
            }
            let mut inputs = Vec::new();
            let mut x: *mut sys::ggml_tensor = std::ptr::null_mut();
            let normed_only = groups.iter().all(|g| g.normed);
            if groups.iter().any(|g| g.normed) && !normed_only {
                sys::ggml_free(ctx);
                bail!("normed and raw columns cannot share one readout");
            }
            for g in groups {
                let n = (g.columns.len() / d) as i64;
                if n == 0 {
                    continue;
                }
                let h = sys::ggml_new_tensor_2d(ctx, sys::ggml_type_GGML_TYPE_F32, d as i64, n);
                sys::ggml_set_input(h);
                inputs.push((h, g.columns));
                let v = match g.transport {
                    Some(t) => sys::ggml_mul_mat(ctx, t.tensor, h),
                    None => h,
                };
                x = if x.is_null() {
                    v
                } else {
                    sys::ggml_concat(ctx, x, v, 1)
                };
            }
            let scaled = if normed_only {
                x
            } else {
                let normed = sys::ggml_rms_norm(ctx, x, self.eps);
                sys::ggml_mul(ctx, normed, self.norm_w)
            };
            let out = sys::ggml_mul_mat(ctx, self.unembed, scaled);
            sys::ggml_set_output(out);
            let graph = sys::ggml_new_graph(ctx);
            sys::ggml_build_forward_expand(graph, out);
            if !sys::ggml_gallocr_alloc_graph(self.galloc, graph) {
                sys::ggml_free(ctx);
                bail!("no GPU memory for the readout of {n_cols} columns");
            }
            for (h, cols) in &inputs {
                sys::ggml_backend_tensor_set(*h, cols.as_ptr() as *const c_void, 0, cols.len() * 4);
            }
            let st = sys::ggml_backend_graph_compute(self.backend, graph);
            if st != sys::ggml_status_GGML_STATUS_SUCCESS {
                sys::ggml_free(ctx);
                bail!("the readout graph failed ({st})");
            }
            let v = self.n_vocab;
            let mut all = vec![0f32; v * n_cols];
            sys::ggml_backend_tensor_get(out, all.as_mut_ptr() as *mut c_void, 0, all.len() * 4);
            sys::ggml_free(ctx);
            Ok(all.chunks(v).map(|c| c.to_vec()).collect())
        }
    }
}

impl Drop for Readout {
    fn drop(&mut self) {
        self.free_transports();
        // SAFETY: each was made once in `new`.
        unsafe {
            sys::ggml_gallocr_free(self.galloc);
            sys::ggml_backend_free(self.backend);
        }
    }
}

/// A ranked readout of one column: the top `k` tokens with their
/// log-probabilities under the column's softmax.
#[derive(Clone, Debug)]
pub struct Ranked {
    pub top: Vec<(i32, f32)>,
}

/// The `k` largest logits as log-probabilities, best first (ties to the
/// lower token id, as a stable sort gives).
pub fn rank(logits: &[f32], k: usize) -> Ranked {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max
        + logits
            .iter()
            .map(|&x| ((x - max) as f64).exp())
            .sum::<f64>()
            .ln() as f32;
    let k = k.min(logits.len());
    let mut idx: Vec<i32> = (0..logits.len() as i32).collect();
    if k > 0 && k < idx.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize]
                .total_cmp(&logits[a as usize])
                .then(a.cmp(&b))
        });
        idx.truncate(k);
    }
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    Ranked {
        top: idx
            .into_iter()
            .map(|i| (i, logits[i as usize] - lse))
            .collect(),
    }
}

// Read by the lens evaluation's pass (next).
#[allow(dead_code)]
/// The rank (0 = best) of `token` among `logits`: how many tokens score
/// strictly higher, plus those of equal score with a lower id.
pub fn rank_of(logits: &[f32], token: i32) -> usize {
    let x = logits[token as usize];
    logits
        .iter()
        .enumerate()
        .filter(|&(i, &y)| y > x || (y == x && (i as i32) < token))
        .count()
}

/// Compare two logit vectors: whether their top `k` agree in order, and
/// the largest absolute difference.
pub fn compare(a: &[f32], b: &[f32], k: usize) -> Result<(bool, f32)> {
    if a.len() != b.len() {
        bail!("{} against {} logits", a.len(), b.len());
    }
    let ta = rank(a, k)
        .top
        .into_iter()
        .map(|(t, _)| t)
        .collect::<Vec<_>>();
    let tb = rank(b, k)
        .top
        .into_iter()
        .map(|(t, _)| t)
        .collect::<Vec<_>>();
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    Ok((ta == tb, max))
}

/// The device the readout runs on: CUDA0.
pub fn cuda_device() -> Result<sys::ggml_backend_dev_t> {
    let name = std::ffi::CString::new("CUDA0")?;
    // SAFETY: a plain lookup in the backend registry.
    let dev = unsafe { sys::ggml_backend_dev_by_name(name.as_ptr()) };
    (!dev.is_null())
        .then_some(dev)
        .context("no CUDA0 device for the readout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_orders_and_normalizes() {
        let logits = [0.0f32, 3.0, 1.0, 3.0, -2.0];
        let r = rank(&logits, 3);
        let ids: Vec<i32> = r.top.iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        let total: f64 = logits.iter().map(|&x| (x as f64).exp()).sum();
        assert!((r.top[0].1 as f64 - (3.0 - total.ln())).abs() < 1e-5);
        assert_eq!(rank_of(&logits, 1), 0);
        assert_eq!(rank_of(&logits, 3), 1);
        assert_eq!(rank_of(&logits, 4), 4);
    }

    #[test]
    fn compare_reports_agreement_and_distance() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.5, 3.0];
        let (same, d) = compare(&a, &b, 2).unwrap();
        assert!(same);
        assert!((d - 0.5).abs() < 1e-6);
        let c = [3.0f32, 2.0, 1.0];
        assert!(!compare(&a, &c, 2).unwrap().0);
    }
}
