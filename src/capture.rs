//! The residual stream of the tokens being placed, read through llama.cpp's
//! public eval callback (`llama_context_params.cb_eval`): the scheduler asks
//! the callback, node by node, whether it wants a tensor; for those it
//! answers yes to, it computes up to that node, synchronizes the backend,
//! and hands the tensor over. This module wants the residual after chosen
//! blocks (`l_out-L`), the indices of the rows whose logits were asked for
//! (the `get_rows` node over `inp_out_ids`), and the logits node
//! (`result_output`), whose first source is the model's own unembedding
//! weight on the GPU. Rows are kept per output row of each micro-batch, so a
//! readout always belongs to exactly the token whose next token is being
//! sampled. llama.cpp is not changed. See capture.md.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::sys;

/// What to capture.
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    /// Blocks whose output residual (`l_out-L`) is read.
    pub layers: Vec<i32>,
    /// Keep every row of every micro-batch, not only the output rows (the
    /// offline readout of a whole prompt).
    pub all_rows: bool,
    /// Also copy the logits rows of `result_output` (verification).
    pub keep_logits: bool,
}

/// The experts a row was routed to at each captured block (the block, the
/// experts).
pub type BlockExperts = Vec<(i32, Vec<i32>)>;

/// One output row of a micro-batch: the residual at each captured block,
/// in `CaptureConfig::layers` order.
#[derive(Clone, Debug)]
pub struct OutputRow {
    pub micro_batch: usize,
    /// The row's index within its micro-batch.
    pub row: i32,
    pub layers: Vec<(i32, Vec<f32>)>,
    pub logits: Option<Vec<f32>>,
    /// The graph's own normed final residual for this row (with `keep_logits`).
    pub normed: Option<Vec<f32>>,
}

/// Every row of one micro-batch at one block (all-rows mode).
// Read by the offline readout of whole prompts (`lens probe`, next).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct BlockRows {
    pub micro_batch: usize,
    pub layer: i32,
    pub n_rows: usize,
    /// `n_rows` rows of `n_embd` floats.
    pub data: Vec<f32>,
}

/// A block's residual captured before the output rows were known.
struct Pending {
    layer: i32,
    n_rows: usize,
    data: Vec<f32>,
}

/// The callback's state. Lives in a `Box` owned by the model: llama.cpp
/// keeps the pointer for the life of the context.
pub struct Capture {
    pub cfg: CaptureConfig,
    /// Ask for nothing (the callback stays installed: llama.cpp fixes it at
    /// context creation; see capture.md for what that costs).
    pub enabled: bool,
    pending: Vec<Pending>,
    /// Output row indices of the current micro-batch, once read.
    ids: Option<Vec<i32>>,
    /// Rows already selected for the current micro-batch, per output.
    selected: Vec<Vec<(i32, Vec<f32>)>>,
    /// The graph's normed output rows of the current micro-batch.
    normed: Vec<Vec<f32>>,
    micro_batch: usize,
    /// Results of the decodes since the last `take`.
    pub outputs: Vec<OutputRow>,
    pub all_rows: Vec<BlockRows>,
    /// The unembedding weight (`result_output`'s first source), as seen.
    pub unembed: *mut sys::ggml_tensor,
    /// The output norm's weight (`result_norm`'s multiplication), as seen.
    pub output_norm: *mut sys::ggml_tensor,
    /// The output norm's epsilon, from its `rms_norm` node's parameters.
    pub norm_eps: Option<f32>,
    /// The first error met; the callback stops asking after it.
    pub error: Option<String>,
    /// Micro-batches finished since the last `take`.
    pub micro_batches: usize,
    /// Print every node the scheduler asks about (`PHI_STREAM_CAPTURE_TRACE=1`).
    pub trace: bool,
    /// Print the row bookkeeping (`PHI_STREAM_CAPTURE_DEBUG=1`).
    pub debug: bool,
    /// Extra nodes copied whole, by name (`PHI_STREAM_CAPTURE_EXTRA=a,b`), for
    /// diagnosis: the last micro-batch's copies.
    pub extra_names: Vec<String>,
    pub extra: Vec<(String, Vec<i64>, Vec<f32>)>,
    /// The experts each output row was routed to at the captured blocks
    /// (`ffn_moe_topk-L`, live: `experts_on`): asked only when on (each asked
    /// node is one more synchronization of the scheduler).
    pub experts_on: bool,
    pending_topk: Vec<(i32, usize, Vec<i32>)>,
    selected_topk: Vec<BlockExperts>,
    /// Since the last `take`: per output row, its micro-batch, its batch
    /// row, and the experts at each captured block.
    pub row_experts: Vec<(usize, i32, BlockExperts)>,
}

impl Capture {
    pub fn new(cfg: CaptureConfig) -> Self {
        Self {
            cfg,
            enabled: true,
            pending: Vec::new(),
            ids: None,
            selected: Vec::new(),
            normed: Vec::new(),
            micro_batch: 0,
            outputs: Vec::new(),
            all_rows: Vec::new(),
            unembed: std::ptr::null_mut(),
            output_norm: std::ptr::null_mut(),
            norm_eps: None,
            error: None,
            micro_batches: 0,
            trace: std::env::var_os("PHI_STREAM_CAPTURE_TRACE").is_some_and(|v| v == "1"),
            debug: std::env::var_os("PHI_STREAM_CAPTURE_DEBUG").is_some_and(|v| v == "1"),
            extra_names: std::env::var("PHI_STREAM_CAPTURE_EXTRA")
                .map(|s| {
                    s.split(',')
                        .filter(|x| !x.is_empty())
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default(),
            extra: Vec::new(),
            experts_on: false,
            pending_topk: Vec::new(),
            selected_topk: Vec::new(),
            row_experts: Vec::new(),
        }
    }

    /// The results of the decodes since the last call, and a fresh start.
    pub fn take(&mut self) -> (Vec<OutputRow>, Vec<BlockRows>) {
        self.micro_batches = 0;
        self.micro_batch = 0;
        self.row_experts.clear();
        (
            std::mem::take(&mut self.outputs),
            std::mem::take(&mut self.all_rows),
        )
    }

    fn fail(&mut self, msg: String) {
        if self.error.is_none() {
            self.error = Some(msg);
        }
    }

    fn wanted_layer(&self, name: &str) -> Option<i32> {
        let l: i32 = name.strip_prefix("l_out-")?.parse().ok()?;
        self.cfg.layers.contains(&l).then_some(l)
    }

    fn wanted_topk(&self, name: &str) -> Option<i32> {
        if !self.experts_on {
            return None;
        }
        let l: i32 = name.strip_prefix("ffn_moe_topk-")?.parse().ok()?;
        self.cfg.layers.contains(&l).then_some(l)
    }
}

/// The tensor's name as a string (up to its NUL).
unsafe fn name_of(t: *const sys::ggml_tensor) -> String {
    let raw = &(*t).name;
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Whether a node is the output row selection. llama.cpp names it
/// `result_norm`: a `get_rows` of the normed final residual over the
/// output row indices (an input it leaves unnamed). The other placement,
/// with the selection inside the last block (llama.cpp's masked next-token
/// embeddings), is refused in `take_ids`, never guessed at.
unsafe fn is_output_rows(t: *const sys::ggml_tensor) -> bool {
    name_of(t) == "result_norm"
}

/// The rows of a 2-D contiguous f32 tensor, as one vector; `None` with the
/// reason when the layout is not that.
unsafe fn rows_of(t: *const sys::ggml_tensor) -> Result<(usize, usize), String> {
    let t = &*t;
    if t.type_ != sys::ggml_type_GGML_TYPE_F32 {
        return Err(format!("type {} is not f32", t.type_));
    }
    let n_embd = t.ne[0] as usize;
    let n_rows = (t.ne[1] * t.ne[2] * t.ne[3]) as usize;
    if t.nb[0] != 4
        || t.nb[1] != n_embd * 4
        || (t.ne[2] > 1 && t.nb[2] != t.nb[1] * t.ne[1] as usize)
    {
        return Err(format!("not contiguous: ne {:?} nb {:?}", t.ne, t.nb));
    }
    Ok((n_embd, n_rows))
}

unsafe fn read_all(t: *const sys::ggml_tensor, n_floats: usize) -> Vec<f32> {
    let mut v = vec![0f32; n_floats];
    if n_floats > 0 {
        sys::ggml_backend_tensor_get(t, v.as_mut_ptr() as *mut c_void, 0, n_floats * 4);
    }
    v
}

unsafe fn read_row(t: *const sys::ggml_tensor, n_embd: usize, row: usize) -> Vec<f32> {
    let mut v = vec![0f32; n_embd];
    sys::ggml_backend_tensor_get(
        t,
        v.as_mut_ptr() as *mut c_void,
        row * n_embd * 4,
        n_embd * 4,
    );
    v
}

/// Select rows `ids` out of `n_rows` rows of `n_embd` floats.
fn select(data: &[f32], n_embd: usize, ids: &[i32]) -> Vec<Vec<f32>> {
    ids.iter()
        .map(|&i| data[i as usize * n_embd..(i as usize + 1) * n_embd].to_vec())
        .collect()
}

impl Capture {
    fn ask(&mut self, t: *const sys::ggml_tensor) -> bool {
        if self.trace {
            // SAFETY: a live node handed over by the scheduler.
            unsafe {
                let src: Vec<String> = (*t)
                    .src
                    .iter()
                    .take_while(|s| !s.is_null())
                    .map(|&s| name_of(s))
                    .collect();
                eprintln!(
                    "capture trace: node {:?} op {} ne {:?} src {:?}",
                    name_of(t),
                    (*t).op,
                    (*t).ne,
                    src
                );
            }
        }
        if !self.enabled || self.error.is_some() {
            return false;
        }
        // SAFETY: the scheduler passes a live node of the graph being computed.
        unsafe {
            let name = name_of(t);
            self.wanted_layer(&name).is_some()
                || self.wanted_topk(&name).is_some()
                || is_output_rows(t)
                || name == "result_output"
                || self.extra_names.contains(&name)
        }
    }

    fn take_layer(&mut self, t: *const sys::ggml_tensor, layer: i32) -> Result<(), String> {
        // SAFETY: the node was computed and its backend synchronized by the
        // scheduler before this call; reads stay inside the tensor.
        unsafe {
            let (n_embd, n_rows) = rows_of(t)?;
            if self.cfg.all_rows {
                let data = read_all(t, n_embd * n_rows);
                self.all_rows.push(BlockRows {
                    micro_batch: self.micro_batch,
                    layer,
                    n_rows,
                    data: data.clone(),
                });
                self.pending.push(Pending {
                    layer,
                    n_rows,
                    data,
                });
                return Ok(());
            }
            match &self.ids {
                // The rows are already only the outputs (a block after the
                // output selection).
                Some(ids) if n_rows == ids.len() => {
                    let rows: Vec<Vec<f32>> = (0..n_rows).map(|r| read_row(t, n_embd, r)).collect();
                    for (j, r) in rows.into_iter().enumerate() {
                        self.selected[j].push((layer, r));
                    }
                }
                Some(ids) => {
                    let ids = ids.clone();
                    for (j, &i) in ids.iter().enumerate() {
                        if i < 0 || i as usize >= n_rows {
                            return Err(format!(
                                "output row {i} outside {n_rows} rows of l_out-{layer}"
                            ));
                        }
                        let r = read_row(t, n_embd, i as usize);
                        self.selected[j].push((layer, r));
                    }
                }
                None => {
                    let data = read_all(t, n_embd * n_rows);
                    if self.debug {
                        eprintln!(
                            "capture: l_out-{layer} pending {n_rows} rows, row 0 [..3] {:?}",
                            &data[..3.min(data.len())]
                        );
                    }
                    self.pending.push(Pending {
                        layer,
                        n_rows,
                        data,
                    });
                }
            }
            Ok(())
        }
    }

    /// A block's selected experts (`ffn_moe_topk-L`: i32, the experts used
    /// by each row): kept until the output rows are known, as a residual is.
    fn take_topk(&mut self, t: *const sys::ggml_tensor, layer: i32) -> Result<(), String> {
        // SAFETY: as in `take_layer`.
        unsafe {
            if (*t).type_ != sys::ggml_type_GGML_TYPE_I32 {
                return Err(format!("ffn_moe_topk-{layer} is not i32"));
            }
            let k = (*t).ne[0] as usize;
            let n_rows = (*t).ne[1] as usize;
            if (*t).nb[1] != k * 4 {
                return Err(format!("ffn_moe_topk-{layer} is not contiguous"));
            }
            let mut data = vec![0i32; k * n_rows];
            if !data.is_empty() {
                sys::ggml_backend_tensor_get(
                    t,
                    data.as_mut_ptr() as *mut c_void,
                    0,
                    data.len() * 4,
                );
            }
            match &self.ids {
                Some(ids) => {
                    for (j, &i) in ids.iter().enumerate() {
                        let i = if n_rows == ids.len() { j } else { i as usize };
                        if i < n_rows && j < self.selected_topk.len() {
                            let r = data[i * k..(i + 1) * k].to_vec();
                            self.selected_topk[j].push((layer, r));
                        }
                    }
                }
                None => self.pending_topk.push((layer, n_rows, data)),
            }
            Ok(())
        }
    }

    fn take_ids(&mut self, t: *const sys::ggml_tensor) -> Result<(), String> {
        // SAFETY: as in `take_layer`; `src[1]` is the ids input, computed.
        unsafe {
            if (*t).op != sys::ggml_op_GGML_OP_GET_ROWS || (*t).src[1].is_null() {
                return Err(format!(
                    "result_norm is op {}, not the get_rows of the output rows: this llama.cpp selects the output rows elsewhere, which the capture does not follow",
                    (*t).op
                ));
            }
            let ids_t = (*t).src[1];
            if (*ids_t).type_ != sys::ggml_type_GGML_TYPE_I32 {
                return Err("inp_out_ids is not i32".into());
            }
            let n = (*ids_t).ne[0] as usize;
            let mut ids = vec![0i32; n];
            if n > 0 {
                sys::ggml_backend_tensor_get(ids_t, ids.as_mut_ptr() as *mut c_void, 0, n * 4);
            }
            self.selected = vec![Vec::new(); n];
            // The experts captured before the selection was known.
            self.selected_topk = vec![Vec::new(); n];
            for (layer, n_rows, data) in std::mem::take(&mut self.pending_topk) {
                let k = data.len().checked_div(n_rows).unwrap_or(0);
                for (j, &i) in ids.iter().enumerate() {
                    if i >= 0 && (i as usize) < n_rows && k > 0 {
                        let r = &data[i as usize * k..(i as usize + 1) * k];
                        self.selected_topk[j].push((layer, r.to_vec()));
                    }
                }
            }
            // Resolve what was captured before the selection was known.
            for p in std::mem::take(&mut self.pending) {
                let n_embd = p.data.len().checked_div(p.n_rows).unwrap_or(0);
                for &i in &ids {
                    if i < 0 || i as usize >= p.n_rows {
                        return Err(format!(
                            "output row {i} outside {} rows of l_out-{}",
                            p.n_rows, p.layer
                        ));
                    }
                }
                for (j, r) in select(&p.data, n_embd, &ids).into_iter().enumerate() {
                    self.selected[j].push((p.layer, r));
                }
            }
            if self.cfg.keep_logits && !ids.is_empty() {
                let (n_embd, n_rows) = rows_of(t)?;
                let all = read_all(t, n_embd * n_rows);
                self.normed = all.chunks(n_embd).map(|c| c.to_vec()).collect();
            }
            self.ids = Some(ids);
            Ok(())
        }
    }

    fn take_logits(&mut self, t: *const sys::ggml_tensor) -> Result<(), String> {
        // SAFETY: as in `take_layer`.
        unsafe {
            if (*t).op != sys::ggml_op_GGML_OP_MUL_MAT {
                return Err(format!(
                    "result_output is op {}, not a matrix multiply (a scaled unembedding?)",
                    (*t).op
                ));
            }
            let w = (*t).src[0];
            if !w.is_null() {
                if self.unembed.is_null() {
                    self.unembed = w;
                } else if self.unembed != w {
                    return Err("the unembedding weight moved between decodes".into());
                }
            }
            let norm = (*t).src[1];
            if !norm.is_null() {
                // result_norm = get_rows(mul(rms_norm(x), w), ids) or
                // mul(rms_norm(x), w): find the multiplication by the weight.
                let mut m = norm;
                if (*m).op == sys::ggml_op_GGML_OP_GET_ROWS {
                    m = (*m).src[0];
                }
                if !m.is_null() && (*m).op == sys::ggml_op_GGML_OP_MUL && !(*m).src[1].is_null() {
                    let w = (*m).src[1];
                    if self.output_norm.is_null() {
                        self.output_norm = w;
                    }
                    let n = (*m).src[0];
                    if !n.is_null()
                        && (*n).op == sys::ggml_op_GGML_OP_RMS_NORM
                        && self.norm_eps.is_none()
                    {
                        self.norm_eps = Some(f32::from_bits((*n).op_params[0] as u32));
                    }
                }
            }
            let ids = self.ids.take().unwrap_or_default();
            let mut selected = std::mem::take(&mut self.selected);
            selected.resize(ids.len(), Vec::new());
            let logits: Vec<Option<Vec<f32>>> = if self.cfg.keep_logits && !ids.is_empty() {
                let (n_vocab, n_rows) = rows_of(t)?;
                if n_rows != ids.len() {
                    return Err(format!("{n_rows} logits rows for {} outputs", ids.len()));
                }
                let all = read_all(t, n_vocab * n_rows);
                (0..n_rows)
                    .map(|r| Some(all[r * n_vocab..(r + 1) * n_vocab].to_vec()))
                    .collect()
            } else {
                vec![None; ids.len()]
            };
            let normed = std::mem::take(&mut self.normed);
            if self.debug {
                for (j, l) in selected.iter().enumerate() {
                    for (layer, v) in l {
                        eprintln!(
                            "capture: output {j} (row {}) l_out-{layer} [..3] {:?}",
                            ids[j],
                            &v[..3.min(v.len())]
                        );
                    }
                }
            }
            let topk = std::mem::take(&mut self.selected_topk);
            for (j, &row) in ids.iter().enumerate() {
                if let Some(e) = topk.get(j).filter(|e| !e.is_empty()) {
                    self.row_experts.push((self.micro_batch, row, e.clone()));
                }
            }
            for (j, (&row, layers)) in ids.iter().zip(selected).enumerate() {
                self.outputs.push(OutputRow {
                    micro_batch: self.micro_batch,
                    row,
                    layers,
                    logits: logits[j].clone(),
                    normed: normed.get(j).cloned(),
                });
            }
            self.pending.clear();
            self.pending_topk.clear();
            self.micro_batch += 1;
            self.micro_batches += 1;
            Ok(())
        }
    }

    fn computed(&mut self, t: *const sys::ggml_tensor) {
        // SAFETY: a live node handed over by the scheduler.
        let name = unsafe { name_of(t) };
        if self.extra_names.contains(&name) {
            // SAFETY: as in `take_layer`.
            unsafe {
                if (*t).type_ == sys::ggml_type_GGML_TYPE_F32 {
                    let n = ((*t).ne[0] * (*t).ne[1] * (*t).ne[2] * (*t).ne[3]) as usize;
                    if (*t).nb[1] == (*t).ne[0] as usize * 4 {
                        let v = read_all(t, n);
                        self.extra.retain(|(k, _, _)| k != &name);
                        self.extra.push((name.clone(), (*t).ne.to_vec(), v));
                    }
                }
            }
        }
        let r = if let Some(l) = self.wanted_layer(&name) {
            self.take_layer(t, l)
        } else if let Some(l) = self.wanted_topk(&name) {
            self.take_topk(t, l)
        } else if unsafe { is_output_rows(t) } {
            self.take_ids(t)
        } else if name == "result_output" {
            self.take_logits(t)
        } else {
            Ok(())
        };
        if let Err(e) = r {
            self.fail(e);
        }
    }
}

/// The callback llama.cpp's scheduler calls (`ggml_backend_sched_eval_callback`).
/// Never unwinds into C: a panic is caught, recorded, and capture stops.
pub unsafe extern "C" fn eval_callback(
    t: *mut sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    let cap = &mut *(user_data as *mut Capture);
    let r = catch_unwind(AssertUnwindSafe(|| {
        if ask {
            cap.ask(t)
        } else {
            cap.computed(t);
            true
        }
    }));
    match r {
        Ok(v) => v,
        Err(_) => {
            cap.fail("the capture panicked".into());
            // Answering a question: no. After a computed node: go on.
            !ask
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_selected_by_index() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let s = select(&data, 4, &[2, 0]);
        assert_eq!(
            s,
            vec![vec![8.0, 9.0, 10.0, 11.0], vec![0.0, 1.0, 2.0, 3.0]]
        );
    }

    #[test]
    fn layers_are_wanted_by_number() {
        let c = Capture::new(CaptureConfig {
            layers: vec![20, 39],
            all_rows: false,
            keep_logits: false,
        });
        assert_eq!(c.wanted_layer("l_out-20"), Some(20));
        assert_eq!(c.wanted_layer("l_out-21"), None);
        assert_eq!(c.wanted_layer("l_out-x"), None);
        assert_eq!(c.wanted_layer("ffn_out-20"), None);
    }
}
