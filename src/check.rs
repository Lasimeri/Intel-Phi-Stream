//! `phi-stream lens check`: the capture and the readout reproduce the
//! model's own logits. The final block's residual of the token being
//! placed is captured (`capture.rs`), decoded by the readout with no
//! transport (`readout.rs`: the output norm and the unembedding, the
//! model's own tensors), and compared with `llama_get_logits_ith` for the
//! same token; the logits row captured from the graph is compared with the
//! same, bit for bit. Three kinds of cycle, the ones the engine makes: a
//! token decoded alone, a token decoded beside another sequence's chunk,
//! and the last token of an injection. Exits with an error on any
//! disagreement. See check.md.

use anyhow::{bail, Context as _, Result};

use crate::llm::{Lane, Llm};
use crate::readout::{compare, cuda_device, Group, Readout};

/// What one compared token gave.
struct Row {
    kind: &'static str,
    pos: i32,
    /// Where the token's row was: its micro-batch and its row in it.
    micro_batch: usize,
    row: i32,
    /// The captured logits row against llama's: largest difference.
    graph_vs_api: f32,
    /// The readout against llama's logits: top-k order equal, largest difference.
    topk_same: bool,
    readout_vs_api: f32,
    /// The host's norm of the captured row against the graph's normed row.
    norm_vs_graph: f32,
    /// The unembedding alone, of the graph's normed row, against llama's logits.
    unembed_vs_api: f32,
}

const TOP_K: usize = 10;
/// The readout and the graph compute the same norm and the same matrix
/// product with the same kernels; anything past rounding is a defect.
const MAX_DIFF: f32 = 1e-2;

fn one(
    llm: &mut Llm,
    readout: &mut Option<Readout>,
    rows: &[i32],
    kind: &'static str,
    pos: i32,
    out: &mut Vec<Row>,
) -> Result<()> {
    if rows.len() != 1 {
        bail!(
            "{kind}: {} logits rows asked for, the check wants one",
            rows.len()
        );
    }
    let last = llm.n_layer() - 1;
    let cap = llm.capture().context("no capture installed")?;
    if let Some(e) = &cap.error {
        bail!("the capture failed: {e}");
    }
    let (outputs, _) = cap.take();
    if outputs.len() != 1 {
        bail!(
            "{kind} at {pos}: {} output rows captured, one asked for",
            outputs.len()
        );
    }
    let o = &outputs[0];
    let h = o
        .layers
        .iter()
        .find(|(l, _)| *l == last)
        .map(|(_, v)| v.clone())
        .with_context(|| format!("{kind} at {pos}: block {last} not captured"))?;
    let graph = o
        .logits
        .clone()
        .context("the logits row was not captured")?;
    if readout.is_none() {
        let cap = llm.capture().context("no capture installed")?;
        let eps = cap
            .norm_eps
            .context("the output norm's epsilon was not seen")?;
        *readout = Some(Readout::new(
            cuda_device()?,
            cap.unembed,
            cap.output_norm,
            eps,
            &[],
        )?);
    }
    let api = llm.logits(rows[0])?.to_vec();
    let r = readout.as_mut().unwrap();
    let ours = r.logits(&[Group {
        transport: None,
        normed: false,
        columns: &h,
    }])?;
    let (_, graph_vs_api) = compare(&graph, &api, TOP_K)?;
    let (topk_same, readout_vs_api) = compare(&ours[0], &api, TOP_K)?;
    // The two stages apart: the norm (on the host, from the captured row
    // and the model's norm weight) and the unembedding alone.
    let graph_normed = o
        .normed
        .clone()
        .context("the graph's normed row was not captured")?;
    let w = r.norm_weight();
    let eps = r.eps();
    let ss: f64 = h.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / h.len() as f64;
    let scale = 1.0 / (ss + eps as f64).sqrt();
    let host_normed: Vec<f32> = h
        .iter()
        .zip(&w)
        .map(|(&x, &wi)| ((x as f64) * scale) as f32 * wi)
        .collect();
    let norm_vs_graph = host_normed
        .iter()
        .zip(&graph_normed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    if std::env::var_os("PHI_STREAM_CAPTURE_DEBUG").is_some_and(|v| v == "1") {
        if let Some(cap) = llm.capture() {
            for (name, ne, v) in &cap.extra {
                eprintln!(
                    "  extra {name} ne {:?} [..6] {:?}",
                    ne,
                    &v[..v.len().min(6)]
                );
            }
        }
        eprintln!(
            "check debug {kind} {pos}: rms(h) {:.4} h[..6] {:?}\n  host normed {:?}\n  graph normed {:?}\n  w {:?}",
            ss.sqrt(),
            &h[..6],
            &host_normed[..6],
            &graph_normed[..6],
            &w[..6]
        );
    }
    let un = r.logits(&[Group {
        transport: None,
        normed: true,
        columns: &graph_normed,
    }])?;
    let (_, unembed_vs_api) = compare(&un[0], &api, TOP_K)?;
    out.push(Row {
        kind,
        pos,
        micro_batch: o.micro_batch,
        row: o.row,
        graph_vs_api,
        topk_same,
        readout_vs_api,
        norm_vs_graph,
        unembed_vs_api,
    });
    Ok(())
}

pub fn check(llm: &mut Llm, prompt: &str, side: &str) -> Result<()> {
    let cap = llm.batch_cap();
    let a = llm.tokenize(prompt, true)?;
    let b = llm.tokenize(side, true)?;
    if a.len() + 1 > cap * 4 || b.len() < 64 {
        bail!("the check's texts are the wrong size");
    }
    let mut readout: Option<Readout> = None;
    let mut out = Vec::new();
    llm.clear();
    if let Some(c) = llm.capture() {
        c.take();
    }

    // The prompt, in chunks; the last chunk asks for the next token.
    let mut pos = 0i32;
    let n_chunks = a.chunks(cap).count();
    let mut last_rows = Vec::new();
    for (i, c) in a.chunks(cap).enumerate() {
        let logits = i + 1 == n_chunks;
        last_rows = llm.decode(&[Lane {
            seq: 0,
            tokens: c,
            pos0: pos,
            logits,
        }])?;
        pos += c.len() as i32;
        if !logits {
            // Chunks without outputs leave nothing to compare.
            if let Some(cp) = llm.capture() {
                let (o, _) = cp.take();
                if !o.is_empty() {
                    bail!(
                        "{} output rows captured from a chunk that asked for none",
                        o.len()
                    );
                }
            }
        }
    }
    one(
        llm,
        &mut readout,
        &last_rows,
        "prompt end",
        pos - 1,
        &mut out,
    )?;
    let mut next = llm.greedy(last_rows[0], true)?;

    // Alone: eight tokens one at a time.
    for _ in 0..8 {
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: &[next],
            pos0: pos,
            logits: true,
        }])?;
        one(llm, &mut readout, &rows, "alone", pos, &mut out)?;
        next = llm.greedy(rows[0], true)?;
        pos += 1;
    }

    // Beside another sequence: it gets the prefix, its first chunk alone,
    // then eight cycles of the live token and a chunk of 16.
    llm.seq_rm(1, -1, -1);
    llm.seq_cp(0, 1, 0, pos);
    let mut rpos = pos;
    llm.decode(&[Lane {
        seq: 1,
        tokens: &b[..16],
        pos0: rpos,
        logits: false,
    }])?;
    if let Some(cp) = llm.capture() {
        cp.take();
    }
    rpos += 16;
    for k in 1..=8usize {
        let chunk = &b[16 * k..16 * (k + 1)];
        let rows = llm.decode(&[
            Lane {
                seq: 0,
                tokens: &[next],
                pos0: pos,
                logits: true,
            },
            Lane {
                seq: 1,
                tokens: chunk,
                pos0: rpos,
                logits: false,
            },
        ])?;
        one(llm, &mut readout, &rows, "beside", pos, &mut out)?;
        next = llm.greedy(rows[0], true)?;
        pos += 1;
        rpos += 16;
    }
    llm.seq_rm(1, -1, -1);

    // An injection: the pending token and 40 tokens in one decode, the last
    // asking for the next token.
    let mut inj = vec![next];
    inj.extend_from_slice(&b[..40]);
    let rows = llm.decode(&[Lane {
        seq: 0,
        tokens: &inj,
        pos0: pos,
        logits: true,
    }])?;
    pos += inj.len() as i32;
    one(llm, &mut readout, &rows, "injection end", pos - 1, &mut out)?;

    println!(
        "{:<14} {:>6} {:>6}  {:>13}  {:>9}  {:>15}  {:>14}  {:>15}",
        "cycle",
        "pos",
        "mb:row",
        "graph vs api",
        "top-10",
        "readout vs api",
        "norm vs graph",
        "unembed vs api"
    );
    let mut bad = 0;
    for r in &out {
        let ok = r.graph_vs_api == 0.0 && r.topk_same && r.readout_vs_api <= MAX_DIFF;
        if !ok {
            bad += 1;
        }
        println!(
            "{:<14} {:>6} {:>6}  {:>13.6}  {:>9}  {:>15.6}  {:>14.6}  {:>15.6}{}",
            r.kind,
            r.pos,
            format!("{}:{}", r.micro_batch, r.row),
            r.graph_vs_api,
            if r.topk_same { "same" } else { "DIFFERENT" },
            r.readout_vs_api,
            r.norm_vs_graph,
            r.unembed_vs_api,
            if ok { "" } else { "   FAIL" }
        );
    }
    let worst = out.iter().map(|r| r.readout_vs_api).fold(0.0, f32::max);
    println!(
        "{} tokens compared; the readout's largest difference {:.6} (limit {MAX_DIFF}); the graph's logits row equal to llama's in {} of {}",
        out.len(),
        worst,
        out.iter().filter(|r| r.graph_vs_api == 0.0).count(),
        out.len()
    );
    if bad > 0 {
        bail!("{bad} of {} tokens disagree", out.len());
    }
    println!("PASS: the captured residual, decoded by the readout, is the model's own next-token distribution");
    Ok(())
}
