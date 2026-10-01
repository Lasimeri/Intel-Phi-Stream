//! `phi-stream lens eval`: does the Jacobian lens read this model? The
//! reference implementation's own evaluation sets (pinned, fetched by
//! `scripts/fetch-lens.sh`) scored the way the paper scores them: for each
//! item a prompt, a readout position and the intermediate concepts the
//! model should be computing there; at that position the residual of every
//! fitted block is read through the lens and, for comparison, through the
//! plain logit lens (no transport); an intermediate is recovered at `k`
//! when one of its single-token forms is among the top `k` of the readout
//! at any block. Reported per set: pass@k for both lenses, the area under
//! pass@k against log k (normalized so that always-first scores 1), and
//! the per-block fraction recovered at 10, which is where the workspace
//! band shows. See eval.md.

use std::time::Instant;

use anyhow::{bail, Context as _, Result};
use serde_json::Value;

use crate::lens::Lens;
use crate::llm::{Lane, Llm};
use crate::readout::{cuda_device, rank_of, Group, Readout};

/// Where an item is read: the paper's conventions per set (the README of
/// the reference's `data/evaluations`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    /// The last prompt token (multihop, multilingual, order-ops: the token
    /// before the target; association: the closing period; typo: the last
    /// fragment of the misspelling).
    Last,
    /// The last newline token (poetry: the end of the couplet's first line).
    LastNewline,
}

pub struct Item {
    pub name: String,
    pub prompt: String,
    pub intermediates: Vec<String>,
}

pub fn load_set(path: &str) -> Result<(String, Position, Vec<Item>)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} (scripts/fetch-lens.sh fetches the sets)"))?;
    let v: Value = serde_json::from_str(&text).with_context(|| format!("{path} is not JSON"))?;
    let items = v
        .get("items")
        .and_then(Value::as_array)
        .with_context(|| format!("{path} has no items"))?;
    let stem = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = stem.trim_start_matches("lens-eval-").to_string();
    let position = if name == "poetry" {
        Position::LastNewline
    } else {
        Position::Last
    };
    let mut out = Vec::new();
    for it in items {
        let prompt = it.get("prompt").and_then(Value::as_str).with_context(|| {
            format!("{path}: an item without a text prompt (chat items are not in these sets)")
        })?;
        let inter: Vec<String> = it
            .get("intermediates")
            .and_then(Value::as_array)
            .context("an item without intermediates")?
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        out.push(Item {
            name: it
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            prompt: prompt.to_string(),
            intermediates: inter,
        });
    }
    Ok((name, position, out))
}

/// The order-of-operations set's synonym expansion. The reference README
/// describes it ("numbers: digit and word forms; operations: symbol and
/// word forms") without publishing the table; this is the table this
/// program uses, stated so it can be checked. Every other set uses the
/// word itself.
pub fn synonyms(set: &str, word: &str) -> Vec<String> {
    if set != "order-ops" {
        return vec![word.to_string()];
    }
    const NUMBERS: [&str; 21] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ];
    if let Ok(n) = word.parse::<usize>() {
        let mut v = vec![word.to_string()];
        if n < NUMBERS.len() {
            v.push(NUMBERS[n].to_string());
        }
        return v;
    }
    let list: &[&str] = match word {
        "addition" => &["addition", "add", "plus", "sum", "+"],
        "subtraction" => &["subtraction", "subtract", "minus", "difference", "-"],
        "multiplication" => &["multiplication", "multiply", "times", "product", "*", "×"],
        "division" => &["division", "divide", "divided", "quotient", "/", "÷"],
        "mod" => &["mod", "modulo", "remainder", "%"],
        "squared" => &["squared", "square", "^", "power"],
        _ => &[],
    };
    if list.is_empty() {
        vec![word.to_string()]
    } else {
        list.iter().map(|s| s.to_string()).collect()
    }
}

/// The single-token forms of a word in this vocabulary: the word with and
/// without a leading space, as one token each.
pub fn single_tokens(llm: &Llm, words: &[String]) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    for w in words {
        for form in [w.clone(), format!(" {w}")] {
            let t = llm.tokenize(&form, false)?;
            if t.len() == 1 && !out.contains(&t[0]) {
                out.push(t[0]);
            }
        }
    }
    Ok(out)
}

/// The area under pass@k against log k for k in 1..=k_max, normalized so
/// that always rank 0 scores 1, exactly: an intermediate first recovered
/// at k = r + 1 contributes (ln k_max - ln(r + 1)) / ln k_max.
pub fn auc(ranks: &[usize], k_max: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let lk = (k_max as f64).ln();
    ranks
        .iter()
        .map(|&r| {
            let k = r + 1;
            if k > k_max {
                0.0
            } else {
                (lk - (k as f64).ln()) / lk
            }
        })
        .sum::<f64>()
        / ranks.len() as f64
}

pub fn pass_at(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    ranks.iter().filter(|&&r| r < k).count() as f64 / ranks.len() as f64
}

/// What one set gave.
pub struct SetResult {
    pub name: String,
    pub items: usize,
    pub scored: usize,
    pub skipped: Vec<String>,
    /// Per scored intermediate: the best rank over blocks, J-lens and logit lens.
    pub j_best: Vec<usize>,
    pub logit_best: Vec<usize>,
    /// The model's own next-token rank of the intermediate (block 39 decoded).
    pub final_rank: Vec<usize>,
    /// Per block: intermediates recovered at 10, J-lens and logit lens.
    pub j_by_layer: Vec<usize>,
    pub logit_by_layer: Vec<usize>,
    pub secs: f64,
}

/// Run one set. `layers` are the lens's fitted blocks; the capture must
/// include them and the final block.
pub fn run_set(
    llm: &mut Llm,
    readout: &mut Readout,
    set: &str,
    position: Position,
    items: &[Item],
    layers: &[i32],
    final_layer: i32,
) -> Result<SetResult> {
    let t0 = Instant::now();
    let mut res = SetResult {
        name: set.to_string(),
        items: items.len(),
        scored: 0,
        skipped: Vec::new(),
        j_best: Vec::new(),
        logit_best: Vec::new(),
        final_rank: Vec::new(),
        j_by_layer: vec![0; layers.len()],
        logit_by_layer: vec![0; layers.len()],
        secs: 0.0,
    };
    let cap = llm.batch_cap();
    for it in items {
        let tokens = llm.tokenize(&it.prompt, false)?;
        let pos = match position {
            Position::Last => tokens.len() - 1,
            Position::LastNewline => {
                let mut p = None;
                for (i, &t) in tokens.iter().enumerate() {
                    if llm.text(&[t]).contains('\n') {
                        p = Some(i);
                    }
                }
                p.with_context(|| format!("{}: no newline token in the prompt", it.name))?
            }
        };
        // The prompt up to and including the readout position; the
        // position asks for logits, so its residuals are captured.
        let prefix = &tokens[..=pos];
        llm.clear();
        if let Some(c) = llm.capture() {
            c.take();
        }
        let n = prefix.chunks(cap).count();
        let mut p0 = 0i32;
        for (i, c) in prefix.chunks(cap).enumerate() {
            llm.decode(&[Lane {
                seq: 0,
                tokens: c,
                pos0: p0,
                logits: i + 1 == n,
            }])?;
            p0 += c.len() as i32;
        }
        let c = llm.capture().context("no capture installed")?;
        if let Some(e) = &c.error {
            bail!("the capture failed: {e}");
        }
        let (outputs, _) = c.take();
        if outputs.len() != 1 {
            bail!(
                "{}: {} output rows captured, one asked for",
                it.name,
                outputs.len()
            );
        }
        let o = &outputs[0];
        let h = |l: i32| -> Result<&[f32]> {
            o.layers
                .iter()
                .find(|(x, _)| *x == l)
                .map(|(_, v)| v.as_slice())
                .with_context(|| format!("block {l} not captured"))
        };
        // Per intermediate, its candidate tokens; skip those with none.
        let mut targets: Vec<(String, Vec<i32>)> = Vec::new();
        for w in &it.intermediates {
            let cands = single_tokens(llm, &synonyms(set, w))?;
            if cands.is_empty() {
                res.skipped.push(format!("{}: {w}", it.name));
            } else {
                targets.push((w.clone(), cands));
            }
        }
        if targets.is_empty() {
            continue;
        }
        // One readout: every block through its transport, every block as
        // it is (the logit lens), and the final block.
        let mut groups = Vec::new();
        for &l in layers {
            groups.push(Group {
                transport: Some(l),
                normed: false,
                columns: h(l)?,
            });
        }
        for &l in layers {
            groups.push(Group {
                transport: None,
                normed: false,
                columns: h(l)?,
            });
        }
        groups.push(Group {
            transport: None,
            normed: false,
            columns: h(final_layer)?,
        });
        let logits = readout.logits(&groups)?;
        let nl = layers.len();
        for (_, cands) in &targets {
            let best = |col: &[f32]| cands.iter().map(|&t| rank_of(col, t)).min().unwrap();
            let j: Vec<usize> = (0..nl).map(|i| best(&logits[i])).collect();
            let lg: Vec<usize> = (0..nl).map(|i| best(&logits[nl + i])).collect();
            for i in 0..nl {
                if j[i] < 10 {
                    res.j_by_layer[i] += 1;
                }
                if lg[i] < 10 {
                    res.logit_by_layer[i] += 1;
                }
            }
            res.j_best.push(*j.iter().min().unwrap());
            res.logit_best.push(*lg.iter().min().unwrap());
            res.final_rank.push(best(&logits[2 * nl]));
            res.scored += 1;
        }
    }
    res.secs = t0.elapsed().as_secs_f64();
    Ok(res)
}

pub fn report(r: &SetResult, layers: &[i32]) {
    println!(
        "\n== {}: {} items, {} intermediates scored, {} skipped (no single-token form), {:.0} s",
        r.name,
        r.items,
        r.scored,
        r.skipped.len(),
        r.secs
    );
    for s in &r.skipped {
        println!("   skipped {s}");
    }
    println!(
        "{:<12} {:>8} {:>8} {:>9} {:>10}",
        "lens", "pass@1", "pass@10", "pass@100", "AUC(1000)"
    );
    for (name, ranks) in [
        ("Jacobian", &r.j_best),
        ("logit", &r.logit_best),
        ("final only", &r.final_rank),
    ] {
        println!(
            "{:<12} {:>8.3} {:>8.3} {:>9.3} {:>10.3}",
            name,
            pass_at(ranks, 1),
            pass_at(ranks, 10),
            pass_at(ranks, 100),
            auc(ranks, 1000)
        );
    }
    let total = r.scored.max(1) as f64;
    let j: Vec<String> = layers
        .iter()
        .zip(&r.j_by_layer)
        .map(|(l, n)| format!("{l}:{:.2}", *n as f64 / total))
        .collect();
    let g: Vec<String> = layers
        .iter()
        .zip(&r.logit_by_layer)
        .map(|(l, n)| format!("{l}:{:.2}", *n as f64 / total))
        .collect();
    println!("recovered at 10, per block, Jacobian: {}", j.join(" "));
    println!("recovered at 10, per block, logit:    {}", g.join(" "));
}

/// Load the lens's transports into the readout.
pub fn load_lens(readout: &mut Readout, lens: &mut Lens) -> Result<Vec<i32>> {
    let layers = lens.header.layers.clone();
    let mats: Vec<(i32, Vec<u16>)> = layers
        .iter()
        .map(|&l| Ok((l, lens.matrix(l)?)))
        .collect::<Result<_>>()?;
    let refs: Vec<(i32, &[u16])> = mats.iter().map(|(l, m)| (*l, m.as_slice())).collect();
    readout.load_transports(&refs)?;
    Ok(layers)
}

/// A readout over the model's own tensors, once the capture has seen them
/// (one decode first).
pub fn readout_for(llm: &mut Llm) -> Result<Readout> {
    llm.clear();
    let t = llm.tokenize("The", false)?;
    llm.decode(&[Lane {
        seq: 0,
        tokens: &t,
        pos0: 0,
        logits: true,
    }])?;
    let c = llm.capture().context("no capture installed")?;
    c.take();
    let eps = c
        .norm_eps
        .context("the output norm's epsilon was not seen")?;
    Readout::new(cuda_device()?, c.unembed, c.output_norm, eps, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_is_one_for_always_first_and_zero_past_k() {
        assert!((auc(&[0, 0, 0], 1000) - 1.0).abs() < 1e-12);
        assert_eq!(auc(&[1000, 5000], 1000), 0.0);
        // Rank 9 (k = 10) contributes (ln 1000 - ln 10) / ln 1000 = 2/3.
        assert!((auc(&[9], 1000) - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(pass_at(&[0, 5, 50], 10), 2.0 / 3.0);
    }

    #[test]
    fn order_ops_expands_and_others_do_not() {
        assert_eq!(synonyms("multihop", "Brazil"), vec!["Brazil".to_string()]);
        assert_eq!(
            synonyms("order-ops", "12"),
            vec!["12".to_string(), "twelve".to_string()]
        );
        assert!(synonyms("order-ops", "multiplication").contains(&"*".to_string()));
    }
}
