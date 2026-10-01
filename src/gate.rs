//! `phi-stream gate`: does the composition give the same next token as a
//! straight sequence? A prompt A is read into the live sequence and T
//! thoughts generated greedily; a text B is read beside it the engine's
//! way (a sequence given A's cells and state, chunks of `chunk`); the
//! composed sequence (A, then B, then the T thoughts caught up) is
//! compared with a fresh sequence fed A B thoughts in one go: the logits
//! of the next token and K further greedy tokens. Everything greedy, so
//! any difference is the composition's, or the kernels' rounding across
//! batch sizes, which the report shows as the logit margin. See gate.md.

use anyhow::{bail, Context as _, Result};

use crate::llm::{argmax, Lane, Llm};
use crate::readout::{compare, cuda_device, Group, Readout};

/// Feed `tokens` into `seq` from `pos0` in chunks; the batch row of the
/// last token comes back.
fn feed(llm: &mut Llm, seq: i32, tokens: &[i32], pos0: i32, chunk: usize) -> Result<i32> {
    let n = tokens.chunks(chunk).count();
    let mut row = 0;
    let mut pos = pos0;
    for (i, c) in tokens.chunks(chunk).enumerate() {
        let rows = llm.decode(&[Lane {
            seq,
            tokens: c,
            pos0: pos,
            logits: i + 1 == n,
        }])?;
        if let Some(&r) = rows.first() {
            row = r;
        }
        pos += c.len() as i32;
    }
    Ok(row)
}

/// `k` greedy tokens after `row`'s logits in `seq` from `pos0`.
fn continue_greedy(llm: &mut Llm, seq: i32, row: i32, pos0: i32, k: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(k);
    let mut next = llm.greedy(row, true)?;
    for pos in pos0..pos0 + k as i32 {
        out.push(next);
        let rows = llm.decode(&[Lane {
            seq,
            tokens: &[next],
            pos0: pos,
            logits: true,
        }])?;
        next = llm.greedy(rows[0], true)?;
    }
    Ok(out)
}

pub fn gate(
    llm: &mut Llm,
    a: &str,
    b: &str,
    thoughts: usize,
    chunk: usize,
    k: usize,
) -> Result<()> {
    let cap = llm.batch_cap();
    let ta = llm.tokenize(a, true)?;
    let tb = llm.tokenize(b, true)?;
    println!(
        "A: {} tokens, B: {} tokens, {thoughts} thoughts, chunks of {chunk}, {k} tokens compared",
        ta.len(),
        tb.len()
    );

    // The live sequence: A, then T greedy thoughts, the last cycle's
    // logits kept for the pending token.
    llm.clear();
    let row = feed(llm, 0, &ta, 0, cap)?;
    let mut g = Vec::new();
    let mut next = llm.greedy(row, true)?;
    let mut pos = ta.len() as i32;
    // Reading B beside the thoughts, as the engine does: sequence 1 gets
    // A's cells and state, its first chunk alone, then one chunk a cycle.
    llm.seq_cp(0, 1, 0, pos);
    let mut fed = 0usize;
    let mut rpos = pos;
    let first = tb.len().min(chunk);
    llm.decode(&[Lane {
        seq: 1,
        tokens: &tb[..first],
        pos0: rpos,
        logits: false,
    }])?;
    fed += first;
    rpos += first as i32;
    let mut produced = 0usize;
    while fed < tb.len() || produced < thoughts {
        let mut lanes = Vec::new();
        if produced < thoughts {
            lanes.push(Lane {
                seq: 0,
                tokens: std::slice::from_ref(&next),
                pos0: pos,
                logits: true,
            });
        }
        let n = chunk.min(tb.len() - fed);
        let c = tb[fed..fed + n].to_vec();
        if n > 0 {
            lanes.push(Lane {
                seq: 1,
                tokens: &c,
                pos0: rpos,
                logits: false,
            });
        }
        let rows = llm.decode(&lanes)?;
        if produced < thoughts {
            g.push(next);
            next = llm.greedy(rows[0], true)?;
            pos += 1;
            produced += 1;
        }
        fed += n;
        rpos += n as i32;
    }
    // Compose: sequence 2 = A's cells, B's cells and state, then the
    // thoughts and the pending token caught up in chunks.
    llm.seq_cp(0, 2, 0, ta.len() as i32);
    llm.seq_cp(1, 2, ta.len() as i32, (ta.len() + tb.len()) as i32);
    llm.seq_rm(1, -1, -1);
    let mut tail = g.clone();
    tail.push(next);
    let row_c = feed(llm, 2, &tail, (ta.len() + tb.len()) as i32, chunk)?;
    let lc: Vec<f32> = llm.logits(row_c)?.to_vec();
    let end = (ta.len() + tb.len() + tail.len()) as i32;
    let cont_c = continue_greedy(llm, 2, row_c, end, k)?;

    // Straight: sequence 3 fed A, B, the thoughts and the pending token in one go.
    let mut all = ta.clone();
    all.extend_from_slice(&tb);
    all.extend_from_slice(&tail);
    llm.seq_rm(3, -1, -1);
    let row_s = feed(llm, 3, &all, 0, cap)?;
    let ls: Vec<f32> = llm.logits(row_s)?.to_vec();
    let cont_s = continue_greedy(llm, 3, row_s, end, k)?;

    // Control: the straight sequence again, fed in chunks of `chunk` as
    // the composed one was, against itself at the full batch. The size of
    // this difference is what the kernels' rounding across batch sizes
    // costs; the composition is judged against it.
    llm.seq_rm(1, -1, -1);
    let row_k = feed(llm, 1, &all, 0, chunk)?;
    let lk: Vec<f32> = llm.logits(row_k)?.to_vec();
    let max_k = lk
        .iter()
        .zip(&ls)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let max_ck = lk
        .iter()
        .zip(&lc)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("control: the straight sequence in chunks of {chunk} against itself at the full batch: max |delta logit| {max_k:.4}, top token {}; against the composed {max_ck:.4}", argmax(&lk));

    let top_c = argmax(&lc);
    let top_s = argmax(&ls);
    let max_d = lc
        .iter()
        .zip(&ls)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let margin = {
        let mut v = ls.clone();
        v.sort_by(|x, y| y.total_cmp(x));
        v[0] - v[1]
    };
    println!(
        "next token: composed {} ({}), straight {} ({}); max |delta logit| {:.4}; the straight top-2 margin {:.4}",
        top_c,
        llm.text(&[top_c]).escape_debug(),
        top_s,
        llm.text(&[top_s]).escape_debug(),
        max_d,
        margin
    );
    let same = cont_c
        .iter()
        .zip(&cont_s)
        .take_while(|(x, y)| x == y)
        .count();
    println!("{k} greedy tokens after: {same} identical before the first difference");
    println!("composed: {}", llm.text(&cont_c).escape_debug());
    println!("straight: {}", llm.text(&cont_s).escape_debug());
    println!("thoughts while reading: {}", llm.text(&g).escape_debug());
    if top_c != top_s {
        println!("VERDICT: the next token differs");
    } else {
        println!("VERDICT: same next token");
    }
    Ok(())
}

/// Decode `tokens` one at a time into `seq` from `pos0` (the chunking the
/// live sequence's own tokens get); the batch row of the last.
fn feed_one_by_one(llm: &mut Llm, seq: i32, tokens: &[i32], pos0: i32) -> Result<i32> {
    let mut row = 0;
    for (i, &t) in tokens.iter().enumerate() {
        let rows = llm.decode(&[Lane {
            seq,
            tokens: &[t],
            pos0: pos0 + i as i32,
            logits: i + 1 == tokens.len(),
        }])?;
        if let Some(&r) = rows.first() {
            row = r;
        }
    }
    Ok(row)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// `gate --snapshot`: a sequence copied from the live one at position p
/// keeps the state at p while the live one goes on writing. The live
/// sequence decodes A and `before` greedy tokens; a snapshot is copied
/// (`seq_cp`, the recurrent state shared until written); the live one
/// decodes the token it chose and `after` more; then the snapshot decodes
/// the second-best token in its place. Compared with a control that
/// decodes exactly the same tokens in exactly the same chunks without any
/// snapshot: the snapshot's logits, and the live sequence's own logits
/// after the copy. Any difference is the snapshot's (or the cache's cell
/// layout, which the control does not share), and is reported.
pub fn gate_snapshot(llm: &mut Llm, a: &str, before: usize, after: usize) -> Result<()> {
    let cap = llm.batch_cap();
    let ta = llm.tokenize(a, true)?;
    llm.clear();
    let mut row = feed(llm, 0, &ta, 0, cap)?;
    let mut gen = Vec::new();
    let mut pos = ta.len() as i32;
    for _ in 0..before {
        let t = llm.greedy(row, true)?;
        gen.push(t);
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: &[t],
            pos0: pos,
            logits: true,
        }])?;
        row = rows[0];
        pos += 1;
    }
    // At the snapshot: the live token is chosen (c), the alternative is the
    // second best (x).
    let l = llm.logits(row)?.to_vec();
    let c = argmax(&l);
    let mut l2 = l.clone();
    l2[c as usize] = f32::NEG_INFINITY;
    let x = argmax(&l2);
    let p1 = pos; // where c (or x) goes
    llm.seq_rm(1, -1, -1);
    llm.seq_cp(0, 1, -1, -1);
    // The live sequence goes on: c, then `after` greedy tokens.
    let mut cont = vec![c];
    let mut row_live = feed_one_by_one(llm, 0, &[c], p1)?;
    for _ in 0..after {
        let t = llm.greedy(row_live, true)?;
        cont.push(t);
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: &[t],
            pos0: p1 + cont.len() as i32 - 1,
            logits: true,
        }])?;
        row_live = rows[0];
    }
    let live_after = llm.logits(row_live)?.to_vec();
    // The snapshot places x.
    let row_snap = feed_one_by_one(llm, 1, &[x], p1)?;
    let snap = llm.logits(row_snap)?.to_vec();
    // Controls, without any snapshot: A in chunks, the same tokens one by
    // one, then x (for the snapshot) or c and the continuation (for the live).
    llm.seq_rm(2, -1, -1);
    feed(llm, 2, &ta, 0, cap)?;
    let mut ctl_tail = gen.clone();
    ctl_tail.push(x);
    let row_ctl = feed_one_by_one(llm, 2, &ctl_tail, ta.len() as i32)?;
    let ctl_snap = llm.logits(row_ctl)?.to_vec();
    llm.seq_rm(2, -1, -1);
    llm.seq_rm(3, -1, -1);
    feed(llm, 3, &ta, 0, cap)?;
    let mut live_tail = gen.clone();
    live_tail.extend_from_slice(&cont);
    let row_ctl_live = feed_one_by_one(llm, 3, &live_tail, ta.len() as i32)?;
    let ctl_live = llm.logits(row_ctl_live)?.to_vec();
    llm.seq_rm(1, -1, -1);
    llm.seq_rm(3, -1, -1);

    let d_snap = max_diff(&snap, &ctl_snap);
    let d_live = max_diff(&live_after, &ctl_live);
    let same_snap = argmax(&snap) == argmax(&ctl_snap);
    let same_live = argmax(&live_after) == argmax(&ctl_live);
    println!(
        "A: {} tokens, {before} tokens before the snapshot, {after} after it on the live sequence; the snapshot placed {:?} where the live sequence placed {:?}",
        ta.len(),
        llm.text(&[x]),
        llm.text(&[c])
    );
    println!(
        "snapshot against its control: largest |delta logit| {d_snap:.6}, next token {}",
        if same_snap { "same" } else { "DIFFERENT" }
    );
    println!("live sequence after the copy against its control: largest |delta logit| {d_live:.6}, next token {}", if same_live { "same" } else { "DIFFERENT" });
    if !same_snap || !same_live || d_snap > 0.5 || d_live > 0.5 {
        bail!("the snapshot did not keep its state, or disturbed the live sequence");
    }
    println!(
        "PASS: a copied sequence keeps the state at the copy while the live one goes on writing"
    );
    Ok(())
}

/// The readout and the graph agree to rounding (`check.rs`); a row that is
/// another lane's misses by whole logits.
const IDENTITY_MAX: f32 = 1e-2;

/// Every output row the capture holds since the last call, read as it is
/// (no transport), must give llama's own logits for the lane asking at the
/// same index of `rows` (the batch's lane order). The largest difference
/// comes back.
fn identity(llm: &mut Llm, readout: &mut Option<Readout>, rows: &[i32], what: &str) -> Result<f32> {
    let last = llm.n_layer() - 1;
    let outputs = {
        let cap = llm.capture().context("no capture installed")?;
        if let Some(e) = &cap.error {
            bail!("the capture failed: {e}");
        }
        if readout.is_none() {
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
        cap.take().0
    };
    if outputs.len() != rows.len() {
        bail!(
            "{what}: {} output rows captured for {} lanes asking",
            outputs.len(),
            rows.len()
        );
    }
    let mut worst = 0f32;
    for (i, (o, &row)) in outputs.iter().zip(rows).enumerate() {
        let h = o
            .layers
            .iter()
            .find(|(l, _)| *l == last)
            .map(|(_, v)| v.as_slice())
            .with_context(|| format!("{what}: block {last} not captured"))?;
        let ours = readout.as_mut().unwrap().logits(&[Group {
            transport: None,
            normed: false,
            columns: h,
        }])?;
        let (same, d) = compare(&ours[0], llm.logits(row)?, 10)?;
        if !same || d > IDENTITY_MAX {
            bail!("{what}: captured row {i} is not lane {i}'s (batch row {row}): top 10 {}, largest difference {d:.4}", if same { "same" } else { "different" });
        }
        worst = worst.max(d);
    }
    Ok(worst)
}

/// Drop what the capture holds (decodes whose rows nobody reads).
fn discard(llm: &mut Llm) {
    if let Some(c) = llm.capture() {
        c.take();
    }
}

/// The best token of `row` other than `c` and the end-of-generation ones.
fn second_best(llm: &Llm, row: i32, c: i32) -> Result<i32> {
    let mut l = llm.logits(row)?.to_vec();
    l[c as usize] = f32::NEG_INFINITY;
    loop {
        let x = argmax(&l);
        if !llm.is_eog(x) {
            return Ok(x);
        }
        l[x as usize] = f32::NEG_INFINITY;
    }
}

/// KL(p || q) of the softmaxes of two logits rows, in nats.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (za, zb) = (lse(a), lse(b));
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let la = x as f64 - za;
            la.exp() * (la - (y as f64 - zb))
        })
        .sum::<f64>()
        .max(0.0)
}

/// One episode's tokens and the logits its lanes gave.
#[derive(Default)]
struct Tape {
    /// The live tokens after the prompt, before the copy; the token in
    /// question and the second best.
    gap: Vec<i32>,
    c: i32,
    x: i32,
    /// The live tokens decoded beside D (the token in question first), and
    /// D's lanes after its first token (the question's rest, then answer
    /// tokens one at a time).
    live: Vec<i32>,
    d: Vec<Vec<i32>>,
    /// Tokens decoded alone after the episode ended: the live sequence's
    /// next ones when kept; the second best and its next ones on the
    /// snapshot when rewound.
    after: Vec<i32>,
    l_live: Vec<Vec<f32>>,
    l_d: Vec<Vec<f32>>,
    l_after: Vec<Vec<f32>>,
}

fn take_seq(llm: &Llm, free: &mut Vec<i32>, what: &str) -> Result<i32> {
    let s = free
        .pop()
        .with_context(|| format!("no sequence free for {what}"))?;
    if llm.seq_pos_max(s) != -1 {
        bail!("sequence {s} is not empty before {what}");
    }
    Ok(s)
}

fn give_seq(llm: &mut Llm, free: &mut Vec<i32>, s: i32) {
    llm.seq_rm(s, -1, -1);
    free.push(s);
}

/// One decode of one token alone in `seq` at `pos`, its row checked; its logits.
fn alone(
    llm: &mut Llm,
    readout: &mut Option<Readout>,
    seq: i32,
    t: i32,
    pos: usize,
) -> Result<(i32, Vec<f32>)> {
    let rows = llm.decode(&[Lane {
        seq,
        tokens: &[t],
        pos0: pos as i32,
        logits: true,
    }])?;
    identity(llm, readout, &rows, "alone")?;
    Ok((rows[0], llm.logits(rows[0])?.to_vec()))
}

/// The prompt in chunks and then `gap` one token at a time: how the live
/// sequence is fed before a copy. The last row.
fn prefill(
    llm: &mut Llm,
    readout: &mut Option<Readout>,
    seq: i32,
    ta: &[i32],
    gap: &[i32],
) -> Result<i32> {
    let mut row = feed(llm, seq, ta, 0, llm.batch_cap())?;
    identity(llm, readout, &[row], "the prompt")?;
    for (i, &t) in gap.iter().enumerate() {
        row = alone(llm, readout, seq, t, ta.len() + i)?.0;
    }
    Ok(row)
}

/// One cycle of the live token and D's lane, the live lane first; each
/// captured row checked to be its own lane's. The two logits rows.
fn beside_cycle(
    llm: &mut Llm,
    readout: &mut Option<Readout>,
    live: (i32, i32, usize),
    d: (i32, &[i32], usize),
) -> Result<(i32, i32, Vec<f32>, Vec<f32>)> {
    let rows = llm.decode(&[
        Lane {
            seq: live.0,
            tokens: &[live.1],
            pos0: live.2 as i32,
            logits: true,
        },
        Lane {
            seq: d.0,
            tokens: d.1,
            pos0: d.2 as i32,
            logits: true,
        },
    ])?;
    identity(llm, readout, &rows, "beside")?;
    Ok((
        rows[0],
        rows[1],
        llm.logits(rows[0])?.to_vec(),
        llm.logits(rows[1])?.to_vec(),
    ))
}

/// D's first token alone: it shares the live sequence's recurrent state
/// (or, in a control, it is D's own prefill's) until it writes its own.
fn d_first(llm: &mut Llm, d: i32, t: i32, at: usize) -> Result<()> {
    llm.decode(&[Lane {
        seq: d,
        tokens: &[t],
        pos0: at as i32,
        logits: false,
    }])?;
    discard(llm);
    Ok(())
}

/// An episode as the engine runs one: the live sequence from the prompt,
/// `gap` greedy tokens alone, a snapshot S and a deliberation D copied
/// from it before the token it chose, D's first token alone, `beside`
/// cycles of the live token and D's lane, then kept (S and D dropped, the
/// live sequence goes on alone) or rewound (the live sequence and D
/// dropped, S places the second-best token and goes on alone).
#[allow(clippy::too_many_arguments)]
fn episode_with_copies(
    llm: &mut Llm,
    readout: &mut Option<Readout>,
    free: &mut Vec<i32>,
    ta: &[i32],
    q: &[i32],
    gap: usize,
    beside: usize,
    after: usize,
    rewind: bool,
) -> Result<Tape> {
    let mut tape = Tape::default();
    let live = take_seq(llm, free, "the live sequence")?;
    let mut row = prefill(llm, readout, live, ta, &[])?;
    let mut next = llm.greedy(row, true)?;
    for _ in 0..gap {
        tape.gap.push(next);
        row = alone(llm, readout, live, next, ta.len() + tape.gap.len() - 1)?.0;
        next = llm.greedy(row, true)?;
    }
    let at = ta.len() + gap;
    tape.c = next;
    tape.x = second_best(llm, row, next)?;
    let s = take_seq(llm, free, "the snapshot")?;
    let d = take_seq(llm, free, "the deliberation")?;
    llm.seq_cp(live, s, -1, -1);
    llm.seq_cp(live, d, -1, -1);
    d_first(llm, d, q[0], at)?;
    let mut d_len = 1;
    let mut d_next = 0;
    for i in 0..beside {
        let lane: Vec<i32> = if i == 0 {
            q[1..].to_vec()
        } else {
            vec![d_next]
        };
        let (rl, rd, ll, ld) =
            beside_cycle(llm, readout, (live, next, at + i), (d, &lane, at + d_len))?;
        tape.live.push(next);
        tape.l_live.push(ll);
        tape.l_d.push(ld);
        d_len += lane.len();
        tape.d.push(lane);
        next = llm.greedy(rl, true)?;
        d_next = llm.greedy(rd, true)?;
    }
    give_seq(llm, free, d);
    let (seq, mut pos, mut t) = if rewind {
        give_seq(llm, free, live);
        (s, at, tape.x)
    } else {
        give_seq(llm, free, s);
        (live, at + beside, next)
    };
    for _ in 0..after {
        let (r, l) = alone(llm, readout, seq, t, pos)?;
        tape.after.push(t);
        tape.l_after.push(l);
        pos += 1;
        t = llm.greedy(r, true)?;
    }
    give_seq(llm, free, seq);
    Ok(tape)
}

/// The same episode without the snapshot: the tape's tokens in the same
/// decodes of the same shapes, in the same cells (D copied from the live
/// sequence as in the run, so the prefix's cells are shared the same way),
/// the live sequence then owning its recurrent state where in the run it
/// shares it with the snapshot until its first write, inside a two-lane
/// batch: the path under test. Rewound, the snapshot's part is played by a
/// sequence prefilled as the live one was, into the same cells. `swap`
/// places the token in question instead of the second best there (the
/// resolution check). The logits come back in a tape.
#[allow(clippy::too_many_arguments)]
fn episode_without_snapshot(
    llm: &mut Llm,
    readout: &mut Option<Readout>,
    free: &mut Vec<i32>,
    ta: &[i32],
    q: &[i32],
    tape: &Tape,
    rewind: bool,
    swap: bool,
) -> Result<Tape> {
    let mut out = Tape::default();
    let at = ta.len() + tape.gap.len();
    let live = take_seq(llm, free, "the control's live sequence")?;
    prefill(llm, readout, live, ta, &tape.gap)?;
    let d = take_seq(llm, free, "the control's deliberation")?;
    llm.seq_cp(live, d, -1, -1);
    d_first(llm, d, q[0], at)?;
    let mut d_len = 1;
    for (i, lane) in tape.d.iter().enumerate() {
        let (_, _, ll, ld) = beside_cycle(
            llm,
            readout,
            (live, tape.live[i], at + i),
            (d, lane, at + d_len),
        )?;
        out.l_live.push(ll);
        out.l_d.push(ld);
        d_len += lane.len();
    }
    give_seq(llm, free, d);
    let (seq, mut pos) = if rewind {
        give_seq(llm, free, live);
        let s = take_seq(llm, free, "the control's snapshot")?;
        prefill(llm, readout, s, ta, &tape.gap)?;
        (s, at)
    } else {
        (live, at + tape.live.len())
    };
    for (i, &t) in tape.after.iter().enumerate() {
        let t = if i == 0 && swap { tape.c } else { t };
        out.l_after.push(alone(llm, readout, seq, t, pos)?.1);
        pos += 1;
    }
    give_seq(llm, free, seq);
    Ok(out)
}

/// The largest |delta logit| and KL between two tapes' rows, per lane
/// (live, D, after), and whether every row's next token agrees.
fn tape_diff(a: &Tape, b: &Tape) -> ([f32; 3], [f64; 3], bool) {
    let mut diff = [0f32; 3];
    let mut kls = [0f64; 3];
    let mut same = true;
    for (k, (ra, rb)) in [
        (&a.l_live, &b.l_live),
        (&a.l_d, &b.l_d),
        (&a.l_after, &b.l_after),
    ]
    .into_iter()
    .enumerate()
    {
        for (x, y) in ra.iter().zip(rb) {
            diff[k] = diff[k].max(max_diff(x, y));
            kls[k] = kls[k].max(kl(y, x));
            same &= argmax(x) == argmax(y);
        }
    }
    (diff, kls, same)
}

/// `gate --reflect`: the reflection loop's lanes (reflect.md) keep every
/// sequence's state, and every captured row is its own lane's. Episodes,
/// kept and rewound in turn, each from a fresh prompt (the gap grows by
/// five tokens an episode, and the sequences and cells freed by one are
/// taken by the next, so the cache's layout drifts): the lanes as the
/// engine runs them, with the snapshot; then twice the same tokens in the
/// same decodes in the same cells without it (`episode_without_snapshot`).
/// The copies pass when they match the control as closely as the control
/// matches itself (the run-to-run floor, which on this backend is exactly
/// zero; at least 0.01 in logit) with the same next tokens.
/// The resolution: on the rewound path, the control again with the token
/// in question placed instead of the second best, whose distance from the
/// lanes must stand far above the limit.
pub fn gate_reflect(
    llm: &mut Llm,
    a: &str,
    question: &str,
    episodes: usize,
    gap: usize,
    beside: usize,
) -> Result<()> {
    const AFTER: usize = 3;
    let ta = llm.tokenize(a, true)?;
    let q = llm.tokenize(question, false)?;
    if q.len() < 2 || q.len() > llm.batch_cap() {
        bail!(
            "the question is {} tokens; it must be 2 to {}",
            q.len(),
            llm.batch_cap()
        );
    }
    llm.clear();
    discard(llm);
    let mut readout = None;
    let mut free = vec![3, 2, 1, 0];
    let mut rows = Vec::new();
    let mut teeth: Vec<f64> = Vec::new();
    for e in 0..episodes {
        let rewind = e % 2 == 1;
        let g = gap + 5 * e;
        let run = episode_with_copies(
            llm,
            &mut readout,
            &mut free,
            &ta,
            &q,
            g,
            beside,
            AFTER,
            rewind,
        )?;
        let ctl =
            episode_without_snapshot(llm, &mut readout, &mut free, &ta, &q, &run, rewind, false)?;
        let ctl2 =
            episode_without_snapshot(llm, &mut readout, &mut free, &ta, &q, &run, rewind, false)?;
        let (d_run, k_run, same) = tape_diff(&run, &ctl);
        let (d_floor, k_floor, same_floor) = tape_diff(&ctl, &ctl2);
        let mut line = format!(
            "episode {e}: {} at position {}: chose {:?}, second {:?}; D {} tokens beside {} live ones\n  copies vs control  max|d| live {:.4} D {:.4} after {:.4}  KL max {:.2e}{}\n  control vs itself  max|d| live {:.4} D {:.4} after {:.4}  KL max {:.2e}{}",
            if rewind { "rewound" } else { "kept" },
            ta.len() + g,
            llm.text(&[run.c]),
            llm.text(&[run.x]),
            1 + run.d.iter().map(Vec::len).sum::<usize>(),
            beside,
            d_run[0],
            d_run[1],
            d_run[2],
            k_run.iter().copied().fold(0.0, f64::max),
            if same { "" } else { "  DIFFERENT next" },
            d_floor[0],
            d_floor[1],
            d_floor[2],
            k_floor.iter().copied().fold(0.0, f64::max),
            if same_floor { "" } else { "  DIFFERENT next" },
        );
        if rewind {
            let wrong =
                episode_without_snapshot(llm, &mut readout, &mut free, &ta, &q, &run, true, true)?;
            let (dw, kw, _) = tape_diff(&run, &wrong);
            line.push_str(&format!(
                "\n  one token different (the chosen one placed, not the second best): max|d| after {:.4}  KL {:.2e}",
                dw[2], kw[2]
            ));
            teeth.push(dw[2] as f64);
        }
        println!("{line}");
        rows.push((d_run, same, d_floor, same_floor));
    }
    let floor = rows.iter().flat_map(|r| r.2).fold(0f32, f32::max);
    let limit = (2.0 * floor).max(0.01);
    let worst = rows.iter().flat_map(|r| r.0).fold(0f32, f32::max);
    let fails = rows
        .iter()
        .filter(|r| !r.1 || r.0.iter().any(|&d| d > limit))
        .count();
    println!("every captured row its own lane's (identity exact in every decode)");
    println!(
        "copies against the control: largest |delta logit| {worst:.4}; the control against itself up to {floor:.4}; limit {limit:.4} and the same next tokens"
    );
    if let Some(t) = teeth.iter().copied().reduce(f64::min) {
        println!(
            "resolution: one token different gives |delta logit| {t:.4} at the least ({:.0} times the limit)",
            t / limit as f64
        );
        if t <= limit as f64 * 4.0 {
            bail!("the comparison cannot resolve a one-token difference: the gate says nothing");
        }
    }
    if rows.iter().any(|r| !r.3) {
        bail!("the control does not reproduce itself: the backend is not deterministic enough for this gate");
    }
    if fails > 0 {
        bail!("{fails} of {episodes} episodes: the copies moved a sequence past the control's own floor");
    }
    println!("PASS: snapshots and deliberations beside the live sequence keep every state, kept or rewound");
    Ok(())
}
