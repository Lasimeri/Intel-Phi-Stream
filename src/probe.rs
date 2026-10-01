//! `phi-stream probe`: what the split gives on this machine. The placement
//! and the GPU's memory, then the rates: a prompt read alone, tokens
//! generated alone, and cycles that carry one live token and a chunk of
//! another sequence's prompt, for several chunk sizes, which is how the
//! engine reads while it thinks. Every number is this run's. See probe.md.

use std::time::Instant;

use anyhow::{Context as _, Result};

use crate::llm::{Lane, Llm};
use crate::readout::{cuda_device, Group, Readout};
use crate::split::gib;

/// The per-token readout's cost in the probe: after every decode that
/// asked for a token, the captured blocks are transported, normed,
/// unembedded and ranked, as the live stream will do. The transports are
/// synthetic (a deterministic pseudo-random float16 matrix per block, the
/// size of a real one): this measures cost, not meaning.
pub struct MindCost {
    pub layers: Vec<i32>,
    readout: Option<Readout>,
    /// Readouts made, and the time they took (seconds).
    pub n: usize,
    pub secs: f64,
}

impl MindCost {
    pub fn new(layers: Vec<i32>) -> Self {
        Self {
            layers,
            readout: None,
            n: 0,
            secs: 0.0,
        }
    }

    fn step(&mut self, llm: &mut Llm) -> Result<()> {
        let t0 = Instant::now();
        let cap = llm.capture().context("no capture installed")?;
        if let Some(e) = &cap.error {
            anyhow::bail!("the capture failed: {e}");
        }
        let (outputs, _) = cap.take();
        if self.readout.is_none() {
            let eps = cap
                .norm_eps
                .context("the output norm's epsilon was not seen")?;
            let (unembed, norm) = (cap.unembed, cap.output_norm);
            let d = 2048usize;
            let mats: Vec<(i32, Vec<u16>)> = self
                .layers
                .iter()
                .map(|&l| (l, synthetic_transport(d, l as u64)))
                .collect();
            let refs: Vec<(i32, &[u16])> = mats.iter().map(|(l, m)| (*l, m.as_slice())).collect();
            self.readout = Some(Readout::new(cuda_device()?, unembed, norm, eps, &refs)?);
        }
        let r = self.readout.as_mut().unwrap();
        for o in outputs {
            let groups: Vec<Group> = o
                .layers
                .iter()
                .map(|(l, h)| Group {
                    transport: Some(*l),
                    normed: false,
                    columns: h.as_slice(),
                })
                .collect();
            std::hint::black_box(r.top(&groups, 8)?);
            self.n += 1;
        }
        self.secs += t0.elapsed().as_secs_f64();
        Ok(())
    }
}

/// A `d x d` float16 matrix of small pseudo-random entries (xorshift64),
/// for timing only.
fn synthetic_transport(d: usize, seed: u64) -> Vec<u16> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ seed.wrapping_mul(0x2545_f491_4f6c_dd1d);
    (0..d * d)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // float16 with exponent 0b01001 (about 1/64) and a random mantissa and sign.
            ((x as u16) & 0x83ff) | (0x09 << 10)
        })
        .collect()
}

/// Read `tokens` into `seq` from `pos0` in chunks of `chunk`, one cycle
/// each; returns tokens per second and the batch row of the last token
/// (its logits are the next token's).
fn read(llm: &mut Llm, seq: i32, tokens: &[i32], pos0: i32, chunk: usize) -> Result<(f64, i32)> {
    let t0 = Instant::now();
    let mut pos = pos0;
    let n = tokens.chunks(chunk).count();
    let mut row = 0;
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
    Ok((tokens.len() as f64 / t0.elapsed().as_secs_f64(), row))
}

pub fn probe(
    llm: &mut Llm,
    prompt: &str,
    n_gen: usize,
    chunks: &[usize],
    mind: &mut Option<MindCost>,
) -> Result<()> {
    let s = &llm.sizes;
    let p = &llm.split;
    println!(
        "model: {} ({} blocks, {:.2} GiB)",
        s.arch,
        p.n_blocks,
        gib(s.block.iter().sum::<u64>() + s.other)
    );
    println!(
        "GPU: {:.2} GiB free of {:.2} at the plan; weights there {:.2} GiB: everything but experts, plus the experts of blocks {}",
        gib(llm.vram.0),
        gib(llm.vram.1),
        gib(p.gpu_bytes),
        crate::split::ranges(&p.gpu_set)
    );
    println!(
        "host memory and the cards: the experts of the other blocks, {:.2} GiB (the cards keep their share of these)",
        gib(p.host_bytes)
    );
    println!(
        "context: {} cells, {} KiB of K and V per token at float16, {:.1} MiB of recurrent state per sequence slot, batch {}",
        llm.n_ctx(),
        s.kv_per_token_f16 / 1024,
        s.recurrent_per_seq as f64 / (1u64 << 20) as f64,
        llm.batch_cap()
    );

    let tokens = llm.tokenize(prompt, true)?;
    let cap = llm.batch_cap();
    // The cards upload their shares at the first multiply: warm up on a
    // few tokens and time nothing of it.
    let warm = tokens.len().min(8);
    let t0 = Instant::now();
    read(llm, 0, &tokens[..warm], 0, cap)?;
    println!(
        "first multiply (the cards' upload included): {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    llm.clear();

    let (pp, row) = read(llm, 0, &tokens, 0, cap)?;
    if let Some(m) = mind.as_mut() {
        // The prompt's end asked for a token: its readout, untimed (the
        // readout's first use starts its backend).
        m.step(llm)?;
        m.n = 0;
        m.secs = 0.0;
    }
    println!(
        "prompt alone, {} tokens in chunks of {}: {:.1} tok/s",
        tokens.len(),
        cap,
        pp
    );

    // Generation alone, greedy, from the prompt.
    let mut pos = tokens.len() as i32;
    let mut next = llm.greedy(row, true)?;
    let t0 = Instant::now();
    for _ in 0..n_gen {
        let rows = llm.decode(&[Lane {
            seq: 0,
            tokens: &[next],
            pos0: pos,
            logits: true,
        }])?;
        next = llm.greedy(rows[0], true)?;
        if let Some(m) = mind.as_mut() {
            m.step(llm)?;
        }
        pos += 1;
    }
    let tg_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_gen as f64;
    if let Some(m) = mind.as_mut() {
        println!(
            "the readout of blocks {:?}, {} tokens: {:.2} ms per token of the {:.1} ms",
            m.layers,
            m.n,
            m.secs * 1000.0 / m.n.max(1) as f64,
            tg_ms
        );
        m.n = 0;
        m.secs = 0.0;
    }
    println!(
        "generation alone, {} tokens: {:.1} ms per token, {:.1} tok/s",
        n_gen,
        tg_ms,
        1000.0 / tg_ms
    );

    // Reading while thinking: sequence 1 gets the live sequence's prefix
    // (cells shared, the recurrent state copied), then every cycle carries
    // the live token and a chunk of the prompt again at the positions the
    // reading sequence is at.
    for &chunk in chunks {
        if chunk + 1 > cap {
            println!("chunk {chunk}: larger than the batch allows, skipped");
            continue;
        }
        llm.seq_rm(1, -1, -1);
        llm.seq_cp(0, 1, 0, pos);
        let mut rpos = pos;
        let mut cycles = 0usize;
        let mut live = next;
        let t0 = Instant::now();
        let mut read_tokens = 0usize;
        for c in tokens.chunks(chunk) {
            let rows = llm.decode(&[
                Lane {
                    seq: 0,
                    tokens: &[live],
                    pos0: pos,
                    logits: true,
                },
                Lane {
                    seq: 1,
                    tokens: c,
                    pos0: rpos,
                    logits: false,
                },
            ])?;
            live = llm.greedy(rows[0], true)?;
            if let Some(m) = mind.as_mut() {
                m.step(llm)?;
            }
            pos += 1;
            rpos += c.len() as i32;
            read_tokens += c.len();
            cycles += 1;
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "reading while thinking, chunks of {chunk}: {:.1} ms per cycle, the stream {:.1} tok/s, the reading {:.1} tok/s ({} tokens, {} cycles)",
            dt * 1000.0 / cycles as f64,
            cycles as f64 / dt,
            read_tokens as f64 / dt,
            read_tokens,
            cycles
        );
        next = live;
    }
    Ok(())
}
