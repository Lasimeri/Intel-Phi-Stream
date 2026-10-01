//! What is on the mind of the stream, token by token: for every token the
//! live sequence places, the residual after a few chosen blocks at that
//! token's own position (`capture.rs`), read through the Jacobian lens
//! (`lens.rs`, `readout.rs`) and ranked on the GPU. Nothing is averaged
//! over tokens: a reading belongs to one position. It runs synchronously in
//! the engine's cycle, right after the decode that placed the token. See
//! mind.md.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context as _, Result};

use crate::lens::Lens;
use crate::llm::Llm;
use crate::readout::{cuda_device, Group, Readout};

/// How the mind is read.
#[derive(Clone, Debug)]
pub struct MindConfig {
    /// The `.jlens` file.
    pub lens: String,
    /// The blocks read through the lens.
    pub layers: Vec<i32>,
    /// Words shown per block.
    pub k: usize,
    /// Also read the final block as it is (the model's own next-token
    /// distribution, its top 64), for the reflection loop's doubt.
    pub final_block: Option<i32>,
}

/// One token's reading.
#[derive(Clone, Debug)]
pub struct Reading {
    /// The position of the token whose residual this is.
    pub pos: i32,
    /// That token's text.
    pub token: String,
    /// Per block, the word-like tokens on its mind, best first, with
    /// log-probabilities under the lens's distribution.
    pub layers: Vec<(i32, Vec<(String, f32)>)>,
    /// The model's own next-token distribution at this token, its top 64
    /// (token, log-probability), when the final block is read.
    pub model_top: Vec<(i32, f32)>,
    /// The readout's own time, milliseconds.
    pub ms: f32,
    /// When the token existed (its decode done), microseconds of real time.
    pub t_us: i64,
}

pub struct Mind {
    readout: Readout,
    pub cfg: MindConfig,
    /// Per token id: whether it decodes to a word (shown); ranks stay over
    /// the whole vocabulary.
    wordlike: Vec<bool>,
    /// The token texts, for display.
    texts: Vec<String>,
    log: Option<File>,
}

/// The reference's display rule (`_meaningful_token_mask`, jlens/vis.py):
/// the token's text, stripped, is non-empty, not a special token, and made
/// of alphanumeric characters with apostrophes or hyphens only inside.
pub fn is_wordlike(text: &str) -> bool {
    let s = text.trim();
    if s.is_empty() || s.contains("<|") || (s.starts_with('<') && s.ends_with('>')) {
        return false;
    }
    let n = s.chars().count();
    s.chars().enumerate().all(|(i, c)| {
        c.is_alphanumeric() || (i > 0 && i + 1 < n && matches!(c, '\'' | '-' | '\u{2019}'))
    })
}

/// GPU top-k fetched per block before the word-like filter.
const FETCH: usize = 64;

impl Mind {
    /// After the first decode (the capture must have seen the model's
    /// unembedding): the readout, the lens's transports for the chosen
    /// blocks, the vocabulary's word mask.
    pub fn new(llm: &mut Llm, cfg: MindConfig, workspace: &Path) -> Result<Self> {
        let cap = llm
            .capture()
            .context("the mind needs the capture installed")?;
        if let Some(e) = &cap.error {
            bail!("the capture failed: {e}");
        }
        let eps = cap
            .norm_eps
            .context("the output norm's epsilon was not seen (decode once first)")?;
        let (unembed, norm) = (cap.unembed, cap.output_norm);
        let mut lens = Lens::open(&cfg.lens)?;
        let mats: Vec<(i32, Vec<u16>)> = cfg
            .layers
            .iter()
            .map(|&l| Ok((l, lens.matrix(l)?)))
            .collect::<Result<_>>()?;
        let refs: Vec<(i32, &[u16])> = mats.iter().map(|(l, m)| (*l, m.as_slice())).collect();
        let readout = Readout::new(cuda_device()?, unembed, norm, eps, &refs)?;
        let n = readout.n_vocab;
        let mut texts = Vec::with_capacity(n);
        let mut wordlike = Vec::with_capacity(n);
        for t in 0..n as i32 {
            let s = piece(llm, t);
            wordlike.push(is_wordlike(&s));
            texts.push(s.trim().to_string());
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(workspace.join("mind.log"))
            .ok();
        Ok(Self {
            readout,
            cfg,
            wordlike,
            texts,
            log,
        })
    }

    /// The reading of the token just decoded at `pos`, from the capture's
    /// first output row (the live lane comes first in every batch, so its
    /// row is the first: `gate --reflect` checks it); none when the decode
    /// asked for no token.
    pub fn read(&mut self, llm: &mut Llm, pos: i32, token: &str) -> Result<Option<Reading>> {
        let t_us = crate::clock::now_us();
        let t0 = Instant::now();
        let cap = llm.capture().context("no capture installed")?;
        if let Some(e) = &cap.error {
            bail!("the capture failed: {e}");
        }
        let (outputs, _) = cap.take();
        let Some(o) = outputs.first() else {
            return Ok(None);
        };
        let groups: Vec<Group> = o
            .layers
            .iter()
            .filter(|(l, _)| self.cfg.layers.contains(l))
            .map(|(l, h)| Group {
                transport: Some(*l),
                normed: false,
                columns: h.as_slice(),
            })
            .collect();
        let mut groups = groups;
        let final_col = match self.cfg.final_block {
            Some(f) => match o.layers.iter().find(|(l, _)| *l == f) {
                Some((_, h)) => {
                    groups.push(Group {
                        transport: None,
                        normed: false,
                        columns: h.as_slice(),
                    });
                    true
                }
                None => false,
            },
            None => false,
        };
        let mut tops = self.readout.top(&groups, FETCH)?;
        let model_top = if final_col {
            groups.pop();
            tops.pop().map(|r| r.top).unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut layers = Vec::new();
        for (g, r) in groups.iter().zip(&tops) {
            let words: Vec<(String, f32)> = r
                .top
                .iter()
                .filter(|(t, _)| self.wordlike.get(*t as usize).copied().unwrap_or(false))
                .take(self.cfg.k)
                .map(|(t, p)| (self.texts[*t as usize].clone(), *p))
                .collect();
            layers.push((g.transport.unwrap_or(-1), words));
        }
        let reading = Reading {
            pos,
            token: token.to_string(),
            layers,
            model_top,
            ms: t0.elapsed().as_secs_f32() * 1000.0,
            t_us,
        };
        if let Some(f) = &mut self.log {
            let _ = writeln!(f, "{}", line(&reading));
        }
        Ok(Some(reading))
    }
}

/// A reading as one line (the socket's `mind` line and `mind.log`):
/// `t=MICROSECONDS pos=P ms=M tok=TEXT l27=w:logp,w:logp l29=...`, the
/// token escaped.
pub fn line(r: &Reading) -> String {
    let mut s = format!(
        "t={} pos={} ms={:.2} tok={}",
        r.t_us,
        r.pos,
        r.ms,
        crate::client::escape(&r.token).replace(' ', "\\s")
    );
    for (l, words) in &r.layers {
        let w: Vec<String> = words.iter().map(|(t, p)| format!("{t}:{p:.2}")).collect();
        s.push_str(&format!(" l{l}={}", w.join(",")));
    }
    s
}

/// A reading back from its line.
pub fn parse_line(s: &str) -> Option<Reading> {
    let mut pos = None;
    let mut ms = 0.0;
    let mut t_us = 0i64;
    let mut token = String::new();
    let mut layers = Vec::new();
    for field in s.split(' ') {
        let (k, v) = field.split_once('=')?;
        match k {
            "pos" => pos = v.parse().ok(),
            "t" => t_us = v.parse().unwrap_or(0),
            "ms" => ms = v.parse().unwrap_or(0.0),
            "tok" => token = crate::client::unescape(&v.replace("\\s", " ")),
            _ if k.starts_with('l') => {
                let l: i32 = k[1..].parse().ok()?;
                let words = if v.is_empty() {
                    Vec::new()
                } else {
                    v.split(',')
                        .filter_map(|wp| {
                            let (w, p) = wp.rsplit_once(':')?;
                            Some((w.to_string(), p.parse().ok()?))
                        })
                        .collect()
                };
                layers.push((l, words));
            }
            _ => {}
        }
    }
    Some(Reading {
        pos: pos?,
        token,
        layers,
        model_top: Vec::new(),
        ms,
        t_us,
    })
}

fn piece(llm: &Llm, t: i32) -> String {
    llm.text(&[t])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_display_rule_is_the_reference_s() {
        assert!(is_wordlike(" Brazil"));
        assert!(is_wordlike("don't"));
        assert!(is_wordlike("well-known"));
        assert!(is_wordlike("42"));
        assert!(!is_wordlike(" "));
        assert!(!is_wordlike("-x"));
        assert!(!is_wordlike("x-"));
        assert!(!is_wordlike("<|im_end|>"));
        assert!(!is_wordlike("<think>"));
        assert!(!is_wordlike("a,b"));
    }

    #[test]
    fn lines_round_trip() {
        let r = Reading {
            pos: 12,
            token: " the end\n".into(),
            layers: vec![
                (20, vec![("grief".into(), -1.25), ("loss".into(), -2.5)]),
                (26, vec![]),
            ],
            model_top: Vec::new(),
            ms: 0.75,
            t_us: 1_790_000_000_123_456,
        };
        let back = parse_line(&line(&r)).unwrap();
        assert_eq!(back.pos, 12);
        assert_eq!(back.token, r.token);
        assert_eq!(back.layers[0].1[1].0, "loss");
        assert!((back.layers[0].1[1].1 + 2.5).abs() < 1e-6);
        assert!(back.layers[1].1.is_empty());
        assert_eq!(back.t_us, 1_790_000_000_123_456);
    }
}
