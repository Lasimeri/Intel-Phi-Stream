//! What is on the mind of the stream, token by token: for every token the
//! live sequence places, the residual after a few chosen blocks at that
//! token's own position (`capture.rs`), read through the Jacobian lens
//! (`lens.rs`, `readout.rs`) and ranked on the GPU. Nothing is averaged
//! over tokens: a reading belongs to one position. It runs synchronously in
//! the engine's cycle, right after the decode that placed the token. See
//! mind.md.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context as _, Result};

use crate::lens::Lens;
use crate::llm::Llm;
use crate::readout::{cuda_device, Group, Readout};
use crate::rotlog::RotLog;

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
    log: RotLog,
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
        let log = RotLog::open(workspace.join("mind.log"));
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
        self.log.line(&line(&reading));
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

/// Whether the lens's word `w` is one the line `said` holds: in it, or a
/// form of one of its words (a common start of four letters or more, three
/// quarters of the shorter word: under "Rebuild and test" the lens's
/// strongest were testing, tests and rebuilt). Both lowercase.
fn in_line(w: &str, said: &str) -> bool {
    if said.contains(w) {
        return true;
    }
    let w: Vec<char> = w.chars().collect();
    said.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.chars().count() >= 4)
        .any(|s| {
            let s: Vec<char> = s.chars().collect();
            let common = w.iter().zip(&s).take_while(|(a, b)| a == b).count();
            common >= 4 && common * 4 >= w.len().min(s.len()) * 3
        })
}

/// Words with weights, strongest first.
type Words = Vec<(String, f32)>;

/// A line's lens words (each word's probability summed over its tokens and
/// the blocks read, `sums`, over `readings`: tokens times blocks), lowercase,
/// three letters or more, strongest first, split by whether the line holds
/// them.
fn split_words(sums: &[(String, f32)], readings: usize, said: &str) -> (Words, Words) {
    let said = said.to_lowercase();
    let mut merged: Vec<(String, f32)> = Vec::new();
    for (w, p) in sums {
        let w = w.trim().to_lowercase();
        if w.chars().count() < 3 {
            continue;
        }
        match merged.iter_mut().find(|(x, _)| *x == w) {
            Some(e) => e.1 += p,
            None => merged.push((w, *p)),
        }
    }
    for e in &mut merged {
        e.1 /= readings.max(1) as f32;
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    merged.into_iter().partition(|(w, _)| !in_line(w, &said))
}

/// What was on its mind over a line that the line does not say, the one
/// rule for the terminal's row under the line (`tui.md`) and the guide
/// lane's lens aside (`engine.md`): none unless the strongest weighs `min`
/// or more; then it and the others of at least half its weight, four at
/// most.
pub fn unsaid(sums: &[(String, f32)], readings: usize, said: &str, min: f32) -> Vec<(String, f32)> {
    let (out, _) = split_words(sums, readings, said);
    let Some(top) = out.first().map(|w| w.1).filter(|t| *t >= min) else {
        return Vec::new();
    };
    out.into_iter()
        .take(4)
        .filter(|w| w.1 >= top / 2.0)
        .collect()
}

/// The placebo of `unsaid`: the `n` strongest lens words the line does say
/// (an aside of the same frame and length whose content is in the context
/// already).
pub fn said(sums: &[(String, f32)], readings: usize, said: &str, n: usize) -> Vec<(String, f32)> {
    let (_, inside) = split_words(sums, readings, said);
    inside.into_iter().take(n).collect()
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

    #[test]
    fn unsaid_leaves_out_the_line_and_its_forms() {
        // Under "3. Rebuild and test" on the live service the strongest
        // were testing, tests and rebuilt: forms of what the line says.
        let sums: Vec<(String, f32)> = [
            ("testing", 1.5),
            ("tests", 0.6),
            ("Rebuilt", 0.9),
            ("verify", 0.9),
            ("using", 0.15),
            ("to", 2.7),
        ]
        .iter()
        .map(|(w, p)| (w.to_string(), *p))
        .collect();
        let line = "3. Rebuild and test";
        let names = |v: Vec<(String, f32)>| v.into_iter().map(|w| w.0).collect::<Vec<_>>();
        // Over 3 readings: verify 0.30, using 0.05 (under half of it).
        assert_eq!(names(unsaid(&sums, 3, line, 0.10)), vec!["verify"]);
        assert!(unsaid(&sums, 3, line, 0.31).is_empty());
        // The placebo: the strongest the line does say.
        assert_eq!(names(said(&sums, 3, line, 2)), vec!["testing", "rebuilt"]);
    }
}
