//! The reflection loop's own logic, apart from the engine that runs it:
//! when a token deserves a second look (the triggers, read from what is on
//! the mind at that token and how sure the model is of it), the controls
//! that keep the loop from running away or locking up (smoothing,
//! hysteresis, habituation, a refractory period and a budget, all in real
//! time), how the question is put, how the answer is read, and the record
//! of each episode. The engine (`engine.rs`) copies the live sequence
//! before the token in question, deliberates in a lane beside the live
//! token, and keeps the token or rewinds onto the copy and places the word
//! the answer chose. See reflect.md.

use std::collections::{HashMap, VecDeque};

use crate::client::{escape, unescape};
use crate::clock;
use crate::engine::Frame;
use crate::mind::Reading;

/// The loop's settings.
#[derive(Clone, Debug)]
pub struct ReflectConfig {
    /// Doubt: the model's own probability of the chosen token below this.
    pub doubt_p: f32,
    /// The smoothed flag score fires above `flag_hi` and re-arms below `flag_lo`.
    pub flag_hi: f32,
    pub flag_lo: f32,
    /// Readings the flag score is smoothed over (an exponential average).
    pub flag_tokens: f32,
    /// A word lit at more than this many readings in a row stops counting
    /// (and is not shown in a question) until it goes out: habituation.
    pub habituate_after: u32,
    /// After a check, no other for this long; after a change, this long
    /// (microseconds of the monotonic clock).
    pub refractory_us: i64,
    pub refractory_change_us: i64,
    /// At most this many checks and changes in any minute.
    pub checks_per_min: usize,
    pub changes_per_min: usize,
    /// The most tokens the word a write names may run to.
    pub answer_tokens: usize,
    /// Words on the mind shown in a question.
    pub words: usize,
    /// A check keeps when keep's share of the choice is at least this (above
    /// 1: every check writes, which tests the rewind).
    pub keep_at: f32,
    /// A choice in which keep and write together hold less than this of the
    /// deliberation's distribution is no answer: unparsed, and the token
    /// stays (the model was writing something else, e.g. the label again).
    pub min_fmt: f32,
    /// Read-only: deliberate fully, never change a token (the measurement's
    /// third arm: the lanes' cost and numerics without the loop's effect).
    pub dry: bool,
}

impl Default for ReflectConfig {
    fn default() -> Self {
        Self {
            doubt_p: 0.30,
            flag_hi: 0.20,
            flag_lo: 0.08,
            flag_tokens: 6.0,
            habituate_after: 24,
            refractory_us: 2_000_000,
            refractory_change_us: 10_000_000,
            checks_per_min: 12,
            changes_per_min: 4,
            answer_tokens: 8,
            keep_at: 0.5,
            min_fmt: 0.2,
            words: 8,
            dry: false,
        }
    }
}

/// Words that, lit in the workspace, say something may be off.
pub const FLAG_WORDS: &[&str] = &[
    "error",
    "errors",
    "bug",
    "bugs",
    "wrong",
    "mistake",
    "mistakes",
    "incorrect",
    "typo",
    "oops",
    "wait",
    "actually",
    "fix",
    "broken",
    "invalid",
    "undefined",
    "null",
    "missing",
    "overflow",
    "fail",
    "failed",
    "failure",
    "careful",
    "wrongly",
];

/// Why a check fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// The model unsure of a word it chose, and its workspace holding other words.
    Doubt,
    /// Words of error or doubt lit in the workspace, smoothed.
    Flag,
}

impl Why {
    pub fn name(self) -> &'static str {
        match self {
            Why::Doubt => "doubt",
            Why::Flag => "flag",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "doubt" => Some(Why::Doubt),
            "flag" => Some(Why::Flag),
            _ => None,
        }
    }
}

/// What the loop saw at one token.
#[derive(Clone, Debug)]
pub struct Signals {
    /// The model's own probability of the chosen token.
    pub p_chosen: f32,
    /// The chosen token is a word (doubt may fire), or at least not blank
    /// and not a control token (a flag may fire).
    pub chosen_word: bool,
    pub chosen_placeable: bool,
    /// Whether the chosen word is among the workspace's top five words at
    /// a block.
    pub chosen_in_band: bool,
    /// The flag score, smoothed over the readings.
    pub flag_smoothed: f32,
}

/// The loop's running state: the signals and the controls.
pub struct Reflector {
    pub cfg: ReflectConfig,
    flag_ema: f32,
    armed: bool,
    /// Per word on the mind, how many readings in a row it has been lit.
    runs: HashMap<String, u32>,
    last_check_mono: i64,
    last_change_mono: i64,
    checks: VecDeque<i64>,
    changes: VecDeque<i64>,
    pub n_checks: u64,
    pub n_changes: u64,
    pub n_unparsed: u64,
}

impl Reflector {
    pub fn new(cfg: ReflectConfig) -> Self {
        Self {
            cfg,
            flag_ema: 0.0,
            armed: true,
            runs: HashMap::new(),
            last_check_mono: i64::MIN / 2,
            last_change_mono: i64::MIN / 2,
            checks: VecDeque::new(),
            changes: VecDeque::new(),
            n_checks: 0,
            n_changes: 0,
            n_unparsed: 0,
        }
    }

    fn habituated(&self, word: &str) -> bool {
        self.runs.get(word).copied().unwrap_or(0) > self.cfg.habituate_after
    }

    /// The runs, one reading on: the words lit now lengthen theirs, the
    /// others are forgotten.
    fn update_runs(&mut self, r: &Reading) {
        let mut lit: Vec<String> = r
            .layers
            .iter()
            .flat_map(|(_, ws)| ws.iter().map(|(w, _)| w.trim().to_lowercase()))
            .collect();
        lit.sort();
        lit.dedup();
        self.runs.retain(|k, _| lit.binary_search(k).is_ok());
        for k in lit {
            *self.runs.entry(k).or_insert(0) += 1;
        }
    }

    /// The flag score of a reading: the probability mass of flag words at
    /// the band's blocks, averaged over the blocks; habituated words do not
    /// count.
    fn flag_score(&self, r: &Reading) -> f32 {
        let mut mass = 0f32;
        for (_, words) in &r.layers {
            for (w, logp) in words {
                let k = w.trim().to_lowercase();
                if FLAG_WORDS.contains(&k.as_str()) && !self.habituated(&k) {
                    mass += logp.exp();
                }
            }
        }
        mass / r.layers.len().max(1) as f32
    }

    /// The signals at one token, read every token: `p_chosen` the model's
    /// own probability of the token it chose, `chosen` its text, `control`
    /// whether it is a control token.
    pub fn signals(&mut self, r: &Reading, p_chosen: f32, chosen: &str, control: bool) -> Signals {
        self.update_runs(r);
        let flag = self.flag_score(r);
        let a = 1.0 / self.cfg.flag_tokens.max(1.0);
        self.flag_ema = (1.0 - a) * self.flag_ema + a * flag;
        let c = chosen.trim().to_lowercase();
        let chosen_in_band = !c.is_empty()
            && r.layers
                .iter()
                .any(|(_, ws)| ws.iter().take(5).any(|(w, _)| w.trim().to_lowercase() == c));
        Signals {
            p_chosen,
            chosen_word: !control && crate::mind::is_wordlike(chosen),
            chosen_placeable: !control && !c.is_empty(),
            chosen_in_band,
            flag_smoothed: self.flag_ema,
        }
    }

    /// Whether the budget of checks or changes for the last minute is spent
    /// (the loop is read-only until it is not).
    pub fn spent(&self, mono: i64) -> bool {
        let minute = 60_000_000;
        let n = |q: &VecDeque<i64>| q.iter().filter(|&&t| mono - t <= minute).count();
        n(&self.checks) >= self.cfg.checks_per_min || n(&self.changes) >= self.cfg.changes_per_min
    }

    /// Whether the last minute's checks and changes are back to half the
    /// budget or less: the spent note re-arms only then (hysteresis, so a
    /// budget that frees one slot and spends it again is noted once).
    pub fn recovered(&self, mono: i64) -> bool {
        let minute = 60_000_000;
        let n = |q: &VecDeque<i64>| q.iter().filter(|&&t| mono - t <= minute).count();
        n(&self.checks) <= self.cfg.checks_per_min / 2
            && n(&self.changes) <= self.cfg.changes_per_min / 2
    }

    /// Whether to check this token now (`mono`: the monotonic clock, us).
    /// The hysteresis re-arms the flag trigger only once the smoothed score
    /// has fallen below `flag_lo`.
    pub fn should_check(&mut self, s: &Signals, mono: i64) -> Option<Why> {
        if self.flag_ema < self.cfg.flag_lo {
            self.armed = true;
        }
        let minute = 60_000_000;
        while self.checks.front().is_some_and(|&t| mono - t > minute) {
            self.checks.pop_front();
        }
        while self.changes.front().is_some_and(|&t| mono - t > minute) {
            self.changes.pop_front();
        }
        if mono - self.last_check_mono < self.cfg.refractory_us
            || mono - self.last_change_mono < self.cfg.refractory_change_us
            || self.checks.len() >= self.cfg.checks_per_min
            || self.changes.len() >= self.cfg.changes_per_min
        {
            return None;
        }
        let why = if s.chosen_placeable && self.armed && self.flag_ema > self.cfg.flag_hi {
            self.armed = false;
            Some(Why::Flag)
        } else if s.chosen_word && s.p_chosen < self.cfg.doubt_p && !s.chosen_in_band {
            Some(Why::Doubt)
        } else {
            None
        };
        if why.is_some() {
            self.last_check_mono = mono;
            self.checks.push_back(mono);
            self.n_checks += 1;
        }
        why
    }

    /// A check changed the token.
    pub fn changed(&mut self, mono: i64) {
        self.last_change_mono = mono;
        self.changes.push_back(mono);
        self.n_changes += 1;
    }

    /// The words on the mind at a reading, for a question: across the
    /// band's blocks, best first, each once, habituated ones left out.
    pub fn band_words(&self, r: &Reading) -> Vec<String> {
        let mut all: Vec<(String, f32)> = r
            .layers
            .iter()
            .flat_map(|(_, ws)| ws.iter().map(|(w, p)| (w.trim().to_string(), *p)))
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out: Vec<String> = Vec::new();
        for (w, _) in all {
            let k = w.to_lowercase();
            if w.is_empty() || self.habituated(&k) || out.iter().any(|o| o.to_lowercase() == k) {
                continue;
            }
            out.push(w);
            if out.len() == self.cfg.words {
                break;
            }
        }
        out
    }
}

/// The question put to the deliberation, in the frame's own style, placed
/// exactly where the token in question would go: the time, the token, the
/// words on the mind there.
pub fn question(frame: Frame, t_us: Option<i64>, chosen: &str, words: &[&str]) -> String {
    let c = chosen.trim().replace('"', "'");
    let on = if words.is_empty() {
        "nothing in particular".to_string()
    } else {
        words.join(", ")
    };
    // The time to the microsecond in the stream; none in a task (a
    // measurement gives the same answer on every run).
    let at = t_us
        .map(|t| format!("at {}, ", clock::hms(t)))
        .unwrap_or_default();
    let body = format!(
        "{at}a check on the next word: you were about to write \"{c}\" here; on your mind here: {on}. Is \"{c}\" right at this place? Keep it, or write the word to use instead."
    );
    match frame {
        Frame::Journal => format!("\n« [{body}]\n{DECISION}"),
        Frame::Chat => format!("\n[{body}]\n{DECISION}"),
    }
}

/// The line the question ends on: the deliberation's first token after it
/// is read as a choice between ` keep` and ` write` (their probabilities,
/// not a sample), and only a write goes on to name the word.
pub const DECISION: &str = "Decision:";

/// The choice's two classes, as the deliberation may continue `Decision:`:
/// each the forms of its word that are one token in the vocabulary (a
/// form split into pieces would lend its first piece's mass to other words).
pub const KEEP_FORMS: &[&str] = &[" keep", " Keep", "keep", "Keep", " KEEP"];
pub const WRITE_FORMS: &[&str] = &[" write", " Write", "write", "Write", " WRITE"];

/// What a write is fed before its word.
pub const WRITE_PREFIX: &str = " write:";

/// What an answer said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Keep,
    /// The text to write instead (as the model gave it).
    Write(String),
    /// Neither: the answer said something else (kept, and counted apart).
    Unparsed,
}

/// The first word of `x`, out of its quotes; a `]` at its end only when
/// unmatched inside it (the question's own bracket closing), and sentence
/// punctuation after a word.
fn first_word(x: &str) -> String {
    let mut w = x
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(['"', '\'', '`'])
        .to_string();
    loop {
        // A `]` closing nothing inside the word, or a sentence's mark
        // after a word.
        let unmatched = w.ends_with(']') && w.matches(']').count() > w.matches('[').count();
        let mark = w.len() > 1
            && w.ends_with([',', '.', ';', ':'])
            && w[..w.len() - 1]
                .chars()
                .last()
                .is_some_and(|c| c.is_alphanumeric() || matches!(c, ']' | ')' | '"' | '\'' | '`'));
        if !(unmatched || mark) {
            break;
        }
        w.pop();
        w = w.trim_matches(['"', '\'', '`']).to_string();
    }
    w
}

/// Read an answer: its first line with anything on it, which says `keep`
/// or `write: X`.
pub fn parse_answer(answer: &str) -> Decision {
    for line in answer.lines() {
        let l = line
            .trim()
            .trim_start_matches(['»', '«', '[', ' ', '-', '*'])
            .trim();
        if l.is_empty() {
            continue;
        }
        // ASCII lowering keeps byte offsets, so `i` indexes `l` too.
        let low = l.to_ascii_lowercase();
        if let Some(i) = low.find("write:") {
            let word = first_word(&l[i + "write:".len()..]);
            return if word.is_empty() {
                Decision::Unparsed
            } else {
                Decision::Write(word)
            };
        }
        if low.starts_with("keep") {
            return Decision::Keep;
        }
        return Decision::Unparsed;
    }
    Decision::Unparsed
}

/// Whether a written word is the protocol's own (`Decision`, `keep`,
/// `write`): the model echoing the question, not naming a word.
pub fn is_protocol_word(word: &str) -> bool {
    let w = word
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    matches!(w.as_str(), "decision" | "keep" | "write")
}

/// The text to place instead, with the chosen token's leading space when
/// it had one (a word inside a sentence keeps its spacing).
pub fn replacement_text(chosen: &str, word: &str) -> String {
    if chosen.starts_with(' ') && !word.starts_with(' ') {
        format!(" {word}")
    } else {
        word.to_string()
    }
}

/// How an episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The answer kept the token.
    Kept,
    /// The answer wrote another token: the live sequence rewound onto the
    /// copy and placed it.
    Changed,
    /// The answer wrote the token it was about to write.
    Same,
    /// The answer said neither; the token stayed.
    Unparsed,
    /// Read-only: the answer would have changed the token.
    Dry,
    /// Something had to go into the live sequence first (a rollover, the
    /// end of a task's answer while it ran past the cap): the check was
    /// dropped and the token stayed.
    Abandoned,
}

impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Kept => "kept",
            Outcome::Changed => "changed",
            Outcome::Same => "same",
            Outcome::Unparsed => "unparsed",
            Outcome::Dry => "dry",
            Outcome::Abandoned => "abandoned",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        [
            Outcome::Kept,
            Outcome::Changed,
            Outcome::Same,
            Outcome::Unparsed,
            Outcome::Dry,
            Outcome::Abandoned,
        ]
        .into_iter()
        .find(|o| o.name() == s)
    }
}

/// One check, start to end: the record (`reflect.log`, the socket's
/// `reflect` line, the terminal's marks).
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    /// When the token in question was chosen, microseconds of real time.
    pub t_us: i64,
    /// Its position.
    pub pos: i32,
    pub why: Why,
    pub chosen: String,
    pub p_chosen: f32,
    pub flag: f32,
    pub words: Vec<String>,
    /// The choice: keep's share of keep and write, and how much of the
    /// deliberation's whole next-token distribution the two hold (low: the
    /// model was not answering the question).
    pub keep: f32,
    pub fmt: f32,
    /// The three likeliest next tokens of the deliberation at `Decision:`,
    /// with their probabilities (where the mass `fmt` misses went).
    pub top: Vec<(String, f32)>,
    /// The deliberation's answer as it wrote it.
    pub answer: String,
    pub outcome: Outcome,
    /// What was placed instead (changed), or would have been (dry).
    pub to: String,
    /// How long the check took (monotonic) and the live tokens placed meanwhile.
    pub ms: f32,
    pub placed: usize,
    /// Positions taken back by a rewind, `[from, to)`: readings of them
    /// already shown belong to the text that was taken back.
    pub back: Option<(i32, i32)>,
}

fn field(s: &str) -> String {
    escape(s).replace(' ', "\\s")
}

fn unfield(s: &str) -> String {
    unescape(&s.replace("\\s", " "))
}

/// An episode as one line: `t=US pos=P why=W outcome=O p=P flag=F keep=K
/// fmt=F ms=M placed=N [back=A-B] chosen=T to=T words=a,b answer=T`.
pub fn line(e: &Episode) -> String {
    let mut s = format!(
        "t={} pos={} why={} outcome={} p={:.4} flag={:.4} keep={:.4} fmt={:.4} ms={:.1} placed={}",
        e.t_us,
        e.pos,
        e.why.name(),
        e.outcome.name(),
        e.p_chosen,
        e.flag,
        e.keep,
        e.fmt,
        e.ms,
        e.placed
    );
    if let Some((a, b)) = e.back {
        s.push_str(&format!(" back={a}-{b}"));
    }
    let words: Vec<String> = e.words.iter().map(|w| field(&w.replace(',', ""))).collect();
    for (i, (t, p)) in e.top.iter().enumerate() {
        s.push_str(&format!(" top{n}={} top{n}p={p:.4}", field(t), n = i + 1));
    }
    s.push_str(&format!(
        " chosen={} to={} words={} answer={}",
        field(&e.chosen),
        field(&e.to),
        words.join(","),
        field(&e.answer)
    ));
    s
}

/// An episode back from its line.
pub fn parse_line(s: &str) -> Option<Episode> {
    let mut e = Episode {
        t_us: 0,
        pos: -1,
        why: Why::Doubt,
        chosen: String::new(),
        p_chosen: 0.0,
        flag: 0.0,
        words: Vec::new(),
        keep: 0.0,
        fmt: 0.0,
        top: Vec::new(),
        answer: String::new(),
        outcome: Outcome::Kept,
        to: String::new(),
        ms: 0.0,
        placed: 0,
        back: None,
    };
    let mut seen_why = false;
    for f in s.split(' ') {
        let (k, v) = f.split_once('=')?;
        match k {
            "t" => e.t_us = v.parse().ok()?,
            "pos" => e.pos = v.parse().ok()?,
            "why" => {
                e.why = Why::parse(v)?;
                seen_why = true;
            }
            "outcome" => e.outcome = Outcome::parse(v)?,
            "p" => e.p_chosen = v.parse().ok()?,
            "flag" => e.flag = v.parse().ok()?,
            "keep" => e.keep = v.parse().ok()?,
            "fmt" => e.fmt = v.parse().ok()?,
            "ms" => e.ms = v.parse().ok()?,
            "placed" => e.placed = v.parse().ok()?,
            "back" => {
                let (a, b) = v.split_once('-')?;
                e.back = Some((a.parse().ok()?, b.parse().ok()?));
            }
            "chosen" => e.chosen = unfield(v),
            "to" => e.to = unfield(v),
            "words" => {
                e.words = v
                    .split(',')
                    .filter(|w| !w.is_empty())
                    .map(unfield)
                    .collect()
            }
            "answer" => e.answer = unfield(v),
            k if k.starts_with("top") => {
                let (n, prob) = match k[3..].strip_suffix('p') {
                    Some(n) => (n, true),
                    None => (&k[3..], false),
                };
                let i: usize = n.parse().ok()?;
                if i == 0 || i > 16 {
                    return None;
                }
                while e.top.len() < i {
                    e.top.push((String::new(), 0.0));
                }
                if prob {
                    e.top[i - 1].1 = v.parse().ok()?;
                } else {
                    e.top[i - 1].0 = unfield(v);
                }
            }
            _ => {}
        }
    }
    (seen_why && e.pos >= 0).then_some(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(words: &[(&str, f32)]) -> Reading {
        Reading {
            pos: 0,
            token: String::new(),
            layers: vec![(29, words.iter().map(|(w, p)| (w.to_string(), *p)).collect())],
            model_top: Vec::new(),
            ms: 0.0,
            t_us: 0,
        }
    }

    #[test]
    fn answers_are_read() {
        assert_eq!(parse_answer("keep"), Decision::Keep);
        assert_eq!(parse_answer("» Keep, it fits."), Decision::Keep);
        assert_eq!(
            parse_answer("write: vector\nbecause"),
            Decision::Write("vector".into())
        );
        assert_eq!(
            parse_answer("\n» write: \"usize\" instead"),
            Decision::Write("usize".into())
        );
        assert_eq!(parse_answer("I think so"), Decision::Unparsed);
        assert_eq!(parse_answer("write:"), Decision::Unparsed);
        assert_eq!(parse_answer(""), Decision::Unparsed);
        assert_eq!(replacement_text(" foo", "bar"), " bar");
        assert_eq!(replacement_text("foo", "bar"), "bar");
    }

    #[test]
    fn brackets_and_punctuation_in_written_code_survive() {
        assert_eq!(
            parse_answer("write: v[i], since"),
            Decision::Write("v[i]".into())
        );
        assert_eq!(parse_answer("write: v[i]]"), Decision::Write("v[i]".into()));
        assert_eq!(
            parse_answer("write: usize]"),
            Decision::Write("usize".into())
        );
        assert_eq!(
            parse_answer("write: `<=`. The bound"),
            Decision::Write("<=".into())
        );
        assert_eq!(
            parse_answer("write: len()."),
            Decision::Write("len()".into())
        );
        assert_eq!(parse_answer("write: ;"), Decision::Write(";".into()));
        assert_eq!(parse_answer("write: x.y"), Decision::Write("x.y".into()));
    }

    #[test]
    fn the_protocols_own_words_are_no_answer() {
        assert!(is_protocol_word(" Decision"));
        assert!(is_protocol_word("keep."));
        assert!(is_protocol_word("`write`"));
        assert!(!is_protocol_word(" usize"));
        assert!(!is_protocol_word("decisions"));
    }

    #[test]
    fn the_question_carries_the_time_the_token_and_the_words() {
        let q = question(
            Frame::Chat,
            Some(clock::now_us()),
            " sort",
            &["order", "stable"],
        );
        assert!(
            q.starts_with("\n[at ") && q.ends_with("]\nDecision:"),
            "{q}"
        );
        assert!(q.contains("\"sort\"") && q.contains("order, stable"), "{q}");
        assert!(question(Frame::Journal, Some(0), "x", &[]).starts_with("\n« [at "));
        // A task's question carries no time.
        let t = question(Frame::Chat, None, " sort", &[]);
        assert!(t.starts_with("\n[a check on the next word"), "{t}");
    }

    #[test]
    fn doubt_fires_on_words_and_the_refractory_period_holds() {
        let mut r = Reflector::new(ReflectConfig::default());
        let rd = reading(&[("cat", -1.0)]);
        let s = r.signals(&rd, 0.1, " dog", false);
        assert!(!s.chosen_in_band);
        assert_eq!(r.should_check(&s, 10_000_000), Some(Why::Doubt));
        // Within the refractory period: nothing.
        assert_eq!(r.should_check(&s, 11_000_000), None);
        // After it: again.
        assert_eq!(r.should_check(&s, 12_500_000), Some(Why::Doubt));
        // Sure of its choice, the choice in the band, punctuation or a
        // control token: no doubt.
        let s2 = r.signals(&rd, 0.9, " dog", false);
        assert_eq!(r.should_check(&s2, 20_000_000), None);
        let s3 = r.signals(&rd, 0.1, " cat", false);
        assert!(s3.chosen_in_band);
        let s4 = r.signals(&rd, 0.1, ");", false);
        assert_eq!(r.should_check(&s4, 30_000_000), None);
        let s5 = r.signals(&rd, 0.1, "dog", true);
        assert_eq!(r.should_check(&s5, 40_000_000), None);
    }

    #[test]
    fn the_flag_needs_its_smoothing_and_rearms_by_hysteresis() {
        let cfg = ReflectConfig {
            refractory_us: 0,
            refractory_change_us: 0,
            ..ReflectConfig::default()
        };
        let mut r = Reflector::new(cfg);
        // p 0.37 and 0.14: a mass of 0.50, a sixth of which after one
        // reading is below flag_hi; it takes three.
        let lit = reading(&[("wrong", -1.0), ("error", -2.0)]);
        let quiet = reading(&[("cat", -0.2)]);
        let s = r.signals(&lit, 0.9, " x", false);
        assert_eq!(r.should_check(&s, 1), None);
        let mut fired = 0;
        for t in 2..10 {
            let s = r.signals(&lit, 0.9, " x", false);
            if r.should_check(&s, t) == Some(Why::Flag) {
                fired += 1;
            }
        }
        assert_eq!(fired, 1, "fires once, then waits to re-arm");
        for t in 10..40 {
            let s = r.signals(&quiet, 0.9, " x", false);
            r.should_check(&s, t);
        }
        let mut again = false;
        for t in 40..50 {
            let s = r.signals(&lit, 0.9, " x", false);
            again |= r.should_check(&s, t) == Some(Why::Flag);
        }
        assert!(again, "re-armed once quiet");
        // A blank token is never checked, flagged or not.
        let s = r.signals(&lit, 0.9, "\n", false);
        assert!(!s.chosen_placeable);
    }

    #[test]
    fn words_lit_too_long_stop_counting_and_leave_the_question() {
        let cfg = ReflectConfig {
            habituate_after: 3,
            ..ReflectConfig::default()
        };
        let mut r = Reflector::new(cfg);
        let lit = reading(&[("wrong", 0.0), ("cat", -1.0)]);
        let mut scores = Vec::new();
        for _ in 0..6 {
            r.update_runs(&lit);
            scores.push(r.flag_score(&lit));
        }
        assert!(scores[0] > 0.9 && scores[2] > 0.9);
        assert_eq!(scores[4], 0.0);
        assert!(r.band_words(&lit).is_empty());
        // Gone for one reading, it counts again.
        r.update_runs(&reading(&[("dog", -1.0)]));
        r.update_runs(&lit);
        assert!(r.flag_score(&lit) > 0.9);
        assert_eq!(
            r.band_words(&lit),
            vec!["wrong".to_string(), "cat".to_string()]
        );
    }

    #[test]
    fn the_budget_caps_checks_per_minute() {
        let cfg = ReflectConfig {
            refractory_us: 0,
            checks_per_min: 2,
            ..ReflectConfig::default()
        };
        let mut r = Reflector::new(cfg);
        let rd = reading(&[("cat", -1.0)]);
        let mut n = 0;
        for t in 0..10 {
            let s = r.signals(&rd, 0.1, " dog", false);
            if r.should_check(&s, t * 1000).is_some() {
                n += 1;
            }
        }
        assert_eq!(n, 2);
        assert!(r.spent(10_000));
        assert!(!r.spent(70_000_000));
        // Spent at 2 of 2; recovered only at half (1) or less.
        assert!(!r.recovered(10_000));
        assert!(r.recovered(70_000_000));
    }

    #[test]
    fn episodes_round_trip_through_their_line() {
        let e = Episode {
            t_us: 1_790_000_000_123_456,
            pos: 812,
            why: Why::Doubt,
            chosen: " i32".into(),
            p_chosen: 0.2134,
            flag: 0.0312,
            words: vec!["usize".into(), "index".into()],
            keep: 0.1875,
            fmt: 0.9312,
            top: vec![
                (" write".into(), 0.7),
                ("\n".into(), 0.05),
                (" a b".into(), 0.01),
            ],
            answer: " write: usize, since it indexes\n".into(),
            outcome: Outcome::Changed,
            to: " usize".into(),
            ms: 412.5,
            placed: 14,
            back: Some((812, 826)),
        };
        assert_eq!(parse_line(&line(&e)), Some(e.clone()));
        let k = Episode {
            outcome: Outcome::Kept,
            back: None,
            to: String::new(),
            words: Vec::new(),
            ..e
        };
        assert_eq!(parse_line(&line(&k)), Some(k));
        assert_eq!(parse_line("t=1 why=nope pos=2"), None);
    }
}
