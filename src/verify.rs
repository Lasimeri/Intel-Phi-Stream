//! What a note of the stream's says about the code, checked against the
//! code (development, `docs/dev.md`). In the first dev sessions the stream
//! noted functions, modules and constants that do not exist, and its notes
//! come back at every rollover and start, so one false note re-seeds the
//! same confabulation again and again. A note is checked when written (and
//! when loaded): every code name it uses is searched for in the
//! repository's text files, every `file:line` is resolved and the line
//! quoted. Nothing is guessed: a name is missing only if no file holds it
//! as a whole word. See verify.md.

use std::fs;
use std::path::Path;

/// The repository's text files: their paths (relative) and contents.
pub struct Repo {
    files: Vec<(String, String)>,
}

/// What a note's code references came to.
#[derive(Debug, Default, PartialEq)]
pub struct Findings {
    /// Code names no file in the repository holds.
    pub missing: Vec<String>,
    /// `file:line` references: the file, the line, and the line's text.
    pub lines: Vec<(String, usize, String)>,
    /// Files named that do not exist, or lines past a file's end.
    pub bad_refs: Vec<String>,
}

impl Findings {
    pub fn clean(&self) -> bool {
        self.missing.is_empty() && self.bad_refs.is_empty()
    }
}

const DIRS: &[&str] = &["src", "scripts", "docs", "tools"];
const EXTS: &[&str] = &["rs", "md", "sh", "c", "h", "toml", "S"];

impl Repo {
    /// The text files under the repository's code and doc directories and
    /// its top level (never `target/` or `.git/`).
    pub fn load(root: &Path) -> Self {
        let mut files = Vec::new();
        let mut stack: Vec<std::path::PathBuf> = DIRS.iter().map(|d| root.join(d)).collect();
        if let Ok(rd) = fs::read_dir(root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_file() {
                    stack.push(p);
                }
            }
        }
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                if let Ok(rd) = fs::read_dir(&p) {
                    stack.extend(rd.flatten().map(|e| e.path()));
                }
            } else if p
                .extension()
                .is_some_and(|x| EXTS.contains(&x.to_string_lossy().as_ref()))
            {
                if let Ok(text) = fs::read_to_string(&p) {
                    let rel = p
                        .strip_prefix(root)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .into_owned();
                    files.push((rel, text));
                }
            }
        }
        Self { files }
    }

    fn holds(&self, name: &str) -> bool {
        self.files.iter().any(|(_, t)| has_word(t, name))
    }

    /// A file by a path as the note wrote it: relative to the root, or a bare
    /// name under one of the directories.
    fn file(&self, path: &str) -> Option<&(String, String)> {
        let path = path.trim_start_matches("./");
        self.files.iter().find(|(p, _)| p == path).or_else(|| {
            self.files
                .iter()
                .find(|(p, _)| p.rsplit('/').next() == Some(path))
        })
    }

    /// The note's code names and file references, checked.
    pub fn check(&self, note: &str) -> Findings {
        let mut f = Findings::default();
        for name in code_names(note) {
            if !self.holds(&name) && !f.missing.contains(&name) {
                f.missing.push(name);
            }
        }
        for (path, line) in file_refs(note) {
            match (self.file(&path), line) {
                (None, _) => f.bad_refs.push(format!("{path} does not exist")),
                (Some(_), None) => {}
                (Some((p, text)), Some(n)) => match text.lines().nth(n.saturating_sub(1)) {
                    Some(l) if n > 0 => {
                        let l: String = l.trim().chars().take(120).collect();
                        f.lines.push((p.clone(), n, l));
                    }
                    _ => f
                        .bad_refs
                        .push(format!("{p} has {} lines, not {n}", text.lines().count())),
                },
            }
        }
        f
    }
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether `text` holds `word` as a whole identifier.
fn has_word(text: &str, word: &str) -> bool {
    let mut from = 0;
    while let Some(i) = text[from..].find(word) {
        let at = from + i;
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        if !before.is_some_and(is_ident_char) && !after.is_some_and(is_ident_char) {
            return true;
        }
        from = at + word.len();
    }
    false
}

/// The code names a note uses: what is inside backticks (split into its
/// identifiers), identifiers with an underscore or `::`, a name called
/// (`name(`), and a name after `mod`, `fn`, `struct`, `enum`, `impl`,
/// `trait` or `const`. Plain words are not code names. File names are
/// `file_refs`'.
pub fn code_names(note: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        let s = s.trim_matches(|c: char| !is_ident_char(c));
        if s.len() >= 3
            && s.chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && s.chars().all(is_ident_char)
            && !out.iter().any(|o| o == s)
        {
            out.push(s.to_string());
        }
    };
    // Inside backticks.
    for (i, part) in note.split('`').enumerate() {
        if i % 2 == 1 && !part.contains(char::is_whitespace)
            || i % 2 == 1 && part.starts_with("mod ")
        {
            for w in part.split(|c: char| !is_ident_char(c)) {
                if !w.is_empty() && !is_file_word(part) {
                    push(w);
                }
            }
        }
    }
    let words: Vec<&str> = note.split_whitespace().collect();
    for (i, w) in words.iter().enumerate() {
        if is_file_word(w) {
            continue;
        }
        // Every identifier in the word (`self.reflection_check`,
        // `mirror.check_line(text)`, `keep_at=0.45`): one with an
        // underscore, one called, every part of a `::` path, and the name
        // after a declaring keyword.
        let path = w.contains("::");
        let declared = i > 0
            && ["mod", "fn", "struct", "enum", "impl", "trait", "const"].contains(&words[i - 1]);
        for (k, part) in w.split(|c: char| !is_ident_char(c)).enumerate() {
            if part.is_empty() {
                continue;
            }
            let called = w.contains(&format!("{part}("));
            let snake = part.contains('_') && part.chars().any(|c| c.is_ascii_lowercase());
            if path || called || snake || (declared && k == 0) {
                push(part);
            }
        }
    }
    out
}

fn is_file_word(w: &str) -> bool {
    let w = w.trim_matches(|c: char| {
        !c.is_ascii_alphanumeric() && c != '.' && c != '/' && c != '_' && c != '-' && c != ':'
    });
    let base = w.split(':').next().unwrap_or(w);
    EXTS.iter().any(|x| base.ends_with(&format!(".{x}"))) && base.len() > 3
}

/// The files a note names (`src/engine.rs`, `engine.rs:449`), with the
/// line when it gives one.
pub fn file_refs(note: &str) -> Vec<(String, Option<usize>)> {
    let mut out = Vec::new();
    let words: Vec<&str> = note.split_whitespace().collect();
    for (i, w) in words.iter().enumerate() {
        let t = w.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && c != '.' && c != '/' && c != '_' && c != '-' && c != ':'
        });
        let t = t.trim_end_matches(['.', ':']);
        if !is_file_word(t) {
            continue;
        }
        let (path, line) = match t.split_once(':') {
            Some((p, n)) => (p, n.split('-').next().and_then(|n| n.parse().ok())),
            None => (t, None),
        };
        // "engine.rs line 449" and "engine.rs 449" give the line too.
        let line = line.or_else(|| {
            let w1 = words.get(i + 1)?;
            let digits = |s: &str| s.trim_matches(|c: char| !c.is_ascii_digit()).parse().ok();
            if w1.trim_start_matches('(') == "line" {
                return digits(words.get(i + 2)?);
            }
            if w1
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, ',' | ')' | ';'))
            {
                return digits(w1);
            }
            None
        });
        if !out
            .iter()
            .any(|(p, l): &(String, Option<usize>)| p == path && *l == line)
        {
            out.push((path.to_string(), line));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> Repo {
        Repo {
            files: vec![
                (
                    "src/reflect.rs".into(),
                    "pub struct ReflectConfig {\n    pub keep_at: f32,\n}\nfn parse_answer() {}\n"
                        .into(),
                ),
                (
                    "src/engine.rs".into(),
                    "fn check_cycle() {\n    let x = 1;\n}\n".into(),
                ),
            ],
        }
    }

    #[test]
    fn the_names_a_note_uses_are_found() {
        let n = code_names("engine.rs 462 calls self.reflection_check which calls mirror.check_line(text) against keep_at=0.45; Mirror is mod mirror inside reflect.rs");
        for w in ["reflection_check", "check_line", "keep_at", "mirror"] {
            assert!(n.contains(&w.to_string()), "{w} in {n:?}");
        }
        assert!(!n
            .iter()
            .any(|w| w == "calls" || w == "inside" || w == "reflect"));
        assert!(code_names("`ReflectConfig::keep_at` and `parse_answer`")
            .contains(&"ReflectConfig".to_string()));
    }

    #[test]
    fn a_false_note_is_caught_and_a_true_one_passes() {
        let r = repo();
        let f = r.check("check_cycle reads keep_at from ReflectConfig::keep_at");
        assert!(f.clean(), "{f:?}");
        let f = r.check(
            "mirror.check_line(text) is called from engine.rs line 2; mod mirror is in reflect.rs",
        );
        assert_eq!(
            f.missing,
            vec!["check_line".to_string(), "mirror".to_string()]
        );
        assert_eq!(
            f.lines,
            vec![("src/engine.rs".to_string(), 2, "let x = 1;".to_string())]
        );
        let f = r.check("see engine.rs:449 and src/check.rs");
        assert!(
            f.bad_refs
                .iter()
                .any(|b| b.contains("has 3 lines, not 449")),
            "{f:?}"
        );
        assert!(
            f.bad_refs
                .iter()
                .any(|b| b.contains("src/check.rs does not exist")),
            "{f:?}"
        );
    }

    #[test]
    fn whole_words_only() {
        assert!(has_word("let keep_at = 1;", "keep_at"));
        assert!(!has_word("let keep_at_all = 1;", "keep_at"));
        assert!(!has_word("mirrors", "mirror"));
    }
}
