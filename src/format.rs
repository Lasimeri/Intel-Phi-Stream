//! The stream's text set for the terminal. Fenced code blocks are kept as
//! written (their indentation, no reflow) and highlighted by the block's
//! language; prose is wrapped by words with its own indentation kept and
//! a hanging indent under list items; the light Markdown the stream
//! writes (headings, **bold**, `code`) is shown as a style instead of its
//! marks. Pure functions over the stream's lines; the terminal (`tui.rs`)
//! draws the rows. See format.md.

use crate::engine::Kind;
use crate::screen::columns;

/// What a character is, for its style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Running text.
    Prose,
    /// Between `**` marks.
    Bold,
    /// A `#` heading's text.
    Heading,
    /// Between single backticks in prose.
    Code,
    /// A fence line (```` ```lang ```` or the closing ```` ``` ````).
    Fence,
    /// Code that is none of the below.
    Plain,
    Keyword,
    Str,
    Comment,
    Number,
    /// A row of what the J-lens read on its mind under a line it wrote
    /// (a line that begins with `LENS_MARK`).
    Lens,
    /// A word a check wrote over, between two `STRUCK` marks.
    Struck,
}

/// Begins a line that is a lens row, not the stream's text (a private-use
/// character, which the model's text does not hold).
pub const LENS_MARK: char = '\u{E000}';
/// On each side of a word a check wrote over.
pub const STRUCK: char = '\u{E001}';

impl Class {
    /// Whether it belongs to a code block (drawn on the code background).
    pub fn in_block(self) -> bool {
        matches!(
            self,
            Class::Fence
                | Class::Plain
                | Class::Keyword
                | Class::Str
                | Class::Comment
                | Class::Number
        )
    }
}

/// A code block's language, from its fence's info string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Rust,
    /// C and C++.
    C,
    /// TypeScript and JavaScript.
    TypeScript,
    Shell,
    Python,
    /// Any other, or none: comments and strings as in C.
    Other,
}

impl Lang {
    pub fn from_info(info: &str) -> Lang {
        let word = info
            .trim()
            .split(|c: char| c.is_whitespace() || c == ',' || c == '{')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match word.as_str() {
            "rust" | "rs" => Lang::Rust,
            "c" | "h" | "cpp" | "c++" | "cc" | "cxx" | "hpp" => Lang::C,
            "typescript" | "ts" | "tsx" | "javascript" | "js" | "jsx" | "mjs" | "cjs" => {
                Lang::TypeScript
            }
            "sh" | "bash" | "shell" | "zsh" | "fish" | "console" => Lang::Shell,
            "python" | "py" => Lang::Python,
            _ => Lang::Other,
        }
    }

    fn keywords(self) -> &'static [&'static str] {
        match self {
            Lang::Rust => &[
                "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else",
                "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
                "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
                "super", "trait", "true", "type", "unsafe", "use", "where", "while",
            ],
            Lang::C | Lang::Other => &[
                "auto",
                "bool",
                "break",
                "case",
                "char",
                "const",
                "continue",
                "default",
                "do",
                "double",
                "else",
                "enum",
                "extern",
                "float",
                "for",
                "goto",
                "if",
                "inline",
                "int",
                "long",
                "register",
                "return",
                "short",
                "signed",
                "sizeof",
                "static",
                "struct",
                "switch",
                "typedef",
                "union",
                "unsigned",
                "void",
                "volatile",
                "while",
                "class",
                "namespace",
                "template",
                "typename",
                "public",
                "private",
                "protected",
                "new",
                "delete",
                "true",
                "false",
                "nullptr",
                "NULL",
                "#include",
                "#define",
                "#if",
                "#ifdef",
                "#ifndef",
                "#endif",
                "#else",
                "#pragma",
            ],
            Lang::TypeScript => &[
                "abstract",
                "any",
                "as",
                "async",
                "await",
                "boolean",
                "break",
                "case",
                "catch",
                "class",
                "const",
                "constructor",
                "continue",
                "declare",
                "default",
                "delete",
                "do",
                "else",
                "enum",
                "export",
                "extends",
                "false",
                "finally",
                "for",
                "from",
                "function",
                "get",
                "if",
                "implements",
                "import",
                "in",
                "instanceof",
                "interface",
                "keyof",
                "let",
                "namespace",
                "never",
                "new",
                "null",
                "number",
                "of",
                "private",
                "protected",
                "public",
                "readonly",
                "return",
                "set",
                "static",
                "string",
                "super",
                "switch",
                "this",
                "throw",
                "true",
                "try",
                "type",
                "typeof",
                "undefined",
                "unknown",
                "var",
                "void",
                "while",
                "yield",
            ],
            Lang::Shell => &[
                "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case",
                "esac", "in", "function", "return", "local", "export", "readonly", "set", "unset",
                "shift", "exit", "exec", "echo", "printf", "read", "cd", "source",
            ],
            Lang::Python => &[
                "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
                "del", "elif", "else", "except", "False", "finally", "for", "from", "global", "if",
                "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise",
                "return", "True", "try", "while", "with", "yield",
            ],
        }
    }

    /// Comments: `#` to the end of the line, or C's `//` and `/* */`.
    fn hash_comments(self) -> bool {
        matches!(self, Lang::Shell | Lang::Python)
    }
}

/// One character set: itself, the kind of its piece of the stream, its class.
pub type Cell = (char, Kind, Class);

/// A fence line: ```` ``` ```` or `~~~`, after up to three spaces, and the
/// info string after it.
fn fence(line: &[(char, Kind)]) -> Option<String> {
    let s: String = line.iter().map(|c| c.0).collect();
    let t = s.trim_start_matches(' ');
    if s.len() - t.len() > 3 {
        return None;
    }
    t.strip_prefix("```")
        .or_else(|| t.strip_prefix("~~~"))
        .map(|rest| rest.trim_start_matches(['`', '~']).trim().to_string())
}

fn ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == '#' || c == '$'
}

fn ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The classes of one line of code in `lang`; `block`: inside a `/* */`
/// comment, carried from line to line.
fn highlight(chars: &[char], lang: Lang, block: &mut bool) -> Vec<Class> {
    let n = chars.len();
    let mut cls = vec![Class::Plain; n];
    let at = |i: usize| chars.get(i).copied().unwrap_or('\0');
    let mut i = 0;
    while i < n {
        if *block {
            cls[i] = Class::Comment;
            if at(i) == '*' && at(i + 1) == '/' {
                cls[i + 1] = Class::Comment;
                *block = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        let c = at(i);
        let line_comment = if lang.hash_comments() {
            c == '#' && (i == 0 || !ident_char(at(i - 1)) && at(i - 1) != '$')
        } else {
            c == '/' && at(i + 1) == '/'
        };
        if line_comment {
            cls[i..].fill(Class::Comment);
            break;
        }
        if !lang.hash_comments() && c == '/' && at(i + 1) == '*' {
            cls[i] = Class::Comment;
            cls[i + 1] = Class::Comment;
            *block = true;
            i += 2;
            continue;
        }
        // A Rust lifetime ('a) or a label is not a string.
        let quote = c == '"'
            || (c == '`' && matches!(lang, Lang::TypeScript | Lang::Shell))
            || (c == '\'' && !(lang == Lang::Rust && ident_start(at(i + 1)) && at(i + 2) != '\''));
        if quote {
            let mut j = i + 1;
            while j < n && at(j) != c {
                j += if at(j) == '\\' { 2 } else { 1 };
            }
            let end = (j + 1).min(n);
            cls[i..end].fill(Class::Str);
            i = end;
            continue;
        }
        if c.is_ascii_digit() && (i == 0 || !ident_char(at(i - 1))) {
            let mut j = i;
            while j < n && (ident_char(at(j)) || at(j) == '.') {
                j += 1;
            }
            cls[i..j].fill(Class::Number);
            i = j;
            continue;
        }
        if ident_start(c) {
            let mut j = i + 1;
            while j < n && ident_char(at(j)) {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            // `#` and `$` begin words only for C's directives.
            if (c == '#' || c == '$') && !lang.keywords().contains(&word.as_str()) {
                i += 1;
                continue;
            }
            if lang.keywords().contains(&word.as_str()) {
                cls[i..j].fill(Class::Keyword);
            }
            i = j;
            continue;
        }
        i += 1;
    }
    cls
}

/// A prose line's cells with its Markdown marks taken out: a heading's
/// `#`s, and the `**` and backtick pairs that close on the line (a mark
/// still open, as while the stream is writing it, is shown as it is).
fn prose(line: &[(char, Kind)]) -> Vec<Cell> {
    let chars: Vec<char> = line.iter().map(|c| c.0).collect();
    let n = chars.len();
    let lead = chars.iter().take_while(|c| **c == ' ').count();
    let hashes = chars[lead..].iter().take_while(|c| **c == '#').count();
    if (1..=6).contains(&hashes) && chars.get(lead + hashes) == Some(&' ') {
        return line[lead + hashes + 1..]
            .iter()
            .map(|&(c, k)| (c, k, Class::Heading))
            .collect();
    }
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        // A word written over, shown between tildes (crossed out where the
        // terminal can, and readable as such in a plain capture).
        if chars[i] == STRUCK {
            if let Some(j) = (i + 1..n).find(|&j| chars[j] == STRUCK) {
                let k = line[i].1;
                out.push(('~', k, Class::Struck));
                out.extend(line[i + 1..j].iter().map(|&(c, k)| (c, k, Class::Struck)));
                out.push(('~', k, Class::Struck));
                i = j + 1;
                continue;
            }
            i += 1;
            continue;
        }
        if chars[i] == '`' {
            if let Some(j) = (i + 1..n).find(|&j| chars[j] == '`') {
                if j > i + 1 {
                    out.extend(line[i + 1..j].iter().map(|&(c, k)| (c, k, Class::Code)));
                    i = j + 1;
                    continue;
                }
            }
        }
        if chars[i] == '*' && chars.get(i + 1) == Some(&'*') {
            if let Some(j) =
                (i + 2..n.saturating_sub(1)).find(|&j| chars[j] == '*' && chars[j + 1] == '*')
            {
                if j > i + 2 {
                    out.extend(line[i + 2..j].iter().map(|&(c, k)| (c, k, Class::Bold)));
                    i = j + 2;
                    continue;
                }
            }
        }
        out.push((line[i].0, line[i].1, Class::Prose));
        i += 1;
    }
    out
}

/// The hanging indent of a prose line: its leading spaces, and a list
/// marker's width after them (`- `, `* `, `1. `).
fn hanging(cells: &[Cell]) -> usize {
    let lead = cells.iter().take_while(|c| c.0 == ' ').count();
    let rest: String = cells[lead..].iter().take(5).map(|c| c.0).collect();
    let marker = if rest.starts_with("- ") || rest.starts_with("* ") {
        2
    } else {
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && rest[digits..].starts_with(". ") {
            digits + 2
        } else {
            0
        }
    };
    lead + marker
}

fn cols(cells: &[Cell]) -> usize {
    cells.iter().map(|c| columns(c.0)).sum()
}

/// Whether a line of text opens or closes a code block (`rows` treats it so).
pub fn is_fence(line: &str) -> bool {
    let cells: Vec<(char, Kind)> = line.chars().map(|c| (c, Kind::Think)).collect();
    fence(&cells).is_some()
}

/// The chat template's marks, which a person reading the stream has no use
/// for (the agent frame's turns are framed by them).
const MARKS: &[&str] = &[
    "<|im_start|>assistant",
    "<|im_start|>user",
    "<|im_start|>system",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
];

/// A line's text without the template's marks, its characters' kinds kept.
fn unmarked(line: &[(char, Kind)]) -> Vec<(char, Kind)> {
    let mut out: Vec<(char, Kind)> = line.to_vec();
    for m in MARKS {
        let m: Vec<char> = m.chars().collect();
        let mut i = 0;
        while i + m.len() <= out.len() {
            if out[i..i + m.len()]
                .iter()
                .map(|c| c.0)
                .eq(m.iter().copied())
            {
                out.drain(i..i + m.len());
            } else {
                i += 1;
            }
        }
    }
    out
}

fn text_of(line: &[(char, Kind)]) -> String {
    line.iter().map(|c| c.0).collect()
}

/// A line of `text` in the kind `kind`.
fn line_of(text: &str, kind: Kind) -> Vec<(char, Kind)> {
    text.chars().map(|c| (c, kind)).collect()
}

/// A tool's result in one line: a command's as how it ended and its
/// output's first line (the call's line above names the command), anything
/// else as its first line; with the count of the lines not shown.
fn result_line(body: &[String]) -> String {
    let more = |n: usize| match n {
        0 => String::new(),
        1 => " (1 more line)".to_string(),
        n => format!(" ({n} more lines)"),
    };
    let Some(first) = body.first() else {
        return String::new();
    };
    if first.starts_with("the command `") {
        let out = body.get(1).map_or("(no output)", |s| s.as_str());
        let rest = more(body.len().saturating_sub(2));
        if let Some((_, after)) = first.split_once("` ended (") {
            let how = after.split_once("); its output").map_or(after, |(h, _)| h);
            return format!("{how}: {out}{rest}");
        }
        if first.contains("did NOT finish") {
            return format!("did not finish (stopped at its time limit): {out}{rest}");
        }
    }
    format!("{first}{}", more(body.len() - 1))
}

/// The stream's lines as a person reads them: the chat template's marks
/// taken out (a line left empty by it goes, and blank lines run to one), a
/// tool call (`<tool_call>` to `</tool_call>`) as one line, `▸ NAME: ` and
/// its first parameter's first line, a tool's result (`<tool_response>` to
/// `</tool_response>`) as one line, `◂ ` and `result_line`. A block still
/// being written is shown so far. Lines a marker begins (`LENS_MARK`) pass
/// as they are.
pub fn readable(lines: &[Vec<(char, Kind)>]) -> Vec<Vec<(char, Kind)>> {
    let mut out: Vec<Vec<(char, Kind)>> = Vec::new();
    let mut i = 0;
    let push = |out: &mut Vec<Vec<(char, Kind)>>, l: Vec<(char, Kind)>| {
        let blank = l.iter().all(|c| c.0 == ' ');
        if blank && out.last().is_none_or(|p| p.iter().all(|c| c.0 == ' ')) {
            return;
        }
        out.push(l);
    };
    while i < lines.len() {
        let raw = &lines[i];
        if raw.first().is_some_and(|c| c.0 == LENS_MARK) {
            out.push(raw.clone());
            i += 1;
            continue;
        }
        let line = unmarked(raw);
        let t = text_of(&line);
        let kind = line.first().map_or(Kind::Think, |c| c.1);
        match t.trim() {
            "<tool_call>" => {
                let (mut name, mut value, mut more) = (String::new(), None::<String>, false);
                let mut in_param = false;
                i += 1;
                while i < lines.len() {
                    let l = text_of(&unmarked(&lines[i]));
                    let lt = l.trim();
                    i += 1;
                    if lt == "</tool_call>" {
                        break;
                    }
                    if let Some(n) = lt.strip_prefix("<function=") {
                        name = n.trim_end_matches('>').to_string();
                    } else if lt.starts_with("<parameter=") {
                        in_param = true;
                    } else if lt == "</parameter>" {
                        in_param = false;
                    } else if in_param && lt != "</function>" {
                        match &value {
                            None if !lt.is_empty() => value = Some(lt.to_string()),
                            Some(_) if !lt.is_empty() => more = true,
                            _ => {}
                        }
                    }
                }
                let v = value.unwrap_or_default();
                let text = format!("▸ {name}: {v}{}", if more { " …" } else { "" });
                push(&mut out, line_of(&text, kind));
            }
            "<tool_response>" => {
                let mut body: Vec<String> = Vec::new();
                i += 1;
                while i < lines.len() {
                    let l = text_of(&unmarked(&lines[i]));
                    let lt = l.trim().to_string();
                    i += 1;
                    if lt == "</tool_response>" {
                        break;
                    }
                    if !lt.is_empty() {
                        body.push(lt);
                    }
                }
                push(
                    &mut out,
                    line_of(&format!("◂ {}", result_line(&body)), kind),
                );
            }
            _ => {
                // A line the marks alone made: gone, not left blank.
                if !(t.trim().is_empty() && !text_of(raw).trim().is_empty()) {
                    push(&mut out, line);
                }
                i += 1;
            }
        }
    }
    out
}

/// The stream's lines (each a run of characters with their kinds, no
/// newlines) as rows of at most `width` columns.
pub fn rows(lines: &[Vec<(char, Kind)>], width: usize) -> Vec<Vec<Cell>> {
    let width = width.max(8);
    let mut out = Vec::new();
    let mut code: Option<Lang> = None;
    let mut block = false;
    for raw in lines {
        // Tabs as four spaces, as a terminal shows code.
        let line: Vec<(char, Kind)> = raw
            .iter()
            .flat_map(|&(c, k)| {
                let n = if c == '\t' { 4 } else { 1 };
                std::iter::repeat_n((if c == '\t' { ' ' } else { c }, k), n)
            })
            .collect();
        // A lens row: its own style, wrapped by words, never code.
        if line.first().is_some_and(|c| c.0 == LENS_MARK) {
            let cells: Vec<Cell> = line[1..]
                .iter()
                .map(|&(c, k)| (c, k, Class::Lens))
                .collect();
            word_wrap(&cells, width, &mut out);
            continue;
        }
        if let Some(info) = fence(&line) {
            code = match code {
                None => Some(Lang::from_info(&info)),
                Some(_) => None,
            };
            block = false;
            hard_wrap(
                &line
                    .iter()
                    .map(|&(c, k)| (c, k, Class::Fence))
                    .collect::<Vec<_>>(),
                width,
                &mut out,
            );
            continue;
        }
        match code {
            Some(lang) => {
                let chars: Vec<char> = line.iter().map(|c| c.0).collect();
                let cls = highlight(&chars, lang, &mut block);
                let cells: Vec<Cell> = line
                    .iter()
                    .zip(cls)
                    .map(|(&(c, k), cl)| (c, k, cl))
                    .collect();
                hard_wrap(&cells, width, &mut out);
            }
            None => word_wrap(&prose(&line), width, &mut out),
        }
    }
    out
}

/// Code: cut at the width, never reflowed; a continuation indented two
/// columns past the line's own indentation, so it reads as one line.
fn hard_wrap(cells: &[Cell], width: usize, out: &mut Vec<Vec<Cell>>) {
    let lead = cells
        .iter()
        .take_while(|c| c.0 == ' ')
        .count()
        .min(width / 2);
    let pad = |k: Kind, cl: Class| vec![(' ', k, cl); lead + 2];
    let mut row: Vec<Cell> = Vec::new();
    let mut c = 0;
    for &cell in cells {
        let w = columns(cell.0);
        if c + w > width && !row.is_empty() {
            out.push(std::mem::take(&mut row));
            row = pad(cell.1, Class::Plain);
            c = lead + 2;
        }
        row.push(cell);
        c += w;
    }
    out.push(row);
}

/// Prose: by words, the line's leading spaces kept, continuations under
/// its hanging indent.
fn word_wrap(cells: &[Cell], width: usize, out: &mut Vec<Vec<Cell>>) {
    let hang = hanging(cells).min(width / 2);
    let mut row: Vec<Cell> = Vec::new();
    let mut c = 0;
    let mut i = 0;
    let n = cells.len();
    while i < n {
        // A word: its non-spaces and the spaces after it.
        let mut j = i;
        while j < n && cells[j].0 != ' ' {
            j += 1;
        }
        while j < n && cells[j].0 == ' ' {
            j += 1;
        }
        let word = &cells[i..j];
        let w = cols(
            &word
                .iter()
                .copied()
                .filter(|x| x.0 != ' ')
                .collect::<Vec<_>>(),
        );
        if c + w > width && c > hang {
            out.push(std::mem::take(&mut row));
            row = vec![(' ', word[0].1, Class::Prose); hang];
            c = hang;
        }
        for &cell in word {
            let cw = columns(cell.0);
            if c + cw > width {
                // A word longer than the row: cut, under the indent.
                if cell.0 == ' ' {
                    continue;
                }
                out.push(std::mem::take(&mut row));
                row = vec![(' ', cell.1, Class::Prose); hang];
                c = hang;
            }
            row.push(cell);
            c += cw;
        }
        i = j;
    }
    out.push(row);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<Vec<(char, Kind)>> {
        text.split('\n')
            .map(|l| l.chars().map(|c| (c, Kind::Think)).collect())
            .collect()
    }

    fn texts(rows: &[Vec<Cell>]) -> Vec<String> {
        rows.iter()
            .map(|r| r.iter().map(|c| c.0).collect())
            .collect()
    }

    fn class_of(rows: &[Vec<Cell>], row: usize, word: &str) -> Class {
        let t: String = rows[row].iter().map(|c| c.0).collect();
        let at = t
            .find(word)
            .unwrap_or_else(|| panic!("{word:?} not in {t:?}"));
        let ci = t[..at].chars().count();
        rows[row][ci].2
    }

    #[test]
    fn code_keeps_its_indentation_and_is_never_reflowed() {
        let r = rows(
            &lines("Here:\n```rust\nfn f() {\n    if x {\n        return 1;\n    }\n}\n```\nafter"),
            40,
        );
        let t = texts(&r);
        assert_eq!(t[2], "fn f() {");
        assert_eq!(t[3], "    if x {");
        assert_eq!(t[4], "        return 1;");
        assert_eq!(class_of(&r, 2, "fn"), Class::Keyword);
        assert_eq!(class_of(&r, 4, "1"), Class::Number);
        assert_eq!(r[1][0].2, Class::Fence);
        assert_eq!(class_of(&r, 8, "after"), Class::Prose);
    }

    #[test]
    fn typescript_is_highlighted() {
        let r = rows(
            &lines("```ts\nexport const f = async (x: number): Promise<string> => `n=${x}`; // done\n```"),
            200,
        );
        assert_eq!(class_of(&r, 1, "export"), Class::Keyword);
        assert_eq!(class_of(&r, 1, "async"), Class::Keyword);
        assert_eq!(class_of(&r, 1, "number"), Class::Keyword);
        assert_eq!(class_of(&r, 1, "`n="), Class::Str);
        assert_eq!(class_of(&r, 1, "// done"), Class::Comment);
        assert_eq!(class_of(&r, 1, "Promise"), Class::Plain);
    }

    #[test]
    fn c_comments_span_lines_and_strings_hold_escapes() {
        let r = rows(
            &lines("```c\n#include <stdio.h>\n/* a\nb */ int x = \"q\\\"\"; // c\n```"),
            80,
        );
        assert_eq!(class_of(&r, 1, "#include"), Class::Keyword);
        assert_eq!(class_of(&r, 2, "a"), Class::Comment);
        assert_eq!(class_of(&r, 3, "b */"), Class::Comment);
        assert_eq!(class_of(&r, 3, "int"), Class::Keyword);
        assert_eq!(class_of(&r, 3, "\"q"), Class::Str);
        assert_eq!(class_of(&r, 3, "// c"), Class::Comment);
    }

    #[test]
    fn rust_lifetimes_are_not_strings() {
        let r = rows(
            &lines("```rust\nfn f<'a>(s: &'a str) -> char { 'x' }\n```"),
            80,
        );
        assert_eq!(class_of(&r, 1, "str"), Class::Plain);
        assert_eq!(class_of(&r, 1, "'x'"), Class::Str);
    }

    #[test]
    fn long_code_lines_are_cut_under_their_indent() {
        let r = rows(
            &lines("```\n    let a_long_name = another_long_name + yet_another;\n```"),
            24,
        );
        let t = texts(&r);
        assert!(t.len() > 3);
        for row in &t {
            assert!(row.chars().map(columns).sum::<usize>() <= 24, "{row:?}");
        }
        assert!(t[2].starts_with("      "), "{:?}", t[2]);
    }

    #[test]
    fn prose_wraps_by_words_with_its_hanging_indent() {
        let r = rows(&lines("- a list item that is long enough to wrap twice here\n  indented text stays indented"), 20);
        let t = texts(&r);
        assert!(t[0].starts_with("- a list"));
        assert!(t[1].starts_with("  "), "{t:?}");
        let last = t.iter().find(|x| x.contains("indented text")).unwrap();
        assert!(last.starts_with("  indented"), "{t:?}");
        for row in &t {
            assert!(row.chars().map(columns).sum::<usize>() <= 20, "{row:?}");
        }
    }

    #[test]
    fn markdown_marks_become_styles_only_when_closed() {
        let r = rows(&lines("# Title\nuse `keep_at` and **this**, not `open"), 80);
        let t = texts(&r);
        assert_eq!(t[0], "Title");
        assert_eq!(r[0][0].2, Class::Heading);
        assert_eq!(t[1], "use keep_at and this, not `open");
        assert_eq!(class_of(&r, 1, "keep_at"), Class::Code);
        assert_eq!(class_of(&r, 1, "this"), Class::Bold);
        assert_eq!(class_of(&r, 1, "`open"), Class::Prose);
    }

    #[test]
    fn an_open_block_stays_code_while_it_streams() {
        let r = rows(&lines("```python\ndef f():\n    return None"), 80);
        assert_eq!(class_of(&r, 1, "def"), Class::Keyword);
        assert_eq!(texts(&r)[2], "    return None");
        assert_eq!(class_of(&r, 2, "None"), Class::Keyword);
    }

    fn readable_texts(text: &str) -> Vec<String> {
        readable(&lines(text))
            .iter()
            .map(|l| l.iter().map(|c| c.0).collect())
            .collect()
    }

    #[test]
    fn a_person_reads_no_template_marks_and_one_line_per_tool() {
        // The shape of an agent turn on the live service (2026-10-02).
        let t = readable_texts(
            "Let me check.\n\n<tool_call>\n<function=run>\n<parameter=command>\ncd repo && git status --short\n</parameter>\n</function>\n</tool_call>\n<|im_start|>user\n<tool_response>\nthe command `git status --short` ended (exit 0, 26 ms); its output:\n M src/a.rs\n M src/b.rs\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nClean enough.",
        );
        assert_eq!(
            t,
            vec![
                "Let me check.",
                "",
                "▸ run: cd repo && git status --short",
                "◂ exit 0, 26 ms: M src/a.rs (1 more line)",
                "Clean enough.",
            ]
        );
    }

    #[test]
    fn a_call_still_being_written_shows_so_far() {
        let t = readable_texts("<tool_call>\n<function=read>\n<parameter=path>\nsrc/engine.rs");
        assert_eq!(t, vec!["▸ read: src/engine.rs"]);
        let t = readable_texts("<tool_call>\n<function=write>\n<parameter=content>\nfn a() {}\nfn b() {}\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(t, vec!["▸ write: fn a() {} …"]);
    }

    #[test]
    fn lens_rows_and_struck_words_have_their_classes() {
        let r = rows(
            &lines(&format!(
                "I {STRUCK}may{STRUCK} likely need it\n{LENS_MARK}on its mind: again 13%"
            )),
            80,
        );
        let t = texts(&r);
        assert_eq!(t[0], "I ~may~ likely need it");
        assert_eq!(class_of(&r, 0, "~may~"), Class::Struck);
        assert_eq!(class_of(&r, 0, "likely"), Class::Prose);
        assert_eq!(t[1], "on its mind: again 13%");
        assert_eq!(r[1][0].2, Class::Lens);
        assert!(is_fence("```rust") && is_fence("```") && !is_fence("a ```"));
    }

    #[test]
    fn wide_characters_count_two_columns() {
        for row in texts(&rows(&lines("而不是 而非 而不是 而非 而不是"), 10)) {
            assert!(row.chars().map(columns).sum::<usize>() <= 10, "{row:?}");
        }
    }
}
