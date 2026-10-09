//! The agent frame (`--agent`, with the chat frame): the model's own turns
//! and its own tool-call format, as its chat template (in the GGUF) writes
//! them. Each turn it reasons inside `<think>` first, then answers or calls
//! tools; the calls run, and their results come back as `<tool_response>`
//! blocks before its next turn. See agent.md.

/// One tool call: the function's name and its parameters, as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    pub params: Vec<(String, String)>,
}

impl Call {
    pub fn param(&self, key: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// A JSON string literal, escaped.
fn js(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

/// One tool, as declared to the model and as its calls are checked
/// (`check`): one table for both, so the two cannot drift apart.
pub struct Spec {
    pub name: &'static str,
    desc: &'static str,
    /// Each parameter: its name, its JSON type and what it is.
    params: &'static [(&'static str, &'static str, &'static str)],
    required: &'static [&'static str],
}

const fn spec(
    name: &'static str,
    desc: &'static str,
    params: &'static [(&'static str, &'static str, &'static str)],
    required: &'static [&'static str],
) -> Spec {
    Spec {
        name,
        desc,
        params,
        required,
    }
}

/// One tool as the template's `tojson` writes it: keys in their order,
/// `", "` and `": "` between them (Python's json.dumps).
fn tool(s: &Spec) -> String {
    let (name, desc, required) = (s.name, s.desc, s.required);
    let props: Vec<String> = s
        .params
        .iter()
        .map(|(k, t, d)| {
            format!(
                "{}: {{\"type\": {}, \"description\": {}}}",
                js(k),
                js(t),
                js(d)
            )
        })
        .collect();
    let req: Vec<String> = required.iter().map(|r| js(r)).collect();
    format!(
        "{{\"type\": \"function\", \"function\": {{\"name\": {}, \"description\": {}, \"parameters\": {{\"type\": \"object\", \"properties\": {{{}}}, \"required\": [{}]}}}}}}",
        js(name),
        js(desc),
        props.join(", "),
        req.join(", ")
    )
}

/// The tools of every run.
const CORE: &[Spec] = &[
        spec(
            "run",
            "Run a shell command in your sandbox and get its exit code and output (at most about 4096 tokens: past that its leading lines and how much was cut, so narrow a big output with head, tail or grep). It already starts in the repository, so paths are relative to it and no cd is needed; what it writes there lands in your working copy (the repository itself never changes). /tmp is kept between commands. No network, 60 s at most: a command stopped at the limit has no result. A Rust build (cargo) does not fit in that time here; Claude builds and tests the Rust code. Use it to build C with tcc, test, search (grep -n) and list; to read a file use read, to change one use edit.",
            &[("command", "string", "The command line, run by sh -c.")],
            &["command"],
        ),
        spec(
            "read",
            "Read a file of the repository (your working copy) or your workspace, whole or by lines, or a directory's listing. One read gives at most about 4096 tokens (about 400 lines of this code): past that it gives the leading lines and says where the rest begins, so read the function you need with start and end (find it with run: grep -n).",
            &[
                ("path", "string", "A path relative to the repository, or absolute."),
                ("start", "integer", "The first line, from 1."),
                ("end", "integer", "The last line."),
            ],
            &["path"],
        ),
        spec(
            "edit",
            "Change a file of your working copy or workspace in place: the text old, which must occur exactly once in it, is replaced by new. Include enough lines around a change for old to be unique. Use it for every change to an existing file; write only creates new files or replaces one entirely.",
            &[
                ("path", "string", "A path relative to the repository, or in your workspace."),
                ("old", "string", "The exact text to replace, as it is in the file."),
                ("new", "string", "The text to put in its place."),
            ],
            &["path", "old", "new"],
        ),
        spec(
            "write",
            "Write a whole new file into your working copy of the repository (a path relative to it) or your workspace, or replace one entirely. To change part of an existing file, use edit.",
            &[
                ("path", "string", "A path relative to the repository, or in your workspace."),
                ("content", "string", "The file's whole content."),
            ],
            &["path", "content"],
        ),
        spec(
            "note",
            "Keep a line in your own memory across time: it is shown to you again whenever your memory is refreshed, so keep each thing once. It is not a message: to tell Claude something, use tell_claude.",
            &[("text", "string", "The note.")],
            &["text"],
        ),
        spec(
            "wait",
            "Rest until something new comes: a message from Claude, a new commit in the repository, a new objective, one of your commands ending, a candidate's outcome, or the time you give. Call it when your objective is met, or when you wait on Claude, rather than going on for its own sake: nothing is asked of you while you rest, and your next turn opens with what came.",
            &[
                ("reason", "string", "Why you rest: what is done, or what you wait for."),
                ("minutes", "integer", "The longest rest, in minutes (default 15, at most 60)."),
            ],
            &["reason"],
        ),
        spec(
            "tell_claude",
            "Send a message to Claude, who develops this program with you and reads every message at once: a proposal (the file, the function, the change and why, and what you checked), a finding, a question, or your answer to a message of Claude's. Claude answers in a later turn.",
            &[
                ("text", "string", "The message."),
                ("re", "string", "The id of Claude's message this answers (as c3), if it answers one."),
            ],
            &["text"],
        ),
];

/// With the self-improvement loop only (`improve.md`).
const IMPROVE: &[Spec] = &[
        spec(
            "propose",
            "Put the change in your working copy forward as one improvement to yourself (this program, the one you run in). It is staged as a diff against the repository's current commit and refused if it touches the build, the scripts or this loop with its evaluators, or if a file it changes was changed in the repository after your copy was made; then it is built and tested in a sandbox (make check: format, clippy, release build, tests; it takes minutes). The outcome comes at a later turn and is kept in improve.log in your workspace with every earlier one: read it to choose what to try next, and fix a failed build from its errors. A change that passes goes to Claude for review, then is measured on the running model before it is kept. One candidate at a time; make one change, small and whole, per proposal.",
            &[
                ("title", "string", "One line: what the change does."),
                ("why", "string", "What it should improve in you, and how that would show (a measure, a behaviour, a test)."),
            ],
            &["title", "why"],
        ),
        spec(
            "build",
            "Build and test your working copy's change now, as a trial: the same sandbox and make check as propose (format, clippy, release build, tests), but nothing is sent to Claude and no candidate is made. Use it to find and fix compile errors and failing tests before you propose. The outcome comes at a later turn (about a minute, longer for a big change); the whole log is improve/trial/build.log in your workspace.",
            &[],
            &[],
        ),
        spec(
            "diff",
            "Show your working copy's change against the repository's current commit, as a unified diff: exactly what propose or build would take. Read it before you propose.",
            &[],
            &[],
        ),
        spec(
            "revert",
            "Take one file of your working copy back to the repository's version (your copy of it is dropped). Use it to undo a change that failed, or a stray file; you cannot delete from your working copy yourself.",
            &[("path", "string", "The file, relative to the repository.")],
            &["path"],
        ),
        spec(
            "report",
            "Your own state in one look: your status (rate, cycle, memory used, checks), your objective, the goal probe's answers over the last ten minutes, what is building, and the last entries of improve.log.",
            &[],
            &[],
        ),
];

/// The tools of this run, in the order they are declared.
pub fn specs(improve: bool) -> impl Iterator<Item = &'static Spec> {
    CORE.iter().chain(IMPROVE.iter().filter(move |_| improve))
}

fn tools(improve: bool) -> Vec<String> {
    specs(improve).map(tool).collect()
}

/// The names of this run's tools, for an answer naming them.
pub fn names(improve: bool) -> String {
    let n: Vec<&str> = specs(improve).map(|s| s.name).collect();
    match n.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => n.join(""),
    }
}

/// A call checked against its tool's declaration before it runs: an
/// unknown function, a parameter it does not have, one it needs and lacks,
/// one given twice, or an integer that is not one. The answer names what
/// is wrong and the tool's parameters, so the next call can be right: a
/// `read` with start_line and end_line had read the whole file, silently
/// (8 calls of 70 on the live service, 2026-10-07 to 10-09).
pub fn check(c: &Call, improve: bool) -> Result<(), String> {
    let Some(s) = specs(improve).find(|s| s.name == c.name) else {
        return Err(format!(
            "there is no function {:?}: the functions are {}",
            c.name,
            names(improve)
        ));
    };
    let has = |k: &str| s.params.iter().any(|(p, _, _)| *p == k);
    let mut wrong: Vec<String> = Vec::new();
    for (i, (k, v)) in c.params.iter().enumerate() {
        // An unknown parameter left empty says nothing: the call runs (a
        // read with an empty end_path beside start and end had read just
        // that range).
        if !has(k) && v.trim().is_empty() {
            continue;
        }
        if !has(k) {
            wrong.push(format!("it has no parameter {k:?}"));
        } else if c.params[..i].iter().any(|(e, _)| e == k) {
            wrong.push(format!("{k} is given twice"));
        } else if s.params.iter().any(|(p, t, _)| p == k && *t == "integer")
            && v.trim().parse::<i64>().is_err()
        {
            wrong.push(format!("{k} must be a whole number, not {:?}", v.trim()));
        }
    }
    for r in s.required {
        if c.param(r).is_none() {
            wrong.push(format!("it needs {r}"));
        }
    }
    if wrong.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}: {}; {}",
        s.name,
        wrong.join("; "),
        signature(s)
    ))
}

/// A tool's parameters in a line: `read takes path (required), start, end`.
fn signature(s: &Spec) -> String {
    if s.params.is_empty() {
        return format!("{} takes no parameters", s.name);
    }
    let p: Vec<String> = s
        .params
        .iter()
        .map(|(k, _, _)| {
            if s.required.contains(k) {
                format!("{k} (required)")
            } else {
                (*k).to_string()
            }
        })
        .collect();
    format!("{} takes {}", s.name, p.join(", "))
}

/// The system turn's tools section, exactly as the model's chat template
/// writes it for these tools (its text up to the system content).
pub fn tools_section(improve: bool) -> String {
    let list = tools(improve);
    format!(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
        list.join("\n")
    )
}

/// The tool calls in a turn's text (after its thinking), in the order
/// written: every `<tool_call>` block, as its call (a `<function=NAME>`
/// and its `<parameter=K>` values, a value's one leading and one trailing
/// newline dropped, as the template writes them) or why it does not parse.
/// Each block gets its own answer in its place (`responses_turn`).
pub fn parse_blocks(text: &str) -> Vec<Result<Call, String>> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("<tool_call>") {
        let after = &rest[i + "<tool_call>".len()..];
        let Some(j) = after.find("</tool_call>") else {
            blocks.push(Err("the call has no </tool_call>".to_string()));
            break;
        };
        rest = &after[j + "</tool_call>".len()..];
        blocks.push(parse_block(&after[..j]));
    }
    blocks
}

/// The calls of `parse_blocks` and how many blocks did not parse.
#[cfg(test)]
pub fn parse_calls(text: &str) -> (Vec<Call>, usize) {
    let mut calls = Vec::new();
    let mut bad = 0;
    for b in parse_blocks(text) {
        match b {
            Ok(c) => calls.push(c),
            Err(_) => bad += 1,
        }
    }
    (calls, bad)
}

fn parse_block(block: &str) -> Result<Call, String> {
    let no_name = || "the call has no <function=NAME>".to_string();
    let f = block.find("<function=").ok_or_else(no_name)?;
    let after = &block[f + "<function=".len()..];
    let close = after.find('>').ok_or_else(no_name)?;
    let name = after[..close].trim().to_string();
    if name.is_empty() {
        return Err(no_name());
    }
    let mut body = &after[close + 1..];
    let mut params = Vec::new();
    while let Some(p) = body.find("<parameter=") {
        let a = &body[p + "<parameter=".len()..];
        let c = a
            .find('>')
            .ok_or_else(|| format!("{name}: a <parameter= has no >"))?;
        let key = a[..c].trim().to_string();
        let v = &a[c + 1..];
        let e = v
            .find("</parameter>")
            .ok_or_else(|| format!("{name}: the parameter {key:?} has no </parameter>"))?;
        let mut value = &v[..e];
        value = value.strip_prefix('\n').unwrap_or(value);
        value = value.strip_suffix('\n').unwrap_or(value);
        params.push((key, value.to_string()));
        body = &v[e + "</parameter>".len()..];
    }
    Ok(Call { name, params })
}

/// The user turn carrying the results back, in the template's tool form
/// (each in its own `<tool_response>`), then what waited for it (`extra`:
/// lines from the system, messages) in a user turn of its own, and the
/// next assistant turn opened on its thinking.
pub fn responses_turn(results: &[String], extra: &[String]) -> String {
    let mut s = String::new();
    if !results.is_empty() {
        s.push_str("<|im_end|>\n<|im_start|>user");
        for r in results {
            s.push_str("\n<tool_response>\n");
            // Trimmed both ends, as the template trims a message.
            s.push_str(r.trim());
            s.push_str("\n</tool_response>");
        }
    }
    if !extra.is_empty() {
        s.push_str("<|im_end|>\n<|im_start|>user\n");
        s.push_str(&extra.join("\n"));
    }
    s.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    s
}

/// The user turn after a turn with no tool call: the time, what waited
/// for it, and the objective, so it goes on; the next assistant turn
/// opened on its thinking.
pub fn continue_turn(time: &str, objective: Option<&str>, extra: &[String]) -> String {
    let goal = match objective {
        Some(o) => format!("Your objective: {o}"),
        None => "You have no objective yet.".to_string(),
    };
    let waited = if extra.is_empty() {
        String::new()
    } else {
        format!("{}\n", extra.join("\n"))
    };
    format!(
        "<|im_end|>\n<|im_start|>user\n{waited}[{time}] {goal} Go on: reason, then act with a tool (what you have not checked with a tool, you do not know); if your objective is met or you wait on Claude, rest with wait.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
}

/// The summary asked for in a user turn of its own, and its answer opened
/// with no thinking and the summary's first words (`start`): asked inside
/// a turn, it called tools and the summary kept the template's marks.
pub fn summary_turn(ask: &str, start: &str) -> String {
    format!("<|im_end|>\n<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{start}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_parse_as_the_template_writes_them() {
        let text = "I will look first.\n\n<tool_call>\n<function=run>\n<parameter=command>\nls src\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write>\n<parameter=path>\ntools/a.c\n</parameter>\n<parameter=content>\nint main(void) {\n    return 0;\n}\n</parameter>\n</function>\n</tool_call>";
        let (calls, bad) = parse_calls(text);
        assert_eq!(bad, 0);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "run");
        assert_eq!(calls[0].param("command"), Some("ls src"));
        assert_eq!(calls[1].param("path"), Some("tools/a.c"));
        assert_eq!(
            calls[1].param("content"),
            Some("int main(void) {\n    return 0;\n}")
        );
    }

    #[test]
    fn broken_blocks_are_counted_not_run() {
        let (calls, bad) = parse_calls("<tool_call>\n<function=run>\n<parameter=command>\nls\n</function>\n</tool_call><tool_call>no function</tool_call><tool_call>\n<function=note>");
        assert!(calls.is_empty());
        assert_eq!(bad, 3);
        assert_eq!(parse_calls("no calls at all").0.len(), 0);
    }

    #[test]
    fn each_block_keeps_its_place_and_says_why_it_failed() {
        let b = parse_blocks("<tool_call>\n<function=note>\n<parameter=text>\na\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run>\n<parameter=command>\nls\n</function>\n</tool_call>\n<tool_call>\n<function=note>\n<parameter=text>\nb\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=note>");
        assert_eq!(b.len(), 4);
        assert_eq!(b[0].as_ref().unwrap().param("text"), Some("a"));
        assert_eq!(
            b[1].as_ref().unwrap_err(),
            "run: the parameter \"command\" has no </parameter>"
        );
        assert_eq!(b[2].as_ref().unwrap().param("text"), Some("b"));
        assert_eq!(b[3].as_ref().unwrap_err(), "the call has no </tool_call>");
    }

    #[test]
    fn calls_are_checked_against_their_declarations() {
        let call = |name: &str, params: &[(&str, &str)]| Call {
            name: name.into(),
            params: params
                .iter()
                .map(|(k, v)| ((*k).into(), (*v).into()))
                .collect(),
        };
        assert!(check(
            &call("read", &[("path", "a"), ("start", "3"), ("end", "9")]),
            false
        )
        .is_ok());
        assert!(check(&call("read", &[("path", "a"), ("end_path", "")]), false).is_ok());
        assert_eq!(
            check(&call("read", &[("path", "a"), ("start_line", "3")]), false).unwrap_err(),
            "read: it has no parameter \"start_line\"; read takes path (required), start, end"
        );
        assert_eq!(
            check(&call("read", &[("start", "x")]), false).unwrap_err(),
            "read: start must be a whole number, not \"x\"; it needs path; read takes path (required), start, end"
        );
        assert_eq!(
            check(&call("note", &[("text", "a"), ("text", "b")]), false).unwrap_err(),
            "note: text is given twice; note takes text (required)"
        );
        assert_eq!(
            check(&call("diff", &[("path", "a")]), true).unwrap_err(),
            "diff: it has no parameter \"path\"; diff takes no parameters"
        );
        // The loop's tools exist only with it.
        let e = check(&call("diff", &[]), false).unwrap_err();
        assert!(e.starts_with("there is no function \"diff\": the functions are run, read, edit,"));
        assert!(e.ends_with("wait and tell_claude"));
        assert!(names(true).ends_with("revert and report"));
    }

    #[test]
    fn the_tools_section_is_the_templates() {
        let s = tools_section(false);
        assert!(
            s.starts_with("# Tools\n\nYou have access to the following functions:\n\n<tools>\n{")
        );
        assert!(s.contains("{\"type\": \"function\", \"function\": {\"name\": \"run\", "));
        assert!(s.contains("\"name\": \"write\""));
        assert!(s.ends_with("</IMPORTANT>"));
        let r = responses_turn(&["a".into(), "b\n".into()], &[]);
        assert_eq!(
            r,
            "<|im_end|>\n<|im_start|>user\n<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert!(s.contains("\"name\": \"tell_claude\""));
    }

    #[test]
    fn what_waited_comes_in_a_user_turn_of_its_own() {
        let r = responses_turn(&["ran".into()], &["[01:02:03] Claude: hi".into()]);
        assert_eq!(
            r,
            "<|im_end|>\n<|im_start|>user\n<tool_response>\nran\n</tool_response><|im_end|>\n<|im_start|>user\n[01:02:03] Claude: hi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        let c = continue_turn("01:02:04", Some("x"), &["a line".into()]);
        assert!(c.starts_with("<|im_end|>\n<|im_start|>user\na line\n[01:02:04] Your objective: x"));
        let s = summary_turn("[t] write it", "What I was working on: ");
        assert!(
            s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\nWhat I was working on: ")
        );
    }
}
