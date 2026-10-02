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

/// One tool as the template's `tojson` writes it: keys in their order,
/// `", "` and `": "` between them (Python's json.dumps).
fn tool(name: &str, desc: &str, props: &[(&str, &str, &str)], required: &[&str]) -> String {
    let props: Vec<String> = props
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

/// The tools; `propose` only with the self-improvement loop (`improve.md`).
fn tools(improve: bool) -> Vec<String> {
    let mut list = vec![
        tool(
            "run",
            "Run a shell command in your sandbox and get its exit code and output (at most about 4096 tokens: past that its leading lines and how much was cut, so narrow a big output with head, tail or grep). It already starts in the repository, so paths are relative to it and no cd is needed; what it writes there lands in your working copy (the repository itself never changes). /tmp is kept between commands. No network, 60 s at most: a command stopped at the limit has no result. A Rust build (cargo) does not fit in that time here; Claude builds and tests the Rust code. Use it to build C with tcc, test, search (grep -n) and list; to read a file use read, to change one use edit.",
            &[("command", "string", "The command line, run by sh -c.")],
            &["command"],
        ),
        tool(
            "read",
            "Read a file of the repository (your working copy) or your workspace, whole or by lines, or a directory's listing. One read gives at most about 4096 tokens (about 400 lines of this code): past that it gives the leading lines and says where the rest begins, so read the function you need with start and end (find it with run: grep -n).",
            &[
                ("path", "string", "A path relative to the repository, or absolute."),
                ("start", "integer", "The first line, from 1."),
                ("end", "integer", "The last line."),
            ],
            &["path"],
        ),
        tool(
            "edit",
            "Change a file of your working copy or workspace in place: the text old, which must occur exactly once in it, is replaced by new. Include enough lines around a change for old to be unique. Use it for every change to an existing file; write only creates new files or replaces one entirely.",
            &[
                ("path", "string", "A path relative to the repository, or in your workspace."),
                ("old", "string", "The exact text to replace, as it is in the file."),
                ("new", "string", "The text to put in its place."),
            ],
            &["path", "old", "new"],
        ),
        tool(
            "write",
            "Write a whole new file into your working copy of the repository (a path relative to it) or your workspace, or replace one entirely. To change part of an existing file, use edit.",
            &[
                ("path", "string", "A path relative to the repository, or in your workspace."),
                ("content", "string", "The file's whole content."),
            ],
            &["path", "content"],
        ),
        tool(
            "note",
            "Keep a line in your own memory across time: it is shown to you again whenever your memory is refreshed, so keep each thing once. It is not a message: to tell Claude something, use tell_claude.",
            &[("text", "string", "The note.")],
            &["text"],
        ),
        tool(
            "wait",
            "Rest until something new comes: a message from Claude, a new commit in the repository, a new objective, or the time you give. Call it when your objective is met, or when you wait on Claude, rather than going on for its own sake: nothing is asked of you while you rest, and your next turn opens with what came.",
            &[
                ("reason", "string", "Why you rest: what is done, or what you wait for."),
                ("minutes", "integer", "The longest rest, in minutes (default 15, at most 60)."),
            ],
            &["reason"],
        ),
        tool(
            "tell_claude",
            "Send a message to Claude, who develops this program with you and reads every message at once: a proposal (the file, the function, the change and why, and what you checked), a finding, a question, or your answer to a message of Claude's. Claude answers in a later turn.",
            &[
                ("text", "string", "The message."),
                ("re", "string", "The id of Claude's message this answers (as c3), if it answers one."),
            ],
            &["text"],
        ),
    ];
    if improve {
        list.push(tool(
            "propose",
            "Put the change in your working copy forward as one improvement to yourself (this program, the one you run in). It is staged as a diff against the repository's current commit and refused if it touches the build, the scripts or this loop with its evaluators, or if a file it changes was changed in the repository after your copy was made; then it is built and tested in a sandbox (make check: format, clippy, release build, tests; it takes minutes). The outcome comes at a later turn and is kept in improve.log in your workspace with every earlier one: read it to choose what to try next, and fix a failed build from its errors. A change that passes goes to Claude for review, then is measured on the running model before it is kept. One candidate at a time; make one change, small and whole, per proposal.",
            &[
                ("title", "string", "One line: what the change does."),
                ("why", "string", "What it should improve in you, and how that would show (a measure, a behaviour, a test)."),
            ],
            &["title", "why"],
        ));
    }
    list
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

/// The tool calls in a turn's text (after its thinking): every
/// `<tool_call>` block with a `<function=NAME>` and its `<parameter=K>`
/// values (a value's one leading and one trailing newline dropped, as the
/// template writes them). A block that does not parse is skipped and
/// counted.
pub fn parse_calls(text: &str) -> (Vec<Call>, usize) {
    let mut calls = Vec::new();
    let mut bad = 0;
    let mut rest = text;
    while let Some(i) = rest.find("<tool_call>") {
        let after = &rest[i + "<tool_call>".len()..];
        let Some(j) = after.find("</tool_call>") else {
            bad += 1;
            break;
        };
        let block = &after[..j];
        rest = &after[j + "</tool_call>".len()..];
        match parse_block(block) {
            Some(c) => calls.push(c),
            None => bad += 1,
        }
    }
    (calls, bad)
}

fn parse_block(block: &str) -> Option<Call> {
    let f = block.find("<function=")?;
    let after = &block[f + "<function=".len()..];
    let close = after.find('>')?;
    let name = after[..close].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut body = &after[close + 1..];
    let mut params = Vec::new();
    while let Some(p) = body.find("<parameter=") {
        let a = &body[p + "<parameter=".len()..];
        let c = a.find('>')?;
        let key = a[..c].trim().to_string();
        let v = &a[c + 1..];
        let e = v.find("</parameter>")?;
        let mut value = &v[..e];
        value = value.strip_prefix('\n').unwrap_or(value);
        value = value.strip_suffix('\n').unwrap_or(value);
        params.push((key, value.to_string()));
        body = &v[e + "</parameter>".len()..];
    }
    Some(Call { name, params })
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
            s.push_str(r.trim_end());
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
