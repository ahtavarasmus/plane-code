//! Cargo command runner exposed to the agent as the `run_cargo` tool.
//!
//! Wraps `cargo check`, `build`, `run`, `test`. Returns a structured envelope
//! the agent can read without parsing freeform text. `check` and `build`
//! reuse the JSON-diagnostic format already consumed by update.rs; `test`
//! parses libtest's structured output; `run` returns stdout/stderr/exit.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Command, Stdio};

/// Largest stdout/stderr slice we'll hand back through the tool
/// surface. Past this the model's attention starts pattern-matching
/// the noise instead of the user's question, which has produced
/// repetition-collapse on full `cargo check` NDJSON dumps (~200 KB).
const MAX_STREAM_BYTES: usize = 8 * 1024;
const MAX_STDERR_BYTES: usize = 4 * 1024;

/// Truncate `s` to at most `limit` bytes at a char boundary, appending
/// a marker that tells the model bytes were dropped. Head-truncation:
/// keeps the start, which is what matters for `cargo run` (program
/// output) and the per-test pass/fail lines in `cargo test`.
fn cap(s: String, limit: usize) -> String {
    if s.len() <= limit {
        return s;
    }
    let mut cutoff = limit;
    while cutoff > 0 && !s.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    let dropped = s.len() - cutoff;
    let mut out = s[..cutoff].to_string();
    out.push_str(&format!(
        "\n[truncated: {dropped} of {} bytes omitted]",
        s.len()
    ));
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCargoRequest {
    /// One of: check | build | run | test
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub stdin: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RunCargoResponse {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    /// Compile errors when command is check/build/run (with json-diagnostic).
    /// Empty when the build is clean. Always present so the agent has a
    /// uniform shape to inspect.
    pub compile_errors: Vec<serde_json::Value>,
    /// Test summary when command is test. Null otherwise.
    pub test_results: Option<serde_json::Value>,
    pub hints: Vec<String>,
}

pub fn run_cargo(workspace: &Path, req: &RunCargoRequest) -> Result<RunCargoResponse> {
    match req.command.as_str() {
        "check" => Ok(run_compile(workspace, "check", &req.args)),
        "build" => Ok(run_compile(workspace, "build", &req.args)),
        "run" => Ok(run_run(workspace, &req.args, req.stdin.as_deref())),
        "test" => Ok(run_test(workspace, &req.args)),
        other => Err(anyhow!(
            "unknown cargo command: {other} (expected check|build|run|test)"
        )),
    }
}

fn run_compile(workspace: &Path, sub: &str, extra_args: &[String]) -> RunCargoResponse {
    let mut cmd = Command::new("cargo");
    cmd.arg(sub)
        .arg("--message-format=json-diagnostic-rendered-ansi")
        .arg("--quiet")
        .current_dir(workspace);
    for a in extra_args {
        cmd.arg(a);
    }
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => return exec_fail(sub, e.to_string()),
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let compile_errors = parse_compiler_errors(&stdout);
    let mut hints = Vec::new();
    if !compile_errors.is_empty() {
        hints.push(format!(
            "{} compile error(s). Address them and retry, or fix in subsequent edits.",
            compile_errors.len()
        ));
    } else {
        hints.push(
            "Verification clean. Safe to finalize your response to the user.".into(),
        );
    }
    // Drop the raw cargo NDJSON stdout: with
    // --message-format=json-diagnostic-rendered-ansi it's one JSON line
    // per compiler-artifact / build-script-executed / fresh marker for
    // every dependency, easily 200 KB on a clean workspace. The agent
    // only needs `compile_errors` (already parsed above) and the
    // `hints` line. Passing the raw stream back has crashed sessions:
    // the model attends to the noise and falls into a repetition loop
    // emitting `{"reason":...}` fragments instead of replying.
    RunCargoResponse {
        command: sub.into(),
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::new(),
        stderr: cap(stderr, MAX_STDERR_BYTES),
        compile_errors,
        test_results: None,
        hints,
    }
}

fn run_run(workspace: &Path, extra_args: &[String], stdin: Option<&str>) -> RunCargoResponse {
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--quiet")
        .current_dir(workspace);
    if !extra_args.is_empty() {
        cmd.arg("--");
        for a in extra_args {
            cmd.arg(a);
        }
    }
    if stdin.is_some() {
        cmd.stdin(Stdio::piped());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return exec_fail("run", e.to_string()),
    };
    if let (Some(input), Some(child_stdin)) = (stdin, child.stdin.take()) {
        use std::io::Write;
        let mut s = child_stdin;
        let _ = s.write_all(input.as_bytes());
        // s drops here, closing stdin so the child sees EOF.
    }
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return exec_fail("run", e.to_string()),
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);
    let mut hints = Vec::new();
    if exit_code != 0 {
        hints.push(format!("Process exited with code {exit_code}."));
    }
    RunCargoResponse {
        command: "run".into(),
        exit_code,
        stdout: cap(stdout, MAX_STREAM_BYTES),
        stderr: cap(stderr, MAX_STDERR_BYTES),
        compile_errors: vec![],
        test_results: None,
        hints,
    }
}

fn run_test(workspace: &Path, extra_args: &[String]) -> RunCargoResponse {
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--no-fail-fast")
        .current_dir(workspace);
    for a in extra_args {
        cmd.arg(a);
    }
    // Note: we don't pass --format=terse / -Z unstable-options because
    // those require the nightly toolchain. parse_test_summary handles
    // libtest's default stable output (the `test foo ... ok` lines and
    // the trailing `test result: ok. N passed; M failed; ...` line).
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => return exec_fail("test", e.to_string()),
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let test_results = parse_test_summary(&stdout, &stderr);
    let exit_code = output.status.code().unwrap_or(-1);
    let mut hints = Vec::new();
    let mut tests_passed = false;
    if let Some(summary) = test_results.as_object() {
        let failed = summary
            .get("failed")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if failed > 0 {
            hints.push(format!(
                "{failed} test(s) failed. Inspect `failures` for the names; \
                 query the failing test by name via query_codebase to see body."
            ));
        } else {
            tests_passed = true;
        }
    }
    if tests_passed {
        hints.push(
            "Verification clean. Safe to finalize your response to the user.".into(),
        );
    }
    // test_results captures the structured pass/fail summary, so the
    // raw stdout is only useful for inspecting a specific failure's
    // panic text. Cap it so a large project's per-test "ok" lines
    // (thousands of them) can't dominate the response.
    RunCargoResponse {
        command: "test".into(),
        exit_code,
        stdout: cap(stdout, MAX_STREAM_BYTES),
        stderr: cap(stderr, MAX_STDERR_BYTES),
        compile_errors: vec![],
        test_results: Some(test_results),
        hints,
    }
}

fn parse_compiler_errors(stdout: &str) -> Vec<serde_json::Value> {
    let mut errors = Vec::new();
    for line in stdout.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let msg = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        let level = msg.get("level").and_then(|s| s.as_str()).unwrap_or("");
        if level != "error" && level != "error: internal compiler error" {
            continue;
        }
        let text = msg.get("message").and_then(|s| s.as_str()).unwrap_or("");
        let code = msg
            .get("code")
            .and_then(|c| c.get("code"))
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let (file, line_no, col) = msg
            .get("spans")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .map(|sp| {
                (
                    sp.get("file_name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    sp.get("line_start").and_then(|v| v.as_u64()).unwrap_or(0),
                    sp.get("column_start").and_then(|v| v.as_u64()).unwrap_or(0),
                )
            })
            .unwrap_or_default();
        errors.push(serde_json::json!({
            "file": file,
            "line": line_no,
            "column": col,
            "message": text,
            "code": code,
        }));
    }
    errors
}

fn parse_test_summary(stdout: &str, stderr: &str) -> serde_json::Value {
    // libtest's terse output prints lines like `test foo::bar ... ok`
    // and a final `test result: ok. N passed; M failed; K ignored; ...`
    let mut passed = 0u64;
    let mut failed_count = 0u64;
    let mut ignored = 0u64;
    let mut failures: Vec<serde_json::Value> = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("test result: ") {
            // Parse `ok. N passed; M failed; K ignored; ...`
            for part in rest.split(';') {
                let part = part.trim().trim_end_matches('.');
                let mut it = part.split_whitespace();
                let n = it.next().and_then(|s| s.parse::<u64>().ok());
                let label = it.next();
                if let (Some(n), Some(label)) = (n, label) {
                    match label {
                        "passed" => passed = n,
                        "failed" => failed_count = n,
                        "ignored" => ignored = n,
                        _ => {}
                    }
                }
            }
        }
        if let Some(rest) = l.strip_prefix("test ") {
            if let Some(name) = rest.strip_suffix(" ... FAILED") {
                failures.push(serde_json::json!({ "test": name.trim() }));
            }
        }
    }
    serde_json::json!({
        "passed": passed,
        "failed": failed_count,
        "ignored": ignored,
        "failures": failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_passes_through_short_strings() {
        assert_eq!(cap("hi".into(), 10), "hi");
        assert_eq!(cap("".into(), 10), "");
    }

    #[test]
    fn cap_truncates_long_strings_with_marker() {
        let s = "x".repeat(20);
        let out = cap(s, 8);
        assert!(out.starts_with("xxxxxxxx"));
        assert!(out.contains("truncated"));
        assert!(out.contains("12 of 20"));
    }

    #[test]
    fn cap_respects_utf8_char_boundary() {
        // "héllo": h=1, é=2 bytes (0xc3 0xa9), then "llo" 3 bytes. Total 6.
        // Capping at 2 would split é; cap() should pull back to 1.
        let s = "héllo".to_string();
        let out = cap(s, 2);
        assert!(out.starts_with("h"));
        assert!(!out.starts_with("h\u{00e9}")); // é didn't fit
        assert!(out.contains("truncated"));
    }
}

fn exec_fail(command: &str, msg: String) -> RunCargoResponse {
    RunCargoResponse {
        command: command.into(),
        exit_code: -1,
        stdout: String::new(),
        stderr: format!("failed to invoke cargo: {msg}"),
        compile_errors: vec![],
        test_results: None,
        hints: vec!["Cargo invocation failed; check that cargo is installed and the workspace path is correct.".into()],
    }
}
