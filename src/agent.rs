//! Agent loop. Drives the LLM with the three ontology tools exposed,
//! dispatches tool calls, and feeds results back until the model emits a
//! plain-text reply (or max_turns is hit).
//!
//! Two entry points:
//!   - `run_once(prompt)`        single-shot: append user msg, drive loop, return.
//!   - `run_turn(prompt)`        REPL turn: same as run_once but designed to be
//!                               called repeatedly with accumulating context.
//!
//! Verbose display is delegated to `display::Display`. The agent itself
//! holds no formatting concerns beyond passing events through.

use crate::cargo_ops::{run_cargo, RunCargoRequest};
use crate::display::Display;
use crate::llm::{BackendConfig, LlmBackend};
use crate::ollama::{ChatMessage, ToolCall};
use crate::ontology::{Ontology, QueryRequest, UpdateRequest};
use crate::read_set::ReadSet;
use crate::sessions;
use crate::tools;
use anyhow::Result;
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct Agent {
    pub workspace: PathBuf,
    pub ontology: Ontology,
    pub llm: LlmBackend,
    /// Static-at-startup config kept around so `/model` can rebuild the
    /// backend in place when the user switches between Ollama and Groq.
    pub backend_config: BackendConfig,
    pub max_turns: usize,
    pub messages: Vec<ChatMessage>,
    pub display: Display,
    /// Stable id used to persist this conversation under
    /// `~/.plane-code/sessions/<workspace-hash>/<id>.json`. Regenerated
    /// on `/clear`; replaced on `/resume` to load into an existing file.
    pub session_id: String,
    /// Set by the UI when the operator presses Esc / Ctrl-C while the
    /// agent is streaming. The chat_stream select polls this flag and
    /// cancels the in-flight request when it flips. Also doubles as a
    /// "stop early" signal for any future long-running tool dispatch.
    pub interrupt: Arc<AtomicBool>,
    /// Coarse busy flag: true while a `run_turn` is in flight. The TUI
    /// reads this to decide whether to draw the input box as editable
    /// or as a "(busy)" placeholder.
    pub busy: Arc<AtomicBool>,
    /// Read-before-edit guardrail. Tracks which entities (Function/Type/
    /// Trait keys) and which files the agent has actually read in this
    /// session. update_ontology consults it before dispatching - edits
    /// against unread targets are bounced with a hint pointing at the
    /// query call to make first.
    pub read_set: ReadSet,
}

impl Agent {
    pub fn new(
        workspace: PathBuf,
        ontology: Ontology,
        llm: LlmBackend,
        backend_config: BackendConfig,
        max_turns: usize,
        display: Display,
    ) -> Self {
        let crate_name = ontology.crate_name.clone();
        let messages = vec![ChatMessage {
            role: "system".into(),
            content: tools::system_prompt(&crate_name),
            thinking: None,
            tool_calls: vec![],
            tool_name: None,
            tool_call_id: None,
        }];
        Self {
            workspace,
            ontology,
            llm,
            backend_config,
            max_turns,
            messages,
            display,
            session_id: sessions::new_session_id(),
            interrupt: Arc::new(AtomicBool::new(false)),
            busy: Arc::new(AtomicBool::new(false)),
            read_set: ReadSet::new(),
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed)
    }

    pub fn set_busy(&self, v: bool) {
        self.busy.store(v, Ordering::Relaxed);
    }

    /// Reset the conversation while preserving the system prompt and the
    /// underlying ontology. Used by the REPL's /clear command. We also
    /// regenerate the session id so the cleared conversation gets its
    /// own slot in /resume rather than overwriting the prior one.
    pub fn clear_history(&mut self) {
        let system = self.messages.first().cloned();
        self.messages.clear();
        if let Some(s) = system {
            self.messages.push(s);
        }
        self.session_id = sessions::new_session_id();
        self.read_set.clear();
    }

    /// Replace the current conversation with the contents of a saved
    /// session. The system prompt from the loaded session is honored
    /// verbatim (so any changes to `tools::system_prompt` since the
    /// session was first created are NOT applied retroactively - the
    /// model sees the same context it had before).
    pub fn resume_session(&mut self, session_id: String, messages: Vec<ChatMessage>) {
        self.session_id = session_id;
        self.read_set.rebuild_from_history(&messages);
        self.messages = messages;
    }

    /// Best-effort persist of the current conversation to disk.
    /// Errors are surfaced as a stderr warning and otherwise swallowed
    /// - a failed save shouldn't take down the REPL.
    fn persist_session(&self) {
        if let Err(e) = sessions::save(
            &self.workspace,
            &self.session_id,
            self.llm.provider(),
            self.llm.model(),
            &self.ontology.crate_name,
            &self.messages,
        ) {
            self.display
                .show_error(&format!("session save: {e}"));
        }
    }

    /// Re-run the indexer and refresh the ontology in place. The system
    /// prompt isn't rebuilt; the dynamic tool definitions get rebuilt
    /// every turn anyway.
    pub fn reindex(&mut self) -> Result<(usize, usize, usize, usize, usize)> {
        let ws = self.workspace.clone();
        self.ontology = Ontology::index(&ws)?;
        Ok((
            self.ontology.functions.len(),
            self.ontology.types.len(),
            self.ontology.traits.len(),
            self.ontology.modules.len(),
            self.ontology.files.len(),
        ))
    }

    /// Single-shot: print the user input, run the loop, return.
    pub async fn run_once(&mut self, prompt: &str) -> Result<()> {
        self.run_turn(prompt).await
    }

    /// Append a user message and drive the agent loop until a final
    /// assistant text appears or `max_turns` is reached. Safe to call
    /// repeatedly; context accumulates across calls.
    ///
    /// Persists the session to disk on the way out regardless of
    /// outcome - even on error or interrupt the user message and any
    /// partial state should be recoverable via /resume.
    pub async fn run_turn(&mut self, prompt: &str) -> Result<()> {
        let result = self.run_turn_inner(prompt).await;
        self.persist_session();
        result
    }

    async fn run_turn_inner(&mut self, prompt: &str) -> Result<()> {
        // Note: the REPL's rustyline prompt already echoes the user's
        // typed input, and single-shot callers passed it as a CLI arg.
        // We don't re-render it here.

        self.messages.push(ChatMessage {
            role: "user".into(),
            content: prompt.into(),
            thinking: None,
            tool_calls: vec![],
            tool_name: None,
            tool_call_id: None,
        });

        // Verification gate: tracks whether the most recent update_ontology
        // has been followed by a run_cargo (any command) yet. The model
        // can pile up multiple edits in a row, query freely, etc., but it
        // can't emit a final text response while updates are unverified.
        // We nudge once with a synthetic user-role message; if the model
        // still bypasses, we accept the response with a visible warning
        // rather than loop forever.
        let mut pending_verification = false;
        let mut already_nudged = false;
        // Empty-response retry: providers occasionally return a stream
        // that finishes with done:true but zero content, zero thinking,
        // and zero tool calls (Ollama session quirk, KV-cache eviction,
        // truncated stream, etc.). One retry with a synthetic nudge is
        // usually enough to wake it back up. After the retry we accept
        // whatever comes back so we don't loop forever.
        let mut already_retried_empty = false;

        for turn in 0..self.max_turns {
            self.display.show_turn_start(turn, self.max_turns);

            // Tool defs regenerated every turn so dynamic enums (file paths,
            // module paths, traits, languages) reflect the live ontology.
            let tool_defs = tools::tool_definitions(&self.ontology);

            // Stream the response so the operator sees thinking tokens
            // arrive in real time. The printer is held in a RefCell so the
            // streaming closure can borrow it across await points without
            // tripping the borrow checker.
            //
            // The TUI thread owns the keyboard; when the operator hits
            // Esc / Ctrl-C, it flips `self.interrupt`. We poll that flag
            // alongside the stream future. No raw-mode dance here -
            // the TUI handles all terminal state.
            let printer = RefCell::new(self.display.stream_printer());
            let started = std::time::Instant::now();
            let interrupt = self.interrupt.clone();

            enum Outcome {
                Done(Result<crate::ollama::ChatResponse>),
                Esc,
            }
            let outcome = tokio::select! {
                biased;
                res = self.llm.chat_stream(&self.messages, &tool_defs, |kind, delta| {
                    printer.borrow_mut().feed(kind, delta);
                }) => {
                    Outcome::Done(res)
                }
                _ = wait_for_interrupt(interrupt.clone()) => {
                    Outcome::Esc
                }
            };
            let wall = started.elapsed();
            printer.borrow_mut().finish();
            drop(printer);

            let resp = match outcome {
                Outcome::Done(Ok(r)) => r,
                Outcome::Done(Err(e)) => {
                    self.display
                        .show_error(&format!("llm chat_stream: {e}"));
                    return Err(e);
                }
                Outcome::Esc => {
                    // Push a placeholder assistant message so the
                    // conversation alternation stays valid for providers
                    // that care (Groq strict-mode, etc.). The user can
                    // type a new prompt to continue.
                    self.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: "[interrupted by user]".into(),
                        thinking: None,
                        tool_calls: vec![],
                        tool_name: None,
                        tool_call_id: None,
                    });
                    self.display.show_error("interrupted (Esc)");
                    return Ok(());
                }
            };
            let assistant = resp.message.clone();

            self.display.show_response_summary(&resp, wall);

            if !assistant.tool_calls.is_empty() {
                self.display.show_tool_dispatch_note(&assistant.tool_calls);
                tracing::info!(turn, calls = assistant.tool_calls.len(), "tool calls");
                let calls = assistant.tool_calls.clone();
                self.messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: assistant.content.clone(),
                    thinking: assistant.thinking.clone(),
                    tool_calls: calls.clone(),
                    tool_name: None,
                    tool_call_id: None,
                });
                for call in &calls {
                    self.display.show_tool_call(call);
                    let result = self.dispatch_tool(call);
                    // Update the verification flag based on which tool
                    // ran and (for update_ontology) whether it actually
                    // succeeded. A rolled-back update doesn't count as
                    // a pending edit.
                    match call.function.name.as_str() {
                        "update_ontology" => {
                            let succeeded = result
                                .get("success")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if succeeded {
                                pending_verification = true;
                            }
                        }
                        "run_cargo" => {
                            pending_verification = false;
                        }
                        _ => {}
                    }
                    self.display.show_tool_result(&call.function.name, &result);
                    let content = match serde_json::to_string(&result) {
                        Ok(s) => s,
                        Err(e) => format!(r#"{{"error":"serialize: {e}"}}"#),
                    };
                    self.messages.push(ChatMessage {
                        role: "tool".into(),
                        content,
                        thinking: None,
                        tool_calls: vec![],
                        tool_name: Some(call.function.name.clone()),
                        tool_call_id: call.id.clone(),
                    });
                }
                continue;
            }

            // No tool calls: model wants to end the turn.
            //
            // Verification gate: if updates were made and not yet
            // verified, append the assistant message + a synthetic
            // user-role nudge, and continue the loop. The model gets one
            // chance to call run_cargo. If it bypasses anyway, we accept
            // the response but warn the operator visibly.
            if pending_verification && !already_nudged {
                already_nudged = true;
                self.messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: assistant.content.clone(),
                    thinking: assistant.thinking.clone(),
                    tool_calls: vec![],
                    tool_name: None,
                    tool_call_id: None,
                });
                self.messages.push(ChatMessage {
                    role: "user".into(),
                    content: "Hold on - your last edit hasn't been verified. \
                              Before I can hand the response back to the \
                              user, call run_cargo with command=test (or \
                              command=check for a quick compile gate). \
                              This is enforced by the harness; one \
                              verification covers all preceding edits."
                        .into(),
                    thinking: None,
                    tool_calls: vec![],
                    tool_name: None,
                    tool_call_id: None,
                });
                continue;
            }

            // Empty-response gate: if the model returned nothing
            // user-visible AND no tool calls, retry once with a hint
            // before giving up. We push the (empty) assistant message
            // first to keep roles alternating, then a synthetic user
            // nudge tailored to whether thinking was emitted.
            let has_content = !assistant.content.trim().is_empty();
            let has_thinking = assistant
                .thinking
                .as_ref()
                .map(|t| !t.trim().is_empty())
                .unwrap_or(false);
            if !has_content && !already_retried_empty {
                already_retried_empty = true;
                self.messages.push(ChatMessage {
                    role: "assistant".into(),
                    content: assistant.content.clone(),
                    thinking: assistant.thinking.clone(),
                    tool_calls: vec![],
                    tool_name: None,
                    tool_call_id: None,
                });
                let hint = if has_thinking {
                    "Your last response had thinking but no user-visible \
                     content and no tool calls. Either reply with text for \
                     the user, or call a tool to make progress - don't end \
                     the turn silently."
                } else {
                    "Your last response came back completely empty (no \
                     content, no thinking, no tool calls). The stream may \
                     have dropped or the context got confused. Try again: \
                     either reply with text for the user, or call a tool."
                };
                self.messages.push(ChatMessage {
                    role: "user".into(),
                    content: hint.into(),
                    thinking: None,
                    tool_calls: vec![],
                    tool_name: None,
                    tool_call_id: None,
                });
                self.display
                    .show_error("empty response; retrying once with a nudge");
                continue;
            }

            // Append the final assistant message and end the turn. If
            // the visible content is empty, dump the raw response so the
            // operator can see exactly what came back.
            self.messages.push(ChatMessage {
                role: "assistant".into(),
                content: assistant.content.clone(),
                thinking: assistant.thinking.clone(),
                tool_calls: vec![],
                tool_name: None,
                tool_call_id: None,
            });
            if pending_verification {
                self.display.show_error(
                    "agent finalized with unverified edits (the model \
                     bypassed the run_cargo gate even after a nudge). \
                     Consider running cargo test manually to confirm.",
                );
            }
            if assistant.content.trim().is_empty() {
                let note = if has_thinking {
                    "model emitted thinking but no user-visible content; \
                     it likely needed to call a tool but didn't"
                } else {
                    "model returned no content, no thinking, and no tool calls"
                };
                self.display.show_raw_response(&resp, note);
            }
            return Ok(());
        }

        self.display.show_max_turns(self.max_turns);
        Ok(())
    }

    fn dispatch_tool(&mut self, call: &ToolCall) -> serde_json::Value {
        match call.function.name.as_str() {
            "query_ontology" => {
                let req: QueryRequest = match serde_json::from_value(call.function.arguments.clone()) {
                    Ok(r) => r,
                    Err(e) => {
                        return serde_json::json!({
                            "error": format!("invalid query_ontology arguments: {e}")
                        });
                    }
                };
                match self.ontology.query(&req) {
                    Ok(resp) => {
                        let val = serde_json::to_value(resp).unwrap_or(serde_json::Value::Null);
                        self.read_set.record_query(&val);
                        val
                    }
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            }
            "update_ontology" => {
                let req: UpdateRequest = match serde_json::from_value(call.function.arguments.clone()) {
                    Ok(r) => r,
                    Err(e) => {
                        return serde_json::json!({
                            "error": format!("invalid update_ontology arguments: {e}")
                        });
                    }
                };
                // Read-before-edit guardrail. If the model is trying to
                // mutate an entity or file it hasn't actually read this
                // session, refuse the call and point it at the right
                // query_ontology call. Without this, small models tend
                // to edit code from imagination - inventing APIs and
                // symbols that don't exist.
                if let Some(reason) = self.read_set.check_update(&req.operation, &req.target) {
                    return serde_json::json!({
                        "success": false,
                        "rollback_reason": "unread_target",
                        "files_changed": [],
                        "compile_status": null,
                        "graph_diff": null,
                        "details": reason.clone(),
                        "hints": [reason],
                    });
                }
                match self.ontology.update(&req) {
                    Ok(resp) => {
                        let val = serde_json::to_value(resp).unwrap_or(serde_json::Value::Null);
                        self.read_set.record_update(&req.operation, &req.target, &val);
                        val
                    }
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            }
            "run_cargo" => {
                let req: RunCargoRequest = match serde_json::from_value(call.function.arguments.clone()) {
                    Ok(r) => r,
                    Err(e) => {
                        return serde_json::json!({
                            "error": format!("invalid run_cargo arguments: {e}")
                        });
                    }
                };
                match run_cargo(&self.workspace, &req) {
                    Ok(resp) => serde_json::to_value(resp).unwrap_or(serde_json::Value::Null),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            }
            "show_flow" => {
                let target = call.function.arguments
                    .get("target")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if target.is_empty() {
                    return serde_json::json!({
                        "error": "show_flow requires a non-empty `target` (function name, path, or `skyline`)"
                    });
                }
                let result = if target == "skyline" {
                    crate::flow_cli::open_skyline(&self.ontology)
                } else {
                    crate::flow_cli::open_flow(&self.ontology, &target)
                };
                match result {
                    Ok(path) => serde_json::json!({
                        "shown": target,
                        "path": path.display().to_string(),
                        "note": "Diagram opened in operator's browser. Continue with \
                                 your final response - the operator will interrupt if \
                                 the visual review reveals a problem."
                    }),
                    Err(e) => serde_json::json!({ "error": e.to_string() }),
                }
            }
            other => serde_json::json!({
                "error": serde_json::Value::String(format!("unknown tool: {other}"))
            }),
        }
    }
}

/// Async helper: poll the shared interrupt flag every 50ms, return
/// when it flips. Used inside the chat_stream select so the TUI's
/// Esc handler can cancel an in-flight LLM request.
async fn wait_for_interrupt(flag: Arc<AtomicBool>) {
    loop {
        if flag.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

