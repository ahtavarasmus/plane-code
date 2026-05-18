//! Read-before-edit guardrail.
//!
//! Tracks which structural entities (Function/Type/Trait keys) and which
//! files the agent has actually seen the contents of in this session.
//! `update_codebase` consults this set before dispatching: if the model
//! tries to edit something it never read, the harness rolls the call back
//! with a hint pointing at the exact `query_codebase` call to make.
//!
//! The motivation is the recurring failure mode where a small model picks
//! a target out of a module listing (signatures only) and writes a "fix"
//! by imagining the body. Forcing a read first plants the real source in
//! the model's context and dramatically reduces hallucinated APIs.
//!
//! What counts as a read:
//!   - `query_codebase object_type=Function|Type|Trait` results - they
//!     return body / source verbatim.
//!   - `query_codebase object_type=File` results - return the file
//!     outline plus gap region content.
//!   - `query_codebase object_type=Module` does NOT count - it shows
//!     signatures only, no bodies.
//!   - `include_links` neighbors do NOT count - they show summaries only.
//!
//! After a successful update_codebase, the edit's target is freshened in
//! the set: the response carried the diff, so the model has up-to-date
//! information and shouldn't be forced to re-query before its next edit.

use crate::ollama::ChatMessage;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
pub struct ReadSet {
    /// Fully-qualified entity keys: `module_path::name`.
    pub entities: HashSet<String>,
    /// Workspace-relative file paths, exactly as the agent addresses them.
    pub files: HashSet<String>,
}

impl ReadSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.entities.clear();
        self.files.clear();
    }

    /// Walk a successful query_codebase result and record any bodies it
    /// exposed. Idempotent; safe to call on partial results.
    pub fn record_query(&mut self, result: &serde_json::Value) {
        let object_type = result
            .get("object_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let results = match result.get("results").and_then(|v| v.as_array()) {
            Some(r) => r,
            None => return,
        };
        for item in results {
            match object_type {
                "Function" | "Type" | "Trait" => {
                    if let (Some(name), Some(mp)) = (
                        item.get("name").and_then(|v| v.as_str()),
                        item.get("module_path").and_then(|v| v.as_str()),
                    ) {
                        self.entities.insert(format!("{mp}::{name}"));
                    }
                }
                "File" => {
                    if let Some(p) = item.get("path").and_then(|v| v.as_str()) {
                        self.files.insert(p.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    /// Freshen the set after a successful update_codebase. The target's
    /// new state is implicitly visible to the model via the diff in the
    /// response, so the next edit on this entity doesn't need a re-query.
    pub fn record_update(
        &mut self,
        op: &str,
        target: &serde_json::Value,
        result: &serde_json::Value,
    ) {
        let success = result
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !success {
            return;
        }
        match op {
            "replace_body" | "replace_item" | "add_function" => {
                if let Some(key) = entity_key(target) {
                    self.entities.insert(key);
                }
            }
            "rename" => {
                if let Some(old) = entity_key(target) {
                    self.entities.remove(&old);
                }
                if let (Some(mp), Some(new_name)) = (
                    target.get("module_path").and_then(|v| v.as_str()),
                    result
                        .get("graph_diff")
                        .and_then(|g| g.get("renamed"))
                        .and_then(|r| r.get("to"))
                        .and_then(|v| v.as_str()),
                ) {
                    self.entities.insert(format!("{mp}::{new_name}"));
                }
            }
            "edit_file" | "create_file" => {
                if let Some(p) = target.get("path").and_then(|v| v.as_str()) {
                    self.files.insert(p.to_string());
                }
            }
            "delete_file" => {
                if let Some(p) = target.get("path").and_then(|v| v.as_str()) {
                    self.files.remove(p);
                }
            }
            _ => {}
        }
    }

    /// Decide whether to block an update. `Some(reason)` means refuse,
    /// `None` means proceed. The reason carries a hint with the exact
    /// query_codebase call the model should make first.
    pub fn check_update(&self, op: &str, target: &serde_json::Value) -> Option<String> {
        match op {
            "replace_body" => {
                let key = entity_key(target)?;
                if self.entities.contains(&key) {
                    return None;
                }
                let (mp, name) = split_key(&key);
                Some(unread_entity_message("Function", mp, name, &key))
            }
            "replace_item" | "rename" => {
                let key = entity_key(target)?;
                if self.entities.contains(&key) {
                    return None;
                }
                let object_type = target
                    .get("object_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Function");
                let (mp, name) = split_key(&key);
                Some(unread_entity_message(object_type, mp, name, &key))
            }
            "edit_file" | "delete_file" => {
                let path = target.get("path").and_then(|v| v.as_str())?;
                if self.files.contains(path) {
                    return None;
                }
                Some(unread_file_message(path))
            }
            // create_file and add_function don't touch existing content.
            _ => None,
        }
    }

    /// Replay session history once on resume so the agent doesn't have
    /// to re-read entities it had already seen before reload. Walks
    /// assistant tool_calls and pairs them with the matching tool
    /// result by tool_call_id.
    pub fn rebuild_from_history(&mut self, messages: &[ChatMessage]) {
        self.clear();

        let mut call_args: HashMap<String, (String, serde_json::Value)> = HashMap::new();
        for msg in messages {
            if msg.role == "assistant" {
                for call in &msg.tool_calls {
                    if let Some(id) = &call.id {
                        call_args.insert(
                            id.clone(),
                            (call.function.name.clone(), call.function.arguments.clone()),
                        );
                    }
                }
                continue;
            }
            if msg.role != "tool" {
                continue;
            }
            let result: serde_json::Value = match serde_json::from_str(&msg.content) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let tool_name = msg.tool_name.as_deref().unwrap_or("");
            match tool_name {
                "query_codebase" => self.record_query(&result),
                "update_codebase" => {
                    if let Some(id) = &msg.tool_call_id {
                        if let Some((_, args)) = call_args.get(id) {
                            let op = args
                                .get("operation")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let target = args
                                .get("target")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);
                            self.record_update(op, &target, &result);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

fn entity_key(target: &serde_json::Value) -> Option<String> {
    let mp = target.get("module_path").and_then(|v| v.as_str())?;
    let name = target.get("name").and_then(|v| v.as_str())?;
    if mp.is_empty() || name.is_empty() {
        return None;
    }
    Some(format!("{mp}::{name}"))
}

fn split_key(key: &str) -> (&str, &str) {
    match key.rfind("::") {
        Some(i) => (&key[..i], &key[i + 2..]),
        None => ("", key),
    }
}

fn unread_entity_message(object_type: &str, mp: &str, name: &str, key: &str) -> String {
    format!(
        "Edit refused: you haven't read `{key}` in this session. The harness blocks \
         updates to entities you haven't actually seen so the model doesn't edit \
         from imagination. Bodies and definitions can change between edits, so \
         pasted snippets and recall don't count - only the body returned by a \
         live query_codebase call does. Make this call first, then retry the \
         edit:\n\
         \n\
         query_codebase {{\"object_type\": \"{object_type}\", \"filters\": \
         {{\"name\": \"{name}\", \"module_path\": \"{mp}\"}}}}\n\
         \n\
         The result includes the body / source - review it carefully before \
         writing your replacement."
    )
}

fn unread_file_message(path: &str) -> String {
    format!(
        "Edit refused: you haven't read `{path}` in this session. The file's \
         current contents are not in your context, so any edit would be blind. \
         Make this call first, then retry the edit:\n\
         \n\
         query_codebase {{\"object_type\": \"File\", \"filters\": \
         {{\"path\": \"{path}\"}}}}\n\
         \n\
         The result includes the file outline plus gap-region text - review \
         it before writing your edit."
    )
}
