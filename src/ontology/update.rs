//! Structural edits over the ontology. Implements the `update_ontology` tool.
//!
//! Compile-check policy (mirrors SPEC.md):
//!   - Parse failures roll back atomically (disk untouched).
//!   - Type/borrow errors are returned in `compile_status` but do NOT
//!     roll back. Multi-step changes can flow through broken intermediates.
//!
//! Operations: replace_body, add_function, rename.
//! `rename` is intentionally naive in v0: word-boundary text replacement
//! across all .rs files. It validates each modified file by re-parsing.

use crate::ontology::indexer;
use crate::ontology::model::*;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use similar::TextDiff;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRequest {
    pub operation: String,
    pub target: serde_json::Value,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
pub struct FileChange {
    pub path: String,
    pub diff: String,
}

#[derive(Debug, Serialize)]
pub struct UpdateResponse {
    pub success: bool,
    pub rollback_reason: Option<String>,
    pub files_changed: Vec<FileChange>,
    pub compile_status: serde_json::Value,
    pub graph_diff: Option<serde_json::Value>,
    pub details: Option<String>,
    /// Just-in-time guidance for the agent based on outcome (rollback
    /// reasons, compile errors, etc). Empty for the clean-success path.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hints: Vec<String>,
}

impl UpdateResponse {
    fn rollback(reason: &str, details: impl Into<String>) -> Self {
        let details: String = details.into();
        let hint = match reason {
            "parse_error" => Some(
                "Edit rolled back: result did not parse as Rust. Disk and \
                 graph are unchanged. Adjust the payload (likely a missing \
                 brace, semicolon, or unbalanced delimiter) and retry. The \
                 body for replace_body must include only the inner block \
                 contents (no outer braces required)."
                    .to_string(),
            ),
            "ambiguous_rename" => Some(
                "Rename was ambiguous. Use a more qualified target or fall \
                 back to per-call-site replace_body."
                    .to_string(),
            ),
            "invalid_target" => Some(
                "Target not found. Re-query the ontology to confirm the \
                 entity exists, then retry with the canonical \
                 module_path + name."
                    .to_string(),
            ),
            "indexed_overlap" => Some(
                "edit_file refused: the requested region overlaps a structural \
                 ontology item. Use the appropriate update_ontology operation \
                 (replace_body / add_function / rename) for the named owner, \
                 or pick one of the gap regions reported in the response."
                    .to_string(),
            ),
            "ambiguous_match" => Some(
                "edit_file find/replace matched multiple locations. Add more \
                 surrounding context to the find string until it is unique, \
                 or address the edit by line range instead."
                    .to_string(),
            ),
            "not_found" => Some(
                "edit_file find string not found. Query the File entity first \
                 to confirm content, or use line-range addressing."
                    .to_string(),
            ),
            "file_exists" => Some(
                "create_file refused: a file already exists at that path. Use \
                 edit_file to modify it, or pick a new path."
                    .to_string(),
            ),
            "file_has_indexed_items" => Some(
                "delete_file refused: the file contains indexed structural \
                 items. Remove those via update_ontology first, or accept the \
                 cascade impact."
                    .to_string(),
            ),
            _ => None,
        };
        Self {
            success: false,
            rollback_reason: Some(reason.into()),
            files_changed: vec![],
            compile_status: serde_json::Value::Null,
            graph_diff: None,
            details: Some(details),
            hints: hint.into_iter().collect(),
        }
    }
}

/// Append a one-line reminder to call run_cargo before finalizing the
/// turn. The harness enforces this as a hard gate, but the hint plants
/// the next-step signal at point of edit so the model doesn't have to
/// rediscover the rule from the system prompt.
fn push_verification_reminder(hints: &mut Vec<String>) {
    hints.push(
        "Edit committed. Before finalizing your response, call run_cargo \
         (test for runtime, check for compile-only). One verification \
         covers all preceding edits."
            .into(),
    );
}

fn compile_hints(compile: &serde_json::Value) -> Vec<String> {
    let summary = compile.get("summary");
    let introduced = summary
        .and_then(|s| s.get("introduced"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let pre_existing = summary
        .and_then(|s| s.get("pre_existing"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let fixed = summary
        .and_then(|s| s.get("fixed"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut hints = Vec::new();
    if introduced > 0 {
        hints.push(format!(
            "{introduced} error(s) introduced by this edit. \
             These are caused by your change - address them before \
             finalizing. Disk is not rolled back; you can keep editing \
             through this state."
        ));
    }
    if pre_existing > 0 {
        hints.push(format!(
            "{pre_existing} pre-existing error(s) in files you didn't \
             touch. These were already broken before this edit. Surface \
             them as TODOs in your final response, but do NOT chase them \
             - they're not yours to fix in this turn."
        ));
    }
    if fixed > 0 {
        hints.push(format!(
            "Edit fixed {fixed} previously-broken error(s)."
        ));
    }
    hints
}

/// Run cargo check, parse errors, and classify each one against a
/// baseline error set. Returns the JSON value the model sees and the
/// fresh error set for caching as the next baseline.
fn run_cargo_check_classified(
    workspace: &Path,
    baseline: &HashSet<ErrorKey>,
) -> (serde_json::Value, HashSet<ErrorKey>) {
    let raw = run_cargo_check(workspace);
    let errors_array = match raw.get("errors").and_then(|v| v.as_array()) {
        Some(a) => a.clone(),
        None => {
            // Either "ok" string or no errors object - means clean
            // build. Compute "fixed" against baseline since previously-
            // broken errors are now gone.
            let summary = serde_json::json!({
                "introduced": 0,
                "pre_existing": 0,
                "fixed": baseline.len(),
            });
            let result = if baseline.is_empty() {
                raw
            } else {
                serde_json::json!({
                    "status": "ok",
                    "summary": summary,
                })
            };
            return (result, HashSet::new());
        }
    };

    let mut current_set = HashSet::new();
    let mut tagged: Vec<serde_json::Value> = Vec::new();
    for err in &errors_array {
        let key = error_key_from_json(err);
        let introduced = !baseline.contains(&key);
        current_set.insert(key);
        let mut tagged_err = err.clone();
        if let Some(obj) = tagged_err.as_object_mut() {
            obj.insert(
                "introduced_by_this_edit".to_string(),
                serde_json::json!(introduced),
            );
        }
        tagged.push(tagged_err);
    }

    let introduced_count = tagged
        .iter()
        .filter(|e| {
            e.get("introduced_by_this_edit")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .count();
    let pre_existing_count = tagged.len() - introduced_count;
    let fixed_count = baseline.iter().filter(|k| !current_set.contains(k)).count();

    let result = serde_json::json!({
        "errors": tagged,
        "summary": {
            "introduced": introduced_count,
            "pre_existing": pre_existing_count,
            "fixed": fixed_count,
        }
    });
    (result, current_set)
}

fn error_key_from_json(e: &serde_json::Value) -> ErrorKey {
    ErrorKey {
        file: e
            .get("file")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        line: e.get("line").and_then(|v| v.as_u64()).unwrap_or(0),
        code: e
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        message: e
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    }
}

/// Public entry to run a classified cargo check (e.g. from main.rs to
/// seed the baseline at session start).
pub fn cargo_check_classified(
    workspace: &Path,
    baseline: &HashSet<ErrorKey>,
) -> (serde_json::Value, HashSet<ErrorKey>) {
    run_cargo_check_classified(workspace, baseline)
}

impl Ontology {
    /// Re-walk the workspace and rebuild the structural graph in place,
    /// preserving cross-edit session state (currently `prev_errors` for
    /// the compile-error classifier). Use this everywhere we'd
    /// previously written `*self = indexer::index(...)`.
    pub fn refresh_index(&mut self) -> Result<()> {
        let preserved_errors = std::mem::take(&mut self.prev_errors);
        let workspace = self.workspace.clone();
        *self = indexer::index(&workspace)?;
        self.prev_errors = preserved_errors;
        Ok(())
    }

    /// Run cargo check using the cached error baseline, classify each
    /// resulting error, and update the cache to be the next baseline.
    /// Used by every editing op so its response can tell the model
    /// which errors it introduced vs which were already there.
    fn run_classified_check(&mut self) -> serde_json::Value {
        let baseline = std::mem::take(&mut self.prev_errors);
        let (compile, fresh) = run_cargo_check_classified(&self.workspace, &baseline);
        self.prev_errors = fresh;
        compile
    }

    pub fn update(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        match req.operation.as_str() {
            "replace_body" => self.op_replace_body(req),
            "replace_item" => self.op_replace_item(req),
            "add_function" => self.op_add_function(req),
            "rename" => self.op_rename(req),
            "edit_file" => self.op_edit_file(req),
            "create_file" => self.op_create_file(req),
            "delete_file" => self.op_delete_file(req),
            other => Err(anyhow!("unknown operation: {other}")),
        }
    }

    fn op_replace_body(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let name = req
            .target
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.name required"))?;
        let module_path = req.target.get("module_path").and_then(|v| v.as_str());
        let new_body = req
            .payload
            .get("new_body")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("payload.new_body required"))?;

        // Defensive: a common LLM mistake is to send the entire function
        // definition as the body (signature + braces and all). Detect that
        // here and bounce with a clear error so the model fixes its
        // payload instead of producing a nested fn declaration that
        // happens to parse but breaks the type.
        let trimmed = new_body.trim();
        let trimmed = trimmed.strip_prefix('{').unwrap_or(trimmed).trim();
        let trimmed = trimmed.strip_suffix('}').unwrap_or(trimmed).trim();
        if syn::parse_str::<syn::ItemFn>(trimmed).is_ok() {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                "payload.new_body parses as a top-level `fn` declaration. \
                 replace_body wants ONLY the inner statements/expressions \
                 of the function body, NOT the signature or surrounding \
                 braces. Example: for `fn f() -> i32 { 1 + 2 }`, send \
                 `new_body: \"1 + 2\"`. To rewrite the whole item \
                 (signature, attrs, doc, body), use replace_item instead.",
            ));
        }

        let func = match self.function(name, module_path).cloned() {
            Some(f) => f,
            None => {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    format!("function not found: {name}"),
                ))
            }
        };

        let file_path = func.file.clone();
        let original = fs::read_to_string(&file_path)?;
        let parsed = match syn::parse_file(&original) {
            Ok(f) => f,
            Err(e) => {
                return Ok(UpdateResponse::rollback(
                    "parse_error",
                    format!("existing file does not parse: {e}"),
                ))
            }
        };

        let new_source = match replace_body_in_source(&parsed, &original, &func.name, new_body) {
            Ok(s) => s,
            Err(e) => return Ok(UpdateResponse::rollback("invalid_target", e.to_string())),
        };

        if let Err(e) = syn::parse_file(&new_source) {
            return Ok(UpdateResponse::rollback(
                "parse_error",
                format!("regenerated source does not parse: {e}"),
            ));
        }

        let diff = make_diff(&original, &new_source, &file_path);
        if !req.dry_run {
            fs::write(&file_path, &new_source)?;
            // Re-index this file's contributions by reparsing the workspace.
            // For v0 we just refresh the whole ontology; cheap on small repos.
            self.refresh_index()?;
        }

        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        let (callers, tests) =
            crate::ontology::query::compute_affected_topology(self, &func.name);
        if !callers.is_empty() || !tests.is_empty() {
            hints.push(format!(
                "{} caller(s) and {} test(s) reference `{}`. \
                 Verify them after the change; if they break, address \
                 with replace_body / replace_item on each, then re-check.",
                callers.len(),
                tests.len(),
                func.name,
            ));
        }

        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed: vec![FileChange {
                path: file_path.display().to_string(),
                diff,
            }],
            compile_status: compile,
            graph_diff: Some(serde_json::json!({
                "edited": format!("{}::{}", func.module_path, func.name),
                "affected": {
                    "callers": callers,
                    "tests": tests,
                },
            })),
            details: None,
            hints,
        })
    }

    /// Replace an indexed item's full presentation (leading attrs + doc
    /// comments + signature + body) with new source. Use this when you need
    /// to add/remove decorators (`#[tracing::instrument]`), change a doc
    /// comment, or alter a function's signature. Use `replace_body` for
    /// narrower edits that only touch the body block.
    fn op_replace_item(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let object_type = req
            .target
            .get("object_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Function");
        let name = req
            .target
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.name required"))?;
        let module_path = req
            .target
            .get("module_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.module_path required"))?;
        let new_source = req
            .payload
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("payload.source required (full item source)"))?;

        // Resolve the file + span from the indexed entity.
        let key = format!("{}::{}", module_path, name);
        let (file_path, line_start, line_end) = match object_type {
            "Function" => self
                .functions
                .get(&key)
                .map(|f| (f.file.clone(), f.line_start, f.line_end)),
            "Type" => self
                .types
                .get(&key)
                .map(|t| (t.file.clone(), t.line_start, t.line_end)),
            "Trait" => self
                .traits
                .get(&key)
                .map(|t| (t.file.clone(), t.line_start, t.line_end)),
            other => {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    format!("replace_item does not support object_type={other}"),
                ))
            }
        }
        .ok_or(())
        .or_else(|_| {
            Ok::<(PathBuf, usize, usize), anyhow::Error>((PathBuf::new(), 0, 0))
        })?;
        if file_path.as_os_str().is_empty() {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                format!("{object_type} {key} not found in ontology"),
            ));
        }

        // Validate the supplied source as a top-level item of the right kind.
        let parse_ok = match object_type {
            "Function" => syn::parse_str::<syn::ItemFn>(new_source).is_ok()
                || syn::parse_str::<syn::ImplItemFn>(new_source).is_ok(),
            "Type" => syn::parse_str::<syn::ItemStruct>(new_source).is_ok()
                || syn::parse_str::<syn::ItemEnum>(new_source).is_ok(),
            "Trait" => syn::parse_str::<syn::ItemTrait>(new_source).is_ok(),
            _ => false,
        };
        if !parse_ok {
            return Ok(UpdateResponse::rollback(
                "parse_error",
                format!(
                    "payload.source does not parse as a {object_type} item. \
                     Provide the full item source including any leading \
                     attributes and doc comments."
                ),
            ));
        }

        let original = fs::read_to_string(&file_path)?;
        let new_file = splice_lines(&original, line_start, line_end, new_source);
        if let Err(e) = syn::parse_file(&new_file) {
            return Ok(UpdateResponse::rollback(
                "parse_error",
                format!("regenerated source does not parse: {e}"),
            ));
        }

        let diff = make_diff(&original, &new_file, &file_path);
        if !req.dry_run {
            fs::write(&file_path, &new_file)?;
            self.refresh_index()?;
        }
        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        let (callers, tests) =
            crate::ontology::query::compute_affected_topology(self, name);
        if !callers.is_empty() || !tests.is_empty() {
            hints.push(format!(
                "{} caller(s) and {} test(s) reference `{}`. \
                 If your change altered the signature or attrs, those \
                 sites may need updating; cargo check will surface any \
                 breakage.",
                callers.len(),
                tests.len(),
                name,
            ));
        }
        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed: vec![FileChange {
                path: file_path.display().to_string(),
                diff,
            }],
            compile_status: compile,
            graph_diff: Some(serde_json::json!({
                "replaced": key,
                "object_type": object_type,
                "affected": {
                    "callers": callers,
                    "tests": tests,
                },
            })),
            details: None,
            hints,
        })
    }

    fn op_add_function(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let module_path = req
            .target
            .get("module_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.module_path required"))?;
        let impl_for_type = req
            .target
            .get("impl_for_type")
            .and_then(|v| v.as_str());
        let source = req
            .payload
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("payload.source required"))?;

        // Validate the new function parses on its own.
        if let Err(e) = syn::parse_str::<syn::ItemFn>(source) {
            // Try as ImplItemFn for impl-block insertions.
            if impl_for_type.is_none() || syn::parse_str::<syn::ImplItemFn>(source).is_err() {
                return Ok(UpdateResponse::rollback(
                    "parse_error",
                    format!("payload.source does not parse as a function: {e}"),
                ));
            }
        }

        // Resolve target file. v0: pick a representative file from the
        // module by looking at any function/type already there.
        let file_path = match find_module_file(self, module_path) {
            Some(p) => p,
            None => {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    format!("could not resolve a file for module {module_path}"),
                ))
            }
        };
        let original = fs::read_to_string(&file_path)?;

        let new_source = if let Some(ty) = impl_for_type {
            match insert_into_impl_block(&original, ty, source) {
                Some(s) => s,
                None => {
                    return Ok(UpdateResponse::rollback(
                        "invalid_target",
                        format!("no impl block for {ty} in {}", file_path.display()),
                    ))
                }
            }
        } else {
            append_top_level(&original, source)
        };

        if let Err(e) = syn::parse_file(&new_source) {
            return Ok(UpdateResponse::rollback(
                "parse_error",
                format!("regenerated source does not parse: {e}"),
            ));
        }

        let diff = make_diff(&original, &new_source, &file_path);
        if !req.dry_run {
            fs::write(&file_path, &new_source)?;
            self.refresh_index()?;
        }

        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed: vec![FileChange {
                path: file_path.display().to_string(),
                diff,
            }],
            compile_status: compile,
            graph_diff: Some(serde_json::json!({
                "nodes_added": [],
                "nodes_removed": [],
                "edges_changed": []
            })),
            details: None,
            hints,
        })
    }

    fn op_rename(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let object_type = req
            .target
            .get("object_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Function");
        let name = req
            .target
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.name required"))?;
        let new_name = req
            .payload
            .get("new_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("payload.new_name required"))?;

        if !is_valid_ident(new_name) {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                format!("{new_name} is not a valid Rust identifier"),
            ));
        }

        // Naive rename: word-boundary replacement across .rs files.
        // Confirm the original name appears at least once (otherwise we
        // bail out as ambiguous / not found).
        let workspace = self.workspace.clone();
        let mut occurrences = 0usize;
        let mut staged: Vec<(PathBuf, String, String)> = vec![];
        for entry in walkdir::WalkDir::new(&workspace)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| {
                let p = e.path();
                p.extension().and_then(|s| s.to_str()) == Some("rs")
                    && !p.components().any(|c| c.as_os_str() == "target")
            })
        {
            let p = entry.path();
            let original = match fs::read_to_string(p) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let (new_src, count) = word_boundary_replace(&original, name, new_name);
            if count > 0 {
                occurrences += count;
                if let Err(e) = syn::parse_file(&new_src) {
                    return Ok(UpdateResponse::rollback(
                        "parse_error",
                        format!(
                            "rename produced unparseable {}: {e}. \
                             v0 rename is naive text replacement; consider \
                             using replace_body for surgical edits.",
                            p.display()
                        ),
                    ));
                }
                staged.push((p.to_path_buf(), original, new_src));
            }
        }

        if occurrences == 0 {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                format!("identifier {name} not found in workspace"),
            ));
        }

        let mut files_changed = vec![];
        for (path, original, new_src) in &staged {
            files_changed.push(FileChange {
                path: path.display().to_string(),
                diff: make_diff(original, new_src, path),
            });
        }

        if !req.dry_run {
            for (path, _, new_src) in &staged {
                fs::write(path, new_src)?;
            }
            self.refresh_index()?;
        }

        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        // Topology around the renamed entity, queried by the appropriate
        // name. For a dry-run the index hasn't changed so look up by old
        // name; otherwise the rename has already landed under new_name.
        let lookup_name = if req.dry_run { name } else { new_name };
        let (callers, tests) =
            crate::ontology::query::compute_affected_topology(self, lookup_name);
        if !callers.is_empty() || !tests.is_empty() {
            hints.push(format!(
                "{} caller(s) and {} test(s) referenced `{}` (now `{}`). \
                 If compile broke, address with replace_body / edit_file \
                 on each. cargo test confirms runtime behavior.",
                callers.len(),
                tests.len(),
                name,
                new_name,
            ));
        }
        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed,
            compile_status: compile,
            graph_diff: Some(serde_json::json!({
                "renamed": { "from": name, "to": new_name, "occurrences": occurrences },
                "object_type": object_type,
                "affected": {
                    "callers": callers,
                    "tests": tests,
                },
            })),
            details: Some(format!(
                "v0 naive rename: {occurrences} occurrence(s) replaced via word-boundary text match. \
                 Confirm compile_status is clean before relying on this."
            )),
            hints,
        })
    }

    fn op_edit_file(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let rel_path = req
            .target
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.path required"))?;
        let abs_path = self.workspace.join(rel_path);

        if !abs_path.exists() {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                format!("file not found: {rel_path}"),
            ));
        }

        let original = match fs::read_to_string(&abs_path) {
            Ok(s) => s,
            Err(e) => {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    format!("cannot read {rel_path} as text: {e}"),
                ))
            }
        };

        // Decide which addressing mode the agent used.
        let mode_find = req.payload.get("find").is_some();
        let mode_lines = req.payload.get("line_start").is_some()
            || req.payload.get("line_end").is_some();
        if mode_find && mode_lines {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                "specify either {find,replace} or {line_start,line_end,replacement}, not both",
            ));
        }

        let (region_start_line, region_end_line, new_content) = if mode_find {
            let find = req.payload.get("find").and_then(|v| v.as_str()).unwrap_or("");
            let replace = req
                .payload
                .get("replace")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if find.is_empty() {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    "payload.find must be non-empty",
                ));
            }
            let matches: Vec<usize> = original.match_indices(find).map(|(i, _)| i).collect();
            if matches.is_empty() {
                return Ok(UpdateResponse::rollback(
                    "not_found",
                    format!("find string not present in {rel_path}"),
                ));
            }
            if matches.len() > 1 {
                return Ok(UpdateResponse::rollback(
                    "ambiguous_match",
                    format!(
                        "find string matched {} times in {rel_path}; add surrounding context to make it unique",
                        matches.len()
                    ),
                ));
            }
            let start = matches[0];
            let end = start + find.len();
            let line_start = byte_to_line(&original, start);
            let line_end = byte_to_line(&original, end.saturating_sub(1));
            let mut out = String::with_capacity(original.len() + replace.len());
            out.push_str(&original[..start]);
            out.push_str(replace);
            out.push_str(&original[end..]);
            (line_start, line_end, out)
        } else if mode_lines {
            let line_start = req
                .payload
                .get("line_start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let line_end = req
                .payload
                .get("line_end")
                .and_then(|v| v.as_u64())
                .unwrap_or(line_start as u64) as usize;
            let replacement = req
                .payload
                .get("replacement")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if line_start == 0 || line_end < line_start {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    "line_start must be >= 1 and line_end >= line_start",
                ));
            }
            let total_lines = original.lines().count();
            if line_end > total_lines {
                return Ok(UpdateResponse::rollback(
                    "invalid_target",
                    format!(
                        "line_end {line_end} exceeds file length {total_lines} in {rel_path}"
                    ),
                ));
            }
            let new_content = splice_lines(&original, line_start, line_end, replacement);
            (line_start, line_end, new_content)
        } else {
            // Whole-file replacement via `content`.
            let content = match req.payload.get("content").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => {
                    return Ok(UpdateResponse::rollback(
                        "invalid_target",
                        "edit_file payload requires one of: {find,replace}, {line_start,line_end,replacement}, or {content}",
                    ))
                }
            };
            let total_lines = original.lines().count().max(1);
            (1, total_lines, content.to_string())
        };

        // Overlap guard. We compare against the most recent indexed_spans.
        if let Some(file_entity) = self.files.get(rel_path) {
            let conflicts: Vec<&crate::ontology::model::IndexedSpan> = file_entity
                .indexed_spans
                .iter()
                .filter(|s| {
                    !(s.line_end < region_start_line || s.line_start > region_end_line)
                })
                .collect();
            if !conflicts.is_empty() {
                let conflict_json: Vec<serde_json::Value> = conflicts
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "kind": s.kind,
                            "owner": s.owner,
                            "line_start": s.line_start,
                            "line_end": s.line_end,
                            "suggested_op": match s.kind.as_str() {
                                "Function" => "replace_body or rename",
                                "Type" | "Trait" => "currently no surgical op; \
                                    edit the surrounding gap or extend update_ontology",
                                _ => "see update_ontology",
                            },
                        })
                    })
                    .collect();
                let outline = crate::ontology::query::compute_outline(
                    file_entity,
                    &self.functions,
                    &self.types,
                    &self.traits,
                );
                let mut resp = UpdateResponse::rollback(
                    "indexed_overlap",
                    format!(
                        "edit region {}..{} in {} overlaps {} indexed span(s)",
                        region_start_line,
                        region_end_line,
                        rel_path,
                        conflicts.len()
                    ),
                );
                resp.graph_diff = Some(serde_json::json!({
                    "conflicts": conflict_json,
                    "outline": outline,
                }));
                return Ok(resp);
            }
        }

        // Parse-validate Rust files. Other languages skip this step.
        let is_rust = rel_path.ends_with(".rs");
        if is_rust {
            if let Err(e) = syn::parse_file(&new_content) {
                return Ok(UpdateResponse::rollback(
                    "parse_error",
                    format!("regenerated {rel_path} does not parse: {e}"),
                ));
            }
        }

        let diff = make_diff(&original, &new_content, &abs_path);
        if !req.dry_run {
            fs::write(&abs_path, &new_content)?;
            self.refresh_index()?;
        }

        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed: vec![FileChange {
                path: abs_path.display().to_string(),
                diff,
            }],
            compile_status: compile,
            graph_diff: Some(serde_json::json!({ "edited": rel_path })),
            details: None,
            hints,
        })
    }

    fn op_create_file(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let rel_path = req
            .target
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.path required"))?;
        let content = req
            .payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let abs_path = self.workspace.join(rel_path);
        if abs_path.exists() {
            return Ok(UpdateResponse::rollback(
                "file_exists",
                format!("path already exists: {rel_path}"),
            ));
        }

        // For .rs files, parse-validate before writing.
        if rel_path.ends_with(".rs") {
            if let Err(e) = syn::parse_file(content) {
                return Ok(UpdateResponse::rollback(
                    "parse_error",
                    format!("content does not parse as Rust: {e}"),
                ));
            }
        }

        let diff = make_diff("", content, &abs_path);
        if !req.dry_run {
            if let Some(parent) = abs_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&abs_path, content)?;
            self.refresh_index()?;
        }

        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed: vec![FileChange {
                path: abs_path.display().to_string(),
                diff,
            }],
            compile_status: compile,
            graph_diff: Some(serde_json::json!({ "created": rel_path })),
            details: None,
            hints,
        })
    }

    fn op_delete_file(&mut self, req: &UpdateRequest) -> Result<UpdateResponse> {
        let rel_path = req
            .target
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("target.path required"))?;
        let abs_path = self.workspace.join(rel_path);
        if !abs_path.exists() {
            return Ok(UpdateResponse::rollback(
                "invalid_target",
                format!("file not found: {rel_path}"),
            ));
        }

        if let Some(file_entity) = self.files.get(rel_path) {
            if !file_entity.indexed_spans.is_empty() {
                let owners: Vec<String> = file_entity
                    .indexed_spans
                    .iter()
                    .map(|s| format!("{} ({})", s.owner, s.kind))
                    .collect();
                let mut resp = UpdateResponse::rollback(
                    "file_has_indexed_items",
                    format!(
                        "{rel_path} contains {} indexed item(s): {}",
                        owners.len(),
                        owners.join(", ")
                    ),
                );
                resp.graph_diff = Some(serde_json::json!({
                    "blocking_items": file_entity.indexed_spans,
                }));
                return Ok(resp);
            }
        }

        let original = fs::read_to_string(&abs_path).unwrap_or_default();
        let diff = make_diff(&original, "", &abs_path);
        if !req.dry_run {
            fs::remove_file(&abs_path)?;
            self.refresh_index()?;
        }

        let compile = self.run_classified_check();
        let mut hints = compile_hints(&compile);
        push_verification_reminder(&mut hints);
        Ok(UpdateResponse {
            success: true,
            rollback_reason: None,
            files_changed: vec![FileChange {
                path: abs_path.display().to_string(),
                diff,
            }],
            compile_status: compile,
            graph_diff: Some(serde_json::json!({ "deleted": rel_path })),
            details: None,
            hints,
        })
    }
}

// ---------- helpers ----------

fn replace_body_in_source(
    parsed: &syn::File,
    original: &str,
    target_name: &str,
    new_body: &str,
) -> Result<String> {
    let span = find_fn_block_span(&parsed.items, target_name)
        .ok_or_else(|| anyhow!("function {target_name} not found in file"))?;

    let line_index = LineIndex::new(original);
    let start = line_index.offset(span.0.line, span.0.column);
    let end = line_index.offset(span.1.line, span.1.column);
    if start > original.len() || end > original.len() || start > end {
        return Err(anyhow!("computed span out of bounds"));
    }

    // Detect the function's indent (column of `fn` line) so the body
    // matches surrounding code. Falls back to 0 if not found.
    let fn_line = original[..start]
        .rfind("\n")
        .map(|p| &original[p + 1..start])
        .unwrap_or("");
    let fn_indent = fn_line.chars().take_while(|c| c.is_whitespace()).count();

    // Defensive: agent might submit body with or without outer braces.
    let body_inner = strip_outer_braces(new_body);

    let close_indent = " ".repeat(fn_indent);
    let indented = reindent(body_inner, fn_indent + 4);

    let mut out = String::with_capacity(original.len() + new_body.len() + 16);
    out.push_str(&original[..start]);
    out.push('{');
    out.push('\n');
    out.push_str(&indented);
    out.push_str(&close_indent);
    out.push('}');
    out.push_str(&original[end..]);
    Ok(out)
}

fn strip_outer_braces(s: &str) -> &str {
    // If the first and last non-whitespace characters are `{` and `}`,
    // return the slice between them. Preserves internal indentation so
    // reindent() can detect the body's base indent correctly.
    let bytes = s.as_bytes();
    let first = bytes.iter().position(|b| !b.is_ascii_whitespace());
    let last = bytes.iter().rposition(|b| !b.is_ascii_whitespace());
    if let (Some(f), Some(l)) = (first, last) {
        if f < l && bytes[f] == b'{' && bytes[l] == b'}' {
            return &s[f + 1..l];
        }
    }
    s
}

/// Re-indent a body block. Detects the minimum indent of non-empty lines
/// in the input (the body's "base indent") and replaces it with `target`
/// spaces, preserving relative nesting.
fn reindent(body: &str, target: usize) -> String {
    let min_indent = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').count())
        .min()
        .unwrap_or(0);
    let prefix = " ".repeat(target);
    let mut out = String::new();
    for line in body.lines() {
        if line.trim().is_empty() {
            out.push('\n');
        } else {
            let stripped = line.get(min_indent..).unwrap_or(line);
            out.push_str(&prefix);
            out.push_str(stripped);
            out.push('\n');
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct LC {
    line: usize,
    column: usize,
}

fn find_fn_block_span(items: &[syn::Item], target: &str) -> Option<(LC, LC)> {
    use syn::spanned::Spanned;
    for item in items {
        match item {
            syn::Item::Fn(f) if f.sig.ident == target => {
                let s = f.block.span();
                return Some((
                    LC {
                        line: s.start().line,
                        column: s.start().column,
                    },
                    LC {
                        line: s.end().line,
                        column: s.end().column,
                    },
                ));
            }
            syn::Item::Impl(imp) => {
                for it in &imp.items {
                    if let syn::ImplItem::Fn(m) = it {
                        if m.sig.ident == target {
                            let s = m.block.span();
                            return Some((
                                LC {
                                    line: s.start().line,
                                    column: s.start().column,
                                },
                                LC {
                                    line: s.end().line,
                                    column: s.end().column,
                                },
                            ));
                        }
                    }
                }
            }
            syn::Item::Mod(m) => {
                if let Some((_, items)) = &m.content {
                    if let Some(span) = find_fn_block_span(items, target) {
                        return Some(span);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

struct LineIndex {
    line_starts: Vec<usize>,
}

impl LineIndex {
    fn new(s: &str) -> Self {
        let mut starts = vec![0];
        for (i, b) in s.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Self { line_starts: starts }
    }
    /// Convert proc-macro2 LineColumn (1-based line, 0-based column in
    /// UTF-8 bytes) to a byte offset in the source.
    fn offset(&self, line: usize, column: usize) -> usize {
        if line == 0 {
            return 0;
        }
        let idx = line - 1;
        let line_start = self.line_starts.get(idx).copied().unwrap_or(0);
        line_start + column
    }
}

fn find_module_file(ont: &Ontology, module_path: &str) -> Option<PathBuf> {
    if let Some(m) = ont.modules.get(module_path) {
        if !m.file.as_os_str().is_empty() {
            return Some(m.file.clone());
        }
    }
    // Fallback: any function in that module
    for f in ont.functions.values() {
        if f.module_path == module_path {
            return Some(f.file.clone());
        }
    }
    for t in ont.types.values() {
        if t.module_path == module_path {
            return Some(t.file.clone());
        }
    }
    None
}

fn append_top_level(source: &str, item_source: &str) -> String {
    let mut out = String::with_capacity(source.len() + item_source.len() + 2);
    out.push_str(source);
    if !source.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(item_source);
    if !item_source.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn insert_into_impl_block(source: &str, type_name: &str, method_source: &str) -> Option<String> {
    let parsed = syn::parse_file(source).ok()?;
    let span = find_impl_close_brace(&parsed.items, type_name)?;
    let line_index = LineIndex::new(source);
    // The block's end span points at the closing `}`. We insert before it.
    let close_off = line_index.offset(span.line, span.column);
    let close_off = close_off.min(source.len()).saturating_sub(1);
    let mut out = String::with_capacity(source.len() + method_source.len() + 4);
    out.push_str(&source[..close_off]);
    out.push('\n');
    out.push_str(method_source.trim_end_matches('\n'));
    out.push_str("\n}");
    if close_off + 1 < source.len() {
        out.push_str(&source[close_off + 1..]);
    }
    Some(out)
}

fn find_impl_close_brace(items: &[syn::Item], type_name: &str) -> Option<LC> {
    use syn::spanned::Spanned;
    for item in items {
        if let syn::Item::Impl(imp) = item {
            let ty = type_path_ident(&imp.self_ty);
            if ty.as_deref() == Some(type_name) {
                let s = imp.brace_token.span.span().end();
                return Some(LC {
                    line: s.line,
                    column: s.column,
                });
            }
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, items)) = &m.content {
                if let Some(s) = find_impl_close_brace(items, type_name) {
                    return Some(s);
                }
            }
        }
    }
    None
}

fn type_path_ident(ty: &syn::Type) -> Option<String> {
    if let syn::Type::Path(p) = ty {
        return p.path.segments.last().map(|s| s.ident.to_string());
    }
    None
}

fn word_boundary_replace(src: &str, from: &str, to: &str) -> (String, usize) {
    let bytes = src.as_bytes();
    let from_bytes = from.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut count = 0;
    while i < bytes.len() {
        if i + from_bytes.len() <= bytes.len() && &bytes[i..i + from_bytes.len()] == from_bytes {
            let before_ok = i == 0 || !is_ident_continue_byte(bytes[i - 1]);
            let after_ok = i + from_bytes.len() == bytes.len()
                || !is_ident_continue_byte(bytes[i + from_bytes.len()]);
            if before_ok && after_ok {
                out.push_str(to);
                i += from_bytes.len();
                count += 1;
                continue;
            }
        }
        // Push the next char (handle multi-byte safely)
        let ch_len = utf8_char_len(bytes[i]);
        out.push_str(std::str::from_utf8(&bytes[i..i + ch_len]).unwrap_or(""));
        i += ch_len;
    }
    (out, count)
}

fn is_ident_continue_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xC0 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn byte_to_line(src: &str, byte_offset: usize) -> usize {
    if byte_offset == 0 {
        return 1;
    }
    let mut line = 1;
    for (i, b) in src.bytes().enumerate() {
        if i >= byte_offset {
            break;
        }
        if b == b'\n' {
            line += 1;
        }
    }
    line
}

fn splice_lines(src: &str, line_start: usize, line_end: usize, replacement: &str) -> String {
    let mut out = String::with_capacity(src.len() + replacement.len());
    let mut current = 1usize;
    let mut wrote_replacement = false;
    let line_iter: Vec<&str> = src.split_inclusive('\n').collect();
    for line in &line_iter {
        if current < line_start || current > line_end {
            out.push_str(line);
        } else if !wrote_replacement {
            out.push_str(replacement);
            if !replacement.ends_with('\n') {
                out.push('\n');
            }
            wrote_replacement = true;
        }
        current += 1;
    }
    if !wrote_replacement {
        // Replacement range is past the file end; append it.
        out.push_str(replacement);
    }
    out
}

fn make_diff(before: &str, after: &str, path: &Path) -> String {
    let diff = TextDiff::from_lines(before, after);
    let mut out = String::new();
    out.push_str(&format!("--- {}\n", path.display()));
    out.push_str(&format!("+++ {}\n", path.display()));
    for hunk in diff.unified_diff().context_radius(2).iter_hunks() {
        out.push_str(&hunk.to_string());
    }
    out
}

fn run_cargo_check(workspace: &Path) -> serde_json::Value {
    let result = Command::new("cargo")
        .arg("check")
        .arg("--message-format=json-diagnostic-rendered-ansi")
        .arg("--quiet")
        .current_dir(workspace)
        .output();
    let output = match result {
        Ok(o) => o,
        Err(e) => {
            return serde_json::json!({
                "errors": [{
                    "message": format!("failed to invoke cargo check: {e}"),
                    "code": "EXEC_FAIL",
                }]
            });
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut errors: Vec<serde_json::Value> = vec![];
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
        let text = msg
            .get("message")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let code = msg
            .get("code")
            .and_then(|c| c.get("code"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let (file, line, column) = msg
            .get("spans")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .map(|sp| {
                (
                    sp.get("file_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    sp.get("line_start").and_then(|v| v.as_u64()).unwrap_or(0),
                    sp.get("column_start").and_then(|v| v.as_u64()).unwrap_or(0),
                )
            })
            .unwrap_or_default();
        errors.push(serde_json::json!({
            "file": file,
            "line": line,
            "column": column,
            "message": text,
            "code": code,
        }));
    }

    if errors.is_empty() && output.status.success() {
        return serde_json::Value::String("ok".into());
    }
    if errors.is_empty() {
        // No JSON errors but cargo check failed: include stderr as a hint.
        return serde_json::json!({
            "errors": [{
                "message": stderr.trim().to_string(),
                "code": "CARGO_FAIL",
            }]
        });
    }
    serde_json::json!({ "errors": errors })
}
