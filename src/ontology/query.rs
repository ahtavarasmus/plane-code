//! Query operations on the ontology. Implements the `query_codebase` tool.
//!
//! v0 ranking is hybrid lexical: substring on name, module_path, signature,
//! and doc, weighted by field. No embeddings yet. Replace `score_*` once
//! an embedding model is wired in.

use crate::ontology::model::*;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Build the `outline` field for a File response. Outline is the agent's
/// only view into a file's contents:
///   - `gap` entries carry full text content for non-graph regions
///     (use statements, const/static, comments between items).
///   - `indexed` entries carry only `signature` (and owner key); body and
///     attribute/doc text are NOT exposed here. To read or edit those,
///     the agent must query the structural entity (Function/Type/Trait)
///     and use the matching update_codebase op.
/// This is what enforces graph-first navigation: File responses tell the
/// agent which structural items live where, but cannot be used as a
/// shortcut to read whole-file source.
pub(crate) fn compute_outline(
    f: &File,
    functions: &HashMap<String, Function>,
    types: &HashMap<String, Type>,
    traits: &HashMap<String, Trait>,
) -> Vec<serde_json::Value> {
    if f.content.is_empty() && f.indexed_spans.is_empty() {
        return vec![];
    }
    let lines: Vec<&str> = f.content.lines().collect();
    let total_lines = lines.len();

    if f.indexed_spans.is_empty() {
        return vec![serde_json::json!({
            "kind": "gap",
            "line_start": 1,
            "line_end": total_lines,
            "content": f.content,
        })];
    }

    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut cursor: usize = 1;
    for span in &f.indexed_spans {
        if span.line_start > cursor {
            out.push(serde_json::json!({
                "kind": "gap",
                "line_start": cursor,
                "line_end": span.line_start - 1,
                "content": slice_lines(&lines, cursor, span.line_start - 1),
            }));
        }
        out.push(serde_json::json!({
            "kind": "indexed",
            "line_start": span.line_start,
            "line_end": span.line_end,
            "owner": span.owner,
            "owner_kind": span.kind,
            "signature": signature_for_outline(span, functions, types, traits),
        }));
        cursor = span.line_end + 1;
    }
    if cursor <= total_lines {
        out.push(serde_json::json!({
            "kind": "gap",
            "line_start": cursor,
            "line_end": total_lines,
            "content": slice_lines(&lines, cursor, total_lines),
        }));
    }
    out
}

fn slice_lines(lines: &[&str], start: usize, end: usize) -> String {
    if start == 0 || start > lines.len() {
        return String::new();
    }
    let s = start - 1;
    let e = end.min(lines.len());
    lines[s..e].join("\n")
}

fn signature_for_outline(
    span: &IndexedSpan,
    functions: &HashMap<String, Function>,
    types: &HashMap<String, Type>,
    traits: &HashMap<String, Trait>,
) -> String {
    match span.kind.as_str() {
        "Function" => functions
            .get(&span.owner)
            .map(|f| f.signature.clone())
            .unwrap_or_default(),
        "Type" => types
            .get(&span.owner)
            .map(|t| {
                let vis = if t.visibility.is_empty() {
                    String::new()
                } else {
                    format!("{} ", t.visibility)
                };
                format!("{vis}{} {}", t.kind, t.name)
            })
            .unwrap_or_default(),
        "Trait" => traits
            .get(&span.owner)
            .map(|t| {
                let vis = if t.visibility.is_empty() {
                    String::new()
                } else {
                    format!("{} ", t.visibility)
                };
                format!("{vis}trait {}", t.name)
            })
            .unwrap_or_default(),
        _ => String::new(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    pub object_type: String,
    /// Whitespace-separated keywords. Each token is grepped against
    /// name, module_path, signature, and doc; per-token hits compose
    /// so a multi-keyword search bubbles entries that match more of
    /// them. Designed to be the first move when the operator
    /// describes intent and you don't yet know the canonical name.
    #[serde(default)]
    pub keywords: Option<String>,
    #[serde(default)]
    pub filters: Option<serde_json::Value>,
    #[serde(default)]
    pub include_links: Option<Vec<String>>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub object_type: String,
    pub results: Vec<serde_json::Value>,
    pub total_matches: usize,
    pub truncated: bool,
    /// Just-in-time guidance for the agent. Only populated when something
    /// in the request or result is worth flagging (zero matches, likely
    /// wrong object_type, truncation). Empty when the call is healthy.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hints: Vec<String>,
}

impl Ontology {
    pub fn query(&self, req: &QueryRequest) -> Result<QueryResponse> {
        let limit = req.limit.clamp(1, 50);
        let filters = parse_filters(req.filters.as_ref());
        let query = req.keywords.as_deref().unwrap_or("").to_lowercase();
        let links: Vec<String> = req.include_links.clone().unwrap_or_default();

        let mut hits: Vec<(f32, serde_json::Value)> = match req.object_type.as_str() {
            "Function" => self
                .functions
                .values()
                .filter(|f| pass_function_filter(f, &filters))
                .map(|f| (score_function(f, &query), self.render_function(f, &links)))
                .collect(),
            "Type" => self
                .types
                .values()
                .filter(|t| pass_type_filter(t, &filters))
                .map(|t| (score_type(t, &query), self.render_type(t, &links)))
                .collect(),
            "Trait" => self
                .traits
                .values()
                .filter(|t| pass_trait_filter(t, &filters))
                .map(|t| (score_trait(t, &query), self.render_trait(t, &links)))
                .collect(),
            "Module" => self
                .modules
                .values()
                .filter(|m| pass_module_filter(m, &filters))
                .map(|m| (score_module(m, &query), self.render_module(m, &links)))
                .collect(),
            "File" => self
                .files
                .values()
                .filter(|f| pass_file_filter(f, &filters))
                .map(|f| (self.score_file(f, &query), self.render_file(f, &query)))
                .collect(),
            other => return Err(anyhow!("unknown object_type: {other}")),
        };

        hits.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total = hits.len();
        let truncated = total > limit;
        let results: Vec<serde_json::Value> =
            hits.into_iter().take(limit).map(|(_, v)| v).collect();

        let hints = self.build_hints(req, total, truncated, &results);
        Ok(QueryResponse {
            object_type: req.object_type.clone(),
            results,
            total_matches: total,
            truncated,
            hints,
        })
    }

    fn build_hints(
        &self,
        req: &QueryRequest,
        total: usize,
        truncated: bool,
        results: &[serde_json::Value],
    ) -> Vec<String> {
        let mut hints = Vec::new();

        let path_filter_value = req
            .filters
            .as_ref()
            .and_then(|v| v.as_object())
            .and_then(|m| {
                m.get("module_path")
                    .or_else(|| m.get("path"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            });

        if total == 0 {
            if path_filter_value.is_some() {
                hints.push(
                    "No matches. Path filters accept three forms: exact \
                     ('crate::module'), suffix at module boundary ('module' \
                     matches 'crate::module'), or prefix at module boundary \
                     ('crate::module' matches descendants like \
                     'crate::module::Type::method'). Try the short name or a \
                     fully qualified path."
                        .into(),
                );
            } else if req.keywords.is_some() {
                hints.push(
                    "No matches. Try different keywords (synonyms, broader \
                     terms), drop filters, or switch object_type. If \
                     nothing produces results the entity may not exist."
                        .into(),
                );
            }
        }

        // The agent searched for something module-shaped via Function/Type/Trait.
        if req.object_type != "Module" {
            if let Some(q) = &req.keywords {
                let ql = q.to_lowercase();
                if ql.contains("module") || q.contains("::") {
                    hints.push(
                        "For questions scoped to a module, file, or area of \
                         code, query object_type=Module with the path filter \
                         and include_links=[functions, types, traits, \
                         submodules]. One call returns the directory-like view."
                            .into(),
                    );
                }
            }
        }

        // Module + functions link returned no functions, but methods exist on
        // types defined in that module. Common confusion: standalone fns vs
        // impl methods.
        if req.object_type == "Module" {
            let asked_functions = req
                .include_links
                .as_ref()
                .map(|v| v.iter().any(|s| s == "functions"))
                .unwrap_or(false);
            if asked_functions {
                if let Some(filter) = &path_filter_value {
                    let standalone_count = results
                        .iter()
                        .filter_map(|r| r.get("links"))
                        .filter_map(|l| l.get("functions"))
                        .filter_map(|f| f.as_array())
                        .map(|a| a.len())
                        .sum::<usize>();
                    let descendant_methods = self
                        .functions
                        .values()
                        .filter(|f| {
                            f.module_path != *filter
                                && f.module_path.starts_with(&format!("{filter}::"))
                        })
                        .count();
                    if standalone_count == 0 && descendant_methods > 0 {
                        hints.push(format!(
                            "This module has 0 standalone functions but {descendant_methods} \
                             methods on types defined in it. To see them, query \
                             object_type=Function with module_path={filter:?} \
                             (the prefix match includes descendants like \
                             '{filter}::TypeName::method')."
                        ));
                    }
                }
            }
        }

        if truncated {
            hints.push(
                "Result set was truncated. Add filters or refine the query \
                 to narrow to the entity you need."
                    .into(),
            );
        }

        // When File matches hit a signature (vs gap content), nudge the
        // agent toward the structural query for full context.
        if req.object_type == "File" {
            let sig_hits = results
                .iter()
                .filter_map(|r| r.get("matches"))
                .filter_map(|m| m.as_array())
                .flat_map(|a| a.iter())
                .filter(|m| m.get("scope").and_then(|v| v.as_str()) == Some("signature"))
                .count();
            if sig_hits > 0 {
                hints.push(format!(
                    "{sig_hits} match(es) hit indexed signatures. For full \
                     structural context (body, callers, callees, fields), \
                     re-query the listed owner via \
                     object_type=Function/Type/Trait. File responses do not \
                     include bodies of indexed items by design."
                ));
            }
        }

        hints
    }

    fn render_function(&self, f: &Function, links: &[String]) -> serde_json::Value {
        let mut v = serde_json::to_value(f).unwrap_or(serde_json::Value::Null);
        v["object_type"] = serde_json::Value::String("Function".into());
        v["dispatch_notes"] = serde_json::Value::Array(vec![]);
        if !links.is_empty() {
            v["links"] = self.function_links(f, links);
        }
        v
    }

    fn render_type(&self, t: &Type, links: &[String]) -> serde_json::Value {
        let mut v = serde_json::to_value(t).unwrap_or(serde_json::Value::Null);
        v["object_type"] = serde_json::Value::String("Type".into());
        if !links.is_empty() {
            v["links"] = self.type_links(t, links);
        }
        v
    }

    fn render_trait(&self, t: &Trait, links: &[String]) -> serde_json::Value {
        let mut v = serde_json::to_value(t).unwrap_or(serde_json::Value::Null);
        v["object_type"] = serde_json::Value::String("Trait".into());
        if !links.is_empty() {
            v["links"] = self.trait_links(t, links);
        }
        v
    }

    fn render_module(&self, m: &Module, links: &[String]) -> serde_json::Value {
        let mut v = serde_json::to_value(m).unwrap_or(serde_json::Value::Null);
        v["object_type"] = serde_json::Value::String("Module".into());
        if !links.is_empty() {
            v["links"] = self.module_links(m, links);
        }
        v
    }

    fn render_file(&self, f: &File, query: &str) -> serde_json::Value {
        let outline = compute_outline(f, &self.functions, &self.types, &self.traits);
        let mut v = serde_json::json!({
            "object_type": "File",
            "path": f.path,
            "extension": f.extension,
            "language": f.language,
            "bytes": f.bytes,
            "outline": outline,
        });
        // Free-text matches are scoped to GAP content and INDEXED signatures.
        // Indexed bodies are intentionally hidden: an agent that wants the
        // body must re-query as Function/Type/Trait. This keeps File from
        // becoming a back door around the graph.
        if !query.is_empty() {
            let q = query.to_lowercase();
            let mut matches: Vec<serde_json::Value> = Vec::new();
            self.collect_outline_matches(f, &q, &mut matches);
            if !matches.is_empty() {
                v["matches"] = serde_json::Value::Array(matches);
            }
        }
        v
    }

    fn collect_outline_matches(
        &self,
        f: &File,
        q: &str,
        out: &mut Vec<serde_json::Value>,
    ) {
        if f.content.is_empty() {
            return;
        }
        let lines: Vec<&str> = f.content.lines().collect();
        let mut cursor: usize = 1;
        let total_lines = lines.len();
        let emit = |m: serde_json::Value, out: &mut Vec<serde_json::Value>| {
            if out.len() < 50 {
                out.push(m);
            }
        };
        for span in &f.indexed_spans {
            // Search the gap region [cursor .. span.line_start - 1]
            if span.line_start > cursor {
                for ln in cursor..span.line_start {
                    if let Some(line) = lines.get(ln - 1) {
                        if line.to_lowercase().contains(q) {
                            emit(
                                serde_json::json!({
                                    "scope": "gap",
                                    "line": ln,
                                    "text": *line,
                                }),
                                out,
                            );
                        }
                    }
                }
            }
            // Search the indexed item's signature only (not the body).
            let sig =
                signature_for_outline(span, &self.functions, &self.types, &self.traits);
            if !sig.is_empty() && sig.to_lowercase().contains(q) {
                emit(
                    serde_json::json!({
                        "scope": "signature",
                        "line": span.line_start,
                        "owner": span.owner,
                        "owner_kind": span.kind,
                        "signature": sig,
                    }),
                    out,
                );
            }
            cursor = span.line_end + 1;
        }
        // Trailing gap.
        if cursor <= total_lines {
            for ln in cursor..=total_lines {
                if let Some(line) = lines.get(ln - 1) {
                    if line.to_lowercase().contains(q) {
                        emit(
                            serde_json::json!({
                                "scope": "gap",
                                "line": ln,
                                "text": *line,
                            }),
                            out,
                        );
                    }
                }
            }
        }
    }

    fn function_links(&self, f: &Function, requested: &[String]) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for link in requested {
            match link.as_str() {
                "callers" => {
                    let arr: Vec<_> = self
                        .functions
                        .values()
                        .filter(|g| g.callees.contains(&f.name))
                        .map(|g| shallow_function(g))
                        .collect();
                    out.insert("callers".into(), serde_json::Value::Array(arr));
                }
                "callees" => {
                    let mut seen = std::collections::HashSet::new();
                    let arr: Vec<_> = f
                        .callees
                        .iter()
                        .filter_map(|name| {
                            self.functions.values().find(|g| g.name == *name)
                        })
                        .filter(|g| seen.insert(g.name.clone()))
                        .map(|g| shallow_function(g))
                        .collect();
                    out.insert("callees".into(), serde_json::Value::Array(arr));
                }
                "tests" => {
                    let arr: Vec<_> = self
                        .functions
                        .values()
                        .filter(|g| g.is_test && g.callees.contains(&f.name))
                        .map(|g| shallow_function(g))
                        .collect();
                    out.insert("tests".into(), serde_json::Value::Array(arr));
                }
                "module" => {
                    if let Some(m) = self.modules.get(&f.module_path) {
                        out.insert(
                            "module".into(),
                            serde_json::to_value(m).unwrap_or(serde_json::Value::Null),
                        );
                    }
                }
                _ => {}
            }
        }
        serde_json::Value::Object(out)
    }

    fn type_links(&self, t: &Type, requested: &[String]) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for link in requested {
            match link.as_str() {
                "fields" => {
                    out.insert(
                        "fields".into(),
                        serde_json::to_value(&t.fields).unwrap_or(serde_json::Value::Null),
                    );
                }
                "impls" => {
                    let prefix = format!("{}::{}::", t.module_path, t.name);
                    let arr: Vec<_> = self
                        .functions
                        .values()
                        .filter(|f| f.module_path == format!("{}::{}", t.module_path, t.name)
                            || f.module_path.starts_with(&prefix))
                        .map(shallow_function)
                        .collect();
                    out.insert("impls".into(), serde_json::Value::Array(arr));
                }
                "used_by_functions" => {
                    let arr: Vec<_> = self
                        .functions
                        .values()
                        .filter(|f| f.signature.contains(&t.name) || f.body.contains(&t.name))
                        .map(shallow_function)
                        .collect();
                    out.insert("used_by_functions".into(), serde_json::Value::Array(arr));
                }
                "module" => {
                    if let Some(m) = self.modules.get(&t.module_path) {
                        out.insert(
                            "module".into(),
                            serde_json::to_value(m).unwrap_or(serde_json::Value::Null),
                        );
                    }
                }
                _ => {}
            }
        }
        serde_json::Value::Object(out)
    }

    fn trait_links(&self, t: &Trait, requested: &[String]) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for link in requested {
            match link.as_str() {
                "methods" => {
                    out.insert(
                        "methods".into(),
                        serde_json::Value::Array(
                            t.methods
                                .iter()
                                .map(|m| serde_json::Value::String(m.clone()))
                                .collect(),
                        ),
                    );
                }
                "implementors" => {
                    // Approximate: any type with an impl that matches the trait name
                    // by string. v0 doesn't track impl-trait edges precisely.
                    out.insert(
                        "implementors".into(),
                        serde_json::Value::String(
                            "v0: implementor lookup not indexed".into(),
                        ),
                    );
                }
                "module" => {
                    if let Some(m) = self.modules.get(&t.module_path) {
                        out.insert(
                            "module".into(),
                            serde_json::to_value(m).unwrap_or(serde_json::Value::Null),
                        );
                    }
                }
                _ => {}
            }
        }
        serde_json::Value::Object(out)
    }

    fn module_links(&self, m: &Module, requested: &[String]) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        for link in requested {
            match link.as_str() {
                "functions" => {
                    let arr: Vec<_> = m
                        .functions
                        .iter()
                        .filter_map(|n| self.functions.get(&format!("{}::{}", m.path, n)))
                        .map(shallow_function)
                        .collect();
                    out.insert("functions".into(), serde_json::Value::Array(arr));
                }
                "types" => {
                    let arr: Vec<_> = m
                        .types
                        .iter()
                        .filter_map(|n| self.types.get(&format!("{}::{}", m.path, n)))
                        .map(shallow_type)
                        .collect();
                    out.insert("types".into(), serde_json::Value::Array(arr));
                }
                "traits" => {
                    let arr: Vec<_> = m
                        .traits
                        .iter()
                        .filter_map(|n| self.traits.get(&format!("{}::{}", m.path, n)))
                        .map(shallow_trait)
                        .collect();
                    out.insert("traits".into(), serde_json::Value::Array(arr));
                }
                "submodules" => {
                    let arr: Vec<_> = m
                        .submodules
                        .iter()
                        .map(|n| serde_json::Value::String(format!("{}::{}", m.path, n)))
                        .collect();
                    out.insert("submodules".into(), serde_json::Value::Array(arr));
                }
                _ => {}
            }
        }
        serde_json::Value::Object(out)
    }
}

/// Compute the topology around a just-edited entity by name. Returns
/// (callers, tests) where:
///   - `callers` = every non-test function whose `callees` list mentions
///     `name`, as shallow refs the model can re-query to inspect.
///   - `tests` = test functions that touch this entity, either directly
///     (their body calls `name`) or one hop away (they call a function
///     that calls `name`). One hop catches the very common pattern where
///     a test calls a wrapper - e.g. tests call `authenticate` which
///     calls `verify_token`; a change to `verify_token` should flag those
///     tests so the model knows to verify them.
///
/// Dedup is by ontology key (`module_path::name`). Trait dispatch and
/// generic resolution remain approximate (callee detection only sees
/// last path segments); this is the same imprecision the rest of the
/// graph already carries.
pub(crate) fn compute_affected_topology(
    ont: &crate::ontology::model::Ontology,
    name: &str,
) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    use std::collections::HashSet;

    let direct_callers: Vec<&Function> = ont
        .functions
        .values()
        .filter(|f| f.callees.iter().any(|c| c == name))
        .collect();

    let mut tests: Vec<serde_json::Value> = Vec::new();
    let mut seen_test_keys: HashSet<String> = HashSet::new();

    // Direct: tests whose body calls `name`.
    for f in &direct_callers {
        if !f.is_test {
            continue;
        }
        let key = format!("{}::{}", f.module_path, f.name);
        if seen_test_keys.insert(key) {
            tests.push(shallow_function(f));
        }
    }

    // One-hop: tests whose body calls one of the direct (non-test) callers.
    let direct_caller_names: HashSet<String> = direct_callers
        .iter()
        .filter(|f| !f.is_test)
        .map(|f| f.name.clone())
        .collect();
    for f in ont.functions.values() {
        if !f.is_test {
            continue;
        }
        if f.callees.iter().any(|c| direct_caller_names.contains(c)) {
            let key = format!("{}::{}", f.module_path, f.name);
            if seen_test_keys.insert(key) {
                tests.push(shallow_function(f));
            }
        }
    }

    let callers: Vec<serde_json::Value> = direct_callers
        .iter()
        .filter(|f| !f.is_test)
        .map(|f| shallow_function(*f))
        .collect();

    (callers, tests)
}

pub(crate) fn shallow_function(f: &Function) -> serde_json::Value {
    serde_json::json!({
        "object_type": "Function",
        "name": f.name,
        "module_path": f.module_path,
        "signature": f.signature,
        "doc_summary": doc_summary(&f.doc),
    })
}

fn shallow_type(t: &Type) -> serde_json::Value {
    serde_json::json!({
        "object_type": "Type",
        "name": t.name,
        "module_path": t.module_path,
        "kind": t.kind,
        "doc_summary": doc_summary(&t.doc),
    })
}

fn shallow_trait(t: &Trait) -> serde_json::Value {
    serde_json::json!({
        "object_type": "Trait",
        "name": t.name,
        "module_path": t.module_path,
        "doc_summary": doc_summary(&t.doc),
    })
}

fn doc_summary(doc: &str) -> String {
    doc.lines().next().unwrap_or("").trim().to_string()
}

fn parse_filters(v: Option<&serde_json::Value>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Some(serde_json::Value::Object(map)) = v {
        for (k, val) in map {
            let s = match val {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            out.insert(k.clone(), s);
        }
    }
    out
}

/// Match a path filter. Accepts:
///   - exact match
///   - suffix at module boundary: filter `"math"` matches `"sandbox::math"`
///   - prefix at module boundary: filter `"sandbox::math"` matches any
///     descendant like `"sandbox::math::Point"` or `"sandbox::math::Point::distance"`.
/// Designed so that filtering Functions by `module_path: "sandbox::math"`
/// also returns methods on types defined in that module.
fn match_path(path: &str, filter: &str) -> bool {
    if path == filter {
        return true;
    }
    if let Some(rest) = path.strip_suffix(filter) {
        if rest.ends_with("::") {
            return true;
        }
    }
    if let Some(rest) = path.strip_prefix(filter) {
        if rest.starts_with("::") {
            return true;
        }
    }
    false
}

fn pass_function_filter(f: &Function, filters: &HashMap<String, String>) -> bool {
    for (k, v) in filters {
        let m = match k.as_str() {
            "name" => &f.name == v,
            "module_path" => match_path(&f.module_path, v),
            "visibility" => &f.visibility == v,
            "is_async" => f.is_async.to_string() == *v,
            "is_test" => f.is_test.to_string() == *v,
            _ => true,
        };
        if !m {
            return false;
        }
    }
    true
}

fn pass_type_filter(t: &Type, filters: &HashMap<String, String>) -> bool {
    for (k, v) in filters {
        let m = match k.as_str() {
            "name" => &t.name == v,
            "module_path" => match_path(&t.module_path, v),
            "kind" => &t.kind == v,
            _ => true,
        };
        if !m {
            return false;
        }
    }
    true
}

fn pass_trait_filter(t: &Trait, filters: &HashMap<String, String>) -> bool {
    for (k, v) in filters {
        let m = match k.as_str() {
            "name" => &t.name == v,
            "module_path" => match_path(&t.module_path, v),
            _ => true,
        };
        if !m {
            return false;
        }
    }
    true
}

fn pass_module_filter(m: &Module, filters: &HashMap<String, String>) -> bool {
    for (k, v) in filters {
        let ok = match k.as_str() {
            "path" => match_path(&m.path, v),
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}

fn pass_file_filter(f: &File, filters: &HashMap<String, String>) -> bool {
    for (k, v) in filters {
        let ok = match k.as_str() {
            "path" => f.path == *v || f.path.ends_with(v.as_str()),
            "extension" => f.extension == *v,
            "language" => f.language == *v,
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}

fn score_function(f: &Function, q: &str) -> f32 {
    if q.is_empty() {
        return 1.0;
    }
    let mut s = 0.0;
    let n = f.name.to_lowercase();
    let mp = f.module_path.to_lowercase();
    let sg = f.signature.to_lowercase();
    let dc = f.doc.to_lowercase();
    if n == q {
        s += 8.0;
    }
    if n.contains(q) {
        s += 3.0;
    }
    if mp.contains(q) {
        s += 1.5;
    }
    if sg.contains(q) {
        s += 1.0;
    }
    if dc.contains(q) {
        s += 2.0;
    }
    for tok in q.split_whitespace() {
        if n.contains(tok) {
            s += 0.5;
        }
        if mp.contains(tok) {
            s += 0.3;
        }
        if dc.contains(tok) {
            s += 0.4;
        }
    }
    // gentle centrality boost
    let caller_count = (f.callees.len() as f32).min(20.0) * 0.05;
    s + caller_count
}

fn score_type(t: &Type, q: &str) -> f32 {
    if q.is_empty() {
        return 1.0;
    }
    let mut s = 0.0;
    let n = t.name.to_lowercase();
    let mp = t.module_path.to_lowercase();
    let dc = t.doc.to_lowercase();
    if n == q {
        s += 8.0;
    }
    if n.contains(q) {
        s += 3.0;
    }
    if mp.contains(q) {
        s += 1.5;
    }
    if dc.contains(q) {
        s += 2.0;
    }
    for tok in q.split_whitespace() {
        if n.contains(tok) {
            s += 0.5;
        }
        if dc.contains(tok) {
            s += 0.4;
        }
    }
    s
}

fn score_trait(t: &Trait, q: &str) -> f32 {
    if q.is_empty() {
        return 1.0;
    }
    let mut s = 0.0;
    let n = t.name.to_lowercase();
    let dc = t.doc.to_lowercase();
    if n == q {
        s += 8.0;
    }
    if n.contains(q) {
        s += 3.0;
    }
    if dc.contains(q) {
        s += 2.0;
    }
    s
}

fn score_module(m: &Module, q: &str) -> f32 {
    if q.is_empty() {
        return 1.0;
    }
    let mut s = 0.0;
    if m.path.to_lowercase().contains(q) {
        s += 3.0;
    }
    s
}

impl Ontology {
    /// Free-text scoring for File results. Searches over GAP content and
    /// indexed-item SIGNATURES, never over indexed bodies, so File search
    /// can't be used as a back door around the graph.
    fn score_file(&self, f: &File, q: &str) -> f32 {
        if q.is_empty() {
            return match f.language.as_str() {
                "rust" => 3.0,
                "binary" => 0.5,
                _ => 1.5,
            };
        }
        let mut s = 0.0;
        let p = f.path.to_lowercase();
        if p == q {
            s += 8.0;
        } else if p.ends_with(q) {
            s += 4.0;
        } else if p.contains(q) {
            s += 2.5;
        }
        let searchable = self.build_searchable_text(f);
        if !searchable.is_empty() {
            let lc = searchable.to_lowercase();
            if lc.contains(q) {
                s += 2.0;
                let extra = lc.matches(q).count().min(20) as f32 * 0.1;
                s += extra;
            }
        }
        s
    }

    fn build_searchable_text(&self, f: &File) -> String {
        if f.content.is_empty() && f.indexed_spans.is_empty() {
            return String::new();
        }
        if f.indexed_spans.is_empty() {
            return f.content.clone();
        }
        let lines: Vec<&str> = f.content.lines().collect();
        let mut out = String::new();
        let mut cursor: usize = 1;
        for span in &f.indexed_spans {
            if span.line_start > cursor {
                for ln in cursor..span.line_start {
                    if let Some(line) = lines.get(ln - 1) {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
            // Signature is part of the searchable surface even though body
            // is hidden. Lets the agent find items by name via File search,
            // then drill into the structural query for body access.
            let sig = signature_for_outline(span, &self.functions, &self.types, &self.traits);
            if !sig.is_empty() {
                out.push_str(&sig);
                out.push('\n');
            }
            cursor = span.line_end + 1;
        }
        let total = lines.len();
        if cursor <= total {
            for ln in cursor..=total {
                if let Some(line) = lines.get(ln - 1) {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        out
    }
}
