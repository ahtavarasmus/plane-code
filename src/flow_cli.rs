//! Glue between the CFG renderer and the operator's browser.
//!
//! Builds CFGs for *every* indexed function in the workspace, resolves
//! each call node's callee against the ontology to discover which other
//! functions it points at, and bundles everything into a single
//! self-contained HTML page. The page lets the operator:
//!
//!   - Click any node to highlight the matching source range in the
//!     side panel (Phase C).
//!   - Click a Call node whose callee is itself an indexed function to
//!     navigate to that function's CFG, with a breadcrumb stack and a
//!     back button (Phase D).
//!   - Pan / zoom the SVG diagram via svg-pan-zoom from CDN.

use crate::flow;
use crate::ontology::{model::Function, Ontology};
use anyhow::Result;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

/// Synthetic bundle key for the workspace skyline view. Distinguished
/// from real ontology keys (which never start with `@`) so both views
/// share one bundle.
const SKYLINE_KEY: &str = "@skyline";

pub fn open_flow(ontology: &Ontology, target: &str) -> Result<PathBuf> {
    let entry = resolve_target(ontology, target)?;
    let entry_key = format!("{}::{}", entry.module_path, entry.name);
    let mut bundle = build_bundle(ontology)?;
    inject_skyline(&mut bundle, ontology);
    let html = render_html(&entry_key, &bundle);

    let safe = entry_key.replace("::", "_");
    let path = std::env::temp_dir().join(format!("planecode-flow-{safe}.html"));
    std::fs::write(&path, &html)?;
    let _ = std::process::Command::new("open").arg(&path).spawn();
    Ok(path)
}

/// Skyline: workspace-level call graph rendered hierarchically so it
/// scales to real codebases. Top level is modules + inter-module
/// edges (typically 10s-100s of nodes - fits mermaid comfortably).
/// Click a module -> inline expansion to its functions and
/// intra-module edges. Click a function -> its CFG. All recursive,
/// all using the same expansion infrastructure.
pub fn open_skyline(ontology: &Ontology) -> Result<PathBuf> {
    let mut bundle = build_bundle(ontology)?;
    inject_skyline(&mut bundle, ontology);
    let html = render_html(SKYLINE_KEY, &bundle);

    let crate_name = if ontology.crate_name.is_empty() {
        "workspace".to_string()
    } else {
        ontology.crate_name.clone()
    };
    let path = std::env::temp_dir().join(format!("planecode-skyline-{crate_name}.html"));
    std::fs::write(&path, &html)?;
    let _ = std::process::Command::new("open").arg(&path).spawn();
    Ok(path)
}

/// Open the skyline view scrolled to a specific node. `focus` may be
/// a function name (`verify_token`), fully-qualified function path
/// (`backend::auth::verify_token`), or a module path (full or trailing
/// suffix - `backend::services` matches `lightfriend::backend::services`).
/// Falls back to a clear error if `focus` matches neither a function
/// nor a module the bundle knows about.
pub fn open_skyline_at(ontology: &Ontology, focus: &str) -> Result<PathBuf> {
    let mut bundle = build_bundle(ontology)?;
    inject_skyline(&mut bundle, ontology);

    let entry_key = resolve_skyline_focus(&bundle, ontology, focus)?;
    let html = render_html(&entry_key, &bundle);

    let safe = entry_key
        .replace("::", "_")
        .replace('@', "")
        .replace(':', "_");
    let path = std::env::temp_dir().join(format!("planecode-skyline-{safe}.html"));
    std::fs::write(&path, &html)?;
    let _ = std::process::Command::new("open").arg(&path).spawn();
    Ok(path)
}

fn resolve_skyline_focus(
    bundle: &HashMap<String, FunctionPayload>,
    ontology: &Ontology,
    focus: &str,
) -> Result<String> {
    // Functions take precedence: a name like `load` is far more
    // likely to be a function than a module. resolve_target gives
    // a precise multi-match error if the function name is ambiguous.
    if let Ok(f) = resolve_target(ontology, focus) {
        let key = format!("{}::{}", f.module_path, f.name);
        if bundle.contains_key(&key) {
            return Ok(key);
        }
    }

    // Module match. Exact path first, then suffix (so the user can
    // pass `services` instead of typing the full crate prefix).
    let module_paths: Vec<String> = bundle
        .keys()
        .filter_map(|k| k.strip_prefix("@module:").map(|s| s.to_string()))
        .collect();

    if module_paths.iter().any(|m| m == focus) {
        return Ok(format!("@module:{focus}"));
    }

    // Segment match: `focus` appearing as a complete `::`-bounded
    // segment anywhere in the path. So `blog` matches both
    // `crate::blog` (suffix) and `crate::blog::posts` (middle).
    // Picking the shallowest tie-breaks toward the parent module,
    // which is usually what the user wants when they type a short
    // name.
    let seg_in_middle = format!("::{focus}::");
    let seg_at_end = format!("::{focus}");
    let seg_at_start = format!("{focus}::");
    let mut matches: Vec<&String> = module_paths
        .iter()
        .filter(|m| {
            m.contains(&seg_in_middle)
                || m.ends_with(&seg_at_end)
                || m.starts_with(&seg_at_start)
        })
        .collect();
    if matches.is_empty() {
        anyhow::bail!(
            "no function or module matches `{focus}`. Pass a function \
             name (`verify_token`), a fully-qualified path \
             (`backend::auth::verify_token`), or a module path \
             (`backend::services`)."
        );
    }
    matches.sort_by_key(|m| m.matches("::").count());
    let min_depth = matches[0].matches("::").count();
    let shallowest: Vec<&&String> = matches
        .iter()
        .take_while(|m| m.matches("::").count() == min_depth)
        .collect();
    if shallowest.len() == 1 {
        return Ok(format!("@module:{}", shallowest[0]));
    }
    let names: Vec<&str> = shallowest.iter().map(|s| s.as_str()).collect();
    anyhow::bail!(
        "module `{focus}` is ambiguous at depth {min_depth}; matches: {}",
        names.join(", ")
    )
}

fn inject_skyline(
    bundle: &mut HashMap<String, FunctionPayload>,
    ontology: &Ontology,
) {
    // Map every function to its containing indexed Module (the
    // deepest Module entity whose path is a prefix of the function's
    // module_path). Methods on impl blocks have module_path like
    // `crate::math::Point`, but their containing Module is `crate::math`.
    let fn_to_module = compute_fn_to_module(ontology);

    // Top-level skyline: modules + inter-module edges.
    let payload = build_module_skyline_payload(ontology, &fn_to_module);
    bundle.insert(SKYLINE_KEY.to_string(), payload);

    // One submap per module: functions in that module + intra-module
    // edges. Synthetic key `@module:<path>` lets the JS click-handler
    // expand a module node inline via the existing callee_key flow.
    let mut by_module: BTreeMap<String, Vec<&Function>> = BTreeMap::new();
    for (key, f) in &ontology.functions {
        if let Some(m) = fn_to_module.get(key) {
            by_module.entry(m.clone()).or_default().push(f);
        }
    }
    for (module_path, fns) in &by_module {
        let submap = build_module_submap_payload(
            module_path,
            fns,
            ontology,
            &fn_to_module,
        );
        bundle.insert(module_payload_key(module_path), submap);
    }
}

fn module_payload_key(module_path: &str) -> String {
    format!("@module:{module_path}")
}

/// For each indexed function, return the path of its deepest
/// containing indexed Module. `sandbox::math::Point::new` ->
/// `sandbox::math` (when `sandbox::math::Point` isn't an indexed
/// Module). Functions with no indexed-module ancestor (e.g. ones
/// outside src/) get the crate root.
fn compute_fn_to_module(ontology: &Ontology) -> HashMap<String, String> {
    let module_paths: HashSet<String> = ontology.modules.keys().cloned().collect();
    let crate_name = if ontology.crate_name.is_empty() {
        "crate".to_string()
    } else {
        ontology.crate_name.clone()
    };
    let mut out = HashMap::new();
    for (key, f) in &ontology.functions {
        let mut candidate = f.module_path.clone();
        // Walk up the module path until we hit an indexed Module.
        let chosen = loop {
            if module_paths.contains(&candidate) {
                break candidate;
            }
            match candidate.rfind("::") {
                Some(i) => candidate.truncate(i),
                None => break crate_name.clone(),
            }
        };
        out.insert(key.clone(), chosen);
    }
    out
}

fn build_module_skyline_payload(
    ontology: &Ontology,
    fn_to_module: &HashMap<String, String>,
) -> FunctionPayload {
    // Stable IDs per module.
    let mut module_set: BTreeSet<String> = BTreeSet::new();
    for m in fn_to_module.values() {
        module_set.insert(m.clone());
    }
    let modules: Vec<String> = module_set.into_iter().collect();
    let mut id_for: HashMap<String, String> = HashMap::new();
    for (i, m) in modules.iter().enumerate() {
        id_for.insert(m.clone(), format!("m{i}"));
    }

    // name -> function keys, for resolving callees (as in the flat
    // skyline). Used to bucket fn-level call edges into module pairs.
    let mut name_to_keys: HashMap<String, Vec<String>> = HashMap::new();
    for (key, f) in &ontology.functions {
        name_to_keys
            .entry(f.name.clone())
            .or_default()
            .push(key.clone());
    }

    // Fn count per module so the label can report scale at a glance.
    let mut fn_count: HashMap<String, usize> = HashMap::new();
    for m in fn_to_module.values() {
        *fn_count.entry(m.clone()).or_insert(0) += 1;
    }

    // Aggregate inter-module edges: for every fn-level callee
    // resolution that crosses module boundaries, emit one edge
    // module_from -> module_to (deduped).
    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    for (from_key, f) in &ontology.functions {
        let from_mod = match fn_to_module.get(from_key) {
            Some(m) => m,
            None => continue,
        };
        for callee_name in &f.callees {
            if let Some(matches) = name_to_keys.get(callee_name) {
                if matches.len() != 1 {
                    continue;
                }
                let to_key = &matches[0];
                if to_key == from_key {
                    continue;
                }
                let to_mod = match fn_to_module.get(to_key) {
                    Some(m) => m,
                    None => continue,
                };
                if from_mod == to_mod {
                    continue;
                }
                let from_id = id_for.get(from_mod).cloned().unwrap_or_default();
                let to_id = id_for.get(to_mod).cloned().unwrap_or_default();
                if !from_id.is_empty() && !to_id.is_empty() {
                    edge_set.insert((from_id, to_id));
                }
            }
        }
    }

    let mut graph_nodes: Vec<GraphNode> = Vec::new();
    let mut node_meta: HashMap<String, NodeMeta> = HashMap::new();
    let mut source = String::new();
    use std::fmt::Write;
    let mut line: usize = 1;
    for m in &modules {
        let id = id_for.get(m).expect("id assigned").clone();
        let count = fn_count.get(m).copied().unwrap_or(0);
        let label = format!("{m}  ({count})");
        graph_nodes.push(GraphNode {
            id: id.clone(),
            kind: "SkylineFn".into(),
            label,
            css_class: Some("modulenode".into()),
        });
        node_meta.insert(
            id,
            NodeMeta {
                line_start: Some(line),
                line_end: Some(line),
                callee_key: Some(module_payload_key(m)),
                kind: "call",
            },
        );
        let _ = writeln!(source, "{m} - {count} fn(s)");
        line += 1;
    }
    let graph_edges: Vec<GraphEdge> = edge_set
        .into_iter()
        .map(|(from, to)| GraphEdge {
            from,
            to,
            label: None,
        })
        .collect();

    let crate_name = if ontology.crate_name.is_empty() {
        "workspace".to_string()
    } else {
        ontology.crate_name.clone()
    };
    let signature = format!(
        "{} module(s) · {} function(s) total · click a module to expand its functions",
        modules.len(),
        ontology.functions.len()
    );

    FunctionPayload {
        label: format!("workspace: {crate_name}"),
        signature,
        body: source,
        // The pre-rendered mermaid is empty - the JS composer will
        // build mermaid from `graph` on render. We keep the field for
        // bundle-shape consistency.
        mermaid: String::new(),
        graph: GraphData {
            nodes: graph_nodes,
            edges: graph_edges,
            groups: Vec::new(),
        },
        node_meta,
    }
}

fn build_module_submap_payload(
    module_path: &str,
    fns: &[&Function],
    ontology: &Ontology,
    fn_to_module: &HashMap<String, String>,
) -> FunctionPayload {
    // Build name resolver across the WHOLE workspace; we still want
    // intra-module edges that resolve to functions in the same module
    // (and we'll skip the rest).
    let mut name_to_keys: HashMap<String, Vec<String>> = HashMap::new();
    for (key, f) in &ontology.functions {
        name_to_keys
            .entry(f.name.clone())
            .or_default()
            .push(key.clone());
    }

    let mut sorted = fns.to_vec();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut id_for: HashMap<String, String> = HashMap::new();
    for (i, f) in sorted.iter().enumerate() {
        let key = format!("{}::{}", f.module_path, f.name);
        id_for.insert(key, format!("fn{i}"));
    }

    let mut graph_nodes: Vec<GraphNode> = Vec::new();
    let mut node_meta: HashMap<String, NodeMeta> = HashMap::new();
    let mut source = String::new();
    use std::fmt::Write;
    let mut line: usize = 1;
    for f in &sorted {
        let key = format!("{}::{}", f.module_path, f.name);
        let id = id_for.get(&key).expect("id assigned").clone();
        let css_class = if f.is_test {
            "testfn"
        } else if f.visibility == "pub" {
            "pubfn"
        } else {
            "privfn"
        };
        graph_nodes.push(GraphNode {
            id: id.clone(),
            kind: "SkylineFn".into(),
            label: f.name.clone(),
            css_class: Some(css_class.into()),
        });
        node_meta.insert(
            id,
            NodeMeta {
                line_start: Some(line),
                line_end: Some(line),
                callee_key: Some(key.clone()),
                kind: "call",
            },
        );
        let _ = writeln!(source, "{} {}", f.name, f.signature);
        line += 1;
    }

    // Intra-module edges only. Cross-module edges are visible at the
    // top level (between module nodes); duplicating them here would
    // clutter without informing.
    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    for f in &sorted {
        let from_key = format!("{}::{}", f.module_path, f.name);
        let from_id = match id_for.get(&from_key) {
            Some(id) => id.clone(),
            None => continue,
        };
        for callee_name in &f.callees {
            if let Some(matches) = name_to_keys.get(callee_name) {
                if matches.len() != 1 {
                    continue;
                }
                let to_key = &matches[0];
                if to_key == &from_key {
                    continue;
                }
                let to_mod = fn_to_module.get(to_key);
                if to_mod.map(|m| m.as_str()) != Some(module_path) {
                    continue;
                }
                if let Some(to_id) = id_for.get(to_key).cloned() {
                    edge_set.insert((from_id.clone(), to_id));
                }
            }
        }
    }
    let graph_edges: Vec<GraphEdge> = edge_set
        .into_iter()
        .map(|(from, to)| GraphEdge {
            from,
            to,
            label: None,
        })
        .collect();

    FunctionPayload {
        label: format!("module: {module_path}"),
        signature: format!(
            "{} fn(s) · click any to expand its CFG inline",
            sorted.len()
        ),
        body: source,
        mermaid: String::new(),
        graph: GraphData {
            nodes: graph_nodes,
            edges: graph_edges,
            groups: Vec::new(),
        },
        node_meta,
    }
}


fn resolve_target<'a>(ontology: &'a Ontology, target: &str) -> Result<&'a Function> {
    let candidates: Vec<&Function> = ontology
        .functions
        .iter()
        .filter_map(|(key, f)| {
            if key == target || f.name == target {
                Some(f)
            } else {
                None
            }
        })
        .collect();
    if candidates.is_empty() {
        anyhow::bail!(
            "no function named `{target}` in the ontology. Pass either a \
             short name (`verify_token`) or a fully-qualified path \
             (`sandbox::auth::verify_token`)."
        );
    }
    if candidates.len() > 1 {
        let names: Vec<String> = candidates
            .iter()
            .map(|f| format!("{}::{}", f.module_path, f.name))
            .collect();
        anyhow::bail!(
            "function `{target}` is ambiguous; matches {} entities. Use \
             a fully-qualified path: {}",
            candidates.len(),
            names.join(", ")
        );
    }
    Ok(candidates[0])
}

/// Per-function payload sent to the browser. `graph` is the structured
/// representation the JS composer uses to emit mermaid (with optional
/// inline expansions); `mermaid` is kept as a fallback / debug aid.
/// `node_meta` maps each `n<i>` id to either a source range (for
/// highlighting) or a callee key (for navigation), or both.
#[derive(serde::Serialize)]
struct FunctionPayload {
    label: String,
    signature: String,
    body: String,
    mermaid: String,
    graph: GraphData,
    node_meta: HashMap<String, NodeMeta>,
}

/// Structured graph data. Nodes and edges describe the topology;
/// `groups` describe optional subgraph groupings (e.g. modules in the
/// skyline view). The browser composer reads this to emit mermaid,
/// rewriting node IDs with prefixes and replacing nodes with nested
/// subgraphs whenever the operator has expanded a callsite.
#[derive(serde::Serialize, Default)]
struct GraphData {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    groups: Vec<NodeGroup>,
}

#[derive(serde::Serialize)]
struct GraphNode {
    id: String,
    /// Visual category - drives shape and color in mermaid:
    /// "Entry" | "Exit" | "Statement" | "Call" | "Branch" | "Return"
    /// | "LoopHeader" | "LoopJump" | "SkylineFn"
    kind: String,
    label: String,
    /// CSS class name (matches a classDef in the rendered mermaid).
    css_class: Option<String>,
}

#[derive(serde::Serialize)]
struct GraphEdge {
    from: String,
    to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
}

/// A subgraph grouping (mermaid `subgraph ... end`). Used by the
/// skyline to group functions by module; per-function CFGs leave this
/// empty.
#[derive(serde::Serialize)]
struct NodeGroup {
    id: String,
    label: String,
    node_ids: Vec<String>,
}

#[derive(serde::Serialize, Default)]
struct NodeMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    line_start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_end: Option<usize>,
    /// Resolved callee key (`module::path::name`) when the call lands
    /// in another indexed function. `None` for stdlib / external /
    /// unresolvable calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    callee_key: Option<String>,
    /// Convenience kind tag the JS uses to decide the click behavior.
    kind: &'static str,
}

fn build_bundle(ontology: &Ontology) -> Result<HashMap<String, FunctionPayload>> {
    // Index function names -> ontology keys for callee resolution.
    // A bare name like "find_port" might match multiple functions in
    // different modules; we only resolve when there's exactly one match
    // so navigation is unambiguous.
    let mut name_to_keys: HashMap<String, Vec<String>> = HashMap::new();
    for (key, f) in &ontology.functions {
        name_to_keys
            .entry(f.name.clone())
            .or_default()
            .push(key.clone());
    }

    let mut bundle = HashMap::new();
    for (key, f) in &ontology.functions {
        let cfg = match flow::build_cfg_from_body_source(&f.body, key) {
            Ok(c) => c,
            Err(_) => continue, // body didn't parse; skip
        };
        let mermaid = flow::to_mermaid_with_clicks(&cfg);

        let mut node_meta: HashMap<String, NodeMeta> = HashMap::new();
        let mut graph_nodes: Vec<GraphNode> = Vec::new();
        for n in &cfg.nodes {
            let id = format!("n{}", n.id);
            let kind_str: &'static str = match &n.kind {
                flow::CfgNodeKind::Entry => "Entry",
                flow::CfgNodeKind::Exit => "Exit",
                flow::CfgNodeKind::Statement => "Statement",
                flow::CfgNodeKind::Call { .. } => "Call",
                flow::CfgNodeKind::Branch { .. } => "Branch",
                flow::CfgNodeKind::Return => "Return",
                flow::CfgNodeKind::LoopHeader => "LoopHeader",
                flow::CfgNodeKind::LoopJump { .. } => "LoopJump",
            };
            let meta_kind = match &n.kind {
                flow::CfgNodeKind::Entry => "entry",
                flow::CfgNodeKind::Exit => "exit",
                flow::CfgNodeKind::Statement => "statement",
                flow::CfgNodeKind::Call { .. } => "call",
                flow::CfgNodeKind::Branch { .. } => "branch",
                flow::CfgNodeKind::Return => "return",
                flow::CfgNodeKind::LoopHeader => "loop",
                flow::CfgNodeKind::LoopJump { .. } => "jump",
            };
            let css_class = match &n.kind {
                flow::CfgNodeKind::Entry | flow::CfgNodeKind::Exit => Some("terminal"),
                flow::CfgNodeKind::Return => Some("returns"),
                flow::CfgNodeKind::Branch { .. } => Some("branch"),
                flow::CfgNodeKind::Call { .. } => Some("calls"),
                flow::CfgNodeKind::LoopHeader => Some("loops"),
                flow::CfgNodeKind::LoopJump { .. } => Some("jumps"),
                flow::CfgNodeKind::Statement => None,
            };
            let mut meta = NodeMeta {
                line_start: n.line_start,
                line_end: n.line_end,
                callee_key: None,
                kind: meta_kind,
            };
            if let flow::CfgNodeKind::Call { callee } = &n.kind {
                if let Some(matches) = name_to_keys.get(callee) {
                    if matches.len() == 1 {
                        meta.callee_key = Some(matches[0].clone());
                    }
                }
            }
            graph_nodes.push(GraphNode {
                id: id.clone(),
                kind: kind_str.to_string(),
                label: n.label.clone(),
                css_class: css_class.map(str::to_string),
            });
            node_meta.insert(id, meta);
        }
        let graph_edges: Vec<GraphEdge> = cfg
            .edges
            .iter()
            .map(|e| GraphEdge {
                from: format!("n{}", e.from),
                to: format!("n{}", e.to),
                label: e.label.clone(),
            })
            .collect();

        // Translate loop_groups into NodeGroups the JS composer renders
        // as mermaid `subgraph` blocks. With each loop body wrapped, the
        // layout engine treats it as a single "blob" off the spine, so
        // back-edges curve around the blob instead of dragging the
        // header below the body.
        let groups: Vec<NodeGroup> = cfg
            .loop_groups
            .iter()
            .enumerate()
            .map(|(i, lg)| {
                // Empty single-space label: mermaid still parses the
                // subgraph, but renders no visible header. The
                // subgraph is purely a layout container for us, so a
                // header would just be visual noise. The JS strip
                // pass also removes the label group entirely as a
                // belt-and-suspenders fallback in case some mermaid
                // version still draws the empty space.
                let _ = lg.header; // not displayed - kept for future use
                NodeGroup {
                    id: format!("loop_g{i}"),
                    label: " ".into(),
                    node_ids: lg.body_nodes.iter().map(|n| format!("n{n}")).collect(),
                }
            })
            .collect();

        bundle.insert(
            key.clone(),
            FunctionPayload {
                label: key.clone(),
                signature: f.signature.clone(),
                body: f.body.clone(),
                mermaid,
                graph: GraphData {
                    nodes: graph_nodes,
                    edges: graph_edges,
                    groups,
                },
                node_meta,
            },
        );
    }
    Ok(bundle)
}

fn truncate_label(s: &str, max: usize) -> String {
    let cleaned: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() <= max {
        cleaned
    } else {
        let truncated: String = cleaned.chars().take(max.saturating_sub(1)).collect();
        format!("{truncated}…")
    }
}

fn render_html(entry_key: &str, bundle: &HashMap<String, FunctionPayload>) -> String {
    let bundle_json = make_script_safe(
        &serde_json::to_string(&bundle).unwrap_or_else(|_| "{}".into()),
    );
    let entry_json = make_script_safe(&json!(entry_key).to_string());

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>planecode flow</title>
<style>
  :root {{
    --bg:        #0d1117;
    --panel:     #11161e;
    --border:    #21262d;
    --fg:        #c9d1d9;
    --muted:     #6e7681;
    --accent:    #58a6ff;
    --highlight: rgba(255, 200, 0, 0.18);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0; height: 100%;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow: hidden;
  }}
  #app {{
    display: grid;
    grid-template-columns: 1fr 480px;
    grid-template-rows: auto 1fr;
    height: 100vh;
  }}
  header {{
    grid-column: 1 / -1;
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }}
  header .title {{
    color: var(--accent);
    font-size: 14px;
    font-weight: 600;
    margin: 0;
  }}
  header .crumbs {{
    color: var(--muted);
    font-size: 12px;
    font-family: ui-monospace, SFMono-Regular, monospace;
  }}
  header .crumbs span.crumb {{
    cursor: pointer;
  }}
  header .crumbs span.crumb:hover {{
    color: var(--accent);
    text-decoration: underline;
  }}
  header .back {{
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--fg);
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }}
  header .back:hover {{
    border-color: var(--accent);
  }}
  header .back:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
  }}
  #graph {{
    overflow: hidden;
    position: relative;
    min-height: 0;
  }}
  #diagram {{
    width: 100%;
    height: 100%;
    display: block;
  }}
  #diagram svg {{
    width: 100% !important;
    height: 100% !important;
    display: block;
  }}
  #graph .zoom-hint {{
    position: absolute;
    bottom: 12px; right: 16px;
    color: var(--muted);
    font-size: 11px;
    pointer-events: none;
  }}
  #side {{
    border-left: 1px solid var(--border);
    background: var(--panel);
    overflow: auto;
  }}
  #side .signature {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    word-break: break-word;
  }}
  #side .source {{
    padding: 12px 0;
  }}
  #side .source pre {{
    margin: 0;
    padding: 0;
    counter-reset: line;
  }}
  #side .source .line {{
    display: grid;
    grid-template-columns: 48px 1fr;
    padding: 0;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    line-height: 1.55;
  }}
  #side .source .line .ln {{
    color: var(--muted);
    text-align: right;
    padding-right: 12px;
    user-select: none;
  }}
  #side .source .line .code {{
    white-space: pre;
    padding-right: 16px;
  }}
  #side .source .line.highlighted {{
    background: var(--highlight);
  }}
  /* Default cursor: nodes show details on hover, no implicit click. */
  #diagram svg g.node {{
    cursor: default;
    transition: filter 120ms ease;
  }}
  #diagram svg g.node:hover {{
    filter: brightness(1.18);
  }}
  /* Navigable nodes: persistent blue glow + pointer cursor so the
     operator can tell at a glance which nodes click into another
     function. Stronger glow on hover to confirm the affordance. */
  #diagram svg g.node.navigable {{
    cursor: pointer;
    filter: drop-shadow(0 0 4px rgba(88, 166, 255, 0.55));
  }}
  #diagram svg g.node.navigable:hover {{
    filter: brightness(1.35) drop-shadow(0 0 10px rgba(88, 166, 255, 0.95));
  }}
  /* The shape outline goes accent-blue too, so even with the filter
     turned off (e.g. a colorblind-mode toggle later) the navigability
     stays readable. */
  #diagram svg g.node.navigable > rect,
  #diagram svg g.node.navigable > polygon,
  #diagram svg g.node.navigable > path {{
    stroke: #58a6ff !important;
    stroke-width: 2.25px !important;
  }}
  /* Externally-driven highlight: applied when the operator hovers a
     source line in the side panel. Distinct color (warm) so it reads
     differently from the navigable-call hover effect. */
  #diagram svg g.node.cfg-highlighted {{
    filter: brightness(1.45) drop-shadow(0 0 6px rgba(255, 200, 0, 0.75));
  }}
  #side .source .line:hover {{
    background: rgba(255, 255, 255, 0.04);
  }}
  /* Expanded subgraph headers are clickable to collapse. The label
     starts with `× ` to signal the affordance, plus we tint it red
     on hover so it reads as a destructive "close" action. */
  #diagram svg g.expanded-cluster-header {{
    cursor: pointer;
  }}
  #diagram svg g.expanded-cluster-header text,
  #diagram svg g.expanded-cluster-header span {{
    transition: fill 120ms ease, color 120ms ease;
  }}
  #diagram svg g.expanded-cluster-header:hover text {{
    fill: #ff7b72 !important;
  }}
  #diagram svg g.expanded-cluster-header:hover span {{
    color: #ff7b72 !important;
  }}
  /* Subgraphs are a layout device for us, not a visual element - we
     wrap loop bodies in clusters so ELK keeps them contained, but
     the gray rect mermaid stamps for them just clutters the diagram.
     Strip it: transparent fill, no stroke. The label still renders
     at the top for legibility ("body of for f in ..."). Match every
     shape mermaid might use for the cluster background (`rect` for
     dagre, `path` for elk). */
  #diagram svg g.subgraph > rect,
  #diagram svg g.subgraph rect.outer,
  #diagram svg g.subgraph > path {{
    fill: transparent !important;
    stroke: none !important;
    stroke-width: 0 !important;
  }}
  /* Cluster label - small, dim, italic. Useful as a "this is a
     loop body" hint but doesn't compete with the actual nodes.
     Mermaid v10 elk renders the label inside `g.label` (not the
     dagre-era `g.cluster-label`), and the actual text sits in a
     <foreignObject><span> or in nested <text> tspans. Match all. */
  #diagram svg g.subgraph > g.label text,
  #diagram svg g.subgraph > g.label tspan,
  #diagram svg g.subgraph > g.label span,
  #diagram svg g.subgraph > g.label foreignObject * {{
    fill: var(--muted) !important;
    color: var(--muted) !important;
    font-size: 11px !important;
    font-style: italic;
    opacity: 0.75;
  }}
  /* Expanded inline subgraphs get a blue accent so nesting reads
     visually as "I am a function expansion" rather than just another
     gray box. */
  #diagram svg g.subgraph.expanded-cluster > rect {{
    fill: rgba(88, 166, 255, 0.045) !important;
    stroke: rgba(88, 166, 255, 0.45) !important;
    stroke-width: 1.5px !important;
  }}
  /* Expanded-cluster labels (inline function expansions) override
     the loop-body styling above: brighter, weight bump, no italic,
     since these subgraphs DO have a real visual presence. */
  #diagram svg g.subgraph.expanded-cluster > g.label text,
  #diagram svg g.subgraph.expanded-cluster > g.label tspan,
  #diagram svg g.subgraph.expanded-cluster > g.label span,
  #diagram svg g.subgraph.expanded-cluster > g.label foreignObject * {{
    fill: var(--fg) !important;
    color: var(--fg) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    font-style: normal !important;
    opacity: 1 !important;
    letter-spacing: 0.2px;
  }}
  /* Round the leaf nodes too - mermaid's default sharp corners
     compound the boxiness. The :not() guards on shapes that already
     have curved geometry. */
  #diagram svg g.node > rect {{
    rx: 6 !important;
    ry: 6 !important;
  }}
</style>
</head>
<body>
<div id="app">
  <header>
    <button class="back" id="backBtn" type="button">&larr; Back</button>
    <h1 class="title" id="titleEl">loading…</h1>
    <span class="crumbs" id="crumbs"></span>
  </header>
  <div id="graph">
    <div id="diagram"></div>
    <div class="zoom-hint">drag to pan · scroll to zoom</div>
  </div>
  <aside id="side">
    <div class="signature" id="sigEl"></div>
    <div class="source" id="sourceEl"></div>
  </aside>
</div>

<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script>
  const BUNDLE = {bundle_json};
  const ENTRY = {entry_json};
  let panZoomInstance = null;
  /// Back-edges removed from the mermaid source (so ELK sees a clean
  /// DAG and places loop bodies BELOW their headers); we draw them
  /// manually as SVG curves after each render. Each entry is
  /// `{{ from: prefixed_id, to: prefixed_id, label: 'loop' | 'continue' }}`.
  let pendingBackEdges = [];
  /// Map of fully-prefixed node id (e.g. `root_n5` or
  /// `root_n5_n0`) -> the rendered SVG `<g>` element for the currently
  /// rendered tree. Rebuilt on every render. Used for hover-highlights
  /// in both directions.
  let currentNodeMap = {{}};
  /// Map of fully-prefixed node id -> the function key whose CFG
  /// scope owns that node. Lets click handlers find the right
  /// expansion tree slot when toggling.
  let currentScopeOf = {{}};
  /// The expansion tree. Root holds the entry function; each
  /// `expansions` entry maps a node id (within the parent's scope) to
  /// a child tree node carrying the expanded function's key and its
  /// own further expansions. `body` of `current` = source shown in the
  /// side panel (the deepest currently-focused function's source).
  let tree = {{ funcKey: ENTRY, expansions: {{}} }};
  /// The currently-focused function key (drives the source panel).
  /// Updated when the operator clicks an expanded subgraph header,
  /// hovers a node, or initially when the page loads.
  let focusedKey = ENTRY;

  mermaid.initialize({{
    startOnLoad: false,
    theme: 'dark',
    flowchart: {{ curve: 'basis', useMaxWidth: false, htmlLabels: false }},
    securityLevel: 'loose', // allow click directives in mermaid source
    // Real codebases blow past mermaid's defaults: a single inline
    // expansion of a 100-fn module can produce thousands of edges.
    // Bump both ceilings so the renderer doesn't refuse to draw.
    maxEdges: 50000,
    maxTextSize: 500000,
  }});

  function escapeHtml(s) {{
    return s.replace(/[&<>"']/g, c => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }})[c]);
  }}

  function renderSourcePanel(payload) {{
    document.getElementById('sigEl').textContent = payload.signature || '';
    const lines = payload.body.split('\n');
    const html = lines.map((line, i) => {{
      const n = i + 1;
      return `<div class="line" id="line-${{n}}" data-line="${{n}}"><span class="ln">${{n}}</span><span class="code">${{escapeHtml(line) || ' '}}</span></div>`;
    }}).join('');
    const sourceEl = document.getElementById('sourceEl');
    sourceEl.innerHTML = `<pre>${{html}}</pre>`;
    // Wire line-hover -> graph-node highlight (inverse direction).
    sourceEl.querySelectorAll('.line').forEach(lineEl => {{
      const lineNo = parseInt(lineEl.dataset.line, 10);
      lineEl.addEventListener('mouseenter', () => highlightGraphNodesForLine(lineNo));
      lineEl.addEventListener('mouseleave', clearGraphHighlights);
    }});
  }}

  function clearGraphHighlights() {{
    Object.values(currentNodeMap).forEach(el => {{
      el.classList.remove('cfg-highlighted');
    }});
  }}

  function highlightGraphNodesForLine(line) {{
    const payload = BUNDLE[focusedKey];
    if (!payload) return;
    clearGraphHighlights();
    for (const id in payload.node_meta) {{
      const meta = payload.node_meta[id];
      const ls = meta.line_start;
      if (!ls) continue;
      const le = meta.line_end || ls;
      if (ls <= line && le >= line) {{
        const el = currentNodeMap[id];
        if (el) el.classList.add('cfg-highlighted');
      }}
    }}
  }}

  function clearHighlights() {{
    document.querySelectorAll('#side .line.highlighted').forEach(el => {{
      el.classList.remove('highlighted');
    }});
  }}

  function highlightRange(start, end) {{
    clearHighlights();
    if (!start) return;
    const realEnd = end || start;
    let firstEl = null;
    for (let n = start; n <= realEnd; n++) {{
      const el = document.getElementById(`line-${{n}}`);
      if (el) {{
        el.classList.add('highlighted');
        if (!firstEl) firstEl = el;
      }}
    }}
    if (firstEl) {{
      firstEl.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
    }}
  }}

  function updateHeader() {{
    document.getElementById('titleEl').textContent = focusedKey;
    document.getElementById('crumbs').textContent =
      Object.keys(tree.expansions).length === 0
        ? ''
        : 'click expanded subgraph headers to collapse · backspace to collapse last expansion';
    document.getElementById('backBtn').disabled =
      Object.keys(tree.expansions).length === 0 && tree.funcKey === ENTRY;
  }}

  /// Mermaid label sanitizer mirroring the Rust side: the renderer's
  /// SVG output treats `<` / `>` as markup and `"` would close the
  /// label wrapper. Everything else is fine inside a quoted label.
  function mermaidLabel(s) {{
    return (s || '').replace(/[\\"<>&]/g, c => ({{
      '"': "'", '\\\\': ' ', '<': '‹', '>': '›', '&': '+',
    }})[c] || c);
  }}

  /// Walk an expansion tree and emit a single mermaid string with
  /// every expanded callsite swapped for a nested subgraph. Each tree
  /// instance gets its own prefix (`root`, `root_n5`, `root_n5_n0`,
  /// etc.) so node IDs never collide across nested copies of the same
  /// function. Direction is LR for the workspace skyline (where there
  /// are many sibling modules and TD produces an unreadably-tall
  /// graph) and TD for everything else (CFGs and module submaps,
  /// where control flow reads naturally top-down).
  ///
  /// CFGs render with `flowchart-elk` so the Eclipse Layout Kernel
  /// handles them: ELK does explicit layered placement and routes
  /// back-edges around the spine on a consistent side, which dagre
  /// (mermaid's default) doesn't guarantee. Skyline keeps dagre
  /// because we want freer placement of many independent module nodes.
  ///
  /// The `%%{{init:...}}%%` block at the top configures ELK directly:
  ///   - `cycleBreakingStrategy: DEPTH_FIRST` tells ELK to break loops
  ///     by reversing the natural back-edge (body_end -> header)
  ///     instead of reversing the forward edges. This is what makes
  ///     the loop body render BELOW the header rather than above it.
  ///     With the default GREEDY strategy ELK sometimes reverses
  ///     header -> body_first, which flips the loop upside-down -
  ///     reader sees code flowing bottom-to-top inside the loop.
  ///   - `nodePlacementStrategy: NETWORK_SIMPLEX` produces compact,
  ///     spine-aligned layouts where parallel branches hug close.
  ///   - `mergeEdges: false` keeps multi-edges between the same node
  ///     pair distinguishable (e.g. true/false from a single branch).
  ///
  /// As edges are emitted we track the indices of "loop"/"continue"/
  /// "break" labels (the non-spine edges) and at the end emit a
  /// `linkStyle` directive that paints them dashed + dim. So even when
  /// the layout engine routes a back-edge across the diagram, the
  /// reader can see at a glance "this is a feedback edge, not a
  /// forward step."
  function composeMermaid(tree) {{
    const isSkyline = tree.funcKey === '@skyline';
    const direction = isSkyline ? 'LR' : 'TD';
    const renderer = isSkyline ? 'flowchart' : 'flowchart-elk';
    const lines = [];
    if (!isSkyline) {{
      // Mermaid parses init JSON with strict JSON.parse - double quotes
      // only. Backtick delimiter so we don't have to escape them.
      // GREEDY_MODEL_ORDER + edge declaration order (spine "done"
      // edge declared before body "next" edge in the Rust CFG
      // builder) tells ELK to treat the loop back-edge as the
      // feedback edge, putting the body subgraph BELOW the header.
      lines.push(`%%{{init: {{ "flowchart": {{ "curve": "basis", "htmlLabels": false, "nodeSpacing": 30, "rankSpacing": 50 }}, "elk": {{ "mergeEdges": false, "nodePlacementStrategy": "NETWORK_SIMPLEX", "cycleBreakingStrategy": "GREEDY_MODEL_ORDER" }} }} }}%%`);
    }}
    lines.push(`${{renderer}} ${{direction}}`);
    const classDefs = new Set();
    const expandedNodeIds = new Set(); // fully-prefixed ids of nodes
                                       // that became subgraphs
    const ctx = {{ edgeIndex: 0, feedbackEdges: [], deferredBackEdges: [] }};
    composeFunctionInto(lines, classDefs, tree, 'root', expandedNodeIds, ctx);
    // Stash on a module-level handle so the post-render wiring can
    // find these and draw SVG curves for them.
    pendingBackEdges = ctx.deferredBackEdges;

    // Standard class definitions used by both CFG and skyline kinds.
    if (classDefs.has('terminal')) lines.push('classDef terminal fill:#1f6feb,stroke:#58a6ff,color:#fff');
    if (classDefs.has('returns')) lines.push('classDef returns fill:#8b3a3a,stroke:#f85149,color:#fff');
    if (classDefs.has('branch')) lines.push('classDef branch fill:#3d2c00,stroke:#d29922,color:#fff');
    if (classDefs.has('calls')) lines.push('classDef calls fill:#1c2b22,stroke:#3fb950,color:#fff');
    if (classDefs.has('loops')) lines.push('classDef loops fill:#1f3a52,stroke:#79c0ff,color:#fff');
    if (classDefs.has('jumps')) lines.push('classDef jumps fill:#3d1f1f,stroke:#ff7b72,color:#fff');
    if (classDefs.has('testfn')) lines.push('classDef testfn fill:#1c2b22,stroke:#3fb950,color:#8b949e,stroke-dasharray: 4 2');
    if (classDefs.has('pubfn')) lines.push('classDef pubfn fill:#11161e,stroke:#58a6ff,color:#c9d1d9');
    if (classDefs.has('privfn')) lines.push('classDef privfn fill:#11161e,stroke:#6e7681,color:#c9d1d9');

    // Dim + dashed for back-edges so the spine reads cleanly.
    if (ctx.feedbackEdges.length > 0) {{
      lines.push(`linkStyle ${{ctx.feedbackEdges.join(',')}} stroke:#79c0ff,stroke-width:1.2px,stroke-dasharray:5 4,opacity:0.7`);
    }}

    return lines.join('\n');
  }}

  function shapeFor(kind, label) {{
    const l = mermaidLabel(label);
    switch (kind) {{
      case 'Entry':
      case 'Exit':
      case 'Return':
        return `(["${{l}}"])`;
      case 'Branch':
        return `{{{{"${{l}}"}}}}`;
      case 'Call':
        return `[/"${{l}}"/]`;
      case 'LoopHeader':
        return `[("${{l}}")]`;
      case 'LoopJump':
        return `["${{l}}"]`;
      case 'SkylineFn':
        return `["${{l}}"]`;
      case 'Statement':
      default:
        return `["${{l}}"]`;
    }}
  }}

  function composeFunctionInto(lines, classDefs, treeNode, prefix, expandedNodeIds, ctx) {{
    const payload = BUNDLE[treeNode.funcKey];
    if (!payload || !payload.graph) return;
    const graph = payload.graph;
    const groupedNodeIds = new Set();
    for (const grp of graph.groups || []) {{
      for (const id of grp.node_ids) groupedNodeIds.add(id);
    }}

    const emitNode = (n) => {{
      const fullId = `${{prefix}}_${{n.id}}`;
      currentScopeOf[fullId] = treeNode;
      // Is this node expanded? The expansion key is the *local* id.
      if (treeNode.expansions[n.id]) {{
        const childTree = treeNode.expansions[n.id];
        const childPayload = BUNDLE[childTree.funcKey];
        const childLabel = childPayload ? childPayload.label : childTree.funcKey;
        // The "× " prefix makes the close affordance obvious to the
        // operator. The whole header is clickable; this just signals
        // it visually.
        lines.push(`subgraph ${{fullId}}["× ${{mermaidLabel(childLabel)}}"]`);
        composeFunctionInto(lines, classDefs, childTree, fullId, expandedNodeIds, ctx);
        lines.push('end');
        expandedNodeIds.add(fullId);
      }} else {{
        lines.push(`${{fullId}}${{shapeFor(n.kind, n.label)}}`);
        if (n.css_class) {{
          classDefs.add(n.css_class);
          lines.push(`class ${{fullId}} ${{n.css_class}}`);
        }}
      }}
    }};

    // Emit grouped nodes inside their group's subgraph.
    for (const grp of graph.groups || []) {{
      const grpId = `${{prefix}}_${{grp.id}}`;
      lines.push(`subgraph ${{grpId}}["${{mermaidLabel(grp.label)}}"]`);
      for (const id of grp.node_ids) {{
        const n = graph.nodes.find(x => x.id === id);
        if (n) emitNode(n);
      }}
      lines.push('end');
    }}
    // Emit nodes that aren't in any group.
    for (const n of graph.nodes) {{
      if (groupedNodeIds.has(n.id)) continue;
      emitNode(n);
    }}

    // Emit edges, rewriting endpoints when the source/target became
    // a subgraph (entry/exit nodes inside the expanded child).
    //
    // Back-edges (loop/continue) are intentionally HELD BACK from the
    // mermaid source: telling ELK about them creates a cycle, and ELK
    // then has to pick which edge to reverse for layering. Even with
    // every config knob we have, mermaid's elk renderer was reversing
    // the forward `next` edge instead of the loop, which flipped the
    // body above the header. Removing back-edges from the layout
    // graph leaves a clean DAG where the body is a forward branch
    // off the header - so it always lands BELOW. We then draw the
    // back-edges manually as SVG curves after mermaid renders.
    for (const e of graph.edges) {{
      const isBack = e.label === 'loop' || e.label === 'continue';
      if (isBack) {{
        ctx.deferredBackEdges.push({{
          from: `${{prefix}}_${{e.from}}`,
          to: `${{prefix}}_${{e.to}}`,
          label: e.label,
        }});
        continue;
      }}
      let from = `${{prefix}}_${{e.from}}`;
      let to = `${{prefix}}_${{e.to}}`;
      if (treeNode.expansions[e.from]) {{
        const childTree = treeNode.expansions[e.from];
        const exit = entryOrExit(childTree.funcKey, 'Exit');
        if (exit) from = `${{prefix}}_${{e.from}}_${{exit.id}}`;
      }}
      if (treeNode.expansions[e.to]) {{
        const childTree = treeNode.expansions[e.to];
        const entry = entryOrExit(childTree.funcKey, 'Entry');
        if (entry) to = `${{prefix}}_${{e.to}}_${{entry.id}}`;
      }}
      if (e.label) {{
        lines.push(`${{from}} -->|"${{mermaidLabel(e.label)}}"| ${{to}}`);
      }} else {{
        lines.push(`${{from}} --> ${{to}}`);
      }}
      if (ctx) {{
        if (e.label === 'break') {{
          ctx.feedbackEdges.push(ctx.edgeIndex);
        }}
        ctx.edgeIndex++;
      }}
    }}
  }}

  function entryOrExit(funcKey, kind) {{
    const payload = BUNDLE[funcKey];
    if (!payload || !payload.graph) return null;
    return payload.graph.nodes.find(n => n.kind === kind);
  }}

  /// Walk the tree to find the subtree at a given prefix path. The
  /// path is a sequence of node IDs separated by `_`, e.g.
  /// "root_n5_n0" -> drill into root.expansions.n5.expansions.n0.
  /// Returns null if any segment is missing.
  function findSubtreeByPrefix(prefix) {{
    if (prefix === 'root') return tree;
    if (!prefix.startsWith('root_')) return null;
    const segs = prefix.substring('root_'.length).split('_');
    let node = tree;
    for (const s of segs) {{
      if (!node.expansions[s]) return null;
      node = node.expansions[s];
    }}
    return node;
  }}

  /// Walk the tree to find the parent subtree of a given prefix and
  /// the local node id at which it's attached. Returns
  /// `{{ parent, localId }}` or null if prefix is the root.
  function findParentSubtree(prefix) {{
    if (prefix === 'root') return null;
    const segs = prefix.substring('root_'.length).split('_');
    const localId = segs[segs.length - 1];
    const parentSegs = segs.slice(0, -1);
    let parent = tree;
    for (const s of parentSegs) {{
      if (!parent.expansions[s]) return null;
      parent = parent.expansions[s];
    }}
    return {{ parent, localId }};
  }}

  /// Resolve the full prefixed DOM id to (treeNode, localId) so click
  /// handlers can find the right tree slot. The fullId is the prefix
  /// path plus the local node id, all joined by `_`. So `root_n5_n0`
  /// drills into root.expansions.n5 (the expanded child) at local id
  /// n0.
  function resolveFullId(fullId) {{
    const segs = fullId.startsWith('root_') ? fullId.substring('root_'.length).split('_') : [];
    const localId = segs.pop();
    let scope = tree;
    for (const s of segs) {{
      if (!scope.expansions[s]) {{
        scope = null; break;
      }}
      scope = scope.expansions[s];
    }}
    return scope ? {{ scope, localId, fullId }} : null;
  }}

  function metaFor(scope, localId) {{
    const payload = BUNDLE[scope.funcKey];
    if (!payload) return null;
    return payload.node_meta[localId] || null;
  }}

  function setFocus(key) {{
    focusedKey = key;
    const payload = BUNDLE[key];
    if (payload) {{
      renderSourcePanel(payload);
    }}
  }}

  /// Show another payload's source in the side panel TEMPORARILY -
  /// just for the duration of a hover. The persistent focus
  /// (`focusedKey`) doesn't move until the operator commits with a
  /// click; on mouseleave we restore.
  function previewSource(key) {{
    const payload = BUNDLE[key];
    if (payload) renderSourcePanel(payload);
  }}

  function restoreFocusedSource() {{
    const payload = BUNDLE[focusedKey];
    if (payload) renderSourcePanel(payload);
  }}

  async function rerender(focusKey, anchorPrefix) {{
    if (focusKey) setFocus(focusKey);
    const diagram = document.getElementById('diagram');

    // Capture previous pan/zoom so we can restore continuity on
    // collapses / hover re-renders (when no anchor is given).
    let prevState = null;
    if (panZoomInstance) {{
      try {{
        prevState = {{
          zoom: panZoomInstance.getZoom(),
          pan: panZoomInstance.getPan(),
        }};
      }} catch (e) {{}}
      try {{ panZoomInstance.destroy(); }} catch (e) {{}}
      panZoomInstance = null;
    }}

    diagram.innerHTML = '';
    const mermaidSrc = composeMermaid(tree);
    let svg;
    try {{
      const result = await mermaid.render('mermaidSvg-' + Date.now(), mermaidSrc);
      svg = result.svg;
    }} catch (e) {{
      // Don't leave the operator staring at a black pane if
      // mermaid chokes. Surface the error inline; full source
      // goes to the console for further diagnosis.
      console.error('mermaid render failed:', e, '\nsource (first 4000 chars):\n', mermaidSrc.slice(0, 4000));
      diagram.innerHTML = `<pre style="color:#ff7b72;padding:24px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px">mermaid render failed:\n${{escapeHtml(e.message || String(e))}}\n\n(check console for full mermaid source)</pre>`;
      updateHeader();
      return;
    }}
    diagram.innerHTML = svg;
    const svgEl = diagram.querySelector('svg');
    if (!svgEl) {{
      diagram.innerHTML = `<pre style="color:#ff7b72;padding:24px">no svg produced by mermaid</pre>`;
      updateHeader();
      return;
    }}
    svgEl.removeAttribute('width');
    svgEl.removeAttribute('height');
    svgEl.style.maxWidth = 'none';
    svgEl.style.width = '100%';
    svgEl.style.height = '100%';
    // Manually-drawn back-edge curves can swing past the bounds mermaid
    // computed for the original DAG, so allow drawing outside the
    // viewBox without clipping.
    svgEl.style.overflow = 'visible';
    await new Promise(r => requestAnimationFrame(r));
    panZoomInstance = svgPanZoom(svgEl, {{
      zoomEnabled: true,
      controlIconsEnabled: false,
      fit: true,
      center: true,
      contain: false,
      minZoom: 0.05,
      maxZoom: 8,
    }});

    // svg-pan-zoom's fit+center init guarantees the entire graph
    // is visible. From that safe baseline we either restore the
    // operator's prior viewport (collapses, hover refreshes) or
    // back off slightly for breathing room (expansions, first
    // render). Animation to the anchor follows below.
    if (prevState && !anchorPrefix) {{
      try {{
        panZoomInstance.zoom(prevState.zoom);
        panZoomInstance.pan(prevState.pan);
      }} catch (e) {{}}
    }} else {{
      try {{ panZoomInstance.zoomBy(0.9); }} catch (e) {{}}
    }}

    wireInteractions(svgEl);
    stripLoopClusterBackgrounds(svgEl);
    drawDeferredBackEdges(svgEl);
    updateHeader();
    clearHighlights();

    // Smooth glide to the freshly-expanded subgraph. Starts from
    // the fit view (whole graph visible), animates camera to the
    // cluster. If lookup or measurement fails the operator still
    // sees the fit view - not an empty pane.
    if (anchorPrefix) {{
      await new Promise(r => requestAnimationFrame(r));
      smoothFocusOn(svgEl, anchorPrefix);
    }}
  }}

  /// Strip the gray rectangle mermaid stamps for every cluster that
  /// represents a loop body. Setting fill="none" via attribute or CSS
  /// kept losing - whatever combination of inline `style=""` and SVG
  /// presentation attributes mermaid v10 elk emits, the rect kept
  /// rendering. So just yank the background element out of the DOM
  /// entirely. Inline function expansions (the `× funcName` clusters)
  /// keep their styling - we identify those by the `expanded-cluster`
  /// class that wireInteractions stamps just before this runs.
  function stripLoopClusterBackgrounds(svgEl) {{
    svgEl.querySelectorAll('g.subgraph').forEach(cluster => {{
      if (cluster.classList.contains('expanded-cluster')) return;
      // The background is whichever shape sits at the start of the
      // cluster group, BEFORE the cluster-label and inner nodes. We
      // look for shape elements (rect/path/polygon) that are direct
      // children, plus an extra hop in case mermaid wraps them in a
      // <g>. Don't touch elements deeper than that - those are the
      // body nodes themselves.
      const candidates = [
        ...cluster.querySelectorAll(':scope > rect'),
        ...cluster.querySelectorAll(':scope > path'),
        ...cluster.querySelectorAll(':scope > polygon'),
        ...cluster.querySelectorAll(':scope > g:first-child > rect'),
        ...cluster.querySelectorAll(':scope > g:first-child > path'),
      ];
      // Also: anything explicitly tagged as a cluster background by
      // mermaid (some versions add a `.background` class).
      cluster.querySelectorAll('.background, .cluster-bkg, rect.outer, rect.cluster').forEach(el => {{
        candidates.push(el);
      }});
      const seen = new Set();
      candidates.forEach(el => {{
        if (seen.has(el)) return;
        seen.add(el);
        // Skip if this element is inside a nested g.node (i.e. a body
        // node's own shape that happens to sit first in the cluster).
        if (el.closest('g.node') && el.closest('g.node') !== cluster) return;
        el.remove();
      }});
      // Also yank the label group - loop body subgraphs are a layout
      // device, not a UI element. The user shouldn't see a "body of..."
      // header above each loop.
      cluster.querySelectorAll(':scope > g.label').forEach(el => el.remove());
    }});
  }}

  /// For each back-edge held back from the mermaid graph, draw an SVG
  /// path from source node up-left to target node. Inserted into
  /// mermaid's main `<g>` so pan/zoom transforms apply uniformly.
  ///
  /// Visual convention: the curve exits the TOP of the source (body
  /// node deep in the loop) and enters the RIGHT side of the target
  /// (loop header sitting on the spine). With body subgraphs placed
  /// to the right of the header, this naturally produces a curve
  /// that swings up-left back to the header - the "right side, down,
  /// left and up again" loop shape the operator asked for.
  function drawDeferredBackEdges(svgEl) {{
    if (!pendingBackEdges || pendingBackEdges.length === 0) return;
    // Mermaid wraps content in a top-level <g> with a transform; that's
    // also where the existing edges live, so adding our paths there
    // means they move with pan/zoom and sit at the correct z-order
    // (above subgraph fills, below labels).
    const root = svgEl.querySelector(':scope > g') || svgEl;
    // CTM has to be from `root`, NOT the outer <svg>. The paths we emit
    // get inserted as children of `root`, so their `d=...` coordinates
    // are read in root's local space - which has root's own transform
    // applied on top of the svg's. Using svgEl.getScreenCTM() here
    // would offset every curve by root's transform.
    const rootCTM = root.getScreenCTM();
    if (!rootCTM) return;
    const inv = rootCTM.inverse();

    // Reuse mermaid's existing arrowhead marker if it stamped one on
    // the rendered graph; otherwise stamp our own.
    const existingMarker = svgEl.querySelector('marker[id*="pointEnd"], marker[id*="arrow"]');
    let markerRef = existingMarker ? `url(#${{existingMarker.id}})` : null;
    if (!markerRef) {{
      const ns = 'http://www.w3.org/2000/svg';
      let defs = svgEl.querySelector('defs');
      if (!defs) {{
        defs = document.createElementNS(ns, 'defs');
        svgEl.insertBefore(defs, svgEl.firstChild);
      }}
      const m = document.createElementNS(ns, 'marker');
      m.setAttribute('id', 'planecode-backedge-arrow');
      m.setAttribute('viewBox', '0 0 10 10');
      m.setAttribute('refX', '8');
      m.setAttribute('refY', '5');
      m.setAttribute('markerWidth', '6');
      m.setAttribute('markerHeight', '6');
      m.setAttribute('orient', 'auto');
      const tri = document.createElementNS(ns, 'path');
      tri.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
      tri.setAttribute('fill', '#79c0ff');
      m.appendChild(tri);
      defs.appendChild(m);
      markerRef = 'url(#planecode-backedge-arrow)';
    }}

    for (const be of pendingBackEdges) {{
      const fromEl = findBackEdgeEndpoint(svgEl, be.from);
      const toEl = findBackEdgeEndpoint(svgEl, be.to);
      if (!fromEl || !toEl) {{
        console.warn('planecode back-edge: missing endpoint', be, 'from?', !!fromEl, 'to?', !!toEl);
        continue;
      }}
      const fromBox = elBoxInUserSpace(svgEl, fromEl, inv);
      const toBox = elBoxInUserSpace(svgEl, toEl, inv);
      if (!fromBox || !toBox) continue;
      const d = buildLoopCurvePath(fromBox, toBox);
      if (!d) continue;
      const ns = 'http://www.w3.org/2000/svg';
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', d);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#79c0ff');
      path.setAttribute('stroke-width', '1.4');
      path.setAttribute('stroke-dasharray', '5 4');
      path.setAttribute('opacity', '0.85');
      path.setAttribute('marker-end', markerRef);
      path.setAttribute('class', 'planecode-backedge');
      root.appendChild(path);

      // Optional label (e.g. "loop") at the curve apex.
      if (be.label) {{
        const txt = document.createElementNS(ns, 'text');
        const apex = curveApex(fromBox, toBox);
        txt.setAttribute('x', apex.x);
        txt.setAttribute('y', apex.y);
        txt.setAttribute('fill', '#79c0ff');
        txt.setAttribute('font-size', '11');
        txt.setAttribute('font-family', 'ui-monospace, SFMono-Regular, monospace');
        txt.setAttribute('opacity', '0.85');
        txt.setAttribute('text-anchor', 'middle');
        txt.textContent = be.label;
        root.appendChild(txt);
      }}
    }}
  }}

  function findBackEdgeEndpoint(svgEl, fullId) {{
    return svgEl.querySelector(`g.node[id^="flowchart-${{fullId}}-"]`)
      || svgEl.querySelector(`g.node[id="flowchart-${{fullId}}"]`)
      || svgEl.querySelector(`g.subgraph[id^="flowchart-${{fullId}}-"]`)
      || svgEl.querySelector(`g.subgraph[id="flowchart-${{fullId}}"]`);
  }}

  /// Convert an element's screen-space bounding rect into a target
  /// `<g>`'s user coordinate space - i.e. the space `path d=...`
  /// values use when the path is appended under that group. Caller
  /// passes the precomputed inverse CTM so we don't recompute it
  /// for every endpoint.
  function elBoxInUserSpace(svgEl, el, invCTM) {{
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return null;
    const tl = svgEl.createSVGPoint();
    tl.x = rect.left;  tl.y = rect.top;
    const br = svgEl.createSVGPoint();
    br.x = rect.right; br.y = rect.bottom;
    const tlS = tl.matrixTransform(invCTM);
    const brS = br.matrixTransform(invCTM);
    return {{
      x: tlS.x,
      y: tlS.y,
      width: brS.x - tlS.x,
      height: brS.y - tlS.y,
      cx: (tlS.x + brS.x) / 2,
      cy: (tlS.y + brS.y) / 2,
    }};
  }}

  /// Build a Bezier curve from `from` (a deep body node) to `to` (the
  /// loop header). The curve exits the source on its right edge,
  /// arcs RIGHT and UP, and enters the target on its right edge.
  /// Result: a clockwise loop on the right side of the body subgraph,
  /// matching the "right -> down -> left -> up" mental model of code
  /// reading.
  function buildLoopCurvePath(from, to) {{
    const startX = from.x + from.width;
    const startY = from.cy;
    const endX = to.x + to.width;
    const endY = to.cy;
    const swing = Math.max(60, Math.abs(startY - endY) * 0.35,
                           Math.abs(startX - endX) * 0.5);
    // Control points push right, then up, then back left into target.
    const c1x = startX + swing;
    const c1y = startY;
    const c2x = endX + swing;
    const c2y = endY;
    return `M ${{startX}} ${{startY}} C ${{c1x}} ${{c1y}}, ${{c2x}} ${{c2y}}, ${{endX}} ${{endY}}`;
  }}

  function curveApex(from, to) {{
    const startX = from.x + from.width;
    const startY = from.cy;
    const endX = to.x + to.width;
    const endY = to.cy;
    const swing = Math.max(60, Math.abs(startY - endY) * 0.35,
                           Math.abs(startX - endX) * 0.5);
    return {{ x: Math.max(startX, endX) + swing * 0.6, y: (startY + endY) / 2 }};
  }}

  function findClusterOrNode(svgEl, fullId) {{
    return [
      svgEl.querySelector(`g.subgraph[id="${{fullId}}"]`),
      svgEl.querySelector(`g.subgraph[id^="${{fullId}}-"]`),
      svgEl.querySelector(`g.subgraph[id^="flowchart-${{fullId}}-"]`),
      svgEl.querySelector(`g.subgraph[id^="flowchart-${{fullId}}"]`),
      svgEl.querySelector(`g.node[id^="flowchart-${{fullId}}-"]`),
    ].find(el => el != null) || null;
  }}

  /// Smoothly animate pan + zoom so the named subgraph lands at
  /// viewport center and fills a comfortable fraction of it.
  /// Screen coords (getBoundingClientRect) sidestep the
  /// transform-stack confusion that plagued the prior getBBox
  /// implementation - we measure where the cluster *actually* is
  /// on screen, derive its position in the SVG's user coord
  /// space, and choose targetPan so the cluster lands at viewport
  /// center under the *new* zoom.
  function smoothFocusOn(svgEl, fullId) {{
    if (!panZoomInstance) return;
    const target = findClusterOrNode(svgEl, fullId);
    if (!target) return;
    const targetRect = target.getBoundingClientRect();
    if (targetRect.width === 0 && targetRect.height === 0) return;
    const svgRect = svgEl.getBoundingClientRect();

    const cxNow = targetRect.left + targetRect.width / 2 - svgRect.left;
    const cyNow = targetRect.top + targetRect.height / 2 - svgRect.top;
    const startPan = panZoomInstance.getPan();
    const startZoom = panZoomInstance.getZoom();

    // Cluster center in SVG user coords. This is invariant under
    // pan/zoom: svg-pan-zoom places it on screen as
    //   screen = pan + clusterUser * zoom.
    // Inverting: clusterUser = (screen - pan) / zoom.
    const clusterUserX = (cxNow - startPan.x) / startZoom;
    const clusterUserY = (cyNow - startPan.y) / startZoom;

    // Fit-to-fill: scale so the cluster fills ~80% of the smaller
    // viewport axis. Always applied, in both directions - large
    // clusters shrink to fit, small clusters magnify so the
    // operator doesn't have to scroll-zoom after every click.
    let zoomMul = 1;
    const tw = targetRect.width;
    const th = targetRect.height;
    if (tw > 0 && th > 0) {{
      zoomMul = Math.min(
        (svgRect.width * 0.8) / tw,
        (svgRect.height * 0.8) / th,
      );
    }}
    // Clamp to a usable range. 6x is enough to read tiny clusters;
    // 0.1 keeps huge ones legible without vanishing into pixels.
    const targetZoom = Math.max(0.1, Math.min(6, startZoom * zoomMul));

    // Choose targetPan so the cluster lands at viewport center
    // under the new zoom. Without the targetZoom factor here the
    // animation slides the cluster off-screen any time zoom
    // changes - exactly the disappearing-graph bug we just hit.
    const targetPan = {{
      x: svgRect.width / 2 - clusterUserX * targetZoom,
      y: svgRect.height / 2 - clusterUserY * targetZoom,
    }};

    animateViewport(startPan, targetPan, startZoom, targetZoom, 450);
  }}

  /// Linear interpolation of pan and zoom in lockstep is
  /// mathematically equivalent to a linear glide of every point on
  /// screen (since svg-pan-zoom's transform is `translate(pan) *
  /// scale(zoom)` and zoom is constant per frame). So a single
  /// cubic-ease envelope on `t` gives the operator a smooth slide
  /// from the previous viewport to the new focus.
  function animateViewport(startPan, targetPan, startZoom, targetZoom, duration) {{
    const startTime = performance.now();
    function step(now) {{
      if (!panZoomInstance) return;
      const raw = Math.min(1, (now - startTime) / duration);
      const t = raw < 0.5
        ? 4 * raw * raw * raw
        : 1 - Math.pow(-2 * raw + 2, 3) / 2;
      const z = startZoom + (targetZoom - startZoom) * t;
      const px = startPan.x + (targetPan.x - startPan.x) * t;
      const py = startPan.y + (targetPan.y - startPan.y) * t;
      try {{
        panZoomInstance.zoom(z);
        panZoomInstance.pan({{ x: px, y: py }});
      }} catch (e) {{ return; }}
      if (raw < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
  }}

  function wireInteractions(svgEl) {{
    currentNodeMap = {{}};
    // Plain nodes
    svgEl.querySelectorAll('g.node').forEach(node => {{
      const fullId = node.id.replace(/^flowchart-/, '').split('-')[0];
      currentNodeMap[fullId] = node;
      const resolved = resolveFullId(fullId);
      if (!resolved) return;
      const meta = metaFor(resolved.scope, resolved.localId);
      if (!meta) return;
      const navigable = !!(meta.callee_key && BUNDLE[meta.callee_key]);
      if (navigable) {{
        node.classList.add('navigable');
        node.setAttribute('title', 'click to expand ' + meta.callee_key + ' inline');
      }} else {{
        node.removeAttribute('title');
      }}
      node.addEventListener('mouseenter', () => {{
        if (meta.callee_key && BUNDLE[meta.callee_key]) {{
          // Hovering a navigable node previews the callee's source
          // without committing the focus change. Click commits.
          previewSource(meta.callee_key);
        }} else if (meta.line_start && BUNDLE[focusedKey] && resolved.scope.funcKey === focusedKey) {{
          highlightRange(meta.line_start, meta.line_end);
        }}
      }});
      node.addEventListener('mouseleave', () => {{
        clearHighlights();
        // Restore the source panel to whatever the operator last
        // committed to via click.
        restoreFocusedSource();
      }});
      node.addEventListener('click', (ev) => {{
        ev.stopPropagation();
        if (!navigable) return;
        if (resolved.scope.expansions[resolved.localId]) {{
          // Collapsing: keep the operator's pan/zoom anchored where
          // it was so they don't lose orientation.
          delete resolved.scope.expansions[resolved.localId];
          rerender();
        }} else {{
          // Expanding: re-anchor on the new subgraph so the operator
          // immediately sees what they just opened.
          resolved.scope.expansions[resolved.localId] = {{
            funcKey: meta.callee_key,
            expansions: {{}},
          }};
          rerender(meta.callee_key, fullId);
        }}
      }});
    }});
    // Wire collapse on every expanded subgraph header. Mermaid
    // renders a subgraph as `<g class="cluster" id="{{id}}">` containing
    // a `<g class="cluster-label">` that holds the title. Different
    // mermaid versions stamp the cluster id with various prefixes /
    // counter suffixes, so we try every known shape until one resolves
    // back to a tree expansion.
    svgEl.querySelectorAll('g.subgraph').forEach(cluster => {{
      const idCandidates = [
        cluster.id,
        cluster.id.replace(/^flowchart-/, ''),
        cluster.id.replace(/-\d+$/, ''),
        cluster.id.replace(/^flowchart-/, '').replace(/-\d+$/, ''),
      ];
      let parent = null;
      for (const c of idCandidates) {{
        const p = findParentSubtree(c);
        if (p && p.parent.expansions[p.localId]) {{
          parent = p;
          break;
        }}
      }}
      if (!parent) return;

      // Mark the cluster itself so CSS can paint it with the blue
      // accent reserved for inline function expansions.
      cluster.classList.add('expanded-cluster');

      // The label is a small `<g class="cluster-label">` with the
      // title text. Make it clickable; the `× funcKey` text already
      // signals "close" to the operator. Stop pan-zoom from grabbing
      // mousedown so the click reliably fires.
      const labelGroup = cluster.querySelector(':scope > g.label, g.cluster-label, .cluster-label');
      if (!labelGroup) return;
      labelGroup.classList.add('expanded-cluster-header');
      labelGroup.style.cursor = 'pointer';
      labelGroup.setAttribute('title', 'click to collapse this expansion');
      labelGroup.addEventListener('mousedown', (ev) => ev.stopPropagation());
      labelGroup.addEventListener('click', (ev) => {{
        ev.stopPropagation();
        ev.preventDefault();
        delete parent.parent.expansions[parent.localId];
        rerender(parent.parent.funcKey);
      }});
    }});
  }}

  function goBack() {{
    // Collapse the deepest expansion.
    const path = findDeepestExpansion(tree, []);
    if (!path) return;
    const {{ parent, localId, parentKey }} = path;
    delete parent.expansions[localId];
    rerender(parentKey);
  }}

  function findDeepestExpansion(node, ancestors) {{
    let deepest = null;
    let deepestDepth = -1;
    function walk(n, depth, parents) {{
      for (const k of Object.keys(n.expansions)) {{
        const child = n.expansions[k];
        if (depth > deepestDepth) {{
          deepest = {{ parent: n, localId: k, parentKey: n.funcKey }};
          deepestDepth = depth;
        }}
        walk(child, depth + 1, [...parents, n]);
      }}
    }}
    walk(node, 0, []);
    return deepest;
  }}

  document.getElementById('backBtn').addEventListener('click', goBack);
  window.addEventListener('keydown', (e) => {{
    if (e.key === 'Backspace' && document.activeElement.tagName !== 'INPUT') {{
      goBack();
    }}
  }});
  window.addEventListener('resize', () => {{
    if (panZoomInstance) {{
      try {{
        panZoomInstance.resize();
        panZoomInstance.fit();
        panZoomInstance.center();
      }} catch (e) {{}}
    }}
  }});

  setFocus(ENTRY);
  rerender();
</script>
</body>
</html>
"#
    )
}

/// Make a JSON string safe to embed inside a `<script>` tag. JSON
/// itself can contain sequences that prematurely terminate the script
/// (or that older JS parsers reject) even though they're valid JSON
/// strings. Specifically:
///   - `</` ends the surrounding `<script>` tag for the HTML parser.
///     We escape `/` as `\/` (a legal JSON escape) so the literal
///     text the JS parser receives is still `</`.
///   - U+2028 / U+2029 were illegal in pre-ES2019 string literals.
///   - `<!--` would start an HTML comment that pre-modern browsers
///     would consume inside the script body.
fn make_script_safe(s: &str) -> String {
    s.replace("</", "<\\/")
        .replace("\u{2028}", "\\u2028")
        .replace("\u{2029}", "\\u2029")
        .replace("<!--", "<\\!--")
}
