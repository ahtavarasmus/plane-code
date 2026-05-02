//! Per-function control-flow graph extraction.
//!
//! Walks a function body's syn AST and produces a CfgGraph: nodes for
//! statements / calls / branches / returns / loops, edges following
//! control flow. Intentionally a SHAPE extractor, not a semantics one -
//! "what could happen syntactically" rather than "what will happen at
//! runtime." Every node corresponds to a syntactic construct in the
//! source so the rendering is deterministic and verifiable.
//!
//! Phase B coverage:
//!   - if / else                      (diamond + true/false branches)
//!   - function and method calls     (call nodes; .await is a tagged call)
//!   - return                          (terminal node)
//!   - sequential let / expr stmts   (plain rectangles)
//!   - loop / while / for             (header + back-edge)
//!   - break / continue                 (jump nodes pointing at loop frame)
//!   - match                            (N-way diamond, one arm per pattern)
//!
//! Not yet expanded (Phase E candidates): `?` propagation, iterator
//! combinator chains, async blocks.

use anyhow::{anyhow, Result};
use quote::ToTokens;
use std::fmt::Write;

type NodeId = usize;

#[derive(Debug, Clone)]
pub struct CfgGraph {
    pub function: String,
    pub nodes: Vec<CfgNode>,
    pub edges: Vec<CfgEdge>,
    pub entry: NodeId,
    pub exit: NodeId,
    /// One entry per loop construct. Each carries the IDs of nodes that
    /// belong to that loop's body so the renderer can wrap them in a
    /// subgraph. With body nodes contained, the layout engine routes
    /// the back-edge around the subgraph instead of letting it govern
    /// rank assignment - which is what flips loop bodies above their
    /// headers in plain ELK output.
    pub loop_groups: Vec<CfgLoopGroup>,
}

#[derive(Debug, Clone)]
pub struct CfgLoopGroup {
    pub header: NodeId,
    pub body_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct CfgNode {
    pub id: NodeId,
    pub kind: CfgNodeKind,
    pub label: String,
    /// Source line range that this node represents, body-relative
    /// (1-indexed, lines within `function.body`). `None` for synthetic
    /// nodes (entry / exit / join / loop end) that don't correspond to
    /// any source location.
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum CfgNodeKind {
    Entry,
    Exit,
    Statement,
    Call {
        #[allow(dead_code)]
        callee: String,
    },
    Branch {
        #[allow(dead_code)]
        condition: String,
    },
    Return,
    /// Header of a loop construct - the node iteration restarts at on
    /// the back-edge from the body's tail. Distinct shape so loops are
    /// visually obvious.
    LoopHeader,
    /// `break` or `continue` jump. Has a sequential edge from the
    /// previous node and a labeled edge to the corresponding target on
    /// the active loop frame.
    LoopJump {
        #[allow(dead_code)]
        kind: LoopJumpKind,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum LoopJumpKind {
    Break,
    Continue,
}

#[derive(Debug, Clone)]
pub struct CfgEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub label: Option<String>,
}

/// Active loop context. `process_*` walkers push a frame on entry to a
/// loop construct and pop it on exit, so `break` and `continue`
/// expressions can resolve to the right targets even when nested.
struct LoopFrame {
    /// Where iteration restarts (loop header for `loop`, condition for
    /// `while`, iterator header for `for`).
    continue_target: NodeId,
    /// Where `break` jumps - the loop's fall-through node.
    break_target: NodeId,
}

impl CfgGraph {
    fn new(function: String) -> Self {
        let mut g = Self {
            function,
            nodes: Vec::new(),
            edges: Vec::new(),
            entry: 0,
            exit: 0,
            loop_groups: Vec::new(),
        };
        g.entry = g.add(CfgNodeKind::Entry, "entry".into());
        g.exit = g.add(CfgNodeKind::Exit, "exit".into());
        g
    }

    fn add(&mut self, kind: CfgNodeKind, label: String) -> NodeId {
        self.add_with_span(kind, label, None, None)
    }

    fn add_with_span(
        &mut self,
        kind: CfgNodeKind,
        label: String,
        line_start: Option<usize>,
        line_end: Option<usize>,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(CfgNode {
            id,
            kind,
            label,
            line_start,
            line_end,
        });
        id
    }

    fn link(&mut self, from: NodeId, to: NodeId, label: Option<String>) {
        self.edges.push(CfgEdge { from, to, label });
    }
}

/// Convert a wrapped-source line (where line 1 is the synthetic
/// `fn __planecode_flow_dummy() {` line) to body-relative line numbers.
/// Returns None if the input is 0 (uninitialized span).
fn body_line(wrapped_line: usize) -> Option<usize> {
    if wrapped_line == 0 {
        None
    } else {
        Some(wrapped_line.saturating_sub(1).max(1))
    }
}

fn span_lines<T: syn::spanned::Spanned>(t: &T) -> (Option<usize>, Option<usize>) {
    let s = t.span();
    (body_line(s.start().line), body_line(s.end().line))
}

/// True for nodes that don't fall through - subsequent statements in
/// the same block are unreachable from these.
fn is_terminating(kind: &CfgNodeKind) -> bool {
    matches!(kind, CfgNodeKind::Return | CfgNodeKind::LoopJump { .. })
}

pub fn build_cfg_from_block(block: &syn::Block, function_label: &str) -> CfgGraph {
    let mut g = CfgGraph::new(function_label.to_string());
    let entry = g.entry;
    let exit = g.exit;
    let mut loops: Vec<LoopFrame> = Vec::new();
    let last = process_block(&mut g, block, entry, &mut loops);
    if last != exit && !is_terminating(&g.nodes[last].kind) {
        g.link(last, exit, None);
    }
    g
}

pub fn build_cfg_from_body_source(body: &str, function_label: &str) -> Result<CfgGraph> {
    let wrapped = format!("fn __planecode_flow_dummy() {{\n{body}\n}}\n");
    let parsed: syn::ItemFn = syn::parse_str(&wrapped)
        .map_err(|e| anyhow!("body did not parse as Rust: {e}"))?;
    Ok(build_cfg_from_block(&parsed.block, function_label))
}

fn process_block(
    g: &mut CfgGraph,
    block: &syn::Block,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    let mut current = prev;
    for stmt in &block.stmts {
        current = process_stmt(g, stmt, current, loops);
        if is_terminating(&g.nodes[current].kind) {
            return current;
        }
    }
    current
}

fn process_stmt(
    g: &mut CfgGraph,
    stmt: &syn::Stmt,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    match stmt {
        syn::Stmt::Local(local) => {
            let label = local_label(local);
            let (ls, le) = span_lines(local);
            let n = g.add_with_span(CfgNodeKind::Statement, label, ls, le);
            g.link(prev, n, None);
            n
        }
        syn::Stmt::Expr(expr, _semi) => process_expr(g, expr, prev, loops),
        syn::Stmt::Item(_) | syn::Stmt::Macro(_) => {
            let label = stmt.to_token_stream().to_string();
            let (ls, le) = span_lines(stmt);
            let n = g.add_with_span(CfgNodeKind::Statement, truncate(&label, 60), ls, le);
            g.link(prev, n, None);
            n
        }
    }
}

fn process_expr(
    g: &mut CfgGraph,
    expr: &syn::Expr,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    match expr {
        syn::Expr::If(if_expr) => process_if(g, if_expr, prev, loops),
        syn::Expr::Match(m) => process_match(g, m, prev, loops),
        syn::Expr::Loop(l) => process_loop(g, l, prev, loops),
        syn::Expr::While(w) => process_while(g, w, prev, loops),
        syn::Expr::ForLoop(f) => process_for(g, f, prev, loops),
        syn::Expr::Break(b) => {
            let target = loops
                .last()
                .map(|frame| frame.break_target)
                .unwrap_or(g.exit);
            let label = match &b.expr {
                Some(e) => format!("break {}", truncate(&expr_to_short(e), 30)),
                None => "break".into(),
            };
            let (ls, le) = span_lines(b);
            let n = g.add_with_span(
                CfgNodeKind::LoopJump {
                    kind: LoopJumpKind::Break,
                },
                label,
                ls,
                le,
            );
            g.link(prev, n, None);
            g.link(n, target, Some("break".into()));
            n
        }
        syn::Expr::Continue(c) => {
            let target = loops
                .last()
                .map(|frame| frame.continue_target)
                .unwrap_or(g.entry);
            let (ls, le) = span_lines(c);
            let n = g.add_with_span(
                CfgNodeKind::LoopJump {
                    kind: LoopJumpKind::Continue,
                },
                "continue".into(),
                ls,
                le,
            );
            g.link(prev, n, None);
            g.link(n, target, Some("continue".into()));
            n
        }
        syn::Expr::Return(ret) => {
            let after_eval = if let Some(inner) = &ret.expr {
                process_expr_for_side_effects(g, inner, prev, loops)
            } else {
                prev
            };
            let (ls, le) = span_lines(ret);
            let n = g.add_with_span(CfgNodeKind::Return, "return".into(), ls, le);
            g.link(after_eval, n, None);
            n
        }
        syn::Expr::Await(aw) => {
            let inner = process_expr(g, &aw.base, prev, loops);
            let (ls, le) = span_lines(aw);
            let n = g.add_with_span(
                CfgNodeKind::Call {
                    callee: "await".into(),
                },
                ".await".into(),
                ls,
                le,
            );
            g.link(inner, n, None);
            n
        }
        syn::Expr::Call(call) => {
            let callee = expr_to_short(&call.func);
            let (ls, le) = span_lines(call);
            let n = g.add_with_span(
                CfgNodeKind::Call {
                    callee: callee.clone(),
                },
                format!("{callee}(...)"),
                ls,
                le,
            );
            g.link(prev, n, None);
            n
        }
        syn::Expr::MethodCall(mc) => {
            // Walk back through the method-call chain. Long chains like
            // `vec.iter().filter().map().collect()` collapse to a SINGLE
            // node so the chart doesn't hairball. The deepest receiver
            // is whatever sits at the bottom of the chain (a path, a
            // call, a literal, etc.). For navigation we pick the most
            // likely resolvable callee name: the deepest function call
            // if there is one (e.g. `find_port` from
            // `find_port().is_some()`), else the outermost method name.
            let mut chain: Vec<&syn::ExprMethodCall> = vec![mc];
            let mut deepest: &syn::Expr = &mc.receiver;
            while let syn::Expr::MethodCall(inner) = deepest {
                chain.push(inner);
                deepest = &inner.receiver;
            }
            chain.reverse();

            let receiver_text = expr_to_short(deepest);
            let methods: Vec<String> = chain
                .iter()
                .map(|m| format!(".{}(...)", m.method))
                .collect();
            let chain_text = format!(
                "{}{}",
                truncate(&receiver_text, 30),
                methods.join("")
            );

            let callee = if let syn::Expr::Call(call) = deepest {
                expr_to_short(&call.func)
            } else {
                chain
                    .last()
                    .expect("chain is non-empty")
                    .method
                    .to_string()
            };

            let (ls, le) = span_lines(mc);
            let n = g.add_with_span(
                CfgNodeKind::Call { callee },
                truncate(&chain_text, 100),
                ls,
                le,
            );
            g.link(prev, n, None);
            n
        }
        syn::Expr::Block(b) => process_block(g, &b.block, prev, loops),
        other => {
            let label = truncate(&expr_to_short(other), 80);
            let (ls, le) = span_lines(other);
            let n = g.add_with_span(CfgNodeKind::Statement, label, ls, le);
            g.link(prev, n, None);
            n
        }
    }
}

fn process_expr_for_side_effects(
    g: &mut CfgGraph,
    expr: &syn::Expr,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    match expr {
        syn::Expr::Call(_) | syn::Expr::MethodCall(_) | syn::Expr::Await(_) => {
            process_expr(g, expr, prev, loops)
        }
        _ => prev,
    }
}

fn process_if(
    g: &mut CfgGraph,
    if_expr: &syn::ExprIf,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    let cond_label = expr_to_short(&if_expr.cond);
    let (ls, le) = span_lines(&if_expr.cond);
    // Pull side-effecting calls out of the condition so they appear as
    // their own Call nodes (and so cross-function navigation can hook
    // them) instead of being absorbed into the diamond's text label.
    let pre = process_expr_for_side_effects(g, &if_expr.cond, prev, loops);
    let cond = g.add_with_span(
        CfgNodeKind::Branch {
            condition: cond_label.clone(),
        },
        format!("if {}", truncate(&cond_label, 60)),
        ls,
        le,
    );
    g.link(pre, cond, None);

    let then_end = process_block(g, &if_expr.then_branch, cond, loops);
    relabel_first_outgoing(g, cond, "true");

    let else_end = match &if_expr.else_branch {
        Some((_, else_expr)) => {
            let before_count = g.edges.len();
            let end = process_expr(g, else_expr, cond, loops);
            if let Some(e) = g.edges.get_mut(before_count) {
                if e.from == cond && e.label.is_none() {
                    e.label = Some("false".into());
                }
            }
            end
        }
        None => cond,
    };

    let join = g.add(CfgNodeKind::Statement, "join".into());
    if !is_terminating(&g.nodes[then_end].kind) {
        g.link(then_end, join, None);
    }
    if if_expr.else_branch.is_none() {
        g.link(cond, join, Some("false".into()));
    } else if !is_terminating(&g.nodes[else_end].kind) && else_end != cond {
        g.link(else_end, join, None);
    }
    join
}

fn process_match(
    g: &mut CfgGraph,
    m: &syn::ExprMatch,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    let scrutinee = expr_to_short(&m.expr);
    let (ls, le) = span_lines(&m.expr);
    let pre = process_expr_for_side_effects(g, &m.expr, prev, loops);
    let head = g.add_with_span(
        CfgNodeKind::Branch {
            condition: scrutinee.clone(),
        },
        format!("match {}", truncate(&scrutinee, 50)),
        ls,
        le,
    );
    g.link(pre, head, None);

    let join = g.add(CfgNodeKind::Statement, "join".into());

    for arm in &m.arms {
        let pat = arm.pat.to_token_stream().to_string();
        let before = g.edges.len();
        let arm_end = process_expr(g, &arm.body, head, loops);
        if let Some(e) = g.edges.get_mut(before) {
            if e.from == head && e.label.is_none() {
                e.label = Some(truncate(&pat, 30));
            }
        }
        if !is_terminating(&g.nodes[arm_end].kind) && arm_end != head {
            g.link(arm_end, join, None);
        }
    }

    join
}

fn process_loop(
    g: &mut CfgGraph,
    l: &syn::ExprLoop,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    let (ls, le) = span_lines(l);
    let header = g.add_with_span(CfgNodeKind::LoopHeader, "loop".into(), ls, le);
    g.link(prev, header, None);
    let exit = g.add(CfgNodeKind::Statement, "loop end".into());

    loops.push(LoopFrame {
        continue_target: header,
        break_target: exit,
    });
    let body_start = g.nodes.len();
    let body_end = process_block(g, &l.body, header, loops);
    let body_nodes: Vec<NodeId> = (body_start..g.nodes.len()).collect();
    loops.pop();

    // Back-edge emitted last so dagre/ELK route the forward path as the
    // primary spine and the loop edge consistently swings around the
    // body. Same trick used in process_while / process_for below.
    if !is_terminating(&g.nodes[body_end].kind) && body_end != header {
        g.link(body_end, header, Some("loop".into()));
    }
    push_loop_group(g, header, body_nodes);
    exit
}

fn process_while(
    g: &mut CfgGraph,
    w: &syn::ExprWhile,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    let cond_str = expr_to_short(&w.cond);
    let (ls, le) = span_lines(&w.cond);
    let pre = process_expr_for_side_effects(g, &w.cond, prev, loops);
    let cond = g.add_with_span(
        CfgNodeKind::Branch {
            condition: cond_str.clone(),
        },
        format!("while {}", truncate(&cond_str, 50)),
        ls,
        le,
    );
    g.link(pre, cond, None);
    let exit = g.add(CfgNodeKind::Statement, "join".into());

    // Spine edge first - same trick as process_for. Keeps the false
    // path on the main vertical spine and pushes the loop body to
    // whichever side the layout engine prefers.
    g.link(cond, exit, Some("false".into()));

    loops.push(LoopFrame {
        continue_target: cond,
        break_target: exit,
    });
    let before = g.edges.len();
    let body_start = g.nodes.len();
    let body_end = process_block(g, &w.body, cond, loops);
    let body_nodes: Vec<NodeId> = (body_start..g.nodes.len()).collect();
    if let Some(e) = g.edges.get_mut(before) {
        if e.from == cond && e.label.is_none() {
            e.label = Some("true".into());
        }
    }
    if !is_terminating(&g.nodes[body_end].kind) && body_end != cond {
        g.link(body_end, cond, Some("loop".into()));
    }
    push_loop_group(g, cond, body_nodes);
    loops.pop();
    exit
}

fn process_for(
    g: &mut CfgGraph,
    f: &syn::ExprForLoop,
    prev: NodeId,
    loops: &mut Vec<LoopFrame>,
) -> NodeId {
    let pat_str = f.pat.to_token_stream().to_string();
    let expr_str = expr_to_short(&f.expr);
    let (ls, le) = span_lines(f);
    let header = g.add_with_span(
        CfgNodeKind::LoopHeader,
        format!(
            "for {} in {}",
            truncate(&pat_str, 20),
            truncate(&expr_str, 30)
        ),
        ls,
        le,
    );
    g.link(prev, header, None);
    let exit = g.add(CfgNodeKind::Statement, "join".into());

    // Emit the spine edge (header -> exit, "done") BEFORE processing
    // the body. Layout engines bias the first declared outgoing edge
    // toward the "natural" side - declaring exit first keeps the spine
    // straight down the page and pushes the loop body off to the side.
    g.link(header, exit, Some("done".into()));

    loops.push(LoopFrame {
        continue_target: header,
        break_target: exit,
    });
    let before = g.edges.len();
    let body_start = g.nodes.len();
    let body_end = process_block(g, &f.body, header, loops);
    let body_nodes: Vec<NodeId> = (body_start..g.nodes.len()).collect();
    if let Some(e) = g.edges.get_mut(before) {
        if e.from == header && e.label.is_none() {
            e.label = Some("next".into());
        }
    }
    if !is_terminating(&g.nodes[body_end].kind) && body_end != header {
        g.link(body_end, header, Some("loop".into()));
    }
    push_loop_group(g, header, body_nodes);
    loops.pop();
    exit
}

/// Add a loop body group, but only with nodes not already claimed by a
/// nested inner loop. Nested loops finish processing before their
/// enclosing loop, so by the time we get here any inner group has
/// already been pushed - a simple "exclude already-claimed" filter is
/// enough to keep each node in exactly one (innermost) group.
fn push_loop_group(g: &mut CfgGraph, header: NodeId, body_nodes: Vec<NodeId>) {
    let claimed: std::collections::HashSet<NodeId> = g
        .loop_groups
        .iter()
        .flat_map(|grp| grp.body_nodes.iter().copied())
        .collect();
    let body_nodes: Vec<NodeId> = body_nodes
        .into_iter()
        .filter(|n| !claimed.contains(n))
        .collect();
    if !body_nodes.is_empty() {
        g.loop_groups.push(CfgLoopGroup { header, body_nodes });
    }
}

fn relabel_first_outgoing(g: &mut CfgGraph, from: NodeId, label: &str) {
    if let Some(e) = g
        .edges
        .iter_mut()
        .rev()
        .find(|e| e.from == from && e.label.is_none())
    {
        e.label = Some(label.into());
    }
}

fn local_label(local: &syn::Local) -> String {
    let pat = local.pat.to_token_stream().to_string();
    if let Some(init) = &local.init {
        let rhs = expr_to_short(&init.expr);
        format!("let {} = {}", truncate(&pat, 30), truncate(&rhs, 50))
    } else {
        format!("let {}", truncate(&pat, 60))
    }
}

fn expr_to_short(expr: &syn::Expr) -> String {
    expr.to_token_stream().to_string()
}

fn truncate(s: &str, max: usize) -> String {
    let cleaned: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if cleaned.chars().count() <= max {
        cleaned
    } else {
        let truncated: String = cleaned.chars().take(max.saturating_sub(1)).collect();
        format!("{truncated}…")
    }
}

// --- mermaid emitter ---

/// Same as `to_mermaid`. Kept as a separate symbol to make it explicit
/// at the callsite that we want the rendered SVG to retain stable node
/// ids the JS click handler can hook on. Currently identical because
/// mermaid emits `id="flowchart-n5-..."` automatically; if we ever
/// switch renderers this is the spot to add per-node `click` directives.
pub fn to_mermaid_with_clicks(g: &CfgGraph) -> String {
    to_mermaid(g)
}

pub fn to_mermaid(g: &CfgGraph) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "%% planecode CFG: {}", g.function);
    let _ = writeln!(out, "flowchart TD");

    for n in &g.nodes {
        let id = node_id(n.id);
        let label = mermaid_label(&n.label);
        let shape = match n.kind {
            CfgNodeKind::Entry | CfgNodeKind::Exit => format!("([\"{label}\"])"),
            CfgNodeKind::Branch { .. } => format!("{{{{\"{label}\"}}}}"),
            CfgNodeKind::Call { .. } => format!("[/\"{label}\"/]"),
            CfgNodeKind::Return => format!("([\"{label}\"])"),
            CfgNodeKind::LoopHeader => format!("[(\"{label}\")]"),
            CfgNodeKind::LoopJump { .. } => format!("[\"{label}\"]"),
            CfgNodeKind::Statement => format!("[\"{label}\"]"),
        };
        let _ = writeln!(out, "    {id}{shape}");
    }

    let _ = writeln!(
        out,
        "    classDef terminal fill:#1f6feb,stroke:#58a6ff,color:#fff"
    );
    let _ = writeln!(
        out,
        "    classDef returns fill:#8b3a3a,stroke:#f85149,color:#fff"
    );
    let _ = writeln!(
        out,
        "    classDef branch fill:#3d2c00,stroke:#d29922,color:#fff"
    );
    let _ = writeln!(
        out,
        "    classDef calls fill:#1c2b22,stroke:#3fb950,color:#fff"
    );
    let _ = writeln!(
        out,
        "    classDef loops fill:#1f3a52,stroke:#79c0ff,color:#fff"
    );
    let _ = writeln!(
        out,
        "    classDef jumps fill:#3d1f1f,stroke:#ff7b72,color:#fff"
    );
    for n in &g.nodes {
        let id = node_id(n.id);
        let class = match n.kind {
            CfgNodeKind::Entry | CfgNodeKind::Exit => "terminal",
            CfgNodeKind::Return => "returns",
            CfgNodeKind::Branch { .. } => "branch",
            CfgNodeKind::Call { .. } => "calls",
            CfgNodeKind::LoopHeader => "loops",
            CfgNodeKind::LoopJump { .. } => "jumps",
            CfgNodeKind::Statement => continue,
        };
        let _ = writeln!(out, "    class {id} {class}");
    }

    for e in &g.edges {
        let from = node_id(e.from);
        let to = node_id(e.to);
        match &e.label {
            Some(label) => {
                // Edge labels need the same escaping as node labels AND
                // must be wrapped in double-quotes; otherwise mermaid
                // parses unquoted parens / dots in something like
                // `|Some (port)|` as syntax.
                let safe = mermaid_label(&label.replace('|', "/"));
                let _ = writeln!(out, "    {from} -->|\"{safe}\"| {to}");
            }
            None => {
                let _ = writeln!(out, "    {from} --> {to}");
            }
        }
    }

    out
}

fn node_id(id: NodeId) -> String {
    format!("n{id}")
}

/// Make a label safe to embed inside `"..."` in a mermaid node shape.
/// Mermaid's SVG renderer treats `<` / `>` as markup even inside quoted
/// labels, so we swap them to unicode angle-quote glyphs that read
/// identically. `&` starts an HTML entity; `"` would close the wrapper;
/// `\` is mermaid's escape; all get replaced. Parens, dots, brackets,
/// and braces are fine inside a quoted label.
fn mermaid_label(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '"' => '\'',
            '\\' => ' ',
            '<' => '‹',
            '>' => '›',
            '&' => '+',
            other => other,
        })
        .collect()
}
