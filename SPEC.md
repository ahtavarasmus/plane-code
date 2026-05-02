# planecode v1 spec

A code ontology + tool surface for AI coding agents working on Rust codebases.

The agent has only two tools: `query_ontology` and `update_ontology`. No grep,
no Read, no file listing. Everything goes through the graph.

## Scope

**In scope (v1):**
- Single Rust workspace (one Cargo.toml at root, possibly with members).
- Workspace-only indexing. Dependency crates are referenced by canonical path
  but not indexed in depth.
- Four object types: `Function`, `Type`, `Trait`, `Module`.

**Out of scope (v1):**
- Non-Rust files (Cargo.toml, *.sql, *.yaml, *.json, build.rs).
- Macro expansion. Macro invocation sites are treated as opaque.
- Field-level read/write tracking as a queryable entity (fields are returned
  embedded inside Type responses, not addressable on their own).
- Const, Static, Macro as first-class object types.
- Cross-crate trait implementor lookup.

If a change requires editing anything out of scope, the agent surfaces it as
a TODO for the user instead of attempting it.

## Backend

Graph indexer built on `ra_ap_*` crates (rust-analyzer as a library). Indexer
runs once on workspace open, then incrementally on file save. Graph stored
in-memory with sled or sqlite for persistence between sessions.

Embedding index: function search documents (name + module_path + first doc
sentence + signature) embedded with a small local model (e.g. all-MiniLM-L6
or similar). Hybrid retrieval: cosine similarity + BM25 lexical overlap +
small centrality boost from caller count.

## Tool 1: query_ontology

```
Description: Query the Rust code ontology. Covers functions, types, traits,
and modules in the current workspace. Use the query field for semantic
search by intent ("verify JWT tokens"), filters for exact lookup once an
anchor is known, and include_links to expand neighborhoods.

Param: object_type
Type: string
Required: yes
Notes: One of: Function, Type, Trait, Module
─────────────────────────────────────────────
Param: query
Type: string | null
Required: no
Default: null
Notes: Free-text semantic search. Ranked across name, module path, doc
  comment first sentence, and signature. Returns top results by hybrid
  embedding + lexical score.
─────────────────────────────────────────────
Param: filters
Type: object | null
Required: no
Default: null
Notes: Exact-match filters. Keys vary by object_type.
  Function: name, module_path, visibility, is_async, is_test
  Type:     name, kind (struct|enum|union|alias), module_path
  Trait:    name, module_path
  Module:   path
─────────────────────────────────────────────
Param: include_links
Type: string[] | null
Required: no
Default: null
Notes: Attach linked objects (see below). Linked objects are returned
  shallow (name, module_path, signature, doc_summary) by default.
─────────────────────────────────────────────
Param: limit
Type: integer
Required: no
Default: 10
Notes: Max 50.
```

### include_links options

```
Function -> callers, callees, tests, module
Type     -> fields, impls, used_by_functions, module
Trait    -> methods, implementors, module
Module   -> functions, types, traits, submodules
```

### Response shape

Function:
```json
{
  "object_type": "Function",
  "name": "verify_jwt_signature",
  "module_path": "auth::tokens",
  "file": "src/auth/tokens.rs",
  "line_start": 42,
  "line_end": 71,
  "signature": "pub fn verify_jwt_signature(token: &str, key: &PublicKey) -> Result<Claims, AuthError>",
  "doc": "Verifies an RS256-signed JWT and returns its claims...",
  "attributes": ["#[tracing::instrument]"],
  "is_async": false,
  "is_unsafe": false,
  "is_test": false,
  "visibility": "pub",
  "body": "<full Rust source of the function>",
  "links": {
    "callers": [
      { "name": "authenticate_request", "module_path": "auth::middleware",
        "signature": "...", "doc_summary": "..." }
    ],
    "callees": [...],
    "tests": [...]
  },
  "dispatch_notes": []
}
```

`dispatch_notes` is non-empty when call edges are imprecise (dyn dispatch,
generic trait bounds). Each note names the callee and explains why the
target set is non-singleton. Empty array means all edges are precise.

Type:
```json
{
  "object_type": "Type",
  "name": "Claims",
  "kind": "struct",
  "module_path": "auth::tokens",
  "file": "src/auth/tokens.rs",
  "line_start": 12,
  "line_end": 24,
  "doc": "...",
  "visibility": "pub",
  "fields": [
    { "name": "sub", "type": "String", "visibility": "pub", "doc": "..." },
    { "name": "exp", "type": "i64",    "visibility": "pub", "doc": "..." }
  ],
  "derives": ["Debug", "Clone", "Serialize", "Deserialize"],
  "source": "<full Rust source of the type definition>",
  "links": {...}
}
```

Trait, Module: analogous shapes.

## Tool 2: update_ontology

```
Description: Apply a structural edit to the Rust workspace. Edits are
addressed by ontology identifiers (name + module_path), not file paths.
Every edit is followed by a compile check; errors are returned in the
response so the agent can iterate.

Param: operation
Type: string
Required: yes
Notes: One of: replace_body, add_function, rename
─────────────────────────────────────────────
Param: target
Type: object
Required: yes
Notes: Identifies the entity to edit. Shape depends on operation.
  replace_body: { name, module_path }
  add_function: { module_path, impl_for_type? }
  rename:       { object_type, name, module_path }
─────────────────────────────────────────────
Param: payload
Type: object
Required: yes
Notes: Operation-specific payload.
  replace_body: { new_body: "<rust source between { and } of fn body>" }
  add_function: { source: "<full fn definition including signature>" }
  rename:       { new_name: string }
─────────────────────────────────────────────
Param: dry_run
Type: boolean
Required: no
Default: false
Notes: If true, computes the diff and runs compile check but does not
  write to disk or update the graph.
```

### Response shape

```json
{
  "success": true,
  "rollback_reason": null,
  "files_changed": [
    { "path": "src/auth/tokens.rs", "diff": "<unified diff>" }
  ],
  "compile_status": "ok",
  "graph_diff": {
    "nodes_added": [],
    "nodes_removed": [],
    "edges_changed": [
      { "from": "auth::middleware::authenticate_request",
        "to": "auth::tokens::verify_jwt_signature",
        "kind": "calls", "change": "renamed_target" }
    ]
  }
}
```

When the edit succeeds at the parse level but the workspace doesn't pass
`cargo check`, the edit is still applied and `compile_status` carries the
errors:

```json
{
  "success": true,
  "rollback_reason": null,
  "files_changed": [...],
  "compile_status": {
    "errors": [
      { "file": "src/auth/middleware.rs", "line": 88, "column": 22,
        "message": "expected 2 arguments, found 1",
        "code": "E0061" }
    ]
  },
  "graph_diff": {...}
}
```

Rollback to pre-edit state happens only on hard failures: unparseable input,
ambiguous rename, or invalid target. In those cases:

```json
{
  "success": false,
  "rollback_reason": "parse_error" | "ambiguous_rename" | "invalid_target",
  "files_changed": [],
  "compile_status": null,
  "graph_diff": null,
  "details": "<failure-specific info>"
}
```

## Compile-check policy

Edits are validated in three layers:

1. **Parse check (atomic, always enforced).** Every edit must produce
   parseable Rust. If parsing fails, disk and graph roll back to pre-edit
   state. The graph indexer cannot represent unparseable code, so this is
   non-negotiable.

2. **Type/borrow check (returned, never auto-rolled-back).** Every
   successful edit triggers `cargo check` on the affected crate. Errors
   are returned in `compile_status` but the workspace stays edited. This
   is what makes multi-step changes possible: an agent renaming a function
   parameter type, then walking through callers, will see errors during
   the intermediate states and watch them resolve as it works through the
   change.

3. **Tests (opt-in).** Not run on edit. Triggered separately when the
   agent considers a logical change complete.

Cascading operations (`rename`, and `change_signature` if added later) are
designed to be self-complete: they update every reference in one call. If
they succeed at the rust-analyzer level the workspace should still
type-check, so they behave as if atomic against compile. Surgical
operations (`replace_body`, `add_function`) are the ones that allow
broken intermediates.

Workflow this enables:
1. Agent plans a multi-step change.
2. Applies edits one at a time. Each response carries current
   `compile_status`.
3. Watches the error set converge to empty.
4. `compile_status: "ok"` confirms the change is structurally complete.

## Operation semantics

**replace_body.** Body between the outermost `{` and `}` is replaced.
Signature, attributes, doc comment, and surrounding whitespace stay intact.
Compile check runs on the containing crate. If the new body changes the set
of callees, the graph is re-indexed for the affected function only.

**add_function.** Insert at the end of the module body, or at the end of
the named impl block when `impl_for_type` is provided. The full source must
include `fn`, signature, and body. Indexer assigns a position, formats
according to rustfmt, and re-indexes the new node.

**rename.** Uses rust-analyzer's rename refactor. All references update
atomically: definition, every call site, doc comment references where
unambiguous, and re-exports. Tests are included. If rust-analyzer reports
ambiguity, the operation aborts with `success: false` and a list of
ambiguous sites for the agent to resolve manually with replace_body.

## What is deliberately not in v1

These are the next obvious additions but we ship without them and see
what hurts:

- `change_signature` operation. rust-analyzer supports it but the refactor
  is rough and call-site rewrites can fail on generic callers. Workaround:
  rename old, add_function new, replace_body on each caller manually.
- `delete_function`, `add_field`, `modify_field` operations.
- Macro-aware indexing.
- `add_dependency` tool for Cargo.toml. Likely v1.5.
- Synthesized doc comments for undocumented functions. Add only if
  retrieval quality data shows it matters.
- Multi-workspace / multi-crate-graph operations.

## Open questions to resolve before building

1. Embedding model choice: local (sentence-transformers/all-MiniLM via
   ort or fastembed-rs) vs. hosted API. Local is the default; hosted
   only if local quality is poor.
2. Index storage: pure in-memory vs. sled vs. sqlite. Pick based on
   workspace size targets. In-memory is fine up to ~50k functions.
3. How to surface "this trait is from external crate X, implementors
   not indexed" without making every result noisy.
4. How the agent gets the workspace root: passed at session start, or
   discovered.
