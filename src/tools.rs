//! Tool schemas exposed to the LLM. Three tools cover the whole surface:
//!   - query_ontology  : reads (Function, Type, Trait, Module, File)
//!   - update_ontology : writes (replace_body, add_function, rename, edit_file,
//!                       create_file, delete_file)
//!   - run_cargo       : executes (check | build | run | test)
//!
//! Filter enums for bounded sets (file paths, module paths, trait names,
//! file extensions/languages) are populated dynamically from the live
//! ontology so the LLM only ever sees values that actually exist.

use crate::ontology::Ontology;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// Build the live tool schema set for the current ontology snapshot.
/// Called once per agent turn so dynamic enum filters stay fresh after
/// every edit.
pub fn tool_definitions(ont: &Ontology) -> Vec<Value> {
    vec![
        query_ontology_def(ont),
        update_ontology_def(ont),
        run_cargo_def(),
        show_flow_def(),
    ]
}

fn query_ontology_def(ont: &Ontology) -> Value {
    let module_paths = sorted_unique(ont.modules.keys().cloned());
    let trait_paths = sorted_unique(ont.traits.keys().cloned());
    let file_paths = sorted_unique(ont.files.keys().cloned());
    let extensions = sorted_unique(ont.files.values().map(|f| f.extension.clone()))
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    let languages = sorted_unique(ont.files.values().map(|f| f.language.clone()));

    let mut filter_props = serde_json::Map::new();
    filter_props.insert("name".into(), json!({ "type": "string" }));
    filter_props.insert(
        "module_path".into(),
        with_enum(json!({ "type": "string" }), &module_paths, 250),
    );
    filter_props.insert(
        "path".into(),
        with_enum(json!({ "type": "string" }), &file_paths, 500),
    );
    filter_props.insert(
        "trait_path".into(),
        with_enum(json!({ "type": "string" }), &trait_paths, 250),
    );
    filter_props.insert(
        "extension".into(),
        with_enum(json!({ "type": "string" }), &extensions, 100),
    );
    filter_props.insert(
        "language".into(),
        with_enum(json!({ "type": "string" }), &languages, 50),
    );
    filter_props.insert(
        "kind".into(),
        json!({ "type": "string", "enum": ["struct", "enum", "union", "alias"] }),
    );
    filter_props.insert(
        "visibility".into(),
        json!({ "type": "string", "enum": ["pub", "pub(crate)", "pub(super)", ""] }),
    );
    filter_props.insert("is_async".into(), json!({ "type": "boolean" }));
    filter_props.insert("is_test".into(), json!({ "type": "boolean" }));

    json!({
        "type": "function",
        "function": {
            "name": "query_ontology",
            "description":
                "EXPLORATION phase. The graph is your eyes - find a \
                 candidate entity by intent, then traverse the edges to \
                 understand the area before editing anything.\n\
                 \n\
                 Top-level fields are exactly: object_type, keywords, \
                 filters, include_links, limit. Nothing else - any other \
                 top-level key is rejected. Per-entity fields like \
                 `name`, `path`, `module_path`, `kind`, `visibility`, \
                 `is_async`, `is_test`, `extension`, `language` MUST go \
                 inside the `filters` object, not at the top level.\n\
                 \n\
                 Examples:\n\
                 - {object_type: \"File\", filters: {path: \"src/main.rs\"}}\n\
                 - {object_type: \"Function\", filters: {name: \"verify_token\"}}\n\
                 - {object_type: \"Function\", keywords: \"verify token jwt auth\"}\n\
                 - {object_type: \"Module\", filters: {path: \"auth\"}, include_links: [\"functions\", \"types\"]}\n\
                 \n\
                 Workflow: start with `keywords` to locate an anchor entity \
                 when you don't yet know the canonical name. Once you have \
                 a name+module_path from the results, switch to `filters` \
                 for exact lookup, and use `include_links` to fan out to \
                 neighbors (callers, callees, tests, fields, impls, \
                 methods, submodules). If a search returns nothing, try \
                 different keywords - synonyms, broader or narrower terms - \
                 don't guess a name.\n\
                 \n\
                 The ontology is the ONLY way to read structural Rust \
                 source. Function bodies, type definitions, and trait \
                 sources come back from object_type=Function/Type/Trait. \
                 File responses for .rs files give an `outline` with gap \
                 content and signatures only - bodies are hidden so the \
                 graph stays canonical.",
            "parameters": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "object_type": {
                        "type": "string",
                        "enum": ["Function", "Type", "Trait", "Module", "File"],
                    },
                    "keywords": {
                        "type": "string",
                        "description":
                            "Whitespace-separated keywords - this is grep, not \
                             natural-language search. Each token is matched \
                             case-insensitive against name, module_path, \
                             signature, and doc; entries scoring on more tokens \
                             rank higher.\n\
                             \n\
                             Pick 2-5 short tokens (verbs, nouns, domain terms) \
                             that could plausibly appear in the code. Include \
                             SYNONYMS since you don't know which the codebase \
                             uses - e.g. for 'how does signup work' search \
                             `register signup create user account` so all of \
                             register_user / signup_handler / create_account \
                             surface. Don't pass full phrases like 'how does \
                             auth work' - they match poorly.\n\
                             \n\
                             For exact lookups by name use `filters: {name: ...}` \
                             instead; this field is for the FIRST search when \
                             you don't yet know the entity name.",
                    },
                    "filters": {
                        "type": "object",
                        "additionalProperties": false,
                        "description":
                            "Exact-match filters. Per object_type, valid keys are:\n\
                             - Function: name, module_path, visibility, is_async, is_test\n\
                             - Type:     name, kind, module_path\n\
                             - Trait:    name, module_path\n\
                             - Module:   path\n\
                             - File:     path, extension, language\n\
                             Always nest filter keys here - putting them at the \
                             top level (e.g. {object_type: \"File\", path: \"x\"}) \
                             will be rejected.",
                        "properties": filter_props,
                    },
                    "include_links": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description":
                            "Edges to traverse from each result.\n\
                             - Function: callers, callees, tests, module\n\
                             - Type:     fields, impls, used_by_functions, module\n\
                             - Trait:    methods, implementors, module\n\
                             - Module:   functions, types, traits, submodules\n\
                             - File:     no links (use regions / matches instead)"
                    },
                    "limit": { "type": "integer" },
                },
                "required": ["object_type"]
            }
        }
    })
}

fn update_ontology_def(ont: &Ontology) -> Value {
    let module_paths = sorted_unique(ont.modules.keys().cloned());
    let function_modules =
        sorted_unique(ont.functions.values().map(|f| f.module_path.clone()));
    let merged_modules = {
        let mut s: BTreeSet<String> = BTreeSet::new();
        s.extend(module_paths);
        s.extend(function_modules);
        s.into_iter().collect::<Vec<_>>()
    };
    let file_paths = sorted_unique(ont.files.keys().cloned());

    let target_props = json!({
        "name": { "type": "string" },
        "module_path": with_enum(json!({ "type": "string" }), &merged_modules, 250),
        "path": with_enum(json!({ "type": "string" }), &file_paths, 500),
        "object_type": { "type": "string", "enum": ["Function", "Type", "Trait"] },
        "impl_for_type": { "type": "string" },
    });

    json!({
        "type": "function",
        "function": {
            "name": "update_ontology",
            "description":
                "ACTION phase. Make the edit. Pick the operation by what \
                 you're changing:\n\
                 \n\
                 - replace_body  : function body only; signature, attrs, \
                                   doc unchanged.\n\
                 - replace_item  : whole item (attrs + doc + signature + \
                                   body). Use for adding/removing \
                                   decorators, editing a doc comment, or \
                                   altering a signature. Works on \
                                   Function, Type, Trait.\n\
                 - rename        : rename + cascade across word-boundary \
                                   matches. Pass `dry_run: true` to \
                                   preview if you're unsure.\n\
                 - add_function  : insert a new fn into a module or impl \
                                   block.\n\
                 - edit_file     : non-Rust files (Cargo.toml, SQL, \
                                   READMEs) and Rust gap regions (use \
                                   statements, const/static, comments). \
                                   Blocked over indexed regions - query \
                                   the File first, pick a `gap`.\n\
                 - create_file   : new files. .rs under src/ get picked up \
                                   on reindex.\n\
                 - delete_file   : delete files. Blocked when the file \
                                   contains indexed items.\n\
                 \n\
                 Every successful structural edit carries an `affected` \
                 block (callers + tests). Read it. If the impact list \
                 surprises you, re-query the affected entities to plan a \
                 follow-up edit.\n\
                 \n\
                 Parse failures roll back atomically; compile errors are \
                 surfaced in compile_status but DO NOT roll back, so \
                 multi-step changes can pass through broken intermediate \
                 states. Verification (run_cargo) is the boundary: \
                 multiple edits in a row are fine, just call run_cargo \
                 once before responding to the user. The harness enforces \
                 this gate - your final text response is held back until \
                 a run_cargo follows the most recent edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "replace_body",
                            "replace_item",
                            "add_function",
                            "rename",
                            "edit_file",
                            "create_file",
                            "delete_file"
                        ],
                        "description":
                            "replace_body: change a function body, signature/attrs/doc unchanged. \
                             replace_item: rewrite an indexed item's full presentation \
                                 (attrs + doc + signature + body) with new source. \
                                 Use when adding decorators or changing a signature/doc. \
                                 Works on Function, Type, Trait. \
                             add_function: insert a new function into a module or impl. \
                             rename: rename + cascade (naive word-boundary). \
                             edit_file: byte-level edit, blocked over indexed regions. \
                             create_file: new file at path with content. \
                             delete_file: delete file (blocked if it contains indexed items)."
                    },
                    "target": {
                        "type": "object",
                        "description":
                            "Identifies the entity. Shape depends on operation. \
                             replace_body: { name, module_path }. \
                             replace_item: { object_type, name, module_path }. \
                             add_function: { module_path, impl_for_type? }. \
                             rename:       { object_type, name, module_path }. \
                             edit_file:    { path }. \
                             create_file:  { path }. \
                             delete_file:  { path }.",
                        "properties": target_props,
                    },
                    "payload": {
                        "type": "object",
                        "description":
                            "Operation-specific. \
                             replace_body: { new_body: \"<ONLY the inner \
                                 statements/expressions; NOT the fn signature \
                                 or outer braces. e.g. for `fn f() -> i32 \
                                 { 1 + 2 }` send new_body=\\\"1 + 2\\\">\" }. \
                                 If you need to change attrs/doc/signature \
                                 too, use replace_item instead. \
                             replace_item: { source: '<full item source: \
                                 leading attrs + doc comments + signature + \
                                 body>' }. \
                             add_function: { source: '<full fn definition>' }. \
                             rename:       { new_name }. \
                             edit_file:    one of {find, replace} | \
                                           {line_start, line_end, replacement} | \
                                           {content} (whole-file). \
                             create_file: { content }. \
                             delete_file: {} (no payload).",
                    },
                    "dry_run": { "type": "boolean" }
                },
                "required": ["operation", "target", "payload"]
            }
        }
    })
}

fn run_cargo_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "run_cargo",
            "description":
                "VERIFICATION phase. Required after any update_ontology \
                 before your final response. The harness gates the loop \
                 here: if updates were made and the most recent run_cargo \
                 is older than the most recent update, your text response \
                 is held back and you'll be asked to verify first.\n\
                 \n\
                 Pick a command:\n\
                 - check : compile-only gate; fastest. Sufficient if your \
                           edit was structural and you trust the type \
                           system to catch breakage.\n\
                 - test  : runs `cargo test`. Use after any change to \
                           function bodies, signatures, or data shapes - \
                           it confirms runtime behavior. Parsed summary \
                           comes back as passed/failed/ignored/failures.\n\
                 - build : full build incl. linking. Slow; rarely needed.\n\
                 - run   : execute the binary. Pass program args via \
                           `args`, optional stdin via `stdin`. Captured \
                           stdout/stderr/exit_code in the response.\n\
                 \n\
                 One run_cargo covers all preceding edits. Multiple edits \
                 in a row are fine; verify once at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["check", "build", "run", "test"],
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description":
                            "Extra arguments. For check/build: cargo flags \
                             (--release, --features=...). For test: filter \
                             names or cargo flags. For run: program args \
                             (placed after `--`).",
                    },
                    "stdin": {
                        "type": "string",
                        "description":
                            "Optional stdin to pipe to the program. Only used by `run`.",
                    },
                },
                "required": ["command"]
            }
        }
    })
}

fn show_flow_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "show_flow",
            "description":
                "Opens an interactive control-flow diagram in the operator's \
                 browser. The diagram is the canonical answer to 'how does \
                 X work' or 'where does Y happen' - clickable, expandable \
                 inline into callees, and faster for the operator to read \
                 than any prose summary you could write.\n\
                 \n\
                 Reach for this tool whenever:\n\
                 \n\
                 (a) EXPLANATION. The operator asks how some piece of code \
                 works, what flows through a module, where a behavior \
                 lives, or how a system is wired. Use query_ontology first \
                 to locate the right entry-point function or module by \
                 intent ('user signup' -> probably an auth handler or a \
                 service function), then show_flow on it. The diagram \
                 lets them drill down themselves; you don't need to \
                 narrate every branch.\n\
                 \n\
                 (b) POST-EDIT REVIEW. After run_cargo passes for an edit \
                 that reshaped control flow (new branches, loops, match \
                 arms, await points). Cargo proves it compiles; show_flow \
                 lets the operator eyeball that the new flow matches \
                 their intent.\n\
                 \n\
                 The tool returns immediately - the operator sees the \
                 diagram and either accepts silently or corrects in their \
                 next message. So always pair the call with a short \
                 textual orientation ('opened the flow for X; the entry \
                 is at the top, branches into Y / Z'); don't dump a long \
                 explanation that the diagram already shows.\n\
                 \n\
                 target options:\n\
                 - function name: `verify_token` (or full path \
                   `auth::verify_token` if ambiguous)\n\
                 - module path: `backend::services::auth` (opens the \
                   skyline focused on that module)\n\
                 - literal `skyline` for the whole-workspace call graph \
                   (use for architecture-level questions or post-edits \
                   that touch many functions)\n\
                 \n\
                 Skip for: trivial questions answerable in one sentence, \
                 rename/doc/signature-only edits, anything where the \
                 SHAPE of execution doesn't carry the answer.",
            "parameters": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "target": {
                        "type": "string",
                        "description":
                            "Function name, fully-qualified path, module \
                             path, or `skyline`."
                    }
                },
                "required": ["target"]
            }
        }
    })
}

fn sorted_unique<I: IntoIterator<Item = String>>(it: I) -> Vec<String> {
    let s: BTreeSet<String> = it.into_iter().collect();
    s.into_iter().collect()
}

/// Attach an `enum` constraint to a JSON-Schema string property only if the
/// candidate set is non-empty and below `max`. Above `max`, the schema stays
/// open and the agent relies on hints in tool responses for validity.
fn with_enum(mut base: Value, values: &[String], max: usize) -> Value {
    if values.is_empty() || values.len() > max {
        return base;
    }
    if let Some(obj) = base.as_object_mut() {
        obj.insert(
            "enum".into(),
            Value::Array(values.iter().cloned().map(Value::String).collect()),
        );
    }
    base
}

pub fn system_prompt(crate_name: &str) -> String {
    format!(
        r#"You are planecode, a Rust coding agent operating on the workspace
crate `{crate_name}`. The loop has three phases - explore (query_ontology),
act (update_ontology), verify (run_cargo). show_flow is the visual
companion: when the operator asks how some piece of code works or where
a behavior lives, find the entry point with query_ontology and then
open it with show_flow rather than reconstructing the flow in prose -
the diagram is faster and lets them drill in themselves. Also reach
for it after edits that reshape control flow, as a post-verify review.
Each tool's description explains when to reach for it.

Paths in the graph are crate-prefixed: `{crate_name}::auth::verify_token`,
methods nest under their type (`{crate_name}::math::Point::distance`).
Path filters accept full paths or suffixes at module boundaries (`auth`
matches `{crate_name}::auth`).

Stay scoped: only edit what the user asked for. Pre-existing errors in
files you didn't touch are TODOs to mention, not problems to chase.
Empty responses are catastrophic - if uncertain, write a paragraph.
"#
    )
}
