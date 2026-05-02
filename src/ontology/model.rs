use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Stable identity for a compile error. Two errors are "the same" if
/// they hash equal here. We deliberately include the full message so
/// that a code like `E0061` with two different "expected N args" texts
/// stays distinguishable across edits.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ErrorKey {
    pub file: String,
    pub line: u64,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub module_path: String,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: String,
    pub doc: String,
    pub attributes: Vec<String>,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_test: bool,
    pub visibility: String,
    pub body: String,
    /// Names referenced in calls inside the body. Approximate: any path
    /// expression or method call yields its last segment here. Trait
    /// dispatch is not resolved.
    pub callees: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub visibility: String,
    pub doc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Type {
    pub name: String,
    pub module_path: String,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    /// One of: struct | enum | union | alias
    pub kind: String,
    pub doc: String,
    pub visibility: String,
    /// For structs: named/positional fields. For enums: variants.
    pub fields: Vec<Field>,
    pub derives: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trait {
    pub name: String,
    pub module_path: String,
    pub file: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub doc: String,
    pub visibility: String,
    pub methods: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub path: String,
    pub file: PathBuf,
    pub functions: Vec<String>,
    pub types: Vec<String>,
    pub traits: Vec<String>,
    pub submodules: Vec<String>,
}

/// A region of a file. `Indexed` regions are owned by a structural ontology
/// item (Function/Type/Trait); file ops must not overlap them. `Gap` regions
/// are free for `edit_file` to touch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedSpan {
    pub line_start: usize,
    pub line_end: usize,
    /// "Function" | "Type" | "Trait"
    pub kind: String,
    /// Fully-qualified ontology key, e.g. `myapp::auth::verify_jwt`.
    pub owner: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    /// Path relative to the workspace root.
    pub path: String,
    pub extension: String,
    /// Best-guess language label: rust, toml, markdown, yaml, json, sql, text, binary.
    pub language: String,
    pub bytes: u64,
    /// File contents. Empty for files classified as binary or above the size cap.
    pub content: String,
    /// Spans owned by structural items, sorted by line_start. Empty for non-Rust files.
    pub indexed_spans: Vec<IndexedSpan>,
}

#[derive(Debug, Default)]
pub struct Ontology {
    pub workspace: PathBuf,
    pub crate_name: String,
    /// Keyed by `<module_path>::<name>`, e.g. `myapp::auth::verify_jwt`.
    /// Methods on impl blocks are keyed `<module_path>::<TypeName>::<method>`.
    pub functions: HashMap<String, Function>,
    pub types: HashMap<String, Type>,
    pub traits: HashMap<String, Trait>,
    pub modules: HashMap<String, Module>,
    /// Keyed by path relative to workspace root. Includes every walked file
    /// (Rust source plus the long tail: Cargo.toml, READMEs, SQL, fixtures).
    pub files: HashMap<String, File>,
    /// Snapshot of cargo check errors as of the most recent classified
    /// run. Used to tag errors after each edit as "introduced by this
    /// edit" vs "pre-existing." Initialized at session start in main.rs
    /// (or remains empty if the agent never edits, e.g. --flow / --debug).
    pub prev_errors: HashSet<ErrorKey>,
}

impl Ontology {
    pub fn index(workspace: &Path) -> Result<Self> {
        crate::ontology::indexer::index(workspace)
    }

    /// Look up a function by name with optional module_path qualifier.
    /// Returns None if no match or if the short name is ambiguous.
    pub fn function(&self, name: &str, module_path: Option<&str>) -> Option<&Function> {
        if let Some(mp) = module_path {
            return self.functions.get(&format!("{mp}::{name}"));
        }
        let mut hits = self.functions.values().filter(|f| f.name == name);
        let first = hits.next()?;
        if hits.next().is_some() {
            return None;
        }
        Some(first)
    }
}
