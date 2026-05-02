//! Workspace indexer. Walks .rs files, parses with `syn`, and populates
//! the in-memory ontology graph.
//!
//! Approximations to be aware of: trait dispatch is not resolved, generic
//! callees fall back to last path segment, macro invocations are opaque,
//! and `mod foo;` declarations rely on walkdir to find the corresponding
//! file rather than being followed explicitly.

use crate::ontology::model::{
    Field, File as FileEntity, Function, IndexedSpan, Module, Ontology, Trait, Type,
};
use anyhow::Result;
use proc_macro2::TokenStream;
use quote::ToTokens;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use syn::{spanned::Spanned, visit::Visit, Item, ItemEnum, ItemFn, ItemStruct, ItemTrait};
use walkdir::WalkDir;

/// Files larger than this (in bytes) get registered as a File entity but with
/// empty `content`. Keeps the ontology snappy on big generated artifacts.
const MAX_FILE_CONTENT_BYTES: u64 = 512 * 1024;

pub fn index(workspace: &Path) -> Result<Ontology> {
    let crate_name = read_crate_name(workspace).unwrap_or_else(|_| "crate".to_string());
    let mut ont = Ontology {
        workspace: workspace.to_path_buf(),
        crate_name: crate_name.clone(),
        ..Default::default()
    };

    for entry in WalkDir::new(workspace)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !is_skipped(e.path()))
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        // Always register the file in the File entity index (Rust or not).
        register_file(&mut ont, workspace, path);

        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let base_module = match derive_base_module(workspace, &crate_name, path) {
            Some(m) => m,
            None => continue,
        };
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let file = match syn::parse_file(&source) {
            Ok(f) => f,
            Err(err) => {
                tracing::warn!("skip unparseable {}: {}", path.display(), err);
                continue;
            }
        };
        for item in &file.items {
            index_item(&mut ont, item, path, &base_module);
        }
    }

    // Now that every structural item has been indexed, attach indexed_spans
    // to each Rust File. The list is sorted by line_start so query.rs can
    // emit gap/indexed regions in document order without re-sorting.
    populate_indexed_spans(&mut ont);

    Ok(ont)
}

fn register_file(ont: &mut Ontology, workspace: &Path, path: &Path) {
    let rel = match path.strip_prefix(workspace) {
        Ok(r) => r.to_string_lossy().to_string(),
        Err(_) => return,
    };
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let language = classify_language(&rel, &extension);
    let bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let content = if language == "binary" || bytes > MAX_FILE_CONTENT_BYTES {
        String::new()
    } else {
        fs::read_to_string(path).unwrap_or_default()
    };
    ont.files.insert(
        rel.clone(),
        FileEntity {
            path: rel,
            extension,
            language,
            bytes,
            content,
            indexed_spans: vec![],
        },
    );
}

fn classify_language(rel_path: &str, extension: &str) -> String {
    match extension {
        "rs" => "rust".into(),
        "toml" => "toml".into(),
        "md" | "markdown" => "markdown".into(),
        "yaml" | "yml" => "yaml".into(),
        "json" => "json".into(),
        "sql" => "sql".into(),
        "txt" => "text".into(),
        "lock" => "toml".into(),
        "" => {
            // Heuristic for extensionless files like README, LICENSE, Makefile.
            let lower = rel_path.to_lowercase();
            if lower.ends_with("readme")
                || lower.ends_with("license")
                || lower.ends_with("makefile")
                || lower.ends_with(".gitignore")
            {
                "text".into()
            } else {
                "binary".into()
            }
        }
        // Common binary types we don't want to load.
        "png" | "jpg" | "jpeg" | "gif" | "ico" | "wasm" | "so" | "dylib" | "rlib" | "o"
        | "a" | "exe" | "bin" => "binary".into(),
        _ => "text".into(),
    }
}

fn populate_indexed_spans(ont: &mut Ontology) {
    use std::collections::HashMap as Map;
    let mut by_path: Map<String, Vec<IndexedSpan>> = Map::new();

    for (key, f) in &ont.functions {
        if let Some(rel) = relative_string(&ont.workspace, &f.file) {
            by_path.entry(rel).or_default().push(IndexedSpan {
                line_start: f.line_start,
                line_end: f.line_end,
                kind: "Function".into(),
                owner: key.clone(),
            });
        }
    }
    for (key, t) in &ont.types {
        if let Some(rel) = relative_string(&ont.workspace, &t.file) {
            by_path.entry(rel).or_default().push(IndexedSpan {
                line_start: t.line_start,
                line_end: t.line_end,
                kind: "Type".into(),
                owner: key.clone(),
            });
        }
    }
    for (key, t) in &ont.traits {
        if let Some(rel) = relative_string(&ont.workspace, &t.file) {
            by_path.entry(rel).or_default().push(IndexedSpan {
                line_start: t.line_start,
                line_end: t.line_end,
                kind: "Trait".into(),
                owner: key.clone(),
            });
        }
    }

    for (rel, mut spans) in by_path {
        spans.sort_by_key(|s| (s.line_start, s.line_end));
        if let Some(file) = ont.files.get_mut(&rel) {
            file.indexed_spans = spans;
        }
    }
}

fn relative_string(workspace: &Path, file: &Path) -> Option<String> {
    file.strip_prefix(workspace)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

fn is_skipped(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str();
        s == "target" || s == ".git" || s == "node_modules"
    })
}

fn read_crate_name(workspace: &Path) -> Result<String> {
    let cargo = workspace.join("Cargo.toml");
    let txt = fs::read_to_string(&cargo)?;
    let mut in_package = false;
    for line in txt.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = line.strip_prefix("name") {
                let rest = rest.trim_start().trim_start_matches('=').trim();
                let name = rest
                    .trim_end_matches(|c: char| c == ',' || c.is_whitespace())
                    .trim_matches('"')
                    .trim_matches('\'')
                    .replace('-', "_");
                if !name.is_empty() {
                    return Ok(name);
                }
            }
        }
    }
    Ok("crate".into())
}

/// Map a file path under `<workspace>/src/...` to a module path string.
/// Returns None for files outside src/.
fn derive_base_module(workspace: &Path, crate_name: &str, file: &Path) -> Option<String> {
    let rel = file.strip_prefix(workspace).ok()?;
    let mut comps = rel.components();
    let first = comps.next()?;
    if first.as_os_str() != "src" {
        return None;
    }
    let comps: Vec<_> = comps.collect();
    if comps.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = vec![crate_name.to_string()];
    let last_idx = comps.len() - 1;
    for (i, c) in comps.iter().enumerate() {
        let s = c.as_os_str().to_string_lossy().to_string();
        if i == last_idx {
            let stem = Path::new(&s).file_stem()?.to_string_lossy().to_string();
            if stem == "lib" || stem == "main" || stem == "mod" {
                // module is the directory above
            } else {
                parts.push(stem);
            }
        } else {
            parts.push(s);
        }
    }
    Some(parts.join("::"))
}

fn index_item(ont: &mut Ontology, item: &Item, path: &Path, module: &str) {
    match item {
        Item::Fn(f) => {
            let func = build_function(f, path, module);
            register_in_module(ont, module, path, "fn", &func.name);
            ont.functions
                .insert(format!("{}::{}", module, func.name), func);
        }
        Item::Struct(s) => {
            let ty = build_struct(s, path, module);
            register_in_module(ont, module, path, "ty", &ty.name);
            ont.types.insert(format!("{}::{}", module, ty.name), ty);
        }
        Item::Enum(e) => {
            let ty = build_enum(e, path, module);
            register_in_module(ont, module, path, "ty", &ty.name);
            ont.types.insert(format!("{}::{}", module, ty.name), ty);
        }
        Item::Trait(t) => {
            let tr = build_trait(t, path, module);
            register_in_module(ont, module, path, "tr", &tr.name);
            ont.traits.insert(format!("{}::{}", module, tr.name), tr);
        }
        Item::Impl(i) => {
            let type_name = type_name_of(&i.self_ty);
            for impl_item in &i.items {
                if let syn::ImplItem::Fn(m) = impl_item {
                    let method_module = format!("{}::{}", module, type_name);
                    let func = build_method(m, path, &method_module);
                    ont.functions
                        .insert(format!("{}::{}", method_module, func.name), func);
                }
            }
        }
        Item::Mod(m) => {
            let sub = format!("{}::{}", module, m.ident);
            register_in_module(ont, module, path, "mod", &m.ident.to_string());
            if let Some((_, items)) = &m.content {
                for inner in items {
                    index_item(ont, inner, path, &sub);
                }
            }
        }
        _ => {}
    }
}

fn register_in_module(ont: &mut Ontology, module: &str, path: &Path, kind: &str, name: &str) {
    let entry = ont
        .modules
        .entry(module.to_string())
        .or_insert_with(|| Module {
            path: module.to_string(),
            file: path.to_path_buf(),
            functions: vec![],
            types: vec![],
            traits: vec![],
            submodules: vec![],
        });
    let bucket = match kind {
        "fn" => &mut entry.functions,
        "ty" => &mut entry.types,
        "tr" => &mut entry.traits,
        "mod" => &mut entry.submodules,
        _ => return,
    };
    if !bucket.contains(&name.to_string()) {
        bucket.push(name.to_string());
    }
}

fn build_function(f: &ItemFn, path: &Path, module: &str) -> Function {
    let name = f.sig.ident.to_string();
    let signature = signature_string(&f.sig, &f.vis);
    let doc = doc_from_attrs(&f.attrs);
    let attributes = non_doc_attrs(&f.attrs);
    let is_async = f.sig.asyncness.is_some();
    let is_unsafe = f.sig.unsafety.is_some();
    let is_test = attributes.iter().any(|a| a.contains("test"));
    let visibility = vis_string(&f.vis);
    let body = block_string(&f.block);
    let callees = collect_callees(&f.block);
    let line_start = fn_start_line(&f.attrs, &f.vis, &f.sig);
    let line_end = f.block.span().end().line;
    Function {
        name,
        module_path: module.to_string(),
        file: path.to_path_buf(),
        line_start,
        line_end,
        signature,
        doc,
        attributes,
        is_async,
        is_unsafe,
        is_test,
        visibility,
        body,
        callees,
    }
}

fn build_method(m: &syn::ImplItemFn, path: &Path, module: &str) -> Function {
    let name = m.sig.ident.to_string();
    let signature = signature_string(&m.sig, &m.vis);
    let doc = doc_from_attrs(&m.attrs);
    let attributes = non_doc_attrs(&m.attrs);
    let is_async = m.sig.asyncness.is_some();
    let is_unsafe = m.sig.unsafety.is_some();
    let is_test = attributes.iter().any(|a| a.contains("test"));
    let visibility = vis_string(&m.vis);
    let body = block_string(&m.block);
    let callees = collect_callees(&m.block);
    let line_start = fn_start_line(&m.attrs, &m.vis, &m.sig);
    let line_end = m.block.span().end().line;
    Function {
        name,
        module_path: module.to_string(),
        file: path.to_path_buf(),
        line_start,
        line_end,
        signature,
        doc,
        attributes,
        is_async,
        is_unsafe,
        is_test,
        visibility,
        body,
        callees,
    }
}

/// First source line owned by a structural item, *including* leading
/// attributes and doc comments. The full presentation - decorators, docs,
/// signature, body - all belong to the structural unit. Edits to any of
/// them go through update_ontology (replace_body for body-only,
/// replace_item for whole rewrites). edit_file cannot touch them, which
/// is what enforces graph-first navigation: to see or change a function's
/// docs/attrs/body, the agent must query Function and use a structural op.
fn fn_start_line(attrs: &[syn::Attribute], vis: &syn::Visibility, sig: &syn::Signature) -> usize {
    item_start_line(attrs, fallback_fn_line(vis, sig))
}

fn fallback_fn_line(vis: &syn::Visibility, sig: &syn::Signature) -> usize {
    match vis {
        syn::Visibility::Inherited => sig.fn_token.span.start().line,
        _ => vis.span().start().line,
    }
}

fn struct_start_line(s: &ItemStruct) -> usize {
    let fallback = match &s.vis {
        syn::Visibility::Inherited => s.struct_token.span.start().line,
        _ => s.vis.span().start().line,
    };
    item_start_line(&s.attrs, fallback)
}

fn enum_start_line(e: &ItemEnum) -> usize {
    let fallback = match &e.vis {
        syn::Visibility::Inherited => e.enum_token.span.start().line,
        _ => e.vis.span().start().line,
    };
    item_start_line(&e.attrs, fallback)
}

fn trait_start_line(t: &ItemTrait) -> usize {
    let fallback = match &t.vis {
        syn::Visibility::Inherited => t.trait_token.span.start().line,
        _ => t.vis.span().start().line,
    };
    item_start_line(&t.attrs, fallback)
}

fn item_start_line(attrs: &[syn::Attribute], fallback: usize) -> usize {
    attrs
        .iter()
        .map(|a| a.span().start().line)
        .min()
        .unwrap_or(fallback)
}

fn build_struct(s: &ItemStruct, path: &Path, module: &str) -> Type {
    let name = s.ident.to_string();
    let doc = doc_from_attrs(&s.attrs);
    let derives = derives_from_attrs(&s.attrs);
    let visibility = vis_string(&s.vis);
    let fields: Vec<Field> = match &s.fields {
        syn::Fields::Named(fs) => fs
            .named
            .iter()
            .map(|f| Field {
                name: f
                    .ident
                    .as_ref()
                    .map(|i| i.to_string())
                    .unwrap_or_default(),
                ty: tokens_to_string(&f.ty),
                visibility: vis_string(&f.vis),
                doc: doc_from_attrs(&f.attrs),
            })
            .collect(),
        syn::Fields::Unnamed(fs) => fs
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| Field {
                name: i.to_string(),
                ty: tokens_to_string(&f.ty),
                visibility: vis_string(&f.vis),
                doc: doc_from_attrs(&f.attrs),
            })
            .collect(),
        syn::Fields::Unit => vec![],
    };
    let source = pretty_item(s);
    let line_start = struct_start_line(s);
    let line_end = s.span().end().line;
    Type {
        name,
        module_path: module.to_string(),
        file: path.to_path_buf(),
        line_start,
        line_end,
        kind: "struct".into(),
        doc,
        visibility,
        fields,
        derives,
        source,
    }
}

fn build_enum(e: &ItemEnum, path: &Path, module: &str) -> Type {
    let name = e.ident.to_string();
    let doc = doc_from_attrs(&e.attrs);
    let derives = derives_from_attrs(&e.attrs);
    let visibility = vis_string(&e.vis);
    let fields: Vec<Field> = e
        .variants
        .iter()
        .map(|v| Field {
            name: v.ident.to_string(),
            ty: variant_shape(&v.fields),
            visibility: "pub".into(),
            doc: doc_from_attrs(&v.attrs),
        })
        .collect();
    let source = pretty_item(e);
    let line_start = enum_start_line(e);
    let line_end = e.span().end().line;
    Type {
        name,
        module_path: module.to_string(),
        file: path.to_path_buf(),
        line_start,
        line_end,
        kind: "enum".into(),
        doc,
        visibility,
        fields,
        derives,
        source,
    }
}

fn build_trait(t: &ItemTrait, path: &Path, module: &str) -> Trait {
    let name = t.ident.to_string();
    let doc = doc_from_attrs(&t.attrs);
    let visibility = vis_string(&t.vis);
    let methods: Vec<String> = t
        .items
        .iter()
        .filter_map(|it| match it {
            syn::TraitItem::Fn(f) => Some(f.sig.ident.to_string()),
            _ => None,
        })
        .collect();
    let source = pretty_item(t);
    let line_start = trait_start_line(t);
    let line_end = t.span().end().line;
    Trait {
        name,
        module_path: module.to_string(),
        file: path.to_path_buf(),
        line_start,
        line_end,
        doc,
        visibility,
        methods,
        source,
    }
}

fn signature_string(sig: &syn::Signature, vis: &syn::Visibility) -> String {
    let v = vis_string(vis);
    let prefix = if v.is_empty() {
        String::new()
    } else {
        format!("{v} ")
    };
    let toks = sig.to_token_stream().to_string();
    format!("{prefix}{toks}")
}

fn vis_string(v: &syn::Visibility) -> String {
    match v {
        syn::Visibility::Public(_) => "pub".into(),
        syn::Visibility::Restricted(r) => r.to_token_stream().to_string(),
        syn::Visibility::Inherited => String::new(),
    }
}

fn doc_from_attrs(attrs: &[syn::Attribute]) -> String {
    let mut docs = vec![];
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    docs.push(s.value().trim().to_string());
                }
            }
        }
    }
    docs.join("\n")
}

fn non_doc_attrs(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| !a.path().is_ident("doc"))
        .map(|a| a.to_token_stream().to_string())
        .collect()
}

fn derives_from_attrs(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut out = vec![];
    for attr in attrs {
        if attr.path().is_ident("derive") {
            let _ = attr.parse_nested_meta(|meta| {
                if let Some(ident) = meta.path.get_ident() {
                    out.push(ident.to_string());
                }
                Ok(())
            });
        }
    }
    out
}

/// Serialize a function body for the agent. Wraps the block in a synthetic
/// fn so prettyplease can format it, then returns just the inner content
/// (without the surrounding braces). Callers see idiomatic Rust, not
/// space-separated tokens.
fn block_string(b: &syn::Block) -> String {
    let stream = quote::quote! { fn __planecode_dummy() #b };
    if let Ok(file) = syn::parse2::<syn::File>(stream) {
        let pretty = prettyplease::unparse(&file);
        if let Some(open) = pretty.find('{') {
            if let Some(close) = pretty.rfind('}') {
                if open < close {
                    return pretty[open + 1..close].trim_matches('\n').to_string();
                }
            }
        }
    }
    let s = b.to_token_stream().to_string();
    let s = s.trim();
    s.strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| s.to_string())
}

fn tokens_to_string<T: ToTokens>(t: &T) -> String {
    t.to_token_stream().to_string()
}

fn pretty_item<T: ToTokens>(t: &T) -> String {
    let stream: TokenStream = t.to_token_stream();
    if let Ok(file) = syn::parse2::<syn::File>(quote::quote! { #stream }) {
        return prettyplease::unparse(&file);
    }
    stream.to_string()
}

fn variant_shape(f: &syn::Fields) -> String {
    match f {
        syn::Fields::Named(_) => "struct-like".into(),
        syn::Fields::Unnamed(u) => {
            let inner: Vec<String> = u.unnamed.iter().map(|f| tokens_to_string(&f.ty)).collect();
            format!("({})", inner.join(", "))
        }
        syn::Fields::Unit => "unit".into(),
    }
}

fn type_name_of(ty: &syn::Type) -> String {
    if let syn::Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            return seg.ident.to_string();
        }
    }
    tokens_to_string(ty)
}

fn collect_callees(b: &syn::Block) -> Vec<String> {
    struct V {
        calls: HashSet<String>,
    }
    impl<'a> Visit<'a> for V {
        fn visit_expr_call(&mut self, c: &'a syn::ExprCall) {
            if let syn::Expr::Path(p) = &*c.func {
                if let Some(seg) = p.path.segments.last() {
                    self.calls.insert(seg.ident.to_string());
                }
            }
            syn::visit::visit_expr_call(self, c);
        }
        fn visit_expr_method_call(&mut self, c: &'a syn::ExprMethodCall) {
            self.calls.insert(c.method.to_string());
            syn::visit::visit_expr_method_call(self, c);
        }
    }
    let mut v = V {
        calls: HashSet::new(),
    };
    v.visit_block(b);
    let mut calls: Vec<String> = v.calls.into_iter().collect();
    calls.sort();
    calls
}

