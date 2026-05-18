use anyhow::{Context, Result};
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches, Parser};
use llm::Provider;
use std::path::PathBuf;

mod agent;
mod cargo_ops;
mod cli;
mod config;
mod display;
mod flow;
mod flow_cli;
mod groq;
mod keys;
mod llm;
mod ollama;
mod ontology;
mod read_set;
mod sessions;
mod tools;
mod tui;

const DEFAULT_NUM_CTX: u32 = 32768;
const DEFAULT_OLLAMA_MODEL: &str = "qwen3:8b";
const DEFAULT_GROQ_MODEL: &str = "llama-3.3-70b-versatile";
const DEFAULT_GROQ_HOST: &str = "https://api.groq.com";

#[derive(Parser, Debug)]
#[command(name = "plane-code", version, about = "Ontology-based Rust coding agent")]
struct Cli {
    /// Path to the Rust workspace to operate on
    #[arg(short, long, default_value = ".")]
    workspace: PathBuf,

    /// LLM backend. `ollama` runs against a local daemon; `groq` calls
    /// the hosted Groq REST API (OpenAI-compatible) and translates
    /// responses into the same shape the agent loop already speaks.
    #[arg(long, value_enum, default_value_t = Provider::Ollama)]
    provider: Provider,

    /// Model name. Default depends on provider: `qwen3:8b` for ollama,
    /// `llama-3.3-70b-versatile` for groq. Override to pick a different
    /// model on either backend.
    #[arg(short, long)]
    model: Option<String>,

    /// Ollama API base URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_host: String,

    /// Groq API base URL. Override only for staging / proxies.
    #[arg(long, default_value = DEFAULT_GROQ_HOST)]
    groq_host: String,

    /// Groq API key. Falls back to the `GROQ_API_KEY` environment
    /// variable when omitted. Required when --provider=groq.
    #[arg(long)]
    api_key: Option<String>,

    /// Maximum agent turns (tool-call rounds) before bailing
    #[arg(long, default_value_t = 30)]
    max_turns: usize,

    /// Ollama context window (num_ctx). Larger values let longer
    /// conversations fit but use more memory. Default 32768.
    #[arg(long)]
    num_ctx: Option<u32>,

    /// Disable Ollama's `think: true`. Some thinking-capable models
    /// (qwen3:8b especially) emit chain-of-thought, then fail to emit
    /// the structured tool_call JSON. Turning thinking off often makes
    /// tool-calling much more reliable on small models, at the cost of
    /// not seeing the model's reasoning.
    #[arg(long, default_value_t = false)]
    no_think: bool,

    /// Skip the model warm-up ping at REPL startup. Without warm-up,
    /// the first prompt of a session pays a ~10-15s weight-load cost.
    #[arg(long, default_value_t = false)]
    no_warm: bool,

    /// Opt in to the experimental full-screen TUI (pinned input box,
    /// status line, scrollback). Off by default - the rustyline-based
    /// REPL is currently more reliable. Re-enable once the TUI's
    /// scroll math + render perf are settled.
    #[arg(long, default_value_t = false)]
    tui: bool,

    /// Print every raw Ollama stream chunk to stderr. Use when the model
    /// generates eval tokens but you don't see content / thinking /
    /// tool_calls in the response, so you can tell where the tokens
    /// actually went.
    #[arg(long, default_value_t = false)]
    trace: bool,

    /// Suppress the verbose tool-call / tool-result rendering. By default
    /// the agent prints every step in full so the operator can see what
    /// the model is doing.
    #[arg(long, default_value_t = false)]
    quiet: bool,

    /// Skip the LLM and instead run a few sample ontology calls. Used for
    /// smoke-testing the query/update plumbing without depending on Ollama.
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Emit a control-flow diagram for the named function as Mermaid text
    /// on stdout. The name is matched against indexed Function entities;
    /// pass either a short name (`verify_token`) or a fully-qualified
    /// path (`sandbox::auth::verify_token`). Pipe to `pbcopy` and paste
    /// into mermaid.live to view, or save to a .mmd file.
    #[arg(long, value_name = "FUNCTION")]
    flow: Option<String>,

    /// Open the workspace skyline: a hierarchical map of modules and
    /// inter-module edges; click a module to expand its functions
    /// inline, click a function to expand its CFG. Pass an optional
    /// FOCUS (function name or module path) to start zoomed in on
    /// that node instead of the top-level workspace map.
    ///
    /// Examples:
    ///   --skyline                     (top-level workspace map)
    ///   --skyline=backend::services   (start expanded on a module)
    ///   --skyline=verify_token        (start expanded on a function)
    #[arg(long, value_name = "FOCUS", num_args = 0..=1, default_missing_value = "")]
    skyline: Option<String>,

    /// User prompt. If omitted, planecode launches an interactive REPL.
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "planecode=info".into()),
        )
        .with_target(false)
        .init();

    // Parse via ArgMatches so we can ask clap whether each flag came
    // from the command line vs. the built-in default. That distinction
    // matters because saved user config (~/.plane-code/config.json)
    // should override the *default* but lose to an explicit CLI flag.
    let matches = Cli::command().get_matches();
    let provider_explicit =
        matches.value_source("provider") == Some(ValueSource::CommandLine);
    let no_think_explicit =
        matches.value_source("no_think") == Some(ValueSource::CommandLine);
    let trace_explicit =
        matches.value_source("trace") == Some(ValueSource::CommandLine);
    let cli = Cli::from_arg_matches(&matches).context("parse cli")?;
    let workspace = cli
        .workspace
        .canonicalize()
        .with_context(|| format!("workspace path {:?} not found", cli.workspace))?;
    let prompt = cli.prompt.join(" ");

    eprintln!("indexing {} ...", workspace.display());
    let mut ontology = ontology::Ontology::index(&workspace)?;
    eprintln!(
        "indexed: {} functions, {} types, {} traits, {} modules, {} files",
        ontology.functions.len(),
        ontology.types.len(),
        ontology.traits.len(),
        ontology.modules.len(),
        ontology.files.len(),
    );

    if cli.debug {
        return run_debug(&mut ontology, &workspace);
    }

    if let Some(target) = cli.flow.as_deref() {
        return run_flow(&ontology, target);
    }

    if let Some(focus) = cli.skyline.as_deref() {
        let path = if focus.is_empty() {
            flow_cli::open_skyline(&ontology)?
        } else {
            flow_cli::open_skyline_at(&ontology, focus)?
        };
        eprintln!("opened {}", path.display());
        return Ok(());
    }

    {
        eprintln!("baselining workspace compile state ...");
        use std::collections::HashSet;
        let (_, errors) =
            ontology::update::cargo_check_classified(&workspace, &HashSet::new());
        let n = errors.len();
        ontology.prev_errors = errors;
        if n == 0 {
            eprintln!("baseline: clean");
        } else {
            eprintln!("baseline: {n} pre-existing compile error(s)");
        }
    }

    let display = display::Display::new(!cli.quiet);
    let backend_config = llm::BackendConfig {
        ollama_host: cli.ollama_host.clone(),
        num_ctx: cli.num_ctx.unwrap_or(DEFAULT_NUM_CTX),
        groq_host: cli.groq_host.clone(),
        api_key: cli.api_key.clone(),
    };

    // Layered defaults: saved user config overrides built-ins, explicit
    // CLI flags override the saved config.
    let saved = config::load().unwrap_or_default();
    let provider = if provider_explicit {
        cli.provider
    } else {
        saved.provider.unwrap_or(cli.provider)
    };
    let initial_model = if let Some(m) = cli.model.clone() {
        m
    } else if !provider_explicit {
        saved.model.unwrap_or_else(|| match provider {
            Provider::Ollama => DEFAULT_OLLAMA_MODEL.to_string(),
            Provider::Groq => DEFAULT_GROQ_MODEL.to_string(),
        })
    } else {
        match provider {
            Provider::Ollama => DEFAULT_OLLAMA_MODEL.to_string(),
            Provider::Groq => DEFAULT_GROQ_MODEL.to_string(),
        }
    };
    let think = if no_think_explicit {
        !cli.no_think
    } else {
        saved.think.unwrap_or(!cli.no_think)
    };
    let trace = if trace_explicit {
        cli.trace
    } else {
        saved.trace.unwrap_or(cli.trace)
    };
    let llm = backend_config.build(provider, initial_model, think, trace)?;
    eprintln!("backend: {} model={}", llm.provider(), llm.model());

    let mut session = agent::Agent::new(
        workspace,
        ontology,
        llm,
        backend_config,
        cli.max_turns,
        display,
    );

    if prompt.trim().is_empty() {
        // Default to the rustyline-based REPL. TUI is experimental
        // and opt-in via --tui until its scroll/render polish is done.
        let is_tty = unsafe { libc::isatty(libc::STDIN_FILENO) } == 1;
        if cli.tui && is_tty {
            tui::run_tui(&mut session, !cli.no_warm).await
        } else {
            cli::run_repl(&mut session, !cli.no_warm).await
        }
    } else {
        if !cli.no_warm {
            warm_up(&session).await;
        }
        session.run_once(&prompt).await
    }
}

fn run_flow(ontology: &ontology::Ontology, target: &str) -> Result<()> {
    let path = flow_cli::open_flow(ontology, target)?;
    eprintln!("opened {}", path.display());
    Ok(())
}

async fn warm_up(session: &agent::Agent) {
    use colored::Colorize;
    print!("{}", "warming model... ".bright_black().italic());
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let started = std::time::Instant::now();
    match session.llm.warm().await {
        Ok(_) => {
            let s = format!("loaded in {:.1}s", started.elapsed().as_secs_f32());
            println!("{}", s.bright_black().italic());
        }
        Err(e) => {
            println!();
            eprintln!(
                "{}: {} (first prompt will pay the load cost)",
                "warm failed".bright_red(),
                e
            );
        }
    }
}

/// Smoke test: exercises the ontology + tools surface without an LLM.
fn run_debug(ontology: &mut ontology::Ontology, workspace: &std::path::Path) -> Result<()> {
    use ontology::{QueryRequest, UpdateRequest};

    println!("\n--- File listing ---");
    let files: Vec<_> = ontology.files.keys().cloned().collect();
    for f in &files {
        println!("  {}", f);
    }

    if let Some(rs_path) = files.iter().find(|p| p.ends_with("auth.rs")) {
        println!("\n--- query_ontology File path={rs_path} (outline-only response) ---");
        let req = QueryRequest {
            object_type: "File".into(),
            keywords: None,
            filters: Some(serde_json::json!({ "path": rs_path })),
            include_links: None,
            limit: 1,
        };
        let resp = ontology.query(&req)?;
        if let Some(file) = resp.results.first() {
            let has_content = file.get("content").is_some();
            println!("contains top-level `content` field: {has_content}");
            if let Some(outline) = file.get("outline").and_then(|v| v.as_array()) {
                println!("outline entries: {}", outline.len());
                for entry in outline {
                    let kind = entry.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
                    let ls = entry.get("line_start").and_then(|v| v.as_u64()).unwrap_or(0);
                    let le = entry.get("line_end").and_then(|v| v.as_u64()).unwrap_or(0);
                    if kind == "indexed" {
                        let owner = entry.get("owner").and_then(|v| v.as_str()).unwrap_or("");
                        let sig = entry.get("signature").and_then(|v| v.as_str()).unwrap_or("");
                        println!("  [{ls}-{le}] indexed owner={owner} sig={sig:?}");
                    } else {
                        let content = entry.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let preview: String = content.chars().take(60).collect();
                        println!("  [{ls}-{le}] gap content={preview:?}");
                    }
                }
            }
        }

        println!(
            "\n--- update_ontology edit_file (overlap test on the doc-comment line) ---"
        );
        let upd = UpdateRequest {
            operation: "edit_file".into(),
            target: serde_json::json!({ "path": rs_path }),
            payload: serde_json::json!({
                "line_start": 1,
                "line_end": 1,
                "replacement": "/// New doc string\n",
            }),
            dry_run: true,
        };
        let r = ontology.update(&upd)?;
        println!(
            "success={}, rollback={:?}, hint(s)={}",
            r.success,
            r.rollback_reason,
            r.hints.len()
        );
    }

    println!("\n--- query_ontology File query='verify_token' ---");
    let req = QueryRequest {
        object_type: "File".into(),
        keywords: Some("verify_token".into()),
        filters: None,
        include_links: None,
        limit: 3,
    };
    let resp = ontology.query(&req)?;
    println!("total_matches={}, hints={:?}", resp.total_matches, resp.hints);
    for r in &resp.results {
        if let (Some(path), Some(matches)) = (
            r.get("path").and_then(|v| v.as_str()),
            r.get("matches").and_then(|v| v.as_array()),
        ) {
            println!("  {}: {} match(es)", path, matches.len());
            for m in matches.iter().take(3) {
                let scope = m.get("scope").and_then(|v| v.as_str()).unwrap_or("?");
                let line = m.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                let descriptor = match scope {
                    "signature" => m
                        .get("owner")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    _ => m
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                };
                println!("    line {line} [{scope}]: {descriptor}");
            }
        }
    }

    println!("\n--- update_ontology edit_file (dry-run) on Cargo.toml ---");
    let upd = UpdateRequest {
        operation: "edit_file".into(),
        target: serde_json::json!({ "path": "Cargo.toml" }),
        payload: serde_json::json!({
            "find": "version = \"0.1.0\"",
            "replace": "version = \"0.1.1\"",
        }),
        dry_run: true,
    };
    let r = ontology.update(&upd)?;
    println!(
        "success={}, rollback={:?}, files_changed={}",
        r.success,
        r.rollback_reason,
        r.files_changed.len()
    );

    println!("\n--- update_ontology replace_body (dry-run) on sandbox::auth::authenticate ---");
    let upd = UpdateRequest {
        operation: "replace_body".into(),
        target: serde_json::json!({
            "name": "authenticate",
            "module_path": "sandbox::auth",
        }),
        payload: serde_json::json!({
            "new_body": "verify_token(token, now).is_ok()\n",
        }),
        dry_run: true,
    };
    let r = ontology.update(&upd)?;
    println!(
        "success={}, rollback={:?}, files_changed={}",
        r.success,
        r.rollback_reason,
        r.files_changed.len()
    );

    println!("\n--- update_ontology replace_item (dry-run) on sandbox::auth::Token ---");
    let upd = UpdateRequest {
        operation: "replace_item".into(),
        target: serde_json::json!({
            "object_type": "Type",
            "name": "Token",
            "module_path": "sandbox::auth",
        }),
        payload: serde_json::json!({
            "source": "/// A bearer authentication token issued to an authenticated user.\n#[derive(Debug, Clone, PartialEq)]\npub struct Token {\n    pub subject: String,\n    pub issued_at: u64,\n    pub expires_at: u64,\n}\n",
        }),
        dry_run: true,
    };
    let r = ontology.update(&upd)?;
    println!(
        "success={}, rollback={:?}, files_changed={}",
        r.success,
        r.rollback_reason,
        r.files_changed.len()
    );

    println!("\n--- run_cargo check ---");
    let r = cargo_ops::run_cargo(
        workspace,
        &cargo_ops::RunCargoRequest {
            command: "check".into(),
            args: vec![],
            stdin: None,
        },
    )?;
    println!(
        "exit={}, errors={}, hints={:?}",
        r.exit_code,
        r.compile_errors.len(),
        r.hints
    );

    Ok(())
}
