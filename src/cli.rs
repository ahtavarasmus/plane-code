//! Interactive REPL. Wraps `Agent` with a rustyline-backed input loop and
//! a small set of slash commands. Designed for debug-mode use: every tool
//! call and result is printed in full via `Display`.
//!
//! Commands:
//!   /help         show command list
//!   /quit, /exit  leave the REPL
//!   /context      dump the model's current message history
//!   /export [F]   write the full next-turn payload (messages + tools) to JSON
//!   /clear        reset conversation (keeps system prompt + ontology)
//!   /reindex      re-walk the workspace and rebuild the ontology graph
//!   /turns N      change the per-message max_turns budget

use crate::agent::Agent;
use crate::config;
use crate::llm::Provider;
use crate::sessions;
use crate::tools;
use anyhow::{Context, Result};
use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Select};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;

/// Curated picker entries for `/model`. Order matches what shows up in
/// the picker. Add a row here to expose a new model in the REPL; the
/// rest of the system has no opinions about which models exist. The
/// (provider, model_id) pair is exactly what `BackendConfig::build`
/// needs.
const MODEL_OPTIONS: &[(Provider, &str)] = &[
    (Provider::Groq, "qwen/qwen3-32b"),
    (Provider::Ollama, "qwen3:32b"),
    (Provider::Ollama, "qwen3:8b"),
];

pub async fn run_repl(agent: &mut Agent, warm: bool) -> Result<()> {
    let mut rl = DefaultEditor::new()?;

    // Pre-load the model so the operator's first prompt doesn't pay the
    // cold-start cost. Skipped under --no-warm.
    if warm {
        print!("{}", "warming model... ".bright_black().italic());
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let started = std::time::Instant::now();
        match agent.llm.warm().await {
            Ok(_) => {
                let s = format!("loaded in {:.1}s", started.elapsed().as_secs_f32());
                println!("{}", s.bright_black().italic());
            }
            Err(e) => {
                println!();
                agent
                    .display
                    .show_error(&format!("warm failed: {e} (first prompt will pay the load cost)"));
            }
        }
    }

    // Intro banner reflects the current ontology snapshot.
    let crate_name = agent.ontology.crate_name.clone();
    agent.display.banner(
        &crate_name,
        (
            agent.ontology.functions.len(),
            agent.ontology.types.len(),
            agent.ontology.traits.len(),
            agent.ontology.modules.len(),
            agent.ontology.files.len(),
        ),
    );

    loop {
        let prompt = format!("{} ", "›".bright_cyan().bold());
        let line = match rl.readline(&prompt) {
            Ok(s) => s,
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C: cancel current input, continue.
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D: exit cleanly.
                println!("{}", "(exit)".bright_black());
                return Ok(());
            }
            Err(ReadlineError::WindowResized) => {
                // Terminal resize: rustyline surfaces this as an Err so
                // the host can react. Just re-prompt; the line buffer
                // was empty (we're between turns), so nothing is lost.
                continue;
            }
            Err(e) => {
                agent.display.show_error(&format!("readline: {e}"));
                return Err(e.into());
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);

        if let Some(cmd) = trimmed.strip_prefix('/') {
            match handle_slash(agent, cmd).await {
                SlashOutcome::Continue => continue,
                SlashOutcome::Quit => return Ok(()),
            }
        }

        if let Err(e) = agent.run_turn(trimmed).await {
            agent.display.show_error(&e.to_string());
        }
    }
}

/// TUI-mode entry point for slash commands. Same logic as the
/// rustyline path; just exposed so `tui.rs` can call into it without
/// needing rustyline's run_repl stack.
pub async fn handle_slash_for_tui(agent: &mut Agent, cmd: &str) -> SlashOutcome {
    handle_slash(agent, cmd).await
}

pub enum SlashOutcome {
    Continue,
    Quit,
}

async fn handle_slash(agent: &mut Agent, cmd: &str) -> SlashOutcome {
    let mut parts = cmd.split_whitespace();
    let head = parts.next().unwrap_or("");
    match head {
        "quit" | "exit" | "q" => {
            println!("{}", "bye".bright_black());
            SlashOutcome::Quit
        }
        "help" | "h" | "?" => {
            print_help();
            SlashOutcome::Continue
        }
        "context" | "ctx" => {
            agent.display.dump_context(&agent.messages);
            SlashOutcome::Continue
        }
        "export" => {
            let path_arg: Option<String> = parts.next().map(str::to_string);
            match export_session(agent, path_arg.as_deref()) {
                Ok(path) => println!(
                    "{} {}",
                    "exported to".bright_black(),
                    path.display().to_string().bright_white()
                ),
                Err(e) => agent.display.show_error(&format!("export: {e}")),
            }
            SlashOutcome::Continue
        }
        "clear" => {
            agent.clear_history();
            println!(
                "{}",
                "history cleared (system prompt and ontology preserved)".bright_black()
            );
            SlashOutcome::Continue
        }
        "resume" => {
            handle_resume(agent);
            SlashOutcome::Continue
        }
        "model" => {
            let arg = parts.next().unwrap_or("").to_string();
            handle_model_switch(agent, &arg);
            SlashOutcome::Continue
        }
        "warm" => {
            print!("{}", "warming model... ".bright_black().italic());
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let started = std::time::Instant::now();
            match agent.llm.warm().await {
                Ok(_) => {
                    let s = format!("loaded in {:.1}s", started.elapsed().as_secs_f32());
                    println!("{}", s.bright_black().italic());
                }
                Err(e) => {
                    println!();
                    agent.display.show_error(&format!("warm failed: {e}"));
                }
            }
            SlashOutcome::Continue
        }
        "think" => {
            let arg = parts.next().unwrap_or("");
            let new_value = match arg {
                "on" | "true" | "1" => Some(true),
                "off" | "false" | "0" => Some(false),
                "" => None,
                _ => {
                    agent.display.show_error("usage: /think on | off");
                    return SlashOutcome::Continue;
                }
            };
            match new_value {
                Some(v) => {
                    agent.llm.set_think(v);
                    println!("{} {}", "think =".bright_black(), v.to_string().bright_white());
                    if let Err(e) = config::update(|c| c.think = Some(v)) {
                        agent.display.show_error(&format!("config save: {e}"));
                    }
                }
                None => {
                    println!(
                        "{} {}",
                        "think =".bright_black(),
                        agent.llm.think().to_string().bright_white()
                    );
                }
            }
            SlashOutcome::Continue
        }
        "trace" => {
            let arg = parts.next().unwrap_or("");
            let new_value = match arg {
                "on" | "true" | "1" => Some(true),
                "off" | "false" | "0" => Some(false),
                "" => None,
                _ => {
                    agent.display.show_error("usage: /trace on | off");
                    return SlashOutcome::Continue;
                }
            };
            match new_value {
                Some(v) => {
                    agent.llm.set_trace(v);
                    let suffix = if v { " (raw stream chunks will print to stderr)" } else { "" };
                    println!(
                        "{} {}{}",
                        "trace =".bright_black(),
                        v.to_string().bright_white(),
                        suffix.bright_black().to_string()
                    );
                    if let Err(e) = config::update(|c| c.trace = Some(v)) {
                        agent.display.show_error(&format!("config save: {e}"));
                    }
                }
                None => {
                    println!(
                        "{} {}",
                        "trace =".bright_black(),
                        agent.llm.trace().to_string().bright_white()
                    );
                }
            }
            SlashOutcome::Continue
        }
        "flow" => {
            let target = parts.next().unwrap_or("");
            if target.is_empty() {
                agent.display.show_error("usage: /flow <function-name>");
                return SlashOutcome::Continue;
            }
            match crate::flow_cli::open_flow(&agent.ontology, target) {
                Ok(path) => println!(
                    "{} {}",
                    "opened".bright_black(),
                    path.display().to_string().bright_white()
                ),
                Err(e) => agent.display.show_error(&e.to_string()),
            }
            SlashOutcome::Continue
        }
        "skyline" => {
            match crate::flow_cli::open_skyline(&agent.ontology) {
                Ok(path) => println!(
                    "{} {}",
                    "opened".bright_black(),
                    path.display().to_string().bright_white()
                ),
                Err(e) => agent.display.show_error(&e.to_string()),
            }
            SlashOutcome::Continue
        }
        "reindex" => match agent.reindex() {
            Ok((fns, types, traits, modules, files)) => {
                println!(
                    "{} {} fn  {} ty  {} tr  {} mod  {} files",
                    "reindexed:".bright_black(),
                    fns,
                    types,
                    traits,
                    modules,
                    files
                );
                SlashOutcome::Continue
            }
            Err(e) => {
                agent.display.show_error(&format!("reindex: {e}"));
                SlashOutcome::Continue
            }
        },
        "turns" => {
            let n = parts.next().and_then(|s| s.parse::<usize>().ok());
            match n {
                Some(n) if n > 0 => {
                    agent.max_turns = n;
                    println!("{} {n}", "max_turns =".bright_black());
                }
                _ => {
                    agent
                        .display
                        .show_error("usage: /turns <positive integer>");
                }
            }
            SlashOutcome::Continue
        }
        other => {
            agent
                .display
                .show_error(&format!("unknown command: /{other}  (try /help)"));
            SlashOutcome::Continue
        }
    }
}

fn print_help() {
    let lines: &[(&str, &str)] = &[
        ("/help", "show this help"),
        ("/quit, /exit, /q", "exit the REPL"),
        ("/context, /ctx", "dump the full model context (every message)"),
        ("/export [path]", "write the next-turn payload (messages + tools) as JSON"),
        ("/clear", "reset conversation (keep system prompt and ontology)"),
        ("/resume", "pick a previous session for this workspace and reload it"),
        ("/reindex", "rebuild the ontology graph from disk"),
        ("/model [name]", "switch backend+model (interactive picker if no arg)"),
        ("/flow <fn>", "open the control-flow diagram for a function"),
        ("/skyline", "open the workspace call-graph skyline"),
        ("/warm", "force-reload the model (after Ollama evicts it)"),
        ("/think on|off", "toggle Ollama's think:true (off helps tool calls)"),
        ("/trace on|off", "dump every raw Ollama stream chunk to stderr"),
        ("/turns N", "set max tool-call rounds per user message"),
    ];
    println!();
    for (cmd, desc) in lines {
        println!("  {:<20} {}", cmd.bright_yellow(), desc);
    }
    println!();
}

/// Handle `/model` with an optional model-name argument. Empty arg
/// opens the interactive picker; non-empty tries to match an entry in
/// MODEL_OPTIONS by exact model id and switches without prompting.
///
/// Conversation history is preserved across the swap. think/trace
/// settings are read off the current backend and copied onto the new
/// one so toggling those mid-session doesn't reset.
fn handle_model_switch(agent: &mut Agent, arg: &str) {
    let current_provider = agent.llm.provider();
    let current_model = agent.llm.model().to_string();
    let current_idx = MODEL_OPTIONS.iter().position(|(p, m)| {
        p.as_str() == current_provider && *m == current_model.as_str()
    });

    let chosen_idx = if arg.is_empty() {
        // Interactive picker. dialoguer handles arrow-key navigation,
        // Enter to confirm, Esc/Ctrl-C to cancel (returns Ok(None)).
        let items: Vec<String> = MODEL_OPTIONS
            .iter()
            .map(|(p, m)| format!("({:<6}) {}", p.as_str(), m))
            .collect();
        let theme = ColorfulTheme::default();
        let select = Select::with_theme(&theme)
            .with_prompt("model")
            .items(&items)
            .default(current_idx.unwrap_or(0));
        match select.interact_opt() {
            Ok(Some(i)) => i,
            Ok(None) => {
                println!("{}", "(model unchanged)".bright_black());
                return;
            }
            Err(e) => {
                agent.display.show_error(&format!("picker: {e}"));
                return;
            }
        }
    } else {
        // Exact-name match against the curated list. Bare model names
        // are unambiguous in our list (Ollama uses `name:tag`, Groq
        // uses `org/name`); first match wins.
        match MODEL_OPTIONS.iter().position(|(_, m)| *m == arg) {
            Some(i) => i,
            None => {
                let known: Vec<String> = MODEL_OPTIONS
                    .iter()
                    .map(|(p, m)| format!("{}:{}", p.as_str(), m))
                    .collect();
                agent.display.show_error(&format!(
                    "unknown model: {arg}\n  known: {}",
                    known.join(", ")
                ));
                return;
            }
        }
    };

    let (provider, model) = MODEL_OPTIONS[chosen_idx];
    if Some(chosen_idx) == current_idx {
        println!(
            "{} {} {}",
            "already on".bright_black(),
            provider.as_str().bright_white(),
            model.bright_white(),
        );
        return;
    }

    let think = agent.llm.think();
    let trace = agent.llm.trace();
    match agent
        .backend_config
        .build(provider, model.to_string(), think, trace)
    {
        Ok(new_backend) => {
            agent.llm = new_backend;
            println!(
                "{} {} {}",
                "switched to".bright_black(),
                provider.as_str().bright_white(),
                model.bright_white(),
            );
            // Persist so the next launch starts with this provider/model
            // pair instead of falling back to the built-in default.
            if let Err(e) = config::update(|c| {
                c.provider = Some(provider);
                c.model = Some(model.to_string());
            }) {
                agent
                    .display
                    .show_error(&format!("config save: {e}"));
            }
            if matches!(provider, Provider::Ollama) {
                println!(
                    "{}",
                    "  (run /warm to preload weights before the next prompt)".bright_black().italic()
                );
            }
        }
        Err(e) => {
            agent.display.show_error(&e.to_string());
        }
    }
}

/// `/resume`: list saved sessions for this workspace, let the operator
/// pick one with arrow keys, then load its messages into the live
/// agent. Auto-saves keep these files current; pruning trims to 30 on
/// new-session start. The model + provider that the loaded session
/// was using is NOT auto-restored - whatever model is currently
/// selected continues - because the operator may want to re-run the
/// same conversation under a different backend.
fn handle_resume(agent: &mut Agent) {
    let listings = match sessions::list(&agent.workspace) {
        Ok(v) => v,
        Err(e) => {
            agent.display.show_error(&format!("list sessions: {e}"));
            return;
        }
    };
    if listings.is_empty() {
        println!("{}", "(no saved sessions for this workspace)".bright_black());
        return;
    }

    let items: Vec<String> = listings
        .iter()
        .map(|l| {
            format!(
                "{:<10} {:>3} msg  {:<20}  {}",
                sessions::relative_time(l.updated_at_unix),
                l.message_count,
                format!("{}/{}", l.provider, l.model),
                l.title
            )
        })
        .collect();

    let theme = ColorfulTheme::default();
    let select = Select::with_theme(&theme)
        .with_prompt("resume session")
        .items(&items)
        .default(0);

    let chosen_idx = match select.interact_opt() {
        Ok(Some(i)) => i,
        Ok(None) => {
            println!("{}", "(no change)".bright_black());
            return;
        }
        Err(e) => {
            agent.display.show_error(&format!("picker: {e}"));
            return;
        }
    };

    let chosen = &listings[chosen_idx];
    let session = match sessions::load(&chosen.path) {
        Ok(s) => s,
        Err(e) => {
            agent.display.show_error(&format!("load: {e}"));
            return;
        }
    };
    let msg_count = session.messages.len();
    let title = chosen.title.clone();
    agent.resume_session(session.id, session.messages);

    // Replay the conversation onto the screen so the operator lands in
    // a terminal that looks like the chat just happened. Without this
    // the prompt drops back to `›` with no visible context, even
    // though the model has all of it in memory.
    println!();
    println!(
        "{}",
        "─────────── replaying session ───────────".bright_black()
    );
    println!();
    agent.display.render_transcript(&agent.messages);
    println!(
        "{}",
        "──────────── end of replay ─────────────".bright_black()
    );
    println!(
        "{} \"{}\" {}",
        "resumed".bright_black(),
        title.bright_white(),
        format!("({msg_count} messages)").bright_black().italic()
    );
}

/// Snapshot of the next-turn payload: exactly what `chat_stream` would
/// send to the LLM if the operator typed something now. Tool definitions
/// are rebuilt against the live ontology since dynamic enums (file
/// paths, module paths, traits) refresh every turn. Written as
/// pretty-printed JSON.
fn export_session(agent: &Agent, dest: Option<&str>) -> Result<PathBuf> {
    let path: PathBuf = match dest {
        Some(s) if !s.is_empty() => PathBuf::from(s),
        _ => {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            PathBuf::from(format!("planecode-session-{stamp}.json"))
        }
    };
    let payload = serde_json::json!({
        "exported_at_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "provider": agent.llm.provider(),
        "model": agent.llm.model(),
        "workspace": agent.workspace.display().to_string(),
        "crate": agent.ontology.crate_name,
        "tools": tools::tool_definitions(&agent.ontology),
        "messages": agent.messages,
    });
    let json = serde_json::to_string_pretty(&payload)
        .context("serialize session")?;
    std::fs::write(&path, json)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}
