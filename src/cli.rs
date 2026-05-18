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
use crate::models::{self, ModelInfo, ModelState};
use crate::sessions;
use crate::tools;
use anyhow::{Context, Result};
use colored::Colorize;
use dialoguer::{theme::ColorfulTheme, Confirm, FuzzySelect, Select};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::collections::HashSet;
use std::path::PathBuf;

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
            handle_model_switch(agent, &arg).await;
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
/// opens the interactive two-step picker (provider, then model);
/// non-empty switches the model directly using the provided string as
/// the model id. Provider for the direct path is inferred from the id
/// shape: `org/name` -> Groq, `name:tag` -> Ollama, else current.
///
/// Models are not hardcoded - both lists come from the providers'
/// APIs at click time so new releases show up without code changes.
///
/// Conversation history is preserved across the swap. think/trace
/// settings are read off the current backend and copied onto the new
/// one so toggling those mid-session doesn't reset.
async fn handle_model_switch(agent: &mut Agent, arg: &str) {
    let current_provider = match agent.llm.provider() {
        "groq" => Provider::Groq,
        _ => Provider::Ollama,
    };
    let current_model = agent.llm.model().to_string();

    if !arg.is_empty() {
        switch_direct(agent, arg, current_provider, &current_model);
        return;
    }

    // Step 1: provider picker.
    let providers = [Provider::Ollama, Provider::Groq];
    let provider_items = [
        format!("{} {}", "ollama".bright_white(), "(local)".bright_black()),
        format!("{} {}", "groq".bright_white(), "(hosted)".bright_black()),
    ];
    let default_provider_idx = providers
        .iter()
        .position(|p| *p == current_provider)
        .unwrap_or(0);
    let theme = ColorfulTheme::default();
    let provider = match Select::with_theme(&theme)
        .with_prompt("provider")
        .items(&provider_items)
        .default(default_provider_idx)
        .interact_opt()
    {
        Ok(Some(i)) => providers[i],
        Ok(None) => {
            println!("{}", "(model unchanged)".bright_black());
            return;
        }
        Err(e) => {
            agent.display.show_error(&format!("picker: {e}"));
            return;
        }
    };

    // Step 2: fetch models for the chosen provider. Print a status line
    // so the operator sees activity during the round-trip.
    print!(
        "{}",
        format!("fetching {} models... ", provider.as_str())
            .bright_black()
            .italic()
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let listed = match provider {
        Provider::Groq => fetch_groq(agent).await,
        Provider::Ollama => fetch_ollama(agent).await,
    };
    let models = match listed {
        Ok(m) => m,
        Err(e) => {
            println!();
            agent.display.show_error(&e.to_string());
            return;
        }
    };
    println!("{}", "done".bright_black().italic());

    if models.is_empty() {
        agent.display.show_error("no models found for this provider");
        return;
    }

    // Step 3: fuzzy-search picker. FuzzySelect lets the operator type
    // to filter the list - useful when Ollama's library has 200+ entries.
    let items: Vec<String> = models.iter().map(render_picker_row).collect();
    let current_idx = models
        .iter()
        .position(|m| m.id == current_model && m.provider == current_provider);
    let prompt_label = if provider == Provider::Ollama {
        "model  -  type to filter, Enter on [available] for size variants"
    } else {
        "model  -  type to filter"
    };
    let chosen_idx = match FuzzySelect::with_theme(&theme)
        .with_prompt(prompt_label)
        .items(&items)
        .default(current_idx.unwrap_or(0))
        .interact_opt()
    {
        Ok(Some(i)) => i,
        Ok(None) => {
            println!("{}", "(model unchanged)".bright_black());
            return;
        }
        Err(e) => {
            agent.display.show_error(&format!("picker: {e}"));
            return;
        }
    };
    let chosen = models[chosen_idx].clone();

    // Step 4: if it's an AvailableRemote Ollama slug, drill into its
    // tag list so the user can pick a size variant before pulling.
    // The base slug alone (e.g. `qwen3`) doesn't tell the user whether
    // they're about to download 500MB or 140GB - the tag does.
    let final_id = if chosen.state == ModelState::AvailableRemote {
        match pick_ollama_tag(&chosen.id, &theme).await {
            Some(full_id) => {
                if let Err(e) = run_pull(agent, &full_id).await {
                    agent.display.show_error(&format!("pull: {e}"));
                    return;
                }
                full_id
            }
            None => return,
        }
    } else {
        chosen.id.clone()
    };

    apply_switch(agent, chosen.provider, final_id);
}

/// Fetch and display tag variants for an Ollama slug, prompt the user
/// to pick one, and return `slug:tag` ready to pull. Returns `None`
/// when the user cancels, the network fetch fails, or the page lists
/// no usable tags - in each case we print a message and bail back to
/// the prompt without switching models.
async fn pick_ollama_tag(slug: &str, theme: &ColorfulTheme) -> Option<String> {
    print!(
        "{}",
        format!("fetching {slug} tags... ").bright_black().italic()
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let tags = match models::list_ollama_tags(slug).await {
        Ok(t) => t,
        Err(e) => {
            println!();
            eprintln!(
                "  {} {}",
                "warn:".bright_yellow(),
                format!("tag list: {e}").bright_black()
            );
            // Fall back to a plain confirm on the bare slug - the daemon
            // will pull the default tag (usually `latest`).
            let confirm = Confirm::with_theme(theme)
                .with_prompt(format!(
                    "couldn't fetch tag sizes; pull {slug} (default tag)?"
                ))
                .default(true)
                .interact_opt();
            return match confirm {
                Ok(Some(true)) => Some(slug.to_string()),
                _ => {
                    println!("{}", "(model unchanged)".bright_black());
                    None
                }
            };
        }
    };
    println!("{}", "done".bright_black().italic());

    if tags.is_empty() {
        println!(
            "{}",
            format!("no tags found for {slug}").bright_yellow()
        );
        return None;
    }

    let items: Vec<String> = tags
        .iter()
        .map(|t| {
            let size = t
                .size_bytes
                .map(|n| format!("  ({})", models::human_size(n)))
                .unwrap_or_else(|| "  (size unknown)".into());
            format!("{}:{}{}", slug, t.tag, size)
        })
        .collect();
    let chosen_tag_idx = match FuzzySelect::with_theme(theme)
        .with_prompt(format!("{slug} tag (type to filter)"))
        .items(&items)
        .default(0)
        .interact_opt()
    {
        Ok(Some(i)) => i,
        Ok(None) => {
            println!("{}", "(model unchanged)".bright_black());
            return None;
        }
        Err(e) => {
            eprintln!("picker: {e}");
            return None;
        }
    };
    let tag = &tags[chosen_tag_idx];
    let full_id = format!("{}:{}", slug, tag.tag);
    let size_hint = tag
        .size_bytes
        .map(|n| format!(" (~{})", models::human_size(n)))
        .unwrap_or_default();
    match Confirm::with_theme(theme)
        .with_prompt(format!("pull {full_id}{size_hint}?"))
        .default(true)
        .interact_opt()
    {
        Ok(Some(true)) => Some(full_id),
        _ => {
            println!("{}", "(model unchanged)".bright_black());
            None
        }
    }
}

/// Direct (non-interactive) switch from `/model <name>`. Provider is
/// inferred from id shape; we trust the user and let the LLM backend
/// surface "model not found" errors at first use rather than
/// pre-validating against a list.
fn switch_direct(
    agent: &mut Agent,
    arg: &str,
    current_provider: Provider,
    current_model: &str,
) {
    let provider = if arg.contains('/') {
        Provider::Groq
    } else if arg.contains(':') {
        Provider::Ollama
    } else {
        current_provider
    };
    if provider == current_provider && arg == current_model {
        println!(
            "{} {} {}",
            "already on".bright_black(),
            provider.as_str().bright_white(),
            arg.bright_white(),
        );
        return;
    }
    apply_switch(agent, provider, arg.to_string());
}

/// Build the new backend, swap it onto the agent, and persist the
/// choice. Shared by interactive and direct paths so they don't drift.
fn apply_switch(agent: &mut Agent, provider: Provider, model: String) {
    let think = agent.llm.think();
    let trace = agent.llm.trace();
    match agent
        .backend_config
        .build(provider, model.clone(), think, trace)
    {
        Ok(new_backend) => {
            agent.llm = new_backend;
            println!(
                "{} {} {}",
                "switched to".bright_black(),
                provider.as_str().bright_white(),
                model.bright_white(),
            );
            if let Err(e) = config::update(|c| {
                c.provider = Some(provider);
                c.model = Some(model.clone());
            }) {
                agent.display.show_error(&format!("config save: {e}"));
            }
            if matches!(provider, Provider::Ollama) {
                println!(
                    "{}",
                    "  (run /warm to preload weights before the next prompt)"
                        .bright_black()
                        .italic()
                );
            }
        }
        Err(e) => {
            agent.display.show_error(&e.to_string());
        }
    }
}

async fn fetch_groq(agent: &Agent) -> Result<Vec<ModelInfo>> {
    let api_key = agent
        .backend_config
        .api_key
        .clone()
        .or_else(|| std::env::var("GROQ_API_KEY").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no GROQ_API_KEY available; pass --api-key=<key> or set GROQ_API_KEY"
            )
        })?;
    models::list_groq(&agent.backend_config.groq_host, &api_key).await
}

/// Fetch both the local (downloaded) and remote (registry catalogue)
/// Ollama lists, merge, dedupe. Local entries appear first; remote
/// entries are filtered to slugs that aren't already represented in
/// `local` (matched by the base name before any `:tag`).
async fn fetch_ollama(agent: &Agent) -> Result<Vec<ModelInfo>> {
    let local = models::list_ollama_local(&agent.backend_config.ollama_host)
        .await
        .map_err(|e| anyhow::anyhow!("ollama daemon: {e}"))?;
    // Best-effort: a failed library scrape shouldn't block the picker.
    let remote = match models::list_ollama_remote().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "  {} {}",
                "warn:".bright_yellow(),
                format!("ollama.com library: {e}").bright_black()
            );
            Vec::new()
        }
    };
    let downloaded_bases: HashSet<String> = local
        .iter()
        .map(|m| m.id.split(':').next().unwrap_or(&m.id).to_string())
        .collect();
    let mut combined = local;
    for r in remote {
        if !downloaded_bases.contains(&r.id) {
            combined.push(r);
        }
    }
    Ok(combined)
}

/// Plain text — no ANSI escapes. dialoguer's FuzzySelect applies its
/// own highlight to the cursor row by wrapping the item string in
/// styling escapes; if the item already contains `\x1b[...m` codes
/// (from `colored`), the highlight terminator lands mid-sequence and
/// the row prints raw `[92m...[0m` text on the selected line. Padding
/// the badge to a fixed width keeps the model column aligned without
/// needing color.
fn render_picker_row(m: &ModelInfo) -> String {
    let badge = match m.state {
        ModelState::Downloaded => "[downloaded]",
        ModelState::AvailableRemote => "[available] ",
        ModelState::Hosted => "[hosted]    ",
    };
    let size = m
        .size_bytes
        .map(|n| format!("  ({})", models::human_size(n)))
        .unwrap_or_default();
    format!("{badge} {}{size}", m.id)
}

/// Pull a model from the Ollama registry, painting per-line progress
/// over a single terminal line. The daemon emits many status updates
/// (one per layer + verifying + success); we use `\r` to overwrite
/// so the operator sees a live indicator instead of a wall of text.
async fn run_pull(agent: &Agent, name: &str) -> Result<()> {
    use std::io::Write;
    println!(
        "{}",
        format!("pulling {name}...").bright_black().italic()
    );
    let host = agent.backend_config.ollama_host.clone();
    let res = models::pull_ollama(&host, name, |status| {
        // Pad to clear any trailing characters from the previous,
        // longer status line. 64 chars covers any realistic message.
        print!("\r  {:<64}", status.bright_black());
        let _ = std::io::stdout().flush();
    })
    .await;
    // Newline so the next prompt doesn't share a line with the last
    // progress chunk.
    println!();
    res
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
