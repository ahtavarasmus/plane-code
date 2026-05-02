//! Verbose terminal display for the agent loop.
//!
//! Designed for debug-mode use: prints every event the model sees in full,
//! no truncation. Streaming-aware: tokens render as they arrive, with
//! `<think>...</think>` blocks split out into a dim italic style so the
//! operator can watch the reasoning unfold separately from the final text.

use crate::ollama::{ChatMessage, ChatResponse, DeltaKind, ToolCall};
use colored::{ColoredString, Colorize};
use serde_json::Value;
use std::io::Write;

pub struct Display {
    /// When false, suppresses verbose output. All `show_*` helpers become
    /// no-ops; errors still print.
    pub verbose: bool,
}

impl Display {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }

    pub fn banner(&self, crate_name: &str, indexed: (usize, usize, usize, usize, usize)) {
        if !self.verbose {
            return;
        }
        let (fns, types, traits, modules, files) = indexed;
        println!();
        println!(
            "{}",
            "═══════════════════════════════════════════════════════════════"
                .bright_black()
        );
        println!(
            "  {}  crate=`{}`",
            "planecode".bold().bright_cyan(),
            crate_name.bright_white()
        );
        println!(
            "  indexed: {} fn  {} ty  {} tr  {} mod  {} files",
            fns.to_string().bright_white(),
            types.to_string().bright_white(),
            traits.to_string().bright_white(),
            modules.to_string().bright_white(),
            files.to_string().bright_white(),
        );
        println!(
            "  type {} for slash commands, {} to exit",
            "/help".bright_yellow(),
            "/quit".bright_yellow()
        );
        println!(
            "{}",
            "═══════════════════════════════════════════════════════════════"
                .bright_black()
        );
        println!();
    }

    pub fn show_turn_start(&self, turn: usize, max_turns: usize) {
        if !self.verbose {
            return;
        }
        let label = format!("turn {}/{}", turn + 1, max_turns);
        println!(
            "{} {}",
            "──".bright_black(),
            label.bright_black().italic()
        );
    }

    /// Make a fresh streaming printer for one model response. Call
    /// `feed(delta)` for every chunk that arrives, then `finish()` once
    /// the stream is complete.
    pub fn stream_printer(&self) -> StreamPrinter {
        StreamPrinter::new(self.verbose)
    }

    /// Inline note printed after the streamed response when the model
    /// chose to dispatch tools. The tool-call payloads themselves are
    /// rendered separately by `show_tool_call`.
    pub fn show_tool_dispatch_note(&self, tool_calls: &[ToolCall]) {
        if !self.verbose || tool_calls.is_empty() {
            return;
        }
        let names: Vec<String> = tool_calls
            .iter()
            .map(|c| c.function.name.clone())
            .collect();
        println!(
            "  {} {}",
            "→ tools:".bright_black(),
            names.join(", ").bright_yellow()
        );
        println!();
    }

    pub fn show_tool_call(&self, call: &ToolCall) {
        if !self.verbose {
            return;
        }
        let header = format!("tool call · {}", call.function.name);
        section_header(&header, "▸", "yellow");
        let args = pretty(&call.function.arguments);
        for line in args.lines() {
            println!("  {}", line.cyan());
        }
        println!();
    }

    pub fn show_tool_result(&self, name: &str, value: &Value) {
        if !self.verbose {
            return;
        }
        let header = format!("tool result · {}", name);
        section_header(&header, "◂", "bright_black");
        let text = pretty(value);
        let line_count = text.lines().count();
        let bytes = text.len();
        for line in text.lines() {
            println!("  {}", line.bright_black());
        }
        println!(
            "  {}",
            format!("({} lines, {} bytes)", line_count, bytes)
                .bright_black()
                .italic()
        );
        println!();
    }

    /// One-line summary after each model turn: wall-clock total,
    /// current context size (the prompt the model just consumed - same
    /// thing as session tokens), and output tokens for this turn.
    ///
    /// `wall` is measured around the chat_stream call by the caller.
    /// Ollama also reports its own total_duration, which we prefer when
    /// non-zero (excludes our serialization overhead). Groq returns 0
    /// for all duration fields so wall is the only signal we have.
    pub fn show_response_summary(&self, resp: &ChatResponse, wall: std::time::Duration) {
        if !self.verbose {
            return;
        }
        let total = resp
            .total_duration
            .map(ns_to_secs)
            .filter(|s| *s > 0.0)
            .unwrap_or_else(|| wall.as_secs_f64());
        let context_tokens = resp.prompt_eval_count.unwrap_or(0);
        let eval_tokens = resp.eval_count.unwrap_or(0);
        let line = format!(
            "  {} total {:.2}s · context {} tok · out {} tok",
            "·".bright_black(),
            total,
            context_tokens,
            eval_tokens,
        );
        println!("{}", line.bright_black().italic());
        println!();
    }

    /// Last-resort visibility: when an assistant turn produces neither
    /// content nor tool calls (or any other "what just happened?"
    /// situation), dump the raw response so the operator can see exactly
    /// what came back from Ollama.
    pub fn show_raw_response(&self, resp: &ChatResponse, note: &str) {
        // Always print, even if !verbose, since this only fires on
        // anomalies the operator definitely wants to see.
        println!(
            "{} {}",
            "▸ raw response".red().bold(),
            note.bright_black().italic()
        );
        let v = serde_json::to_value(resp).unwrap_or(Value::Null);
        let text = pretty(&v);
        for line in text.lines() {
            println!("  {}", line.red());
        }
        println!();
    }

    pub fn show_max_turns(&self, max_turns: usize) {
        if !self.verbose {
            return;
        }
        println!(
            "{}",
            format!("[agent hit max_turns={max_turns} without a final response]")
                .bright_red()
        );
    }

    pub fn show_error(&self, msg: &str) {
        eprintln!("{}: {}", "error".bright_red().bold(), msg);
    }

    /// Re-render a saved conversation in the same visual style the
    /// live REPL produces, so /resume lands the operator in a screen
    /// that looks like the chat just happened. Skips the system prompt
    /// (no operator value), elides per-turn summary lines (no token
    /// counts available for replayed turns), and uses the same headers
    /// and indents as the streaming printer.
    pub fn render_transcript(&self, messages: &[ChatMessage]) {
        if !self.verbose {
            return;
        }
        for m in messages {
            match m.role.as_str() {
                "system" => continue,
                "user" => {
                    // Mirror the rustyline prompt: cyan ›, then content.
                    let first = m.content.lines().next().unwrap_or("");
                    let rest: Vec<&str> = m.content.lines().skip(1).collect();
                    println!("{} {}", "›".bright_cyan().bold(), first);
                    for line in rest {
                        println!("  {}", line);
                    }
                    println!();
                }
                "assistant" => {
                    if let Some(t) = &m.thinking {
                        if !t.trim().is_empty() {
                            section_header("thinking", "•", "magenta");
                            for line in t.lines() {
                                println!("  {}", line.bright_black().italic());
                            }
                            println!();
                        }
                    }
                    if !m.content.trim().is_empty() {
                        section_header("assistant", "▸", "green");
                        for line in m.content.lines() {
                            println!("  {}", line);
                        }
                        println!();
                    }
                    if !m.tool_calls.is_empty() {
                        let names: Vec<String> = m
                            .tool_calls
                            .iter()
                            .map(|c| c.function.name.clone())
                            .collect();
                        println!(
                            "  {} {}",
                            "→ tools:".bright_black(),
                            names.join(", ").bright_yellow()
                        );
                        for call in &m.tool_calls {
                            let header = format!("tool call · {}", call.function.name);
                            section_header(&header, "▸", "yellow");
                            let args = pretty(&call.function.arguments);
                            for line in args.lines() {
                                println!("  {}", line.cyan());
                            }
                            println!();
                        }
                    }
                }
                "tool" => {
                    let name = m.tool_name.as_deref().unwrap_or("?");
                    let header = format!("tool result · {}", name);
                    section_header(&header, "◂", "bright_black");
                    // Tool results are stored as JSON strings in the
                    // history; pretty-print if parseable, else dump raw.
                    let parsed: Result<Value, _> = serde_json::from_str(&m.content);
                    let text = match parsed {
                        Ok(v) => pretty(&v),
                        Err(_) => m.content.clone(),
                    };
                    for line in text.lines() {
                        println!("  {}", line.bright_black());
                    }
                    println!();
                }
                _ => {}
            }
        }
    }

    pub fn dump_context(&self, messages: &[ChatMessage]) {
        println!();
        println!(
            "{}",
            "═════════════ message history (model context) ═════════════"
                .bright_black()
        );
        for (i, m) in messages.iter().enumerate() {
            let role = match m.role.as_str() {
                "system" => m.role.magenta().bold(),
                "user" => m.role.blue().bold(),
                "assistant" => m.role.green().bold(),
                "tool" => m.role.yellow().bold(),
                _ => m.role.normal(),
            };
            let header = format!("[{i:>3}] {role}");
            if let Some(name) = &m.tool_name {
                println!("{header}  ({})", name.cyan());
            } else {
                println!("{header}");
            }
            if !m.content.is_empty() {
                for line in m.content.lines() {
                    println!("    {}", line.bright_black());
                }
            }
            for call in &m.tool_calls {
                let args = pretty(&call.function.arguments);
                println!(
                    "    {} {}",
                    "→ tool_call".yellow(),
                    call.function.name.bright_yellow()
                );
                for line in args.lines() {
                    println!("      {}", line.cyan());
                }
            }
            println!();
        }
        println!(
            "{}",
            "════════════════════════════════════════════════════════════"
                .bright_black()
        );
        println!();
    }
}

fn section_header(label: &str, glyph: &str, color: &str) {
    let painted: ColoredString = match color {
        "blue" => label.blue().bold(),
        "green" => label.green().bold(),
        "yellow" => label.yellow().bold(),
        "magenta" => label.magenta().bold(),
        "bright_black" => label.bright_black().bold(),
        _ => label.normal().bold(),
    };
    println!("{} {}", glyph, painted);
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
}

fn ns_to_secs(ns: u64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

// --------------------- streaming printer ---------------------

pub struct StreamPrinter {
    verbose: bool,
    /// Tracks which kind we last emitted, so we can drop a separator
    /// line when the stream switches (thinking -> content typically).
    last_kind: Option<DeltaKind>,
    thinking_header_shown: bool,
    content_header_shown: bool,
    /// Whether anything was printed at all (used by finish to add a
    /// terminating blank line).
    emitted_any: bool,
}

impl StreamPrinter {
    fn new(verbose: bool) -> Self {
        Self {
            verbose,
            last_kind: None,
            thinking_header_shown: false,
            content_header_shown: false,
            emitted_any: false,
        }
    }

    /// Feed a chunk of streamed text. `kind` says whether it came from
    /// the model's thinking field or its visible content field; the
    /// printer routes it to the appropriate styled output.
    pub fn feed(&mut self, kind: DeltaKind, delta: &str) {
        if !self.verbose || delta.is_empty() {
            return;
        }
        if self.last_kind != Some(kind) {
            // Switching channels: blank line between sections.
            if self.last_kind.is_some() {
                println!();
            }
            self.last_kind = Some(kind);
            self.print_header(kind);
        }
        match kind {
            DeltaKind::Thinking => self.emit_thinking(delta),
            DeltaKind::Content => self.emit_content(delta),
        }
        let _ = std::io::stdout().flush();
        self.emitted_any = true;
    }

    /// Flush any final state and add a trailing blank line if anything
    /// was rendered.
    pub fn finish(&mut self) {
        if !self.verbose {
            return;
        }
        if self.emitted_any {
            println!();
            println!();
        }
        let _ = std::io::stdout().flush();
    }

    fn print_header(&mut self, kind: DeltaKind) {
        match kind {
            DeltaKind::Thinking => {
                if !self.thinking_header_shown {
                    section_header("thinking", "•", "magenta");
                    self.thinking_header_shown = true;
                }
            }
            DeltaKind::Content => {
                if !self.content_header_shown {
                    section_header("assistant", "▸", "green");
                    self.content_header_shown = true;
                }
            }
        }
        // Two-space indent before the first character of each section.
        print!("  ");
    }

    fn emit_content(&mut self, text: &str) {
        for ch in text.chars() {
            if ch == '\n' {
                print!("\n  ");
            } else {
                print!("{ch}");
            }
        }
    }

    fn emit_thinking(&mut self, text: &str) {
        // Dim italic gray. We re-style per character chunk so newlines
        // get the indent treatment without baking ANSI codes into
        // partial-line state.
        let mut chunk = String::new();
        for ch in text.chars() {
            if ch == '\n' {
                if !chunk.is_empty() {
                    print!("{}", chunk.bright_black().italic());
                    chunk.clear();
                }
                print!("\n  ");
            } else {
                chunk.push(ch);
            }
        }
        if !chunk.is_empty() {
            print!("{}", chunk.bright_black().italic());
        }
    }
}
