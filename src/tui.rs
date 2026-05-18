//! Full-screen terminal UI: scrollback above, status line, input box pinned at bottom.
//!
//! Architecture:
//!   - We `dup2` stdout into a pipe so anything `Display` (or anything
//!     else) prints with `println!` ends up on a reader task instead
//!     of the screen. The reader splits incoming bytes by newlines
//!     and ships each line through `ui_tx` to the render loop.
//!   - The render loop owns all terminal output via `ratatui` +
//!     `crossterm` raw mode. It draws three regions every frame:
//!     scrollback (whatever has been printed so far, ANSI parsed),
//!     a one-row status line (workspace + branch + busy state), and
//!     a bordered input box at the bottom. The input box stays put
//!     while output flows above it - same shape as Claude Code.
//!   - User input is collected character-by-character in `app.input`.
//!     On Enter, the line is forwarded to a handler (slash command or
//!     agent.run_turn). The agent runs inline on the same task; while
//!     it's running, the render loop continues firing on a tick so
//!     streaming output keeps painting.
//!   - Esc during streaming flips an `Arc<AtomicBool>` interrupt flag
//!     that `agent.run_turn` checks inside its chat_stream select.
//!     This replaces the old keys.rs raw-mode dance: the TUI already
//!     owns the keyboard, so the agent just polls a flag.
//!
//! What's NOT here yet (deliberate scope):
//!   - Mouse / scroll-wheel scrollback navigation (PgUp/PgDn work).
//!   - In-TUI picker for /model and /resume - we temporarily leave
//!     the alt screen, run dialoguer, then re-enter. Not pretty but
//!     it works.
//!   - Multi-line input. Enter submits, no shift-Enter for newline.

use crate::agent::Agent;
use crate::cli::{handle_slash_for_tui, SlashOutcome};
use ansi_to_tui::IntoText;
use anyhow::{Context, Result};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind,
        KeyModifiers, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io::{self, Read, Stdout};
use std::os::unix::io::FromRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Output events flowing IN to the render loop.
pub enum UiOut {
    /// Raw bytes from the captured stdout pipe. May contain partial
    /// lines (mid-token streaming chunks), embedded `\n` newlines,
    /// or both. The render loop appends to the last scrollback line
    /// and splits on `\n` to start new lines. This is what makes
    /// live token-by-token streaming work: chunks land here as they
    /// arrive instead of being buffered until end-of-line.
    Append(String),
    /// Update the right-side "busy" indicator on the status line.
    Status(String),
}

/// Default app icon left in the status line. The right side shows the
/// busy state (idle / streaming / processing).
struct App {
    scrollback: Vec<String>,
    input: String,
    cursor: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    history_saved: String,
    status_left: String,
    status_right: String,
    /// Lines from the bottom (0 = stuck to bottom). PgUp/PgDn change this.
    scroll_back: usize,
    /// True while agent.run_turn is in flight; Esc + Ctrl-C trigger interrupt.
    busy: bool,
}

impl App {
    fn new(status_left: String) -> Self {
        Self {
            scrollback: Vec::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            history_saved: String::new(),
            status_left,
            status_right: "ready".into(),
            scroll_back: 0,
            busy: false,
        }
    }

    /// Append a chunk that may contain partial text and embedded
    /// newlines. Splits on `\n` to start new scrollback lines while
    /// extending the current trailing line in place. This is what
    /// makes token-by-token streaming look live: one character lands,
    /// one character renders.
    fn append(&mut self, chunk: &str) {
        if self.scrollback.is_empty() {
            self.scrollback.push(String::new());
        }
        for ch in chunk.chars() {
            if ch == '\n' {
                self.scrollback.push(String::new());
            } else if ch != '\r' {
                self.scrollback.last_mut().unwrap().push(ch);
            }
        }
        self.scroll_back = 0;
    }

    fn render(&self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),    // scrollback
                Constraint::Length(1), // status line
                Constraint::Length(3), // input box (with border)
            ])
            .split(area);

        // Scrollback: parse ANSI from accumulated lines.
        let scrollback_text = self.build_scrollback_text();
        let total_lines = scrollback_text.lines.len();
        let height = chunks[0].height as usize;
        // scroll_back=0 -> show last `height` lines. scroll_back=N -> shift up by N.
        let skip = total_lines
            .saturating_sub(height)
            .saturating_sub(self.scroll_back);
        let scrollback_widget = Paragraph::new(scrollback_text)
            .scroll((skip as u16, 0))
            .wrap(Wrap { trim: false });
        f.render_widget(scrollback_widget, chunks[0]);

        // Status line.
        let busy_color = if self.busy { Color::Yellow } else { Color::Green };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(
                self.status_left.as_str(),
                Style::default().add_modifier(Modifier::DIM),
            ),
            Span::raw("  "),
            Span::styled(
                self.status_right.as_str(),
                Style::default().fg(busy_color),
            ),
        ]));
        f.render_widget(status, chunks[1]);

        // Input box.
        let input_block = Block::default().borders(Borders::ALL);
        let prefix = if self.busy { "(busy) " } else { "› " };
        let input_text = Line::from(vec![
            Span::styled(prefix, Style::default().fg(Color::Cyan)),
            Span::raw(self.input.as_str()),
        ]);
        let input_widget = Paragraph::new(input_text).block(input_block);
        f.render_widget(input_widget, chunks[2]);

        // Cursor inside input box.
        if !self.busy {
            let prefix_w = prefix.chars().count() as u16;
            let cursor_x = chunks[2].x + 1 + prefix_w + self.cursor as u16;
            let cursor_y = chunks[2].y + 1;
            f.set_cursor_position((cursor_x, cursor_y));
        }
    }

    fn build_scrollback_text(&self) -> Text<'_> {
        // Concatenate all scrollback lines, parse ANSI once. ratatui's
        // Paragraph handles vertical scroll via its `scroll` setter, so
        // we just hand it the full text.
        let mut buf = String::new();
        for line in &self.scrollback {
            buf.push_str(line);
            buf.push('\n');
        }
        // ansi-to-tui parses ANSI escape sequences emitted by the
        // `colored` crate into ratatui Spans with proper styling.
        buf.into_text()
            .unwrap_or_else(|_| Text::raw(self.scrollback.join("\n")))
    }
}

/// What user input or key events resolve to once mapped through App.
enum Action {
    None,
    Submit(String),
    Interrupt,
    Quit,
}

fn handle_key(app: &mut App, key: event::KeyEvent) -> Action {
    if key.kind != KeyEventKind::Press && key.kind != KeyEventKind::Repeat {
        // Ignore Release events (some terminals emit them when
        // kitty-protocol or similar is active).
        return Action::None;
    }

    // Streaming mode: only Esc / Ctrl-C / Ctrl-D do anything; typing
    // is buffered into input but won't render visibly until busy ends.
    if app.busy {
        match key.code {
            KeyCode::Esc => return Action::Interrupt,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Action::Interrupt;
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Action::Quit;
            }
            _ => return Action::None,
        }
    }

    // Idle mode.
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') => {
                app.input.clear();
                app.cursor = 0;
                return Action::None;
            }
            KeyCode::Char('d') => {
                if app.input.is_empty() {
                    return Action::Quit;
                }
                return Action::None;
            }
            KeyCode::Char('a') => {
                app.cursor = 0;
                return Action::None;
            }
            KeyCode::Char('e') => {
                app.cursor = app.input.chars().count();
                return Action::None;
            }
            KeyCode::Char('u') => {
                app.input.clear();
                app.cursor = 0;
                return Action::None;
            }
            KeyCode::Char('w') => {
                let bytes = app.input.as_bytes();
                let mut new_cursor = byte_offset(&app.input, app.cursor);
                while new_cursor > 0 && bytes[new_cursor - 1].is_ascii_whitespace() {
                    new_cursor -= 1;
                }
                while new_cursor > 0 && !bytes[new_cursor - 1].is_ascii_whitespace() {
                    new_cursor -= 1;
                }
                let cursor_byte = byte_offset(&app.input, app.cursor);
                app.input.replace_range(new_cursor..cursor_byte, "");
                app.cursor = char_offset(&app.input, new_cursor);
                return Action::None;
            }
            _ => return Action::None,
        }
    }

    match key.code {
        KeyCode::Char(c) => {
            let byte_at = byte_offset(&app.input, app.cursor);
            app.input.insert(byte_at, c);
            app.cursor += 1;
            app.history_idx = None;
        }
        KeyCode::Backspace => {
            if app.cursor > 0 {
                let prev_byte = byte_offset(&app.input, app.cursor - 1);
                let cur_byte = byte_offset(&app.input, app.cursor);
                app.input.replace_range(prev_byte..cur_byte, "");
                app.cursor -= 1;
            }
        }
        KeyCode::Left => {
            if app.cursor > 0 {
                app.cursor -= 1;
            }
        }
        KeyCode::Right => {
            if app.cursor < app.input.chars().count() {
                app.cursor += 1;
            }
        }
        KeyCode::Home => app.cursor = 0,
        KeyCode::End => app.cursor = app.input.chars().count(),
        KeyCode::Up => {
            if app.history.is_empty() {
                return Action::None;
            }
            let new_idx = match app.history_idx {
                Some(i) if i > 0 => i - 1,
                Some(i) => i,
                None => {
                    app.history_saved = std::mem::take(&mut app.input);
                    app.history.len() - 1
                }
            };
            app.history_idx = Some(new_idx);
            app.input = app.history[new_idx].clone();
            app.cursor = app.input.chars().count();
        }
        KeyCode::Down => {
            if let Some(i) = app.history_idx {
                if i + 1 < app.history.len() {
                    app.history_idx = Some(i + 1);
                    app.input = app.history[i + 1].clone();
                    app.cursor = app.input.chars().count();
                } else {
                    app.history_idx = None;
                    app.input = std::mem::take(&mut app.history_saved);
                    app.cursor = app.input.chars().count();
                }
            }
        }
        KeyCode::PageUp => {
            app.scroll_back = app.scroll_back.saturating_add(10);
        }
        KeyCode::PageDown => {
            app.scroll_back = app.scroll_back.saturating_sub(10);
        }
        KeyCode::Enter => {
            let text = std::mem::take(&mut app.input);
            app.cursor = 0;
            app.history_idx = None;
            if !text.is_empty() {
                app.history.push(text.clone());
                return Action::Submit(text);
            }
        }
        KeyCode::Esc => {
            app.input.clear();
            app.cursor = 0;
            app.history_idx = None;
        }
        _ => {}
    }
    Action::None
}

/// Convert a `chars` index into a byte offset into the string. Cheap
/// for short inputs (which our prompt is).
fn byte_offset(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

fn char_offset(s: &str, byte_idx: usize) -> usize {
    s.char_indices().take_while(|(i, _)| *i < byte_idx).count()
}

/// Workspace-name + git-branch description for the status line. Built
/// once at TUI start; if branch changes mid-session we'd need to
/// re-poll, but that's a niche need.
fn build_status_left(workspace: &Path) -> String {
    let dir_name = workspace
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| workspace.display().to_string());
    let branch = git2::Repository::open(workspace)
        .ok()
        .and_then(|repo| {
            repo.head()
                .ok()
                .and_then(|h| h.shorthand().map(String::from))
        })
        .unwrap_or_else(|| "no-branch".to_string());
    format!("{} · {}", dir_name, branch)
}

/// State returned by `capture_stdout`. Hold this until TUI exit; on
/// drop, fd 1 is restored to the real terminal.
struct StdoutCapture {
    saved_for_restore: libc::c_int,
}

impl Drop for StdoutCapture {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_for_restore, libc::STDOUT_FILENO);
            libc::close(self.saved_for_restore);
        }
    }
}

/// Redirect stdout (fd 1) into a pipe and spawn a reader that ships
/// incoming bytes (newlines and all) to `ui_tx` as they arrive.
/// Returns:
///   - a `File` opened on a separate dup of the original stdout fd -
///     hand this to ratatui's CrosstermBackend so its drawing reaches
///     the real terminal, not the pipe.
///   - a `StdoutCapture` guard that restores fd 1 on drop, so the
///     shell sees a normal terminal when plane-code exits.
///
/// Crucial ordering: `EnterAlternateScreen` and `enable_raw_mode` must
/// happen AFTER this returns, because we want them applied to the
/// saved-real-stdout fd that ratatui draws on, not to the pipe fd
/// that's now masquerading as fd 1.
fn capture_stdout(
    ui_tx: mpsc::UnboundedSender<UiOut>,
) -> Result<(std::fs::File, StdoutCapture)> {
    unsafe {
        let mut fds = [0; 2];
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error()).context("pipe");
        }
        let read_fd = fds[0];
        let write_fd = fds[1];

        // Two duplicate fds pointing at the real stdout terminal:
        // one for ratatui to draw to, one for restoring fd 1 on drop.
        // Both must be made before the dup2 swap below.
        let ratatui_fd = libc::dup(libc::STDOUT_FILENO);
        let saved_for_restore = libc::dup(libc::STDOUT_FILENO);
        if ratatui_fd < 0 || saved_for_restore < 0 {
            libc::close(read_fd);
            libc::close(write_fd);
            if ratatui_fd >= 0 {
                libc::close(ratatui_fd);
            }
            if saved_for_restore >= 0 {
                libc::close(saved_for_restore);
            }
            return Err(io::Error::last_os_error()).context("dup stdout");
        }

        if libc::dup2(write_fd, libc::STDOUT_FILENO) < 0 {
            libc::close(ratatui_fd);
            libc::close(saved_for_restore);
            libc::close(read_fd);
            libc::close(write_fd);
            return Err(io::Error::last_os_error()).context("dup2 stdout");
        }
        libc::close(write_fd);

        // Make Rust's stdout unbuffered so streamed tokens appear live
        // instead of in batches when the buffer fills.
        let stdout_file = libc::fdopen(libc::STDOUT_FILENO, b"w\0".as_ptr() as *const _);
        if !stdout_file.is_null() {
            libc::setvbuf(stdout_file, std::ptr::null_mut(), libc::_IONBF, 0);
        }

        // Reader thread: ship raw bytes to the UI as they arrive.
        // We deliberately do NOT buffer until newline - streaming
        // tokens often arrive without newlines for many seconds, and
        // the operator wants to see them live. The App handles
        // splitting on `\n` into new scrollback lines.
        //
        // We DO have a small UTF-8 boundary buffer: a chunk may end
        // mid-character (e.g. an emoji split across two reads). We
        // hold the trailing incomplete bytes and prepend them to the
        // next read so we never decode garbled text.
        let read_file = std::fs::File::from_raw_fd(read_fd);
        std::thread::spawn(move || {
            let mut reader = read_file;
            let mut buf = [0u8; 4096];
            let mut leftover: Vec<u8> = Vec::new();
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        leftover.extend_from_slice(&buf[..n]);
                        let valid_up_to = match std::str::from_utf8(&leftover) {
                            Ok(_) => leftover.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if valid_up_to == 0 {
                            continue;
                        }
                        let s = String::from_utf8_lossy(&leftover[..valid_up_to]).into_owned();
                        leftover.drain(..valid_up_to);
                        if ui_tx.send(UiOut::Append(s)).is_err() {
                            return;
                        }
                    }
                    Err(_) => break,
                }
            }
            if !leftover.is_empty() {
                let s = String::from_utf8_lossy(&leftover).into_owned();
                let _ = ui_tx.send(UiOut::Append(s));
            }
        });

        let real_stdout = std::fs::File::from_raw_fd(ratatui_fd);
        Ok((real_stdout, StdoutCapture { saved_for_restore }))
    }
}

/// Main entry: spin up the TUI, drive the agent.
pub async fn run_tui(agent: &mut Agent, warm: bool) -> Result<()> {
    // Channels.
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiOut>();
    let (key_tx, mut key_rx) = mpsc::unbounded_channel::<event::Event>();

    // Capture stdout into ui_tx FIRST. Returns a File on the saved
    // real-stdout fd; we hand that to ratatui so its drawing bypasses
    // the pipe and reaches the actual terminal.
    let (real_stdout, _stdout_capture) = capture_stdout(ui_tx.clone())?;

    // Now bring up the terminal using the saved-real-stdout fd. Order
    // matters: enable_raw_mode operates on the controlling tty (so
    // it's fine to call after the swap), but EnterAlternateScreen
    // writes escape codes that must reach the real terminal - hence
    // executing them via `terminal.backend_mut()` whose writer is
    // `real_stdout`.
    enable_raw_mode().context("enable raw mode")?;
    let backend = CrosstermBackend::new(real_stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;
    execute!(
        terminal.backend_mut(),
        EnterAlternateScreen,
        EnableMouseCapture
    )
    .context("enter alt screen")?;

    let interrupt = agent.interrupt.clone();

    // Key reader thread - blocks on crossterm events, ships them to
    // the async loop.
    {
        let key_tx = key_tx.clone();
        std::thread::spawn(move || loop {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(evt) => {
                        if key_tx.send(evt).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                },
                Ok(false) => continue,
                Err(_) => return,
            }
        });
    }

    let status_left = build_status_left(&agent.workspace);
    let mut app = App::new(status_left);

    // Warm the model only if warm=true AND the backend actually needs
    // it (Ollama benefits from preloading weights; Groq's warm is a
    // no-op so the user-visible "warming..." line is just noise).
    if warm && agent.llm.provider() == "ollama" {
        println!("warming model...");
        if let Err(e) = agent.llm.warm().await {
            println!("warm failed: {e}");
        }
    }
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

    // Tick interval to ensure we redraw periodically (in case nothing
    // else happens, e.g. waiting for user input).
    let mut tick = tokio::time::interval(Duration::from_millis(50));

    loop {
        // Drain any pending UI output before drawing. Append events
        // carry raw chunks - possibly partial mid-token text - so we
        // need to render them as they come in for live streaming.
        while let Ok(msg) = ui_rx.try_recv() {
            match msg {
                UiOut::Append(s) => app.append(&s),
                UiOut::Status(s) => app.status_right = s,
            }
        }

        // Reflect agent's interrupt state in busy indicator.
        app.busy = agent.is_busy();
        if app.busy {
            app.status_right = "streaming · esc to interrupt".into();
        } else {
            app.status_right = "ready".into();
        }

        terminal.draw(|f| app.render(f))?;

        tokio::select! {
            biased;
            _ = tick.tick() => {
                // periodic redraw, no event
            }
            Some(evt) = key_rx.recv() => {
                // Mouse-wheel scrollback. ScrollUp moves further back
                // in history (scroll_back grows); ScrollDown moves
                // toward the bottom (scroll_back shrinks).
                if let Event::Mouse(m) = &evt {
                    match m.kind {
                        MouseEventKind::ScrollUp => {
                            app.scroll_back = app.scroll_back.saturating_add(3);
                        }
                        MouseEventKind::ScrollDown => {
                            app.scroll_back = app.scroll_back.saturating_sub(3);
                        }
                        _ => {}
                    }
                }
                if let Event::Key(k) = evt {
                    let action = handle_key(&mut app, k);
                    match action {
                        Action::None => {}
                        Action::Quit => break,
                        Action::Interrupt => {
                            interrupt.store(true, Ordering::Relaxed);
                        }
                        Action::Submit(text) => {
                            // Print user's input back into scrollback so
                            // the conversation shows what they typed.
                            println!("{}", format!("> {}", text));
                            // Slash command vs prompt.
                            if let Some(cmd) = text.strip_prefix('/') {
                                match handle_slash_for_tui(agent, cmd).await {
                                    SlashOutcome::Continue => {}
                                    SlashOutcome::Quit => break,
                                }
                            } else {
                                interrupt.store(false, Ordering::Relaxed);
                                agent.set_busy(true);
                                let res = agent.run_turn(&text).await;
                                agent.set_busy(false);
                                if let Err(e) = res {
                                    eprintln!("turn error: {e}");
                                }
                            }
                        }
                    }
                }
                // Resize / mouse events: ignore for now; ratatui's
                // next draw will pick up the new size automatically.
            }
        }
    }

    // Restore terminal. LeaveAlternateScreen must go through the
    // ratatui backend's writer (which still points at the real
    // stdout); after that, dropping `_stdout_capture` restores fd 1
    // so the shell sees a working stdout again.
    let _ = execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    disable_raw_mode().ok();
    drop(terminal);
    drop(_stdout_capture);
    Ok(())
}
