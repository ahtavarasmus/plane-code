//! Esc-during-stream detection.
//!
//! Architecture: the main thread owns terminal mode via the
//! `RawInputGuard` returned by `enter_raw_input()`. A separate
//! spawn_blocking watcher only polls for keystrokes; it never touches
//! termios. This ordering matters because the watcher's lifetime is
//! detached - we don't await its JoinHandle - so any state it left
//! behind would leak. By keeping all `tcsetattr` calls on the main
//! thread, the RAII guard's Drop runs in normal control flow and we
//! get back a sane terminal even on panic or early return.
//!
//! "Raw input" here means: line buffering off (`ICANON`), echo off,
//! signal generation (`ISIG`) off so Ctrl-C can be intercepted by the
//! watcher rather than killing the process - BUT output post-processing
//! (`OPOST`/`ONLCR`) stays ON so `\n` in `println!` still translates
//! to `\r\n` and prints render normally. crossterm's `enable_raw_mode`
//! disables `OPOST` (it calls `cfmakeraw`), which is why we use
//! direct termios calls here instead.
//!
//! Keystrokes that aren't Esc are consumed and discarded - typing
//! during a stream is lost, same as Claude Code.
//!
//! Recovery: if the process gets SIGKILLed mid-stream, raw mode will
//! leak. From the shell, run `stty sane` and press Ctrl-J (real LF,
//! since Enter is sending CR which won't be translated until OPOST is
//! restored).
//!
//! Why not crossterm's enable_raw_mode? Two reasons: (1) cfmakeraw
//! disables OPOST which mangles output, (2) crossterm uses a process-
//! global mutex to remember the prior termios, so nested or concurrent
//! enable/disable pairs can corrupt the saved state. Direct termios
//! with our own RAII guard is simpler and correct by construction.

use crossterm::event::{poll, read, Event, KeyCode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

/// RAII guard for raw-input mode. Drop restores the original termios.
/// Ownership stays on the main thread so Drop runs in normal control
/// flow.
pub struct RawInputGuard {
    original: libc::termios,
    /// Set to false if the original mode wasn't captured (non-tty,
    /// headless test). Drop becomes a no-op.
    valid: bool,
}

impl Drop for RawInputGuard {
    fn drop(&mut self) {
        if !self.valid {
            return;
        }
        // Restore the captured termios, but also force OPOST + ONLCR
        // on. If the captured state had them off (rustyline 14
        // sometimes leaves the terminal that way after its prompt
        // mode), preserving as-is would leave the user staring at
        // stair-stepped output every time they hit run_turn. The
        // operator-visible expectation is "newlines work"; forcing
        // these flags on at restore upholds that even when we inherit
        // a degraded state.
        let mut restore = self.original;
        restore.c_oflag |= libc::OPOST | libc::ONLCR;
        unsafe {
            let _ = libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &restore);
        }
    }
}

/// Switch stdin to character-by-character input (no line buffering,
/// no echo, no signal generation) while leaving output processing
/// untouched. The returned guard restores the original mode on drop.
///
/// Returns `Ok(None)` when stdin isn't a tty (running under a pipe,
/// headless harness) - the caller should treat this as "Esc detection
/// not available" and proceed without it.
pub fn enter_raw_input() -> std::io::Result<Option<RawInputGuard>> {
    unsafe {
        if libc::isatty(libc::STDIN_FILENO) == 0 {
            return Ok(None);
        }
        let mut original: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(libc::STDIN_FILENO, &mut original) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut raw = original;
        // Input flags: turn off CR-to-NL translation, parity check,
        // strip 8th bit, software flow control.
        raw.c_iflag &= !(libc::BRKINT
            | libc::ICRNL
            | libc::INPCK
            | libc::ISTRIP
            | libc::IXON);
        // Local flags: no echo, no canonical mode (line buffering),
        // no signal generation, no extended input.
        raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG);
        // Output: FORCE OPOST + ONLCR on. cfmakeraw would clear them
        // (mangling \n in output), and rustyline's prompt mode also
        // sometimes returns a termios that already has them off. We
        // want raw input + cooked output regardless of inherited
        // state, so set them explicitly.
        raw.c_oflag |= libc::OPOST | libc::ONLCR;
        // Char size 8 bits.
        raw.c_cflag |= libc::CS8;
        // VMIN=0, VTIME=0: read returns immediately with whatever is
        // available, even nothing. crossterm's poll/read use this.
        raw.c_cc[libc::VMIN] = 0;
        raw.c_cc[libc::VTIME] = 0;
        if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Some(RawInputGuard {
            original,
            valid: true,
        }))
    }
}

/// Spawn a detached blocking task that polls stdin for Esc. The task
/// does NOT manage terminal mode - the caller must have already
/// entered raw input via `enter_raw_input()` before spawning, and
/// must hold the guard until after the task has been signalled to
/// exit.
///
/// Signalling: when `cancel` flips to true, the task wakes within
/// ~50ms and exits cleanly. When Esc is observed, `esc_tx.send(())`
/// fires before the task returns.
pub fn spawn_esc_watcher(cancel: Arc<AtomicBool>, esc_tx: oneshot::Sender<()>) {
    tokio::task::spawn_blocking(move || {
        loop {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            match poll(Duration::from_millis(50)) {
                Ok(true) => {
                    if let Ok(Event::Key(k)) = read() {
                        if k.code == KeyCode::Esc {
                            let _ = esc_tx.send(());
                            return;
                        }
                        // Other keystrokes are consumed and discarded.
                    }
                }
                Ok(false) => continue,
                Err(_) => return,
            }
        }
    });
}
