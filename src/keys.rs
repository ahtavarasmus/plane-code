//! Single-purpose helper: watch stdin for Esc while a long-running async
//! task is in flight, signaling the caller via a oneshot when the key
//! fires. Used by the agent loop to let the operator abort an in-flight
//! LLM stream the same way Claude Code does.
//!
//! Implementation: spawn_blocking thread enables raw mode and polls the
//! crossterm event queue every 50ms. The caller flips an AtomicBool
//! when its own future wins the race; the watcher checks that flag
//! between polls and exits cleanly, restoring terminal mode. If raw
//! mode can't be enabled (non-tty, headless test, etc.) we just park
//! the thread until cancelled - Esc detection silently no-ops, the
//! stream still completes normally.
//!
//! Keystrokes that aren't Esc (including Ctrl-C, arrows, plain letters)
//! are consumed and discarded. This means typing during a stream is
//! lost - same behavior as Claude Code.
//!
//! Drop guard ensures raw mode is disabled even on panic.
//!
//! Notes:
//!   - We treat any Esc keypress as the abort signal regardless of
//!     KeyEventKind, since Press is the only kind emitted on Unix
//!     terminals without explicit kitty-protocol setup.
//!   - The blocking thread is detached: when the parent's stream wins,
//!     we set `cancel` and move on without awaiting the watcher. It'll
//!     wake within ~50ms and clean up after itself.
//!   - This module has no dependencies on the rest of plane-code so
//!     it's safe to call from anywhere that has tokio + crossterm.

use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

struct RawGuard;
impl Drop for RawGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

/// Spawn a detached blocking task that watches for Esc. When pressed,
/// `esc_tx.send(())` fires; the parent learns about it via its
/// `oneshot::Receiver`. When the parent's other future wins, it sets
/// `cancel` to true and the watcher exits without sending.
///
/// `esc_tx` is consumed; only one signal possible per watcher.
pub fn spawn_esc_watcher(cancel: Arc<AtomicBool>, esc_tx: oneshot::Sender<()>) {
    tokio::task::spawn_blocking(move || {
        if enable_raw_mode().is_err() {
            // Raw mode unavailable - park until cancelled so we don't
            // burn cpu, but no Esc detection possible.
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(100));
            }
            return;
        }
        let _guard = RawGuard;

        loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            // Short timeout so we re-check `cancel` quickly when the
            // parent's future wins. Longer timeouts feel laggy on exit.
            match poll(Duration::from_millis(50)) {
                Ok(true) => {
                    if let Ok(Event::Key(k)) = read() {
                        if k.code == KeyCode::Esc {
                            let _ = esc_tx.send(());
                            return;
                        }
                        // Discard everything else - typing during a
                        // stream gets eaten, matching Claude Code.
                    }
                }
                Ok(false) => continue,
                Err(_) => break,
            }
        }
    });
}
