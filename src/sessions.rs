//! Per-workspace session persistence + listing for /resume.
//!
//! Each REPL session writes one JSON file at
//! `~/.plane-code/sessions/<workspace-hash>/<session-id>.json` after
//! every successful turn. The full message history + provider/model
//! snapshot is preserved so /resume can reload it verbatim into a new
//! Agent.
//!
//! Workspace scoping: sessions are bucketed by a hash of the workspace
//! path so /resume in project A doesn't surface project B's chats.
//!
//! Pruning: when a brand-new session writes its first file, we trim
//! the per-workspace bucket down to MAX_SESSIONS by deleting the
//! oldest. Resumed sessions don't trigger pruning (they overwrite an
//! existing file).
//!
//! No new dependencies: session ids are nanosecond timestamps in hex,
//! workspace hashes are std DefaultHasher. Good enough for an
//! interactive REPL where collisions would require sub-nanosecond
//! repeated startup, which can't happen.

use crate::ollama::ChatMessage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_SESSIONS: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub provider: String,
    pub model: String,
    pub workspace: String,
    #[serde(rename = "crate")]
    pub crate_name: String,
    pub messages: Vec<ChatMessage>,
}

/// Display row for the picker. We read all session files on /resume
/// since <30 sessions is fast enough that an index file isn't worth
/// the extra complexity.
pub struct SessionListing {
    pub path: PathBuf,
    pub updated_at_unix: u64,
    pub provider: String,
    pub model: String,
    pub message_count: usize,
    pub title: String,
}

pub fn new_session_id() -> String {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{ns:x}")
}

fn workspace_hash(workspace: &Path) -> String {
    let mut h = DefaultHasher::new();
    workspace.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn workspace_dir(workspace: &Path) -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| anyhow::anyhow!("$HOME unset"))?;
    let dir = PathBuf::from(home)
        .join(".plane-code")
        .join("sessions")
        .join(workspace_hash(workspace));
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

fn session_path(workspace: &Path, id: &str) -> Result<PathBuf> {
    Ok(workspace_dir(workspace)?.join(format!("{id}.json")))
}

/// Persist a session to disk. Preserves the original `created_at`
/// across updates. Triggers a prune sweep when the file is brand new
/// (i.e. first save of a fresh session) so the bucket stays bounded.
pub fn save(
    workspace: &Path,
    id: &str,
    provider: &str,
    model: &str,
    crate_name: &str,
    messages: &[ChatMessage],
) -> Result<PathBuf> {
    let path = session_path(workspace, id)?;
    let is_new = !path.exists();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let created_at = if is_new {
        now
    } else {
        match fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<Session>(&s).ok())
        {
            Some(prev) => prev.created_at_unix,
            None => now,
        }
    };
    let session = Session {
        id: id.to_string(),
        created_at_unix: created_at,
        updated_at_unix: now,
        provider: provider.to_string(),
        model: model.to_string(),
        workspace: workspace.display().to_string(),
        crate_name: crate_name.to_string(),
        messages: messages.to_vec(),
    };
    let json = serde_json::to_string_pretty(&session).context("serialize session")?;
    fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;

    if is_new {
        // Best-effort: prune failure shouldn't fail the save.
        let _ = prune(workspace);
    }
    Ok(path)
}

pub fn load(path: &Path) -> Result<Session> {
    let s = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let session: Session =
        serde_json::from_str(&s).with_context(|| format!("parse {}", path.display()))?;
    Ok(session)
}

/// All sessions for this workspace, newest first by `updated_at`.
/// Skips files that fail to parse rather than erroring (a stray
/// corrupt file shouldn't break /resume).
pub fn list(workspace: &Path) -> Result<Vec<SessionListing>> {
    let dir = workspace_dir(workspace)?;
    let mut entries: Vec<SessionListing> = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let s = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let session: Session = match serde_json::from_str(&s) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let title = first_user_title(&session.messages);
        entries.push(SessionListing {
            path,
            updated_at_unix: session.updated_at_unix,
            provider: session.provider,
            model: session.model,
            message_count: session.messages.len(),
            title,
        });
    }
    entries.sort_by(|a, b| b.updated_at_unix.cmp(&a.updated_at_unix));
    Ok(entries)
}

fn first_user_title(messages: &[ChatMessage]) -> String {
    let raw = messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("(no user message)");
    let cleaned: String = raw
        .chars()
        .map(|c| if c == '\n' || c == '\r' || c == '\t' { ' ' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    let max_chars = 60;
    if trimmed.chars().count() > max_chars {
        let mut s: String = trimmed.chars().take(max_chars - 3).collect();
        s.push_str("...");
        s
    } else {
        trimmed.to_string()
    }
}

/// Trim the workspace bucket to MAX_SESSIONS, deleting the oldest by
/// `updated_at`. Returns count actually removed.
fn prune(workspace: &Path) -> Result<usize> {
    let sessions = list(workspace)?;
    if sessions.len() <= MAX_SESSIONS {
        return Ok(0);
    }
    let mut count = 0;
    for entry in sessions.into_iter().skip(MAX_SESSIONS) {
        if fs::remove_file(&entry.path).is_ok() {
            count += 1;
        }
    }
    Ok(count)
}

/// Human-readable "Xs ago" / "Xm ago" / "Xh ago" / "Xd ago".
pub fn relative_time(unix_secs: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(unix_secs);
    let diff = now.saturating_sub(unix_secs);
    if diff < 60 {
        format!("{diff}s ago")
    } else if diff < 3600 {
        format!("{}m ago", diff / 60)
    } else if diff < 86400 {
        format!("{}h ago", diff / 3600)
    } else {
        format!("{}d ago", diff / 86400)
    }
}
