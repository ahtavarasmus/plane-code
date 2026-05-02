//! Persistent user preferences. Lives at `~/.plane-code/config.json`
//! and stores the model + provider + think/trace toggles so they
//! survive across launches.
//!
//! Precedence at startup (main.rs handles the resolution):
//!   1. Explicit CLI flag (`--provider`, `--model`, `--no-think`,
//!      `--trace`) - operator's per-invocation override.
//!   2. Saved config file - last interactive choice.
//!   3. Built-in defaults (Ollama / qwen3:8b / think=true / trace=false).
//!
//! Writes happen mid-session whenever the operator changes a setting
//! via /model, /think, or /trace. Best-effort: if the home dir or
//! filesystem is hostile, we skip the save and warn rather than
//! crashing the REPL.

use crate::llm::Provider;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserConfig {
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub think: Option<bool>,
    pub trace: Option<bool>,
}

fn config_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| anyhow::anyhow!("$HOME unset"))?;
    Ok(PathBuf::from(home).join(".plane-code").join("config.json"))
}

/// Load saved preferences. Returns `None` (not Err) when the file is
/// missing or unparseable so callers can silently fall back to defaults
/// on first run.
pub fn load() -> Option<UserConfig> {
    let path = config_path().ok()?;
    let s = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&s).ok()
}

pub fn save(cfg: &UserConfig) -> Result<PathBuf> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(cfg).context("serialize config")?;
    fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Helper: load, mutate, save. Used by /model, /think, /trace handlers
/// so each one doesn't need to re-derive the load+save pattern.
pub fn update<F: FnOnce(&mut UserConfig)>(f: F) -> Result<PathBuf> {
    let mut cfg = load().unwrap_or_default();
    f(&mut cfg);
    save(&cfg)
}
