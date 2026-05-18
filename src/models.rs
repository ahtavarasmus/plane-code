//! Dynamic model discovery for the `/model` picker. Both providers
//! expose lists at runtime so we don't have to hardcode names that
//! get stale fast.
//!
//! Endpoints:
//!   - Groq:   GET {groq_host}/openai/v1/models      (Bearer auth)
//!   - Ollama: GET {ollama_host}/api/tags            (locally downloaded)
//!   - Ollama: GET https://ollama.com/library        (public registry,
//!             HTML; we scan for `href="/library/<slug>"`)
//!   - Ollama: POST {ollama_host}/api/pull (streaming NDJSON progress)
//!
//! All listers return `Result<Vec<ModelInfo>>`. The picker degrades
//! gracefully when one source fails: e.g. a failed library scrape
//! still leaves downloaded Ollama models pickable; a failed Groq fetch
//! surfaces the error with an API-key hint.

use crate::llm::Provider;
use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub provider: Provider,
    pub id: String,
    pub state: ModelState,
    /// Bytes on disk. Populated for downloaded Ollama models;
    /// otherwise None.
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// Present locally (Ollama only).
    Downloaded,
    /// Listed in the Ollama public registry but not pulled.
    AvailableRemote,
    /// Hosted by Groq; no local "downloaded" concept.
    Hosted,
}

pub async fn list_groq(host: &str, api_key: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/openai/v1/models", host.trim_end_matches('/'));
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| anyhow!("build http client: {e}"))?;
    let resp = http
        .get(&url)
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(|e| anyhow!("groq /v1/models request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("groq /v1/models returned {status}: {text}"));
    }
    let parsed: GroqModelsResponse = resp
        .json()
        .await
        .map_err(|e| anyhow!("groq /v1/models parse: {e}"))?;
    let mut out: Vec<ModelInfo> = parsed
        .data
        .into_iter()
        .filter(|m| m.active.unwrap_or(true))
        .map(|m| ModelInfo {
            provider: Provider::Groq,
            id: m.id,
            state: ModelState::Hosted,
            size_bytes: None,
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

#[derive(Deserialize)]
struct GroqModelsResponse {
    data: Vec<GroqModel>,
}

#[derive(Deserialize)]
struct GroqModel {
    id: String,
    #[serde(default)]
    active: Option<bool>,
}

pub async fn list_ollama_local(host: &str) -> Result<Vec<ModelInfo>> {
    let url = format!("{}/api/tags", host.trim_end_matches('/'));
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| anyhow!("build http client: {e}"))?;
    let resp = http
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("ollama /api/tags request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("ollama /api/tags returned {status}: {text}"));
    }
    let parsed: OllamaTagsResponse = resp
        .json()
        .await
        .map_err(|e| anyhow!("ollama /api/tags parse: {e}"))?;
    let mut out: Vec<ModelInfo> = parsed
        .models
        .into_iter()
        .map(|m| ModelInfo {
            provider: Provider::Ollama,
            id: m.name,
            state: ModelState::Downloaded,
            size_bytes: Some(m.size),
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaLocalModel>,
}

#[derive(Deserialize)]
struct OllamaLocalModel {
    name: String,
    #[serde(default)]
    size: u64,
}

/// Scrape ollama.com/library for the public registry. Returns base
/// model slugs (no `:tag`), e.g. "qwen3", "llama3.2". The caller pairs
/// these with the local tags list to label downloaded vs available.
///
/// Best-effort: if the HTML format changes and the regex misses, we
/// return an empty list and the picker still works off `/api/tags`.
pub async fn list_ollama_remote() -> Result<Vec<ModelInfo>> {
    let url = "https://ollama.com/library";
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| anyhow!("build http client: {e}"))?;
    let resp = http
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow!("ollama.com/library fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("ollama.com/library returned {}", resp.status()));
    }
    let body = resp
        .text()
        .await
        .map_err(|e| anyhow!("ollama.com/library body: {e}"))?;
    Ok(extract_library_slugs(&body))
}

/// Scan an Ollama library HTML body for `href="/library/<slug>"` and
/// return one `ModelInfo` per unique slug.
///
/// Pure function so it stays unit-testable without network access.
/// Slug validation matches Ollama's URL grammar: lowercase ASCII,
/// digits, and `.`/`_`/`-`. Anything that doesn't fit (e.g. nested
/// paths like `/library/foo/bar`) is skipped.
fn extract_library_slugs(body: &str) -> Vec<ModelInfo> {
    const PREFIX: &str = "href=\"/library/";
    let mut out: Vec<ModelInfo> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut search_from = 0usize;
    while let Some(rel) = body[search_from..].find(PREFIX) {
        let start = search_from + rel + PREFIX.len();
        let rest = &body[start..];
        let end = rest.find('"').unwrap_or(rest.len());
        let slug = &rest[..end];
        search_from = start + end;
        if slug.is_empty() {
            continue;
        }
        let valid = slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-');
        if !valid {
            continue;
        }
        if seen.insert(slug.to_string()) {
            out.push(ModelInfo {
                provider: Provider::Ollama,
                id: slug.to_string(),
                state: ModelState::AvailableRemote,
                size_bytes: None,
            });
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// One tag (variant) of an Ollama model, with the size advertised on
/// ollama.com. `size_bytes` is None when we couldn't find a size near
/// the tag anchor (rare, but the picker should still let the user
/// choose).
#[derive(Debug, Clone)]
pub struct OllamaTag {
    pub tag: String,
    pub size_bytes: Option<u64>,
}

/// Fetch the per-tag list for a single Ollama model by scraping its
/// detail page. Used when the user picks an `AvailableRemote` slug in
/// the picker - they then pick a specific tag/size before pulling.
pub async fn list_ollama_tags(slug: &str) -> Result<Vec<OllamaTag>> {
    let url = format!("https://ollama.com/library/{slug}");
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| anyhow!("build http client: {e}"))?;
    let resp = http
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow!("ollama.com/library/{slug} fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!(
            "ollama.com/library/{slug} returned {}",
            resp.status()
        ));
    }
    let body = resp
        .text()
        .await
        .map_err(|e| anyhow!("ollama.com/library/{slug} body: {e}"))?;
    Ok(extract_model_tags(slug, &body))
}

/// Pure extraction: pull `<slug>:<tag>` anchors out of the detail
/// page HTML, then scan each anchor's body for a size like "5.2GB"
/// or "523MB". Split out from the network call so it's unit-testable.
fn extract_model_tags(slug: &str, body: &str) -> Vec<OllamaTag> {
    let prefix = format!("href=\"/library/{slug}:");
    let mut out: Vec<OllamaTag> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut search_from = 0usize;
    while let Some(rel) = body[search_from..].find(&prefix) {
        let tag_start = search_from + rel + prefix.len();
        let rest = &body[tag_start..];
        let tag_end = rest.find('"').unwrap_or(rest.len());
        let tag = &rest[..tag_end];
        search_from = tag_start + tag_end;
        if tag.is_empty() {
            continue;
        }
        let valid_tag = tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-');
        if !valid_tag {
            continue;
        }
        if !seen.insert(tag.to_string()) {
            continue;
        }
        // Anchor body: scan from the href forward to the next `</a>`.
        // Size text lives somewhere in that span.
        let anchor_body_end = body[tag_start..]
            .find("</a>")
            .map(|p| tag_start + p)
            .unwrap_or(body.len().min(tag_start + 2000));
        let anchor_body = &body[tag_start..anchor_body_end];
        let size_bytes = find_size_in_text(anchor_body);
        out.push(OllamaTag {
            tag: tag.to_string(),
            size_bytes,
        });
    }
    // Sort by size ascending so the smallest variant lands at the top.
    // Tags without size fall last.
    out.sort_by(|a, b| match (a.size_bytes, b.size_bytes) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.tag.cmp(&b.tag),
    });
    out
}

/// Scan a text span for the first size literal like "5.2GB", "523MB",
/// "16KB", "2TB". Returns the value in bytes (using 1024-base for the
/// suffix so 1GB = 1024^3, consistent with `human_size`).
fn find_size_in_text(text: &str) -> Option<u64> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // Capture the numeric literal: digits, optional `.`, more digits.
        let num_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        // Optional whitespace between number and unit.
        let mut j = i;
        while j < bytes.len() && bytes[j] == b' ' {
            j += 1;
        }
        if j >= bytes.len() {
            continue;
        }
        // Match the longest unit suffix that fits.
        let unit_slice = &text[j..text.len().min(j + 2)];
        let (multiplier, unit_len) = match unit_slice {
            "TB" => (1024u64.pow(4), 2),
            "GB" => (1024u64.pow(3), 2),
            "MB" => (1024u64.pow(2), 2),
            "KB" => (1024u64, 2),
            _ => {
                // Single-letter? "B" alone is rarely standalone in this
                // context; skip to avoid false positives on prose like
                // "1B parameters" that aren't byte counts.
                continue;
            }
        };
        // Must be word-bounded on the right so "GBs" or "GBP" don't match.
        let after = j + unit_len;
        if after < bytes.len() {
            let c = bytes[after];
            if c.is_ascii_alphanumeric() {
                continue;
            }
        }
        let num_str = &text[num_start..i];
        if let Ok(num) = num_str.parse::<f64>() {
            return Some((num * multiplier as f64) as u64);
        }
    }
    None
}

/// Pull a model from the Ollama registry. Streams NDJSON progress
/// lines from the daemon; `on_progress` receives a short status string
/// per parsed line so the caller can paint a progress indicator.
pub async fn pull_ollama<F>(host: &str, name: &str, mut on_progress: F) -> Result<()>
where
    F: FnMut(&str),
{
    let url = format!("{}/api/pull", host.trim_end_matches('/'));
    // Long timeout: large models take many minutes on a slow link. The
    // request body sets stream=true so we read incrementally rather
    // than waiting for the whole pull to finish before any response.
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| anyhow!("build http client: {e}"))?;
    let body = serde_json::json!({ "name": name, "stream": true });
    let mut resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow!("ollama /api/pull request failed: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("ollama /api/pull returned {status}: {text}"));
    }
    let mut buffer: Vec<u8> = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| anyhow!("pull chunk error: {e}"))?
    {
        buffer.extend_from_slice(&chunk);
        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buffer.drain(..=pos).collect();
            let s = std::str::from_utf8(&line).unwrap_or("").trim();
            if s.is_empty() {
                continue;
            }
            // Progress lines look like:
            //   {"status":"pulling 4f...","digest":"...","completed":1234,"total":5678}
            //   {"status":"verifying sha256 digest"}
            //   {"status":"success"}
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                let status = v.get("status").and_then(|v| v.as_str()).unwrap_or("");
                let completed = v.get("completed").and_then(|v| v.as_u64());
                let total = v.get("total").and_then(|v| v.as_u64());
                let msg = match (completed, total) {
                    (Some(c), Some(t)) if t > 0 => {
                        let pct = (c as f64 / t as f64 * 100.0).round() as u64;
                        format!("{status}  {pct}%")
                    }
                    _ => status.to_string(),
                };
                if !msg.is_empty() {
                    on_progress(&msg);
                }
                if let Some(err) = v.get("error").and_then(|v| v.as_str()) {
                    return Err(anyhow!("ollama pull error: {err}"));
                }
            }
        }
    }
    Ok(())
}

/// Format a byte count for the picker (1.2G, 850M, etc).
pub fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0}K", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_distinct_library_slugs() {
        let html = r#"
            <a href="/library/qwen3"><h2>qwen3</h2></a>
            <a href="/library/llama3.2"><h2>llama3.2</h2></a>
            <a href="/library/qwen3"><h2>qwen3 again</h2></a>
            <a href="/library/deepseek-r1">...</a>
            <a href="/library/foo/bar">nested, skipped</a>
            <a href="/library/Bad-Slug">uppercase, skipped</a>
            <a href="/library/">empty, skipped</a>
        "#;
        let out = extract_library_slugs(html);
        let ids: Vec<&str> = out.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["deepseek-r1", "llama3.2", "qwen3"]);
        assert!(out.iter().all(|m| m.state == ModelState::AvailableRemote));
        assert!(out.iter().all(|m| m.provider == Provider::Ollama));
    }

    #[test]
    fn human_size_rounds_sensibly() {
        assert_eq!(human_size(512), "512B");
        assert_eq!(human_size(2_048), "2K");
        assert_eq!(human_size(5 * 1024 * 1024), "5M");
        assert_eq!(human_size(2 * 1024 * 1024 * 1024), "2.0G");
    }

    #[test]
    fn finds_sizes_in_anchor_body() {
        assert_eq!(find_size_in_text("model 5.2GB context"), Some((5.2 * 1024_f64.powi(3)) as u64));
        assert_eq!(find_size_in_text("523MB · 40K"), Some(523 * 1024 * 1024));
        assert_eq!(find_size_in_text("16KB total"), Some(16 * 1024));
        assert_eq!(find_size_in_text("2TB pool"), Some(2 * 1024_u64.pow(4)));
        // No false positive on parameter counts written like "8B params".
        assert_eq!(find_size_in_text("8B parameters"), None);
        // No false positive on "GBP" or "MBs".
        assert_eq!(find_size_in_text("100GBP fee"), None);
        // No size at all.
        assert_eq!(find_size_in_text("just text"), None);
    }

    #[test]
    fn extracts_tags_with_sizes_and_sorts_by_size() {
        let html = r#"
            <a href="/library/qwen3:0.6b">qwen3:0.6b 523MB 40K context Text</a>
            <a href="/library/qwen3:8b">qwen3:8b 5.2GB 40K context Text</a>
            <a href="/library/qwen3:30b">qwen3:30b 19GB Text</a>
            <a href="/library/qwen3:235b">qwen3:235b 142GB Text</a>
            <a href="/library/qwen3:latest">qwen3:latest 5.2GB Text</a>
            <a href="/library/qwen3:bad/nested">should skip</a>
        "#;
        let tags = extract_model_tags("qwen3", html);
        let names: Vec<&str> = tags.iter().map(|t| t.tag.as_str()).collect();
        // Sorted by size ascending: 523MB, 5.2GB (twice), 19GB, 142GB.
        // 8b and latest both 5.2GB - relative order between them is
        // not guaranteed by the size comparator, so just check by set.
        assert_eq!(names[0], "0.6b");
        assert!(names.contains(&"8b"));
        assert!(names.contains(&"latest"));
        assert_eq!(names[3], "30b");
        assert_eq!(names[4], "235b");
        assert_eq!(tags.len(), 5);
        assert_eq!(tags[0].size_bytes, Some(523 * 1024 * 1024));
        assert_eq!(tags[4].size_bytes, Some((142.0 * 1024_f64.powi(3)) as u64));
    }
}
