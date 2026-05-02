//! Backend dispatch. The agent loop talks to one `LlmBackend`; the enum
//! decides at the call site whether to hit the local Ollama daemon or
//! the hosted Groq API. Ollama is the canonical shape - Groq translates
//! its OpenAI-shape responses into Ollama's `ChatMessage` /
//! `ChatResponse` / `ToolCall` types before returning, so the agent
//! and display code never has to branch on backend.
//!
//! Static dispatch (enum) over trait objects: there are exactly two
//! variants, the closure passed to `chat_stream` is generic on `FnMut`,
//! and we want the REPL's `agent.llm.set_think(true)` style mutations
//! without async-trait boxing.

use crate::groq::GroqClient;
use crate::ollama::{ChatMessage, ChatResponse, DeltaKind, OllamaClient};
use anyhow::{anyhow, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Ollama,
    Groq,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Ollama => "ollama",
            Provider::Groq => "groq",
        }
    }
}

/// Static-at-startup configuration shared by both backends. Lives on
/// the agent so the REPL's /model command can rebuild a backend
/// in-place without re-running `main`. The api_key field captures the
/// `--api-key` CLI value (when given); switching to Groq later falls
/// back to the GROQ_API_KEY env var if this is None, so the user can
/// rely on either source.
pub struct BackendConfig {
    pub ollama_host: String,
    pub num_ctx: u32,
    pub groq_host: String,
    pub api_key: Option<String>,
}

impl BackendConfig {
    /// Build a fresh backend for a (provider, model) pair. think/trace
    /// are passed in by the caller (typically copied from the previous
    /// backend's state) so toggling those mid-session doesn't reset
    /// when the user switches model.
    pub fn build(
        &self,
        provider: Provider,
        model: String,
        think: bool,
        trace: bool,
    ) -> Result<LlmBackend> {
        match provider {
            Provider::Ollama => Ok(LlmBackend::Ollama(OllamaClient::new(
                self.ollama_host.clone(),
                model,
                self.num_ctx,
                think,
                trace,
            ))),
            Provider::Groq => {
                // Trim whitespace defensively. Common gotcha: keys
                // pulled from a file or pasted into shell rc files
                // pick up a trailing newline, which makes the header
                // "Bearer gsk_...\n" and Groq returns 401 with no
                // hint that whitespace was the issue.
                let api_key = self
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("GROQ_API_KEY").ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        anyhow!(
                            "groq backend selected but no API key found. Pass \
                             --api-key=<key> or set GROQ_API_KEY in the environment."
                        )
                    })?;
                Ok(LlmBackend::Groq(GroqClient::new(
                    self.groq_host.clone(),
                    model,
                    api_key,
                    think,
                    trace,
                )))
            }
        }
    }
}

pub enum LlmBackend {
    Ollama(OllamaClient),
    Groq(GroqClient),
}

impl LlmBackend {
    pub async fn warm(&self) -> Result<()> {
        match self {
            LlmBackend::Ollama(c) => c.warm().await,
            LlmBackend::Groq(c) => c.warm().await,
        }
    }

    pub async fn chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        on_delta: F,
    ) -> Result<ChatResponse>
    where
        F: FnMut(DeltaKind, &str),
    {
        match self {
            LlmBackend::Ollama(c) => c.chat_stream(messages, tools, on_delta).await,
            LlmBackend::Groq(c) => c.chat_stream(messages, tools, on_delta).await,
        }
    }

    pub fn set_think(&mut self, v: bool) {
        match self {
            LlmBackend::Ollama(c) => c.think = v,
            LlmBackend::Groq(c) => c.think = v,
        }
    }

    pub fn set_trace(&mut self, v: bool) {
        match self {
            LlmBackend::Ollama(c) => c.trace = v,
            LlmBackend::Groq(c) => c.trace = v,
        }
    }

    pub fn think(&self) -> bool {
        match self {
            LlmBackend::Ollama(c) => c.think,
            LlmBackend::Groq(c) => c.think,
        }
    }

    pub fn trace(&self) -> bool {
        match self {
            LlmBackend::Ollama(c) => c.trace,
            LlmBackend::Groq(c) => c.trace,
        }
    }

    /// Human-readable provider name for diagnostic output.
    pub fn provider(&self) -> &'static str {
        match self {
            LlmBackend::Ollama(_) => "ollama",
            LlmBackend::Groq(_) => "groq",
        }
    }

    pub fn model(&self) -> &str {
        match self {
            LlmBackend::Ollama(c) => &c.model,
            LlmBackend::Groq(c) => &c.model,
        }
    }
}
