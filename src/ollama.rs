//! Minimal Ollama HTTP client. Talks to /api/chat with tool-calling enabled.
//!
//! Two modes:
//!   - `chat`         non-streaming, returns the whole response when ready.
//!   - `chat_stream`  streams NDJSON chunks; the callback receives a
//!                    `DeltaKind` per fragment so the operator can render
//!                    thinking and content separately. Returns the final
//!                    aggregated message + Ollama timing stats.
//!
//! Ollama's chat endpoint splits a thinking-model's reasoning out of
//! `<think>...</think>` tags into a dedicated `message.thinking` field
//! when `think: true` is set (the default for thinking-capable models).
//! We capture both fields and emit them as distinct deltas; otherwise
//! qwen3 etc. silently consumes its eval budget on reasoning the operator
//! never sees.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub host: String,
    pub model: String,
    pub num_ctx: u32,
    pub think: bool,
    /// When true, every raw stream chunk is dumped to stderr. Use for
    /// debugging weird responses (tokens generated but no visible
    /// content/tool_calls/thinking - means the model emitted something
    /// in a field we're not capturing or in a chunk we ignored).
    pub trace: bool,
    pub http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(host: String, model: String, num_ctx: u32, think: bool, trace: bool) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .expect("reqwest client");
        Self {
            host,
            model,
            num_ctx,
            think,
            trace,
            http,
        }
    }

    fn options(&self) -> Value {
        serde_json::json!({
            "temperature": 0.2,
            "num_ctx": self.num_ctx,
        })
    }

    /// Pre-load the model into memory so the operator's first real prompt
    /// doesn't pay the ~10-15s weight-load + Metal kernel JIT cost. Hits
    /// `/api/generate` with an empty prompt: Ollama treats this as a
    /// load-only request and returns once the model is resident. The
    /// keep_alive value matches the typical session length so the model
    /// stays hot through routine idle periods between prompts.
    pub async fn warm(&self) -> Result<()> {
        let body = serde_json::json!({
            "model": self.model,
            "prompt": "",
            "keep_alive": "30m",
        });
        let url = format!("{}/api/generate", self.host.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("ollama warm request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("ollama warm returned {status}: {text}"));
        }
        // Discard the body; we only care that the load completed.
        let _ = resp.bytes().await;
        Ok(())
    }

    /// Non-streaming chat. Kept as a fallback / debugging helper; the
    /// agent loop uses `chat_stream` so the operator sees output as it
    /// arrives.
    #[allow(dead_code)]
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
    ) -> Result<ChatResponse> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": false,
            "think": self.think,
            "options": self.options(),
        });
        let url = format!("{}/api/chat", self.host.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("ollama request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("ollama returned {status}: {text}"));
        }
        let parsed: ChatResponse = resp
            .json()
            .await
            .map_err(|e| anyhow!("ollama response parse: {e}"))?;
        Ok(parsed)
    }

    /// Streaming chat. `on_delta` is called with `(DeltaKind, &str)` for
    /// every non-empty content or thinking chunk as it arrives. Returns
    /// the final ChatResponse with accumulated content, accumulated
    /// thinking, any tool_calls from the final chunk, and Ollama's
    /// timing stats.
    pub async fn chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        mut on_delta: F,
    ) -> Result<ChatResponse>
    where
        F: FnMut(DeltaKind, &str),
    {
        // Up to two attempts. Ollama occasionally drops the streaming
        // connection mid-generation (memory pressure, model swap, h2
        // RST), surfacing as `error decoding response body`. The agent
        // hasn't acted on anything yet at this point - it's safe to
        // retry from scratch. If we've already streamed visible bytes
        // to the operator, we re-issue silently; the second response
        // is what we return. Drop both attempts and propagate.
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..2u32 {
            match self.chat_stream_once(messages, tools, &mut on_delta).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    let msg = e.to_string();
                    let retriable = msg.contains("decoding response body")
                        || msg.contains("stream chunk error")
                        || msg.contains("connection closed")
                        || msg.contains("error trying to connect");
                    if !retriable || attempt == 1 {
                        return Err(e);
                    }
                    eprintln!(
                        "[ollama] stream dropped ({msg}); retrying once"
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("ollama: no attempts made")))
    }

    async fn chat_stream_once<F>(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        on_delta: &mut F,
    ) -> Result<ChatResponse>
    where
        F: FnMut(DeltaKind, &str),
    {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": true,
            "think": self.think,
            "options": self.options(),
        });
        let url = format!("{}/api/chat", self.host.trim_end_matches('/'));
        let mut resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("ollama request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("ollama returned {status}: {text}"));
        }

        let mut buffer: Vec<u8> = Vec::new();
        let mut accumulated_content = String::new();
        let mut accumulated_thinking = String::new();
        let mut accumulated_tool_calls: Vec<ToolCall> = Vec::new();
        let mut final_message: Option<ChatMessage> = None;
        let mut total_duration: Option<u64> = None;
        let mut prompt_eval_count: Option<u64> = None;
        let mut prompt_eval_duration: Option<u64> = None;
        let mut eval_count: Option<u64> = None;
        let mut eval_duration: Option<u64> = None;

        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| anyhow!("ollama stream chunk error: {e}"))?
        {
            buffer.extend_from_slice(&chunk);
            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
                let line = std::str::from_utf8(&line_bytes).unwrap_or("").trim();
                if line.is_empty() {
                    continue;
                }
                if self.trace {
                    eprintln!("[ollama-chunk] {line}");
                }
                let parsed: ChatStreamChunk = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(anyhow!(
                            "ollama chunk parse: {e}; line={line}"
                        ));
                    }
                };
                if let Some(thinking) = parsed.message.thinking.as_deref() {
                    if !thinking.is_empty() {
                        on_delta(DeltaKind::Thinking, thinking);
                        accumulated_thinking.push_str(thinking);
                    }
                }
                if !parsed.message.content.is_empty() {
                    on_delta(DeltaKind::Content, &parsed.message.content);
                    accumulated_content.push_str(&parsed.message.content);
                }
                // Tool calls can appear in non-final chunks (Ollama emits
                // them as soon as the model finishes the call structure,
                // which may be before done=true). Aggregate across every
                // chunk so we don't lose them.
                if !parsed.message.tool_calls.is_empty() {
                    accumulated_tool_calls.extend(parsed.message.tool_calls.clone());
                }
                if parsed.done {
                    let mut msg = parsed.message;
                    msg.content = accumulated_content.clone();
                    msg.thinking = if accumulated_thinking.is_empty() {
                        None
                    } else {
                        Some(accumulated_thinking.clone())
                    };
                    if !accumulated_tool_calls.is_empty() {
                        msg.tool_calls = accumulated_tool_calls.clone();
                    }
                    final_message = Some(msg);
                    total_duration = parsed.total_duration;
                    prompt_eval_count = parsed.prompt_eval_count;
                    prompt_eval_duration = parsed.prompt_eval_duration;
                    eval_count = parsed.eval_count;
                    eval_duration = parsed.eval_duration;
                }
            }
        }

        let msg = final_message.unwrap_or_else(|| ChatMessage {
            role: "assistant".into(),
            content: accumulated_content,
            thinking: if accumulated_thinking.is_empty() {
                None
            } else {
                Some(accumulated_thinking)
            },
            tool_calls: accumulated_tool_calls,
            tool_name: None,
            tool_call_id: None,
        });
        Ok(ChatResponse {
            message: msg,
            done: true,
            total_duration,
            prompt_eval_count,
            prompt_eval_duration,
            eval_count,
            eval_duration,
        })
    }
}

/// Which slot the streamed text came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaKind {
    /// User-visible response text.
    Content,
    /// Reasoning emitted by a thinking model. Ollama returns this in
    /// `message.thinking` when `think: true`.
    Thinking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    /// Reasoning. Present on assistant messages from thinking-capable
    /// models when `think: true`. Skipped on serialize so we don't echo
    /// it back to Ollama in subsequent turns (the API expects content +
    /// tool_calls only on the input side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// OpenAI-compatible tool-call correlation id. Ignored by Ollama
    /// (which matches tool replies by name) but required by OpenAI-shape
    /// backends (Groq) so they can pair a `role: tool` message with the
    /// originating assistant tool_call. Set on tool-role messages and on
    /// the ToolCall.id when the assistant first emits the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatStreamChunk {
    message: ChatMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
}
