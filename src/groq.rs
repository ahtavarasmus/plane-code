//! Groq HTTP client. Speaks the OpenAI-compatible REST API at
//! `/openai/v1/chat/completions` and translates everything to the
//! ollama-shape `ChatMessage` / `ChatResponse` / `ToolCall` types so the
//! rest of the codebase doesn't need to know which backend is wired up.
//!
//! Translation rules (OpenAI -> ollama-shape):
//!   - assistant `delta.content`   -> DeltaKind::Content + ChatMessage.content
//!   - assistant `delta.reasoning` -> DeltaKind::Thinking + ChatMessage.thinking
//!     (Groq emits this for reasoning models like gpt-oss, qwen3-32b,
//!     deepseek-r1 when reasoning_format=parsed.)
//!   - `delta.tool_calls[]`        -> accumulated by index, arguments
//!     concatenated as a JSON string and parsed once finish_reason fires.
//!     Result becomes ToolCall.function.arguments (Value), matching how
//!     Ollama already returns tool calls.
//!   - usage.prompt_tokens / completion_tokens -> ChatResponse
//!     prompt_eval_count / eval_count. Duration fields stay None since
//!     Groq doesn't report nanoseconds the way Ollama does.
//!
//! Reverse direction (request side):
//!   - Ollama-shape `tool_calls` on assistant messages get re-serialized
//!     with arguments as a JSON string (OpenAI requires string, Ollama
//!     accepts object - we always stored object).
//!   - role=tool messages get rewritten to OpenAI's
//!     `{role, tool_call_id, content}` shape.

use crate::ollama::{ChatMessage, ChatResponse, DeltaKind, ToolCall, ToolCallFunction};
use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct GroqClient {
    pub host: String,
    pub model: String,
    pub api_key: String,
    /// Mostly inert (Groq doesn't have a `think: true` switch the way
    /// Ollama does), but kept on the struct so `/think` and `/trace`
    /// REPL toggles work uniformly across backends. When true, sets
    /// `reasoning_format: parsed` so reasoning-capable models emit their
    /// chain-of-thought in a dedicated `delta.reasoning` field instead
    /// of inline `<think>` tags. Models that don't expose reasoning
    /// ignore the parameter.
    pub think: bool,
    pub trace: bool,
    pub http: reqwest::Client,
}

impl GroqClient {
    pub fn new(host: String, model: String, api_key: String, think: bool, trace: bool) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .expect("reqwest client");
        Self {
            host,
            model,
            api_key,
            think,
            trace,
            http,
        }
    }

    /// No-op. Groq is hosted; there's nothing to load. Returns Ok so the
    /// REPL's `warm` ceremony runs without surfacing a fake error.
    pub async fn warm(&self) -> Result<()> {
        Ok(())
    }

    /// Streaming chat. Mirrors `OllamaClient::chat_stream` semantics: the
    /// callback fires once per non-empty content/thinking fragment, and
    /// the returned `ChatResponse` carries accumulated content + thinking
    /// + tool_calls. Single-shot - no Ollama-style retry layer here
    /// because Groq's HTTP path is more stable and the streaming format
    /// is line-deterministic; if it fails, we surface the error.
    pub async fn chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        mut on_delta: F,
    ) -> Result<ChatResponse>
    where
        F: FnMut(DeltaKind, &str),
    {
        let oa_messages = translate_messages_to_openai(messages);

        let mut body = json!({
            "model": self.model,
            "messages": oa_messages,
            "stream": true,
            "temperature": 0.2,
            "stream_options": { "include_usage": true },
        });
        if !tools.is_empty() {
            // Tool definitions in `tools.rs` are already OpenAI-shape
            // (`{type:"function", function:{name, description, parameters}}`)
            // - Ollama happens to accept the same shape. Pass through.
            body["tools"] = Value::Array(tools.to_vec());
        }
        if self.think {
            body["reasoning_format"] = json!("parsed");
        }

        let url = format!(
            "{}/openai/v1/chat/completions",
            self.host.trim_end_matches('/')
        );
        let mut resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("groq request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("groq returned {status}: {text}"));
        }

        let mut buffer: Vec<u8> = Vec::new();
        let mut accumulated_content = String::new();
        let mut accumulated_thinking = String::new();
        let mut tc_acc: Vec<ToolCallAcc> = Vec::new();
        let mut usage: Option<UsageInfo> = None;
        let mut saw_done = false;

        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| anyhow!("groq stream chunk error: {e}"))?
        {
            buffer.extend_from_slice(&chunk);
            // SSE events are separated by a blank line ("\n\n"). Drain
            // the buffer one event at a time; partial events stay parked.
            while let Some(end) = find_event_boundary(&buffer) {
                let event_bytes: Vec<u8> = buffer.drain(..end).collect();
                let event_str = std::str::from_utf8(&event_bytes).unwrap_or("");
                for line in event_str.lines() {
                    let line = line.trim_end_matches('\r');
                    if !line.starts_with("data:") {
                        continue;
                    }
                    let payload = line[5..].trim();
                    if payload.is_empty() {
                        continue;
                    }
                    if payload == "[DONE]" {
                        saw_done = true;
                        continue;
                    }
                    if self.trace {
                        eprintln!("[groq-chunk] {payload}");
                    }
                    let parsed: GroqStreamChunk = match serde_json::from_str(payload) {
                        Ok(v) => v,
                        Err(e) => {
                            return Err(anyhow!("groq chunk parse: {e}; payload={payload}"));
                        }
                    };
                    if let Some(u) = parsed.usage {
                        usage = Some(u);
                    }
                    for choice in parsed.choices {
                        let delta = choice.delta;
                        if let Some(content) = delta.content.as_deref() {
                            if !content.is_empty() {
                                on_delta(DeltaKind::Content, content);
                                accumulated_content.push_str(content);
                            }
                        }
                        if let Some(reasoning) = delta.reasoning.as_deref() {
                            if !reasoning.is_empty() {
                                on_delta(DeltaKind::Thinking, reasoning);
                                accumulated_thinking.push_str(reasoning);
                            }
                        }
                        if let Some(tcs) = delta.tool_calls {
                            for tcd in tcs {
                                let idx = tcd.index.unwrap_or(0);
                                while tc_acc.len() <= idx {
                                    tc_acc.push(ToolCallAcc::default());
                                }
                                let slot = &mut tc_acc[idx];
                                if let Some(id) = tcd.id {
                                    slot.id = Some(id);
                                }
                                if let Some(f) = tcd.function {
                                    if let Some(n) = f.name {
                                        if !n.is_empty() {
                                            slot.name = n;
                                        }
                                    }
                                    if let Some(args) = f.arguments {
                                        slot.arguments_str.push_str(&args);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !saw_done && self.trace {
            eprintln!("[groq] stream ended without [DONE] sentinel");
        }

        let msg = build_final_message(accumulated_content, accumulated_thinking, tc_acc);
        Ok(ChatResponse {
            message: msg,
            done: true,
            total_duration: None,
            prompt_eval_count: usage.as_ref().map(|u| u.prompt_tokens),
            prompt_eval_duration: None,
            eval_count: usage.as_ref().map(|u| u.completion_tokens),
            eval_duration: None,
        })
    }
}

#[derive(Default)]
struct ToolCallAcc {
    id: Option<String>,
    name: String,
    arguments_str: String,
}

fn build_final_message(
    content: String,
    thinking: String,
    tcs: Vec<ToolCallAcc>,
) -> ChatMessage {
    let tool_calls: Vec<ToolCall> = tcs
        .into_iter()
        .filter(|acc| !acc.name.is_empty())
        .map(|acc| {
            // Arguments arrive as a JSON-stringified blob. Parse to
            // restore the object shape that `ToolCallFunction.arguments`
            // (a `serde_json::Value`) expects - this matches what the
            // Ollama backend already returns. Empty string -> empty
            // object so downstream `serde_json::from_value::<Request>`
            // doesn't choke. Unparseable string -> wrap as Value::String
            // so the raw payload at least survives to display.
            let arguments: Value = if acc.arguments_str.trim().is_empty() {
                Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&acc.arguments_str)
                    .unwrap_or_else(|_| Value::String(acc.arguments_str.clone()))
            };
            ToolCall {
                id: acc.id,
                function: ToolCallFunction {
                    name: acc.name,
                    arguments,
                },
            }
        })
        .collect();
    ChatMessage {
        role: "assistant".into(),
        content,
        thinking: if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        },
        tool_calls,
        tool_name: None,
        tool_call_id: None,
    }
}

/// SSE events end with a blank line. Find the position of the trailing
/// "\n\n" (or "\r\n\r\n") and return the byte offset *past* that
/// terminator so a drain at that index pulls one whole event out of the
/// buffer.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(p + 4);
    }
    if let Some(p) = buf.windows(2).position(|w| w == b"\n\n") {
        return Some(p + 2);
    }
    None
}

fn translate_messages_to_openai(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role.as_str() {
            "tool" => {
                // OpenAI: { role: "tool", tool_call_id, content }.
                // We always have a tool_call_id when the assistant
                // message that triggered this tool reply also came from
                // an OpenAI-shape backend (Groq sets call.id). For
                // assistants that came from Ollama mid-stream and got
                // replayed against Groq, fall back to a synthetic id
                // derived from the tool name so the request still
                // round-trips. Such mixed sessions aren't a supported
                // mode but we'd rather degrade than 400.
                let tool_call_id = m
                    .tool_call_id
                    .clone()
                    .or_else(|| m.tool_name.as_ref().map(|n| format!("call_{n}")))
                    .unwrap_or_else(|| "call_unknown".into());
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": m.content,
                }));
            }
            "assistant" => {
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), json!("assistant"));
                // OpenAI accepts content as null when only tool_calls
                // are present. Empty string is also accepted.
                if m.content.is_empty() {
                    obj.insert("content".into(), Value::Null);
                } else {
                    obj.insert("content".into(), json!(m.content));
                }
                if !m.tool_calls.is_empty() {
                    let tcs: Vec<Value> = m
                        .tool_calls
                        .iter()
                        .enumerate()
                        .map(|(i, tc)| {
                            let id = tc
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("call_{i}_{}", tc.function.name));
                            // OpenAI requires arguments to be a STRING.
                            let args_str = serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_else(|_| "{}".into());
                            json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": args_str,
                                }
                            })
                        })
                        .collect();
                    obj.insert("tool_calls".into(), Value::Array(tcs));
                }
                out.push(Value::Object(obj));
            }
            _ => {
                // user, system: pass content through unchanged.
                out.push(json!({
                    "role": m.role,
                    "content": m.content,
                }));
            }
        }
    }
    out
}

#[derive(Debug, Deserialize)]
struct GroqStreamChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    // finish_reason is informational; we don't gate on it because the
    // [DONE] sentinel + EOF already tell us the stream is over.
    #[serde(default)]
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning models (gpt-oss, qwen3-32b reasoning, deepseek-r1)
    /// expose chain-of-thought here when `reasoning_format=parsed`.
    /// Other models leave this absent.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<DeltaToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct DeltaToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}
