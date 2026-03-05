use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use crate::config::LLMConfig;
use std::env;
use tokio::time::{sleep, timeout, Duration};
use tracing::warn;

#[derive(Debug, Clone)]
pub struct LLMClient {
    client: Client,
    api_key: String,
    config: LLMConfig,
    api_url: String,
}

#[derive(Debug, Serialize, Clone)]
struct GroqRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Debug, Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Clone)]
struct GroqResponse {
    choices: Option<Vec<Choice>>,
}

#[derive(Debug, Deserialize, Clone)]
struct Choice {
    message: Option<ResponseMessage>,
}

#[derive(Debug, Deserialize, Clone)]
struct ResponseMessage {
    content: Option<String>,
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let api_key = env::var("GROQ_API_KEY")
            .context("GROQ_API_KEY not found in .env. Get one at https://console.groq.com/keys")?;

        let api_url = config
            .api_url
            .clone()
            .unwrap_or_else(|| "https://api.groq.com/openai/v1/chat/completions".into());

        Ok(Self {
            client: Client::new(),
            api_key,
            config,
            api_url,
        })
    }

    pub async fn generate_content(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        self.generate_groq(system_prompt, user_message).await
    }

    /// Отправляет запрос к LLM с повторными попытками при rate limit (429) или таймауте.
    /// Стратегия: до 3 повторных попыток с экспоненциальной задержкой 2s → 4s → 8s.
    async fn generate_groq(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        const MAX_RETRIES: u32 = 3;
        const REQUEST_TIMEOUT_SECS: u64 = 30;
        const RETRY_DELAYS: [u64; 3] = [2, 4, 8];

        let mut last_error = anyhow::anyhow!("No attempts made");

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_DELAYS[(attempt as usize - 1).min(RETRY_DELAYS.len() - 1)];
                warn!("LLM retry attempt {} of {} after {}s delay", attempt, MAX_RETRIES, delay);
                sleep(Duration::from_secs(delay)).await;
            }

            let result = timeout(
                Duration::from_secs(REQUEST_TIMEOUT_SECS),
                self.send_request(system_prompt, user_message),
            )
            .await;

            match result {
                Err(_elapsed) => {
                    warn!("LLM request timed out after {}s (attempt {})", REQUEST_TIMEOUT_SECS, attempt);
                    last_error = anyhow::anyhow!("Request timed out after {}s", REQUEST_TIMEOUT_SECS);
                    // Таймаут — повторяем
                }
                Ok(Err(e)) if e.to_string().contains("LLM_RATE_LIMIT_429") => {
                    warn!("LLM rate limit hit (attempt {}): {}", attempt, e);
                    last_error = e;
                    // Rate limit — повторяем с задержкой
                }
                Ok(Err(e)) => {
                    // Любая другая ошибка — сразу возвращаем без retry
                    return Err(e);
                }
                Ok(Ok(text)) => return Ok(text),
            }
        }

        Err(last_error)
    }

    async fn send_request(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let req_body = GroqRequest {
            model: self.config.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt.into(),
                },
                Message {
                    role: "user".into(),
                    content: user_message.into(),
                },
            ],
            temperature: self.config.temperature,
        };

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await
            .context("Failed to send request to Groq API")?;

        let status = response.status();

        if status == StatusCode::TOO_MANY_REQUESTS {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM_RATE_LIMIT_429: {}", error_text);
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API Error {}: {}", status, error_text);
        }

        let groq_resp: GroqResponse = response
            .json()
            .await
            .context("Failed to parse API response")?;

        groq_resp
            .choices
            .and_then(|c| c.first().cloned())
            .and_then(|c| c.message)
            .and_then(|m| m.content)
            .ok_or_else(|| anyhow::anyhow!("Empty response from API"))
    }
}
