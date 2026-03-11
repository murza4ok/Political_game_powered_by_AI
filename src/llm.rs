use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use crate::config::LLMConfig;
use std::env;
use tokio::time::{sleep, timeout, Duration};
use tracing::warn;

// Yandex Cloud поддерживает OpenAI-совместимый API для сторонних моделей (Qwen и др.).
// Yandex-специфичный gRPC API (foundationModels/v1/completion) работает только для YandexGPT.
// Используем OpenAI-совместимый формат для всех моделей — он работает и для YandexGPT тоже.

#[derive(Debug, Clone)]
pub struct LLMClient {
    client: Client,
    api_key: String,
    config: LLMConfig,
    api_url: String,
}

#[derive(Debug, Serialize, Clone)]
struct ChatRequest {
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
struct ChatResponse {
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
        let api_key = env::var("DEEPSEEK_API_KEY")
            .context("DEEPSEEK_API_KEY not found in .env. Get one at https://platform.deepseek.com/")?;

        let api_url = config
            .api_url
            .clone()
            .unwrap_or_else(|| "https://api.deepseek.com/chat/completions".into());

        Ok(Self {
            client: Client::new(),
            api_key,
            config,
            api_url,
        })
    }

    pub async fn generate_content(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        self.call_api(system_prompt, user_message).await
    }

    /// Отправляет запрос к LLM с повторными попытками при rate limit (429) или таймауте.
    /// Стратегия: до 3 повторных попыток с экспоненциальной задержкой 2s → 4s → 8s.
    async fn call_api(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        const MAX_RETRIES: u32 = 3;
        const REQUEST_TIMEOUT_SECS: u64 = 60;
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
                }
                Ok(Err(e)) if e.to_string().contains("LLM_RATE_LIMIT_429") => {
                    warn!("LLM rate limit hit (attempt {}): {}", attempt, e);
                    last_error = e;
                }
                Ok(Err(e)) => {
                    return Err(e);
                }
                Ok(Ok(text)) => return Ok(text),
            }
        }

        Err(last_error)
    }

    async fn send_request(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        let req_body = ChatRequest {
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
            .context("Failed to send request to DeepSeek API")?;

        let status = response.status();

        if status == StatusCode::TOO_MANY_REQUESTS {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM_RATE_LIMIT_429: {}", error_text);
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API Error {}: {}", status, error_text);
        }

        let resp: ChatResponse = response
            .json()
            .await
            .context("Failed to parse DeepSeek API response")?;

        resp
            .choices
            .and_then(|c| c.first().cloned())
            .and_then(|c| c.message)
            .and_then(|m| m.content)
            .ok_or_else(|| anyhow::anyhow!("Empty response from DeepSeek API"))
    }
}

// ── Yandex AI client ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct YandexLLMClient {
    client: Client,
    api_key: String,
    folder_id: String,
    model: String,
    temperature: f64,
}

impl YandexLLMClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        let api_key = env::var("YANDEX_API_KEY")
            .context("YANDEX_API_KEY not found in .env. Get one at https://console.yandex.cloud/")?;
        let folder_id = env::var("YANDEX_FOLDER_ID")
            .context("YANDEX_FOLDER_ID not found in .env. Find it in Yandex Cloud console.")?;
        // Модель задаётся через YANDEX_MODEL. Варианты:
        //   qwen3-235b-a22b-fp8/latest                — Qwen3 235B (без цензуры на геополитику)
        //   yandexgpt-lite/latest                     — лёгкая YandexGPT (мало фильтров)
        //   yandexgpt/latest                          — полная YandexGPT (строгие фильтры)
        let model = env::var("YANDEX_MODEL")
            .unwrap_or_else(|_| "yandexgpt-lite/latest".to_string());

        Ok(Self {
            client: Client::new(),
            api_key,
            folder_id,
            model,
            temperature: config.temperature as f64,
        })
    }

    pub async fn generate_content(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        self.call_api(system_prompt, user_message).await
    }

    async fn call_api(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        const MAX_RETRIES: u32 = 3;
        const REQUEST_TIMEOUT_SECS: u64 = 120;
        const RETRY_DELAYS: [u64; 3] = [2, 4, 8];

        let mut last_error = anyhow::anyhow!("No attempts made");

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_DELAYS[(attempt as usize - 1).min(RETRY_DELAYS.len() - 1)];
                warn!("Yandex LLM retry attempt {} of {} after {}s delay", attempt, MAX_RETRIES, delay);
                sleep(Duration::from_secs(delay)).await;
            }

            let result = timeout(
                Duration::from_secs(REQUEST_TIMEOUT_SECS),
                self.send_request(system_prompt, user_message),
            )
            .await;

            match result {
                Err(_elapsed) => {
                    warn!("Yandex request timed out after {}s (attempt {})", REQUEST_TIMEOUT_SECS, attempt);
                    last_error = anyhow::anyhow!("Yandex request timed out after {}s", REQUEST_TIMEOUT_SECS);
                }
                Ok(Err(e)) if e.to_string().contains("LLM_RATE_LIMIT_429") => {
                    warn!("Yandex rate limit hit (attempt {}): {}", attempt, e);
                    last_error = e;
                }
                Ok(Err(e)) => {
                    return Err(e);
                }
                Ok(Ok(text)) => return Ok(text),
            }
        }

        Err(last_error)
    }

    async fn send_request(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        // Yandex Cloud OpenAI-совместимый API:
        //   - Работает для всех моделей (Qwen, YandexGPT и др.)
        //   - Эндпоинт: /v1/chat/completions  (НЕ /foundationModels/v1/completion)
        //   - Модель: полный URI gpt://<folder_id>/<model>/<version>
        //   - Формат сообщений: role + content (стандарт OpenAI)
        let model_uri = format!("gpt://{}/{}", self.folder_id, self.model);

        let req_body = ChatRequest {
            model: model_uri,
            messages: vec![
                Message { role: "system".into(), content: system_prompt.into() },
                Message { role: "user".into(),   content: user_message.into() },
            ],
            temperature: self.temperature as f32,
        };

        let response = self
            .client
            .post("https://llm.api.cloud.yandex.net/v1/chat/completions")
            .header("Authorization", format!("Api-Key {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await
            .context("Failed to send request to Yandex AI API")?;

        let status = response.status();

        if status == StatusCode::TOO_MANY_REQUESTS {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM_RATE_LIMIT_429: {}", error_text);
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Yandex API Error {}: {}", status, error_text);
        }

        let resp: ChatResponse = response
            .json()
            .await
            .context("Failed to parse Yandex AI API response")?;

        let text = resp
            .choices
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.message)
            .and_then(|m| m.content)
            .ok_or_else(|| anyhow::anyhow!("Empty response from Yandex AI API"))?;

        // Детектируем отказ модели и логируем явно
        let refusal_phrases = ["не могу обсуждать", "не могу помочь", "не могу ответить",
                               "невозможно обсуждать", "отказываюсь", "не в состоянии"];
        if refusal_phrases.iter().any(|p| text.to_lowercase().contains(p)) {
            warn!("⚠ Yandex отказал (модель: {}). Ответ: {}", self.model, &text[..text.len().min(200)]);
            eprintln!("\n⚠  YANDEX ОТКАЗ [модель: {}]\n   Ответ: {}\n",
                self.model, &text[..text.len().min(300)]);
        }

        Ok(text)
    }
}

// ── Unified LLM client ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum AnyLLMClient {
    DeepSeek(LLMClient),
    Yandex(YandexLLMClient),
}

impl AnyLLMClient {
    pub async fn generate_content(&self, system: &str, user: &str) -> Result<String> {
        match self {
            AnyLLMClient::DeepSeek(c) => c.generate_content(system, user).await,
            AnyLLMClient::Yandex(c)   => c.generate_content(system, user).await,
        }
    }
}
