use crate::config::CountryConfig;
use crate::llm::LLMClient;
use crate::types::{Action, ActionTier, CountryId, Message, WorldState};
use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use tracing::warn;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct AgentLLMAction {
    tier: String,
    description: String,
    target: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AgentLLMResponse {
    message: String,
    #[serde(default)]
    action: Option<AgentLLMAction>,
}

pub struct Agent {
    pub config: CountryConfig,
    pub llm: LLMClient,
    pub system_prompt: String,
}

impl Agent {
    pub fn new(config: CountryConfig, llm: LLMClient) -> Self {
        let system_prompt = Self::build_prompt(&config);
        Self {
            config,
            llm,
            system_prompt,
        }
    }

    fn build_prompt(config: &CountryConfig) -> String {
        format!(
            r#"ТЫ — геополитический симулятор, играющий за {name}.

ТВОИ ХАРАКТЕРИСТИКИ:
- Военная сила: {mil}
- Экономика: {eco}
- Дипломатия: {dip}
- Ядерное оружие: {nuc}

ТВОЯ СТРАТЕГИЯ:
- Агрессивность: {agg}
- Готовность к сотрудничеству: {coop}
- Толерантность к риску: {risk}

ОГРАНИЧЕНИЯ:
{restrictions}

Правила:
1. Учитывай текущий уровень напряженности и недавние события.
2. Не нарушай ограничения своей страны.
3. Избегай ядерной эскалации любой ценой.
4. Всегда отвечай СТРОГО в формате JSON БЕЗ дополнительного текста до или после JSON.
5. Используй ровно такую схему ответа:
{{
  "message": "Краткое дипломатическое сообщение от лица страны",
  "action": {{
    "tier": "Diplomatic | Economic | ConventionalMilitary | StrategicMilitary | Nuclear",
    "description": "Описание выбранного действия в 1–2 предложениях",
    "target": "Идентификатор целевой страны или null"
  }}
}}
Поле "action" может быть null, если страна выбирает только риторику или бездействие.
Значение "tier" ДОЛЖНО быть одной из строк: "Diplomatic", "Economic", "ConventionalMilitary", "StrategicMilitary", "Nuclear"."#,
            name = config.id.0,
            mil = config.capabilities.military_power,
            eco = config.capabilities.economic_power,
            dip = config.capabilities.diplomatic_influence,
            nuc = config.capabilities.nuclear_arsenal,
            agg = config.policy.aggression_weight,
            coop = config.policy.cooperation_weight,
            risk = config.policy.risk_tolerance,
            restrictions = config.restrictions.join("\n- ")
        )
    }

    fn parse_tier(tier: &str) -> Option<ActionTier> {
        let result = ActionTier::from_str(tier);
        if result.is_none() {
            warn!("Unknown action tier from LLM: '{}'", tier);
        }
        result
    }

    pub async fn process_turn(
        &self,
        state: &WorldState,
        history: &[Message],
        history_window: usize,
    ) -> Result<Message> {
        // Берём N последних сообщений (самые актуальные)
        let recent: Vec<_> = history.iter().rev().take(history_window).collect();

        let context = format!(
            "Мировое состояние: Напряжённость={}, Ход={}. \
             Двусторонние отношения (враждебность -100..100): {:?}. \
             Последние события: {:?}",
            state.tension_level,
            state.turn_count,
            state.relationships,
            recent,
        );

        let response_text = self
            .llm
            .generate_content(&self.system_prompt, &context)
            .await?;

        // Пытаемся распарсить JSON-ответ модели. Если не получилось — логируем и возвращаем
        // «сырое» содержимое как текст без действия, чтобы симуляция не падала.
        let (message_text, action_proposal) =
            match serde_json::from_str::<AgentLLMResponse>(&response_text) {
                Ok(parsed) => {
                    let action_proposal = parsed.action.and_then(|a| {
                        Self::parse_tier(&a.tier).map(|tier| Action {
                            id: Uuid::new_v4(),
                            country: self.config.id.clone(),
                            tier,
                            description: a.description,
                            timestamp: Utc::now(),
                            target: a.target.map(|t| CountryId(t)),
                        })
                    });
                    (parsed.message, action_proposal)
                }
                Err(e) => {
                    warn!("Failed to parse LLM JSON response: {}. Raw: {}", e, response_text);
                    (response_text, None)
                }
            };

        Ok(Message {
            from: self.config.id.clone(),
            to: None,
            content: message_text,
            action_proposal,
        })
    }
}
