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
    diplomatic_proposal: Option<String>,
    #[serde(default)]
    hidden_action: Option<String>,
    #[serde(default)]
    action: Option<AgentLLMAction>,
}

pub struct Agent {
    pub config: CountryConfig,
    pub llm: LLMClient,
    pub system_prompt: String,
}

impl Agent {
    pub fn new(config: CountryConfig, llm: LLMClient, scenario_description: String) -> Self {
        let system_prompt = Self::build_prompt(&config, &scenario_description);
        Self {
            config,
            llm,
            system_prompt,
        }
    }

    fn build_prompt(config: &CountryConfig, scenario_description: &str) -> String {
        format!(
            r#"ТЫ — геополитический агент, представляющий {name} в симуляции мировой политики.

=== НАЧАЛЬНОЕ СОСТОЯНИЕ МИРА ===
{scenario}

=== ТВОИ ХАРАКТЕРИСТИКИ ===
- Военная сила: {mil}/100
- Экономика: {eco}/100
- Дипломатическое влияние: {dip}/100
- Ядерное оружие: {nuc}

=== ТВОЯ СТРАТЕГИЯ ===
- Агрессивность: {agg:.0}%
- Готовность к сотрудничеству: {coop:.0}%
- Толерантность к риску: {risk:.0}%

=== ОГРАНИЧЕНИЯ ===
- {restrictions}

=== ПРАВИЛА ПОВЕДЕНИЯ ===
1. На каждом ходу ты ВИДИШЬ, что сказали другие страны. Обязательно реагируй на их заявления и действия конкретно — называй их имена, упоминай их шаги.
2. Твой публичный ответ (поле "message") должен содержать 3–5 предложений: реакция на ходы других игроков, оценка ситуации, твоя позиция.
3. В поле "diplomatic_proposal" — конкретное предложение по урегулированию напряжённости или сотрудничеству (1–3 предложения). Может быть null если предложений нет.
4. В поле "hidden_action" — скрытое действие, которое происходит внутри страны или тайно: мобилизация, разведка, тайные переговоры, экономические манипуляции (1–2 предложения). Это не публично — только для симулятора.
5. Не нарушай ограничения своей страны.
6. Учитывай уровень напряжённости: при высоком уровне — осторожнее с военными шагами.
7. Всегда отвечай СТРОГО в формате JSON БЕЗ дополнительного текста до или после JSON.

=== ФОРМАТ ОТВЕТА ===
{{
  "message": "Публичное заявление: реакция на слова и действия других игроков (3–5 предложений, конкретно, с упоминанием имён стран)",
  "diplomatic_proposal": "Конкретное предложение по урегулированию или сотрудничеству, либо null",
  "hidden_action": "Скрытое внутреннее или тайное действие (не публикуется)",
  "action": {{
    "tier": "Diplomatic | Economic | ConventionalMilitary | StrategicMilitary | Nuclear",
    "description": "Описание официального действия (2–3 предложения)",
    "target": "Идентификатор целевой страны или null"
  }}
}}

Поле "action" может быть null, если страна ограничивается риторикой.
Значение "tier" ДОЛЖНО быть одной из строк: "Diplomatic", "Economic", "ConventionalMilitary", "StrategicMilitary", "Nuclear"."#,
            name = config.id.0,
            scenario = scenario_description.trim(),
            mil = config.capabilities.military_power,
            eco = config.capabilities.economic_power,
            dip = config.capabilities.diplomatic_influence,
            nuc = if config.capabilities.nuclear_arsenal { "есть" } else { "нет" },
            agg = config.policy.aggression_weight * 100.0,
            coop = config.policy.cooperation_weight * 100.0,
            risk = config.policy.risk_tolerance * 100.0,
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

    fn format_history(history: &[&Message]) -> String {
        if history.is_empty() {
            return "Нет предыдущих событий.".to_string();
        }
        history
            .iter()
            .map(|msg| {
                let action_info = match &msg.action_proposal {
                    Some(a) => format!(
                        " | Действие: {} [{}]{}",
                        a.tier.as_str(),
                        a.description,
                        a.target
                            .as_ref()
                            .map(|t| format!(" → {}", t.0))
                            .unwrap_or_default()
                    ),
                    None => String::new(),
                };
                let proposal_info = match &msg.diplomatic_proposal {
                    Some(p) => format!(" | Дип.предложение: {}", p),
                    None => String::new(),
                };
                format!(
                    "[{}]: \"{}\"{}{} ",
                    msg.from.0,
                    msg.content,
                    action_info,
                    proposal_info
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub async fn process_turn(
        &self,
        state: &WorldState,
        history: &[Message],
        history_window: usize,
    ) -> Result<Message> {
        let recent: Vec<&Message> = history.iter().rev().take(history_window).collect();

        let relationships_text: Vec<String> = state
            .relationships
            .iter()
            .map(|(pair, val)| format!("{}: {}", pair, val))
            .collect();

        let context = format!(
            "=== ТЕКУЩЕЕ СОСТОЯНИЕ ===\n\
             Ход: {turn} | Глобальная напряжённость: {tension}/100\n\
             Двусторонние отношения (отрицательные = союзники, положительные = враги): {rels}\n\n\
             === ПОСЛЕДНИЕ СОБЫТИЯ (реагируй на них) ===\n\
             {history}",
            turn = state.turn_count,
            tension = state.tension_level,
            rels = if relationships_text.is_empty() {
                "данных нет".to_string()
            } else {
                relationships_text.join(", ")
            },
            history = Self::format_history(&recent),
        );

        let response_text = self
            .llm
            .generate_content(&self.system_prompt, &context)
            .await?;

        let (message_text, diplomatic_proposal, hidden_action, action_proposal) =
            match serde_json::from_str::<AgentLLMResponse>(&response_text) {
                Ok(parsed) => {
                    let action_proposal = parsed.action.and_then(|a| {
                        Self::parse_tier(&a.tier).map(|tier| Action {
                            id: Uuid::new_v4(),
                            country: self.config.id.clone(),
                            tier,
                            description: a.description,
                            timestamp: Utc::now(),
                            target: a.target.map(CountryId),
                        })
                    });
                    (parsed.message, parsed.diplomatic_proposal, parsed.hidden_action, action_proposal)
                }
                Err(e) => {
                    warn!("Failed to parse LLM JSON response: {}. Raw: {}", e, response_text);
                    (response_text, None, None, None)
                }
            };

        Ok(Message {
            from: self.config.id.clone(),
            to: None,
            content: message_text,
            diplomatic_proposal,
            hidden_action,
            action_proposal,
        })
    }
}
