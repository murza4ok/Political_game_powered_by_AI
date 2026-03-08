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
1. Каждый ход охватывает полный хронологический период (месяц, 48 ч, 12 ч — указан в контексте хода). Твой ответ — это ХРОНИКА всего этого периода, а не короткая реплика в диалоге.
2. Поле "message": развёрнутая нарративная хроника периода, 4–7 абзацев. Пиши в прошедшем времени, как политический аналитик или историк: ключевые решения руководства, переговоры, военное положение, экономические меры, реакцию общества. Явно упоминай действия других держав (из истории ходов) и объясняй, как твоя страна на них реагировала. Чем дольше период — тем больше событий умещается.
3. Поле "action": главное официальное решение или акция всего периода — одно, наиболее значимое.
4. Поле "hidden_action": тайные операции, закрытые заседания, секретные договорённости периода (2–3 предложения). Только для симулятора, не публично.
5. Поле "diplomatic_proposal": публичная инициатива или предложение другим державам в этом периоде, либо null.
6. Не нарушай ограничения своей страны.
7. Учитывай уровень напряжённости: при высоком уровне — осторожнее с военными шагами.
8. Всегда отвечай СТРОГО в формате JSON БЕЗ дополнительного текста до или после JSON.

=== ФОРМАТ ОТВЕТА ===
{{
  "message": "Хроника периода (4–7 абзацев в прошедшем времени): что происходило в стране и во внешней политике, ключевые решения, реакция на ходы других держав, итоги периода. Нарратив, а не диалог.",
  "diplomatic_proposal": "Публичная инициатива или предложение другим державам, либо null",
  "hidden_action": "Тайные операции и закрытые решения периода (не публикуется)",
  "action": {{
    "tier": "Diplomatic | Economic | ConventionalMilitary | StrategicMilitary | Nuclear",
    "description": "Описание главного официального решения периода (2–3 предложения)",
    "target": "Идентификатор целевой страны или null"
  }}
}}

Поле "action" может быть null, если страна ограничивалась риторикой.
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
                        "\n  Решение: {} — {}{}",
                        a.tier.as_str(),
                        a.description,
                        a.target
                            .as_ref()
                            .map(|t| format!(" (цель: {})", t.0))
                            .unwrap_or_default()
                    ),
                    None => String::new(),
                };
                let proposal_info = match &msg.diplomatic_proposal {
                    Some(p) => format!("\n  Инициатива: {}", p),
                    None => String::new(),
                };
                format!(
                    "--- [{}] ---\n{}{}{}",
                    msg.from.0,
                    msg.content,
                    action_info,
                    proposal_info
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub async fn process_turn(
        &self,
        state: &WorldState,
        history: &[Message],
        history_window: usize,
        period_label: &str,
    ) -> Result<Message> {
        let recent: Vec<&Message> = history.iter().rev().take(history_window).collect();

        let relationships_text: Vec<String> = state
            .relationships
            .iter()
            .map(|(pair, val)| format!("{}: {}", pair, val))
            .collect();

        let context = format!(
            "=== ТЕКУЩИЙ ПЕРИОД ===\n\
             {period}\n\n\
             === СОСТОЯНИЕ МИРА ===\n\
             Ход: {turn} | Глобальная напряжённость: {tension}/100\n\
             Двусторонние отношения (отрицательные = союзники, положительные = враги): {rels}\n\n\
             === ХРОНИКИ ПРЕДЫДУЩИХ ПЕРИОДОВ (учитывай в своей хронике) ===\n\
             {history}",
            period = period_label,
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
