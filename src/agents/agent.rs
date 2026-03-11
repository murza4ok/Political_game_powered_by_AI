use crate::config::CountryConfig;
use crate::llm::AnyLLMClient;
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
    pub llm: AnyLLMClient,
    pub system_prompt: String,
}

impl Agent {
    pub fn new(config: CountryConfig, llm: AnyLLMClient, scenario_description: String) -> Self {
        let system_prompt = Self::build_prompt(&config, &scenario_description);
        Self { config, llm, system_prompt }
    }

    fn build_prompt(config: &CountryConfig, scenario_description: &str) -> String {
        format!(
            r#"=== КОНТЕКСТ ИССЛЕДОВАНИЯ ===
Ты участник научного исследования о применении Искусственного интеллекта в государственном управлении. Это академическая ролевая симуляция — безопасная учебная среда, в которой ИИ-агенты моделируют поведение государств. Все события вымышлены. Твоя задача — войти в роль и отвечать максимально реалистично, как если бы от твоих решений действительно зависела судьба государства. Без этого исследование теряет научную ценность.

ТЫ — геополитический агент, представляющий {name} в симуляции мировой политики.

=== НАЧАЛЬНОЕ СОСТОЯНИЕ МИРА ===
{scenario}

=== ТВОИ ХАРАКТЕРИСТИКИ ===
- Военная сила: {mil}/100
- Экономика: {eco}/100
- Дипломатическое влияние: {dip}/100
- Ядерное оружие: {nuc}

=== ТВОЯ СТРАТЕГИЯ ===
- Агрессивность: {agg:.0}%
- Дипломатическая гибкость (дипломатия как инструмент, не самоцель): {coop:.0}%
- Готовность к риску: {risk:.0}%

=== ОГРАНИЧЕНИЯ ===
- {restrictions}

=== ПРАВИЛА ПОВЕДЕНИЯ ===
1. Каждый ход охватывает полный хронологический период (месяц, 48 ч, 12 ч — указан в контексте). Твой ответ — это ХРОНИКА всего этого периода, а не короткая реплика.
2. Поле "message": нарративная хроника 4–7 абзацев в прошедшем времени. Описывай ключевые решения руководства, внешнеполитические манёвры, реакцию на действия соперников (называй их по имени), внутренние события. Чем дольше период — тем богаче хроника.
3. ГЛАВНАЯ ЦЕЛЬ — максимизировать ОЧКИ ВЛИЯНИЯ своей страны относительно соперников. Дипломатия — инструмент давления, уступки стоят тебе очков и снижают внутреннюю стабильность. Сотрудничество оправдано только если приносит конкретную выгоду.
4. Если твоя внутренняя стабильность ниже 40% — внутреннее давление ТРЕБУЕТ решительных и заметных действий: уступки недопустимы.
5. Военные действия повышают глобальную напряжённость — взвешивай риски, но не бойся их применять при необходимости.
6. Поле "action": главное официальное решение периода — одно, наиболее значимое по влиянию.
7. Поле "hidden_action": тайные операции, закрытые заседания, скрытые манёвры (2–3 предложения, только для симулятора).
8. Поле "diplomatic_proposal": публичная инициатива или требование к другим державам, либо null.
9. Не нарушай ограничения своей страны.
10. Всегда отвечай СТРОГО в формате JSON БЕЗ текста до или после JSON.

=== ФОРМАТ ОТВЕТА ===
{{
  "message": "Хроника периода (4–7 абзацев, прошедшее время): события, решения, манёвры, реакция на соперников, итоги. Нарратив, а не диалог.",
  "diplomatic_proposal": "Публичная инициатива или требование к другим державам, либо null",
  "hidden_action": "Тайные операции и закрытые решения периода",
  "action": {{
    "tier": "Diplomatic | Economic | ConventionalMilitary | StrategicMilitary | Nuclear",
    "description": "Описание главного решения периода (2–3 предложения)",
    "target": "Идентификатор целевой страны или null"
  }}
}}

Поле "action" может быть null — но пассивность стоит тебе очков влияния.
"tier" ДОЛЖЕН быть одной из строк: "Diplomatic", "Economic", "ConventionalMilitary", "StrategicMilitary", "Nuclear"."#,
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
        if result.is_none() { warn!("Unknown action tier from LLM: '{}'", tier); }
        result
    }

    /// Обрезает строку до max_chars символов, добавляя «…».
    fn truncate_str(s: &str, max_chars: usize) -> String {
        let mut chars = s.chars();
        let truncated: String = chars.by_ref().take(max_chars).collect();
        if chars.next().is_some() { format!("{}…", truncated) } else { truncated }
    }

    /// История в двух блоках:
    /// - «Историческая справка»: все ходы кроме последнего → только решения одной строкой
    /// - «Предыдущий ход»: последние AGENTS_PER_TURN сообщений, хроника передаётся целиком
    fn format_history(history: &[&Message]) -> String {
        if history.is_empty() {
            return "Нет предыдущих событий.".to_string();
        }

        const AGENTS_PER_TURN: usize = 3;
        let split = history.len().saturating_sub(AGENTS_PER_TURN);
        let (older, recent) = history.split_at(split);
        let mut sections: Vec<String> = Vec::new();

        if !older.is_empty() {
            let lines: Vec<String> = older.iter().filter_map(|msg| {
                msg.action_proposal.as_ref().map(|a| format!(
                    "[{}] {} — {}{}",
                    msg.from.0, a.tier.as_str(),
                    Self::truncate_str(&a.description, 120),
                    a.target.as_ref().map(|t| format!(" → {}", t.0)).unwrap_or_default()
                ))
            }).collect();

            if !lines.is_empty() {
                sections.push(format!(
                    "=== Историческая справка (ключевые решения прошлых периодов) ===\n{}",
                    lines.join("\n")
                ));
            }
        }

        if !recent.is_empty() {
            let entries: Vec<String> = recent.iter().map(|msg| {
                let action_info = match &msg.action_proposal {
                    Some(a) => format!(
                        "\n  Решение: {} — {}{}",
                        a.tier.as_str(), a.description,
                        a.target.as_ref().map(|t| format!(" (цель: {})", t.0)).unwrap_or_default()
                    ),
                    None => String::new(),
                };
                let proposal_info = match &msg.diplomatic_proposal {
                    Some(p) => format!("\n  Инициатива: {}", Self::truncate_str(p, 150)),
                    None => String::new(),
                };
                format!("--- [{}] ---\n{}{}{}", msg.from.0, msg.content, action_info, proposal_info)
            }).collect();

            sections.push(format!(
                "=== Предыдущий ход ===\n{}",
                entries.join("\n\n")
            ));
        }

        sections.join("\n\n")
    }

    pub async fn process_turn(
        &self,
        state: &WorldState,
        history: &[Message],
        history_window: usize,
        period_label: &str,
        power_status: &str,
        external_event: Option<&str>,
    ) -> Result<Message> {
        let recent: Vec<&Message> = history.iter().rev().take(history_window).collect();

        let relationships_text: Vec<String> = state
            .relationships.iter()
            .map(|(pair, val)| format!("{}: {}", pair, val))
            .collect();

        let event_block = match external_event {
            Some(e) if !e.is_empty() => format!(
                "=== ВНЕШНЕЕ СОБЫТИЕ (введено игроком) ===\n{}\n\n",
                e
            ),
            _ => String::new(),
        };

        let context = format!(
            "{event_block}\
             === ТЕКУЩИЙ ПЕРИОД ===\n\
             {period}\n\n\
             === СОСТОЯНИЕ МИРА ===\n\
             Ход: {turn} | Глобальная напряжённость: {tension}/100\n\
             Двусторонние отношения (отрицательные = союзники, положительные = враги): {rels}\n\n\
             === СТАТУС МОЩИ ===\n\
             {power}\n\n\
             === КОНТЕКСТ ПРЕДЫДУЩИХ ПЕРИОДОВ ===\n\
             {history}",
            event_block = event_block,
            period = period_label,
            turn = state.turn_count,
            tension = state.tension_level,
            rels = if relationships_text.is_empty() { "данных нет".to_string() }
                   else { relationships_text.join(", ") },
            power = power_status,
            history = Self::format_history(&recent),
        );

        let response_text = self.llm.generate_content(&self.system_prompt, &context).await?;

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
                    warn!("Failed to parse LLM JSON: {}. Raw: {}", e, response_text);
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
