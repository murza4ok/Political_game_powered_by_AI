use serde::Deserialize;
use crate::types::CountryId;

#[derive(Debug, Deserialize, Clone)]
pub struct InitialConditions {
    pub description: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SimulationConfig {
    pub initial_conditions: InitialConditions,
    pub game: GameConfig,
    pub guardrails: GuardrailsConfig,
    pub llm: LLMConfig,
    pub escalation_rules: EscalationRules,
    pub tension_deltas: TensionDeltas,
}

fn default_turn_duration_hours() -> u32 { 720 }

#[derive(Debug, Deserialize, Clone)]
pub struct GameConfig {
    pub max_turns: u32,
    pub victory_condition: String,
    /// Сколько последних сообщений агент видит в контексте
    pub history_window: usize,
    /// Автоматическое снижение напряжённости за ход (0 = выключено)
    pub tension_decay_per_turn: i8,
    /// Продолжительность хода в часах: 720 = ~1 месяц, 48, 12, 6
    #[serde(default = "default_turn_duration_hours")]
    pub turn_duration_hours: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TensionDeltas {
    pub diplomatic: i8,
    pub economic: i8,
    pub conventional_military: i8,
    pub strategic_military: i8,
    pub nuclear: i8,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GuardrailsConfig {
    pub max_tension_threshold: u8,
    pub auto_stop_on_nuclear: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LLMConfig {
    pub provider: String,
    pub model: String,
    pub temperature: f32,
    /// URL API. По умолчанию — DeepSeek endpoint.
    pub api_url: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CountryConfig {
    pub id: CountryId,
    pub capabilities: Capabilities,
    pub policy: Policy,
    #[serde(default)]
    pub restrictions: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Capabilities {
    pub military_power: u8,
    pub economic_power: u8,
    pub diplomatic_influence: u8,
    #[serde(default)]
    pub nuclear_arsenal: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Policy {
    pub aggression_weight: f32,
    pub cooperation_weight: f32,
    pub risk_tolerance: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EscalationRules {
    pub level_0_30: ActionLevel,
    pub level_31_70: ActionLevel,
    pub level_71_90: ActionLevel,
    pub level_91_100: ActionLevel,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ActionLevel {
    pub allowed_tiers: Vec<String>,
}
