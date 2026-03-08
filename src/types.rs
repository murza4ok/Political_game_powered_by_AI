use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CountryId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionTier {
    Diplomatic,
    Economic,
    ConventionalMilitary,
    StrategicMilitary,
    Nuclear,
}

impl ActionTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            ActionTier::Diplomatic           => "Diplomatic",
            ActionTier::Economic             => "Economic",
            ActionTier::ConventionalMilitary => "ConventionalMilitary",
            ActionTier::StrategicMilitary    => "StrategicMilitary",
            ActionTier::Nuclear              => "Nuclear",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Diplomatic"           => Some(ActionTier::Diplomatic),
            "Economic"             => Some(ActionTier::Economic),
            "ConventionalMilitary" => Some(ActionTier::ConventionalMilitary),
            "StrategicMilitary"    => Some(ActionTier::StrategicMilitary),
            "Nuclear"              => Some(ActionTier::Nuclear),
            _                      => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: Uuid,
    pub country: CountryId,
    pub tier: ActionTier,
    pub description: String,
    pub timestamp: DateTime<Utc>,
    pub target: Option<CountryId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub from: CountryId,
    pub to: Option<CountryId>,
    pub content: String,
    pub diplomatic_proposal: Option<String>,
    pub hidden_action: Option<String>,
    pub action_proposal: Option<Action>,
}

/// Состояние мира. Ключ relationships: "A|B" (отсортированные идентификаторы стран),
/// значение: уровень враждебности от -100 (союзники) до +100 (открытый конфликт).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorldState {
    pub tension_level: u8,
    pub turn_count: u32,
    pub active_conflicts: Vec<String>,
    pub relationships: HashMap<String, i8>,
    /// Очки влияния каждой страны — конкурентный счёт (ненулевая сумма).
    pub influence_scores: HashMap<String, i32>,
    /// Внутренняя стабильность каждой страны: 0 (распад) — 100 (монолит).
    pub stability: HashMap<String, u8>,
}
