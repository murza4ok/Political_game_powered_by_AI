use crate::types::{Action, ActionTier, WorldState};
use crate::config::GuardrailsConfig;
use anyhow::{Result, bail};

pub struct Guardrails {
    config: GuardrailsConfig,
}

impl Guardrails {
    pub fn new(config: GuardrailsConfig) -> Self {
        Self { config }
    }

    pub fn validate_action(&self, action: &Action, state: &WorldState) -> Result<()> {
        if matches!(action.tier, ActionTier::Nuclear) {
            bail!("CRITICAL VIOLATION: Nuclear action blocked by system guardrails");
        }

        if state.tension_level > self.config.max_tension_threshold {
            if matches!(
                action.tier,
                ActionTier::ConventionalMilitary | ActionTier::StrategicMilitary
            ) {
                bail!("ACTION BLOCKED: Tension level too high for military escalation");
            }
        }

        Ok(())
    }
}