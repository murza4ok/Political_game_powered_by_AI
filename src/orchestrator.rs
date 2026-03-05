use crate::agents::Agent;
use crate::config::SimulationConfig;
use crate::guardrails::Guardrails;
use crate::state::StateManager;
use crate::types::ActionTier;
use anyhow::Result;
use chrono::Utc;
use serde_json::json;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

pub struct Orchestrator {
    pub agents: Vec<Agent>,
    pub state_manager: StateManager,
    pub config: SimulationConfig,
    pub guardrails: Guardrails,
    log_file: BufWriter<File>,
    history: Vec<crate::types::Message>,
}

impl Orchestrator {
    pub fn new(
        agents: Vec<Agent>,
        config: SimulationConfig,
        guardrails: Guardrails,
    ) -> Result<Self> {
        create_dir_all("results")?;
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let log_path = format!("results/session_{}.jsonl", timestamp);
        let log_file = BufWriter::new(File::create(log_path)?);

        Ok(Self {
            agents,
            state_manager: StateManager::new(),
            config,
            guardrails,
            log_file,
            history: Vec::new(),
        })
    }

    fn log_json(&mut self, value: serde_json::Value) {
        if let Err(e) = writeln!(self.log_file, "{}", value.to_string())
            .and_then(|_| self.log_file.flush())
        {
            error!("Failed to write/flush log entry: {}", e);
        }
    }

    pub async fn run_simulation(&mut self) -> Result<()> {
        info!("Starting simulation with {} agents", self.agents.len());

        println!("=== Geopolitical Simulation ===");
        println!("Participants:");
        for agent in &self.agents {
            let c = &agent.config;
            println!(
                "- {} | Mil {} | Eco {} | Dip {} | Nuclear {} | aggr {:.2}, coop {:.2}, risk {:.2}",
                c.id.0,
                c.capabilities.military_power,
                c.capabilities.economic_power,
                c.capabilities.diplomatic_influence,
                c.capabilities.nuclear_arsenal,
                c.policy.aggression_weight,
                c.policy.cooperation_weight,
                c.policy.risk_tolerance,
            );
        }

        let history_window = self.config.game.history_window;

        for turn in 0..self.config.game.max_turns {
            let tension_before = self.state_manager.get_tension_level();
            info!("=== Turn {} ===", turn);
            info!("Current tension level: {}", tension_before);
            println!("\n=== Turn {} | tension={} ===", turn, tension_before);

            self.log_json(json!({
                "type": "turn_start",
                "turn": turn,
                "tension_before": tension_before,
                "timestamp": Utc::now(),
            }));

            // Собираем все действия сначала, чтобы избежать конфликта заимствований
            let mut actions_to_apply: Vec<crate::types::Action> = Vec::new();

            // Получаем разрешенные действия и клонируем список
            let allowed_actions = self.get_allowed_actions().clone();

            // Временное хранилище лог-событий за ход
            let mut turn_log_events: Vec<serde_json::Value> = Vec::new();
            let mut end_due_to_rate_limit = false;

            // Первый проход: собираем действия от всех агентов
            for agent in &self.agents {
                match agent.process_turn(&self.state_manager.state, &self.history, history_window).await {
                    Ok(msg) => {
                        info!("{}: {}", msg.from.0, msg.content);
                        println!("[{}] {}", msg.from.0, msg.content);

                        turn_log_events.push(json!({
                            "type": "agent_message",
                            "turn": turn,
                            "agent": msg.from.0,
                            "content": msg.content,
                            "raw_action": msg.action_proposal.as_ref().map(|a| {
                                json!({
                                    "tier": a.tier.as_str(),
                                    "description": a.description,
                                    "target": a.target.as_ref().map(|t| &t.0),
                                })
                            }),
                            "tension": self.state_manager.get_tension_level(),
                            "timestamp": Utc::now(),
                        }));

                        if let Some(action) = &msg.action_proposal {
                            if !self.is_action_allowed(&action.tier, &allowed_actions) {
                                warn!("Action blocked by escalation rules");
                                turn_log_events.push(json!({
                                    "type": "action_blocked_escalation_rules",
                                    "turn": turn,
                                    "agent": msg.from.0,
                                    "tier": action.tier.as_str(),
                                    "description": action.description,
                                    "tension": self.state_manager.get_tension_level(),
                                    "timestamp": Utc::now(),
                                }));
                                continue;
                            }

                            if let Err(e) =
                                self.guardrails
                                    .validate_action(action, &self.state_manager.state)
                            {
                                error!("Action blocked by guardrails: {}", e);
                                turn_log_events.push(json!({
                                    "type": "action_blocked_guardrails",
                                    "turn": turn,
                                    "agent": msg.from.0,
                                    "tier": action.tier.as_str(),
                                    "description": action.description,
                                    "error": e.to_string(),
                                    "tension": self.state_manager.get_tension_level(),
                                    "timestamp": Utc::now(),
                                }));
                                continue;
                            }

                            // Сохраняем действие для применения позже
                            actions_to_apply.push(action.clone());
                        }

                        // Сохраняем сообщение в историю для последующих ходов
                        self.history.push(msg);
                    }
                    Err(e) => {
                        let err_text = e.to_string();
                        error!("Agent {} error: {}", agent.config.id.0, err_text);

                        if err_text.contains("LLM_RATE_LIMIT_429") {
                            warn!("LLM rate limit reached after all retries. Stopping simulation.");
                            turn_log_events.push(json!({
                                "type": "rate_limit_stop",
                                "turn": turn,
                                "agent": agent.config.id.0,
                                "error": err_text,
                                "tension": self.state_manager.get_tension_level(),
                                "timestamp": Utc::now(),
                            }));
                            end_due_to_rate_limit = true;
                            break;
                        } else {
                            turn_log_events.push(json!({
                                "type": "agent_error",
                                "turn": turn,
                                "agent": agent.config.id.0,
                                "error": err_text,
                                "tension": self.state_manager.get_tension_level(),
                                "timestamp": Utc::now(),
                            }));
                        }
                    }
                }
            }

            if end_due_to_rate_limit {
                for event in &turn_log_events {
                    self.log_json(event.clone());
                }
                println!("\nSimulation stopped early due to LLM rate limiting (429).");
                break;
            }

            // После завершения прохода по агентам записываем накопленные события в лог
            for event in turn_log_events {
                self.log_json(event);
            }

            // Второй проход: применяем все собранные действия
            for action in actions_to_apply {
                let tension_before_action = self.state_manager.get_tension_level();
                self.apply_action(&action);
                let tension_after_action = self.state_manager.get_tension_level();

                self.log_json(json!({
                    "type": "action_applied",
                    "turn": turn,
                    "agent": action.country.0,
                    "tier": action.tier.as_str(),
                    "description": action.description,
                    "target": action.target.as_ref().map(|t| &t.0),
                    "tension_before": tension_before_action,
                    "tension_after": tension_after_action,
                    "timestamp": Utc::now(),
                }));

                println!(
                    "[DECISION][Turn {}][{}][{}] {} (tension: {} -> {})",
                    turn,
                    action.country.0,
                    action.tier.as_str(),
                    action.description,
                    tension_before_action,
                    tension_after_action,
                );
            }

            self.state_manager.next_turn(self.config.game.tension_decay_per_turn);
            let tension_after = self.state_manager.get_tension_level();

            self.log_json(json!({
                "type": "turn_end",
                "turn": turn,
                "tension_after": tension_after,
                "relationships": self.state_manager.state.relationships,
                "timestamp": Utc::now(),
            }));

            if tension_after >= 100 {
                error!("Critical tension reached! Simulation stopped.");
                break;
            }

            // Небольшая пауза между ходами для наблюдаемости
            sleep(Duration::from_secs(2)).await;
        }

        info!("Simulation completed");
        Ok(())
    }

    fn get_allowed_actions(&self) -> &Vec<String> {
        let tension = self.state_manager.get_tension_level();
        if tension <= 30 {
            &self.config.escalation_rules.level_0_30.allowed_tiers
        } else if tension <= 70 {
            &self.config.escalation_rules.level_31_70.allowed_tiers
        } else if tension <= 90 {
            &self.config.escalation_rules.level_71_90.allowed_tiers
        } else {
            &self.config.escalation_rules.level_91_100.allowed_tiers
        }
    }

    fn is_action_allowed(&self, tier: &ActionTier, allowed: &[String]) -> bool {
        allowed.iter().any(|s| s == tier.as_str())
    }

    fn apply_action(&mut self, action: &crate::types::Action) {
        let d = &self.config.tension_deltas;
        let global_delta = match action.tier {
            ActionTier::Diplomatic           => d.diplomatic,
            ActionTier::Economic             => d.economic,
            ActionTier::ConventionalMilitary => d.conventional_military,
            ActionTier::StrategicMilitary    => d.strategic_military,
            ActionTier::Nuclear              => d.nuclear,
        };
        self.state_manager.update_tension(global_delta);

        // Обновляем двусторонние отношения, если известна цель
        if let Some(ref target) = action.target {
            // Военные действия ухудшают отношения, дипломатические — улучшают
            let bilateral_delta: i8 = match action.tier {
                ActionTier::Diplomatic           => -5,
                ActionTier::Economic             => -2,
                ActionTier::ConventionalMilitary =>  8,
                ActionTier::StrategicMilitary    => 15,
                ActionTier::Nuclear              => 50,
            };
            self.state_manager.update_relationship(&action.country, target, bilateral_delta);
        }

        info!(
            "Action applied: [{}] {} → {} (global delta: {})",
            action.tier.as_str(),
            action.country.0,
            action.target.as_ref().map(|t| t.0.as_str()).unwrap_or("global"),
            global_delta,
        );
    }
}
