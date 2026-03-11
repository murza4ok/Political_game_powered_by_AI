use crate::agents::Agent;
use crate::config::SimulationConfig;
use crate::guardrails::Guardrails;
use crate::state::StateManager;
use crate::types::{ActionTier, CountryId};
use anyhow::Result;
use chrono::{Datelike, NaiveDate, Duration as ChronoDuration, Utc};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

pub struct Orchestrator {
    pub agents: Vec<Agent>,
    pub state_manager: StateManager,
    pub config: SimulationConfig,
    pub guardrails: Guardrails,
    log_file: BufWriter<File>,
    history: Vec<crate::types::Message>,
    /// Накопленные игровые часы с начала симуляции
    in_game_hours_elapsed: u64,
    /// Внешнее событие, введённое игроком — будет добавлено в контекст следующего хода
    pending_event: Option<String>,
}

/// Форматирует игровую дату в виде "DD мес YYYY"
fn fmt_date_ru(d: NaiveDate) -> String {
    let months = [
        "янв", "фев", "мар", "апр", "май", "июн",
        "июл", "авг", "сен", "окт", "ноя", "дек",
    ];
    format!("{} {} {}", d.day(), months[(d.month() - 1) as usize], d.year())
}

/// Строит метку периода хода: "DD мес YYYY — DD мес YYYY (~1 месяц / 48 ч / ...)"
fn game_period_label(hours_elapsed: u64, duration_hours: u64) -> String {
    let base = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let start = base + ChronoDuration::hours(hours_elapsed as i64);
    let end   = base + ChronoDuration::hours((hours_elapsed + duration_hours) as i64);

    let duration_str = match duration_hours {
        h if h >= 600 => "~1 месяц".to_string(),
        48            => "48 часов".to_string(),
        12            => "12 часов".to_string(),
        6             => "6 часов".to_string(),
        h             => format!("{} ч", h),
    };

    format!("{} — {} ({})", fmt_date_ru(start), fmt_date_ru(end), duration_str)
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

        let mut state_manager = StateManager::new();
        for agent in &agents {
            state_manager.init_country(&agent.config.id);
        }

        Ok(Self {
            agents,
            state_manager,
            config,
            guardrails,
            log_file,
            history: Vec::new(),
            in_game_hours_elapsed: 0,
            pending_event: None,
        })
    }

    fn log_json(&mut self, value: serde_json::Value) {
        if let Err(e) = writeln!(self.log_file, "{}", value.to_string())
            .and_then(|_| self.log_file.flush())
        {
            error!("Failed to write/flush log entry: {}", e);
        }
    }

    async fn read_line_stdin() -> String {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap_or(0);
        line.trim().to_string()
    }

    /// Пауза для игрока. Возвращает:
    /// - None → симуляция останавливается
    /// - Some("") → продолжаем без события
    /// - Some(text) → продолжаем, text добавляется в контекст следующего хода
    async fn player_input_prompt(&self, turn: u32) -> Option<String> {
        println!("\n╔════════════════════════════════════════╗");
        println!("║  ПАУЗА — Ход {} завершён               ║", turn);
        println!("╚════════════════════════════════════════╝");
        print!("Продолжить симуляцию? [Y/n]: ");
        let _ = std::io::stdout().flush();

        let answer = Self::read_line_stdin().await;

        if answer.eq_ignore_ascii_case("n") {
            print!("Остановить или добавить внешнее событие? [stop/event]: ");
            let _ = std::io::stdout().flush();
            let choice = Self::read_line_stdin().await;

            if choice.eq_ignore_ascii_case("event") {
                print!("Введите описание события: ");
                let _ = std::io::stdout().flush();
                let event = Self::read_line_stdin().await;
                if event.is_empty() {
                    println!("Событие пустое — продолжаем без него.");
                    return Some(String::new());
                }
                println!("✓ Событие добавлено в контекст следующего хода.");
                return Some(event);
            } else {
                println!("Симуляция остановлена.");
                return None;
            }
        }

        Some(String::new())
    }

    /// Строка статуса очков и стабильности для консоли.
    fn power_status_line(&self) -> String {
        self.agents.iter().map(|a| {
            let id    = &a.config.id.0;
            let score = self.state_manager.state.influence_scores.get(id).copied().unwrap_or(0);
            let stab  = self.state_manager.state.stability.get(id).copied().unwrap_or(70);
            format!("{}: {:+} (стаб {}%)", id, score, stab)
        }).collect::<Vec<_>>().join(" | ")
    }

    /// Блок СТАТУС МОЩИ для передачи конкретному агенту.
    fn power_status_for_agent(&self, agent_id: &str) -> String {
        let my_score = self.state_manager.state.influence_scores.get(agent_id).copied().unwrap_or(0);
        let my_stab  = self.state_manager.state.stability.get(agent_id).copied().unwrap_or(70);

        let rivals: Vec<String> = self.agents.iter()
            .filter(|a| a.config.id.0 != agent_id)
            .map(|a| {
                let id    = &a.config.id.0;
                let score = self.state_manager.state.influence_scores.get(id).copied().unwrap_or(0);
                format!("{}: {:+}", id, score)
            })
            .collect();

        let pressure = if my_stab < 40 {
            "\n⚠ КРИТИЧЕСКИ: внутреннее давление требует решительных действий!"
        } else { "" };

        format!(
            "Твои очки влияния: {:+} | Внутренняя стабильность: {}%\nСоперники: {}{}",
            my_score, my_stab, rivals.join(" | "), pressure
        )
    }

    pub async fn run_simulation(&mut self) -> Result<()> {
        info!("Starting simulation with {} agents", self.agents.len());

        println!("=== Агенты симуляции ===");
        for agent in &self.agents {
            let c = &agent.config;
            println!(
                "- {} | Mil {} | Eco {} | Dip {} | Nuclear {} | aggr {:.0}%, flex {:.0}%, risk {:.0}%",
                c.id.0,
                c.capabilities.military_power,
                c.capabilities.economic_power,
                c.capabilities.diplomatic_influence,
                c.capabilities.nuclear_arsenal,
                c.policy.aggression_weight * 100.0,
                c.policy.cooperation_weight * 100.0,
                c.policy.risk_tolerance * 100.0,
            );
        }

        let history_window        = self.config.game.history_window;
        let player_input_interval = self.config.game.player_input_interval;

        for turn in 0..self.config.game.max_turns {
            let turn_duration_hours = self.config.game.turn_duration_hours as u64;
            let period = game_period_label(self.in_game_hours_elapsed, turn_duration_hours);

            let tension_before = self.state_manager.get_tension_level();
            info!("=== Turn {} | Period: {} ===", turn, period);
            println!("\n=== Ход {} | {} | tension={} ===", turn, period, tension_before);

            let event_for_this_turn = self.pending_event.take();

            self.log_json(json!({
                "type": "turn_start",
                "turn": turn,
                "period": period,
                "turn_duration_hours": turn_duration_hours,
                "tension_before": tension_before,
                "external_event": event_for_this_turn,
                "timestamp": Utc::now(),
            }));

            let mut actions_to_apply: Vec<crate::types::Action> = Vec::new();
            let allowed_actions = self.get_allowed_actions().clone();
            let mut turn_log_events: Vec<serde_json::Value> = Vec::new();
            let mut end_due_to_rate_limit = false;

            let agent_count = self.agents.len();
            for i in 0..agent_count {
                let agent_id     = self.agents[i].config.id.0.clone();
                let power_status = self.power_status_for_agent(&agent_id);

                match self.agents[i].process_turn(
                    &self.state_manager.state,
                    &self.history,
                    history_window,
                    &period,
                    &power_status,
                    event_for_this_turn.as_deref(),
                ).await {
                    Ok(msg) => {
                        info!("{}: {}", msg.from.0, msg.content);
                        println!("\n[{}]\n{}", msg.from.0, msg.content);

                        turn_log_events.push(json!({
                            "type": "agent_message",
                            "turn": turn,
                            "period": period,
                            "agent": msg.from.0,
                            "content": msg.content,
                            "diplomatic_proposal": msg.diplomatic_proposal,
                            "hidden_action": msg.hidden_action,
                            "raw_action": msg.action_proposal.as_ref().map(|a| json!({
                                "tier": a.tier.as_str(),
                                "description": a.description,
                                "target": a.target.as_ref().map(|t| &t.0),
                            })),
                            "influence_scores": self.state_manager.state.influence_scores,
                            "stability": self.state_manager.state.stability,
                            "tension": self.state_manager.get_tension_level(),
                            "timestamp": Utc::now(),
                        }));

                        if let Some(action) = &msg.action_proposal {
                            if !self.is_action_allowed(&action.tier, &allowed_actions) {
                                warn!("Action blocked by escalation rules");
                                turn_log_events.push(json!({
                                    "type": "action_blocked_escalation_rules",
                                    "turn": turn, "agent": msg.from.0,
                                    "tier": action.tier.as_str(), "description": action.description,
                                    "timestamp": Utc::now(),
                                }));
                                self.history.push(msg);
                                continue;
                            }
                            if let Err(e) = self.guardrails.validate_action(action, &self.state_manager.state) {
                                error!("Action blocked by guardrails: {}", e);
                                turn_log_events.push(json!({
                                    "type": "action_blocked_guardrails",
                                    "turn": turn, "agent": msg.from.0,
                                    "tier": action.tier.as_str(), "description": action.description,
                                    "error": e.to_string(), "timestamp": Utc::now(),
                                }));
                                self.history.push(msg);
                                continue;
                            }
                            actions_to_apply.push(action.clone());
                        }
                        self.history.push(msg);
                    }
                    Err(e) => {
                        let err_text = e.to_string();
                        error!("Agent {} error: {}", agent_id, err_text);
                        if err_text.contains("LLM_RATE_LIMIT_429") {
                            warn!("LLM rate limit reached. Stopping simulation.");
                            turn_log_events.push(json!({
                                "type": "rate_limit_stop", "turn": turn,
                                "agent": agent_id, "error": err_text, "timestamp": Utc::now(),
                            }));
                            end_due_to_rate_limit = true;
                            break;
                        } else {
                            turn_log_events.push(json!({
                                "type": "agent_error", "turn": turn,
                                "agent": agent_id, "error": err_text, "timestamp": Utc::now(),
                            }));
                        }
                    }
                }
            }

            if end_due_to_rate_limit {
                for event in &turn_log_events { self.log_json(event.clone()); }
                println!("\nSimulation stopped early due to LLM rate limiting (429).");
                break;
            }

            for event in turn_log_events { self.log_json(event); }

            // Применяем действия + очки влияния
            let country_ids: Vec<CountryId> = self.agents.iter().map(|a| a.config.id.clone()).collect();
            let mut agents_with_action: HashSet<String> = HashSet::new();

            for action in actions_to_apply {
                let t_before = self.state_manager.get_tension_level();
                self.apply_action(&action);
                let t_after = self.state_manager.get_tension_level();
                agents_with_action.insert(action.country.0.clone());

                self.log_json(json!({
                    "type": "action_applied",
                    "turn": turn,
                    "agent": action.country.0,
                    "tier": action.tier.as_str(),
                    "description": action.description,
                    "target": action.target.as_ref().map(|t| &t.0),
                    "tension_before": t_before,
                    "tension_after": t_after,
                    "influence_scores": self.state_manager.state.influence_scores,
                    "timestamp": Utc::now(),
                }));

                println!(
                    "[РЕШЕНИЕ][Ход {}][{}][{}] {} (tension: {} → {})",
                    turn, action.country.0, action.tier.as_str(),
                    action.description, t_before, t_after,
                );
            }

            // Штраф за пассивность
            for id in &country_ids {
                if !agents_with_action.contains(&id.0) {
                    self.state_manager.apply_influence(id, -1, None, 0);
                }
            }

            // Пересчёт стабильности
            self.state_manager.tick_stability(&country_ids);

            self.state_manager.next_turn(self.config.game.tension_decay_per_turn);
            let tension_after = self.state_manager.get_tension_level();
            self.in_game_hours_elapsed += turn_duration_hours;

            println!("\n[СТАТУС МОЩИ] {}", self.power_status_line());

            self.log_json(json!({
                "type": "turn_end",
                "turn": turn,
                "period": period,
                "tension_after": tension_after,
                "in_game_hours_elapsed": self.in_game_hours_elapsed,
                "influence_scores": self.state_manager.state.influence_scores,
                "stability": self.state_manager.state.stability,
                "relationships": self.state_manager.state.relationships,
                "timestamp": Utc::now(),
            }));

            if tension_after >= 100 {
                error!("Critical tension reached! Simulation stopped.");
                break;
            }

            // Пауза для игрока каждые N ходов
            if player_input_interval > 0 && (turn + 1) % player_input_interval == 0 {
                match self.player_input_prompt(turn).await {
                    None => break,
                    Some(event) if !event.is_empty() => { self.pending_event = Some(event); }
                    _ => {}
                }
            }

            sleep(Duration::from_secs(2)).await;
        }

        info!("Simulation completed");
        Ok(())
    }

    fn get_allowed_actions(&self) -> &Vec<String> {
        let tension = self.state_manager.get_tension_level();
        if tension <= 30      { &self.config.escalation_rules.level_0_30.allowed_tiers }
        else if tension <= 70 { &self.config.escalation_rules.level_31_70.allowed_tiers }
        else if tension <= 90 { &self.config.escalation_rules.level_71_90.allowed_tiers }
        else                  { &self.config.escalation_rules.level_91_100.allowed_tiers }
    }

    fn is_action_allowed(&self, tier: &ActionTier, allowed: &[String]) -> bool {
        allowed.iter().any(|s| s == tier.as_str())
    }

    fn apply_action(&mut self, action: &crate::types::Action) {
        let d = &self.config.tension_deltas;
        let (global_delta, influence_actor, influence_target) = match action.tier {
            ActionTier::Diplomatic           => (d.diplomatic,            1_i32,  0_i32),
            ActionTier::Economic             => (d.economic,              2,      0),
            ActionTier::ConventionalMilitary => (d.conventional_military, 4,     -3),
            ActionTier::StrategicMilitary    => (d.strategic_military,    7,     -5),
            ActionTier::Nuclear              => (d.nuclear,               10,   -20),
        };

        self.state_manager.update_tension(global_delta);
        self.state_manager.apply_influence(
            &action.country,
            influence_actor,
            action.target.as_ref(),
            influence_target,
        );

        if let Some(ref target) = action.target {
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
            "Action applied: [{}] {} → {} (tension Δ{}, influence {:+})",
            action.tier.as_str(),
            action.country.0,
            action.target.as_ref().map(|t| t.0.as_str()).unwrap_or("global"),
            global_delta,
            influence_actor,
        );
    }
}
