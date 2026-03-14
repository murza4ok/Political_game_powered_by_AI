use anyhow::{Context, Result};
use chrono::Utc;
use geopolitical_sim::config::LLMConfig;
use geopolitical_sim::llm::{LLMClient, YandexLLMClient};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;

// ── Data structures ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ActionEntry {
    agent: String,
    tier: String,
    target: Option<String>,
    description: String,
}

#[derive(Debug, Clone)]
struct TurnSnapshot {
    turn: u32,
    period: String,
    tension: u8,
    influence: HashMap<String, i32>,
    stability: HashMap<String, u8>,
    actions: Vec<ActionEntry>,
}

#[derive(Debug, Clone, PartialEq)]
enum EndCondition {
    NuclearStrike,
    MaxTurns,
    RateLimit,
    PlayerStop,
    Unknown,
}

impl EndCondition {
    fn label(&self) -> &'static str {
        match self {
            EndCondition::NuclearStrike => "Ядерный удар",
            EndCondition::MaxTurns => "Максимум ходов",
            EndCondition::RateLimit => "Лимит API",
            EndCondition::PlayerStop => "Остановлено игроком",
            EndCondition::Unknown => "Неизвестно",
        }
    }
}

#[derive(Debug, Clone)]
struct SessionData {
    filename: String,
    turns: Vec<TurnSnapshot>,
    end_condition: EndCondition,
    final_turn: u32,
}

// ── UX helpers ─────────────────────────────────────────────────────────────────

fn print_banner() {
    println!(
        r"
╔══════════════════════════════════════════════════════════╗
║       АНАЛИЗАТОР ГЕОПОЛИТИЧЕСКОЙ СИМУЛЯЦИИ               ║
║  • Итоговая таблица и графики по каждой сессии           ║
║  • AI-анализ от DeepSeek и Yandex                        ║
║  • Сравнительный анализ при выборе нескольких сессий     ║
╚══════════════════════════════════════════════════════════╝"
    );
}

fn read_line() -> String {
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line).unwrap_or(0);
    line.trim().to_string()
}

// ── Session listing ────────────────────────────────────────────────────────────

fn list_sessions() -> Result<Vec<(String, String, u64)>> {
    // Returns Vec<(filename, display_name, size_bytes)>
    let results_dir = Path::new("results");
    if !results_dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions: Vec<(String, String, u64)> = fs::read_dir(results_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".jsonl") && name.starts_with("session_") {
                let size = entry.metadata().ok()?.len();
                let display = format!("{} ({:.1} KB)", name, size as f64 / 1024.0);
                Some((name, display, size))
            } else {
                None
            }
        })
        .collect();

    sessions.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
    Ok(sessions)
}

// ── JSONL parsing ──────────────────────────────────────────────────────────────

fn parse_session(path: &str) -> Result<SessionData> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path))?;

    let filename = Path::new(path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut turn_map: HashMap<u32, TurnSnapshot> = HashMap::new();
    let mut end_condition = EndCondition::Unknown;
    let mut max_turn_seen: u32 = 0;

    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .with_context(|| format!("JSON parse error at line {}", lineno + 1))?;

        let event_type = v["type"].as_str().unwrap_or("");

        match event_type {
            "turn_start" => {
                let turn = v["turn"].as_u64().unwrap_or(0) as u32;
                let period = v["period"].as_str().unwrap_or("").to_string();
                let tension = v["tension_before"].as_u64().unwrap_or(0) as u8;
                max_turn_seen = max_turn_seen.max(turn);

                turn_map.entry(turn).or_insert_with(|| TurnSnapshot {
                    turn,
                    period,
                    tension,
                    influence: HashMap::new(),
                    stability: HashMap::new(),
                    actions: Vec::new(),
                });
            }
            "turn_end" => {
                let turn = v["turn"].as_u64().unwrap_or(0) as u32;
                let tension = v["tension_after"].as_u64().unwrap_or(0) as u8;
                let period = v["period"].as_str().unwrap_or("").to_string();

                let snap = turn_map.entry(turn).or_insert_with(|| TurnSnapshot {
                    turn,
                    period: period.clone(),
                    tension,
                    influence: HashMap::new(),
                    stability: HashMap::new(),
                    actions: Vec::new(),
                });

                snap.tension = tension;
                if !period.is_empty() {
                    snap.period = period;
                }

                if let Some(scores) = v["influence_scores"].as_object() {
                    for (k, val) in scores {
                        snap.influence
                            .insert(k.clone(), val.as_i64().unwrap_or(0) as i32);
                    }
                }
                if let Some(stabs) = v["stability"].as_object() {
                    for (k, val) in stabs {
                        snap.stability
                            .insert(k.clone(), val.as_u64().unwrap_or(70) as u8);
                    }
                }
            }
            "action_applied" => {
                let turn = v["turn"].as_u64().unwrap_or(0) as u32;
                let agent = v["agent"].as_str().unwrap_or("").to_string();
                let tier = v["tier"].as_str().unwrap_or("").to_string();
                let target = v["target"].as_str().map(|s| s.to_string());
                let description = v["description"].as_str().unwrap_or("").to_string();

                let snap = turn_map.entry(turn).or_insert_with(|| TurnSnapshot {
                    turn,
                    period: String::new(),
                    tension: 0,
                    influence: HashMap::new(),
                    stability: HashMap::new(),
                    actions: Vec::new(),
                });

                snap.actions.push(ActionEntry {
                    agent,
                    tier,
                    target,
                    description,
                });
            }
            "simulation_end" => {
                let reason = v["reason"].as_str().unwrap_or("");
                end_condition = match reason {
                    "nuclear_strike" => EndCondition::NuclearStrike,
                    _ => EndCondition::Unknown,
                };
            }
            "rate_limit_stop" => {
                end_condition = EndCondition::RateLimit;
            }
            _ => {}
        }
    }

    // Sort turns
    let mut turns: Vec<TurnSnapshot> = turn_map.into_values().collect();
    turns.sort_by_key(|t| t.turn);

    // If no explicit simulation_end event, infer from context
    if end_condition == EndCondition::Unknown && !turns.is_empty() {
        // Check if it ended at max_turns (default 50)
        end_condition = if max_turn_seen >= 49 {
            EndCondition::MaxTurns
        } else {
            EndCondition::PlayerStop
        };
    }

    let final_turn = turns.last().map(|t| t.turn).unwrap_or(0);

    Ok(SessionData {
        filename,
        turns,
        end_condition,
        final_turn,
    })
}

// ── Compact LLM summary ────────────────────────────────────────────────────────

fn build_compact_summary(session: &SessionData) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Сессия: {}, {} ходов, конец: {}",
        session.filename,
        session.turns.len(),
        session.end_condition.label()
    ));
    lines.push(String::new());

    for snap in &session.turns {
        // Actions line
        let action_parts: Vec<String> = snap
            .actions
            .iter()
            .map(|a| {
                let target = a.target.as_deref().unwrap_or("global");
                format!("{}: {}→{}", a.agent, a.tier, target)
            })
            .collect();

        let actions_str = if action_parts.is_empty() {
            "нет действий".to_string()
        } else {
            action_parts.join(" | ")
        };

        // Scores line
        let mut score_parts: Vec<String> = snap
            .influence
            .iter()
            .map(|(k, v)| format!("{}={:+}", k, v))
            .collect();
        score_parts.sort();
        let scores_str = score_parts.join(", ");

        lines.push(format!(
            "Ход {:2} | {} | {} | Напряжённость: {}",
            snap.turn, actions_str, scores_str, snap.tension
        ));
    }

    // Final state
    if let Some(last) = session.turns.last() {
        lines.push(String::new());
        let mut final_parts: Vec<String> = last
            .influence
            .iter()
            .map(|(k, v)| format!("{}={:+}", k, v))
            .collect();
        final_parts.sort();

        let mut stab_parts: Vec<String> = last
            .stability
            .iter()
            .map(|(k, v)| format!("{}={}%", k, v))
            .collect();
        stab_parts.sort();

        lines.push(format!(
            "Финал: {} | Стабильность: {}",
            final_parts.join(", "),
            stab_parts.join(", ")
        ));
    }

    lines.join("\n")
}

// ── LLM calls ─────────────────────────────────────────────────────────────────

async fn call_llm_analysis(summary: &str, llm_config: &LLMConfig) -> (String, String) {
    let prompt = format!(
        "Сделай анализ этой игровой сессии политической симуляции:\n\n{}",
        summary
    );
    let system = "Ты эксперт по геополитике и стратегическим играм. Анализируй ходы, стратегии держав, критические решения и итоги симуляции. Отвечай на русском языке.";

    let deepseek_future = async {
        match LLMClient::new(llm_config.clone()) {
            Ok(client) => client
                .generate_content(system, &prompt)
                .await
                .unwrap_or_else(|e| format!("Ошибка DeepSeek: {}", e)),
            Err(e) => format!("Не удалось создать DeepSeek клиент: {}", e),
        }
    };

    let yandex_future = async {
        match YandexLLMClient::new(llm_config) {
            Ok(client) => client
                .generate_content(system, &prompt)
                .await
                .unwrap_or_else(|e| format!("Ошибка Yandex: {}", e)),
            Err(e) => format!("Не удалось создать Yandex клиент: {}", e),
        }
    };

    tokio::join!(deepseek_future, yandex_future)
}

// ── HTML generation ────────────────────────────────────────────────────────────

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn render_session_section(session: &SessionData, deepseek_analysis: &str, yandex_analysis: &str) -> String {
    let countries = ["Russia", "China", "USA"];

    // Data arrays for charts
    let turn_labels: Vec<String> = session.turns.iter().map(|t| t.turn.to_string()).collect();
    let labels_json = serde_json::to_string(&turn_labels).unwrap_or_default();

    // Influence per country over turns
    let mut influence_datasets = Vec::new();
    let colors = [
        ("Russia", "#e74c3c", "#c0392b"),
        ("China", "#f39c12", "#d68910"),
        ("USA", "#3498db", "#2980b9"),
    ];
    for (country, color, border) in &colors {
        let data: Vec<i32> = session
            .turns
            .iter()
            .map(|t| t.influence.get(*country).copied().unwrap_or(0))
            .collect();
        let data_json = serde_json::to_string(&data).unwrap_or_default();
        influence_datasets.push(format!(
            r#"{{label:"{country}",data:{data_json},borderColor:"{border}",backgroundColor:"{color}33",fill:false,tension:0.3}}"#
        ));
    }

    // Tension over turns
    let tension_data: Vec<u8> = session.turns.iter().map(|t| t.tension).collect();
    let tension_json = serde_json::to_string(&tension_data).unwrap_or_default();

    // Action tier counts per country
    let mut tier_counts: HashMap<&str, HashMap<String, u32>> = HashMap::new();
    for country in &countries {
        tier_counts.insert(country, HashMap::new());
    }
    for snap in &session.turns {
        for action in &snap.actions {
            let entry = tier_counts
                .entry(action.agent.as_str())
                .or_default()
                .entry(action.tier.clone())
                .or_insert(0);
            *entry += 1;
        }
    }

    let tiers = ["Diplomatic", "Economic", "ConventionalMilitary", "StrategicMilitary", "Nuclear"];
    let tier_colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#8e44ad"];
    let mut tier_datasets = Vec::new();
    for (tier, tcolor) in tiers.iter().zip(tier_colors.iter()) {
        let data: Vec<u32> = countries
            .iter()
            .map(|c| {
                tier_counts
                    .get(c)
                    .and_then(|m| m.get(*tier))
                    .copied()
                    .unwrap_or(0)
            })
            .collect();
        let data_json = serde_json::to_string(&data).unwrap_or_default();
        tier_datasets.push(format!(
            r#"{{label:"{tier}",data:{data_json},backgroundColor:"{tcolor}"}}"#
        ));
    }

    // Final scores table
    let last = session.turns.last();
    let final_rows: String = countries
        .iter()
        .map(|c| {
            let score = last
                .and_then(|t| t.influence.get(*c))
                .copied()
                .unwrap_or(0);
            let stab = last
                .and_then(|t| t.stability.get(*c))
                .copied()
                .unwrap_or(70);
            let winner_class = if score == last.map(|t| *t.influence.values().max().unwrap_or(&0)).unwrap_or(0) {
                " style=\"font-weight:bold;background:#ffffcc\""
            } else {
                ""
            };
            format!(
                "<tr{winner_class}><td>{c}</td><td>{score:+}</td><td>{stab}%</td></tr>"
            )
        })
        .collect();

    let chart_id = session.filename.replace('.', "_").replace('-', "_");

    format!(
        r#"<section class="session-section">
  <h2>Сессия: {filename}</h2>
  <p><strong>Ходов:</strong> {turns} &nbsp;|&nbsp;
     <strong>Конец:</strong> {end} &nbsp;|&nbsp;
     <strong>Финальный ход:</strong> {final_turn}</p>

  <h3>Итоговые показатели</h3>
  <table>
    <thead><tr><th>Держава</th><th>Очки влияния</th><th>Стабильность</th></tr></thead>
    <tbody>{final_rows}</tbody>
  </table>

  <div class="charts-grid">
    <div class="chart-box">
      <h3>Очки влияния по ходам</h3>
      <canvas id="inf_{chart_id}"></canvas>
    </div>
    <div class="chart-box">
      <h3>Глобальная напряжённость</h3>
      <canvas id="ten_{chart_id}"></canvas>
    </div>
    <div class="chart-box">
      <h3>Действия по тиру</h3>
      <canvas id="act_{chart_id}"></canvas>
    </div>
  </div>

  <script>
  (function(){{
    var labels = {labels_json};
    new Chart(document.getElementById('inf_{chart_id}'), {{
      type:'line', data:{{labels:labels, datasets:[{influence_ds}]}},
      options:{{responsive:true, plugins:{{legend:{{position:'top'}}}},
               scales:{{y:{{title:{{display:true,text:'Очки влияния'}}}}}}}}
    }});
    new Chart(document.getElementById('ten_{chart_id}'), {{
      type:'line',
      data:{{labels:labels, datasets:[{{label:'Напряжённость',data:{tension_json},
             borderColor:'#e74c3c',backgroundColor:'#e74c3c33',fill:true,tension:0.3}}]}},
      options:{{responsive:true, scales:{{y:{{min:0,max:100,
               title:{{display:true,text:'Уровень напряжённости'}}}}}}}}
    }});
    new Chart(document.getElementById('act_{chart_id}'), {{
      type:'bar',
      data:{{labels:{country_labels}, datasets:[{tier_ds}]}},
      options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},
               scales:{{x:{{stacked:true}},y:{{stacked:true,
               title:{{display:true,text:'Количество действий'}}}}}}}}
    }});
  }})();
  </script>

  <div class="ai-block">
    <h3>Анализ DeepSeek</h3>
    <div class="ai-text">{deepseek_analysis}</div>
  </div>
  <div class="ai-block yandex">
    <h3>Анализ Yandex</h3>
    <div class="ai-text">{yandex_analysis}</div>
  </div>
</section>"#,
        filename = html_escape(&session.filename),
        turns = session.turns.len(),
        end = session.end_condition.label(),
        final_turn = session.final_turn,
        chart_id = chart_id,
        labels_json = labels_json,
        influence_ds = influence_datasets.join(","),
        tension_json = tension_json,
        country_labels = serde_json::to_string(&countries).unwrap_or_default(),
        tier_ds = tier_datasets.join(","),
        deepseek_analysis = html_escape(deepseek_analysis).replace('\n', "<br>"),
        yandex_analysis = html_escape(yandex_analysis).replace('\n', "<br>"),
    )
}

fn render_comparison_section(sessions: &[SessionData]) -> String {
    let rows: String = sessions
        .iter()
        .map(|s| {
            let last = s.turns.last();
            let winner = last
                .and_then(|t| {
                    t.influence
                        .iter()
                        .max_by_key(|(_, v)| *v)
                        .map(|(k, _)| k.as_str())
                })
                .unwrap_or("?");

            let scores: Vec<String> = ["Russia", "China", "USA"]
                .iter()
                .map(|c| {
                    let v = last
                        .and_then(|t| t.influence.get(*c))
                        .copied()
                        .unwrap_or(0);
                    format!("{}: {:+}", c, v)
                })
                .collect();

            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                html_escape(&s.filename),
                winner,
                scores.join(", "),
                s.turns.len(),
                s.end_condition.label()
            )
        })
        .collect();

    format!(
        r#"<section class="session-section comparison">
  <h2>Сравнительный анализ сессий</h2>
  <table>
    <thead>
      <tr>
        <th>Сессия</th><th>Победитель</th><th>Финальные очки</th>
        <th>Ходов</th><th>Условие конца</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>"#
    )
}

fn render_html(sessions: &[SessionData], session_analyses: &[(String, String)], include_comparison: bool) -> String {
    let now = Utc::now().format("%Y-%m-%d %H:%M UTC");

    let mut body = String::new();
    for (i, session) in sessions.iter().enumerate() {
        let (deepseek, yandex) = &session_analyses[i];
        body.push_str(&render_session_section(session, deepseek, yandex));
    }

    if include_comparison && sessions.len() > 1 {
        body.push_str(&render_comparison_section(sessions));
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Анализ геополитической симуляции</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #fff; text-align: center; margin: 20px 0 4px; font-size: 1.8rem; }}
  .subtitle {{ text-align: center; color: #aaa; margin-bottom: 30px; }}
  .session-section {{ background: #16213e; border-radius: 12px; padding: 24px; margin-bottom: 30px; border: 1px solid #0f3460; }}
  .session-section.comparison {{ background: #0d1b2a; }}
  h2 {{ color: #e94560; margin-bottom: 12px; font-size: 1.3rem; }}
  h3 {{ color: #a8dadc; margin: 16px 0 8px; font-size: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ background: #0f3460; color: #e0e0e0; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #2a2a4a; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #1e2d4a; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
  .chart-box {{ background: #0d1b2a; border-radius: 8px; padding: 16px; }}
  .chart-box canvas {{ max-height: 260px; }}
  .ai-block {{ background: #0a0a1a; border-left: 4px solid #e94560; border-radius: 6px; padding: 16px; margin: 16px 0; }}
  .ai-block.yandex {{ border-left-color: #f39c12; }}
  .ai-text {{ line-height: 1.7; font-size: 0.92rem; color: #ccc; white-space: pre-wrap; word-wrap: break-word; }}
  p {{ color: #bbb; margin: 6px 0; font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>Анализ геополитической симуляции</h1>
<p class="subtitle">Сгенерировано {now}</p>
{body}
</body>
</html>"#
    )
}

// ── CSV generation ─────────────────────────────────────────────────────────────

fn render_csv(sessions: &[SessionData]) -> String {
    let header = "session,turn,period,tension,Russia_influence,China_influence,USA_influence,\
Russia_stability,China_stability,USA_stability,\
Russia_action_tier,Russia_action_target,\
China_action_tier,China_action_target,\
USA_action_tier,USA_action_target,end_condition";

    let mut rows = vec![header.to_string()];

    for session in sessions {
        for snap in &session.turns {
            // Find first action per country
            let mut country_actions: HashMap<&str, (&str, Option<&str>)> = HashMap::new();
            for action in &snap.actions {
                country_actions
                    .entry(action.agent.as_str())
                    .or_insert((action.tier.as_str(), action.target.as_deref()));
            }

            let get = |c: &str| {
                country_actions
                    .get(c)
                    .map(|(t, tgt)| {
                        format!("{},{}", t, tgt.unwrap_or(""))
                    })
                    .unwrap_or_else(|| ",".to_string())
            };

            let row = format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                session.filename,
                snap.turn,
                snap.period.replace(',', ";"),
                snap.tension,
                snap.influence.get("Russia").copied().unwrap_or(0),
                snap.influence.get("China").copied().unwrap_or(0),
                snap.influence.get("USA").copied().unwrap_or(0),
                snap.stability.get("Russia").copied().unwrap_or(70),
                snap.stability.get("China").copied().unwrap_or(70),
                snap.stability.get("USA").copied().unwrap_or(70),
                get("Russia"),
                get("China"),
                get("USA"),
                session.end_condition.label(),
            );
            rows.push(row);
        }
    }

    rows.join("\n")
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    print_banner();

    // List available sessions
    let sessions_list = list_sessions()?;

    if sessions_list.is_empty() {
        eprintln!("✗ Ошибка: в папке results/ не найдено файлов сессий (*.jsonl)");
        return Ok(());
    }

    println!("\nДоступные сессии в results/:");
    for (i, (_, display, _)) in sessions_list.iter().enumerate() {
        println!("  [{}] {}", i + 1, display);
    }

    println!();
    print!(r#"По каким сессиям провести анализ? (номера через запятую, например: 1,3 или "all"): "#);
    let _ = io::stdout().flush();
    let input = read_line();

    let selected_indices: Vec<usize> = if input.trim().eq_ignore_ascii_case("all") {
        (0..sessions_list.len()).collect()
    } else {
        let mut indices = Vec::new();
        for part in input.split(',') {
            let part = part.trim();
            match part.parse::<usize>() {
                Ok(n) if n >= 1 && n <= sessions_list.len() => indices.push(n - 1),
                Ok(_) => {
                    eprintln!("✗ Ошибка: номер {} вне диапазона 1–{}", part, sessions_list.len());
                    return Ok(());
                }
                Err(_) => {
                    eprintln!("✗ Ошибка: некорректный ввод '{}'", part);
                    return Ok(());
                }
            }
        }
        indices
    };

    if selected_indices.is_empty() {
        eprintln!("✗ Ошибка: не выбрано ни одной сессии");
        return Ok(());
    }

    let include_comparison = selected_indices.len() > 1;

    // Ask about comparison if multiple
    if include_comparison {
        print!("Включить сравнительный анализ? [Y/n]: ");
        let _ = io::stdout().flush();
        let cmp_answer = read_line();
        let _include_comparison = !cmp_answer.eq_ignore_ascii_case("n");
    }

    // Parse selected sessions
    let mut sessions: Vec<SessionData> = Vec::new();
    for idx in &selected_indices {
        let filename = &sessions_list[*idx].0;
        let path = format!("results/{}", filename);
        let session = parse_session(&path)
            .with_context(|| format!("Ошибка разбора файла {}", filename))?;
        sessions.push(session);
    }

    // Build LLM config from env (use deepseek config values)
    let llm_config = LLMConfig {
        provider: "deepseek".to_string(),
        model: std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string()),
        temperature: 0.7,
        api_url: Some("https://api.deepseek.com/chat/completions".to_string()),
    };

    // Run LLM analysis for each session
    let mut session_analyses: Vec<(String, String)> = Vec::new();
    for session in &sessions {
        let summary = build_compact_summary(session);
        let (deepseek_text, yandex_text) = call_llm_analysis(&summary, &llm_config).await;
        session_analyses.push((deepseek_text, yandex_text));
    }

    // Generate output files
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let html_path = format!("results/report_{}.html", timestamp);
    let csv_path = format!("results/report_{}.csv", timestamp);

    let html_content = render_html(&sessions, &session_analyses, include_comparison);
    fs::write(&html_path, html_content)
        .with_context(|| format!("Ошибка записи {}", html_path))?;

    let csv_content = render_csv(&sessions);
    fs::write(&csv_path, "\u{FEFF}".to_owned() + &csv_content) // BOM for Excel UTF-8
        .with_context(|| format!("Ошибка записи {}", csv_path))?;

    println!(
        "\n✓ Анализ завершён:\n  HTML: {}\n  CSV:  {}",
        html_path, csv_path
    );

    Ok(())
}
