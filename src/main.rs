use geopolitical_sim::agents::Agent;
use geopolitical_sim::config_loader::ConfigLoader;
use geopolitical_sim::guardrails::Guardrails;
use geopolitical_sim::llm::{AnyLLMClient, LLMClient, YandexLLMClient};
use geopolitical_sim::orchestrator::Orchestrator;
use anyhow::Result;
use std::io::{self, BufRead, Write};
use tracing_subscriber;

fn print_banner() {
    println!(r"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       Г Е О П О Л И Т И Ч Е С К А Я   С И М У Л Я Ц И Я                   ║
║                                                                              ║
║             США  •  Россия  •  Китай                                        ║
║                                                                              ║
║       Многоагентная симуляция на основе LLM                                 ║
║       Каждая держава управляется независимым ИИ-агентом                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
");
}

fn read_line_sync() -> String {
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line).unwrap_or(0);
    line.trim().to_string()
}

/// Показывает начальный сценарий и предлагает игроку добавить дополнение.
/// Возвращает (возможно дополненный) текст сценария.
fn run_pregame_flow(scenario_desc: &str) -> String {
    println!("=== НАЧАЛЬНЫЙ СЦЕНАРИЙ ===\n");
    println!("{}\n", scenario_desc.trim());
    println!("══════════════════════════════════════════════════════════════════════════════");
    print!("Хотите добавить что-то к сценарию? [y/N]: ");
    let _ = io::stdout().flush();

    let answer = read_line_sync();

    if answer.eq_ignore_ascii_case("y") {
        println!("Введите дополнение к сценарию:");
        print!("> ");
        let _ = io::stdout().flush();

        let addition = read_line_sync();
        if addition.is_empty() {
            println!("Дополнение не введено — используется исходный сценарий.\n");
            return scenario_desc.to_string();
        }

        println!("\n✓ Дополнение добавлено к сценарию.\n");
        return format!("{}\n\n=== ДОПОЛНЕНИЕ ИГРОКА ===\n{}", scenario_desc.trim(), addition.trim());
    }

    println!("Начинаем симуляцию...\n");
    scenario_desc.to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    dotenv::dotenv().ok();

    // ── Phase A: загрузка конфига ──────────────────────────────────────────
    let config_loader = ConfigLoader::load("config/scenario", "config/countries")?;

    let llm_config = config_loader.simulation.llm.clone();
    let base_scenario = config_loader.simulation.initial_conditions.description.clone();

    // ── Phase B: pre-game UX ───────────────────────────────────────────────
    print_banner();
    let scenario_desc = run_pregame_flow(&base_scenario);

    // ── Phase C: создание агентов с роутингом по LLM-провайдеру ──────────
    let deepseek_client = LLMClient::new(llm_config.clone())?;
    let yandex_client = YandexLLMClient::new(&llm_config)?;

    let mut agents: Vec<Agent> = config_loader
        .countries
        .into_iter()
        .map(|country| {
            let llm: AnyLLMClient = if country.id.0 == "Russia" {
                AnyLLMClient::Yandex(yandex_client.clone())
            } else {
                AnyLLMClient::DeepSeek(deepseek_client.clone())
            };
            Agent::new(country, llm, scenario_desc.clone())
        })
        .collect();

    // Россия ходит первой
    agents.sort_by(|a, _b| {
        if a.config.id.0 == "Russia" { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
    });

    let guardrails = Guardrails::new(config_loader.simulation.guardrails.clone());

    let mut orchestrator = Orchestrator::new(
        agents,
        config_loader.simulation,
        guardrails,
    )?;

    orchestrator.run_simulation().await?;

    Ok(())
}
