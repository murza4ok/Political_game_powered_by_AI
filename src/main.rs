use geopolitical_sim::agents::Agent;
use geopolitical_sim::config_loader::ConfigLoader;
use geopolitical_sim::guardrails::Guardrails;
use geopolitical_sim::llm::LLMClient;
use geopolitical_sim::orchestrator::Orchestrator;
use anyhow::Result;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    dotenv::dotenv().ok();

    let config_loader = ConfigLoader::load("config/scenario", "config/countries")?;

    let llm_config = config_loader.simulation.llm.clone();
    let llm_client = LLMClient::new(llm_config)?;

    let agents: Vec<Agent> = config_loader
        .countries
        .into_iter()
        .map(|country| Agent::new(country, llm_client.clone()))
        .collect();

    let guardrails = Guardrails::new(config_loader.simulation.guardrails.clone());

    let mut orchestrator = Orchestrator::new(
        agents,
        config_loader.simulation,
        guardrails,
    )?;

    orchestrator.run_simulation().await?;

    Ok(())
}