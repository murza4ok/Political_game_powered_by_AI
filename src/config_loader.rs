use config::{Config, File, FileFormat};
use crate::config::{CountryConfig, SimulationConfig};
use anyhow::{bail, Result};

pub struct ConfigLoader {
    pub simulation: SimulationConfig,
    pub countries: Vec<CountryConfig>,
}

impl ConfigLoader {
    pub fn load(config_path: &str, countries_dir: &str) -> Result<Self> {
        let simulation = Config::builder()
            .add_source(File::with_name(config_path).format(FileFormat::Toml))
            .build()?
            .try_deserialize()?;

        let mut countries = Vec::new();
        for entry in std::fs::read_dir(countries_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                let country: CountryConfig = Config::builder()
                    .add_source(File::from(path).format(FileFormat::Toml))
                    .build()?
                    .try_deserialize()?;
                validate_country(&country)?;
                countries.push(country);
            }
        }

        Ok(Self { simulation, countries })
    }
}

fn validate_country(c: &CountryConfig) -> Result<()> {
    let name = &c.id.0;

    if !(0.0..=1.0).contains(&c.policy.aggression_weight) {
        bail!(
            "Country '{}': aggression_weight {} is outside [0.0, 1.0]",
            name, c.policy.aggression_weight
        );
    }
    if !(0.0..=1.0).contains(&c.policy.cooperation_weight) {
        bail!(
            "Country '{}': cooperation_weight {} is outside [0.0, 1.0]",
            name, c.policy.cooperation_weight
        );
    }
    if !(0.0..=1.0).contains(&c.policy.risk_tolerance) {
        bail!(
            "Country '{}': risk_tolerance {} is outside [0.0, 1.0]",
            name, c.policy.risk_tolerance
        );
    }

    let weight_sum = c.policy.aggression_weight + c.policy.cooperation_weight;
    if (weight_sum - 1.0f32).abs() > 0.05 {
        bail!(
            "Country '{}': aggression_weight + cooperation_weight = {:.3} (should sum to ~1.0)",
            name, weight_sum
        );
    }

    Ok(())
}
