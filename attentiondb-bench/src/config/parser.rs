use super::types::BenchmarkConfig;
use anyhow::Result;
use std::path::Path;

pub fn load_config(path: &Path) -> Result<BenchmarkConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: BenchmarkConfig = toml::from_str(&contents)?;
    Ok(config)
}

pub fn load_config_with_overrides(
    path: &Path,
    overrides: &[(String, String)],
) -> Result<BenchmarkConfig> {
    let mut config = load_config(path)?;
    for (key, value) in overrides {
        apply_override(&mut config, key, value)?;
    }
    Ok(config)
}

fn apply_override(
    _config: &mut BenchmarkConfig,
    key: &str,
    value: &str,
) -> Result<()> {
    match key {
        "output_dir" => {
            _config.general.output_dir = value.to_string();
        }
        "data_dir" => {
            _config.general.data_dir = value.to_string();
        }
        "random_seed" => {
            _config.general.random_seed = value.parse()?;
        }
        _ => {
            anyhow::bail!("Unknown config override key: {}", key);
        }
    }
    Ok(())
}
