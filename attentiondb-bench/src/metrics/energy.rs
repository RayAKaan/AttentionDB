use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyMeasurement {
    pub total_queries: usize,
    pub total_energy_joules: f64,
    pub energy_per_query_mj: f64,
    pub measurement_method: EnergyMethod,
    pub available: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EnergyMethod {
    IntelRapl,
    NvidiaSmi,
    NotAvailable,
}

pub struct EnergyMonitor;

impl EnergyMonitor {
    #[cfg(target_os = "linux")]
    pub fn read_rapl_uj() -> Option<u64> {
        std::fs::read_to_string(
            "/sys/class/powercap/intel-rapl:0/energy_uj"
        )
        .ok()
        .and_then(|s| s.trim().parse().ok())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn read_rapl_uj() -> Option<u64> {
        None
    }

    pub fn measure_energy<F, T>(
        f: F,
        num_queries: usize,
    ) -> (T, EnergyMeasurement)
    where
        F: FnOnce() -> T,
    {
        let start_uj = Self::read_rapl_uj();
        let result = f();
        let end_uj = Self::read_rapl_uj();

        match (start_uj, end_uj) {
            (Some(start), Some(end)) => {
                let delta_uj = if end >= start {
                    (end - start) as f64
                } else {
                    (u32::MAX as u64 - start + end) as f64
                };

                let total_joules = delta_uj / 1_000_000.0;
                let energy_mj = if num_queries > 0 {
                    (total_joules * 1000.0) / num_queries as f64
                } else { 0.0 };

                (result, EnergyMeasurement {
                    total_queries: num_queries,
                    total_energy_joules: total_joules,
                    energy_per_query_mj: energy_mj,
                    measurement_method: EnergyMethod::IntelRapl,
                    available: true,
                })
            }
            _ => {
                (result, EnergyMeasurement {
                    total_queries: num_queries,
                    total_energy_joules: 0.0,
                    energy_per_query_mj: 0.0,
                    measurement_method: EnergyMethod::NotAvailable,
                    available: false,
                })
            }
        }
    }

    #[cfg(target_os = "linux")]
    pub fn read_gpu_energy_joules() -> Option<f64> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        let watts: f64 = String::from_utf8(output.stdout).ok()?
            .trim().parse().ok()?;
        Some(watts)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn read_gpu_energy_joules() -> Option<f64> {
        None
    }
}
