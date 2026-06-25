use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub general: GeneralConfig,
    pub translation: TranslationConfig,
    pub run_protocol: RunProtocolConfig,
    pub pareto_sweep: ParetoSweepConfig,
    pub throughput: ThroughputConfig,
    pub network_calibration: NetworkCalibrationConfig,
    pub power_analysis: PowerAnalysisConfig,
    pub scales: ScaleConfig,
    pub difficulties: DifficultyConfig,
    pub databases: DatabaseConfig,
    pub experiments: ExperimentConfig,
    pub real_datasets: RealDatasetConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub random_seed: u64,
    pub vector_dimension: usize,
    pub warmup_runs: usize,
    pub bootstrap_resamples: usize,
    pub confidence_level: f64,
    pub data_dir: String,
    pub output_dir: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            random_seed: 42,
            vector_dimension: 128,
            warmup_runs: 5,
            bootstrap_resamples: 10_000,
            confidence_level: 0.95,
            data_dir: "datasets/".into(),
            output_dir: "results/".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    pub primary_strategy: String,
    pub sensitivity_strategies: Vec<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            primary_strategy: "SemanticOnly".into(),
            sensitivity_strategies: vec!["Concatenation".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunProtocolConfig {
    pub tier1_max_scale: usize,
    pub tier2_max_scale: usize,
    pub tier1_full_runs: usize,
    pub tier2_index_builds: usize,
    pub tier2_queries_per_build: usize,
    pub tier3_query_runs: usize,
}

impl Default for RunProtocolConfig {
    fn default() -> Self {
        Self {
            tier1_max_scale: 100_000,
            tier2_max_scale: 1_000_000,
            tier1_full_runs: 30,
            tier2_index_builds: 5,
            tier2_queries_per_build: 6,
            tier3_query_runs: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RunProtocol {
    Full { num_runs: usize },
    Split { num_index_builds: usize, query_runs_per_build: usize },
    QueryOnly { num_query_runs: usize },
}

impl RunProtocol {
    pub fn for_scale(n_docs: usize, config: &RunProtocolConfig) -> Self {
        match n_docs {
            0..=100_000 => Self::Full { num_runs: config.tier1_full_runs },
            100_001..=1_000_000 => Self::Split {
                num_index_builds: config.tier2_index_builds,
                query_runs_per_build: config.tier2_queries_per_build,
            },
            _ => Self::QueryOnly { num_query_runs: config.tier3_query_runs },
        }
    }

    pub fn total_query_runs(&self) -> usize {
        match self {
            Self::Full { num_runs } => *num_runs,
            Self::Split { num_index_builds, query_runs_per_build } => {
                num_index_builds * query_runs_per_build
            }
            Self::QueryOnly { num_query_runs } => *num_query_runs,
        }
    }

    pub fn justification(&self) -> &'static str {
        match self {
            Self::Full { .. } =>
                "Full teardown-rebuild protocol: all runs are statistically independent.",
            Self::Split { .. } =>
                "Split protocol: multiple independent index builds with multiple query sessions each.",
            Self::QueryOnly { .. } =>
                "Query-only protocol: single index build, multiple independent query sessions.",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSweepConfig {
    pub ef_search_values: Vec<usize>,
    pub queries_per_point: usize,
    pub reps_per_point: usize,
    pub target_recalls: Vec<f64>,
    pub concurrency: usize,
}

impl Default for ParetoSweepConfig {
    fn default() -> Self {
        Self {
            ef_search_values: vec![10, 20, 40, 80, 100, 150, 200, 400, 800],
            queries_per_point: 1000,
            reps_per_point: 5,
            target_recalls: vec![0.90, 0.95, 0.99],
            concurrency: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputConfig {
    pub concurrency_levels: Vec<usize>,
    pub warmup_duration_secs: u64,
    pub measurement_duration_secs: u64,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            concurrency_levels: vec![1, 2, 4, 8, 16, 32, 64],
            warmup_duration_secs: 10,
            measurement_duration_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCalibrationConfig {
    pub enabled: bool,
    pub num_null_queries: usize,
}

impl Default for NetworkCalibrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_null_queries: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysisConfig {
    pub effect_sizes: Vec<f64>,
    pub desired_power: f64,
    pub estimated_num_comparisons: usize,
}

impl Default for PowerAnalysisConfig {
    fn default() -> Self {
        Self {
            effect_sizes: vec![0.2, 0.5, 0.8],
            desired_power: 0.80,
            estimated_num_comparisons: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleConfig {
    pub values: Vec<usize>,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            values: vec![2_000, 10_000, 25_000, 50_000, 100_000],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyConfig {
    pub values: Vec<String>,
}

impl Default for DifficultyConfig {
    fn default() -> Self {
        Self {
            values: vec!["Easy".into(), "Medium".into(), "Hard".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    #[serde(default)]
    pub attentiondb_cpu: DatabaseEntry,
    #[serde(default)]
    pub attentiondb_gpu: DatabaseEntry,
    #[serde(default)]
    pub qdrant: DatabaseEntry,
    #[serde(default)]
    pub milvus: DatabaseEntry,
    #[serde(default)]
    pub weaviate: DatabaseEntry,
    #[serde(default)]
    pub pgvector: DatabaseEntry,
    #[serde(default)]
    pub elasticsearch: DatabaseEntry,
    #[serde(default)]
    pub pinecone: DatabaseEntry,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            attentiondb_cpu: DatabaseEntry { enabled: true, host: "localhost".into(), port: 7070, transport: "LocalGrpc".into(), note: None, requires_cuda: Some(false) },
            attentiondb_gpu: DatabaseEntry { enabled: false, host: "localhost".into(), port: 7071, transport: "LocalGrpc".into(), note: None, requires_cuda: Some(true) },
            qdrant: DatabaseEntry { enabled: true, host: "localhost".into(), port: 6334, transport: "LocalGrpc".into(), note: None, requires_cuda: None },
            milvus: DatabaseEntry { enabled: false, host: "localhost".into(), port: 19530, transport: "LocalGrpc".into(), note: None, requires_cuda: None },
            weaviate: DatabaseEntry { enabled: false, host: "localhost".into(), port: 8080, transport: "LocalRest".into(), note: None, requires_cuda: None },
            pgvector: DatabaseEntry { enabled: true, host: "localhost".into(), port: 5432, transport: "LocalTcp".into(), note: None, requires_cuda: None },
            elasticsearch: DatabaseEntry { enabled: false, host: "localhost".into(), port: 9200, transport: "LocalRest".into(), note: None, requires_cuda: None },
            pinecone: DatabaseEntry { enabled: false, host: String::new(), port: 0, transport: "Cloud".into(), note: Some("Cloud latency is not comparable to local systems".into()), requires_cuda: None },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatabaseEntry {
    pub enabled: bool,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub port: u16,
    #[serde(default)]
    pub transport: String,
    pub note: Option<String>,
    pub requires_cuda: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    #[serde(default)]
    pub run_standard: bool,
    #[serde(default)]
    pub run_head_ablations: bool,
    #[serde(default)]
    pub run_failure_modes: bool,
    #[serde(default)]
    pub run_ann_benchmark_compat: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            run_standard: true,
            run_head_ablations: true,
            run_failure_modes: true,
            run_ann_benchmark_compat: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealDatasetConfig {
    #[serde(default)]
    pub sift128: RealDatasetEntry,
    #[serde(default)]
    pub glove100: RealDatasetEntry,
    #[serde(default)]
    pub nytimes256: RealDatasetEntry,
    #[serde(default)]
    pub gist960: RealDatasetEntry,
}

impl Default for RealDatasetConfig {
    fn default() -> Self {
        Self {
            sift128: RealDatasetEntry { enabled: true, path: "datasets/sift-128-euclidean.hdf5".into(), note: None },
            glove100: RealDatasetEntry { enabled: true, path: "datasets/glove-100-angular.hdf5".into(), note: None },
            nytimes256: RealDatasetEntry { enabled: true, path: "datasets/nytimes-256-angular.hdf5".into(), note: None },
            gist960: RealDatasetEntry { enabled: false, path: "datasets/gist-960-euclidean.hdf5".into(), note: Some("960-dim is expensive; enable for camera-ready".into()) },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RealDatasetEntry {
    pub enabled: bool,
    #[serde(default)]
    pub path: String,
    pub note: Option<String>,
}
