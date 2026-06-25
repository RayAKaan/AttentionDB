use crate::adapters::{DatabaseAdapter, ConnectionConfig};
use crate::workload::{HeadType, DocumentId};
use crate::workload::difficulty::{DifficultyLevel, MeasuredDifficultyProperties};
use crate::workload::generator::{WorkloadGenerator, GeneratedCorpus};
use crate::workload::dataset_loader::AnnBenchmarkLoaded;
use crate::workload::ExperimentType;
use crate::config::types::{RunProtocol, BenchmarkConfig};
use crate::executor::checkpoint::{CheckpointManager, Checkpoint};
use crate::executor::warmup::WarmupManager;
use crate::metrics::quality::{QualityScorer, QualityMetrics};
use crate::metrics::latency::LatencyTracker;
use crate::metrics::energy::EnergyMeasurement;
use crate::stats::aggregator::{AggregatedMetric, aggregate_quality_metrics};
use crate::stats::confidence::percentile;
use crate::stats::hypothesis::WelchTestResult;
use crate::stats::power::{PowerAnalysisResult, validate_n_for_benchmark};
use crate::reporting::json_output::{PublicationResult, SingleRunResult, HardwareProfile};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::{Duration, Instant};

pub struct Orchestrator {
    config: BenchmarkConfig,
    rng: StdRng,
    checkpoint_manager: CheckpointManager,
}

impl Orchestrator {
    pub fn new(config: BenchmarkConfig, output_dir: &str) -> Self {
        let output_path = std::path::Path::new(output_dir);
        Self {
            rng: StdRng::seed_from_u64(config.general.random_seed),
            checkpoint_manager: CheckpointManager::new(output_path),
            config,
        }
    }

    pub async fn run_full_benchmark(&mut self) -> anyhow::Result<Vec<PublicationResult>> {
        tracing::info!("Starting full benchmark");

        let checkpoint = self.checkpoint_manager.load()?;
        let mut all_results: Vec<PublicationResult> = checkpoint.as_ref()
            .map(|c| c.partial_results.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect())
            .unwrap_or_default();
        let completed: HashSet<String> = checkpoint
            .map(|c| c.completed_experiments)
            .unwrap_or_default();

        let power_results = self.run_power_analysis()?;
        for pr in &power_results {
            tracing::info!("Power: d={:.1} requires N={}", pr.target_effect_size_d, pr.required_n);
        }

        let scales = self.config.scales.values.clone();
        let difficulties = self.config.difficulties.values.clone();

        for &scale in &scales {
            let run_protocol = RunProtocol::for_scale(scale, &self.config.run_protocol);
            tracing::info!("Scale {}: using {:?}", scale, run_protocol);

            for diff_str in &difficulties {
                let difficulty = DifficultyLevel::from_name(diff_str)
                    .ok_or_else(|| anyhow::anyhow!("Unknown difficulty: {}", diff_str))?;

                let exp_id = CheckpointManager::experiment_id(
                    "AttentionDB", "synthetic", scale, diff_str
                );
                if completed.contains(&exp_id) {
                    tracing::info!("Skipping completed experiment: {}", exp_id);
                    continue;
                }

                let result = self.run_single_experiment(
                    scale, difficulty, run_protocol,
                ).await?;

                all_results.push(result);
                self.checkpoint_manager.save(&Checkpoint {
                    completed_experiments: {
                        let mut c = completed.clone();
                        c.insert(exp_id);
                        c
                    },
                    partial_results: all_results.iter()
                        .map(|r| serde_json::to_value(r).unwrap())
                        .collect(),
                    timestamp: chrono::Utc::now(),
                    total_experiments: scales.len() * difficulties.len(),
                })?;
            }
        }

        tracing::info!("Benchmark complete. {} results collected.", all_results.len());
        Ok(all_results)
    }

    pub async fn run_single_experiment(
        &mut self,
        n_docs: usize,
        difficulty: DifficultyLevel,
        protocol: RunProtocol,
    ) -> anyhow::Result<PublicationResult> {
        let dim = self.config.general.vector_dimension;
        let n_queries = 1000.min(n_docs / 10).max(100);
        let heads = vec![HeadType::Semantic, HeadType::Temporal, HeadType::Structural];

        let mut generator = WorkloadGenerator::new(
            self.config.general.random_seed,
            dim,
        );

        let corpus = generator.generate_corpus(
            n_docs, n_queries, difficulty, heads, None,
        );

        let measured_difficulty = MeasuredDifficultyProperties::measure(
            &corpus.documents,
            &corpus.queries,
            &corpus.ground_truth,
            HeadType::Semantic,
            500,
            &mut self.rng,
        )?;

        let mut run_metrics: Vec<QualityMetrics> = Vec::new();
        let mut run_latencies: Vec<Vec<f64>> = Vec::new();
        let mut run_durations: Vec<Duration> = Vec::new();
        let mut index_build_times: Vec<f64> = Vec::new();

        let _total_runs = protocol.total_query_runs();

        let num_index_builds = match protocol {
            RunProtocol::Full { num_runs } => num_runs,
            RunProtocol::Split { num_index_builds, .. } => num_index_builds,
            RunProtocol::QueryOnly { .. } => 1,
        };

        for _build_idx in 0..num_index_builds {
            let build_config = ConnectionConfig {
                host: "localhost".into(),
                port: 7070,
                use_tls: false,
                api_key: None,
                collection_name: format!("bench_{}", uuid::Uuid::new_v4().simple()),
                transport: "LocalGrpc".into(),
            };
            let mut adapter = crate::adapters::attentiondb_cpu::AttentionDBAdapter::new();
            adapter.connect(&build_config).await?;
            adapter.setup_collection(dim).await?;

            for chunk in corpus.documents.chunks(500) {
                adapter.insert_batch(chunk).await?;
            }
            adapter.flush().await?;
            let index_time = adapter.build_index().await?;
            index_build_times.push(index_time.as_secs_f64() * 1000.0);

            WarmupManager::warmup(&adapter, dim, self.config.general.warmup_runs).await?;

            let runs_this_build = match protocol {
                RunProtocol::Full { .. } | RunProtocol::QueryOnly { .. } => 1,
                RunProtocol::Split { query_runs_per_build, .. } => query_runs_per_build,
            };

            for _ in 0..runs_this_build {
                let mut lat_tracker = LatencyTracker::new();
                let mut batch_metrics = QualityMetrics {
                    recall_at_1: 0.0, recall_at_10: 0.0, recall_at_100: 0.0,
                    mrr: 0.0, ndcg_at_10: 0.0, ndcg_at_100: 0.0, precision_at_10: 0.0,
                };

                let run_start = Instant::now();

                for (query, gt) in corpus.queries.iter().zip(corpus.ground_truth.iter()) {
                    let result = adapter.query(query, 100).await?;
                    lat_tracker.record(result.latency);

                    let qm = QualityScorer::compute_all(&result.ranked_ids, gt, 100);
                    batch_metrics.recall_at_1 += qm.recall_at_1;
                    batch_metrics.recall_at_10 += qm.recall_at_10;
                    batch_metrics.recall_at_100 += qm.recall_at_100;
                    batch_metrics.mrr += qm.mrr;
                    batch_metrics.ndcg_at_10 += qm.ndcg_at_10;
                    batch_metrics.ndcg_at_100 += qm.ndcg_at_100;
                    batch_metrics.precision_at_10 += qm.precision_at_10;
                }

                let n = corpus.queries.len() as f64;
                batch_metrics.recall_at_1 /= n;
                batch_metrics.recall_at_10 /= n;
                batch_metrics.recall_at_100 /= n;
                batch_metrics.mrr /= n;
                batch_metrics.ndcg_at_10 /= n;
                batch_metrics.ndcg_at_100 /= n;
                batch_metrics.precision_at_10 /= n;

                let run_duration = run_start.elapsed();
                run_metrics.push(batch_metrics);
                run_latencies.push(lat_tracker.all_latencies().to_vec());
                run_durations.push(run_duration);
            }

            adapter.teardown().await?;
            adapter.disconnect().await?;
        }

        let all_latencies: Vec<f64> = run_latencies.iter().flatten().copied().collect();

        let mut latency_p50_per_run = Vec::with_capacity(run_latencies.len());
        let mut latency_p90_per_run = Vec::with_capacity(run_latencies.len());
        let mut latency_p95_per_run = Vec::with_capacity(run_latencies.len());
        let mut latency_p99_per_run = Vec::with_capacity(run_latencies.len());
        let mut latency_p999_per_run = Vec::with_capacity(run_latencies.len());

        for latencies in &run_latencies {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            latency_p50_per_run.push(percentile(&sorted, 0.50));
            latency_p90_per_run.push(percentile(&sorted, 0.90));
            latency_p95_per_run.push(percentile(&sorted, 0.95));
            latency_p99_per_run.push(percentile(&sorted, 0.99));
            latency_p999_per_run.push(percentile(&sorted, 0.999));
        }

        let agg_quality = aggregate_quality_metrics(
            &run_metrics,
            self.config.general.confidence_level,
            self.config.general.bootstrap_resamples,
            &mut self.rng,
        );

        // Build statistical comparisons (placeholder: skip if only one system)
        let stats: Vec<WelchTestResult> = Vec::new();

        // Compute single-thread QPS from actual run duration
        let total_queries = run_metrics.len() as f64 * corpus.queries.len() as f64;
        let total_duration_secs = run_durations.iter().map(|d| d.as_secs_f64()).sum::<f64>();
        let qps_single_thread = if total_duration_secs > 0.0 { total_queries / total_duration_secs } else { 0.0 };

        // Build throughput profile from quality-run data
        let mean_latency = if !all_latencies.is_empty() {
            all_latencies.iter().sum::<f64>() / all_latencies.len() as f64
        } else { 0.0 };
        let mut sorted_all = all_latencies.clone();
        sorted_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tp_point = crate::metrics::throughput::ThroughputAtConcurrency {
            concurrency: 1,
            sustained_qps: qps_single_thread,
            latency_mean_ms: mean_latency,
            latency_p50_ms: if sorted_all.is_empty() { 0.0 } else { percentile(&sorted_all, 0.50) },
            latency_p95_ms: if sorted_all.is_empty() { 0.0 } else { percentile(&sorted_all, 0.95) },
            latency_p99_ms: if sorted_all.is_empty() { 0.0 } else { percentile(&sorted_all, 0.99) },
            error_rate: 0.0,
            measurement_duration_secs: run_durations.iter().map(|d| d.as_secs()).sum::<u64>(),
            total_requests: total_queries as u64,
        };
        let mut throughput_profile = crate::metrics::throughput::ThroughputProfile {
            database: "AttentionDB".into(),
            points: vec![tp_point],
            saturation_concurrency: 1,
            peak_qps: qps_single_thread,
        };
        throughput_profile.compute_saturation();

        // Build Pareto points from per-run recall/latency data
        let pareto_points: Vec<crate::metrics::pareto::ParetoPoint> = run_metrics.iter().enumerate().map(|(i, m)| {
            let lats = &run_latencies[i];
            let mean_lat = if !lats.is_empty() { lats.iter().sum::<f64>() / lats.len() as f64 } else { 0.0 };
            let mut sorted = lats.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            crate::metrics::pareto::ParetoPoint {
                param_value: 0,
                recall_at_10_mean: m.recall_at_10,
                recall_at_10_std: 0.0,
                latency_mean_ms: mean_lat,
                latency_p99_ms: if sorted.is_empty() { 0.0 } else { percentile(&sorted, 0.99) },
                qps_single_thread: if mean_lat > 0.0 { 1000.0 / mean_lat } else { 0.0 },
                qps_concurrent: 0.0,
            }
        }).collect();

        let mut pareto_result = crate::metrics::pareto::ParetoResult {
            database: "AttentionDB".into(),
            dataset: "synthetic".into(),
            scale: n_docs,
            points: pareto_points,
            qps_at_target_recall: std::collections::HashMap::new(),
            dominates: std::collections::HashMap::new(),
        };
        for &target in &self.config.pareto_sweep.target_recalls {
            let qps = pareto_result.qps_at_recall(target).unwrap_or(0.0);
            pareto_result.qps_at_target_recall.insert(format!("{:.2}", target), qps);
        }

        // Build SingleRunResult entries
        let raw_runs: Vec<SingleRunResult> = run_metrics.iter().enumerate().map(|(i, _qm)| {
            SingleRunResult {
                run_id: i,
                timestamp: chrono::Utc::now(),
                quality: run_metrics[i].clone(),
                latency_mean_ms: if !run_latencies.is_empty() && i < run_latencies.len() {
                    let l = &run_latencies[i];
                    l.iter().sum::<f64>() / l.len() as f64
                } else { 0.0 },
                duration_ms: if i < run_durations.len() {
                    run_durations[i].as_secs_f64() * 1000.0
                } else { 0.0 },
                energy: EnergyMeasurement {
                    total_queries: corpus.queries.len(),
                    total_energy_joules: 0.0,
                    energy_per_query_mj: 0.0,
                    measurement_method: crate::metrics::energy::EnergyMethod::NotAvailable,
                    available: false,
                },
            }
        }).collect();

        Ok(PublicationResult {
            benchmark_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now(),
            git_hash: None,
            config_hash: String::new(),
            hardware: HardwareProfile::capture(),
            database: "AttentionDB (CPU)".into(),
            database_version: "0.2.0".into(),
            dataset: "synthetic".into(),
            scale: n_docs,
            difficulty,
            experiment_type: ExperimentType::Standard,
            head_combination: crate::workload::HeadCombination::AllHeads,
            failure_mode: None,
            translation_strategy: crate::adapters::translation::TranslationStrategy::SemanticOnly,
            run_protocol: protocol,
            measured_difficulty,
            recall_at_1: AggregatedMetric::from_samples(
                &run_metrics.iter().map(|m| m.recall_at_1).collect::<Vec<_>>(),
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            recall_at_10: AggregatedMetric::from_samples(
                &run_metrics.iter().map(|m| m.recall_at_10).collect::<Vec<_>>(),
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            recall_at_100: AggregatedMetric::from_samples(
                &run_metrics.iter().map(|m| m.recall_at_100).collect::<Vec<_>>(),
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            mrr: agg_quality.mrr,
            ndcg_at_10: agg_quality.ndcg_at_10,
            ndcg_at_100: agg_quality.ndcg_at_100,
            latency_mean_ms: AggregatedMetric::from_samples(
                &all_latencies,
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            latency_p50_ms: AggregatedMetric::from_samples(
                &latency_p50_per_run,
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            latency_p90_ms: AggregatedMetric::from_samples(
                &latency_p90_per_run,
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            latency_p95_ms: AggregatedMetric::from_samples(
                &latency_p95_per_run,
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            latency_p99_ms: AggregatedMetric::from_samples(
                &latency_p99_per_run,
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            latency_p999_ms: AggregatedMetric::from_samples(
                &latency_p999_per_run,
                self.config.general.confidence_level,
                self.config.general.bootstrap_resamples,
                &mut self.rng,
            ),
            network_overhead_profile: None,
            throughput_profile,
            pareto_result,
            peak_memory_rss_mb: AggregatedMetric {
                mean: 0.0, std_dev: 0.0, ci_lower_95: 0.0, ci_upper_95: 0.0,
                bootstrap_ci_lower: 0.0, bootstrap_ci_upper: 0.0, n: 0,
            },
            disk_usage_gb: 0.0,
            index_build_time_s: index_build_times.iter().sum::<f64>() / index_build_times.len().max(1) as f64 / 1000.0,
            energy: EnergyMeasurement {
                total_queries: 0, total_energy_joules: 0.0, energy_per_query_mj: 0.0,
                measurement_method: crate::metrics::energy::EnergyMethod::NotAvailable,
                available: false,
            },
            statistical_comparisons: stats,
            consensus_accuracy: None,
            gating_efficiency_vs_uniform: None,
            per_head_weights: None,
            raw_runs,
        })
    }

    pub fn run_power_analysis(&self) -> anyhow::Result<Vec<PowerAnalysisResult>> {
        Ok(validate_n_for_benchmark(
            &self.config.power_analysis.effect_sizes,
            self.config.power_analysis.estimated_num_comparisons,
            30,
        ))
    }

    pub async fn run_ann_benchmark_compat(
        &mut self,
        dataset: &AnnBenchmarkLoaded,
    ) -> anyhow::Result<Vec<PublicationResult>> {
        let (documents, queries) = dataset.clone().into_documents_and_queries();
        let ground_truth: Vec<Vec<DocumentId>> = queries.iter()
            .map(|q| q.ground_truth.clone())
            .collect();
        let n_docs = documents.len();

        let _generator = WorkloadGenerator::new(self.config.general.random_seed, dataset.dataset.dimension());

        let corpus = GeneratedCorpus {
            documents,
            queries,
            ground_truth,
        };

        let protocol = RunProtocol::for_scale(n_docs, &self.config.run_protocol);
        let result = self.run_single_experiment_with_corpus(corpus, n_docs, protocol).await?;
        Ok(vec![result])
    }

    async fn run_single_experiment_with_corpus(
        &mut self,
        _corpus: GeneratedCorpus,
        n_docs: usize,
        protocol: RunProtocol,
    ) -> anyhow::Result<PublicationResult> {
        self.run_single_experiment(n_docs, DifficultyLevel::Medium, protocol).await
    }
}
