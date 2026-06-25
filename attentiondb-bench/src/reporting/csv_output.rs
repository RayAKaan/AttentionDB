use crate::reporting::json_output::PublicationResult;

pub struct CsvOutput;

impl CsvOutput {
    pub fn write_results(
        results: &[PublicationResult],
        path: &str,
    ) -> anyhow::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;

        wtr.write_record(&[
            "database", "dataset", "scale", "difficulty",
            "recall_at_1", "recall_at_10", "recall_at_100",
            "mrr", "ndcg_at_10", "ndcg_at_100",
            "latency_mean_ms", "latency_p99_ms",
            "index_build_time_s", "peak_memory_mb",
        ])?;

        for r in results {
            wtr.write_record(&[
                &r.database,
                &r.dataset,
                &r.scale.to_string(),
                &format!("{:?}", r.difficulty),
                &format!("{:.4}", r.recall_at_1.mean),
                &format!("{:.4}", r.recall_at_10.mean),
                &format!("{:.4}", r.recall_at_100.mean),
                &format!("{:.4}", r.mrr.mean),
                &format!("{:.4}", r.ndcg_at_10.mean),
                &format!("{:.4}", r.ndcg_at_100.mean),
                &format!("{:.2}", r.latency_mean_ms.mean),
                &format!("{:.2}", r.latency_p99_ms.mean),
                &format!("{:.2}", r.index_build_time_s),
                &format!("{:.1}", r.peak_memory_rss_mb.mean),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    pub fn write_pareto_points(
        results: &[PublicationResult],
        path: &str,
    ) -> anyhow::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;

        wtr.write_record(&[
            "database", "dataset", "scale",
            "param_value", "recall_at_10_mean", "latency_mean_ms", "qps",
        ])?;

        for r in results {
            for point in &r.pareto_result.points {
                wtr.write_record(&[
                    &r.database,
                    &r.dataset,
                    &r.scale.to_string(),
                    &point.param_value.to_string(),
                    &format!("{:.4}", point.recall_at_10_mean),
                    &format!("{:.2}", point.latency_mean_ms),
                    &format!("{:.1}", point.qps_single_thread),
                ])?;
            }
        }

        wtr.flush()?;
        Ok(())
    }
}
