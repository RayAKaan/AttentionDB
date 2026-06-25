use crate::reporting::json_output::PublicationResult;
use crate::metrics::pareto::ParetoResult;

pub struct LatexOutput;

impl LatexOutput {
    pub fn generate_main_comparison_table(
        results: &[PublicationResult],
        scale: usize,
    ) -> String {
        let relevant: Vec<&PublicationResult> = results.iter()
            .filter(|r| r.scale == scale
                && matches!(r.experiment_type, crate::workload::ExperimentType::Standard))
            .collect();

        let mut table = String::new();
        table.push_str("\\begin{table}[htbp]\n\\centering\n");
        table.push_str(&format!(
            "\\caption{{Retrieval quality comparison at $n={}$. \
             Significance markers use Benjamini-Hochberg FDR correction at $q=0.05$. \
             $\\dagger$: significantly worse than AttentionDB. \
             Quality metrics use binary relevance.}}\n", scale
        ));
        table.push_str("\\label{tab:main}\n");
        table.push_str("\\begin{tabular}{lcccccc}\n\\toprule\n");
        table.push_str("System & R@1 & R@10 & MRR & NDCG@10 & Mean Lat. (ms) & QPS \\\\\n");
        table.push_str("\\midrule\n");

        for r in &relevant {
            let sig = r.statistical_comparisons.iter()
                .find(|c| c.label.contains("NDCG"))
                .and_then(|c| c.significant_bh_fdr05)
                .unwrap_or(false);
            let marker = if sig && r.database != "AttentionDB (CPU)" {
                "$^{\\dagger}$"
            } else { "" };

            let margin = |m: &crate::stats::aggregator::AggregatedMetric| -> f64 {
                (m.ci_upper_95 - m.ci_lower_95) / 2.0
            };

            table.push_str(&format!(
                "{}{} & {:.3}$\\pm${:.3} & {:.3}$\\pm${:.3} & {:.3}$\\pm${:.3} & \
                 {:.3}$\\pm${:.3} & {:.2}$\\pm${:.2} & {:.0} \\\\\n",
                r.database, marker,
                r.recall_at_1.mean, margin(&r.recall_at_1),
                r.recall_at_10.mean, margin(&r.recall_at_10),
                r.mrr.mean, margin(&r.mrr),
                r.ndcg_at_10.mean, margin(&r.ndcg_at_10),
                r.latency_mean_ms.mean, margin(&r.latency_mean_ms),
                r.throughput_profile.peak_qps,
            ));
        }

        table.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        table
    }

    pub fn generate_pareto_table(pareto_results: &[ParetoResult]) -> String {
        let mut table = String::new();

        table.push_str("\\begin{table}[htbp]\n\\centering\n");
        table.push_str("\\caption{Queries per second at target recall levels. \
            Higher is better. $-$: system did not achieve this recall level.}\n");
        table.push_str("\\label{tab:pareto}\n");
        table.push_str("\\begin{tabular}{lcccc}\n\\toprule\n");
        table.push_str("System & QPS@R90 & QPS@R95 & QPS@R99 & Peak QPS \\\\\n");
        table.push_str("\\midrule\n");

        for pr in pareto_results {
            let qps90 = pr.qps_at_recall(0.90)
                .map_or("$-$".to_string(), |q| format!("{:.0}", q));
            let qps95 = pr.qps_at_recall(0.95)
                .map_or("$-$".to_string(), |q| format!("{:.0}", q));
            let qps99 = pr.qps_at_recall(0.99)
                .map_or("$-$".to_string(), |q| format!("{:.0}", q));
            let peak: f64 = pr.points.iter()
                .map(|p| p.qps_single_thread)
                .fold(f64::NEG_INFINITY, f64::max);

            table.push_str(&format!(
                "{} & {} & {} & {} & {:.0} \\\\\n",
                pr.database, qps90, qps95, qps99, peak
            ));
        }

        table.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        table
    }

    pub fn generate_ablation_table(results: &[PublicationResult]) -> String {
        let mut table = String::new();
        let attentiondb: Vec<&PublicationResult> = results.iter()
            .filter(|r| r.database.contains("AttentionDB"))
            .collect();

        table.push_str("\\begin{table}[htbp]\n\\centering\n");
        table.push_str("\\caption{AttentionDB head ablation study. \
            Shows contribution of each head type and gating mechanism.}\n");
        table.push_str("\\begin{tabular}{lcccc}\n\\toprule\n");
        table.push_str("Head Config & R@10 & NDCG@10 & Lat. (ms) & Gating Eff. \\\\\n");
        table.push_str("\\midrule\n");

        for r in &attentiondb {
            let gating_eff = r.gating_efficiency_vs_uniform
                .map(|g| format!("{:.2}", g))
                .unwrap_or_else(|| "--".to_string());

            table.push_str(&format!(
                "{} & {:.4} & {:.4} & {:.2} & {} \\\\\n",
                format!("{:?}", r.head_combination),
                r.recall_at_10.mean,
                r.ndcg_at_10.mean,
                r.latency_mean_ms.mean,
                gating_eff,
            ));
        }

        table.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        table
    }
}
