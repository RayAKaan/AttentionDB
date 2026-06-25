use crate::reporting::json_output::PublicationResult;
use tabled::{Table, Tabled};

#[derive(Tabled)]
struct ComparisonRow {
    #[tabled(rename = "System")]
    database: String,
    #[tabled(rename = "R@1")]
    recall_at_1: String,
    #[tabled(rename = "R@10")]
    recall_at_10: String,
    #[tabled(rename = "NDCG@10")]
    ndcg_at_10: String,
    #[tabled(rename = "Lat (ms)")]
    latency: String,
    #[tabled(rename = "QPS")]
    qps: String,
    #[tabled(rename = "Significant")]
    significant: String,
}

pub struct TableRenderer;

impl TableRenderer {
    pub fn render_comparison_table(
        results: &[PublicationResult],
    ) -> String {
        let rows: Vec<ComparisonRow> = results.iter().map(|r| {
            let sig = r.statistical_comparisons.first()
                .and_then(|c| c.significant_bh_fdr05)
                .map(|s| if s { "†" } else { "" })
                .unwrap_or("");

            ComparisonRow {
                database: r.database.clone(),
                recall_at_1: format!("{:.3}±{:.3}", r.recall_at_1.mean, r.recall_at_1.std_dev),
                recall_at_10: format!("{:.3}±{:.3}", r.recall_at_10.mean, r.recall_at_10.std_dev),
                ndcg_at_10: format!("{:.3}±{:.3}", r.ndcg_at_10.mean, r.ndcg_at_10.std_dev),
                latency: format!("{:.2}±{:.2}", r.latency_mean_ms.mean, r.latency_mean_ms.std_dev),
                qps: format!("{:.0}", r.throughput_profile.peak_qps),
                significant: sig.to_string(),
            }
        }).collect();

        let mut table = Table::new(&rows);
        table.with(tabled::settings::Style::modern());
        table.to_string()
    }

    pub fn render_difficulty_table(
        results: &[PublicationResult],
    ) -> String {
        use tabled::*;

        #[derive(Tabled)]
        struct DifficultyRow {
            #[tabled(rename = "System")]
            database: String,
            #[tabled(rename = "Difficulty")]
            difficulty: String,
            #[tabled(rename = "Contrast")]
            contrast: String,
            #[tabled(rename = "LID")]
            lid: String,
            #[tabled(rename = "Hubness")]
            hubness: String,
            #[tabled(rename = "Unambiguous")]
            unambiguous: String,
        }

        let rows: Vec<DifficultyRow> = results.iter().map(|r| {
            DifficultyRow {
                database: r.database.clone(),
                difficulty: format!("{:?}", r.difficulty),
                contrast: format!("{:.3}", r.measured_difficulty.contrast_ratio),
                lid: format!("{:.1}", r.measured_difficulty.mean_lid),
                hubness: format!("{:.2}", r.measured_difficulty.hubness_skewness),
                unambiguous: format!("{:.2}", r.measured_difficulty.unambiguous_top1_fraction),
            }
        }).collect();

        let mut table = Table::new(&rows);
        table.with(tabled::settings::Style::modern());
        table.to_string()
    }
}
