use crate::metrics::pareto::ParetoResult;
use plotters::prelude::*;

pub struct ParetoPlot;

impl ParetoPlot {
    pub fn render_pareto_curve(
        pareto_results: &[ParetoResult],
        output_path: &str,
        title: &str,
    ) -> anyhow::Result<()> {
        let root = SVGBackend::new(output_path, (800, 600))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let mut max_recall = 0.95f64;
        let mut max_qps = 10.0f64;
        let mut min_recall = 0.5f64;
        let mut min_qps = 1.0f64;

        for pr in pareto_results {
            for pt in &pr.points {
                if pt.recall_at_10_mean > max_recall {
                    max_recall = pt.recall_at_10_mean;
                }
                if pt.recall_at_10_mean < min_recall {
                    min_recall = pt.recall_at_10_mean;
                }
                if pt.qps_single_thread > max_qps {
                    max_qps = pt.qps_single_thread;
                }
                if pt.qps_single_thread < min_qps {
                    min_qps = pt.qps_single_thread;
                }
            }
        }

        // Add margins
        max_recall = (max_recall + 0.05).min(1.0);
        min_recall = (min_recall - 0.05).max(0.0);
        max_qps *= 1.2;
        min_qps = 0.0;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("Arial", 24).into_font())
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                min_recall..max_recall,
                min_qps..max_qps,
            )?;

        chart.configure_mesh()
            .x_desc("Recall@10")
            .y_desc("Queries per Second (QPS)")
            .axis_desc_style(("Arial", 16))
            .draw()?;

        let colors = [
            &RED, &BLUE, &GREEN, &BLACK, &MAGENTA, &CYAN, &YELLOW,
        ];

        for (i, pr) in pareto_results.iter().enumerate() {
            let color = colors[i % colors.len()];

            let mut data: Vec<(f64, f64)> = pr.points.iter()
                .map(|p| (p.recall_at_10_mean, p.qps_single_thread))
                .collect();
            data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            chart.draw_series(LineSeries::new(
                data.iter().map(|&(r, q)| (r, q)),
                color.clone(),
            ))?
            .label(&pr.database)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.clone())
            });

            for &(r, q) in &data {
                chart.draw_series(std::iter::once(Circle::new(
                    (r, q), 3, color.filled(),
                )))?;
            }
        }

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        tracing::info!("Pareto plot saved to {}", output_path);
        Ok(())
    }
}
