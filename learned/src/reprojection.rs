use crate::projection::ProjectionMatrix;
use crate::error::LearnedError;

pub struct ReprojectionJob {
    pub collection: String,
    pub old_projection: ProjectionMatrix,
    pub new_projection: ProjectionMatrix,
}

impl ReprojectionJob {
    pub fn new(
        collection: &str,
        old: ProjectionMatrix,
        new: ProjectionMatrix,
    ) -> Self {
        Self {
            collection: collection.to_string(),
            old_projection: old,
            new_projection: new,
        }
    }

    pub fn run(&self) -> Result<(), LearnedError> {
        self.run_with_callbacks(|_| vec![], |_, _, _| {})
    }

    pub fn run_with_callbacks<F, U>(
        &self,
        mut fetch_records: F,
        mut update_vector: U,
    ) -> Result<(), LearnedError>
    where
        F: FnMut(&str) -> Vec<(u64, String, Vec<f32>)>,
        U: FnMut(&str, u64, &[f32]),
    {
        println!("[Reprojection] Starting active job for collection: {}", self.collection);
        println!("  Target Dim: {}", self.new_projection.config.dim);
        println!("  Target Heads: {}", self.new_projection.config.num_heads);

        let items = fetch_records(&self.collection);
        println!("[Reprojection] Retrieved {} vector entries for batch remapping", items.len());

        for (id, head_name, raw_vec) in items {
            let reprojected = self.new_projection.project_key(&raw_vec);
            update_vector(&head_name, id, &reprojected);
        }

        println!("[Reprojection] Complete for collection: {}", self.collection);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reprojection_job_creation() {
        let config = crate::projection::ProjectionConfig::default();
        let old = ProjectionMatrix::new(config.clone());
        let new = ProjectionMatrix::new(config);
        let job = ReprojectionJob::new("test", old, new);
        assert_eq!(job.collection, "test");
    }

    #[test]
    fn test_reprojection_run_succeeds() {
        let config = crate::projection::ProjectionConfig::default();
        let old = ProjectionMatrix::new(config.clone());
        let new = ProjectionMatrix::new(config);
        let job = ReprojectionJob::new("papers", old, new);
        assert!(job.run().is_ok());
    }

    #[test]
    fn test_authentic_reprojection_callbacks() {
        let config = crate::projection::ProjectionConfig { dim: 4, num_heads: 1, head_dim: 4 };
        let old = ProjectionMatrix::new(config.clone());
        let new = ProjectionMatrix::new(config);
        let job = ReprojectionJob::new("papers", old, new);

        let mut updated_items = Vec::new();
        let result = job.run_with_callbacks(
            |coll| {
                if coll == "papers" {
                    vec![(1, "semantic".to_string(), vec![1.0, 0.0, 0.0, 0.0])]
                } else {
                    vec![]
                }
            },
            |head, id, new_vec| {
                updated_items.push((head.to_string(), id, new_vec.to_vec()));
            }
        );

        assert!(result.is_ok());
        assert_eq!(updated_items.len(), 1);
        assert_eq!(updated_items[0].0, "semantic");
        assert_eq!(updated_items[0].1, 1);
        assert_eq!(updated_items[0].2.len(), 4);
    }
}
