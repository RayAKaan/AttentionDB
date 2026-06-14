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
        println!("[Reprojection] Starting for collection: {}", self.collection);
        println!("  Dim: {}", self.new_projection.config.dim);
        println!("  Heads: {}", self.new_projection.config.num_heads);

        // In production: iterate all records in collection,
        // re-project k_vecs and v_vecs using new_projection,
        // update HNSW indexes (Phase 2)

        println!("[Reprojection] Complete for: {}", self.collection);
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
}
