use rand::Rng;

pub struct ChaosTester {
    pub cluster_size: usize,
}

impl ChaosTester {
    pub fn new(cluster_size: usize) -> Self {
        Self { cluster_size }
    }

    pub fn kill_random_node(&self) -> usize {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..self.cluster_size)
    }

    pub fn partition_network(&self, duration_secs: u64) -> u64 {
        duration_secs
    }

    pub fn verify_availability(&self, healthy_count: usize) -> bool {
        healthy_count as f64 >= self.cluster_size as f64 * 0.5
    }

    pub fn inject_latency(&self, base_ms: u64) -> u64 {
        let mut rng = rand::thread_rng();
        base_ms + rng.gen_range(0..100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_node_in_range() {
        let chaos = ChaosTester::new(5);
        for _ in 0..100 {
            let killed = chaos.kill_random_node();
            assert!(killed < 5);
        }
    }

    #[test]
    fn test_availability_threshold() {
        let chaos = ChaosTester::new(5);
        assert!(chaos.verify_availability(3));  // 3/5 >= 50%
        assert!(!chaos.verify_availability(2)); // 2/5 < 50%
    }

    #[test]
    fn test_partition_returns_duration() {
        let chaos = ChaosTester::new(3);
        assert_eq!(chaos.partition_network(30), 30);
    }
}
