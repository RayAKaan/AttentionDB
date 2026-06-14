use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ReadReplica {
    pub id: u32,
    pub shard_id: u32,
    pub address: String,
    pub last_applied_index: u64,
    pub healthy: bool,
}

impl ReadReplica {
    pub fn new(id: u32, shard_id: u32, address: &str) -> Self {
        Self { id, shard_id, address: address.to_string(), last_applied_index: 0, healthy: true }
    }

    pub fn apply_log(&mut self, index: u64) {
        self.last_applied_index = index;
    }

    pub fn mark_unhealthy(&mut self) {
        self.healthy = false;
    }
}

pub struct ReplicaManager {
    pub replicas: HashMap<u32, Vec<ReadReplica>>, // shard_id -> replicas
}

impl ReplicaManager {
    pub fn new() -> Self {
        Self { replicas: HashMap::new() }
    }

    pub fn add_replica(&mut self, replica: ReadReplica) {
        self.replicas.entry(replica.shard_id).or_default().push(replica);
    }

    pub fn remove_replica(&mut self, shard_id: u32, replica_id: u32) {
        if let Some(replicas) = self.replicas.get_mut(&shard_id) {
            replicas.retain(|r| r.id != replica_id);
        }
    }

    pub fn get_healthy_replicas(&self, shard_id: u32) -> Vec<&ReadReplica> {
        self.replicas
            .get(&shard_id)
            .map(|reps| reps.iter().filter(|r| r.healthy).collect())
            .unwrap_or_default()
    }

    pub fn all_replicas_for_shard(&self, shard_id: u32) -> Option<&Vec<ReadReplica>> {
        self.replicas.get(&shard_id)
    }

    pub fn total_replicas(&self) -> usize {
        self.replicas.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_replicas() {
        let mut rm = ReplicaManager::new();
        rm.add_replica(ReadReplica::new(1, 1, "10.0.0.11:7400"));
        rm.add_replica(ReadReplica::new(2, 1, "10.0.0.12:7400"));
        rm.add_replica(ReadReplica::new(3, 2, "10.0.1.11:7400"));
        assert_eq!(rm.total_replicas(), 3);
    }

    #[test]
    fn test_healthy_replicas() {
        let mut rm = ReplicaManager::new();
        let r1 = ReadReplica::new(1, 1, "addr1");
        let mut r2 = ReadReplica::new(2, 1, "addr2");
        r2.mark_unhealthy();
        rm.add_replica(r1);
        rm.add_replica(r2);
        assert_eq!(rm.get_healthy_replicas(1).len(), 1);
    }

    #[test]
    fn test_remove_replica() {
        let mut rm = ReplicaManager::new();
        rm.add_replica(ReadReplica::new(1, 1, "addr"));
        rm.remove_replica(1, 1);
        assert_eq!(rm.total_replicas(), 0);
    }

    #[test]
    fn test_apply_log() {
        let mut r = ReadReplica::new(1, 1, "addr");
        r.apply_log(42);
        assert_eq!(r.last_applied_index, 42);
    }
}
