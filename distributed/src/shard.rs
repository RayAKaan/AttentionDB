use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HeadPartition {
    pub head_name: String,
    pub shard_id: u32,
    pub node_address: String,
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub id: u32,
    pub heads: Vec<String>,
    pub address: String,
    pub is_leader: bool,
}

impl Shard {
    pub fn new(id: u32, heads: Vec<String>, address: &str) -> Self {
        Self { id, heads, address: address.to_string(), is_leader: false }
    }

    pub fn assign_head(&mut self, head: &str) {
        if !self.heads.contains(&head.to_string()) {
            self.heads.push(head.to_string());
        }
    }
}

pub struct ShardManager {
    pub shards: HashMap<u32, Shard>,
}

impl ShardManager {
    pub fn new() -> Self {
        Self { shards: HashMap::new() }
    }

    pub fn add_shard(&mut self, shard: Shard) {
        self.shards.insert(shard.id, shard);
    }

    pub fn remove_shard(&mut self, id: u32) -> Option<Shard> {
        self.shards.remove(&id)
    }

    pub fn get_shard(&self, id: u32) -> Option<&Shard> {
        self.shards.get(&id)
    }

    pub fn get_shard_for_head(&self, head: &str) -> Option<&Shard> {
        self.shards.values().find(|s| s.heads.contains(&head.to_string()))
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn list_shards(&self) -> Vec<u32> {
        self.shards.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_shard() {
        let mut sm = ShardManager::new();
        sm.add_shard(Shard::new(1, vec!["semantic".into()], "10.0.0.1:7400"));
        assert_eq!(sm.shard_count(), 1);
        assert!(sm.get_shard(1).is_some());
    }

    #[test]
    fn test_get_shard_for_head() {
        let mut sm = ShardManager::new();
        sm.add_shard(Shard::new(1, vec!["semantic".into(), "temporal".into()], "10.0.0.1:7400"));
        sm.add_shard(Shard::new(2, vec!["structural".into()], "10.0.0.2:7400"));
        assert_eq!(sm.get_shard_for_head("temporal").unwrap().id, 1);
        assert_eq!(sm.get_shard_for_head("structural").unwrap().id, 2);
    }

    #[test]
    fn test_remove_shard() {
        let mut sm = ShardManager::new();
        sm.add_shard(Shard::new(1, vec![], "addr"));
        assert!(sm.remove_shard(1).is_some());
        assert_eq!(sm.shard_count(), 0);
    }

    #[test]
    fn test_assign_head() {
        let mut shard = Shard::new(1, vec![], "addr");
        shard.assign_head("semantic");
        shard.assign_head("semantic"); // duplicate
        assert_eq!(shard.heads.len(), 1);
    }
}
