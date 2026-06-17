//! Distributed Sharding & Consistent Hash Ring Engine
//!
//! Implements virtual node balanced consistent hashing to cleanly partition
//! multi-head vector indexes and relational document chunks across quorum peers.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher, DefaultHasher};

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
    pub ring: BTreeMap<u64, u32>,
    pub virtual_nodes: usize,
}

fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

impl ShardManager {
    pub fn new() -> Self {
        Self::with_virtual_nodes(100)
    }

    pub fn with_virtual_nodes(vnodes: usize) -> Self {
        Self {
            shards: HashMap::new(),
            ring: BTreeMap::new(),
            virtual_nodes: vnodes,
        }
    }

    pub fn add_shard(&mut self, shard: Shard) {
        let shard_id = shard.id;
        for v in 0..self.virtual_nodes {
            let key = format!("shard:{}:vnode:{}", shard_id, v);
            let hash = hash_str(&key);
            self.ring.insert(hash, shard_id);
        }
        self.shards.insert(shard_id, shard);
    }

    pub fn remove_shard(&mut self, id: u32) -> Option<Shard> {
        for v in 0..self.virtual_nodes {
            let key = format!("shard:{}:vnode:{}", id, v);
            let hash = hash_str(&key);
            self.ring.remove(&hash);
        }
        self.shards.remove(&id)
    }

    pub fn get_shard(&self, id: u32) -> Option<&Shard> {
        self.shards.get(&id)
    }

    pub fn get_shard_for_key(&self, routing_key: &str) -> Option<&Shard> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = hash_str(routing_key);
        let shard_id = match self.ring.range(hash..).next() {
            Some((_, &s_id)) => s_id,
            None => *self.ring.values().next().unwrap(),
        };
        self.get_shard(shard_id)
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

impl Default for ShardManager {
    fn default() -> Self {
        Self::new()
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
        shard.assign_head("semantic");
        assert_eq!(shard.heads.len(), 1);
    }

    #[test]
    fn test_consistent_hash_ring_routing_and_rebalancing() {
        let mut sm = ShardManager::with_virtual_nodes(200);
        sm.add_shard(Shard::new(1, vec![], "10.0.0.1:7400"));
        sm.add_shard(Shard::new(2, vec![], "10.0.0.2:7400"));
        sm.add_shard(Shard::new(3, vec![], "10.0.0.3:7400"));

        let mut routing_counts = HashMap::new();
        for i in 0..10_000 {
            let key = format!("document_uuid_{}", i);
            let s_id = sm.get_shard_for_key(&key).unwrap().id;
            *routing_counts.entry(s_id).or_insert(0) += 1;
        }

        assert_eq!(routing_counts.len(), 3);
        for &count in routing_counts.values() {
            assert!(count > 2000 && count < 5000);
        }

        sm.remove_shard(2);
        let remapped_id = sm.get_shard_for_key("document_uuid_42").unwrap().id;
        assert!(remapped_id == 1 || remapped_id == 3);
    }
}
