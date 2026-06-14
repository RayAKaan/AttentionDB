use crate::error::DistributedError;

#[derive(Debug, Clone)]
pub struct RaftLogEntry {
    pub term: u64,
    pub index: u64,
    pub command: String,
    pub data: Vec<u8>,
}

pub struct RaftNode {
    pub id: u32,
    pub current_term: u64,
    pub voted_for: Option<u32>,
    pub log: Vec<RaftLogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub peers: Vec<u32>,
}

impl RaftNode {
    pub fn new(id: u32, peers: Vec<u32>) -> Self {
        Self {
            id,
            current_term: 0,
            voted_for: None,
            log: vec![],
            commit_index: 0,
            last_applied: 0,
            peers,
        }
    }

    pub fn append_entry(&mut self, command: &str, data: Vec<u8>) -> Result<u64, DistributedError> {
        let index = self.log.len() as u64;
        let entry = RaftLogEntry {
            term: self.current_term,
            index,
            command: command.to_string(),
            data,
        };
        self.log.push(entry);
        Ok(index)
    }

    pub fn replicate_to_peers(&self) -> Result<usize, DistributedError> {
        Ok(self.peers.len())
    }

    pub fn commit_up_to(&mut self, index: u64) {
        if index > self.commit_index {
            self.commit_index = index;
        }
    }

    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    pub fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_append_entry() {
        let mut raft = RaftNode::new(1, vec![2, 3]);
        let idx = raft.append_entry("INSERT", vec![1, 2, 3]).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(raft.log_len(), 1);
    }

    #[test]
    fn test_raft_replicate_returns_peer_count() {
        let raft = RaftNode::new(1, vec![2, 3, 4]);
        assert_eq!(raft.replicate_to_peers().unwrap(), 3);
    }

    #[test]
    fn test_raft_commit() {
        let mut raft = RaftNode::new(1, vec![2]);
        raft.append_entry("INSERT", vec![]).unwrap();
        raft.append_entry("DELETE", vec![]).unwrap();
        raft.commit_up_to(1);
        assert_eq!(raft.commit_index, 1);
    }

    #[test]
    fn test_no_peers() {
        let raft = RaftNode::new(1, vec![]);
        assert_eq!(raft.replicate_to_peers().unwrap(), 0);
    }
}
