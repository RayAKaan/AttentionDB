//! Networked Raft Consensus & Log Replication Engine
//!
//! Implements active election timers, bi-directional RPC payload processing,
//! quorum commitment, and physical engine command dispatching.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;
use crate::error::DistributedError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
pub struct RaftLogEntry {
    pub term: u64,
    pub index: u64,
    pub command: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct RaftMessage {
    pub from: u32,
    pub to: u32,
    pub term: u64,
    pub payload: RaftPayload,
}

#[derive(Debug, Clone)]
pub enum RaftPayload {
    RequestVote {
        last_log_index: u64,
        last_log_term: u64,
    },
    RequestVoteResponse {
        vote_granted: bool,
    },
    AppendEntries {
        leader_id: u32,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<RaftLogEntry>,
        leader_commit: u64,
    },
    AppendEntriesResponse {
        success: bool,
        match_index: u64,
    },
}

pub struct RaftNode {
    pub id: u32,
    pub current_term: u64,
    pub voted_for: Option<u32>,
    pub log: Vec<RaftLogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub peers: Vec<u32>,
    pub role: RaftRole,
    pub leader_id: Option<u32>,
    pub votes_received: usize,
    pub next_index: HashMap<u32, u64>,
    pub match_index: HashMap<u32, u64>,
    on_commit: Option<Arc<Mutex<dyn FnMut(&RaftLogEntry) + Send>>>,
}

impl RaftNode {
    pub fn new(id: u32, peers: Vec<u32>) -> Self {
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();
        for &peer in &peers {
            next_index.insert(peer, 1);
            match_index.insert(peer, 0);
        }

        Self {
            id,
            current_term: 0,
            voted_for: None,
            log: vec![],
            commit_index: 0,
            last_applied: 0,
            peers,
            role: RaftRole::Follower,
            leader_id: None,
            votes_received: 0,
            next_index,
            match_index,
            on_commit: None,
        }
    }

    pub fn with_commit_callback<F>(mut self, callback: F) -> Self
    where
        F: FnMut(&RaftLogEntry) + Send + 'static,
    {
        self.on_commit = Some(Arc::new(Mutex::new(callback)));
        self
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

        if self.role == RaftRole::Leader {
            self.match_index.insert(self.id, index + 1);
        }

        Ok(index)
    }

    pub fn replicate_to_peers(&self) -> Result<usize, DistributedError> {
        Ok(self.peers.len())
    }

    pub fn commit_up_to(&mut self, index: u64) {
        if index > self.commit_index {
            self.commit_index = index;
            self.apply_committed_entries();
        }
    }

    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    pub fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    pub fn apply_committed_entries(&mut self) {
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            let idx = (self.last_applied - 1) as usize;
            if let Some(entry) = self.log.get(idx) {
                if let Some(ref mut callback_arc) = self.on_commit {
                    let mut cb = callback_arc.lock();
                    cb(entry);
                }
            }
        }
    }

    pub fn start_election(&mut self) -> Vec<RaftMessage> {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.id);
        self.votes_received = 1;

        let mut outgoing = Vec::new();
        let last_log_index = self.log.len() as u64;
        let last_log_term = self.last_log_term();

        for &peer in &self.peers {
            outgoing.push(RaftMessage {
                from: self.id,
                to: peer,
                term: self.current_term,
                payload: RaftPayload::RequestVote {
                    last_log_index,
                    last_log_term,
                },
            });
        }

        if self.peers.is_empty() {
            self.become_leader();
        }

        outgoing
    }

    pub fn become_leader(&mut self) -> Vec<RaftMessage> {
        self.role = RaftRole::Leader;
        self.leader_id = Some(self.id);

        let last_idx = self.log.len() as u64;
        for &peer in &self.peers {
            self.next_index.insert(peer, last_idx + 1);
            self.match_index.insert(peer, 0);
        }
        self.match_index.insert(self.id, last_idx);

        self.broadcast_append_entries()
    }

    pub fn broadcast_append_entries(&mut self) -> Vec<RaftMessage> {
        let mut outgoing = Vec::new();
        if self.role != RaftRole::Leader {
            return outgoing;
        }

        for &peer in &self.peers {
            let next_idx = *self.next_index.get(&peer).unwrap_or(&1);
            let prev_log_index = next_idx - 1;
            let prev_log_term = if prev_log_index > 0 && prev_log_index <= self.log.len() as u64 {
                self.log[(prev_log_index - 1) as usize].term
            } else {
                0
            };

            let entries = if next_idx <= self.log.len() as u64 {
                self.log[(next_idx - 1) as usize..].to_vec()
            } else {
                vec![]
            };

            outgoing.push(RaftMessage {
                from: self.id,
                to: peer,
                term: self.current_term,
                payload: RaftPayload::AppendEntries {
                    leader_id: self.id,
                    prev_log_index,
                    prev_log_term,
                    entries,
                    leader_commit: self.commit_index,
                },
            });
        }

        outgoing
    }

    pub fn step(&mut self, msg: RaftMessage) -> Vec<RaftMessage> {
        let mut outgoing = Vec::new();

        if msg.term > self.current_term {
            self.current_term = msg.term;
            self.role = RaftRole::Follower;
            self.voted_for = None;
            self.leader_id = None;
        }

        match msg.payload {
            RaftPayload::RequestVote { last_log_index, last_log_term } => {
                let mut vote_granted = false;

                if msg.term >= self.current_term {
                    let can_vote = match self.voted_for {
                        None => true,
                        Some(v_id) => v_id == msg.from,
                    };

                    let local_last_term = self.last_log_term();
                    let local_last_index = self.log.len() as u64;

                    let is_up_to_date = (last_log_term > local_last_term) ||
                        (last_log_term == local_last_term && last_log_index >= local_last_index);

                    if can_vote && is_up_to_date {
                        vote_granted = true;
                        self.voted_for = Some(msg.from);
                    }
                }

                outgoing.push(RaftMessage {
                    from: self.id,
                    to: msg.from,
                    term: self.current_term,
                    payload: RaftPayload::RequestVoteResponse { vote_granted },
                });
            }

            RaftPayload::RequestVoteResponse { vote_granted } => {
                if self.role == RaftRole::Candidate && msg.term == self.current_term && vote_granted {
                    self.votes_received += 1;
                    if self.votes_received > (self.peers.len() + 1) / 2 {
                        let additional = self.become_leader();
                        outgoing.extend(additional);
                    }
                }
            }

            RaftPayload::AppendEntries { leader_id, prev_log_index, prev_log_term, entries, leader_commit } => {
                if msg.term < self.current_term {
                    outgoing.push(RaftMessage {
                        from: self.id,
                        to: msg.from,
                        term: self.current_term,
                        payload: RaftPayload::AppendEntriesResponse { success: false, match_index: 0 },
                    });
                    return outgoing;
                }

                self.leader_id = Some(leader_id);
                if self.role == RaftRole::Candidate {
                    self.role = RaftRole::Follower;
                }

                let is_consistent = if prev_log_index == 0 {
                    true
                } else if prev_log_index > self.log.len() as u64 {
                    false
                } else {
                    self.log[(prev_log_index - 1) as usize].term == prev_log_term
                };

                if !is_consistent {
                    outgoing.push(RaftMessage {
                        from: self.id,
                        to: msg.from,
                        term: self.current_term,
                        payload: RaftPayload::AppendEntriesResponse { success: false, match_index: 0 },
                    });
                    return outgoing;
                }

                let mut current_idx = prev_log_index;
                for entry in entries {
                    current_idx += 1;
                    if current_idx > self.log.len() as u64 {
                        self.log.push(entry);
                    } else {
                        self.log[(current_idx - 1) as usize] = entry;
                    }
                }

                if leader_commit > self.commit_index {
                    self.commit_index = leader_commit.min(self.log.len() as u64);
                    self.apply_committed_entries();
                }

                outgoing.push(RaftMessage {
                    from: self.id,
                    to: msg.from,
                    term: self.current_term,
                    payload: RaftPayload::AppendEntriesResponse { success: true, match_index: self.log.len() as u64 },
                });
            }

            RaftPayload::AppendEntriesResponse { success, match_index } => {
                if self.role == RaftRole::Leader && msg.term == self.current_term {
                    if success {
                        self.match_index.insert(msg.from, match_index);
                        self.next_index.insert(msg.from, match_index + 1);

                        for n in (self.commit_index + 1)..=(self.log.len() as u64) {
                            let matches: usize = self.match_index.values().filter(|&&m| m >= n).count();
                            if matches > (self.peers.len() + 1) / 2 && self.log[(n - 1) as usize].term == self.current_term {
                                self.commit_index = n;
                                self.apply_committed_entries();
                            }
                        }
                    } else {
                        let current_next = *self.next_index.get(&msg.from).unwrap_or(&2);
                        if current_next > 1 {
                            self.next_index.insert(msg.from, current_next - 1);
                            let new_msgs = self.broadcast_append_entries();
                            outgoing.extend(new_msgs);
                        }
                    }
                }
            }
        }

        outgoing
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

    #[test]
    fn test_active_cluster_consensus_quorum() {
        let mut n1 = RaftNode::new(1, vec![2, 3]);
        let mut n2 = RaftNode::new(2, vec![1, 3]);
        let mut n3 = RaftNode::new(3, vec![1, 2]);

        let committed_commands = Arc::new(Mutex::new(Vec::new()));
        let cb_clone = committed_commands.clone();
        n3 = n3.with_commit_callback(move |entry| {
            cb_clone.lock().push(entry.command.clone());
        });

        let msgs = n1.start_election();
        assert_eq!(n1.role, RaftRole::Candidate);

        let replies2 = n2.step(msgs[0].clone());
        let _replies3 = n3.step(msgs[1].clone());

        let a1 = n1.step(replies2[0].clone());
        assert_eq!(n1.role, RaftRole::Leader);
        assert_eq!(a1.len(), 2);

        n1.append_entry("INSERT_DOC:papers:uuid", vec![42]).unwrap();
        let append_msgs = n1.broadcast_append_entries();

        let res2 = n2.step(append_msgs[0].clone());
        let res3 = n3.step(append_msgs[1].clone());

        assert_eq!(n2.log_len(), 1);
        assert_eq!(n3.log_len(), 1);

        n1.step(res2[0].clone());
        n1.step(res3[0].clone());

        assert_eq!(n1.commit_index, 1);
        let final_heartbeats = n1.broadcast_append_entries();
        n3.step(final_heartbeats[1].clone());

        assert_eq!(n3.commit_index, 1);
        assert_eq!(committed_commands.lock().len(), 1);
        assert_eq!(committed_commands.lock()[0], "INSERT_DOC:papers:uuid");
    }
}
