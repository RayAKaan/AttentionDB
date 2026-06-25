use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, warn};
use crate::error::DistributedError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole { Follower, Candidate, Leader }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RaftLogEntry {
    pub term: u64,
    pub index: u64,
    pub command: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RaftMessage {
    pub from: u32,
    pub to: u32,
    pub term: u64,
    pub payload: RaftPayload,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RaftPayload {
    RequestVote { last_log_index: u64, last_log_term: u64 },
    RequestVoteResponse { vote_granted: bool },
    AppendEntries { leader_id: u32, prev_log_index: u64, prev_log_term: u64, entries: Vec<RaftLogEntry>, leader_commit: u64 },
    AppendEntriesResponse { success: bool, match_index: u64 },
}

pub struct RaftTransport;

impl RaftTransport {
    pub async fn send_to_peer(addr: &str, msg: &RaftMessage) -> Result<RaftMessage, DistributedError> {
        let mut stream = TcpStream::connect(addr).await
            .map_err(|e| DistributedError::Network(format!("connect: {}", e)))?;
        let data = bincode::serialize(msg)
            .map_err(|e| DistributedError::Network(format!("serialize: {}", e)))?;
        let len = data.len() as u32;
        stream.write_all(&len.to_le_bytes()).await
            .map_err(|e| DistributedError::Network(format!("write_len: {}", e)))?;
        stream.write_all(&data).await
            .map_err(|e| DistributedError::Network(format!("write: {}", e)))?;
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await
            .map_err(|e| DistributedError::Network(format!("read_len: {}", e)))?;
        let resp_len = u32::from_le_bytes(len_buf) as usize;
        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await
            .map_err(|e| DistributedError::Network(format!("read: {}", e)))?;
        bincode::deserialize(&resp_buf)
            .map_err(|e| DistributedError::Network(format!("deserialize: {}", e)))
    }

    pub async fn handle_incoming_message(node: &mut RaftNode, stream: &mut TcpStream) -> Result<(), DistributedError> {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await
            .map_err(|e| DistributedError::Network(format!("read_len: {}", e)))?;
        let msg_len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; msg_len];
        stream.read_exact(&mut buf).await
            .map_err(|e| DistributedError::Network(format!("read: {}", e)))?;
        let msg: RaftMessage = bincode::deserialize(&buf)
            .map_err(|e| DistributedError::Network(format!("deserialize: {}", e)))?;
        let responses = node.step(msg);
        for resp in responses {
            let data = bincode::serialize(&resp)
                .map_err(|e| DistributedError::Network(format!("serialize: {}", e)))?;
            let len = data.len() as u32;
            stream.write_all(&len.to_le_bytes()).await
                .map_err(|e| DistributedError::Network(format!("write_len: {}", e)))?;
            stream.write_all(&data).await
                .map_err(|e| DistributedError::Network(format!("write: {}", e)))?;
        }
        Ok(())
    }
}

pub struct RaftNode {
    pub id: u32,
    pub current_term: u64,
    pub voted_for: Option<u32>,
    pub log: Vec<RaftLogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub peers: Vec<u32>,
    pub peer_addresses: HashMap<u32, String>,
    pub role: RaftRole,
    pub leader_id: Option<u32>,
    pub votes_received: usize,
    pub next_index: HashMap<u32, u64>,
    pub match_index: HashMap<u32, u64>,
    on_commit: Option<Arc<Mutex<dyn FnMut(&RaftLogEntry) + Send>>>,
    pub election_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
}

impl RaftNode {
    pub fn new(id: u32, peers: Vec<u32>) -> Self {
        let mut next = HashMap::new();
        let mut match_idx = HashMap::new();
        let mut addrs = HashMap::new();
        for &p in &peers {
            next.insert(p, 1);
            match_idx.insert(p, 0);
            addrs.insert(p, format!("127.0.0.1:{}", 7401 + p));
        }
        Self {
            id, peers, current_term: 0, voted_for: None, log: vec![],
            commit_index: 0, last_applied: 0, peer_addresses: addrs,
            role: RaftRole::Follower, leader_id: None, votes_received: 0,
            next_index: next, match_index: match_idx, on_commit: None,
            election_timeout_ms: 150, heartbeat_interval_ms: 50,
        }
    }

    pub fn with_commit_callback<F>(mut self, cb: F) -> Self
    where F: FnMut(&RaftLogEntry) + Send + 'static {
        self.on_commit = Some(Arc::new(Mutex::new(cb)));
        self
    }

    pub fn append_entry(&mut self, command: &str, data: Vec<u8>) -> Result<u64, DistributedError> {
        let idx = self.log.len() as u64;
        self.log.push(RaftLogEntry { term: self.current_term, index: idx, command: command.into(), data });
        if self.role == RaftRole::Leader { self.match_index.insert(self.id, idx + 1); }
        Ok(idx)
    }

    pub fn commit_up_to(&mut self, index: u64) {
        if index > self.commit_index { self.commit_index = index; self.apply_committed_entries(); }
    }

    pub fn log_len(&self) -> usize { self.log.len() }

    pub fn last_log_term(&self) -> u64 { self.log.last().map(|e| e.term).unwrap_or(0) }

    pub fn apply_committed_entries(&mut self) {
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.log.get((self.last_applied - 1) as usize) {
                if let Some(ref cb) = self.on_commit { cb.lock()(entry); }
            }
        }
    }

    pub fn start_election(&mut self) -> Vec<RaftMessage> {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.id);
        self.votes_received = 1;
        let (lli, llt) = (self.log.len() as u64, self.last_log_term());
        let msgs: Vec<RaftMessage> = self.peers.iter().map(|&p| RaftMessage {
            from: self.id, to: p, term: self.current_term,
            payload: RaftPayload::RequestVote { last_log_index: lli, last_log_term: llt },
        }).collect();
        if self.peers.is_empty() { self.become_leader(); }
        msgs
    }

    pub fn become_leader(&mut self) -> Vec<RaftMessage> {
        self.role = RaftRole::Leader;
        self.leader_id = Some(self.id);
        let last = self.log.len() as u64;
        for &p in &self.peers { self.next_index.insert(p, last + 1); self.match_index.insert(p, 0); }
        self.match_index.insert(self.id, last);
        self.broadcast_append_entries()
    }

    pub fn broadcast_append_entries(&self) -> Vec<RaftMessage> {
        if self.role != RaftRole::Leader { return vec![]; }
        self.peers.iter().map(|&p| {
            let next = *self.next_index.get(&p).unwrap_or(&1);
            let prev = if next > 0 { next - 1 } else { 0 };
            let prev_term = if prev > 0 && prev <= self.log.len() as u64 { self.log[(prev-1) as usize].term } else { 0 };
            let entries = if next <= self.log.len() as u64 { self.log[(next-1) as usize..].to_vec() } else { vec![] };
            RaftMessage {
                from: self.id, to: p, term: self.current_term,
                payload: RaftPayload::AppendEntries {
                    leader_id: self.id, prev_log_index: prev, prev_log_term: prev_term,
                    entries, leader_commit: self.commit_index,
                },
            }
        }).collect()
    }

    pub fn step(&mut self, msg: RaftMessage) -> Vec<RaftMessage> {
        let mut out = Vec::new();
        if msg.term > self.current_term {
            self.current_term = msg.term; self.role = RaftRole::Follower; self.voted_for = None; self.leader_id = None;
        }
        match msg.payload {
            RaftPayload::RequestVote { last_log_index, last_log_term } => {
                let grant = msg.term >= self.current_term
                    && self.voted_for.map_or(true, |v| v == msg.from)
                    && (last_log_term > self.last_log_term() || (last_log_term == self.last_log_term() && last_log_index >= self.log.len() as u64));
                if grant { self.voted_for = Some(msg.from); }
                out.push(RaftMessage { from: self.id, to: msg.from, term: self.current_term, payload: RaftPayload::RequestVoteResponse { vote_granted: grant } });
            }
            RaftPayload::RequestVoteResponse { vote_granted } => {
                if self.role == RaftRole::Candidate && msg.term == self.current_term && vote_granted {
                    self.votes_received += 1;
                    if self.votes_received > (self.peers.len() + 1) / 2 { out.extend(self.become_leader()); }
                }
            }
            RaftPayload::AppendEntries { leader_id, prev_log_index, prev_log_term, entries, leader_commit } => {
                if msg.term < self.current_term {
                    out.push(RaftMessage { from: self.id, to: msg.from, term: self.current_term, payload: RaftPayload::AppendEntriesResponse { success: false, match_index: 0 } });
                    return out;
                }
                self.leader_id = Some(leader_id);
                if self.role == RaftRole::Candidate { self.role = RaftRole::Follower; }
                let ok = prev_log_index == 0 || (prev_log_index <= self.log.len() as u64 && self.log[(prev_log_index-1) as usize].term == prev_log_term);
                if !ok {
                    out.push(RaftMessage { from: self.id, to: msg.from, term: self.current_term, payload: RaftPayload::AppendEntriesResponse { success: false, match_index: 0 } });
                    return out;
                }
                let mut ci = prev_log_index;
                for e in entries { ci += 1; if ci > self.log.len() as u64 { self.log.push(e); } else { self.log[(ci-1) as usize] = e; } }
                if leader_commit > self.commit_index { self.commit_index = leader_commit.min(self.log.len() as u64); self.apply_committed_entries(); }
                out.push(RaftMessage { from: self.id, to: msg.from, term: self.current_term, payload: RaftPayload::AppendEntriesResponse { success: true, match_index: self.log.len() as u64 } });
            }
            RaftPayload::AppendEntriesResponse { success, match_index } => {
                if self.role == RaftRole::Leader && msg.term == self.current_term {
                    if success {
                        self.match_index.insert(msg.from, match_index);
                        self.next_index.insert(msg.from, match_index + 1);
                        for n in (self.commit_index + 1)..=(self.log.len() as u64) {
                            if self.match_index.values().filter(|&&m| m >= n).count() > (self.peers.len() + 1) / 2 && self.log[(n-1) as usize].term == self.current_term {
                                self.commit_index = n; self.apply_committed_entries();
                            }
                        }
                    } else {
                        let cur = *self.next_index.get(&msg.from).unwrap_or(&2);
                        if cur > 1 { self.next_index.insert(msg.from, cur - 1); out.extend(self.broadcast_append_entries()); }
                    }
                }
            }
        }
        out
    }

    pub async fn start(self: Arc<Self>, listen_addr: String) {
        let listener = TcpListener::bind(&listen_addr).await
            .expect("Failed to bind Raft listener");
        info!("[Raft {}] Listening on {}", self.id, listen_addr);
        loop {
            match listener.accept().await {
                Ok((mut stream, _)) => {
                    let node = self.clone();
                    tokio::spawn(async move {
                        let mut node = node;
                        if let Err(e) = RaftTransport::handle_incoming_message(&mut node, &mut stream).await {
                            warn!("[Raft {}] handle error: {}", node.id, e);
                        }
                    });
                }
                Err(e) => warn!("[Raft] accept error: {}", e),
            }
        }
    }

    pub async fn replicate_to_peers(&self) -> Vec<Result<RaftMessage, DistributedError>> {
        let mut results = Vec::new();
        for &p in &self.peers {
            if let Some(addr) = self.peer_addresses.get(&p) {
                let hb = RaftMessage {
                    from: self.id, to: p, term: self.current_term,
                    payload: RaftPayload::AppendEntries {
                        leader_id: self.id, prev_log_index: 0, prev_log_term: 0,
                        entries: vec![], leader_commit: self.commit_index,
                    },
                };
                results.push(RaftTransport::send_to_peer(addr, &hb).await);
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raft_append_entry() {
        let mut r = RaftNode::new(1, vec![2, 3]);
        assert_eq!(r.append_entry("X", vec![1,2,3]).unwrap(), 0);
        assert_eq!(r.log_len(), 1);
    }

    #[test]
    fn test_raft_commit() {
        let mut r = RaftNode::new(1, vec![2]);
        r.append_entry("I", vec![]).unwrap();
        r.append_entry("D", vec![]).unwrap();
        r.commit_up_to(1);
        assert_eq!(r.commit_index, 1);
    }

    #[test]
    fn test_leader_election_quorum() {
        let mut n1 = RaftNode::new(1, vec![2, 3]);
        let mut n2 = RaftNode::new(2, vec![1, 3]);
        let mut n3 = RaftNode::new(3, vec![1, 2]);
        let cmds = Arc::new(Mutex::new(Vec::new()));
        let c = cmds.clone();
        n3 = n3.with_commit_callback(move |e| { c.lock().push(e.command.clone()); });
        let msgs = n1.start_election();
        assert_eq!(n1.role, RaftRole::Candidate);
        let r2 = n2.step(msgs[0].clone());
        let _r3 = n3.step(msgs[1].clone());
        let a1 = n1.step(r2[0].clone());
        assert_eq!(n1.role, RaftRole::Leader);
        assert_eq!(a1.len(), 2);
        n1.append_entry("INSERT_DOC:papers:uuid", vec![42]).unwrap();
        let am = n1.broadcast_append_entries();
        let res2 = n2.step(am[0].clone());
        let res3 = n3.step(am[1].clone());
        assert_eq!(n2.log_len(), 1);
        assert_eq!(n3.log_len(), 1);
        n1.step(res2[0].clone());
        n1.step(res3[0].clone());
        assert_eq!(n1.commit_index, 1);
        let fhb = n1.broadcast_append_entries();
        n3.step(fhb[1].clone());
        assert_eq!(n3.commit_index, 1);
        assert_eq!(cmds.lock().len(), 1);
        assert_eq!(cmds.lock()[0], "INSERT_DOC:papers:uuid");
    }
}