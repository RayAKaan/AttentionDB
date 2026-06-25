use crate::error::DistributedError;
use crate::raft::RaftMessage;

pub trait RaftTransport: Send + Sync {
    fn send_message(&self, peer_addr: &str, msg: &RaftMessage) -> Result<RaftMessage, DistributedError>;
}

pub struct HttpRaftTransport {
    pub client: reqwest::blocking::Client,
}

impl HttpRaftTransport {
    pub fn new() -> Self {
        Self { client: reqwest::blocking::Client::new() }
    }
}

impl Default for HttpRaftTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftTransport for HttpRaftTransport {
    fn send_message(&self, peer_addr: &str, msg: &RaftMessage) -> Result<RaftMessage, DistributedError> {
        let url = format!("http://{}/raft", peer_addr);
        let body = serde_json::to_vec(msg)
            .map_err(|e| DistributedError::Network(format!("serialization: {}", e)))?;
        let resp = self.client.post(&url)
            .body(body)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .map_err(|e| DistributedError::Network(format!("request: {}", e)))?;
        let response: RaftMessage = serde_json::from_slice(&resp.bytes()
            .map_err(|e| DistributedError::Network(format!("read response: {}", e)))?)
            .map_err(|e| DistributedError::Network(format!("deserialize response: {}", e)))?;
        Ok(response)
    }
}
