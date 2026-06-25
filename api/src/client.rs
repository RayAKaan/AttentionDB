use tonic::transport::Channel;

pub mod attentiondb {
    tonic::include_proto!("attentiondb");
}

use attentiondb::attention_db_client::AttentionDbClient;
use attentiondb::AttendRequest;

pub struct AttentionDBClient {
    client: AttentionDbClient<Channel>,
}

impl AttentionDBClient {
    pub async fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let addr = if !addr.starts_with("http") {
            format!("http://{}", addr)
        } else {
            addr.to_string()
        };

        let channel = Channel::from_shared(addr)?.connect().await?;
        let client = AttentionDbClient::new(channel);
        Ok(Self { client })
    }

    pub async fn attend(
        &mut self,
        collection: &str,
        query: &str,
        heads: Vec<String>,
        top_k: u32,
    ) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
        let request = tonic::Request::new(AttendRequest {
            collection: collection.to_string(),
            query: query.to_string(),
            heads,
            top_k,
            min_weight: 0.01,
            temporal_decay: None,
            offset: 0,
            hybrid: false,
            bm25_weight: 0.3,
            vector_weight: 0.7,
            query_text: String::new(),
        });

        let response = self.client.attend(request).await?;
        let results = response.into_inner().results
            .into_iter()
            .map(|r| (r.id, r.score))
            .collect();

        Ok(results)
    }

    pub async fn health(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let request = tonic::Request::new(attentiondb::HealthRequest {});
        let response = self.client.health_check(request).await?;
        Ok(response.into_inner().status)
    }
}
