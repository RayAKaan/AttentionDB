use tonic::{Request, Response, Status};

pub mod attentiondb {
    tonic::include_proto!("attentiondb");
}

use attentiondb::attention_db_server::AttentionDb;
use attentiondb::{
    AttendRequest, AttendResponse, InsertRequest, InsertResponse,
    DeleteRequest, DeleteResponse, HealthRequest, HealthResponse,
};

pub struct AttentionDBService;

#[tonic::async_trait]
impl AttentionDb for AttentionDBService {
    async fn attend(
        &self,
        request: Request<AttendRequest>,
    ) -> Result<Response<AttendResponse>, Status> {
        let req = request.into_inner();
        println!("[gRPC] Attend: collection={}, query=\"{}\", heads={:?}, top_k={}",
                 req.collection, req.query, req.heads, req.top_k);

        let response = AttendResponse {
            results: vec![],
            latency_ms: 1.2,
            effective_sample_size: 4.7,
        };
        Ok(Response::new(response))
    }

    async fn insert(
        &self,
        _request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let response = InsertResponse {
            id: uuid::Uuid::new_v4().to_string(),
            success: true,
        };
        Ok(Response::new(response))
    }

    async fn delete(
        &self,
        _request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let response = DeleteResponse { success: true };
        Ok(Response::new(response))
    }

    async fn health_check(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.5.0".to_string(),
        };
        Ok(Response::new(response))
    }
}
