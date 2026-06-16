use tonic::{Request, Response, Status};

pub mod attentiondb {
    tonic::include_proto!("attentiondb");
}

use attentiondb::attention_db_server::AttentionDb;
use attentiondb::{
    AttendRequest, AttendResponse, InsertRequest, InsertResponse,
    DeleteRequest, DeleteResponse, HealthRequest, HealthResponse,
    CreateCollectionRequest, CreateCollectionResponse,
    GetCollectionSettingsRequest, GetCollectionSettingsResponse,
    AlterCollectionRequest, AlterCollectionResponse,
    CollectionSettings,
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

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        println!("[gRPC] CreateCollection: collection={}", req.collection);

        let mut hnsw_settings = attentiondb_hnsw::CollectionSettings::default();
        if let Some(s) = req.settings {
            hnsw_settings.ef_search = s.ef_search as usize;
            hnsw_settings.ef_construction = s.ef_construction as usize;
            hnsw_settings.max_nb_connection = s.max_connections as usize;
            hnsw_settings.similarity_metric = s.similarity;
            hnsw_settings.enable_exact_reranking = s.exact_rerank;
            hnsw_settings.enable_gpu_fusion = s.enable_gpu_fusion;
        }

        hnsw_settings.validate().map_err(|e| Status::invalid_argument(e))?;

        let field_info: Vec<String> = req.fields.iter()
            .map(|f| format!("{}:{}", f.name, f.r#type))
            .collect();

        let msg = format!(
            "Created collection '{}' with fields [{}] and settings (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
            req.collection,
            field_info.join(", "),
            hnsw_settings.ef_search,
            hnsw_settings.ef_construction,
            hnsw_settings.max_nb_connection,
            hnsw_settings.similarity_metric,
            hnsw_settings.enable_exact_reranking,
        );

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            message: msg,
        }))
    }

    async fn get_collection_settings(
        &self,
        _request: Request<GetCollectionSettingsRequest>,
    ) -> Result<Response<GetCollectionSettingsResponse>, Status> {
        let settings = CollectionSettings {
            ef_search: 64,
            ef_construction: 400,
            max_connections: 16,
            similarity: "cosine".to_string(),
            exact_rerank: true,
            enable_gpu_fusion: false,
        };
        Ok(Response::new(GetCollectionSettingsResponse {
            settings: Some(settings),
        }))
    }

    async fn alter_collection(
        &self,
        request: Request<AlterCollectionRequest>,
    ) -> Result<Response<AlterCollectionResponse>, Status> {
        let req = request.into_inner();
        println!("[gRPC] AlterCollection: collection={}", req.collection);

        let s = req.settings.unwrap_or_default();
        let hnsw_settings = attentiondb_hnsw::CollectionSettings {
            ef_search: s.ef_search as usize,
            ef_construction: s.ef_construction as usize,
            max_nb_connection: s.max_connections as usize,
            similarity_metric: s.similarity,
            enable_exact_reranking: s.exact_rerank,
            enable_gpu_fusion: s.enable_gpu_fusion,
        };

        hnsw_settings.validate().map_err(|e| Status::invalid_argument(e))?;

        let msg = format!(
            "Altered collection '{}' settings to (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
            req.collection,
            hnsw_settings.ef_search,
            hnsw_settings.ef_construction,
            hnsw_settings.max_nb_connection,
            hnsw_settings.similarity_metric,
            hnsw_settings.enable_exact_reranking,
        );

        Ok(Response::new(AlterCollectionResponse {
            success: true,
            message: msg,
        }))
    }
}
