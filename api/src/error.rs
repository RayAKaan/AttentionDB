use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("gRPC error: {0}")]
    Grpc(String),

    #[error("REST error: {0}")]
    Rest(String),

    #[error("Client error: {0}")]
    Client(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}
