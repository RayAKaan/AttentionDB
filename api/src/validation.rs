//! Request Validation — Input sanitization, size limits, parameter bounds
//!
//! Prevents abuse, injection, and resource exhaustion attacks.

use tonic::Status;
use attentiondb_core::constants::{
    MAX_COLLECTION_NAME_LEN, MAX_HEADS, MAX_TOP_K,
    MAX_DIMENSION, MAX_FIELDS, MAX_FIELD_VALUE_BYTES,
};

/// Maximum allowed pagination page size.
pub const MAX_PAGE_SIZE: u32 = 1000;
/// Maximum allowed pagination page number.
pub const MAX_PAGE_NUMBER: u32 = 1000;
/// Maximum request body size in bytes (REST).
pub const MAX_REQUEST_BODY_BYTES: usize = 10_485_760; // 10 MB

/// Validate a collection name.
pub fn validate_collection_name(name: &str) -> Result<(), Status> {
    if name.is_empty() {
        return Err(Status::invalid_argument("Collection name cannot be empty"));
    }
    if name.len() > MAX_COLLECTION_NAME_LEN {
        return Err(Status::invalid_argument(
            format!("Collection name too long: {} chars (max {})", name.len(), MAX_COLLECTION_NAME_LEN)
        ));
    }
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return Err(Status::invalid_argument(
            format!("Collection name contains invalid characters: '{}'. Only alphanumeric, underscore, and hyphen allowed.", name)
        ));
    }
    Ok(())
}

/// Validate top_k parameter.
pub fn validate_top_k(top_k: u32) -> Result<(), Status> {
    if top_k == 0 {
        return Err(Status::invalid_argument("top_k must be at least 1"));
    }
    if top_k > MAX_TOP_K {
        return Err(Status::invalid_argument(
            format!("top_k too large: {} (max {})", top_k, MAX_TOP_K)
        ));
    }
    Ok(())
}

/// Validate pagination parameters.
pub fn validate_page(page: u32) -> Result<(), Status> {
    if page == 0 {
        return Err(Status::invalid_argument("page must be at least 1"));
    }
    if page > MAX_PAGE_NUMBER {
        return Err(Status::invalid_argument(
            format!("page number too large: {} (max {})", page, MAX_PAGE_NUMBER)
        ));
    }
    Ok(())
}

/// Validate pagination page size.
pub fn validate_page_size(page_size: u32) -> Result<(), Status> {
    if page_size == 0 {
        return Err(Status::invalid_argument("page_size must be at least 1"));
    }
    if page_size > MAX_PAGE_SIZE {
        return Err(Status::invalid_argument(
            format!("page_size too large: {} (max {})", page_size, MAX_PAGE_SIZE)
        ));
    }
    Ok(())
}

/// Validate head names.
pub fn validate_heads(heads: &[String]) -> Result<(), Status> {
    if heads.len() > MAX_HEADS {
        return Err(Status::invalid_argument(
            format!("Too many heads: {} (max {})", heads.len(), MAX_HEADS)
        ));
    }
    for head in heads {
        if head.is_empty() || head.len() > 64 {
            return Err(Status::invalid_argument(
                format!("Invalid head name: '{}' (must be 1-64 chars)", head)
            ));
        }
        if !head.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(Status::invalid_argument(
                format!("Head name '{}' contains invalid characters", head)
            ));
        }
    }
    Ok(())
}

/// Validate a parsed vector has reasonable dimensions.
pub fn validate_vector_dimension(dim: usize) -> Result<(), Status> {
    if dim == 0 {
        return Err(Status::invalid_argument("Vector dimension cannot be 0"));
    }
    if dim > MAX_DIMENSION {
        return Err(Status::invalid_argument(
            format!("Vector dimension too large: {} (max {})", dim, MAX_DIMENSION)
        ));
    }
    Ok(())
}

/// Validate document fields count and sizes.
pub fn validate_fields(fields: &std::collections::HashMap<String, String>) -> Result<(), Status> {
    if fields.len() > MAX_FIELDS {
        return Err(Status::invalid_argument(
            format!("Too many fields: {} (max {})", fields.len(), MAX_FIELDS)
        ));
    }
    for (key, value) in fields {
        if key.is_empty() || key.len() > 256 {
            return Err(Status::invalid_argument(
                format!("Invalid field name length: {} (must be 1-256)", key.len())
            ));
        }
        if value.len() > MAX_FIELD_VALUE_BYTES {
            return Err(Status::invalid_argument(
                format!("Field '{}' value too large: {} bytes (max {})", key, value.len(), MAX_FIELD_VALUE_BYTES)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_collection_names() {
        assert!(validate_collection_name("papers").is_ok());
        assert!(validate_collection_name("my_collection_2026").is_ok());
        assert!(validate_collection_name("test-collection").is_ok());
    }

    #[test]
    fn test_invalid_collection_names() {
        assert!(validate_collection_name("").is_err());
        assert!(validate_collection_name("has spaces").is_err());
        assert!(validate_collection_name("has;semicolons").is_err());
        assert!(validate_collection_name(&"x".repeat(200)).is_err());
    }

    #[test]
    fn test_valid_top_k() {
        assert!(validate_top_k(1).is_ok());
        assert!(validate_top_k(100).is_ok());
        assert!(validate_top_k(10_000).is_ok());
    }

    #[test]
    fn test_invalid_top_k() {
        assert!(validate_top_k(0).is_err());
        assert!(validate_top_k(10_001).is_err());
    }

    #[test]
    fn test_valid_heads() {
        assert!(validate_heads(&["semantic".into(), "temporal".into()]).is_ok());
        assert!(validate_heads(&[]).is_ok());
    }

    #[test]
    fn test_invalid_heads() {
        assert!(validate_heads(&["has spaces".into()]).is_err());
        assert!(validate_heads(&["".into()]).is_err());
        let too_many: Vec<String> = (0..33).map(|i| format!("h{}", i)).collect();
        assert!(validate_heads(&too_many).is_err());
    }

    #[test]
    fn test_vector_dimension() {
        assert!(validate_vector_dimension(64).is_ok());
        assert!(validate_vector_dimension(4096).is_ok());
        assert!(validate_vector_dimension(0).is_err());
        assert!(validate_vector_dimension(5000).is_err());
    }

    #[test]
    fn test_pagination_validation() {
        assert!(validate_page(1).is_ok());
        assert!(validate_page(100).is_ok());
        assert!(validate_page(MAX_PAGE_NUMBER).is_ok());
        assert!(validate_page(0).is_err());
        assert!(validate_page(MAX_PAGE_NUMBER + 1).is_err());

        assert!(validate_page_size(1).is_ok());
        assert!(validate_page_size(10).is_ok());
        assert!(validate_page_size(MAX_PAGE_SIZE).is_ok());
        assert!(validate_page_size(0).is_err());
        assert!(validate_page_size(MAX_PAGE_SIZE + 1).is_err());
    }
}
