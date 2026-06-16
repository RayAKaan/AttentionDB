use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HeadType {
    Semantic,
    Structural,
    Temporal,
    Relational,
    FieldSpecific(String),
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    pub name: String,
    pub head_type: HeadType,
    pub dim: usize,
    pub weight: f32,
    pub fields: Vec<String>,
    pub settings: Option<attentiondb_hnsw::CollectionSettings>,
}

impl HeadConfig {
    pub fn new(name: &str, head_type: HeadType, dim: usize) -> Self {
        Self {
            name: name.to_string(),
            head_type,
            dim,
            weight: 1.0,
            fields: vec![],
            settings: None,
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_fields(mut self, fields: Vec<String>) -> Self {
        self.fields = fields;
        self
    }

    pub fn with_settings(mut self, settings: attentiondb_hnsw::CollectionSettings) -> Self {
        self.settings = Some(settings);
        self
    }
}
