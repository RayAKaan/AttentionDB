use crate::workload::{Document, HeadEmbedding, HeadType, Query};
use crate::workload::difficulty::DifficultyLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnnBenchmarkDataset {
    Sift128Euclidean,
    GloVe100Angular,
    NYTimes256Angular,
    Gist960Euclidean,
    MsMarco,
}

impl AnnBenchmarkDataset {
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Sift128Euclidean  => "sift-128-euclidean.hdf5",
            Self::GloVe100Angular   => "glove-100-angular.hdf5",
            Self::NYTimes256Angular => "nytimes-256-angular.hdf5",
            Self::Gist960Euclidean  => "gist-960-euclidean.hdf5",
            Self::MsMarco           => "msmarco-v2.hdf5",
        }
    }

    pub fn fvecs_filename(&self) -> &'static str {
        match self {
            Self::Sift128Euclidean  => "sift-128-euclidean.fvecs",
            Self::GloVe100Angular   => "glove-100-angular.fvecs",
            Self::NYTimes256Angular => "nytimes-256-angular.fvecs",
            Self::Gist960Euclidean  => "gist-960-euclidean.fvecs",
            Self::MsMarco           => "msmarco.fvecs",
        }
    }

    pub fn metric(&self) -> crate::workload::DistanceMetric {
        match self {
            Self::Sift128Euclidean | Self::Gist960Euclidean => crate::workload::DistanceMetric::Euclidean,
            Self::GloVe100Angular | Self::NYTimes256Angular => crate::workload::DistanceMetric::Cosine,
            Self::MsMarco => crate::workload::DistanceMetric::DotProduct,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::Sift128Euclidean  => 128,
            Self::GloVe100Angular   => 100,
            Self::NYTimes256Angular => 256,
            Self::Gist960Euclidean  => 960,
            Self::MsMarco           => 768,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnBenchmarkLoaded {
    pub train_vectors: Vec<Vec<f32>>,
    pub test_vectors: Vec<Vec<f32>>,
    pub neighbors: Vec<Vec<usize>>,
    pub distances: Vec<Vec<f32>>,
    pub dataset: AnnBenchmarkDataset,
}

impl AnnBenchmarkLoaded {
    pub fn load(path: &Path, dataset: AnnBenchmarkDataset) -> anyhow::Result<Self> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext {
            "hdf5" => Self::load_hdf5(path, dataset),
            "fvecs" => Self::load_fvecs(path, dataset),
            _ => anyhow::bail!("Unsupported dataset format: {}. Use .hdf5 or .fvecs", ext),
        }
    }

    #[cfg(feature = "hdf5")]
    fn load_hdf5(path: &Path, dataset: AnnBenchmarkDataset) -> anyhow::Result<Self> {
        let file = hdf5::File::open(path)?;

        let train: hdf5::Dataset = file.dataset("train")?;
        let test: hdf5::Dataset  = file.dataset("test")?;
        let neighbors: hdf5::Dataset = file.dataset("neighbors")?;
        let distances: hdf5::Dataset = file.dataset("distances")?;

        let train_flat: Vec<f32> = train.read_raw()?;
        let test_flat: Vec<f32>  = test.read_raw()?;
        let neigh_flat: Vec<i32> = neighbors.read_raw()?;
        let dist_flat: Vec<f32>  = distances.read_raw()?;

        let dim = dataset.dimension();
        let n_train = train_flat.len() / dim;
        let n_test  = test_flat.len() / dim;
        let k_gt    = neigh_flat.len() / n_test;

        let train_vectors: Vec<Vec<f32>> = train_flat
            .chunks_exact(dim)
            .map(|c| c.to_vec())
            .collect();

        let test_vectors: Vec<Vec<f32>> = test_flat
            .chunks_exact(dim)
            .map(|c| c.to_vec())
            .collect();

        let neighbors: Vec<Vec<usize>> = neigh_flat
            .chunks_exact(k_gt)
            .map(|c| c.iter().map(|&i| i as usize).collect())
            .collect();

        let distances: Vec<Vec<f32>> = dist_flat
            .chunks_exact(k_gt)
            .map(|c| c.to_vec())
            .collect();

        tracing::info!(
            "Loaded HDF5 {}: {} train, {} test, {}D, GT@{}",
            path.display(), n_train, n_test, dim, k_gt
        );

        Ok(Self { train_vectors, test_vectors, neighbors, distances, dataset })
    }

    #[cfg(not(feature = "hdf5"))]
    fn load_hdf5(_path: &Path, _dataset: AnnBenchmarkDataset) -> anyhow::Result<Self> {
        anyhow::bail!("HDF5 support not enabled (feature 'hdf5' required). \
                      Use .fvecs format or enable the hdf5 feature.")
    }

    fn load_fvecs(path: &Path, dataset: AnnBenchmarkDataset) -> anyhow::Result<Self> {
        let data = std::fs::read(path)?;
        let dim = dataset.dimension();
        let vector_byte_size = 4 + dim * 4; // 4 bytes for dim + dim * 4 bytes for floats

        if data.len() < vector_byte_size {
            anyhow::bail!("File too small for fvecs format");
        }

        let mut vectors = Vec::new();
        let mut offset = 0;

        while offset + vector_byte_size <= data.len() {
            let vec_dim = u32::from_le_bytes(
                data[offset..offset + 4].try_into().unwrap()
            ) as usize;
            offset += 4;

            if vec_dim != dim {
                anyhow::bail!("Dimension mismatch: expected {} but got {} at offset {}", dim, vec_dim, offset - 4);
            }

            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                let val = f32::from_le_bytes(
                    data[offset..offset + 4].try_into().unwrap()
                );
                vec.push(val);
                offset += 4;
            }
            vectors.push(vec);
        }

        // Split 90% train, 10% test
        let split = (vectors.len() as f64 * 0.9) as usize;
        let train_vectors = vectors[..split].to_vec();
        let test_vectors = vectors[split..].to_vec();

        // Brute-force compute ground truth for queries
        let neighbors: Vec<Vec<usize>> = test_vectors.iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = train_vectors.iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let dist: f32 = q.iter().zip(d.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (i, dist)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.iter().take(100).map(|&(i, _)| i).collect()
            })
            .collect();

        let distances: Vec<Vec<f32>> = test_vectors.iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = train_vectors.iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let dist: f32 = q.iter().zip(d.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (i, dist)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.iter().take(100).map(|&(_, d)| d).collect()
            })
            .collect();

        tracing::info!(
            "Loaded fvecs {}: {} train, {} test, {}D",
            path.display(), train_vectors.len(), test_vectors.len(), dim
        );

        Ok(Self { train_vectors, test_vectors, neighbors, distances, dataset })
    }

    pub fn into_documents_and_queries(self) -> (Vec<Document>, Vec<Query>) {
        let documents: Vec<Document> = self.train_vectors
            .into_iter()
            .enumerate()
            .map(|(i, vec)| Document {
                id: format!("doc_{}", i),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec,
                }],
                metadata: HashMap::new(),
                timestamp: None,
            })
            .collect();

        let queries: Vec<Query> = self.test_vectors
            .into_iter()
            .zip(self.neighbors.into_iter())
            .enumerate()
            .map(|(i, (vec, gt_indices))| Query {
                id: format!("query_{}", i),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec,
                }],
                enabled_heads: vec![HeadType::Semantic],
                ground_truth: gt_indices
                    .iter()
                    .map(|&idx| format!("doc_{}", idx))
                    .collect(),
                difficulty: DifficultyLevel::Medium,
                failure_mode: None,
            })
            .collect();

        (documents, queries)
    }
}

pub fn download_ann_benchmark_datasets(data_dir: &Path) -> anyhow::Result<()> {
    let datasets = [
        ("sift-128-euclidean.hdf5", "http://ann-benchmarks.com/sift-128-euclidean.hdf5"),
    ];

    for (filename, url) in &datasets {
        let path = data_dir.join(filename);
        if path.exists() {
            tracing::info!("Dataset {} already exists, skipping", filename);
            continue;
        }

        tracing::info!("Downloading {}...", filename);
        let resp = reqwest::blocking::get(*url)?;
        let bytes = resp.bytes()?;
        std::fs::write(&path, &bytes)?;
        tracing::info!("Downloaded {} ({} bytes)", filename, bytes.len());
    }

    Ok(())
}
