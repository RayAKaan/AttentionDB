use crate::persistence::error::PersistenceError;
use std::path::Path;
use std::fs::File;
use std::io::{Read, Write};

pub fn compact_index(dir: &Path) -> Result<usize, PersistenceError> {
    let vectors_path = dir.join("vectors.bin");
    if !vectors_path.exists() {
        return Err(PersistenceError::IndexNotFound(vectors_path.to_string_lossy().to_string()));
    }

    let mut file = File::open(&vectors_path)?;
    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;

    let mut vectors = Vec::with_capacity(count);
    for _ in 0..count {
        let mut id_buf = [0u8; 8];
        file.read_exact(&mut id_buf)?;
        let id = u64::from_le_bytes(id_buf);

        let mut dim_buf = [0u8; 4];
        file.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut val_buf = [0u8; 4];
            file.read_exact(&mut val_buf)?;
            vec.push(f32::from_le_bytes(val_buf));
        }
        vectors.push((id, vec));
    }

    let temp_path = dir.join("vectors.bin.compact");
    let mut out_file = File::create(&temp_path)?;
    out_file.write_all(&(vectors.len() as u64).to_le_bytes())?;

    for (id, vec) in &vectors {
        out_file.write_all(&id.to_le_bytes())?;
        out_file.write_all(&(vec.len() as u32).to_le_bytes())?;
        for val in vec {
            out_file.write_all(&val.to_le_bytes())?;
        }
    }
    out_file.flush()?;

    std::fs::rename(&temp_path, &vectors_path)?;

    Ok(vectors.len())
}
