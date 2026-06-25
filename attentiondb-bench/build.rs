fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "milvus")]
    tonic_build::configure()
        .build_client(true)
        .compile(
            &["proto/milvus.proto", "proto/common.proto", "proto/schema.proto"],
            &["proto/"],
        )?;
    Ok(())
}
