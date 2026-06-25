use attentiondb_storage::record::Record;
use attentiondb_storage::sstable::{SSTableReader, SSTableWriter};
use rand::Rng;
use tempfile::tempdir;

#[test]
fn write_and_read_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.sst");

    let mut writer = SSTableWriter::create(&path).unwrap();
    for i in 0..100u32 {
        let rec = Record::new(i.to_string().into_bytes(), vec![i as u8]);
        writer.append(&rec).unwrap();
    }
    writer.finish().unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    let mut count = 0;
    for res in reader.iter().unwrap() {
        let _ = res.unwrap();
        count += 1;
    }
    assert_eq!(count, 100);
}

#[test]
fn detect_crc_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.sst");

    let mut writer = SSTableWriter::create(&path).unwrap();
    writer
        .append(&Record::new(b"k".to_vec(), b"v".to_vec()))
        .unwrap();
    writer.finish().unwrap();

    // flip a byte in the payload to corrupt CRC
    let mut data = std::fs::read(&path).unwrap();
    let len = data.len();
    if len > 10 {
        data[len - 1] ^= 0x01;
    }
    std::fs::write(&path, &data).unwrap();

    let res = SSTableReader::open(&path);
    assert!(res.is_err());
}

#[test]
fn fallback_to_v1_format() {
    // v1 format was raw bincode payload; write that directly and ensure reader can read it
    let dir = tempdir().unwrap();
    let path = dir.path().join("v1.sst");

    let mut file = std::fs::File::create(&path).unwrap();
    let v: Vec<Record> = (0..10)
        .map(|i| Record::new(i.to_string().into_bytes(), vec![i as u8]))
        .collect();
    let payload = bincode::serialize(&v).unwrap();
    std::io::Write::write_all(&mut file, &payload).unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    let mut count = 0;
    for res in reader.iter().unwrap() {
        res.unwrap();
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn large_payloads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.sst");

    let mut writer = SSTableWriter::create(&path).unwrap();
    // 1MB payloads
    let mut rng = rand::thread_rng();
    for i in 0..10 {
        let mut v = vec![0u8; 1024 * 1024];
        rng.fill(&mut v[..]);
        writer
            .append(&Record::new(i.to_string().into_bytes(), v))
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    let mut count = 0;
    for res in reader.iter().unwrap() {
        res.unwrap();
        count += 1;
    }
    assert_eq!(count, 10);
}
