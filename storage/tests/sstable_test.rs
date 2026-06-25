use attentiondb_storage::sstable::{SSTableReader, SSTableWriter};
use tempfile::tempdir;

#[test]
fn write_and_read_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.sst");

    let mut writer = SSTableWriter::new(&path).unwrap();
    for i in 0..100u32 {
        let key = i.to_string().into_bytes();
        let val = vec![i as u8];
        writer.append(key, val).unwrap();
    }
    writer.flush().unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    assert_eq!(reader.iter().count(), 100);
}

#[test]
fn detect_crc_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.sst");

    let mut writer = SSTableWriter::new(&path).unwrap();
    writer.append(b"k".to_vec(), b"v".to_vec()).unwrap();
    writer.flush().unwrap();

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
fn large_payloads() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.sst");

    let mut writer = SSTableWriter::new(&path).unwrap();
    for i in 0..10u32 {
        let key = i.to_string().into_bytes();
        let val = vec![i as u8; 1024 * 1024];
        writer.append(key, val).unwrap();
    }
    writer.flush().unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    assert_eq!(reader.iter().count(), 10);
}

#[test]
fn get_by_key() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("get.sst");

    let mut writer = SSTableWriter::new(&path).unwrap();
    writer.append(b"alpha".to_vec(), b"one".to_vec()).unwrap();
    writer.append(b"beta".to_vec(), b"two".to_vec()).unwrap();
    writer.flush().unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    let entry = reader.get(b"alpha").unwrap();
    assert_eq!(entry.value, b"one");
    assert!(reader.get(b"gamma").is_none());
}
