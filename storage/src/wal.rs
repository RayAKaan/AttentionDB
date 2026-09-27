//! Authoritative segmented Write-Ahead Log (WAL v2) — Phase 1.
//!
//! This is the SINGLE authoritative mutation log for database logical state.
//! Design properties (see docs/wal.md):
//!
//! - **Segmented**: files named `WAL/<start_seq, zero-padded 20 digits>.wal`.
//!   Rotation at a configurable maximum segment size.
//! - **Framed**: every record is `[magic u32][len u32][body][crc32 u32]` where the
//!   CRC covers `magic|len|body`. Length-prefixed framing makes torn-tail recovery
//!   deterministic (the old format was a raw bincode stream with guessed boundaries).
//! - **Versioned**: every record carries `version`; an unknown version is a hard
//!   error — we never reinterpret a newer format as an older one.
//! - **Checksummed over the full record** (old WAL checksummed only the payload,
//!   so header corruption went undetected).
//! - **Strict sequence numbers**: seq increases by exactly 1 per record. Duplicate,
//!   regressing, or gapped sequences are corruption errors.
//! - **Torn tail vs mid-log corruption**: a truncated/garbage final frame is repaired
//!   by truncation (reported); corruption with valid data after it is fatal.
//! - **Durability modes**: `Sync` (fsync per append), `GroupCommit` (flush to OS page
//!   cache per append; process-crash safe, not machine-crash safe), `Async`
//!   (userspace buffered). These are documented guarantees, never overstated.
//!
//! Record payloads are opaque bytes defined by the engine layer; this module only
//! handles framing, sequencing, durability, and recovery mechanics.

use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Format version of the record body layout. Bump on any incompatible change.
pub const WAL_FORMAT_VERSION: u16 = 2;
/// Frame magic: "WAL2".
pub const FRAME_MAGIC: u32 = 0x5741_4C32;
/// Directory holding segments (relative to the database directory).
pub const WAL_DIR_NAME: &str = "WAL";

/// Logical operation kinds recorded in the WAL. Payload interpretation is the
/// engine's responsibility; the WAL guarantees only framing + integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RecordKind {
    BeginTxn = 1,
    TxnOp = 2,
    CommitTxn = 3,
    AbortTxn = 4,
    CreateCollection = 5,
    DropCollection = 6,
    AlterCollection = 7,
    InsertDocument = 8,
    UpdateDocument = 9,
    DeleteDocument = 10,
    UpsertDocument = 11,
    CheckpointMarker = 12,
}

impl RecordKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        use RecordKind::*;
        Some(match v {
            1 => BeginTxn,
            2 => TxnOp,
            3 => CommitTxn,
            4 => AbortTxn,
            5 => CreateCollection,
            6 => DropCollection,
            7 => AlterCollection,
            8 => InsertDocument,
            9 => UpdateDocument,
            10 => DeleteDocument,
            11 => UpsertDocument,
            12 => CheckpointMarker,
            _ => return None,
        })
    }
}

/// One durable mutation. Fields use fixed-width primitives with sentinels
/// (`0` / zero-uuid = "not applicable") rather than enums-with-variants so the
/// on-disk layout stays explicit and stable across bincode versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalRecord {
    pub version: u16,
    pub seq: u64,
    pub kind: u8,
    /// transaction id (0 = not part of a transaction)
    pub txn_id: u64,
    /// collection name ("" = not collection-scoped)
    pub collection: String,
    /// document uuid bytes (16, all-zero = n/a)
    pub doc_id: [u8; 16],
    /// internal numeric id (0 = n/a)
    pub numeric_id: u64,
    /// engine-defined payload (self-describing, versioned by `version`)
    pub payload: Vec<u8>,
}

impl WalRecord {
    pub fn new(seq: u64, kind: RecordKind) -> Self {
        Self {
            version: WAL_FORMAT_VERSION,
            seq,
            kind: kind as u8,
            txn_id: 0,
            collection: String::new(),
            doc_id: [0u8; 16],
            numeric_id: 0,
            payload: Vec::new(),
        }
    }

    pub fn kind(&self) -> Option<RecordKind> {
        RecordKind::from_u8(self.kind)
    }
}

/// Explicit durability semantics for appends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Durability {
    /// flush userspace buffers + fsync before acknowledging. Machine-crash durable.
    Sync,
    /// flush to OS page cache before acknowledging. Process-crash durable.
    GroupCommit,
    /// buffered in userspace only. Neither process- nor machine-crash durable.
    Async,
}

/// Result of replaying the log.
/// Durable WAL watermark record (Phase 3E E1). Written atomically at every
/// segment creation; absent for legacy databases (pre-adoption) and for a
/// brand-new empty database.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct WalState {
    pub format_version: u32,
    /// highest sequence number inside COMPLETED segments at write time
    /// (equal to the last seq before the active segment started; 0 while the
    /// first segment is still the active one)
    pub high_watermark: u64,
    /// sequence number the active segment was created at
    pub active_start: u64,
}

pub const WAL_STATE_FORMAT_VERSION: u32 = 1;
pub const WAL_STATE_FILE: &str = "wal-state.json";

/// Atomically persist the WAL state record (tmp -> rename -> fsync dir).
pub fn write_wal_state(wal_dir: &Path, state: &WalState) -> Result<(), crate::error::StorageError> {
    let tmp = wal_dir.join(format!("{WAL_STATE_FILE}.tmp"));
    let final_path = wal_dir.join(WAL_STATE_FILE);
    {
        let mut f = File::create(&tmp)?;
        serde_json::to_writer_pretty(&mut f, state)
            .map_err(|e| crate::error::StorageError::Serialization(e.to_string()))?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &final_path)?;
    crate::catalog::fsync_dir(wal_dir)?;
    Ok(())
}

/// Load the durable WAL state record.
/// - `Ok(None)` — no record (legacy database or brand-new): invariant not yet
///   authoritative; the open path proceeds with pre-E1 semantics.
/// - `Err` — a record exists but cannot be parsed: an integrity record that
///   cannot be trusted must refuse the open, never be ignored.
pub fn read_wal_state(wal_dir: &Path) -> Result<Option<WalState>, crate::error::StorageError> {
    let path = wal_dir.join(WAL_STATE_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)?;
    if bytes.is_empty() {
        return Err(crate::error::StorageError::corruption(
            path.display().to_string(),
            "WAL state record is empty",
        ));
    }
    serde_json::from_slice::<WalState>(&bytes)
        .map(Some)
        .map_err(|e| {
            crate::error::StorageError::corruption(
                path.display().to_string(),
                format!("WAL state record unreadable: {e}"),
            )
        })
}

#[derive(Debug, Default)]
pub struct ReplayOutcome {
    pub records: Vec<WalRecord>,
    /// trailing bytes were an incomplete/garbage final frame; they were truncated
    pub torn_tail: bool,
    /// highest sequence number observed (0 if empty)
    pub last_seq: u64,
    /// lowest sequence observed (0 if empty)
    pub first_seq: u64,
}

pub struct Wal {
    dir: PathBuf,
    durability: Durability,
    max_segment_bytes: u64,
    writer: Option<BufWriter<File>>,
    active_path: Option<PathBuf>,
    active_start_seq: u64,
    active_bytes: u64,
    next_seq: u64,
    last_synced_seq: u64,
}

impl Wal {
    /// Open (or create) the WAL directory and prepare for appends.
    ///
    /// Scans existing segments, validates naming, repairs nothing — repair happens
    /// in [`Wal::replay`] which is called by recovery before any appends.
    pub fn open(
        dir: &Path,
        durability: Durability,
        max_segment_bytes: u64,
    ) -> Result<Self, crate::error::StorageError> {
        std::fs::create_dir_all(dir)?;
        let wal = Self {
            dir: dir.to_path_buf(),
            durability,
            max_segment_bytes: max_segment_bytes.max(1024),
            writer: None,
            active_path: None,
            active_start_seq: 0,
            active_bytes: 0,
            next_seq: 1,
            last_synced_seq: 0,
        };
        Ok(wal)
    }

    fn segment_path(&self, start_seq: u64) -> PathBuf {
        self.dir.join(format!("{:020}.wal", start_seq))
    }

    fn list_segments(&self) -> Result<Vec<(u64, PathBuf)>, crate::error::StorageError> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("wal") {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    match stem.parse::<u64>() {
                        Ok(n) => out.push((n, p)),
                        Err(_) => {
                            return Err(crate::error::StorageError::corruption(
                                p.display().to_string(),
                                "WAL segment filename is not a sequence number",
                            ))
                        }
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Replay all records with `seq > min_seq_exclusive`, in sequence order.
    ///
    /// - Torn final frame (crash during append): truncated, `outcome.torn_tail = true`.
    /// - Mid-log corruption (invalid frame with valid frames after it, bad version,
    ///   duplicate/gapped sequence): hard [`StorageError::Corruption`] — never skipped.
    pub fn replay(
        &mut self,
        min_seq_exclusive: u64,
    ) -> Result<ReplayOutcome, crate::error::StorageError> {
        let segments = self.list_segments()?;
        let mut outcome = ReplayOutcome::default();
        let mut expected_seq: u64 = 0;
        let mut seen_any = false;

        for (seg_start, seg_path) in &segments {
            let mut buf = Vec::new();
            File::open(seg_path)?.read_to_end(&mut buf)?;
            let file_name = seg_path.display().to_string();
            let mut offset = 0usize;
            let mut first_in_segment = true;

            while offset < buf.len() {
                match Self::parse_frame(&buf[offset..]) {
                    Ok((record, frame_len)) => {
                        // Version gate: never reinterpret unknown formats.
                        if record.version != WAL_FORMAT_VERSION {
                            return Err(crate::error::StorageError::UnsupportedVersion {
                                file: file_name.clone(),
                                found: record.version as u32,
                                supported: WAL_FORMAT_VERSION as u32,
                            });
                        }
                        // Segments are named after the sequence number of their first
                        // record, and sequences are strictly +1. Validate both.
                        if first_in_segment {
                            if record.seq != *seg_start {
                                return Err(crate::error::StorageError::corruption(
                                    file_name.clone(),
                                    format!(
                                        "segment starts at seq={} but first record has seq={}",
                                        seg_start, record.seq
                                    ),
                                ));
                            }
                            if seen_any && record.seq != expected_seq + 1 {
                                return Err(crate::error::StorageError::corruption(
                                    file_name.clone(),
                                    format!(
                                        "sequence gap between segments at seq={} (expected {})",
                                        record.seq,
                                        expected_seq + 1
                                    ),
                                ));
                            }
                            first_in_segment = false;
                        } else if record.seq != expected_seq + 1 {
                            return Err(crate::error::StorageError::corruption(
                                file_name.clone(),
                                format!(
                                    "sequence gap/regression at seq={} (expected {})",
                                    record.seq,
                                    expected_seq + 1
                                ),
                            ));
                        }
                        seen_any = true;
                        expected_seq = record.seq;
                        outcome.last_seq = record.seq;
                        if record.seq > min_seq_exclusive {
                            if outcome.records.is_empty() {
                                outcome.first_seq = record.seq;
                            }
                            outcome.records.push(record);
                        }
                        offset += frame_len;
                    }
                    Err(FrameError::Truncated) => {
                        // Torn tail: everything from `offset` onward is an incomplete
                        // frame (crash during append). Truncate to the last good frame.
                        Self::truncate_file(seg_path, offset)?;
                        outcome.torn_tail = true;
                        tracing::warn!(
                            file = %file_name,
                            truncated_bytes = buf.len() - offset,
                            "WAL torn tail detected and truncated (recoverable)"
                        );
                        break;
                    }
                    Err(FrameError::Corrupt(detail)) => {
                        // Classification:
                        // - valid frame(s) after the bad one → mid-log corruption: FATAL
                        // - bad tail consisting of all zero bytes → crash zero-padding
                        //   (common on journaled filesystems after a torn write): repair
                        // - any other non-zero garbage tail → corruption: FATAL.
                        // We never silently skip arbitrary corrupted records.
                        let rest = &buf[offset..];
                        let zero_padded = rest.iter().all(|&b| b == 0);
                        if Self::has_valid_frame_after(&buf, offset + 1) || !zero_padded {
                            return Err(crate::error::StorageError::corruption(
                                file_name.clone(),
                                format!(
                                    "{} corruption at offset {}: {}",
                                    if zero_padded {
                                        "mid-log"
                                    } else {
                                        "non-zero tail"
                                    },
                                    offset,
                                    detail
                                ),
                            ));
                        }
                        Self::truncate_file(seg_path, offset)?;
                        outcome.torn_tail = true;
                        tracing::warn!(
                            file = %file_name,
                            truncated_bytes = rest.len(),
                            "WAL zero-padded tail from torn write detected and truncated (recoverable)"
                        );
                        break;
                    }
                }
            }
        }

        // Prepare append state at the end of the (possibly truncated) last segment.
        if let Some((_, last_path)) = segments.last() {
            let meta = std::fs::metadata(last_path)?;
            self.active_path = Some(last_path.clone());
            self.active_start_seq = segments.last().unwrap().0;
            self.active_bytes = meta.len();
            self.writer = Some(BufWriter::new(
                OpenOptions::new().append(true).open(last_path)?,
            ));
        }
        // The sequence allocator continues after BOTH the WAL tail and the
        // checkpoint floor: a freshly-rotated (empty) active segment after a
        // checkpoint has no records, but sequences must never regress below
        // what earlier segments already consumed (they may have been trimmed).
        self.next_seq = outcome.last_seq.max(min_seq_exclusive) + 1;
        self.last_synced_seq = self.next_seq - 1;

        Ok(outcome)
    }

    fn parse_frame(buf: &[u8]) -> Result<(WalRecord, usize), FrameError> {
        if buf.len() < 8 {
            return Err(FrameError::Truncated);
        }
        let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != FRAME_MAGIC {
            return Err(FrameError::Corrupt(format!(
                "bad frame magic {magic:#010x}"
            )));
        }
        let len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        if len > 512 * 1024 * 1024 {
            return Err(FrameError::Corrupt(format!(
                "frame length {len} exceeds sane maximum"
            )));
        }
        let frame_len = 8 + len + 4;
        if buf.len() < frame_len {
            return Err(FrameError::Truncated);
        }
        let body = &buf[8..8 + len];
        let crc_stored = u32::from_be_bytes([
            buf[8 + len],
            buf[8 + len + 1],
            buf[8 + len + 2],
            buf[8 + len + 3],
        ]);
        let mut hasher = Hasher::new();
        hasher.update(&buf[..8 + len]);
        if hasher.finalize() != crc_stored {
            return Err(FrameError::Corrupt("checksum mismatch".to_string()));
        }
        let record: WalRecord = bincode::deserialize(body)
            .map_err(|e| FrameError::Corrupt(format!("body deserialization failed: {e}")))?;
        Ok((record, frame_len))
    }

    fn truncate_file(path: &Path, len_bytes: usize) -> Result<(), crate::error::StorageError> {
        let f = OpenOptions::new().write(true).open(path)?;
        f.set_len(len_bytes as u64)?;
        f.sync_all()?;
        Ok(())
    }

    fn has_valid_frame_after(buf: &[u8], from: usize) -> bool {
        if from >= buf.len() {
            return false;
        }
        for i in from..buf.len() {
            if i + 8 <= buf.len() {
                let magic = u32::from_be_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                if magic == FRAME_MAGIC && Self::parse_frame(&buf[i..]).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Append a record. Returns its assigned sequence number.
    pub fn append(&mut self, mut record: WalRecord) -> Result<u64, crate::error::StorageError> {
        if record.version != WAL_FORMAT_VERSION {
            return Err(crate::error::StorageError::InvalidArgument(format!(
                "refusing to append record with version {} (supported {})",
                record.version, WAL_FORMAT_VERSION
            )));
        }
        record.seq = self.next_seq;
        let body = bincode::serialize(&record)
            .map_err(|e| crate::error::StorageError::Wal(e.to_string()))?;

        // Rotate if the active segment is full (before writing, so segments stay bounded).
        if self.writer.is_some() && self.active_bytes >= self.max_segment_bytes {
            self.rotate()?;
        }

        // Ensure an active segment exists.
        if self.writer.is_none() {
            let start = if record.seq > 1 { record.seq } else { 1 };
            self.open_segment(start)?;
        }

        let mut frame = Vec::with_capacity(body.len() + 12);
        frame.extend_from_slice(&FRAME_MAGIC.to_be_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
        frame.extend_from_slice(&body);
        let mut hasher = Hasher::new();
        hasher.update(&frame);
        frame.extend_from_slice(&hasher.finalize().to_be_bytes());

        let writer = self.writer.as_mut().expect("active segment");
        writer
            .write_all(&frame)
            .map_err(|e| crate::error::StorageError::Wal(format!("WAL append failed: {e}")))?;
        // E2 crash gate: frame bytes are in the userspace buffer; no flush yet.
        crate::crashgate::GATE_AFTER_WRITE.hit();

        match self.durability {
            Durability::Sync => {
                writer.flush()?;
                writer.get_ref().sync_all()?;
                self.last_synced_seq = record.seq;
                // E2 crash gate: frame is fsynced (machine-durable boundary).
                crate::crashgate::GATE_AFTER_FSYNC.hit();
            }
            Durability::GroupCommit => {
                writer.flush()?;
                // E2 crash gate: frame is in the OS page cache (process-crash
                // durable boundary); not fsynced.
                crate::crashgate::GATE_AFTER_FLUSH.hit();
            }
            Durability::Async => {}
        }

        self.active_bytes += frame.len() as u64;
        self.next_seq += 1;
        Ok(record.seq)
    }

    fn open_segment(&mut self, start_seq: u64) -> Result<(), crate::error::StorageError> {
        let path = self.segment_path(start_seq);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        self.writer = Some(BufWriter::new(file));
        self.active_path = Some(path);
        self.active_start_seq = start_seq;
        self.active_bytes = 0;
        // E1 WAL-integrity invariant: a durable watermark record is written
        // whenever a segment is created (first append or rotation). It anchors
        // (a) that the WAL dir must keep at least the recorded active segment
        // and (b) how far completed segments had advanced — so deleting
        // required WAL history can no longer masquerade as a fresh/trimmed
        // database. Atomic tmp -> rename -> dir fsync, like the manifest.
        write_wal_state(
            &self.dir,
            &WalState {
                format_version: WAL_STATE_FORMAT_VERSION,
                high_watermark: self.next_seq - 1,
                active_start: start_seq,
            },
        )?;
        // E3 window: durable watermark record written for this segment.
        crate::crashgate::GATE_ROTATE_AFTER_STATE_WRITE.hit();
        Ok(())
    }

    /// Close the active segment and start a new one at the next sequence number.
    pub fn rotate(&mut self) -> Result<(), crate::error::StorageError> {
        if let Some(w) = self.writer.take() {
            let mut w = w;
            w.flush()?;
            w.get_ref().sync_all()?;
            self.last_synced_seq = self.next_seq - 1;
            // E3 window: completed segment durable; new segment + in-flight
            // frame not yet written (rotation precedes the frame write).
            crate::crashgate::GATE_ROTATE_AFTER_OLD_FSYNC.hit();
        }
        let next_start = self.next_seq;
        self.open_segment(next_start)
    }

    /// Flush userspace buffers (visible to the OS, not necessarily durable).
    pub fn flush(&mut self) -> Result<(), crate::error::StorageError> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
        }
        Ok(())
    }

    /// Flush + fsync: everything appended so far becomes machine-crash durable.
    pub fn fsync(&mut self) -> Result<(), crate::error::StorageError> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        self.last_synced_seq = self.next_seq - 1;
        Ok(())
    }

    /// Delete obsolete segments: a segment is removable when every record in it has
    /// seq <= `checkpoint_seq` (i.e. recovery no longer depends on it) and it is not
    /// the active segment. Never deletes data the checkpoint doesn't cover.
    pub fn retain_from(
        &mut self,
        checkpoint_seq: u64,
    ) -> Result<usize, crate::error::StorageError> {
        let segments = self.list_segments()?;
        if segments.is_empty() {
            return Ok(0);
        }
        let mut removed = 0;
        for (i, (start, path)) in segments.iter().enumerate() {
            let is_active = Some(path.clone()) == self.active_path;
            if is_active {
                break; // never delete the segment we are appending to
            }
            let end = segments.get(i + 1).map(|(next_start, _)| next_start - 1);
            match end {
                Some(end) if end <= checkpoint_seq => {
                    std::fs::remove_file(path)?;
                    removed += 1;
                }
                _ => break, // segments are ordered; first non-removable stops us
            }
            let _ = start;
        }
        Ok(removed)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn durability(&self) -> Durability {
        self.durability
    }

    pub fn set_durability(&mut self, d: Durability) {
        self.durability = d;
    }

    pub fn active_segment(&self) -> Option<&Path> {
        self.active_path.as_deref()
    }

    pub fn total_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Ok(segs) = self.list_segments() {
            for (_, p) in segs {
                if let Ok(m) = std::fs::metadata(&p) {
                    total += m.len();
                }
            }
        }
        total
    }
}

enum FrameError {
    Truncated,
    Corrupt(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn rec(kind: RecordKind, payload: &[u8]) -> WalRecord {
        let mut r = WalRecord::new(0, kind);
        r.payload = payload.to_vec();
        r
    }

    #[test]
    fn append_replay_roundtrip() {
        let dir = tmpdir();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        for i in 0..10u64 {
            wal.append(rec(RecordKind::InsertDocument, format!("p{i}").as_bytes()))
                .unwrap();
        }
        drop(wal);
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        let out = wal.replay(0).unwrap();
        assert_eq!(out.records.len(), 10);
        assert!(!out.torn_tail);
        assert_eq!(out.records[7].payload, b"p7");
        assert_eq!(out.last_seq, 10);
        assert_eq!(wal.next_seq(), 11);
    }

    #[test]
    fn replay_since_checkpoint_filters() {
        let dir = tmpdir();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        for _ in 0..10 {
            wal.append(rec(RecordKind::InsertDocument, b"x")).unwrap();
        }
        drop(wal);
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        let out = wal.replay(5).unwrap();
        assert_eq!(out.records.len(), 5); // seqs 6..10
        assert_eq!(out.first_seq, 6);
    }

    #[test]
    fn torn_tail_is_truncated_and_reported() {
        let dir = tmpdir();
        {
            let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
            for i in 0..5 {
                wal.append(rec(RecordKind::InsertDocument, format!("p{i}").as_bytes()))
                    .unwrap();
            }
            wal.fsync().unwrap();
        }
        // Simulate a torn write: append garbage partial bytes to the segment.
        let seg = {
            let mut entries: Vec<_> = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wal"))
                .collect();
            entries.sort();
            entries.pop().unwrap()
        };
        {
            let mut f = OpenOptions::new().append(true).open(&seg).unwrap();
            f.write_all(&[0u8; 13]).unwrap(); // partial frame
        }
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        let out = wal.replay(0).unwrap();
        assert!(out.torn_tail);
        assert_eq!(out.records.len(), 5);
        // WAL remains usable: append + reopen round-trips.
        wal.append(rec(RecordKind::InsertDocument, b"after"))
            .unwrap();
        drop(wal);
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        let out = wal.replay(0).unwrap();
        assert_eq!(out.records.len(), 6);
        assert!(!out.torn_tail);
    }

    #[test]
    fn mid_log_corruption_is_fatal() {
        let dir = tmpdir();
        {
            let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
            for _ in 0..5 {
                wal.append(rec(RecordKind::InsertDocument, b"x")).unwrap();
            }
            wal.fsync().unwrap();
        }
        let seg = {
            let mut entries: Vec<_> = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wal"))
                .collect();
            entries.sort();
            entries.pop().unwrap()
        };
        // Flip a byte inside the SECOND record's body (offset: 8+len0+4 + 8 + 8 ..)
        let mut bytes = std::fs::read(&seg).unwrap();
        let after_first = 8 + {
            let l = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
            l + 4
        };
        bytes[after_first + 20] ^= 0xFF; // inside frame 2 (magic/len intact, body damaged)
        std::fs::write(&seg, &bytes).unwrap();

        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        let err = wal.replay(0).unwrap_err();
        assert!(matches!(err, crate::error::StorageError::Corruption { .. }));
    }

    #[test]
    fn unknown_version_is_rejected() {
        let dir = tmpdir();
        {
            let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
            wal.append(rec(RecordKind::InsertDocument, b"x")).unwrap();
        }
        let seg = {
            let mut entries: Vec<_> = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wal"))
                .collect();
            entries.sort();
            entries.pop().unwrap()
        };
        // Corrupt the version field at the start of the frame body. This breaks the
        // CRC as well; both are corruption-class errors and must be rejected.
        let mut bytes = std::fs::read(&seg).unwrap();
        let body_start = 8; // magic(4) + len(4)
        let v = u16::from_le_bytes([bytes[body_start], bytes[body_start + 1]]);
        assert_eq!(v, WAL_FORMAT_VERSION);
        bytes[body_start] = 42;
        std::fs::write(&seg, &bytes).unwrap();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        assert!(matches!(
            wal.replay(0),
            Err(crate::error::StorageError::Corruption { .. }
                | crate::error::StorageError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn segments_rotate_and_retain_by_checkpoint() {
        let dir = tmpdir();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 64).unwrap(); // tiny segments
        let mut last = 0;
        for i in 0..20 {
            last = wal
                .append(rec(
                    RecordKind::InsertDocument,
                    format!("payload-{i:010}").as_bytes(),
                ))
                .unwrap();
        }
        assert!(
            wal.list_segments().unwrap().len() > 1,
            "rotation should have occurred"
        );
        wal.fsync().unwrap();
        drop(wal);

        let mut wal = Wal::open(dir.path(), Durability::Sync, 64).unwrap();
        let out = wal.replay(0).unwrap();
        assert_eq!(out.records.len(), 20);

        // Retain only what checkpoint 15 covers: segments fully below seq<=15 go away.
        let removed = wal.retain_from(15).unwrap();
        assert!(removed >= 1);
        let segs = wal.list_segments().unwrap();
        assert!(!segs.is_empty());
        // No removable data was lost: replay still yields everything after checkpoint.
        let out = wal.replay(15).unwrap();
        assert_eq!(out.records.len(), 5);
        assert_eq!(out.records.last().unwrap().seq, last);
    }

    #[test]
    fn durability_sync_survives_reopen_without_flush() {
        // Sync mode: record must be durable immediately after append returns.
        let dir = tmpdir();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        wal.append(rec(RecordKind::InsertDocument, b"durable"))
            .unwrap();
        drop(wal); // no explicit fsync — Sync already did it
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        let out = wal.replay(0).unwrap();
        assert_eq!(out.records.len(), 1);
    }

    #[test]
    fn corrupted_segment_filename_detected() {
        let dir = tmpdir();
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(dir.path().join("notaseq.wal"), b"junk").unwrap();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        assert!(wal.replay(0).is_err());
    }

    /// §34 fuzz: the WAL parser must NEVER panic on arbitrary bytes —
    /// replay either succeeds (intact prefix) or returns Err (corruption
    /// detected). Deterministic PRNG so failures are reproducible.
    #[test]
    fn replay_fuzz_random_bytes_never_panics() {
        for seed in 0..64u64 {
            let dir = tmpdir();
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEF;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let len = 1 + (next() % 4096) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let seg = dir.path().join("00000000000000000001.wal");
            std::fs::write(&seg, &bytes).unwrap();
            // Ok or Err are both acceptable; a panic is the failure.
            if let Ok(mut wal) = Wal::open(dir.path(), Durability::Async, u64::MAX) {
                let _ = wal.replay(0);
            }
        }
    }

    /// §34 fuzz: random truncations of a VALID WAL must always yield a clean
    /// prefix replay or an error — never a panic, never a fabricated record
    /// with a sequence that breaks continuity.
    #[test]
    fn replay_fuzz_truncated_valid_wal_never_panics() {
        let src = tmpdir();
        {
            let mut wal = Wal::open(src.path(), Durability::Sync, 1 << 20).unwrap();
            for i in 0..40u64 {
                wal.append(rec(
                    RecordKind::InsertDocument,
                    format!("payload-{i}").as_bytes(),
                ))
                .unwrap();
            }
            wal.flush().unwrap();
        }
        let valid = std::fs::read(src.path().join("00000000000000000001.wal")).unwrap();
        assert!(!valid.is_empty());
        for seed in 0..32u64 {
            let dir = tmpdir();
            let mut x = seed.wrapping_mul(0x2545F4914F6CDD1D) ^ 0xA5A5A5A5;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let cut = 1 + (x as usize) % valid.len();
            let seg = dir.path().join("00000000000000000001.wal");
            std::fs::write(&seg, &valid[..cut]).unwrap();
            // Ok or Err are both acceptable; a panic is the failure.
            if let Ok(mut wal) = Wal::open(dir.path(), Durability::Async, u64::MAX) {
                if let Ok(outcome) = wal.replay(0) {
                    // recovered records must form a strict prefix of the seq space
                    for (k, r) in outcome.records.iter().enumerate() {
                        assert_eq!(r.seq, (k + 1) as u64, "seed {seed}: non-prefix replay");
                    }
                }
            }
        }
    }

    // ---------- Phase 3E E1: durable watermark record ----------

    #[test]
    fn wal_state_written_on_first_append_and_rotation() {
        let dir = tmpdir();
        {
            let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
            // no segment yet -> no state record
            assert!(read_wal_state(dir.path()).unwrap().is_none());
            for i in 0..5u64 {
                wal.append(rec(RecordKind::InsertDocument, format!("p{i}").as_bytes()))
                    .unwrap();
            }
            let ws = read_wal_state(dir.path())
                .unwrap()
                .expect("state after first append");
            // first segment still active: completed coverage is empty (0),
            // active segment starts at 1
            assert_eq!(ws.high_watermark, 0);
            assert_eq!(ws.active_start, 1);
            wal.rotate().unwrap();
            let ws2 = read_wal_state(dir.path())
                .unwrap()
                .expect("state after rotate");
            // seqs 1..=5 are now in a completed segment
            assert_eq!(ws2.high_watermark, 5);
            assert_eq!(ws2.active_start, 6);
            wal.append(rec(RecordKind::InsertDocument, b"x")).unwrap();
            let ws3 = read_wal_state(dir.path()).unwrap().unwrap();
            // the new record lives in the ACTIVE segment: no anchor until the
            // next rotation (documented boundary)
            assert_eq!(ws3.high_watermark, 5);
            assert_eq!(ws3.active_start, 6);
        }
    }

    #[test]
    fn wal_state_corrupt_record_is_an_error_not_ignored() {
        let dir = tmpdir();
        let mut wal = Wal::open(dir.path(), Durability::Sync, 1 << 20).unwrap();
        wal.append(rec(RecordKind::InsertDocument, b"p")).unwrap();
        std::fs::write(dir.path().join(WAL_STATE_FILE), b"garbage{").unwrap();
        assert!(read_wal_state(dir.path()).is_err());
    }
}
