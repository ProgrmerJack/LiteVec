//! Write-ahead log for crash safety.
//!
//! All mutations are recorded in the WAL before being applied to the main file.
//! On recovery, the WAL is replayed to restore consistency.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// WAL record types.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalRecordType {
    InsertVector = 0,
    DeleteVector = 1,
    UpdateMetadata = 2,
    Checkpoint = 3,
}

impl WalRecordType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::InsertVector),
            1 => Some(Self::DeleteVector),
            2 => Some(Self::UpdateMetadata),
            3 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

/// A single WAL record.
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub record_type: WalRecordType,
    pub collection_id: u16,
    pub payload: Vec<u8>,
}

/// WAL record header layout:
/// - Record Length: u32 (4 bytes) — total payload size
/// - Record Type: u8 (1 byte)
/// - Collection ID: u16 (2 bytes)
/// - Payload: [u8; record_length]
/// - CRC32: u32 (4 bytes)
const WAL_RECORD_HEADER_SIZE: usize = 4 + 1 + 2; // 7 bytes
const WAL_CRC_SIZE: usize = 4;

/// Write-ahead log manager.
pub struct Wal {
    path: PathBuf,
    file: Option<std::fs::File>,
    records: Vec<WalRecord>,
}

impl Wal {
    /// Open or create a WAL file.
    pub fn open(db_path: &Path) -> Result<Self> {
        let wal_path = wal_path_for(db_path);

        let mut wal = Self {
            path: wal_path.clone(),
            file: None,
            records: Vec::new(),
        };

        // If WAL exists, read all records for replay
        if wal_path.exists() {
            wal.records = Self::read_records(&wal_path)?;
        }

        Ok(wal)
    }

    /// Create a WAL that operates purely in memory (for in-memory databases).
    pub fn in_memory() -> Self {
        Self {
            path: PathBuf::new(),
            file: None,
            records: Vec::new(),
        }
    }

    /// Returns true if there are records that need to be replayed.
    pub fn needs_replay(&self) -> bool {
        !self.records.is_empty()
    }

    /// Get records for replay.
    pub fn records_for_replay(&self) -> &[WalRecord] {
        &self.records
    }

    /// Append a record to the WAL.
    pub fn append(&mut self, record: WalRecord) -> Result<()> {
        // Write to file if file-backed
        if !self.path.as_os_str().is_empty() {
            let file = self.ensure_file()?;
            Self::write_record(file, &record)?;
            file.flush()?;
        }
        self.records.push(record);
        Ok(())
    }

    /// Clear the WAL after a successful checkpoint.
    pub fn clear(&mut self) -> Result<()> {
        self.records.clear();
        if let Some(file) = self.file.take() {
            drop(file);
        }
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }

    /// Ensure the WAL file is open.
    fn ensure_file(&mut self) -> Result<&mut std::fs::File> {
        if self.file.is_none() {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            self.file = Some(file);
        }
        Ok(self.file.as_mut().unwrap())
    }

    /// Write a single record to a file.
    fn write_record(file: &mut std::fs::File, record: &WalRecord) -> Result<()> {
        let payload_len = record.payload.len() as u32;
        let mut buf =
            Vec::with_capacity(WAL_RECORD_HEADER_SIZE + record.payload.len() + WAL_CRC_SIZE);

        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.push(record.record_type as u8);
        buf.extend_from_slice(&record.collection_id.to_le_bytes());
        buf.extend_from_slice(&record.payload);

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        file.write_all(&buf)?;
        Ok(())
    }

    /// Read all records from a WAL file.
    fn read_records(path: &Path) -> Result<Vec<WalRecord>> {
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut records = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            // Need at least header + CRC
            if offset + WAL_RECORD_HEADER_SIZE + WAL_CRC_SIZE > data.len() {
                break; // Truncated record — stop replay here
            }

            let payload_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            let record_type_byte = data[offset + 4];
            let collection_id =
                u16::from_le_bytes(data[offset + 5..offset + 7].try_into().unwrap());

            let total_record_size = WAL_RECORD_HEADER_SIZE + payload_len + WAL_CRC_SIZE;
            if offset + total_record_size > data.len() {
                break; // Truncated record
            }

            let record_data = &data[offset..offset + WAL_RECORD_HEADER_SIZE + payload_len];
            let stored_crc = u32::from_le_bytes(
                data[offset + WAL_RECORD_HEADER_SIZE + payload_len..offset + total_record_size]
                    .try_into()
                    .unwrap(),
            );
            let computed_crc = crc32fast::hash(record_data);

            if stored_crc != computed_crc {
                return Err(Error::WalCorruption(format!(
                    "CRC mismatch at offset {offset}: stored={stored_crc:#x}, computed={computed_crc:#x}"
                )));
            }

            let record_type = WalRecordType::from_u8(record_type_byte).ok_or_else(|| {
                Error::WalCorruption(format!("Unknown record type: {record_type_byte}"))
            })?;

            let payload = data
                [offset + WAL_RECORD_HEADER_SIZE..offset + WAL_RECORD_HEADER_SIZE + payload_len]
                .to_vec();

            records.push(WalRecord {
                record_type,
                collection_id,
                payload,
            });

            offset += total_record_size;
        }

        Ok(records)
    }
}

/// Get the WAL file path for a given database file path.
pub fn wal_path_for(db_path: &Path) -> PathBuf {
    let mut wal_path = db_path.as_os_str().to_owned();
    wal_path.push("-wal");
    PathBuf::from(wal_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("litevec_test_wal_{name}_{}.lv", std::process::id()))
    }

    fn cleanup(db_path: &Path) {
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(wal_path_for(db_path));
    }

    #[test]
    fn test_wal_path() {
        let db_path = Path::new("/tmp/test.lv");
        let wal = wal_path_for(db_path);
        assert_eq!(wal, PathBuf::from("/tmp/test.lv-wal"));
    }

    #[test]
    fn test_wal_append_and_replay() {
        let db_path = temp_db_path("append_replay");
        cleanup(&db_path);

        // Write records
        {
            let mut wal = Wal::open(&db_path).unwrap();
            assert!(!wal.needs_replay());

            wal.append(WalRecord {
                record_type: WalRecordType::InsertVector,
                collection_id: 0,
                payload: vec![1, 2, 3, 4],
            })
            .unwrap();

            wal.append(WalRecord {
                record_type: WalRecordType::DeleteVector,
                collection_id: 1,
                payload: vec![5, 6],
            })
            .unwrap();
        }

        // Reopen and replay
        {
            let wal = Wal::open(&db_path).unwrap();
            assert!(wal.needs_replay());
            let records = wal.records_for_replay();
            assert_eq!(records.len(), 2);
            assert_eq!(records[0].record_type, WalRecordType::InsertVector);
            assert_eq!(records[0].collection_id, 0);
            assert_eq!(records[0].payload, vec![1, 2, 3, 4]);
            assert_eq!(records[1].record_type, WalRecordType::DeleteVector);
            assert_eq!(records[1].collection_id, 1);
            assert_eq!(records[1].payload, vec![5, 6]);
        }

        cleanup(&db_path);
    }

    #[test]
    fn test_wal_clear() {
        let db_path = temp_db_path("clear");
        cleanup(&db_path);

        {
            let mut wal = Wal::open(&db_path).unwrap();
            wal.append(WalRecord {
                record_type: WalRecordType::InsertVector,
                collection_id: 0,
                payload: vec![1, 2, 3],
            })
            .unwrap();
            wal.clear().unwrap();
        }

        // After clear, no records to replay
        {
            let wal = Wal::open(&db_path).unwrap();
            assert!(!wal.needs_replay());
        }

        cleanup(&db_path);
    }

    #[test]
    fn test_wal_in_memory() {
        let mut wal = Wal::in_memory();
        assert!(!wal.needs_replay());

        wal.append(WalRecord {
            record_type: WalRecordType::UpdateMetadata,
            collection_id: 0,
            payload: b"hello".to_vec(),
        })
        .unwrap();

        assert!(wal.needs_replay());
        assert_eq!(wal.records_for_replay().len(), 1);
    }

    #[test]
    fn test_wal_crc_integrity() {
        let db_path = temp_db_path("crc");
        cleanup(&db_path);

        // Write a record
        {
            let mut wal = Wal::open(&db_path).unwrap();
            wal.append(WalRecord {
                record_type: WalRecordType::InsertVector,
                collection_id: 0,
                payload: vec![1, 2, 3],
            })
            .unwrap();
        }

        // Corrupt the WAL file
        {
            let wal_path = wal_path_for(&db_path);
            let mut data = std::fs::read(&wal_path).unwrap();
            if let Some(last) = data.last_mut() {
                *last ^= 0xFF; // Flip bits in CRC
            }
            std::fs::write(&wal_path, &data).unwrap();
        }

        // Reopen should detect corruption
        let result = Wal::open(&db_path);
        assert!(result.is_err());

        cleanup(&db_path);
    }
}
