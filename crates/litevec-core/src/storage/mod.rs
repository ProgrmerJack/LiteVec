//! Storage engine.
//!
//! Provides a trait-based abstraction over file-backed and in-memory storage.

pub mod file;
pub mod page;
pub mod wal;

use crate::error::Result;

/// Abstract storage backend trait.
pub trait StorageBackend: Send + Sync {
    /// Read a page by page ID.
    fn read_page(&self, page_id: u64) -> &[u8];

    /// Write data to a page.
    fn write_page(&mut self, page_id: u64, data: &[u8]);

    /// Allocate `count` new pages, return starting page ID.
    fn allocate_pages(&mut self, count: u64) -> Result<u64>;

    /// Flush all changes to persistent storage.
    fn flush(&self) -> Result<()>;

    /// Total number of pages.
    fn page_count(&self) -> u64;

    /// Page size in bytes.
    fn page_size(&self) -> usize;
}

/// In-memory storage backend (for `Database::open_memory()` and testing).
pub struct MemoryStorage {
    pages: Vec<Vec<u8>>,
    page_size: usize,
}

impl MemoryStorage {
    /// Create a new in-memory storage with the given page size.
    pub fn new(page_size: usize) -> Self {
        // Allocate page 0 (header)
        let mut header_page = vec![0u8; page_size];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        page::write_header(&mut header_page, page_size as u32, 0, now);

        Self {
            pages: vec![header_page],
            page_size,
        }
    }
}

impl StorageBackend for MemoryStorage {
    fn read_page(&self, page_id: u64) -> &[u8] {
        &self.pages[page_id as usize]
    }

    fn write_page(&mut self, page_id: u64, data: &[u8]) {
        let page = &mut self.pages[page_id as usize];
        page[..data.len()].copy_from_slice(data);
    }

    fn allocate_pages(&mut self, count: u64) -> Result<u64> {
        let start = self.pages.len() as u64;
        for _ in 0..count {
            self.pages.push(vec![0u8; self.page_size]);
        }
        Ok(start)
    }

    fn flush(&self) -> Result<()> {
        Ok(()) // No-op for memory
    }

    fn page_count(&self) -> u64 {
        self.pages.len() as u64
    }

    fn page_size(&self) -> usize {
        self.page_size
    }
}

/// Implement StorageBackend for FileStorage.
impl StorageBackend for file::FileStorage {
    fn read_page(&self, page_id: u64) -> &[u8] {
        self.read_page(page_id)
    }

    fn write_page(&mut self, page_id: u64, data: &[u8]) {
        self.write_page(page_id, data);
    }

    fn allocate_pages(&mut self, count: u64) -> Result<u64> {
        self.allocate_pages(count)
    }

    fn flush(&self) -> Result<()> {
        self.flush()
    }

    fn page_count(&self) -> u64 {
        self.page_count()
    }

    fn page_size(&self) -> usize {
        self.page_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage_basics() {
        let mut storage = MemoryStorage::new(4096);
        assert_eq!(storage.page_count(), 1);

        // Header should be valid
        let header = storage.read_page(0);
        assert!(page::validate_magic(header));

        // Allocate pages
        let start = storage.allocate_pages(3).unwrap();
        assert_eq!(start, 1);
        assert_eq!(storage.page_count(), 4);

        // Write and read
        let data = vec![0xAB; 4096];
        storage.write_page(1, &data);
        let read_back = storage.read_page(1);
        assert_eq!(read_back[0], 0xAB);
        assert_eq!(read_back[4095], 0xAB);
    }

    #[test]
    fn test_memory_storage_flush() {
        let storage = MemoryStorage::new(4096);
        assert!(storage.flush().is_ok());
    }
}
