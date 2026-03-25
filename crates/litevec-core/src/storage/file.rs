//! Memory-mapped file storage engine.

use std::path::{Path, PathBuf};

use memmap2::{MmapMut, MmapOptions};

use crate::error::{Error, Result};

use super::page;
#[cfg(test)]
use super::page::DEFAULT_PAGE_SIZE;

/// File-backed storage engine using memory-mapped I/O.
pub struct FileStorage {
    file: std::fs::File,
    mmap: MmapMut,
    page_size: usize,
    file_size: u64,
    path: PathBuf,
}

impl FileStorage {
    /// Open or create a database file at the given path.
    pub fn open(path: &Path, page_size: usize) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let file_size = file.metadata()?.len();
        let is_new = file_size == 0;

        if is_new {
            // Allocate first page for the header
            file.set_len(page_size as u64)?;
        }

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let mut storage = Self {
            file,
            mmap,
            page_size,
            file_size: if is_new { page_size as u64 } else { file_size },
            path: path.to_path_buf(),
        };

        if is_new {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let mut header = vec![0u8; page_size];
            page::write_header(&mut header, page_size as u32, 0, now);
            storage.write_page(0, &header);
            storage.flush()?;
        } else {
            // Validate existing file
            if file_size < page_size as u64 {
                return Err(Error::InvalidFile(
                    "File too small to contain a valid header".to_string(),
                ));
            }
            let header = storage.read_page(0);
            if !page::validate_magic(header) {
                return Err(Error::InvalidFile("Invalid magic bytes".to_string()));
            }
            let version = page::read_version(header);
            if version != page::FORMAT_VERSION {
                return Err(Error::InvalidFile(format!(
                    "Unsupported format version: {version}"
                )));
            }
        }

        Ok(storage)
    }

    /// Read a page by page ID.
    pub fn read_page(&self, page_id: u64) -> &[u8] {
        let offset = page_id as usize * self.page_size;
        &self.mmap[offset..offset + self.page_size]
    }

    /// Write data to a page. `data` must be <= page_size.
    pub fn write_page(&mut self, page_id: u64, data: &[u8]) {
        let offset = page_id as usize * self.page_size;
        self.mmap[offset..offset + data.len()].copy_from_slice(data);
    }

    /// Allocate `count` new pages at the end of the file. Returns the starting page ID.
    pub fn allocate_pages(&mut self, count: u64) -> Result<u64> {
        let start_page = self.file_size / self.page_size as u64;
        let new_size = self.file_size + count * self.page_size as u64;
        self.file.set_len(new_size)?;
        self.mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
        self.file_size = new_size;
        Ok(start_page)
    }

    /// Flush all changes to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Total number of pages.
    pub fn page_count(&self) -> u64 {
        self.file_size / self.page_size as u64
    }

    /// Page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Path to the database file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Remap after external file size change.
    pub fn remap(&mut self) -> Result<()> {
        self.file_size = self.file.metadata()?.len();
        self.mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("litevec_test_{name}_{}.lv", std::process::id()))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_create_and_open() {
        let path = temp_path("create_open");
        cleanup(&path);

        {
            let storage = FileStorage::open(&path, DEFAULT_PAGE_SIZE).unwrap();
            assert_eq!(storage.page_count(), 1);
            let header = storage.read_page(0);
            assert!(page::validate_magic(header));
        }

        // Reopen
        {
            let storage = FileStorage::open(&path, DEFAULT_PAGE_SIZE).unwrap();
            assert_eq!(storage.page_count(), 1);
            let header = storage.read_page(0);
            assert!(page::validate_magic(header));
        }

        cleanup(&path);
    }

    #[test]
    fn test_allocate_and_write() {
        let path = temp_path("alloc_write");
        cleanup(&path);

        {
            let mut storage = FileStorage::open(&path, DEFAULT_PAGE_SIZE).unwrap();
            let page_id = storage.allocate_pages(2).unwrap();
            assert_eq!(page_id, 1);
            assert_eq!(storage.page_count(), 3);

            let data = vec![42u8; DEFAULT_PAGE_SIZE];
            storage.write_page(page_id, &data);
            storage.flush().unwrap();

            let read_back = storage.read_page(page_id);
            assert_eq!(read_back[0], 42);
            assert_eq!(read_back[DEFAULT_PAGE_SIZE - 1], 42);
        }

        // Reopen and verify
        {
            let storage = FileStorage::open(&path, DEFAULT_PAGE_SIZE).unwrap();
            assert_eq!(storage.page_count(), 3);
            let read_back = storage.read_page(1);
            assert_eq!(read_back[0], 42);
        }

        cleanup(&path);
    }

    #[test]
    fn test_invalid_file() {
        let path = temp_path("invalid");
        cleanup(&path);

        // Write garbage
        std::fs::write(&path, b"NOT_A_LITEVEC_FILE").unwrap();

        let result = FileStorage::open(&path, DEFAULT_PAGE_SIZE);
        assert!(result.is_err());

        cleanup(&path);
    }
}
