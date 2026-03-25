//! Page layout and management.

/// Default page size: 4 KiB.
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// File header magic bytes: "LVEC"
pub const MAGIC: &[u8; 4] = b"LVEC";

/// Current file format version.
pub const FORMAT_VERSION: u32 = 1;

/// File header layout (stored in page 0).
///
/// ```text
/// Offset  Size  Field
///   0       4    Magic: b"LVEC"
///   4       4    Version: u32
///   8       4    Page Size: u32
///  12       4    Flags: u32 (reserved)
///  16       4    Collection Count: u32
///  20       8    Created At: u64 (unix timestamp)
///  28    rest    Reserved (padding to page size)
/// ```
pub const HEADER_MAGIC_OFFSET: usize = 0;
pub const HEADER_VERSION_OFFSET: usize = 4;
pub const HEADER_PAGE_SIZE_OFFSET: usize = 8;
pub const HEADER_FLAGS_OFFSET: usize = 12;
pub const HEADER_COLLECTION_COUNT_OFFSET: usize = 16;
pub const HEADER_CREATED_AT_OFFSET: usize = 20;
pub const HEADER_SIZE: usize = 28;

/// Write the file header into a buffer.
pub fn write_header(buf: &mut [u8], page_size: u32, collection_count: u32, created_at: u64) {
    buf[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4].copy_from_slice(MAGIC);
    buf[HEADER_VERSION_OFFSET..HEADER_VERSION_OFFSET + 4]
        .copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf[HEADER_PAGE_SIZE_OFFSET..HEADER_PAGE_SIZE_OFFSET + 4]
        .copy_from_slice(&page_size.to_le_bytes());
    buf[HEADER_FLAGS_OFFSET..HEADER_FLAGS_OFFSET + 4].copy_from_slice(&0u32.to_le_bytes());
    buf[HEADER_COLLECTION_COUNT_OFFSET..HEADER_COLLECTION_COUNT_OFFSET + 4]
        .copy_from_slice(&collection_count.to_le_bytes());
    buf[HEADER_CREATED_AT_OFFSET..HEADER_CREATED_AT_OFFSET + 8]
        .copy_from_slice(&created_at.to_le_bytes());
}

/// Read and validate the magic bytes from a header.
pub fn validate_magic(buf: &[u8]) -> bool {
    buf.len() >= 4 && &buf[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4] == MAGIC
}

/// Read the format version from a header.
pub fn read_version(buf: &[u8]) -> u32 {
    u32::from_le_bytes(
        buf[HEADER_VERSION_OFFSET..HEADER_VERSION_OFFSET + 4]
            .try_into()
            .unwrap(),
    )
}

/// Read the page size from a header.
pub fn read_page_size(buf: &[u8]) -> u32 {
    u32::from_le_bytes(
        buf[HEADER_PAGE_SIZE_OFFSET..HEADER_PAGE_SIZE_OFFSET + 4]
            .try_into()
            .unwrap(),
    )
}

/// Read the collection count from a header.
pub fn read_collection_count(buf: &[u8]) -> u32 {
    u32::from_le_bytes(
        buf[HEADER_COLLECTION_COUNT_OFFSET..HEADER_COLLECTION_COUNT_OFFSET + 4]
            .try_into()
            .unwrap(),
    )
}

/// Update the collection count in a header buffer.
pub fn write_collection_count(buf: &mut [u8], count: u32) {
    buf[HEADER_COLLECTION_COUNT_OFFSET..HEADER_COLLECTION_COUNT_OFFSET + 4]
        .copy_from_slice(&count.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_header() {
        let mut buf = vec![0u8; DEFAULT_PAGE_SIZE];
        write_header(&mut buf, 4096, 0, 1700000000);

        assert!(validate_magic(&buf));
        assert_eq!(read_version(&buf), FORMAT_VERSION);
        assert_eq!(read_page_size(&buf), 4096);
        assert_eq!(read_collection_count(&buf), 0);
    }

    #[test]
    fn test_invalid_magic() {
        let buf = vec![0u8; 64];
        assert!(!validate_magic(&buf));
    }

    #[test]
    fn test_update_collection_count() {
        let mut buf = vec![0u8; DEFAULT_PAGE_SIZE];
        write_header(&mut buf, 4096, 0, 1700000000);
        assert_eq!(read_collection_count(&buf), 0);

        write_collection_count(&mut buf, 5);
        assert_eq!(read_collection_count(&buf), 5);
    }
}
