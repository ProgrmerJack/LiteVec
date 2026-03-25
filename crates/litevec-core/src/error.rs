//! Error types for LiteVec.

/// All errors that LiteVec operations can produce.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },

    #[error("Collection '{0}' already exists")]
    CollectionExists(String),

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Vector '{0}' not found")]
    VectorNotFound(String),

    #[error("Invalid database file: {0}")]
    InvalidFile(String),

    #[error("WAL corruption: {0}")]
    WalCorruption(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// A specialized Result type for LiteVec operations.
pub type Result<T> = std::result::Result<T, Error>;
