//! # LiteVec
//!
//! The embedded vector database. No server. No Docker. No config.
//!
//! ```rust,no_run
//! use litevec::Database;
//!
//! let db = Database::open("my_vectors.lv").unwrap();
//! let collection = db.create_collection("docs", 384).unwrap();
//! ```

pub use litevec_core::*;
