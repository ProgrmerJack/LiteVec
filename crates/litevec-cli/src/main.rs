use std::io::{self, BufRead};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use litevec_core::{CollectionConfig, Database, DistanceType, Filter, IndexType};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "litevec",
    version,
    about = "The embedded vector database. No server. No Docker. No config."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new collection
    Create {
        /// Database file path
        db: PathBuf,
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Vector dimension
        #[arg(long)]
        dimension: u32,
        /// Distance metric (cosine, euclidean, dot)
        #[arg(long, default_value = "cosine")]
        distance: String,
        /// Index type (flat, hnsw, auto)
        #[arg(long, default_value = "auto")]
        index: String,
    },
    /// Insert vectors from a JSONL file or stdin
    Insert {
        /// Database file path
        db: PathBuf,
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Input JSONL file (omit for stdin)
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Search for similar vectors
    Search {
        /// Database file path
        db: PathBuf,
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Query vector as JSON array (e.g., '[0.1, 0.2, 0.3]')
        #[arg(long)]
        vector: String,
        /// Number of results
        #[arg(long, short, default_value = "10")]
        k: usize,
        /// Filter as JSON (e.g., '{"category": "docs"}')
        #[arg(long)]
        filter: Option<String>,
    },
    /// Get a vector by ID
    Get {
        /// Database file path
        db: PathBuf,
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Vector ID
        id: String,
    },
    /// Delete a vector by ID
    Delete {
        /// Database file path
        db: PathBuf,
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Vector ID
        id: String,
    },
    /// List collections in a database
    List {
        /// Database file path
        db: PathBuf,
    },
    /// Show database or collection info
    Info {
        /// Database file path
        db: PathBuf,
        /// Optional collection name
        #[arg(long)]
        collection: Option<String>,
    },
}

fn parse_distance(s: &str) -> Result<DistanceType, String> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceType::Cosine),
        "euclidean" | "l2" => Ok(DistanceType::Euclidean),
        "dot" | "dotproduct" | "dot_product" => Ok(DistanceType::DotProduct),
        other => Err(format!("unknown distance type: {other}")),
    }
}

fn parse_index(s: &str) -> Result<IndexType, String> {
    match s.to_lowercase().as_str() {
        "flat" => Ok(IndexType::Flat),
        "hnsw" => Ok(IndexType::Hnsw),
        "auto" => Ok(IndexType::Auto),
        other => Err(format!("unknown index type: {other}")),
    }
}

fn parse_simple_filter(value: &Value) -> Option<Filter> {
    match value {
        Value::Object(map) => {
            let filters: Vec<Filter> = map
                .iter()
                .map(|(k, v)| Filter::Eq(k.clone(), v.clone()))
                .collect();
            match filters.len() {
                0 => None,
                1 => Some(filters.into_iter().next().unwrap()),
                _ => Some(Filter::And(filters)),
            }
        }
        _ => None,
    }
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Create {
            db,
            collection,
            dimension,
            distance,
            index,
        } => {
            let dist = parse_distance(&distance)?;
            let idx = parse_index(&index)?;
            let db = Database::open(&db)?;
            let config = CollectionConfig {
                dimension,
                distance: dist,
                index: idx,
                ..CollectionConfig::new(dimension)
            };
            db.create_collection_with_config(&collection, config)?;
            println!(
                "Created collection '{collection}' (dim={dimension}, distance={distance}, index={index})"
            );
        }

        Commands::Insert {
            db,
            collection,
            input,
        } => {
            let db = Database::open(&db)?;
            let col = db.get_collection(&collection).ok_or_else(|| {
                format!(
                    "Collection '{collection}' not found. Create it first with 'litevec create'."
                )
            })?;

            let reader: Box<dyn BufRead> = match input {
                Some(path) => Box::new(io::BufReader::new(std::fs::File::open(path)?)),
                None => Box::new(io::stdin().lock()),
            };

            let mut count = 0;
            for line in reader.lines() {
                let line = line?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let obj: Value = serde_json::from_str(line)?;
                let id = obj["id"]
                    .as_str()
                    .ok_or("each line must have an 'id' string field")?;
                let vector: Vec<f32> = obj["vector"]
                    .as_array()
                    .ok_or("each line must have a 'vector' array field")?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                let metadata = obj
                    .get("metadata")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));

                col.insert(id, &vector, metadata)?;
                count += 1;
            }
            println!("Inserted {count} vectors into '{collection}'");
        }

        Commands::Search {
            db,
            collection,
            vector,
            k,
            filter,
        } => {
            let db = Database::open(&db)?;
            let col = db
                .get_collection(&collection)
                .ok_or(format!("Collection '{collection}' not found"))?;

            let query: Vec<f32> = serde_json::from_str(&vector)?;

            let mut search = col.search(&query, k);
            if let Some(filter_str) = filter {
                let filter_val: Value = serde_json::from_str(&filter_str)?;
                if let Some(f) = parse_simple_filter(&filter_val) {
                    search = search.filter(f);
                }
            }

            let results = search.execute()?;
            for (i, r) in results.iter().enumerate() {
                let meta = serde_json::to_string(&r.metadata)?;
                println!("{}. {} (distance: {:.6}) {}", i + 1, r.id, r.distance, meta);
            }
            if results.is_empty() {
                println!("No results found.");
            }
        }

        Commands::Get { db, collection, id } => {
            let db = Database::open(&db)?;
            let col = db
                .get_collection(&collection)
                .ok_or(format!("Collection '{collection}' not found"))?;

            match col.get(&id)? {
                Some(record) => {
                    println!("ID: {}", record.id);
                    println!("Dimension: {}", record.vector.len());
                    let preview: String = record
                        .vector
                        .iter()
                        .take(5)
                        .map(|v| format!("{v:.4}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let suffix = if record.vector.len() > 5 {
                        format!(", ... ({} total)", record.vector.len())
                    } else {
                        String::new()
                    };
                    println!("Vector: [{preview}{suffix}]");
                    println!(
                        "Metadata: {}",
                        serde_json::to_string_pretty(&record.metadata)?
                    );
                }
                None => {
                    println!("Vector '{id}' not found.");
                }
            }
        }

        Commands::Delete { db, collection, id } => {
            let db = Database::open(&db)?;
            let col = db
                .get_collection(&collection)
                .ok_or(format!("Collection '{collection}' not found"))?;

            if col.delete(&id)? {
                println!("Deleted '{id}' from '{collection}'");
            } else {
                println!("Vector '{id}' not found in '{collection}'");
            }
        }

        Commands::List { db } => {
            let db = Database::open(&db)?;
            let names = db.list_collections();
            if names.is_empty() {
                println!("No collections.");
            } else {
                println!("Collections:");
                for name in &names {
                    if let Some(col) = db.get_collection(name) {
                        println!("  {name}: {} vectors, dim={}", col.len(), col.dimension());
                    }
                }
            }
        }

        Commands::Info {
            db: path,
            collection,
        } => {
            let db = Database::open(&path)?;
            let names = db.list_collections();

            if let Some(col_name) = collection {
                let col = db
                    .get_collection(&col_name)
                    .ok_or(format!("Collection '{col_name}' not found"))?;
                println!("Collection: {col_name}");
                println!("  Vectors: {}", col.len());
                println!("  Dimension: {}", col.dimension());
            } else {
                let file_size = std::fs::metadata(&path)
                    .map(|m| format_bytes(m.len()))
                    .unwrap_or_else(|_| "unknown".to_string());
                println!("Database: {} ({})", path.display(), file_size);
                println!("Collections: {}", names.len());
                for name in &names {
                    if let Some(col) = db.get_collection(name) {
                        println!("  {name}: {} vectors, dim={}", col.len(), col.dimension());
                    }
                }
            }
        }
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
