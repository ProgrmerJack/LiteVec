use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Instant;

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
    /// Start a lightweight HTTP API server
    Serve {
        /// Database file path
        db: PathBuf,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// Run built-in benchmarks
    Bench {
        /// Vector dimension
        #[arg(long, default_value = "128")]
        dimension: u32,
        /// Number of vectors to insert
        #[arg(long, default_value = "10000")]
        count: usize,
        /// Number of search queries
        #[arg(long, default_value = "100")]
        queries: usize,
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

        Commands::Serve { db, port } => {
            run_serve(db, port)?;
        }

        Commands::Bench {
            dimension,
            count,
            queries,
        } => {
            run_bench(dimension, count, queries)?;
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

// ---------------------------------------------------------------------------
// serve — lightweight HTTP API for quick testing (not production)
// ---------------------------------------------------------------------------

fn run_serve(db_path: PathBuf, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open(&db_path)?;
    let db = std::sync::Arc::new(db);

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("LiteVec HTTP server listening on http://{addr}");
    println!("Database: {}", db_path.display());
    println!("Press Ctrl+C to stop.\n");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };

        let db = db.clone();
        // Handle each request synchronously (lightweight server for quick testing)
        if let Err(e) = handle_http_request(&mut stream, &db) {
            let _ = write!(stream, "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n{{\"error\":\"{e}\"}}");
        }
    }

    Ok(())
}

fn handle_http_request(
    stream: &mut std::net::TcpStream,
    db: &Database,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(stream.try_clone()?);

    // Parse request line
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        send_json(stream, 400, r#"{"error":"bad request"}"#)?;
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1];

    // Read headers to find Content-Length
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
        if let Some(val) = trimmed.strip_prefix("content-length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
    }

    // Read body
    let body = if content_length > 0 {
        let mut buf = vec![0u8; content_length];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).unwrap_or_default()
    } else {
        String::new()
    };

    // Route
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();

    match (method, segments.as_slice()) {
        // GET /collections — list all collections
        ("GET", ["collections"]) => {
            let names = db.list_collections();
            let json = serde_json::to_string(&names)?;
            send_json(stream, 200, &json)?;
        }

        // POST /collections/{name}?dimension=N — create collection
        ("POST", ["collections", name]) => {
            let params: Value = if body.is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(&body)?
            };
            let dim = params["dimension"].as_u64().unwrap_or(128) as u32;
            db.create_collection(name, dim)?;
            send_json(stream, 201, &format!(r#"{{"created":"{name}","dimension":{dim}}}"#))?;
        }

        // POST /collections/{name}/insert — insert vector
        ("POST", ["collections", name, "insert"]) => {
            let col = db
                .get_collection(name)
                .ok_or(format!("collection '{name}' not found"))?;
            let params: Value = serde_json::from_str(&body)?;
            let id = params["id"].as_str().ok_or("missing 'id'")?;
            let vector: Vec<f32> = params["vector"]
                .as_array()
                .ok_or("missing 'vector'")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            let metadata = params.get("metadata").cloned().unwrap_or(Value::Object(Default::default()));
            col.insert(id, &vector, metadata)?;
            send_json(stream, 200, r#"{"ok":true}"#)?;
        }

        // POST /collections/{name}/search — search
        ("POST", ["collections", name, "search"]) => {
            let col = db
                .get_collection(name)
                .ok_or(format!("collection '{name}' not found"))?;
            let params: Value = serde_json::from_str(&body)?;
            let vector: Vec<f32> = params["vector"]
                .as_array()
                .ok_or("missing 'vector'")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            let k = params["k"].as_u64().unwrap_or(10) as usize;
            let results = col.search(&vector, k).execute()?;
            let out: Vec<Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id,
                        "distance": r.distance,
                        "metadata": r.metadata,
                    })
                })
                .collect();
            send_json(stream, 200, &serde_json::to_string(&out)?)?;
        }

        // GET /collections/{name}/{id} — get vector by ID
        ("GET", ["collections", name, id]) if *id != "search" && *id != "insert" => {
            let col = db
                .get_collection(name)
                .ok_or(format!("collection '{name}' not found"))?;
            match col.get(id)? {
                Some(record) => {
                    let out = serde_json::json!({
                        "id": record.id,
                        "vector": record.vector,
                        "metadata": record.metadata,
                    });
                    send_json(stream, 200, &serde_json::to_string(&out)?)?;
                }
                None => {
                    send_json(stream, 404, r#"{"error":"not found"}"#)?;
                }
            }
        }

        // DELETE /collections/{name}/{id} — delete vector
        ("DELETE", ["collections", name, id]) => {
            let col = db
                .get_collection(name)
                .ok_or(format!("collection '{name}' not found"))?;
            let deleted = col.delete(id)?;
            send_json(stream, 200, &format!(r#"{{"deleted":{deleted}}}"#))?;
        }

        _ => {
            send_json(stream, 404, r#"{"error":"not found"}"#)?;
        }
    }

    Ok(())
}

fn send_json(
    stream: &mut std::net::TcpStream,
    status: u16,
    body: &str,
) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

// ---------------------------------------------------------------------------
// bench — built-in benchmarks
// ---------------------------------------------------------------------------

fn run_bench(
    dimension: u32,
    count: usize,
    queries: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("LiteVec Built-in Benchmark");
    println!("==========================");
    println!("Dimension:      {dimension}");
    println!("Vectors:        {count}");
    println!("Search queries: {queries}");
    println!();

    let db = Database::open_memory()?;

    // --- Insert benchmark ---
    let config = CollectionConfig {
        dimension,
        distance: DistanceType::Cosine,
        index: IndexType::Hnsw,
        ..CollectionConfig::new(dimension)
    };
    let col = db.create_collection_with_config("bench", config)?;

    println!("Inserting {count} vectors...");
    let start = Instant::now();
    for i in 0..count {
        let vector = random_vector(dimension as usize, i as u64);
        let metadata = serde_json::json!({"i": i});
        col.insert(&format!("v{i}"), &vector, metadata)?;
    }
    let insert_elapsed = start.elapsed();
    let insert_per_sec = count as f64 / insert_elapsed.as_secs_f64();
    println!(
        "  Insert:  {:.2}s total, {:.0} vectors/sec",
        insert_elapsed.as_secs_f64(),
        insert_per_sec
    );

    // --- Search benchmark ---
    println!("Running {queries} searches (k=10)...");
    let start = Instant::now();
    for i in 0..queries {
        let query = random_vector(dimension as usize, (count + i) as u64);
        let _results = col.search(&query, 10).execute()?;
    }
    let search_elapsed = start.elapsed();
    let search_avg_us = search_elapsed.as_micros() as f64 / queries as f64;
    let search_per_sec = queries as f64 / search_elapsed.as_secs_f64();
    println!(
        "  Search:  {:.2}s total, {:.1} µs/query, {:.0} queries/sec",
        search_elapsed.as_secs_f64(),
        search_avg_us,
        search_per_sec
    );

    // --- Filtered search benchmark ---
    println!("Running {queries} filtered searches (k=10, i < {})...", count / 2);
    let start = Instant::now();
    for i in 0..queries {
        let query = random_vector(dimension as usize, (count * 2 + i) as u64);
        let _results = col
            .search(&query, 10)
            .filter(Filter::Lt("i".to_string(), (count / 2) as f64))
            .execute()?;
    }
    let filtered_elapsed = start.elapsed();
    let filtered_avg_us = filtered_elapsed.as_micros() as f64 / queries as f64;
    println!(
        "  Filtered: {:.2}s total, {:.1} µs/query",
        filtered_elapsed.as_secs_f64(),
        filtered_avg_us
    );

    println!("\nDone.");
    Ok(())
}

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(((state >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}
