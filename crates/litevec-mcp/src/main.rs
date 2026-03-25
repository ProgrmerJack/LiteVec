use std::io::{self, BufRead, Write};

use clap::Parser;
use litevec_core::{Database, Filter};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "litevec-mcp", about = "MCP server for LiteVec")]
struct Cli {
    /// Path to the LiteVec database file
    #[arg(long)]
    database: String,
}

fn main() {
    let cli = Cli::parse();

    let db = match Database::open(&cli.database) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Failed to open database at {}: {e}", cli.database);
            std::process::exit(1);
        }
    };

    eprintln!("litevec-mcp: database opened at {}", cli.database);

    let stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    for line in stdin.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("litevec-mcp: stdin read error: {e}");
                break;
            }
        };

        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("litevec-mcp: invalid JSON: {e}");
                continue;
            }
        };

        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(json!({}));

        eprintln!("litevec-mcp: received method={method}");

        let response = match method {
            "initialize" => Some(handle_initialize(&id)),
            "initialized" => None, // notification, no response
            "tools/list" => Some(handle_tools_list(&id)),
            "tools/call" => Some(handle_tools_call(&id, &params, &db)),
            _ => id
                .as_ref()
                .map(|_| make_error_response(&id, -32601, "Method not found")),
        };

        if let Some(resp) = response {
            let out = serde_json::to_string(&resp).expect("failed to serialize response");
            let _ = writeln!(stdout, "{out}");
            let _ = stdout.flush();
        }
    }

    eprintln!("litevec-mcp: shutting down");
}

// ---------------------------------------------------------------------------
// Protocol handlers
// ---------------------------------------------------------------------------

fn handle_initialize(id: &Option<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "litevec-mcp", "version": "0.1.0" }
        }
    })
}

fn handle_tools_list(id: &Option<Value>) -> Value {
    let tools = vec![
        tool_def(
            "litevec_create_collection",
            "Create a new vector collection",
            json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "dimension": { "type": "integer", "description": "Vector dimension" }
                },
                "required": ["collection", "dimension"]
            }),
        ),
        tool_def(
            "litevec_insert",
            "Insert a vector into a collection",
            json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "id": { "type": "string", "description": "Vector ID" },
                    "vector": { "type": "array", "items": { "type": "number" }, "description": "Vector values" },
                    "metadata": { "type": "object", "description": "Optional metadata" }
                },
                "required": ["collection", "id", "vector"]
            }),
        ),
        tool_def(
            "litevec_search",
            "Search for similar vectors",
            json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "vector": { "type": "array", "items": { "type": "number" }, "description": "Query vector" },
                    "k": { "type": "integer", "description": "Number of results (default 10)" },
                    "filter": { "type": "object", "description": "Optional metadata filter" }
                },
                "required": ["collection", "vector"]
            }),
        ),
        tool_def(
            "litevec_get",
            "Get a vector by ID",
            json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "id": { "type": "string", "description": "Vector ID" }
                },
                "required": ["collection", "id"]
            }),
        ),
        tool_def(
            "litevec_delete",
            "Delete a vector by ID",
            json!({
                "type": "object",
                "properties": {
                    "collection": { "type": "string", "description": "Collection name" },
                    "id": { "type": "string", "description": "Vector ID" }
                },
                "required": ["collection", "id"]
            }),
        ),
        tool_def(
            "litevec_info",
            "Get database info (list collections and sizes)",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),
    ];

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "tools": tools }
    })
}

fn handle_tools_call(id: &Option<Value>, params: &Value, db: &Database) -> Value {
    let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let result = match tool_name {
        "litevec_create_collection" => tool_create_collection(&args, db),
        "litevec_insert" => tool_insert(&args, db),
        "litevec_search" => tool_search(&args, db),
        "litevec_get" => tool_get(&args, db),
        "litevec_delete" => tool_delete(&args, db),
        "litevec_info" => tool_info(db),
        _ => Err(format!("Unknown tool: {tool_name}")),
    };

    match result {
        Ok(text) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{ "type": "text", "text": text }]
            }
        }),
        Err(e) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{ "type": "text", "text": format!("Error: {e}") }],
                "isError": true
            }
        }),
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

fn tool_create_collection(args: &Value, db: &Database) -> Result<String, String> {
    let name = arg_str(args, "collection")?;
    let dim = arg_u32(args, "dimension")?;
    db.create_collection(&name, dim)
        .map_err(|e| e.to_string())?;
    Ok(format!("Created collection '{name}' with dimension {dim}"))
}

fn tool_insert(args: &Value, db: &Database) -> Result<String, String> {
    let coll_name = arg_str(args, "collection")?;
    let id = arg_str(args, "id")?;
    let vector = arg_f32_vec(args, "vector")?;
    let metadata = args.get("metadata").cloned().unwrap_or(json!({}));

    let coll = db
        .get_collection(&coll_name)
        .ok_or_else(|| format!("Collection '{coll_name}' not found"))?;
    coll.insert(&id, &vector, metadata)
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "Inserted vector '{id}' into collection '{coll_name}'"
    ))
}

fn tool_search(args: &Value, db: &Database) -> Result<String, String> {
    let coll_name = arg_str(args, "collection")?;
    let vector = arg_f32_vec(args, "vector")?;
    let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let coll = db
        .get_collection(&coll_name)
        .ok_or_else(|| format!("Collection '{coll_name}' not found"))?;

    let filter_val = args.get("filter");
    let query = coll.search(&vector, k);
    let query = if let Some(f) = filter_val {
        let filter = parse_filter(f)?;
        query.filter(filter)
    } else {
        query
    };

    let results = query.execute().map_err(|e| e.to_string())?;

    let items: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "distance": r.distance,
                "metadata": r.metadata,
            })
        })
        .collect();

    Ok(serde_json::to_string(&items).unwrap())
}

fn tool_get(args: &Value, db: &Database) -> Result<String, String> {
    let coll_name = arg_str(args, "collection")?;
    let id = arg_str(args, "id")?;

    let coll = db
        .get_collection(&coll_name)
        .ok_or_else(|| format!("Collection '{coll_name}' not found"))?;

    match coll.get(&id).map_err(|e| e.to_string())? {
        Some(record) => Ok(serde_json::to_string(&json!({
            "id": record.id,
            "vector": record.vector,
            "metadata": record.metadata,
        }))
        .unwrap()),
        None => Err(format!(
            "Vector '{id}' not found in collection '{coll_name}'"
        )),
    }
}

fn tool_delete(args: &Value, db: &Database) -> Result<String, String> {
    let coll_name = arg_str(args, "collection")?;
    let id = arg_str(args, "id")?;

    let coll = db
        .get_collection(&coll_name)
        .ok_or_else(|| format!("Collection '{coll_name}' not found"))?;

    let deleted = coll.delete(&id).map_err(|e| e.to_string())?;
    if deleted {
        Ok(format!(
            "Deleted vector '{id}' from collection '{coll_name}'"
        ))
    } else {
        Err(format!(
            "Vector '{id}' not found in collection '{coll_name}'"
        ))
    }
}

fn tool_info(db: &Database) -> Result<String, String> {
    let names = db.list_collections();
    let mut collections = Vec::new();
    for name in &names {
        if let Some(coll) = db.get_collection(name) {
            collections.push(json!({
                "name": name,
                "dimension": coll.dimension(),
                "size": coll.len(),
            }));
        }
    }
    Ok(serde_json::to_string(&json!({ "collections": collections })).unwrap())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tool_def(name: &str, description: &str, input_schema: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema
    })
}

fn make_error_response(id: &Option<Value>, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
}

fn arg_str(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| format!("Missing or invalid argument: {key}"))
}

fn arg_u32(args: &Value, key: &str) -> Result<u32, String> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| format!("Missing or invalid argument: {key}"))
}

fn arg_f32_vec(args: &Value, key: &str) -> Result<Vec<f32>, String> {
    args.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .ok_or_else(|| format!("Missing or invalid argument: {key}"))
}

fn parse_filter(val: &Value) -> Result<Filter, String> {
    let obj = val.as_object().ok_or("Filter must be an object")?;

    // Simple equality filter: {"field": "value"} or {"$and": [...]}
    if let Some(and) = obj.get("$and") {
        let arr = and.as_array().ok_or("$and must be an array")?;
        let filters: Result<Vec<Filter>, String> = arr.iter().map(parse_filter).collect();
        return Ok(Filter::And(filters?));
    }
    if let Some(or) = obj.get("$or") {
        let arr = or.as_array().ok_or("$or must be an array")?;
        let filters: Result<Vec<Filter>, String> = arr.iter().map(parse_filter).collect();
        return Ok(Filter::Or(filters?));
    }

    // Single-field filters: {"field": value} => Eq, {"field": {"$gt": n}} => Gt, etc.
    let mut filters = Vec::new();
    for (field, value) in obj {
        if let Some(inner) = value.as_object() {
            for (op, v) in inner {
                let f = match op.as_str() {
                    "$eq" => Filter::Eq(field.clone(), v.clone()),
                    "$ne" => Filter::Ne(field.clone(), v.clone()),
                    "$gt" => Filter::Gt(field.clone(), v.as_f64().ok_or("$gt requires number")?),
                    "$gte" => Filter::Gte(field.clone(), v.as_f64().ok_or("$gte requires number")?),
                    "$lt" => Filter::Lt(field.clone(), v.as_f64().ok_or("$lt requires number")?),
                    "$lte" => Filter::Lte(field.clone(), v.as_f64().ok_or("$lte requires number")?),
                    "$in" => {
                        let arr = v.as_array().ok_or("$in requires array")?;
                        Filter::In(field.clone(), arr.clone())
                    }
                    _ => return Err(format!("Unknown filter operator: {op}")),
                };
                filters.push(f);
            }
        } else {
            filters.push(Filter::Eq(field.clone(), value.clone()));
        }
    }

    match filters.len() {
        0 => Err("Empty filter".into()),
        1 => Ok(filters.into_iter().next().unwrap()),
        _ => Ok(Filter::And(filters)),
    }
}
