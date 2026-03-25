/**
 * LiteVec Node.js Quickstart Example
 *
 * Demonstrates: create database → insert vectors → search → get results
 *
 * Requirements:
 *   npm install litevec
 */

const { Database } = require("litevec");

async function main() {
  // Open an in-memory database
  const db = Database.openMemory();

  // Create a collection for 128-dimensional vectors
  const collection = db.createCollection("documents", 128);

  // Sample vectors (in practice, these come from an embedding model)
  const documents = [
    {
      id: "doc1",
      vector: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.1)),
      metadata: JSON.stringify({ title: "Introduction to AI", category: "ai" }),
    },
    {
      id: "doc2",
      vector: Array.from({ length: 128 }, (_, i) => Math.cos(i * 0.1)),
      metadata: JSON.stringify({
        title: "Machine Learning Basics",
        category: "ml",
      }),
    },
    {
      id: "doc3",
      vector: Array.from({ length: 128 }, (_, i) =>
        Math.sin(i * 0.1 + Math.PI / 4)
      ),
      metadata: JSON.stringify({
        title: "Neural Networks",
        category: "deep-learning",
      }),
    },
  ];

  // Insert vectors
  console.log("Inserting vectors...");
  for (const doc of documents) {
    collection.insert(doc.id, doc.vector, doc.metadata);
  }
  console.log(`Inserted ${collection.len()} vectors\n`);

  // Search for similar vectors
  const query = Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.1 + 0.1));
  console.log("Searching for similar vectors...");
  const results = collection.search(query, 3);

  console.log("Results:");
  for (const result of results) {
    const meta = JSON.parse(result.metadata || "{}");
    console.log(`  ${result.id}: distance=${result.distance.toFixed(4)}`);
    console.log(`    Title: ${meta.title || "N/A"}`);
  }

  // Get a specific vector
  console.log("\nGetting doc1...");
  const record = collection.get("doc1");
  if (record) {
    console.log(`  ID: ${record.id}`);
    console.log(`  Dimension: ${record.vector.length}`);
    console.log(`  Metadata: ${record.metadata}`);
  }

  // Delete a vector
  console.log("\nDeleting doc2...");
  const deleted = collection.delete("doc2");
  console.log(`  Deleted: ${deleted}`);
  console.log(`  Collection size: ${collection.len()}`);

  console.log("\n✅ Done!");
}

main().catch(console.error);
