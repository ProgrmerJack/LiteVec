export interface SearchResult {
  id: string;
  distance: number;
  /** JSON-encoded metadata string. Parse with `JSON.parse()`. */
  metadata: string;
}

export interface VectorRecord {
  id: string;
  vector: number[];
  /** JSON-encoded metadata string. Parse with `JSON.parse()`. */
  metadata: string;
}

export interface BatchItem {
  id: string;
  vector: number[];
  /** Optional JSON-encoded metadata string. */
  metadata?: string;
}

export class Database {
  static open(path: string): Database;
  static openMemory(): Database;
  createCollection(name: string, dimension: number): Collection;
  getCollection(name: string): Collection;
  deleteCollection(name: string): void;
  listCollections(): string[];
  checkpoint(): void;
  /** Create a backup snapshot at the given path. */
  createBackup(path: string): void;
}

export class Collection {
  insert(id: string, vector: number[], metadata?: string): void;
  /**
   * Insert multiple vectors in a batch.
   * Returns the number of inserted items.
   */
  insertBatch(items: BatchItem[]): number;
  /**
   * Search for the k nearest vectors.
   * An optional JSON filter string restricts results.
   * Supports operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or.
   */
  search(query: number[], k: number, filter?: string): SearchResult[];
  /** Full-text keyword search across metadata string fields. */
  textSearch(query: string, limit?: number): SearchResult[];
  /** Combined vector + keyword search using Reciprocal Rank Fusion. */
  hybridSearch(vector: number[], query: string, k?: number): SearchResult[];
  get(id: string): VectorRecord | null;
  delete(id: string): boolean;
  /** Update metadata for an existing vector. Metadata is a JSON string. */
  updateMetadata(id: string, metadata: string): void;
  /** Create a secondary index on a metadata field. */
  createIndex(field: string): void;
  /** Drop a secondary index on a metadata field. */
  dropIndex(field: string): void;
  /** List the currently indexed metadata fields. */
  indexedFields(): string[];
  len(): number;
  isEmpty(): boolean;
  dimension(): number;
  name(): string;
}
