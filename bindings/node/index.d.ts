export interface SearchResult {
  id: string;
  distance: number;
  metadata: string;
}

export interface VectorRecord {
  id: string;
  vector: number[];
  metadata: string;
}

export class Database {
  static open(path: string): Database;
  static openMemory(): Database;
  createCollection(name: string, dimension: number): Collection;
  getCollection(name: string): Collection;
  deleteCollection(name: string): void;
  listCollections(): string[];
  checkpoint(): void;
}

export class Collection {
  insert(id: string, vector: number[], metadata?: string): void;
  search(query: number[], k: number): SearchResult[];
  get(id: string): VectorRecord | null;
  delete(id: string): boolean;
  len(): number;
  dimension(): number;
  name(): string;
}
