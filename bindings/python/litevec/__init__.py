"""LiteVec — The embedded vector database."""

from litevec._native import (
    PyDatabase as Database,
    PyCollection as Collection,
    PySearchResult as SearchResult,
)

__all__ = ["open", "open_memory", "Database", "Collection", "SearchResult"]


def open(path: str) -> Database:
    """Open or create a LiteVec database at the given path."""
    return Database.open(path)


def open_memory() -> Database:
    """Open an in-memory LiteVec database."""
    return Database.open_memory()
