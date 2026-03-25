//! Query builder and execution.

use crate::error::Result;
use crate::types::{Filter, SearchResult};

/// A builder for configuring and executing a vector search.
pub struct SearchQuery {
    pub(crate) query: Vec<f32>,
    pub(crate) k: usize,
    pub(crate) filter: Option<Filter>,
    pub(crate) ef_search: Option<usize>,
    pub(crate) collection: crate::collection::CollectionInner,
}

impl SearchQuery {
    /// Add a metadata filter.
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Override the ef_search parameter for this query.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Execute the search and return results.
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        self.collection
            .execute_search(&self.query, self.k, self.filter.as_ref(), self.ef_search)
    }
}
