use super::NativeNgramIndex;
use crate::fingerprint::exact_fingerprint;
use crate::index::workspace::QueryWorkspace;
use crate::ngrams::char_ngram_slices;
use crate::types::{GramId, NeighborTuple};

impl NativeNgramIndex {
    /// Return exact top-k Jaccard neighbors.
    ///
    /// Exact mode visits every posting list for query grams. It uses a reusable
    /// vector-backed workspace instead of allocating a candidate HashMap for
    /// each query. The result ordering is deterministic: higher score first,
    /// lower training index on ties.
    pub(crate) fn query_topk_exact_impl(
        &self,
        query_norm: &str,
        k: usize,
        workspace: &mut QueryWorkspace,
    ) -> Vec<NeighborTuple> {
        self.query_topk_with_mode(query_norm, k, QueryMode::Exact, workspace)
    }

    /// Return fast approximate top-k Jaccard neighbors.
    ///
    /// Fast mode uses bounded rare-gram candidate collection followed by exact
    /// reranking inside that candidate set. It is intentionally separate from
    /// exact mode so approximate results cannot be mistaken for exact search.
    pub(crate) fn query_topk_fast_impl(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        let mut workspace = QueryWorkspace::new(self.gram_sets.len());
        self.query_topk_with_mode(query_norm, k, QueryMode::Fast, &mut workspace)
    }

    pub(crate) fn query_topk_impl(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        let mut workspace = QueryWorkspace::new(self.gram_sets.len());
        let mode = if self.mode == "fast" {
            QueryMode::Fast
        } else {
            QueryMode::Exact
        };
        self.query_topk_with_mode(query_norm, k, mode, &mut workspace)
    }

    pub(crate) fn query_topk_with_workspace(
        &self,
        query_norm: &str,
        k: usize,
        workspace: &mut QueryWorkspace,
    ) -> Vec<NeighborTuple> {
        let mode = if self.mode == "fast" {
            QueryMode::Fast
        } else {
            QueryMode::Exact
        };
        self.query_topk_with_mode(query_norm, k, mode, workspace)
    }

    fn query_topk_with_mode(
        &self,
        query_norm: &str,
        k: usize,
        mode: QueryMode,
        workspace: &mut QueryWorkspace,
    ) -> Vec<NeighborTuple> {
        if k == 0 {
            return Vec::new();
        }

        if let Some(index) = self.exact_map.get(&exact_fingerprint(query_norm)) {
            return vec![(Some(*index), 1.0, true)];
        }

        let query_grams = char_ngram_slices(query_norm, &self.ngram_orders);
        let (query_count, query_ids) = self.query_gram_ids_from_grams(query_grams);
        if query_count == 0 || query_ids.is_empty() {
            return Vec::new();
        }

        match mode {
            QueryMode::Exact => self.rank_exact_candidates(query_count, &query_ids, k, workspace),
            QueryMode::Fast => {
                let intersection_counts = self.candidate_counts_fast(&query_ids);
                self.rank_fast_candidates(query_count, &query_ids, intersection_counts, k)
            }
        }
    }

    pub(crate) fn query_gram_ids_from_grams(&self, grams: Vec<&[u8]>) -> (usize, Vec<GramId>) {
        let query_count = grams.len();
        let mut query_ids: Vec<GramId> = grams
            .iter()
            .filter_map(|gram| self.gram_to_id.get(*gram).copied())
            .collect();
        query_ids.sort_unstable();
        (query_count, query_ids)
    }
}

#[derive(Clone, Copy)]
enum QueryMode {
    Exact,
    Fast,
}
