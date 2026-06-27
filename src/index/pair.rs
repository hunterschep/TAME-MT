use super::NativeNgramIndex;
use crate::ngrams::char_ngram_slices;
use crate::similarity::jaccard_ids;
use crate::types::{GramId, NeighborTuple};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

impl NativeNgramIndex {
    /// Rerank a caller-supplied pair candidate set by min(source, target).
    ///
    /// Pair exposure here is exact only within the candidate indices supplied
    /// by Python. Report generation labels this as top-k reranking unless a
    /// separate exact threshold pass proves no-false-negative threshold flags.
    pub(crate) fn best_pair_candidate_impl(
        &self,
        target_index: &NativeNgramIndex,
        source_query_norm: &str,
        target_query_norms: &[String],
        candidate_indices: &[usize],
    ) -> PyResult<NeighborTuple> {
        if candidate_indices.is_empty() {
            return Ok((None, 0.0, false));
        }
        let mut sorted_candidates = candidate_indices.to_vec();
        sorted_candidates.sort_unstable();
        sorted_candidates.dedup();

        let source_grams = char_ngram_slices(source_query_norm, &self.ngram_orders);
        let (source_count, source_ids) = self.query_gram_ids_from_grams(source_grams);
        let target_queries: Vec<(usize, Vec<GramId>)> = target_query_norms
            .iter()
            .map(|query| {
                target_index
                    .query_gram_ids_from_grams(char_ngram_slices(query, &target_index.ngram_orders))
            })
            .collect();

        let mut best_index: Option<usize> = None;
        let mut best_score = 0.0_f64;
        for index in sorted_candidates {
            if index >= self.gram_sets.len() || index >= target_index.gram_sets.len() {
                return Err(PyIndexError::new_err(format!(
                    "candidate index out of range: {index}"
                )));
            }
            let source_score = jaccard_ids(source_count, &source_ids, &self.gram_sets[index]);
            let target_score = target_queries
                .iter()
                .map(|(target_count, target_ids)| {
                    jaccard_ids(*target_count, target_ids, &target_index.gram_sets[index])
                })
                .fold(0.0_f64, f64::max);
            let pair_score = source_score.min(target_score);
            if pair_score > best_score {
                best_index = Some(index);
                best_score = pair_score;
            }
        }
        Ok((best_index, best_score, best_score == 1.0))
    }
}
