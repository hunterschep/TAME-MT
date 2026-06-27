use super::NativeNgramIndex;
use crate::ngrams::char_ngram_slices;
use crate::similarity::{intersection_size_ids, jaccard_ids};
use crate::types::{CandidateCountMap, GramId, NeighborTuple};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

impl NativeNgramIndex {
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

    pub(crate) fn query_topk_impl(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        if k == 0 {
            return Vec::new();
        }

        if let Some(index) = self.exact_map.get(query_norm) {
            return vec![(Some(*index), 1.0, true)];
        }

        let query_grams = char_ngram_slices(query_norm, &self.ngram_orders);
        let (query_count, query_ids) = self.query_gram_ids_from_grams(query_grams);
        if query_count == 0 || query_ids.is_empty() {
            return Vec::new();
        }

        let intersection_counts = if self.mode == "fast" {
            self.candidate_counts_fast(&query_ids)
        } else {
            self.candidate_counts_exact(&query_ids)
        };
        self.rank_candidates(query_count, &query_ids, intersection_counts, k)
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

    fn candidate_counts_exact(&self, query_ids: &[GramId]) -> CandidateCountMap {
        let mut counts = CandidateCountMap::default();
        for gram_id in query_ids {
            for index in &self.postings[*gram_id as usize] {
                *counts.entry(*index as usize).or_insert(0) += 1;
            }
        }
        counts
    }

    fn candidate_counts_fast(&self, query_ids: &[GramId]) -> CandidateCountMap {
        let mut ranked_grams: Vec<(usize, GramId)> = query_ids
            .iter()
            .map(|gram_id| (self.postings[*gram_id as usize].len(), *gram_id))
            .collect();
        ranked_grams.sort_unstable_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
        });

        let mut selected: Vec<GramId> = ranked_grams
            .iter()
            .filter_map(|(posting_count, gram_id)| {
                if *posting_count > 0 && *posting_count <= self.posting_limit {
                    Some(*gram_id)
                } else {
                    None
                }
            })
            .take(self.candidate_gram_limit)
            .collect();

        if selected.is_empty() {
            selected = ranked_grams
                .iter()
                .filter_map(|(posting_count, gram_id)| {
                    if *posting_count > 0 {
                        Some(*gram_id)
                    } else {
                        None
                    }
                })
                .take(self.candidate_gram_limit)
                .collect();
        }

        let mut counts = CandidateCountMap::default();
        for gram_id in selected {
            for index in self.postings[gram_id as usize]
                .iter()
                .take(self.posting_limit)
            {
                *counts.entry(*index as usize).or_insert(0) += 1;
                if counts.len() >= self.max_candidates {
                    break;
                }
            }
            if counts.len() >= self.max_candidates {
                break;
            }
        }

        if counts.len() > self.rerank_limit {
            let mut ranked_counts: Vec<(usize, usize)> = counts.into_iter().collect();
            ranked_counts.sort_unstable_by(|left, right| {
                right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
            });
            ranked_counts.truncate(self.rerank_limit);
            return ranked_counts.into_iter().collect();
        }
        counts
    }

    fn rank_candidates(
        &self,
        query_count: usize,
        query_ids: &[GramId],
        intersection_counts: CandidateCountMap,
        k: usize,
    ) -> Vec<NeighborTuple> {
        let mut results: Vec<NeighborTuple> = Vec::with_capacity(intersection_counts.len());
        for index in intersection_counts.keys() {
            let candidate_grams = &self.gram_sets[*index];
            let intersection = intersection_size_ids(query_ids, candidate_grams);
            let union = query_count + candidate_grams.len() - intersection;
            let score = if union == 0 {
                1.0
            } else {
                intersection as f64 / union as f64
            };
            if score > 0.0 {
                results.push((Some(*index), score, false));
            }
        }
        results.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| {
                left.0
                    .unwrap_or(usize::MAX)
                    .cmp(&right.0.unwrap_or(usize::MAX))
            })
        });
        results.truncate(k);
        results
    }
}
