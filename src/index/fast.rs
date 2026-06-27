use super::NativeNgramIndex;
use crate::similarity::intersection_size_ids;
use crate::types::{CandidateCountMap, GramId, NeighborTuple};

impl NativeNgramIndex {
    /// Collect bounded rare-gram candidates for approximate retrieval.
    ///
    /// This deliberately does not power canonical exposure metrics. It is a
    /// throughput path for explicitly approximate runs: select low-frequency
    /// postings under configured caps, then let `rank_fast_candidates` rerank
    /// that bounded set exactly.
    pub(crate) fn candidate_counts_fast(&self, query_ids: &[GramId]) -> CandidateCountMap {
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

    /// Rerank the fast candidate set with exact Jaccard inside that set.
    ///
    /// The final scores are exact for the retained candidates, but recall is
    /// bounded by `candidate_counts_fast`, so callers must label this mode as
    /// approximate and validate it for paper-critical runs.
    pub(crate) fn rank_fast_candidates(
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
