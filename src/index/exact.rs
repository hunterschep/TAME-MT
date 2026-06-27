use super::NativeNgramIndex;
use crate::index::workspace::QueryWorkspace;
use crate::types::{CandidateCountMap, GramId, NeighborTuple};

impl NativeNgramIndex {
    /// Rank exact Jaccard candidates using all query posting lists.
    ///
    /// This is the canonical no-false-negative top-k path. It reuses
    /// vector-backed candidate counts from `QueryWorkspace`, visits query grams
    /// in increasing document-frequency order, and applies a length-only upper
    /// bound once the current kth score is known. Ties remain deterministic:
    /// higher score first, then lower training index.
    pub(crate) fn rank_exact_candidates(
        &self,
        query_count: usize,
        query_ids: &[GramId],
        k: usize,
        workspace: &mut QueryWorkspace,
    ) -> Vec<NeighborTuple> {
        if query_ids.len() > u16::MAX as usize {
            return self.rank_large_query_candidates(query_count, query_ids, k);
        }

        workspace.reset_counts();
        workspace.ordered_query_ids.clear();
        workspace.ordered_query_ids.extend_from_slice(query_ids);
        workspace.ordered_query_ids.sort_unstable_by(|left, right| {
            self.postings[*left as usize]
                .len()
                .cmp(&self.postings[*right as usize].len())
                .then_with(|| left.cmp(right))
        });

        for gram_id in workspace.ordered_query_ids.iter().copied() {
            for index in &self.postings[gram_id as usize] {
                let index = *index as usize;
                if workspace.counts[index] == 0 {
                    workspace.touched.push(index);
                }
                workspace.counts[index] += 1;
            }
        }

        let mut results: Vec<NeighborTuple> = Vec::new();
        let mut kth_score = 0.0_f64;
        for index in workspace.touched.iter().copied() {
            let candidate_len = self.gram_sets[index].len();
            if results.len() >= k
                && jaccard_length_upper_bound(query_count, candidate_len) < kth_score
            {
                continue;
            }

            let intersection = usize::from(workspace.counts[index]);
            let union = query_count + candidate_len - intersection;
            let score = if union == 0 {
                1.0
            } else {
                intersection as f64 / union as f64
            };
            if score > 0.0 {
                results.push((Some(index), score, false));
                if results.len() > k.saturating_mul(2).max(k + 1) {
                    sort_and_truncate(&mut results, k);
                    kth_score = results.last().map(|item| item.1).unwrap_or(0.0);
                }
            }
        }

        sort_and_truncate(&mut results, k);
        workspace.reset_counts();
        results
    }

    fn rank_large_query_candidates(
        &self,
        query_count: usize,
        query_ids: &[GramId],
        k: usize,
    ) -> Vec<NeighborTuple> {
        let mut counts = CandidateCountMap::default();
        for gram_id in query_ids {
            for index in &self.postings[*gram_id as usize] {
                *counts.entry(*index as usize).or_insert(0) += 1;
            }
        }

        let mut results: Vec<NeighborTuple> = Vec::new();
        let mut kth_score = 0.0_f64;
        for (index, intersection) in counts {
            let candidate_len = self.gram_sets[index].len();
            if results.len() >= k
                && jaccard_length_upper_bound(query_count, candidate_len) < kth_score
            {
                continue;
            }

            let union = query_count + candidate_len - intersection;
            let score = if union == 0 {
                1.0
            } else {
                intersection as f64 / union as f64
            };
            if score > 0.0 {
                results.push((Some(index), score, false));
                if results.len() > k.saturating_mul(2).max(k + 1) {
                    sort_and_truncate(&mut results, k);
                    kth_score = results.last().map(|item| item.1).unwrap_or(0.0);
                }
            }
        }

        sort_and_truncate(&mut results, k);
        results
    }
}

fn sort_and_truncate(results: &mut Vec<NeighborTuple>, k: usize) {
    results.sort_unstable_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| {
            left.0
                .unwrap_or(usize::MAX)
                .cmp(&right.0.unwrap_or(usize::MAX))
        })
    });
    results.truncate(k);
}

fn jaccard_length_upper_bound(left_len: usize, right_len: usize) -> f64 {
    if left_len == 0 && right_len == 0 {
        return 1.0;
    }
    if left_len == 0 || right_len == 0 {
        return 0.0;
    }
    let shorter = left_len.min(right_len);
    let longer = left_len.max(right_len);
    shorter as f64 / longer as f64
}
