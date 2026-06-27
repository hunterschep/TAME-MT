use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type NeighborTuple = (Option<usize>, f64, bool);
type GramId = u32;
type DocId = u32;
type GramToIdMap = HashMap<Vec<u8>, GramId>;
type ExactMap = HashMap<String, usize>;
type CandidateCountMap = HashMap<usize, usize>;

#[pyclass]
#[derive(Serialize, Deserialize)]
struct NativeNgramIndex {
    gram_sets: Vec<Vec<GramId>>,
    postings: Vec<Vec<DocId>>,
    gram_to_id: GramToIdMap,
    exact_map: ExactMap,
    ngram_orders: Vec<usize>,
    mode: String,
    candidate_gram_limit: usize,
    posting_limit: usize,
    max_candidates: usize,
    rerank_limit: usize,
}

#[pymethods]
impl NativeNgramIndex {
    #[new]
    #[pyo3(signature = (
        normalized_lines,
        ngram_orders,
        mode,
        candidate_gram_limit,
        posting_limit,
        max_candidates,
        rerank_limit
    ))]
    fn new(
        normalized_lines: Vec<String>,
        ngram_orders: Vec<usize>,
        mode: String,
        candidate_gram_limit: usize,
        posting_limit: usize,
        max_candidates: usize,
        rerank_limit: usize,
    ) -> PyResult<Self> {
        if ngram_orders.is_empty() || ngram_orders.contains(&0) {
            return Err(PyValueError::new_err(
                "ngram_orders must contain positive integers",
            ));
        }
        if mode != "exact" && mode != "fast" {
            return Err(PyValueError::new_err("native mode must be exact or fast"));
        }
        if candidate_gram_limit == 0
            || posting_limit == 0
            || max_candidates == 0
            || rerank_limit == 0
            || rerank_limit > max_candidates
        {
            return Err(PyValueError::new_err(
                "native fast-mode limits must be positive and rerank <= max_candidates",
            ));
        }

        if normalized_lines.len() > DocId::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "native backend supports at most {} training segments",
                DocId::MAX
            )));
        }

        let mut gram_sets: Vec<Vec<GramId>> = Vec::with_capacity(normalized_lines.len());
        let mut postings: Vec<Vec<DocId>> = Vec::new();
        let mut gram_to_id = GramToIdMap::default();
        let mut exact_map = ExactMap::default();

        for (idx, line) in normalized_lines.iter().enumerate() {
            exact_map.entry(line.clone()).or_insert(idx);
            let grams = char_ngram_slices(line, &ngram_orders);
            let mut doc_grams: Vec<GramId> = Vec::with_capacity(grams.len());
            for gram in grams {
                let gram_id = if let Some(gram_id) = gram_to_id.get(gram) {
                    *gram_id
                } else {
                    let next_id = postings.len();
                    if next_id > GramId::MAX as usize {
                        return Err(PyValueError::new_err(format!(
                            "native backend supports at most {} unique n-grams",
                            GramId::MAX
                        )));
                    }
                    let gram_id = next_id as GramId;
                    gram_to_id.insert(gram.to_vec(), gram_id);
                    postings.push(Vec::new());
                    gram_id
                };
                doc_grams.push(gram_id);
            }
            doc_grams.sort_unstable();
            doc_grams.dedup();
            for gram_id in &doc_grams {
                postings[*gram_id as usize].push(idx as DocId);
            }
            gram_sets.push(doc_grams);
        }

        for indices in postings.iter_mut() {
            indices.sort_unstable();
        }

        Ok(Self {
            gram_sets,
            postings,
            gram_to_id,
            exact_map,
            ngram_orders,
            mode,
            candidate_gram_limit,
            posting_limit,
            max_candidates,
            rerank_limit,
        })
    }

    #[getter]
    fn backend_name(&self) -> String {
        format!("native_{}", self.mode)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let payload = self.serialize_index()?;
        Ok(PyBytes::new(py, &payload))
    }

    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let index: Self = bincode::deserialize(data)
            .map_err(|exc| PyValueError::new_err(format!("invalid native index bytes: {exc}")))?;
        index.validate_invariants()?;
        Ok(index)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let payload = self.serialize_index()?;
        std::fs::write(path, payload)
            .map_err(|exc| PyOSError::new_err(format!("failed to write native index: {exc}")))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let payload = std::fs::read(path)
            .map_err(|exc| PyOSError::new_err(format!("failed to read native index: {exc}")))?;
        Self::from_bytes(&payload)
    }

    fn doc_count(&self) -> usize {
        self.gram_sets.len()
    }

    fn gram_count(&self) -> usize {
        self.postings.len()
    }

    fn contains_exact(&self, query_norm: &str) -> bool {
        self.exact_map.contains_key(query_norm)
    }

    fn query_topk(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        self.query_topk_impl(query_norm, k)
    }

    fn batch_query_topk(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        k: usize,
    ) -> Vec<Vec<NeighborTuple>> {
        py.allow_threads(|| {
            queries
                .par_iter()
                .map(|query| self.query_topk_impl(query, k))
                .collect()
        })
    }

    fn score_candidate(&self, query_norm: &str, index: usize) -> PyResult<f64> {
        if index >= self.gram_sets.len() {
            return Err(PyIndexError::new_err(format!(
                "candidate index out of range: {index}"
            )));
        }
        let query_grams = char_ngram_slices(query_norm, &self.ngram_orders);
        let (query_count, query_ids) = self.query_gram_ids_from_grams(query_grams);
        Ok(jaccard_ids(query_count, &query_ids, &self.gram_sets[index]))
    }

    fn score_candidates(
        &self,
        query_norm: &str,
        indices: Vec<usize>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let query_grams = char_ngram_slices(query_norm, &self.ngram_orders);
        let (query_count, query_ids) = self.query_gram_ids_from_grams(query_grams);
        let mut scores = Vec::with_capacity(indices.len());
        for index in indices {
            if index >= self.gram_sets.len() {
                return Err(PyIndexError::new_err(format!(
                    "candidate index out of range: {index}"
                )));
            }
            scores.push((
                index,
                jaccard_ids(query_count, &query_ids, &self.gram_sets[index]),
            ));
        }
        Ok(scores)
    }

    fn best_pair_candidate(
        &self,
        target_index: PyRef<'_, NativeNgramIndex>,
        source_query_norm: &str,
        target_query_norms: Vec<String>,
        candidate_indices: Vec<usize>,
    ) -> PyResult<NeighborTuple> {
        self.best_pair_candidate_impl(
            &target_index,
            source_query_norm,
            &target_query_norms,
            &candidate_indices,
        )
    }

    fn batch_best_pair_candidates(
        &self,
        py: Python<'_>,
        target_index: PyRef<'_, NativeNgramIndex>,
        source_query_norms: Vec<String>,
        target_query_norms_by_segment: Vec<Vec<String>>,
        candidate_indices_by_segment: Vec<Vec<usize>>,
    ) -> PyResult<Vec<NeighborTuple>> {
        if source_query_norms.len() != target_query_norms_by_segment.len()
            || source_query_norms.len() != candidate_indices_by_segment.len()
        {
            return Err(PyValueError::new_err(
                "source queries, target queries, and candidate lists must have the same length",
            ));
        }
        let target_index_ref: &NativeNgramIndex = &target_index;
        py.allow_threads(|| {
            source_query_norms
                .par_iter()
                .zip(target_query_norms_by_segment.par_iter())
                .zip(candidate_indices_by_segment.par_iter())
                .map(
                    |((source_query_norm, target_query_norms), candidate_indices)| {
                        self.best_pair_candidate_impl(
                            target_index_ref,
                            source_query_norm,
                            target_query_norms,
                            candidate_indices,
                        )
                    },
                )
                .collect()
        })
    }
}

impl NativeNgramIndex {
    fn validate_invariants(&self) -> PyResult<()> {
        if self.ngram_orders.is_empty() || self.ngram_orders.contains(&0) {
            return Err(PyValueError::new_err(
                "invalid native index: ngram_orders must contain positive integers",
            ));
        }
        if self.mode != "exact" && self.mode != "fast" {
            return Err(PyValueError::new_err(
                "invalid native index: mode must be exact or fast",
            ));
        }
        if self.candidate_gram_limit == 0
            || self.posting_limit == 0
            || self.max_candidates == 0
            || self.rerank_limit == 0
            || self.rerank_limit > self.max_candidates
        {
            return Err(PyValueError::new_err(
                "invalid native index: fast-mode limits must be positive and rerank <= max_candidates",
            ));
        }
        if self.gram_sets.len() > DocId::MAX as usize {
            return Err(PyValueError::new_err(
                "invalid native index: document count exceeds native limit",
            ));
        }
        if self.postings.len() > GramId::MAX as usize {
            return Err(PyValueError::new_err(
                "invalid native index: n-gram count exceeds native limit",
            ));
        }
        if self.gram_to_id.len() != self.postings.len() {
            return Err(PyValueError::new_err(
                "invalid native index: gram map and posting table sizes differ",
            ));
        }
        if self.exact_map.len() > self.gram_sets.len() {
            return Err(PyValueError::new_err(
                "invalid native index: exact-match map is larger than document table",
            ));
        }

        let mut seen_gram_ids = vec![false; self.postings.len()];
        for (gram, gram_id) in &self.gram_to_id {
            if gram.is_empty() {
                return Err(PyValueError::new_err(
                    "invalid native index: gram map contains an empty n-gram",
                ));
            }
            let gram_index = *gram_id as usize;
            if gram_index >= self.postings.len() {
                return Err(PyValueError::new_err(
                    "invalid native index: gram id points past posting table",
                ));
            }
            if seen_gram_ids[gram_index] {
                return Err(PyValueError::new_err(
                    "invalid native index: duplicate gram id in gram map",
                ));
            }
            seen_gram_ids[gram_index] = true;
        }
        if seen_gram_ids.iter().any(|seen| !seen) {
            return Err(PyValueError::new_err(
                "invalid native index: posting table contains an unmapped gram id",
            ));
        }

        for index in self.exact_map.values() {
            if *index >= self.gram_sets.len() {
                return Err(PyValueError::new_err(
                    "invalid native index: exact-match entry points past document table",
                ));
            }
        }

        let mut doc_gram_counts = vec![0_usize; self.postings.len()];
        for doc_grams in &self.gram_sets {
            if !is_strictly_increasing_u32(doc_grams) {
                return Err(PyValueError::new_err(
                    "invalid native index: document gram ids must be sorted and unique",
                ));
            }
            for gram_id in doc_grams {
                let gram_index = *gram_id as usize;
                if gram_index >= self.postings.len() {
                    return Err(PyValueError::new_err(
                        "invalid native index: document gram id points past posting table",
                    ));
                }
                doc_gram_counts[gram_index] += 1;
            }
        }

        for (gram_index, posting) in self.postings.iter().enumerate() {
            if !is_strictly_increasing_u32(posting) {
                return Err(PyValueError::new_err(
                    "invalid native index: postings must be sorted and unique",
                ));
            }
            if posting.len() != doc_gram_counts[gram_index] {
                return Err(PyValueError::new_err(
                    "invalid native index: posting length does not match document gram count",
                ));
            }
            let gram_id = gram_index as GramId;
            for doc_id in posting {
                let doc_index = *doc_id as usize;
                if doc_index >= self.gram_sets.len() {
                    return Err(PyValueError::new_err(
                        "invalid native index: posting points past document table",
                    ));
                }
                if self.gram_sets[doc_index].binary_search(&gram_id).is_err() {
                    return Err(PyValueError::new_err(
                        "invalid native index: posting/document gram cross-reference is inconsistent",
                    ));
                }
            }
        }
        Ok(())
    }

    fn best_pair_candidate_impl(
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

    fn serialize_index(&self) -> PyResult<Vec<u8>> {
        bincode::serialize(self).map_err(|exc| {
            PyValueError::new_err(format!("failed to serialize native index: {exc}"))
        })
    }

    fn query_topk_impl(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
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

    fn query_gram_ids_from_grams(&self, grams: Vec<&[u8]>) -> (usize, Vec<GramId>) {
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

#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeNgramIndex>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    Ok(())
}

fn char_ngram_slices<'a>(text: &'a str, orders: &[usize]) -> Vec<&'a [u8]> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut offsets: Vec<usize> = text.char_indices().map(|(idx, _)| idx).collect();
    offsets.push(text.len());
    let char_count = offsets.len() - 1;
    let min_order = orders.iter().copied().min().unwrap_or(1);
    if char_count < min_order {
        return vec![text.as_bytes()];
    }

    let mut grams: Vec<&[u8]> = Vec::new();
    for order in orders {
        if *order <= char_count {
            for start in 0..=(char_count - *order) {
                let byte_start = offsets[start];
                let byte_end = offsets[start + *order];
                grams.push(&text.as_bytes()[byte_start..byte_end]);
            }
        }
    }

    grams.sort_unstable();
    grams.dedup();
    grams
}

fn jaccard_ids(query_count: usize, query_ids: &[GramId], candidate: &[GramId]) -> f64 {
    if query_count == 0 && candidate.is_empty() {
        return 1.0;
    }
    if query_count == 0 || candidate.is_empty() {
        return 0.0;
    }
    let intersection = intersection_size_ids(query_ids, candidate);
    let union = query_count + candidate.len() - intersection;
    intersection as f64 / union as f64
}

fn intersection_size_ids(left: &[GramId], right: &[GramId]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut count = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

fn is_strictly_increasing_u32(values: &[u32]) -> bool {
    values.windows(2).all(|items| items[0] < items[1])
}

#[cfg(test)]
mod tests {
    use super::{char_ngram_slices, jaccard_ids, NativeNgramIndex};

    fn valid_native_index() -> NativeNgramIndex {
        NativeNgramIndex::new(
            vec!["abcdef".to_string(), "abcxyz".to_string()],
            vec![3, 4, 5],
            "exact".to_string(),
            8,
            500,
            3_000,
            1_000,
        )
        .unwrap()
    }

    fn load_error(payload: &[u8]) -> pyo3::PyErr {
        pyo3::prepare_freethreaded_python();
        match NativeNgramIndex::from_bytes(payload) {
            Ok(_) => panic!("corrupt native index unexpectedly loaded"),
            Err(err) => err,
        }
    }

    #[test]
    fn char_ngrams_short_text_matches_python_semantics() {
        assert_eq!(char_ngram_slices("ab", &[3, 4, 5]), vec![b"ab".as_slice()]);
    }

    #[test]
    fn jaccard_handles_empty_sets() {
        let empty: Vec<u32> = Vec::new();
        assert_eq!(jaccard_ids(0, &empty, &empty), 1.0);
        assert_eq!(jaccard_ids(0, &empty, &[1]), 0.0);
    }

    #[test]
    fn native_index_from_bytes_accepts_valid_roundtrip() {
        let index = valid_native_index();
        let payload = bincode::serialize(&index).unwrap();
        let restored = NativeNgramIndex::from_bytes(&payload).unwrap();

        assert_eq!(restored.doc_count(), index.doc_count());
        assert_eq!(
            restored.query_topk_impl("abcdeg", 2),
            index.query_topk_impl("abcdeg", 2)
        );
    }

    #[test]
    fn native_index_from_bytes_rejects_out_of_range_gram_id() {
        let mut index = valid_native_index();
        let first_gram = index.gram_to_id.keys().next().unwrap().clone();
        *index.gram_to_id.get_mut(&first_gram).unwrap() = 99;
        let payload = bincode::serialize(&index).unwrap();

        let err = load_error(&payload);

        assert!(err
            .to_string()
            .contains("gram id points past posting table"));
    }

    #[test]
    fn native_index_from_bytes_rejects_unsorted_doc_grams() {
        let mut index = valid_native_index();
        if index.gram_sets[0].len() < 2 {
            panic!("test fixture must have at least two grams");
        }
        index.gram_sets[0].swap(0, 1);
        let payload = bincode::serialize(&index).unwrap();

        let err = load_error(&payload);

        assert!(err
            .to_string()
            .contains("document gram ids must be sorted and unique"));
    }

    #[test]
    fn native_index_from_bytes_rejects_posting_to_missing_document() {
        let mut index = valid_native_index();
        let posting_index = index
            .postings
            .iter()
            .position(|posting| posting.len() == 1)
            .unwrap();
        index.postings[posting_index][0] = 99;
        let payload = bincode::serialize(&index).unwrap();

        let err = load_error(&payload);

        assert!(err
            .to_string()
            .contains("posting points past document table"));
    }
}
