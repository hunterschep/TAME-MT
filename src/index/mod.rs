mod exact;
mod fast;
mod pair;
mod query;
#[cfg(test)]
mod tests;
mod validation;
mod workspace;

use crate::fingerprint::exact_fingerprint;
use crate::index::workspace::QueryWorkspace;
use crate::ngrams::char_ngram_slices;
use crate::types::{DocId, ExactMap, GramId, GramToIdMap, NeighborTuple};
use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct NativeNgramIndex {
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
            exact_map.entry(exact_fingerprint(line)).or_insert(idx);
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
        let index: Self = rmp_serde::from_slice(data)
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
        self.exact_map.contains_key(&exact_fingerprint(query_norm))
    }

    fn query_topk(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        self.query_topk_impl(query_norm, k)
    }

    fn query_topk_exact(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        let mut workspace = QueryWorkspace::new(self.gram_sets.len());
        self.query_topk_exact_impl(query_norm, k, &mut workspace)
    }

    fn query_topk_fast(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        self.query_topk_fast_impl(query_norm, k)
    }

    fn batch_query_topk(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        k: usize,
    ) -> Vec<Vec<NeighborTuple>> {
        py.detach(|| {
            queries
                .par_iter()
                .map_init(
                    || QueryWorkspace::new(self.gram_sets.len()),
                    |workspace, query| self.query_topk_with_workspace(query, k, workspace),
                )
                .collect()
        })
    }

    fn batch_query_topk_exact(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        k: usize,
    ) -> Vec<Vec<NeighborTuple>> {
        py.detach(|| {
            queries
                .par_iter()
                .map_init(
                    || QueryWorkspace::new(self.gram_sets.len()),
                    |workspace, query| self.query_topk_exact_impl(query, k, workspace),
                )
                .collect()
        })
    }

    fn batch_query_topk_fast(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        k: usize,
    ) -> Vec<Vec<NeighborTuple>> {
        py.detach(|| {
            queries
                .par_iter()
                .map(|query| self.query_topk_fast_impl(query, k))
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
        Ok(crate::similarity::jaccard_ids(
            query_count,
            &query_ids,
            &self.gram_sets[index],
        ))
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
                crate::similarity::jaccard_ids(query_count, &query_ids, &self.gram_sets[index]),
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
        py.detach(|| {
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
    fn serialize_index(&self) -> PyResult<Vec<u8>> {
        rmp_serde::to_vec(self).map_err(|exc| {
            PyValueError::new_err(format!("failed to serialize native index: {exc}"))
        })
    }
}
