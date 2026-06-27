use super::NativeNgramIndex;
use crate::types::{DocId, GramId};
use crate::validation::is_strictly_increasing_u32;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

impl NativeNgramIndex {
    pub(crate) fn validate_invariants(&self) -> PyResult<()> {
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
}
