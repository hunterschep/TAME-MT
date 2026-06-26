use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

type NeighborTuple = (Option<usize>, f64, bool);

#[pyclass]
struct NativeNgramIndex {
    normalized_lines: Vec<String>,
    gram_sets: Vec<Vec<String>>,
    gram_counts: Vec<usize>,
    postings: HashMap<String, Vec<usize>>,
    exact_map: HashMap<String, usize>,
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

        let mut gram_sets: Vec<Vec<String>> = Vec::with_capacity(normalized_lines.len());
        let mut gram_counts: Vec<usize> = Vec::with_capacity(normalized_lines.len());
        let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
        let mut exact_map: HashMap<String, usize> = HashMap::new();

        for (idx, line) in normalized_lines.iter().enumerate() {
            exact_map.entry(line.clone()).or_insert(idx);
            let grams = char_ngrams(line, &ngram_orders);
            gram_counts.push(grams.len());
            for gram in &grams {
                postings.entry(gram.clone()).or_default().push(idx);
            }
            gram_sets.push(grams);
        }

        for indices in postings.values_mut() {
            indices.sort_unstable();
        }

        Ok(Self {
            normalized_lines,
            gram_sets,
            gram_counts,
            postings,
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
        let query_grams = char_ngrams(query_norm, &self.ngram_orders);
        Ok(jaccard_sorted(&query_grams, &self.gram_sets[index]))
    }
}

impl NativeNgramIndex {
    fn query_topk_impl(&self, query_norm: &str, k: usize) -> Vec<NeighborTuple> {
        if k == 0 {
            return Vec::new();
        }

        let query_grams = char_ngrams(query_norm, &self.ngram_orders);
        if query_grams.is_empty() {
            if let Some(index) = self.exact_map.get(query_norm) {
                return vec![(Some(*index), 1.0, true)];
            }
            return Vec::new();
        }

        if let Some(index) = self.exact_map.get(query_norm) {
            return vec![(Some(*index), 1.0, true)];
        }

        let intersection_counts = if self.mode == "fast" {
            self.candidate_counts_fast(&query_grams)
        } else {
            self.candidate_counts_exact(&query_grams)
        };
        self.rank_candidates(query_norm, &query_grams, intersection_counts, k)
    }

    fn candidate_counts_exact(&self, query_grams: &[String]) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for gram in query_grams {
            if let Some(indices) = self.postings.get(gram) {
                for index in indices {
                    *counts.entry(*index).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    fn candidate_counts_fast(&self, query_grams: &[String]) -> HashMap<usize, usize> {
        let mut ranked_grams: Vec<(usize, &String)> = query_grams
            .iter()
            .map(|gram| {
                let posting_count = self.postings.get(gram).map_or(0, Vec::len);
                (posting_count, gram)
            })
            .collect();
        ranked_grams.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(right.1)));

        let mut selected: Vec<&String> = ranked_grams
            .iter()
            .filter_map(|(posting_count, gram)| {
                if *posting_count > 0 && *posting_count <= self.posting_limit {
                    Some(*gram)
                } else {
                    None
                }
            })
            .take(self.candidate_gram_limit)
            .collect();

        if selected.is_empty() {
            selected = ranked_grams
                .iter()
                .filter_map(|(posting_count, gram)| {
                    if *posting_count > 0 {
                        Some(*gram)
                    } else {
                        None
                    }
                })
                .take(self.candidate_gram_limit)
                .collect();
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for gram in selected {
            if let Some(indices) = self.postings.get(gram) {
                for index in indices.iter().take(self.posting_limit) {
                    *counts.entry(*index).or_insert(0) += 1;
                    if counts.len() >= self.max_candidates {
                        break;
                    }
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
        query_norm: &str,
        query_grams: &[String],
        intersection_counts: HashMap<usize, usize>,
        k: usize,
    ) -> Vec<NeighborTuple> {
        let mut results: Vec<NeighborTuple> = Vec::with_capacity(intersection_counts.len());
        for index in intersection_counts.keys() {
            let candidate_grams = &self.gram_sets[*index];
            let intersection = intersection_size_sorted(query_grams, candidate_grams);
            let union = query_grams.len() + self.gram_counts[*index] - intersection;
            let score = if union == 0 {
                1.0
            } else {
                intersection as f64 / union as f64
            };
            if score > 0.0 {
                results.push((
                    Some(*index),
                    score,
                    self.normalized_lines[*index] == query_norm,
                ));
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

fn char_ngrams(text: &str, orders: &[usize]) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let min_order = orders.iter().copied().min().unwrap_or(1);
    if chars.len() < min_order {
        return vec![text.to_string()];
    }

    let mut grams: HashSet<String> = HashSet::new();
    for order in orders {
        if *order <= chars.len() {
            for start in 0..=(chars.len() - *order) {
                grams.insert(chars[start..start + *order].iter().collect());
            }
        }
    }

    let mut sorted: Vec<String> = grams.into_iter().collect();
    sorted.sort_unstable();
    sorted
}

fn jaccard_sorted(left: &[String], right: &[String]) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = intersection_size_sorted(left, right);
    let union = left.len() + right.len() - intersection;
    intersection as f64 / union as f64
}

fn intersection_size_sorted(left: &[String], right: &[String]) -> usize {
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

#[cfg(test)]
mod tests {
    use super::{char_ngrams, jaccard_sorted};

    #[test]
    fn char_ngrams_short_text_matches_python_semantics() {
        assert_eq!(char_ngrams("ab", &[3, 4, 5]), vec!["ab"]);
    }

    #[test]
    fn jaccard_handles_empty_sets() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(jaccard_sorted(&empty, &empty), 1.0);
        assert_eq!(jaccard_sorted(&empty, &char_ngrams("abc", &[3])), 0.0);
    }
}
