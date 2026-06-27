use crate::types::GramId;

pub(crate) fn jaccard_ids(query_count: usize, query_ids: &[GramId], candidate: &[GramId]) -> f64 {
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

pub(crate) fn intersection_size_ids(left: &[GramId], right: &[GramId]) -> usize {
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
