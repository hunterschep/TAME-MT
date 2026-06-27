pub(crate) fn char_ngram_slices<'a>(text: &'a str, orders: &[usize]) -> Vec<&'a [u8]> {
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
