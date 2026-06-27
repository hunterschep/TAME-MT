use super::workspace::QueryWorkspace;
use super::NativeNgramIndex;
use crate::ngrams::char_ngram_slices;
use crate::similarity::jaccard_ids;

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
    pyo3::Python::initialize();
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
    let payload = rmp_serde::to_vec(&index).unwrap();
    let restored = NativeNgramIndex::from_bytes(&payload).unwrap();

    assert_eq!(restored.doc_count(), index.doc_count());
    assert_eq!(
        restored.query_topk_impl("abcdeg", 2),
        index.query_topk_impl("abcdeg", 2)
    );
}

#[test]
fn exact_query_workspace_can_be_reused_across_queries() {
    let index = valid_native_index();
    let mut workspace = QueryWorkspace::new(index.doc_count());
    let queries = ["abcdeg", "abcxyy", "nomatch", "abcdef", "abcdeg"];

    for query in queries {
        assert_eq!(
            index.query_topk_exact_impl(query, 2, &mut workspace),
            index.query_topk_impl(query, 2)
        );
    }
}

#[test]
fn native_index_from_bytes_rejects_out_of_range_gram_id() {
    let mut index = valid_native_index();
    let first_gram = index.gram_to_id.keys().next().unwrap().clone();
    *index.gram_to_id.get_mut(&first_gram).unwrap() = 99;
    let payload = rmp_serde::to_vec(&index).unwrap();

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
    let payload = rmp_serde::to_vec(&index).unwrap();

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
    let payload = rmp_serde::to_vec(&index).unwrap();

    let err = load_error(&payload);

    assert!(err
        .to_string()
        .contains("posting points past document table"));
}
