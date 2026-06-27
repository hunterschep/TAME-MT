use crate::types::GramId;

pub(crate) struct QueryWorkspace {
    pub(crate) counts: Vec<u16>,
    pub(crate) touched: Vec<usize>,
    pub(crate) ordered_query_ids: Vec<GramId>,
}

impl QueryWorkspace {
    pub(crate) fn new(doc_count: usize) -> Self {
        Self {
            counts: vec![0; doc_count],
            touched: Vec::new(),
            ordered_query_ids: Vec::new(),
        }
    }

    pub(crate) fn reset_counts(&mut self) {
        for index in self.touched.drain(..) {
            self.counts[index] = 0;
        }
    }
}
