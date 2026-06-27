use std::collections::HashMap;

pub(crate) type NeighborTuple = (Option<usize>, f64, bool);
pub(crate) type GramId = u32;
pub(crate) type DocId = u32;
pub(crate) type GramToIdMap = HashMap<Vec<u8>, GramId>;
pub(crate) type ExactMap = HashMap<String, usize>;
pub(crate) type CandidateCountMap = HashMap<usize, usize>;
