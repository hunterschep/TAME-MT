pub(crate) fn is_strictly_increasing_u32(values: &[u32]) -> bool {
    values.windows(2).all(|items| items[0] < items[1])
}
