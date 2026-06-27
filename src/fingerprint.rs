pub(crate) type ExactFingerprint = [u8; 16];

pub(crate) fn exact_fingerprint(text: &str) -> ExactFingerprint {
    let digest = blake3::hash(text.as_bytes());
    let mut key = [0_u8; 16];
    key.copy_from_slice(&digest.as_bytes()[..16]);
    key
}
