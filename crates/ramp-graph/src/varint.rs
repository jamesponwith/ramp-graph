//! Order-preserving variable-length integer encoding.
//!
//! Each value is a length byte (0..=8) followed by that many big-endian bytes with
//! leading zeros dropped. Shorter encodings always hold smaller numbers, so a
//! byte-wise comparison of encoded tuples matches a numeric comparison of the tuples:
//! the btree's natural key order is the order we want.

use crate::{GraphError as Error, Result};

/// Longest encoding of one `u64`.
pub(crate) const MAX_LEN: usize = 9;

/// Appends the encoding of `x` to `buf`.
pub(crate) fn push(buf: &mut Vec<u8>, x: u64) {
    let bytes = x.to_be_bytes();
    let digits = bytes
        .get(bytes.iter().take_while(|b| **b == 0).count()..)
        .unwrap_or_default();
    buf.push(u8::try_from(digits.len()).unwrap_or(8));
    buf.extend_from_slice(digits);
}

/// Encodes a tuple of integers.
pub(crate) fn pack(xs: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(xs.len() * MAX_LEN);
    for &x in xs {
        push(&mut buf, x);
    }
    buf
}

/// Decodes one integer from the front of `buf`, advancing it.
///
/// # Errors
/// Fails on storage errors or corrupt data.
pub(crate) fn take(buf: &mut &[u8]) -> Result<u64> {
    let (&len, rest) = buf
        .split_first()
        .ok_or(Error::Corrupt("truncated varint"))?;
    let (digits, rest) = rest
        .split_at_checked(usize::from(len))
        .filter(|(d, _)| d.len() <= 8)
        .ok_or(Error::Corrupt("bad varint length"))?;
    *buf = rest;
    Ok(digits.iter().fold(0, |acc, &d| (acc << 8) | u64::from(d)))
}

/// Decodes the last integer of an encoded tuple (index keys end in a log ID).
///
/// # Errors
/// Fails on storage errors or corrupt data.
pub(crate) fn last(mut buf: &[u8]) -> Result<u64> {
    let mut x = take(&mut buf)?;
    while !buf.is_empty() {
        x = take(&mut buf)?;
    }
    Ok(x)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn known_encodings() {
        assert_eq!(pack(&[0]), [0], "zero is a bare length byte");
        assert_eq!(pack(&[1]), [1, 1], "one byte");
        assert_eq!(pack(&[256]), [2, 1, 0], "two bytes");
        assert_eq!(
            pack(&[u64::MAX]),
            [8, 255, 255, 255, 255, 255, 255, 255, 255],
            "max"
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(take(&mut &[][..]).is_err(), "empty");
        assert!(take(&mut &[2, 1][..]).is_err(), "truncated");
        assert!(
            take(&mut &[9, 0, 0, 0, 0, 0, 0, 0, 0, 0][..]).is_err(),
            "too long"
        );
    }

    proptest! {
        #[test]
        fn roundtrip(xs in prop::collection::vec(any::<u64>(), 1..6)) {
            let enc = pack(&xs);
            let mut buf = enc.as_slice();
            for &x in &xs {
                prop_assert_eq!(take(&mut buf)?, x);
            }
            prop_assert!(buf.is_empty());
            prop_assert_eq!(last(&enc)?, *xs.last().unwrap_or(&0));
        }

        #[test]
        fn order_preserving(a in prop::collection::vec(any::<u64>(), 3), b in prop::collection::vec(any::<u64>(), 3)) {
            prop_assert_eq!(pack(&a).cmp(&pack(&b)), a.cmp(&b));
        }
    }
}
