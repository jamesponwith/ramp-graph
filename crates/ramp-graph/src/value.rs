//! Typed property values: JSON-shaped data stored as msgpack (as upstream's server does).

use crate::{GraphError, LogId, Result, Txn};

pub use serde_json::Value;

/// Encodes a value for storage.
///
/// # Errors
/// Never in practice; msgpack can represent every JSON value.
pub fn encode(v: &Value) -> Result<Vec<u8>> {
    rmp_serde::to_vec(v).map_err(|e| GraphError::Value(e.to_string()))
}

/// Decodes a stored value.
///
/// # Errors
/// Fails if `bytes` is not exactly one msgpack value or has non-string map keys.
pub fn decode(bytes: &[u8]) -> Result<Value> {
    let mut rest = bytes;
    let v = rmp_serde::from_read(&mut rest).map_err(|e| GraphError::Value(e.to_string()))?;
    // `from_slice` ignores trailing bytes, which would read raw strings as numbers.
    if rest.is_empty() {
        Ok(v)
    } else {
        Err(GraphError::Value(
            "trailing bytes after msgpack value".to_owned(),
        ))
    }
}

impl Txn<'_> {
    /// Sets property `key` of `parent` to a typed value.
    ///
    /// # Errors
    /// As [`Txn::set`].
    pub fn set_value(&mut self, parent: LogId, key: &str, v: &Value) -> Result<()> {
        self.set(parent, key.as_bytes(), &encode(v)?).map(drop)
    }

    /// Typed value of property `key` of `parent` in view `before`.
    ///
    /// # Errors
    /// Fails on storage errors or a value that is not msgpack.
    pub fn value(&self, parent: LogId, key: &str, before: Option<LogId>) -> Result<Option<Value>> {
        let Some(crate::Entry {
            record: crate::Record::Prop { val, .. },
            ..
        }) = self.prop(parent, key.as_bytes(), before)?
        else {
            return Ok(None);
        };
        decode(self.string(val)?).map(Some)
    }
}
