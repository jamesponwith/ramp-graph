//! Rendering graph objects to JSON, applying client input, and the depth/cost adapters.

use std::collections::HashMap;
use std::io::Write;

use ramp_graph::value::{self, Value};

use ramp_graph::{Direction, Entry, LogId, Projection, Record, StrId, Txn};
use serde_json::{Map, Number};

use crate::api::ApiError;

/// Kv domain holding seed payloads, in arrival order.
pub(crate) const SEEDS: &[u8] = b"lg.seeds";

/// Keys clients may not set on nodes (native fields, and `depth`, which the server owns).
const NODE_RESERVED: &[&str] = &[
    "ID",
    "type",
    "value",
    "typeID",
    "valueID",
    "edges",
    "edgeIDs",
    "edge_count",
    "neighbors",
    "neighborIDs",
    "neighbor_count",
    "neighbor_types",
    "outbound",
    "outboundIDs",
    "outbound_count",
    "inbound",
    "inboundIDs",
    "inbound_count",
    "depth",
];
const EDGE_RESERVED: &[&str] = &[
    "ID", "type", "value", "srcID", "tgtID", "src", "tgt", "typeID", "valueID",
];

/// Python-style truthiness, as upstream tests `seed` and role flags.
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// # Errors
/// The HTTP error to send instead.
fn text(t: &Txn<'_>, id: u64) -> Result<String, ApiError> {
    Ok(String::from_utf8_lossy(t.string(id)?).into_owned())
}

/// Properties of `parent` as a JSON object. Values that are not msgpack show as strings.
///
/// # Errors
/// The HTTP error to send instead.
pub(crate) fn props_dict(
    t: &Txn<'_>,
    parent: LogId,
    view: Option<LogId>,
) -> Result<Map<String, Value>, ApiError> {
    let mut out = Map::new();
    for p in t.props(parent, view)? {
        if let Record::Prop { key, val, .. } = p?.record {
            let bytes = t.string(val)?;
            let v = value::decode(bytes)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned()));
            out.insert(text(t, key)?, v);
        }
    }
    Ok(out)
}

/// Writes query results straight into a response buffer.
///
/// An object's native fields and properties go out as JSON without an intermediate
/// `Value` tree; properties come from the projection when there is one; strings that
/// repeat (keys, types) are fetched once per response.
pub(crate) struct Render<'t, 'p> {
    pub(crate) t: &'t Txn<'t>,
    pub(crate) proj: Option<&'p Projection>,
    strings: HashMap<StrId, String>,
    /// Property values as JSON, by value string ID; values repeat far more than objects.
    values: HashMap<StrId, Vec<u8>>,
}

/// Interned strings remembered per response before the cache is cleared.
const STRINGS: usize = 1 << 12;

impl<'t, 'p> Render<'t, 'p> {
    pub(crate) fn new(t: &'t Txn<'t>, proj: Option<&'p Projection>) -> Self {
        Self {
            t,
            proj,
            strings: HashMap::new(),
            values: HashMap::new(),
        }
    }

    /// A property value rendered as JSON. Values that are not msgpack show as strings.
    ///
    /// # Errors
    /// Storage errors.
    fn value_json(&mut self, id: StrId) -> Result<&[u8], ApiError> {
        if !self.values.contains_key(&id) {
            if self.values.len() >= STRINGS {
                self.values.clear();
            }
            let bytes = self.t.string(id)?;
            let json = match value::decode(bytes) {
                Ok(v) => serde_json::to_vec(&v),
                Err(_) => serde_json::to_vec(&String::from_utf8_lossy(bytes)),
            }
            .map_err(|e| ApiError::internal(e.to_string()))?;
            self.values.insert(id, json);
        }
        Ok(self.values.get(&id).map_or(&[], Vec::as_slice))
    }

    /// The text of a string that tends to repeat (a key or a type).
    ///
    /// # Errors
    /// Storage errors.
    fn cached(&mut self, id: StrId) -> Result<&str, ApiError> {
        if !self.strings.contains_key(&id) {
            // ponytail: flat cap, cleared when full; an LRU if a response has more hot strings
            if self.strings.len() >= STRINGS {
                self.strings.clear();
            }
            self.strings.insert(id, text(self.t, id)?);
        }
        Ok(self.strings.get(&id).map_or("", String::as_str))
    }

    /// Writes `e` as `as_dict` would render it, in `view`.
    ///
    /// # Errors
    /// Storage errors.
    pub(crate) fn json(
        &mut self,
        out: &mut Vec<u8>,
        e: &Entry,
        view: Option<LogId>,
    ) -> Result<(), ApiError> {
        let t = self.t;
        let s = |out: &mut Vec<u8>, v: &str| {
            serde_json::to_writer(out, v).map_err(|e| ApiError::internal(e.to_string()))
        };
        let n = |out: &mut Vec<u8>, v: u64| write!(out, "{v}").map_err(ApiError::io);
        out.extend_from_slice(b"{\"ID\":");
        n(out, e.id)?;
        match e.record {
            Record::Node { ty, val } | Record::Edge { ty, val, .. } => {
                out.extend_from_slice(b",\"type\":");
                s(out, self.cached(ty)?)?;
                out.extend_from_slice(b",\"value\":");
                s(out, &String::from_utf8_lossy(t.string(val)?))?;
                if let Record::Edge { src, tgt, .. } = e.record {
                    out.extend_from_slice(b",\"srcID\":");
                    n(out, src)?;
                    out.extend_from_slice(b",\"tgtID\":");
                    n(out, tgt)?;
                }
            }
            Record::Prop { .. } | Record::Deletion { .. } => {}
        }
        // The projection covers exactly the current view; anything else reads the index.
        let mut props: Vec<(StrId, StrId)> = Vec::new();
        match self.proj.filter(|_| view.is_none()) {
            Some(pj) => props.extend(pj.props_of(e.id).iter().map(|&(_, k, v)| (k, v))),
            None => {
                for p in t.props(e.id, view)? {
                    if let Record::Prop { key, val, .. } = p?.record {
                        props.push((key, val));
                    }
                }
            }
        }
        for (key, val) in props {
            out.push(b',');
            s(out, self.cached(key)?)?;
            out.push(b':');
            let json = self.value_json(val)?;
            out.extend_from_slice(json);
        }
        out.push(b'}');
        Ok(())
    }
}

/// Upstream `as_dict`: native fields, then properties.
///
/// # Errors
/// The HTTP error to send instead.
pub(crate) fn as_dict(t: &Txn<'_>, e: &Entry, view: Option<LogId>) -> Result<Value, ApiError> {
    let mut d = Map::new();
    d.insert("ID".to_owned(), e.id.into());
    match e.record {
        Record::Node { ty, val } => {
            d.insert("type".to_owned(), text(t, ty)?.into());
            d.insert("value".to_owned(), text(t, val)?.into());
        }
        Record::Edge { ty, val, src, tgt } => {
            d.insert("type".to_owned(), text(t, ty)?.into());
            d.insert("value".to_owned(), text(t, val)?.into());
            d.insert("srcID".to_owned(), src.into());
            d.insert("tgtID".to_owned(), tgt.into());
        }
        Record::Prop { .. } | Record::Deletion { .. } => {}
    }
    d.extend(props_dict(t, e.id, view)?);
    Ok(Value::Object(d))
}

/// Upstream `format_edge`: `as_dict` with `src`/`tgt` summaries instead of IDs.
///
/// # Errors
/// The HTTP error to send instead.
pub(crate) fn format_edge(t: &Txn<'_>, e: &Entry, view: Option<LogId>) -> Result<Value, ApiError> {
    let Value::Object(mut d) = as_dict(t, e, view)? else {
        return Err(ApiError::internal("dict is not an object"));
    };
    if let Record::Edge { src, tgt, .. } = e.record {
        for (key, id) in [("src", src), ("tgt", tgt)] {
            let mut n = Map::new();
            n.insert("ID".to_owned(), id.into());
            if let Some(Entry {
                record: Record::Node { ty, val },
                ..
            }) = t.entry(id)?
            {
                n.insert("type".to_owned(), text(t, ty)?.into());
                n.insert("value".to_owned(), text(t, val)?.into());
            }
            d.insert(key.to_owned(), Value::Object(n));
        }
        d.remove("srcID");
        d.remove("tgtID");
    }
    Ok(Value::Object(d))
}

/// Upstream `merge_values`: objects merge deeply; arrays become the sorted union when
/// their items are all numbers, all strings, or all booleans; anything else is replaced.
pub(crate) fn merge(old: Option<Value>, new: &Value) -> Value {
    match (old, new) {
        (Some(Value::Object(mut a)), Value::Object(b)) => {
            for (k, v) in b {
                let prev = a.remove(k);
                a.insert(k.clone(), merge(prev, v));
            }
            Value::Object(a)
        }
        (old, Value::Array(b)) => {
            let mut items = match old {
                Some(Value::Array(a)) => a,
                _ => Vec::new(),
            };
            items.extend(b.iter().cloned());
            let key = |v: &Value| match v {
                Value::Number(n) => Some((0, n.as_f64().unwrap_or(f64::NAN), String::new())),
                Value::String(s) => Some((1, 0.0, s.clone())),
                Value::Bool(b) => Some((2, f64::from(u8::from(*b)), String::new())),
                Value::Null | Value::Array(_) | Value::Object(_) => None,
            };
            let keys: Option<Vec<_>> = items.iter().map(key).collect();
            match keys {
                Some(keys) if keys.windows(2).all(|w| matches!(w, [a, b] if a.0 == b.0)) => {
                    let mut pairs: Vec<_> = keys.into_iter().zip(items).collect();
                    pairs.sort_by(|(a, _), (b, _)| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
                    pairs.dedup_by(|(a, _), (b, _)| a == b);
                    Value::Array(pairs.into_iter().map(|(_, v)| v).collect())
                }
                _ => new.clone(),
            }
        }
        (_, new) => new.clone(),
    }
}

/// Number as JSON, integral when it is one (depths are usually whole).
fn num(f: f64) -> Value {
    if f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0 {
        format!("{f:.0}").parse::<i64>().map_or_else(
            |_| Number::from_f64(f).map_or(Value::Null, Value::Number),
            Value::from,
        )
    } else {
        Number::from_f64(f).map_or(Value::Null, Value::Number)
    }
}

/// Applies a POST body (`seed`, `meta`, `chains`, `nodes`, `edges`) to a write txn.
pub(crate) struct Input<'t, 'a> {
    pub(crate) t: &'t mut Txn<'a>,
    pub(crate) create: bool,
    pub(crate) seed: bool,
}

impl Input<'_, '_> {
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn apply(t: &mut Txn<'_>, data: &Value, create: bool) -> Result<(), ApiError> {
        let Value::Object(obj) = data else {
            return Err(ApiError::new(400, "body must be an object"));
        };
        let seed = obj.get("seed").is_some_and(truthy);
        let mut input = Input { t, create, seed };
        if seed {
            input.t.fifo_push(SEEDS, &[&value::encode(data)?])?;
        }
        if let Some(meta) = obj.get("meta") {
            input.meta(meta)?;
        }
        if let Some(chains) = obj.get("chains") {
            for chain in as_array(chains, "chains")? {
                input.chain(as_array(chain, "chain")?)?;
            }
        }
        if let Some(nodes) = obj.get("nodes") {
            for n in as_array(nodes, "nodes")? {
                input.node(n)?;
            }
        }
        if let Some(edges) = obj.get("edges") {
            for e in as_array(edges, "edges")? {
                input.edge(e, None, None)?;
            }
        }
        Ok(())
    }

    /// Merges graph properties.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn meta(&mut self, meta: &Value) -> Result<(), ApiError> {
        let Value::Object(m) = meta else {
            return Err(ApiError::new(400, "meta must be an object"));
        };
        for (k, v) in m {
            set_merged(self.t, 0, k, v)?;
        }
        Ok(())
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn chain(&mut self, chain: &[Value]) -> Result<(), ApiError> {
        if chain.len().is_multiple_of(2) {
            return Err(ApiError::new(409, "Bad data - chain length must be odd"));
        }
        let nodes = chain
            .iter()
            .step_by(2)
            .map(|n| self.node(n))
            .collect::<Result<Vec<_>, _>>()?;
        for (edge, ends) in chain.iter().skip(1).step_by(2).zip(nodes.windows(2)) {
            if let [src, tgt] = ends {
                self.edge(edge, Some(*src), Some(*tgt))?;
            }
        }
        Ok(())
    }

    fn props<'v>(&self, obj: &'v Map<String, Value>, reserved: &[&str]) -> Vec<(&'v str, Value)> {
        let mut props: Vec<(&str, Value)> = obj
            .iter()
            .filter(|(k, _)| !reserved.contains(&k.as_str()))
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        if self.seed {
            props.retain(|(k, _)| *k != "seed");
            props.push(("seed", Value::Bool(true)));
        } else if !self.create {
            props.retain(|(k, _)| *k != "seed");
        }
        props
    }

    /// Finds (by `ID`) or resolves (by `type`/`value`) a node and merges its properties.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn node(&mut self, n: &Value) -> Result<Entry, ApiError> {
        let bad = || ApiError::new(409, format!("Bad node: {n}"));
        let Value::Object(obj) = n else {
            return Err(bad());
        };
        let entry = if let Some(id) = obj.get("ID").filter(|_| !self.create) {
            let id = id.as_u64().ok_or_else(bad)?;
            self.t
                .entry(id)?
                .filter(|e| e.next == 0 && matches!(e.record, Record::Node { .. }))
                .ok_or_else(bad)?
        } else {
            let (Some(ty), Some(val)) = (
                obj.get("type").and_then(scalar),
                obj.get("value").and_then(scalar),
            ) else {
                return Err(bad());
            };
            self.t.node(ty.as_bytes(), val.as_bytes())?
        };
        for (k, v) in self.props(obj, NODE_RESERVED) {
            set_merged(self.t, entry.id, k, &v)?;
        }
        Ok(entry)
    }

    /// Finds (by `ID`) or resolves (by `src`/`tgt`/`type`/`value`) an edge and merges its
    /// properties. `cost` must be in [0, 1] and may only decrease; other values are dropped.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn edge(
        &mut self,
        e: &Value,
        src: Option<Entry>,
        tgt: Option<Entry>,
    ) -> Result<Entry, ApiError> {
        let bad = || ApiError::new(409, format!("Bad edge: {e}"));
        let Value::Object(obj) = e else {
            return Err(bad());
        };
        let entry = if let Some(id) = obj.get("ID").filter(|_| !self.create) {
            let id = id.as_u64().ok_or_else(bad)?;
            self.t
                .entry(id)?
                .filter(|e| e.next == 0 && matches!(e.record, Record::Edge { .. }))
                .ok_or_else(bad)?
        } else {
            let mut end = |given: Option<Entry>, key| match given {
                Some(n) => Ok(n),
                None => self.node(obj.get(key).ok_or_else(bad)?),
            };
            let (src, tgt) = (end(src, "src")?, end(tgt, "tgt")?);
            let ty = obj.get("type").and_then(scalar).ok_or_else(bad)?;
            let val = obj
                .get("value")
                .map_or(Some(String::new()), scalar)
                .ok_or_else(bad)?;
            self.t
                .edge(src.id, tgt.id, ty.as_bytes(), val.as_bytes())
                .map_err(|_| bad())?
        };
        let mut props = self.props(obj, EDGE_RESERVED);
        let current = self
            .t
            .value(entry.id, "cost", None)
            .ok()
            .flatten()
            .and_then(|v| v.as_f64());
        props.retain(|(k, v)| {
            *k != "cost"
                || v.as_f64()
                    .is_some_and(|c| (0.0..=1.0).contains(&c) && current.is_none_or(|cur| c < cur))
        });
        for (k, v) in props {
            set_merged(self.t, entry.id, k, &v)?;
        }
        Ok(entry)
    }
}

/// # Errors
/// The HTTP error to send instead.
fn as_array<'v>(v: &'v Value, what: &str) -> Result<&'v [Value], ApiError> {
    v.as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| ApiError::new(409, format!("Bad data - {what} must be a list")))
}

/// Node/edge type or value as upstream's default serializer stores it.
fn scalar(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(true) => Some("True".to_owned()),
        Value::Bool(false) => Some("False".to_owned()),
        Value::Null => Some(String::new()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

/// # Errors
/// The HTTP error to send instead.
fn set_merged(t: &mut Txn<'_>, parent: LogId, key: &str, v: &Value) -> Result<(), ApiError> {
    let old = t.value(parent, key, None).ok().flatten();
    Ok(t.set_value(parent, key, &merge(old, v))?)
}

fn depth(t: &Txn<'_>, node: LogId) -> f64 {
    t.value(node, "depth", None)
        .ok()
        .flatten()
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::INFINITY)
}

/// # Errors
/// The HTTP error to send instead.
fn live(t: &Txn<'_>, id: LogId) -> Result<Option<Record>, ApiError> {
    Ok(t.entry(id)?.filter(|e| e.next == 0).map(|e| e.record))
}

/// Upstream's server adapters, run over everything logged since `start` (including what
/// they log themselves, so depth changes cascade):
/// - a truthy `seed` property on a node sets its `depth` to 0;
/// - a new edge, or an edge `cost`, pulls one endpoint's depth to the other's + cost (1 by default);
/// - a node `depth` caps each neighbor's depth at depth + 1.
///
/// # Errors
/// The HTTP error to send instead.
pub(crate) fn run_adapters(t: &mut Txn<'_>, start: LogId) -> Result<(), ApiError> {
    let mut x = start;
    while x < t.next_id()? {
        let Some(e) = t.entry(x)? else { break };
        x += 1;
        match e.record {
            Record::Edge { src, tgt, .. } => relax(t, src, tgt, 1.0)?,
            Record::Prop { parent, key, val } => {
                let key = match t.string(key)? {
                    k @ (b"seed" | b"depth" | b"cost") => k.to_vec(),
                    _ => continue,
                };
                let v = value::decode(t.string(val)?).unwrap_or(Value::Null);
                match (key.as_slice(), live(t, parent)?) {
                    (b"seed", Some(Record::Node { .. })) if truthy(&v) => {
                        t.set_value(parent, "depth", &Value::from(0))?;
                    }
                    (b"depth", Some(Record::Node { .. })) => {
                        if let Some(d) = v.as_f64() {
                            cascade(t, parent, d + 1.0)?;
                        }
                    }
                    (b"cost", Some(Record::Edge { src, tgt, .. })) => {
                        if let Some(c) = v.as_f64() {
                            relax(t, src, tgt, c)?;
                        }
                    }
                    _ => {}
                }
            }
            Record::Node { .. } | Record::Deletion { .. } => {}
        }
    }
    Ok(())
}

/// # Errors
/// The HTTP error to send instead.
fn relax(t: &mut Txn<'_>, src: LogId, tgt: LogId, cost: f64) -> Result<(), ApiError> {
    if live(t, src)?.is_none() || live(t, tgt)?.is_none() {
        return Ok(());
    }
    let (ds, dt) = (depth(t, src), depth(t, tgt));
    if ds + cost < dt {
        t.set_value(tgt, "depth", &num(ds + cost))?;
    } else if dt + cost < ds {
        t.set_value(src, "depth", &num(dt + cost))?;
    }
    Ok(())
}

/// # Errors
/// The HTTP error to send instead.
fn cascade(t: &mut Txn<'_>, node: LogId, d: f64) -> Result<(), ApiError> {
    let mut neighbors = Vec::new();
    for e in t.node_edges(node, Direction::Both, None, None)? {
        if let Record::Edge { src, tgt, .. } = e?.record {
            neighbors.extend([src, tgt].into_iter().filter(|&n| n != node));
        }
    }
    neighbors.sort_unstable();
    neighbors.dedup();
    for n in neighbors {
        if depth(t, n) > d {
            t.set_value(n, "depth", &num(d))?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn merge_values() {
        let m = |a: Value, b: Value| merge(Some(a), &b);
        assert_eq!(
            m(json!({"a": 1, "b": {"c": 1}}), json!({"b": {"d": 2}})),
            json!({"a": 1, "b": {"c": 1, "d": 2}})
        );
        assert_eq!(m(json!([3, 1]), json!([2, 1])), json!([1, 2, 3]));
        assert_eq!(m(json!(["b"]), json!(["a", "b"])), json!(["a", "b"]));
        assert_eq!(
            m(json!([1]), json!(["a"])),
            json!(["a"]),
            "mixed types overwrite"
        );
        assert_eq!(m(json!(5), json!([2])), json!([2]));
        assert_eq!(m(json!({"a": 1}), json!(2)), json!(2));
        assert_eq!(merge(None, &json!([2, 2])), json!([2]));
    }

    #[test]
    fn render_matches_as_dict() {
        // Nodes, edges, msgpack and raw property values, nested properties, a historical
        // view, with and without a projection: the same object `as_dict` builds. Keys
        // come out in upstream's order (natives, then properties by key ID) rather than
        // sorted, so compare parsed.
        let dir = tempfile::tempdir().unwrap();
        let graph = ramp_graph::Graph::open(dir.path().join("g.db")).unwrap();
        let mut txn = graph.write().unwrap();
        let na = txn.node(b"t", b"a \"quoted\"").unwrap().id;
        let nb = txn.node(b"t", b"b").unwrap().id;
        txn.set_value(na, "k", &json!({"n": [1, 2.5, "x"], "b": true}))
            .unwrap();
        txn.set_value(na, "zz", &json!(null)).unwrap();
        let raw = txn.set(na, b"raw", b"not msgpack \xff").unwrap().id;
        txn.set_value(raw, "deep", &json!("d")).unwrap();
        let edge = txn.edge(na, nb, b"e", b"v").unwrap().id;
        txn.set_value(edge, "w", &json!(7)).unwrap();
        txn.commit().unwrap();
        let mut txn = graph.write().unwrap();
        txn.set_value(nb, "later", &json!("yes")).unwrap();
        txn.commit().unwrap();
        let txn = graph.read().unwrap();
        let mid = txn.next_id().unwrap() - 1;
        let check = |render: &mut Render<'_, '_>, id: LogId, view: Option<LogId>| {
            let entry = txn.entry(id).unwrap().unwrap();
            let mut out = Vec::new();
            render.json(&mut out, &entry, view).unwrap();
            let got: Value = serde_json::from_slice(&out).unwrap();
            assert_eq!(
                got,
                as_dict(&txn, &entry, view).unwrap(),
                "{id} at {view:?}"
            );
            assert!(out.starts_with(b"{\"ID\":"), "natives first");
        };
        let mut render = Render::new(&txn, None);
        for id in [na, nb, edge, raw] {
            check(&mut render, id, None);
            check(&mut render, id, Some(mid));
        }
        let proj = txn.projection().unwrap();
        let mut render = Render::new(&txn, Some(&proj));
        for id in [na, nb, edge, raw] {
            check(&mut render, id, None);
            check(&mut render, id, Some(mid));
        }
    }

    #[test]
    fn numbers() {
        assert_eq!(num(2.0), json!(2));
        assert_eq!(num(1.5), json!(1.5));
        assert_eq!(num(f64::INFINITY), Value::Null);
    }
}
