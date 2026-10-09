//! LGQL execution: ad-hoc queries against a view, streaming queries over a log range.
//!
//! A filter key resolves to a native field first, then to a property (msgpack value,
//! see [`crate::value`]), then walks nested object keys. Native fields:
//! - both: `ID`, `type`, `value`, `typeID`, `valueID`
//! - nodes: `edge_count`, `inbound_count`, `outbound_count`, `neighbor_count`;
//!   ID lists `edges`/`edgeIDs`, `inbound`/`inboundIDs`, `outbound`/`outboundIDs`,
//!   `neighbors`/`neighborIDs` (upstream yields objects for the former, which LGQL can
//!   only test for existence; IDs keep that and make `:array`/list tests useful);
//!   `neighbor_types` (`{type: count}`)
//! - edges: `srcID`, `tgtID`, `src`, `tgt` (the latter two continue into the node: `src.type`)
//!
//! A key that does not resolve fails every test, including the negated ones.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde_json::{Number, Value};

use crate::lgql::{Cmp, Filter, Kind, Pattern, Slot, Test};
use crate::{Direction, Entry, GraphError, LogId, Record, Result, StrId, Txn, value, varint};

const fn kind(e: &Entry) -> Option<Kind> {
    match e.record {
        Record::Node { .. } => Some(Kind::Node),
        Record::Edge { .. } => Some(Kind::Edge),
        Record::Prop { .. } | Record::Deletion { .. } => None,
    }
}

/// String IDs of every key-path segment and equality literal in a query's filters,
/// looked up once per call instead of once per object tested (strings cannot change
/// while a query runs).
#[derive(Debug, Default)]
pub(crate) struct Keys {
    keys: HashMap<String, Option<StrId>>,
    /// Per key of a single-key `=`/`!=` test, the ID of each string literal: of its
    /// raw bytes for `type`/`value`, of its msgpack encoding for a property. `None`
    /// means no object holds that string.
    literals: HashMap<String, HashMap<String, Option<StrId>>>,
}

impl Keys {
    /// # Errors
    /// Fails on storage errors.
    fn new(t: &Txn<'_>, patterns: &[Pattern]) -> Result<Self> {
        let mut k = Self::default();
        for f in patterns
            .iter()
            .flat_map(|p| &p.slots)
            .flat_map(|s| &s.filters)
        {
            for seg in &f.path {
                // `type`/`value` are native on every object a slot can hold.
                if !k.keys.contains_key(seg) && !NATIVE_STR.contains(&seg.as_str()) {
                    k.keys.insert(seg.clone(), t.string_id(seg.as_bytes())?);
                }
            }
            let (Some(vals), [key]) = (eq_strings(f), f.path.as_slice()) else {
                continue;
            };
            let packed = !NATIVE_STR.contains(&key.as_str());
            let literals = k.literals.entry(key.clone()).or_default();
            for v in vals {
                if !literals.contains_key(v) {
                    let id = if packed {
                        t.string_id(&value::encode(&Value::from(v))?)?
                    } else {
                        t.string_id(v.as_bytes())?
                    };
                    literals.insert(v.to_owned(), id);
                }
            }
        }
        Ok(k)
    }

    /// Whether any of `key`'s literals `literals` is interned as `id`.
    fn any_is(&self, key: &str, literals: &[&str], id: StrId) -> bool {
        self.literals.get(key).is_some_and(|m| {
            literals
                .iter()
                .any(|l| m.get(*l).copied().flatten() == Some(id))
        })
    }
}

/// Native string fields of nodes and edges, the only objects a slot can hold.
const NATIVE_STR: [&str; 2] = ["type", "value"];

/// The literals of a single-key `=`/`!=` test, if every one is a string.
fn eq_strings(f: &Filter) -> Option<Vec<&str>> {
    match (f.path.as_slice(), &f.test) {
        ([_], Test::In(vals) | Test::NotIn(vals)) => {
            vals.iter().map(Value::as_str).collect::<Option<Vec<_>>>()
        }
        _ => None,
    }
}

/// What stays fixed while one chain is expanded.
#[derive(Debug, Clone, Copy)]
struct Walk<'w> {
    p: &'w Pattern,
    view: Option<LogId>,
    keys: &'w Keys,
}

type Seeds<'s> = Box<dyn Iterator<Item = Result<Entry>> + 's>;

impl Txn<'_> {
    /// Runs `patterns` against view `before`, calling `sink(pattern index, chain)` for
    /// each match (chains hold the kept slots, in order) until it returns `false`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn query(
        &self,
        patterns: &[Pattern],
        before: Option<LogId>,
        mut sink: impl FnMut(usize, Vec<Entry>) -> bool,
    ) -> Result<()> {
        let view = self.view(before)?;
        let keys = Keys::new(self, patterns)?;
        for (pi, p) in patterns.iter().enumerate() {
            let Some(slot) = p.slots.get(p.seed) else {
                continue;
            };
            for seed in self.seeds(slot, view)? {
                let seed = seed?;
                if self.matches(&seed, slot, view, &keys)?
                    && !self.expand(p, p.seed, seed, view, &keys, &mut |c| sink(pi, c))?
                {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Streaming query over log entries `start..=stop` (`None` = to the end): reports
    /// each chain at the entry where one of its slots starts matching, evaluated in the
    /// view just after that entry. Calls `sink(pattern index, entry id, chain)` until
    /// it returns `false`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn mquery(
        &self,
        patterns: &[Pattern],
        start: LogId,
        stop: Option<LogId>,
        mut sink: impl FnMut(usize, LogId, Vec<Entry>) -> bool,
    ) -> Result<()> {
        let next = self.next_id()?;
        let keys = Keys::new(self, patterns)?;
        let end = stop.map_or(next, |s| s.saturating_add(1).min(next));
        // Edge slots testing `src.*`/`tgt.*` change when an endpoint's property does.
        let via_ends = patterns.iter().flat_map(|p| &p.slots).any(|s| {
            s.kind == Kind::Edge
                && s.filters
                    .iter()
                    .any(|f| matches!(f.path.first().map(String::as_str), Some("src" | "tgt")))
        });
        for x in start.max(1)..end {
            let Some(entry) = self.entry(x)? else {
                continue;
            };
            let (before, after) = (Some(x), Some(x + 1));
            let mut cands = Vec::new();
            let touch = |txn: &Self, parent: LogId, cands: &mut Vec<LogId>| -> Result<()> {
                cands.push(parent);
                if via_ends
                    && matches!(
                        txn.entry(parent)?,
                        Some(Entry {
                            record: Record::Node { .. },
                            ..
                        })
                    )
                {
                    for e in txn.node_edges(parent, Direction::Both, None, after)? {
                        cands.push(e?.id);
                    }
                }
                Ok(())
            };
            match entry.record {
                Record::Node { .. } => cands.push(x),
                Record::Edge { src, tgt, .. } => cands.extend([x, src, tgt]),
                Record::Prop { parent, .. } => touch(self, parent, &mut cands)?,
                Record::Deletion { target } => match self.entry(target)?.map(|e| e.record) {
                    Some(Record::Edge { src, tgt, .. }) => cands.extend([src, tgt]),
                    Some(Record::Prop { parent, .. }) => touch(self, parent, &mut cands)?,
                    Some(Record::Node { .. }) => {
                        for e in self.node_edges(target, Direction::Both, None, before)? {
                            if let Record::Edge { src, tgt, .. } = e?.record {
                                cands.extend([src, tgt]);
                            }
                        }
                    }
                    Some(Record::Deletion { .. }) | None => {}
                },
            }
            cands.sort_unstable();
            cands.dedup();

            let mut emitted = HashSet::new();
            let mut stopped = false;
            for id in cands {
                let Some(obj) = self.entry(id)?.filter(|o| o.live_at(after)) else {
                    continue;
                };
                let existed = id < x && obj.live_at(before);
                for (pi, p) in patterns.iter().enumerate() {
                    for (si, slot) in p.slots.iter().enumerate() {
                        if !self.matches(&obj, slot, after, &keys)?
                            || existed && self.matches(&obj, slot, before, &keys)?
                        {
                            continue;
                        }
                        self.expand(p, si, obj, after, &keys, &mut |c| {
                            if emitted.insert((pi, c.iter().map(|e| e.id).collect::<Vec<_>>()))
                                && !sink(pi, x, c)
                            {
                                stopped = true;
                            }
                            !stopped
                        })?;
                        if stopped {
                            return Ok(());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Candidate objects for a pattern's seed slot.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn seeds<'s>(&'s self, slot: &Slot, view: Option<LogId>) -> Result<Seeds<'s>> {
        let strs = |key| -> Option<Vec<&str>> {
            slot.eq_set(key)
                .map(|vs| vs.into_iter().filter_map(Value::as_str).collect())
        };
        if let Some(ids) = slot.eq_set("ID") {
            let mut out = Vec::new();
            for id in ids.into_iter().filter_map(Value::as_u64) {
                if let Some(e) = self.entry(id)?
                    && view.is_none_or(|b| id < b)
                    && e.live_at(view)
                    && kind(&e) == Some(slot.kind)
                {
                    out.push(Ok(e));
                }
            }
            return Ok(Box::new(out.into_iter()));
        }
        let table = match slot.kind {
            Kind::Node => self.g.t.node_idx,
            Kind::Edge => self.g.t.edge_idx,
        };
        Ok(match (slot.kind, strs("type"), strs("value")) {
            (Kind::Node, Some(ts), Some(vs)) => {
                let mut out = Vec::new();
                for t in &ts {
                    for v in &vs {
                        out.extend(self.node_lookup(t.as_bytes(), v.as_bytes(), view)?.map(Ok));
                    }
                }
                Box::new(out.into_iter())
            }
            (_, Some(ts), _) => {
                let its = ts
                    .iter()
                    .map(|t| self.by_type(table, Some(t.as_bytes()), view))
                    .collect::<Result<Vec<_>>>()?;
                Box::new(its.into_iter().flatten())
            }
            (_, None, _) => Box::new(self.by_type(table, None, view)?),
        })
    }

    /// Completes chains around `seed` sitting in slot `s`. Returns `false` once `sink` does.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn expand(
        &self,
        p: &Pattern,
        s: usize,
        seed: Entry,
        view: Option<LogId>,
        keys: &Keys,
        sink: &mut dyn FnMut(Vec<Entry>) -> bool,
    ) -> Result<bool> {
        let n = p.slots.len();
        // Fill rightwards from the seed, then leftwards: (slot to fill, filled neighbor).
        let order: Vec<(usize, usize)> = (s + 1..n)
            .map(|i| (i, i - 1))
            .chain((0..s).rev().map(|i| (i, i + 1)))
            .collect();
        let mut chain = vec![seed; n];
        let mut used = Vec::new();
        if p.slots.get(s).is_some_and(|x| x.uniq) {
            used.push(seed.id);
        }
        let walk = Walk { p, view, keys };
        self.step(&walk, &order, &mut chain, &mut used, sink)
    }

    /// # Errors
    /// Fails on storage errors.
    fn step(
        &self,
        w: &Walk<'_>,
        order: &[(usize, usize)],
        chain: &mut [Entry],
        seen: &mut Vec<LogId>,
        sink: &mut dyn FnMut(Vec<Entry>) -> bool,
    ) -> Result<bool> {
        let Walk { p, view, keys } = *w;
        let Some((&(pos, from), rest)) = order.split_first() else {
            let kept = p
                .slots
                .iter()
                .zip(chain.iter())
                .filter(|(s, _)| s.keep)
                .map(|(_, e)| *e)
                .collect();
            return Ok(sink(kept));
        };
        let bad = || GraphError::Corrupt("pattern slot out of range");
        let (slot, prev, cur) = (
            p.slots.get(pos).ok_or_else(bad)?,
            p.slots.get(from).ok_or_else(bad)?,
            *chain.get(from).ok_or_else(bad)?,
        );
        let dir = if pos > from { prev.fwd } else { prev.bwd };
        // An edge slot with `type=` walks only those types' edges (via the [node, type] index).
        let types = (slot.kind == Kind::Edge)
            .then(|| slot.eq_set("type"))
            .flatten();
        let types: Option<Vec<&str>> =
            types.map(|ts| ts.into_iter().filter_map(Value::as_str).collect());
        for cand in self.neighbors(&cur, dir, types.as_deref(), view)? {
            if slot.uniq && seen.contains(&cand.id) || !self.matches(&cand, slot, view, keys)? {
                continue;
            }
            *chain.get_mut(pos).ok_or_else(bad)? = cand;
            if slot.uniq {
                seen.push(cand.id);
            }
            let go = self.step(w, rest, chain, seen, sink)?;
            if slot.uniq {
                seen.pop();
            }
            if !go {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Edges of a node, or endpoints of an edge (`Out` = target, `In` = source).
    ///
    /// # Errors
    /// Fails on storage errors.
    fn neighbors(
        &self,
        e: &Entry,
        dir: Direction,
        types: Option<&[&str]>,
        view: Option<LogId>,
    ) -> Result<Vec<Entry>> {
        match (e.record, types) {
            (Record::Node { .. }, None) => self.node_edges(e.id, dir, None, view)?.collect(),
            (Record::Node { .. }, Some(types)) => {
                let mut out = Vec::new();
                for t in types {
                    for edge in self.node_edges(e.id, dir, Some(t.as_bytes()), view)? {
                        out.push(edge?);
                    }
                }
                Ok(out)
            }
            (Record::Edge { src, tgt, .. }, _) => {
                let ids = match dir {
                    Direction::In => vec![src],
                    Direction::Out => vec![tgt],
                    Direction::Both if src == tgt => vec![src],
                    Direction::Both => vec![src, tgt],
                };
                ids.into_iter()
                    .map(|id| self.entry(id)?.ok_or(GraphError::NotFound(id, "node")))
                    .collect()
            }
            (Record::Prop { .. } | Record::Deletion { .. }, _) => Ok(Vec::new()),
        }
    }

    /// Decoded value of property `key` on `parent`; undecodable (non-msgpack) values resolve to nothing.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn prop_value(
        &self,
        parent: LogId,
        key: &str,
        view: Option<LogId>,
        keys: &Keys,
    ) -> Result<Option<Value>> {
        let id = match keys.keys.get(key) {
            Some(&id) => id,
            None => self.string_id(key.as_bytes())?,
        };
        let Some(id) = id else { return Ok(None) };
        match self.lookup(self.g.t.prop_idx, &varint::pack(&[parent, id]), view)? {
            Some(Entry {
                record: Record::Prop { val, .. },
                ..
            }) => Ok(value::decode(self.string(val)?).ok()),
            _ => Ok(None),
        }
    }

    /// Whether `e` satisfies every filter of `slot` in `view`.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn matches(&self, e: &Entry, slot: &Slot, view: Option<LogId>, keys: &Keys) -> Result<bool> {
        if kind(e) != Some(slot.kind) {
            return Ok(false);
        }
        for f in &slot.filters {
            let ok = match self.eq_by_id(e, f, view, keys)? {
                Some(ok) => ok,
                None => self
                    .resolve(e, &f.path, view, keys)?
                    .is_some_and(|v| holds(&f.test, &v)),
            };
            if !ok {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// A single-key `=`/`!=` test against string literals, decided by comparing interned
    /// IDs instead of materialising the value. `None` when the test is not of that
    /// shape or the key is a computed native field, which [`resolve`](Self::resolve)
    /// handles. Same result as the slow path: a key that does not resolve fails both tests.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn eq_by_id(
        &self,
        e: &Entry,
        f: &Filter,
        view: Option<LogId>,
        keys: &Keys,
    ) -> Result<Option<bool>> {
        let (Some(literals), [k]) = (eq_strings(f), f.path.as_slice()) else {
            return Ok(None);
        };
        let negated = matches!(f.test, Test::NotIn(_));
        let id = match (e.record, k.as_str()) {
            (Record::Node { ty, .. } | Record::Edge { ty, .. }, "type") => ty,
            (Record::Node { val, .. } | Record::Edge { val, .. }, "value") => val,
            // `src`/`tgt` are handled before properties in `resolve`.
            (Record::Edge { .. }, "src" | "tgt") => return Ok(None),
            _ => {
                if self.native(e, k, view)?.is_some() {
                    return Ok(None);
                }
                let Some(key) = keys.keys.get(k).copied().flatten() else {
                    return Ok(Some(false));
                };
                return Ok(Some(
                    match self.lookup(self.g.t.prop_idx, &varint::pack(&[e.id, key]), view)? {
                        Some(Entry {
                            record: Record::Prop { val, .. },
                            ..
                        }) => keys.any_is(k, &literals, val) != negated,
                        _ => false,
                    },
                ));
            }
        };
        Ok(Some(keys.any_is(k, &literals, id) != negated))
    }

    /// The value at `path` on `e` in `view`, if any.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub(crate) fn resolve(
        &self,
        e: &Entry,
        path: &[String],
        view: Option<LogId>,
        keys: &Keys,
    ) -> Result<Option<Value>> {
        let Some((k0, rest)) = path.split_first() else {
            return Ok(None);
        };
        if let (Record::Edge { src: end, .. }, "src") | (Record::Edge { tgt: end, .. }, "tgt") =
            (e.record, k0.as_str())
        {
            if rest.is_empty() {
                return Ok(Some(Value::from(end)));
            }
            let Some(node) = self.entry(end)? else {
                return Ok(None);
            };
            return self.resolve(&node, rest, view, keys);
        }
        let base = match self.native(e, k0, view)? {
            Some(v) => Some(v),
            None => self.prop_value(e.id, k0, view, keys)?,
        };
        Ok(base.and_then(|b| {
            rest.iter().try_fold(b, |v, k| match v {
                Value::Object(mut m) => m.remove(k),
                Value::Null
                | Value::Bool(_)
                | Value::Number(_)
                | Value::String(_)
                | Value::Array(_) => None,
            })
        }))
    }

    /// A native field of `e` other than `src`/`tgt`, or `None` if `k0` is not one.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn native(&self, e: &Entry, k0: &str, view: Option<LogId>) -> Result<Option<Value>> {
        let s = |id| -> Result<Value> {
            Ok(Value::String(
                String::from_utf8_lossy(self.string(id)?).into_owned(),
            ))
        };
        let edges =
            |dir| -> Result<Vec<Entry>> { self.node_edges(e.id, dir, None, view)?.collect() };
        let degree = |dir| -> Result<Value> { Ok(Value::from(edges(dir)?.len())) };
        let ids = |dir| -> Result<Value> { Ok(edges(dir)?.iter().map(|e| e.id).collect()) };
        // Distinct nodes at the other end of this node's edges (self-loops excluded).
        let neighbors = || -> Result<Vec<LogId>> {
            let mut out: Vec<LogId> = edges(Direction::Both)?
                .iter()
                .filter_map(|x| match x.record {
                    Record::Edge { src, tgt, .. } => Some(if src == e.id { tgt } else { src }),
                    Record::Node { .. } | Record::Prop { .. } | Record::Deletion { .. } => None,
                })
                .filter(|&n| n != e.id)
                .collect();
            out.sort_unstable();
            out.dedup();
            Ok(out)
        };
        Ok(match (e.record, k0) {
            (_, "ID") => Some(Value::from(e.id)),
            (Record::Node { ty, .. } | Record::Edge { ty, .. }, "type") => Some(s(ty)?),
            (Record::Node { val, .. } | Record::Edge { val, .. }, "value") => Some(s(val)?),
            (Record::Node { ty, .. } | Record::Edge { ty, .. }, "typeID") => Some(Value::from(ty)),
            (Record::Node { val, .. } | Record::Edge { val, .. }, "valueID") => {
                Some(Value::from(val))
            }
            (Record::Node { .. }, "edge_count") => Some(degree(Direction::Both)?),
            (Record::Node { .. }, "inbound_count") => Some(degree(Direction::In)?),
            (Record::Node { .. }, "outbound_count") => Some(degree(Direction::Out)?),
            (Record::Node { .. }, "edges" | "edgeIDs") => Some(ids(Direction::Both)?),
            (Record::Node { .. }, "inbound" | "inboundIDs") => Some(ids(Direction::In)?),
            (Record::Node { .. }, "outbound" | "outboundIDs") => Some(ids(Direction::Out)?),
            (Record::Node { .. }, "neighbors" | "neighborIDs") => {
                Some(neighbors()?.into_iter().collect())
            }
            (Record::Node { .. }, "neighbor_count") => Some(Value::from(neighbors()?.len())),
            (Record::Node { .. }, "neighbor_types") => {
                let mut types = serde_json::Map::new();
                for n in neighbors()? {
                    if let Some(Entry {
                        record: Record::Node { ty, .. },
                        ..
                    }) = self.entry(n)?
                    {
                        let count = types
                            .entry(s(ty)?.as_str().unwrap_or_default().to_owned())
                            .or_insert(Value::from(0));
                        *count = Value::from(count.as_u64().unwrap_or(0) + 1);
                    }
                }
                Some(Value::Object(types))
            }
            (Record::Edge { src, .. }, "srcID") => Some(Value::from(src)),
            (Record::Edge { tgt, .. }, "tgtID") => Some(Value::from(tgt)),
            _ => None,
        })
    }
}

impl Pattern {
    /// Whether a one-slot pattern's filters all hold on a plain JSON object (keys are
    /// walked as nested object fields; nothing is native).
    #[must_use]
    pub fn matches_value(&self, v: &Value) -> bool {
        let [slot] = self.slots.as_slice() else {
            return false;
        };
        slot.filters.iter().all(|f| {
            f.path
                .iter()
                .try_fold(v, |v, k| v.as_object()?.get(k))
                .is_some_and(|v| holds(&f.test, v))
        })
    }
}

fn holds(test: &Test, v: &Value) -> bool {
    match test {
        Test::Exists => true,
        Test::In(vals) => vals.iter().any(|x| eq(x, v)),
        Test::NotIn(vals) => !vals.iter().any(|x| eq(x, v)),
        Test::Re(rs) => v.as_str().is_some_and(|s| rs.iter().any(|r| r.is_match(s))),
        Test::NotRe(rs) => v
            .as_str()
            .is_some_and(|s| !rs.iter().any(|r| r.is_match(s))),
        Test::Is(tys) => tys.iter().any(|t| t.has(v)),
        Test::IsNot(tys) => !tys.iter().any(|t| t.has(v)),
        Test::Cmp(c, x) => order(v, x).is_some_and(|o| match c {
            Cmp::Lt => o.is_lt(),
            Cmp::Le => o.is_le(),
            Cmp::Gt => o.is_gt(),
            Cmp::Ge => o.is_ge(),
        }),
    }
}

/// Equality with `1 == 1.0`; booleans are never numbers.
fn eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => num_cmp(x, y) == Some(Ordering::Equal),
        _ => a == b,
    }
}

/// Numbers compare with numbers, strings with strings; anything else is unordered.
fn order(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => num_cmp(x, y),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn num_cmp(x: &Number, y: &Number) -> Option<Ordering> {
    match (x.as_i64(), y.as_i64(), x.as_u64(), y.as_u64()) {
        (Some(x), Some(y), ..) => Some(x.cmp(&y)),
        (.., Some(x), Some(y)) => Some(x.cmp(&y)),
        _ => x.as_f64()?.partial_cmp(&y.as_f64()?),
    }
}
