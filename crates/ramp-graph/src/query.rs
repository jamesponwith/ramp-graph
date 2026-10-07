//! LGQL execution: ad-hoc queries against a view, streaming queries over a log range.
//!
//! A filter key resolves to a native field first, then to a property (msgpack value,
//! see [`crate::value`]), then walks nested object keys. Native fields:
//! - both: `ID`, `type`, `value`, `typeID`, `valueID`
//! - nodes: `edge_count`, `inbound_count`, `outbound_count`
//! - edges: `srcID`, `tgtID`, `src`, `tgt` (the latter two continue into the node: `src.type`)
//!
//! A key that does not resolve fails every test, including the negated ones.

use std::cmp::Ordering;
use std::collections::HashSet;

use serde_json::{Number, Value};

use crate::lgql::{Cmp, Kind, Pattern, Slot, Test};
use crate::{Direction, Entry, GraphError, LogId, Record, Result, Txn, value};

const fn kind(e: &Entry) -> Option<Kind> {
    match e.record {
        Record::Node { .. } => Some(Kind::Node),
        Record::Edge { .. } => Some(Kind::Edge),
        Record::Prop { .. } | Record::Deletion { .. } => None,
    }
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
        for (pi, p) in patterns.iter().enumerate() {
            let Some(slot) = p.slots.get(p.seed) else {
                continue;
            };
            for seed in self.seeds(slot, view)? {
                let seed = seed?;
                if self.matches(&seed, slot, view)?
                    && !self.expand(p, p.seed, seed, view, &mut |c| sink(pi, c))?
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
                        if !self.matches(&obj, slot, after)?
                            || existed && self.matches(&obj, slot, before)?
                        {
                            continue;
                        }
                        self.expand(p, si, obj, after, &mut |c| {
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
        self.step(p, &order, &mut chain, &mut used, view, sink)
    }

    /// # Errors
    /// Fails on storage errors.
    fn step(
        &self,
        p: &Pattern,
        order: &[(usize, usize)],
        chain: &mut [Entry],
        seen: &mut Vec<LogId>,
        view: Option<LogId>,
        sink: &mut dyn FnMut(Vec<Entry>) -> bool,
    ) -> Result<bool> {
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
        for cand in self.neighbors(&cur, dir, view)? {
            if slot.uniq && seen.contains(&cand.id) || !self.matches(&cand, slot, view)? {
                continue;
            }
            *chain.get_mut(pos).ok_or_else(bad)? = cand;
            if slot.uniq {
                seen.push(cand.id);
            }
            let go = self.step(p, rest, chain, seen, view, sink)?;
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
    fn neighbors(&self, e: &Entry, dir: Direction, view: Option<LogId>) -> Result<Vec<Entry>> {
        match e.record {
            Record::Node { .. } => self.node_edges(e.id, dir, None, view)?.collect(),
            Record::Edge { src, tgt, .. } => {
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
            Record::Prop { .. } | Record::Deletion { .. } => Ok(Vec::new()),
        }
    }

    /// Whether `e` satisfies every filter of `slot` in `view`.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn matches(&self, e: &Entry, slot: &Slot, view: Option<LogId>) -> Result<bool> {
        if kind(e) != Some(slot.kind) {
            return Ok(false);
        }
        for f in &slot.filters {
            if !self
                .resolve(e, &f.path, view)?
                .is_some_and(|v| holds(&f.test, &v))
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The value at `path` on `e` in `view`, if any.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn resolve(&self, e: &Entry, path: &[String], view: Option<LogId>) -> Result<Option<Value>> {
        let Some((k0, rest)) = path.split_first() else {
            return Ok(None);
        };
        let s = |id| -> Result<Value> {
            Ok(Value::String(
                String::from_utf8_lossy(self.string(id)?).into_owned(),
            ))
        };
        let degree = |dir| -> Result<Value> {
            let edges = self
                .node_edges(e.id, dir, None, view)?
                .try_fold(0_u64, |n, e| e.map(|_| n + 1))?;
            Ok(Value::from(edges))
        };
        let native = match (e.record, k0.as_str()) {
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
            (Record::Edge { src, .. }, "srcID") => Some(Value::from(src)),
            (Record::Edge { tgt, .. }, "tgtID") => Some(Value::from(tgt)),
            (Record::Edge { src: end, .. }, "src") | (Record::Edge { tgt: end, .. }, "tgt") => {
                if rest.is_empty() {
                    return Ok(Some(Value::from(end)));
                }
                let Some(node) = self.entry(end)? else {
                    return Ok(None);
                };
                return self.resolve(&node, rest, view);
            }
            _ => None,
        };
        let base = match native {
            Some(v) => Some(v),
            None => match self.prop(e.id, k0.as_bytes(), view)? {
                // Undecodable values (not msgpack) resolve to nothing.
                Some(Entry {
                    record: Record::Prop { val, .. },
                    ..
                }) => value::decode(self.string(val)?).ok(),
                _ => None,
            },
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
