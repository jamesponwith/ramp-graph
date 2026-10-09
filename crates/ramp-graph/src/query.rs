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
use crate::projection::Projection;
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
            for v in vals.iter().filter_map(Value::as_str) {
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
    fn any_is(&self, key: &str, literals: &[Value], id: StrId) -> bool {
        self.literals.get(key).is_some_and(|m| {
            literals
                .iter()
                .filter_map(Value::as_str)
                .any(|l| m.get(l).copied().flatten() == Some(id))
        })
    }
}

/// Native string fields of nodes and edges, the only objects a slot can hold.
const NATIVE_STR: [&str; 2] = ["type", "value"];

/// The literals of a single-key `=`/`!=` test, if every one is a string.
fn eq_strings(f: &Filter) -> Option<&[Value]> {
    match (f.path.as_slice(), &f.test) {
        ([_], Test::In(vals) | Test::NotIn(vals)) if vals.iter().all(Value::is_string) => {
            Some(vals)
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
    proj: Option<&'w Projection>,
}

type Seeds<'s> = Box<dyn Iterator<Item = Result<Entry>> + 's>;

/// Share `k` of `n` of a seed stream: a contiguous slice where the source is one,
/// every `n`th seed otherwise.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Part {
    pub(crate) k: usize,
    pub(crate) n: usize,
}

impl Part {
    const ALL: Self = Self { k: 0, n: 1 };

    fn slice<T>(self, s: &[T]) -> &[T] {
        let n = self.n.max(1);
        let (lo, hi) = (s.len() * self.k / n, s.len() * (self.k + 1) / n);
        s.get(lo..hi).unwrap_or_default()
    }

    fn takes(self, i: usize) -> bool {
        i % self.n.max(1) == self.k
    }
}

/// "Row unknown" for an entry that did not come from a projection.
const NO_ROW: u32 = u32::MAX;

/// Buffers reused across chains so expansion allocates nothing per hop or result.
#[derive(Debug, Default)]
struct Scratch {
    /// Candidate lists with their projection rows, one per recursion depth, kept for reuse.
    pool: Vec<(Vec<Entry>, Vec<u32>)>,
    /// The kept slots of the chain being reported.
    kept: Vec<Entry>,
    /// Slot fill order for the current pattern and seed slot.
    order: Vec<(usize, usize)>,
    /// The chain being filled.
    chain: Vec<Entry>,
    /// IDs held by `uniq` slots along the chain.
    used: Vec<LogId>,
}

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
        sink: impl FnMut(usize, &[Entry]) -> bool,
    ) -> Result<()> {
        // A cached projection is used only when it projects exactly this txn's view.
        let proj = match self.view(before)? {
            None => self
                .g
                .cached_projection()
                .filter(|s| s.end() == self.state.next_log),
            Some(_) => None,
        };
        self.query_with(proj.as_deref(), patterns, before, sink)
    }

    /// [`query`](Self::query) against `proj` (which must project this txn's current
    /// view) or, with `None`, against LMDB.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub(crate) fn query_with(
        &self,
        proj: Option<&Projection>,
        patterns: &[Pattern],
        before: Option<LogId>,
        mut sink: impl FnMut(usize, &[Entry]) -> bool,
    ) -> Result<()> {
        self.run(proj, patterns, before, Part::ALL, &mut sink)
    }

    /// The query loop: this `part` of the seeds is expanded; `sink` returning `false`
    /// ends it.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub(crate) fn run(
        &self,
        proj: Option<&Projection>,
        patterns: &[Pattern],
        before: Option<LogId>,
        part: Part,
        sink: &mut dyn FnMut(usize, &[Entry]) -> bool,
    ) -> Result<()> {
        let view = self.view(before)?;
        let keys = Keys::new(self, patterns)?;
        let mut scratch = Scratch::default();
        for (pi, p) in patterns.iter().enumerate() {
            let Some(slot) = p.slots.get(p.seed) else {
                continue;
            };
            let walk = Walk {
                p,
                view,
                keys: &keys,
                proj,
            };
            let (seeds, satisfied) = self.seeds(slot, view, &keys, proj, part)?;
            let single = p.slots.len() == 1 && slot.keep;
            for seed in seeds {
                let seed = seed?;
                if !self.matches_except(&seed, slot, view, &keys, proj, satisfied)? {
                    continue;
                }
                // A one-slot pattern has nothing to expand: the seed is the chain.
                let go = if single {
                    sink(pi, std::slice::from_ref(&seed))
                } else {
                    self.expand(&walk, p.seed, seed, &mut scratch, &mut |c| sink(pi, c))?
                };
                if !go {
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
        mut sink: impl FnMut(usize, LogId, &[Entry]) -> bool,
    ) -> Result<()> {
        let next = self.next_id()?;
        let keys = Keys::new(self, patterns)?;
        let mut scratch = Scratch::default();
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
                        if !self.matches(&obj, slot, after, &keys, None)?
                            || existed && self.matches(&obj, slot, before, &keys, None)?
                        {
                            continue;
                        }
                        let walk = Walk {
                            p,
                            view: after,
                            keys: &keys,
                            proj: None,
                        };
                        self.expand(&walk, si, obj, &mut scratch, &mut |c| {
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
    fn seeds<'s>(
        &'s self,
        slot: &Slot,
        view: Option<LogId>,
        keys: &Keys,
        proj: Option<&'s Projection>,
        part: Part,
    ) -> Result<(Seeds<'s>, Option<usize>)> {
        let strs = |key| -> Option<Vec<&str>> {
            slot.eq_set(key)
                .map(|vs| vs.into_iter().filter_map(Value::as_str).collect())
        };
        let share = move |it: Seeds<'s>| -> Seeds<'s> {
            Box::new(
                it.enumerate()
                    .filter(move |(i, _)| part.takes(*i))
                    .map(|(_, e)| e),
            )
        };
        if let Some(ids) = slot.eq_set("ID") {
            let mut out = Vec::new();
            for id in ids.into_iter().filter_map(Value::as_u64) {
                let found = match proj {
                    Some(pj) => pj.entry(id),
                    None => self.entry(id)?,
                };
                if let Some(e) = found
                    && view.is_none_or(|b| id < b)
                    && e.live_at(view)
                    && kind(&e) == Some(slot.kind)
                {
                    out.push(Ok(e));
                }
            }
            return Ok((share(Box::new(out.into_iter())), None));
        }
        if let Some(pj) = proj {
            return self.projection_seeds(pj, slot, keys, strs("type"), strs("value"), part);
        }
        let table = match slot.kind {
            Kind::Node => self.g.t.node_idx,
            Kind::Edge => self.g.t.edge_idx,
        };
        let it: Seeds<'s> = match (slot.kind, strs("type"), strs("value")) {
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
        };
        Ok((share(it), None))
    }

    /// [`seeds`](Self::seeds) from a projection: by type group, by a property equality
    /// (the first single-key `=` on string literals), or every row of the slot's kind.
    /// `(type, value)` pairs still go through the node index.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn projection_seeds<'s>(
        &'s self,
        pj: &'s Projection,
        slot: &Slot,
        keys: &Keys,
        types: Option<Vec<&str>>,
        values: Option<Vec<&str>>,
        part: Part,
    ) -> Result<(Seeds<'s>, Option<usize>)> {
        let edges = slot.kind == Kind::Edge;
        // The one filter a seed source already decided, so `matches` can skip it.
        let only = |pred: &dyn Fn(&Filter) -> bool| -> Option<usize> {
            let mut hits = slot.filters.iter().enumerate().filter(|(_, f)| pred(f));
            let first = hits.next()?.0;
            hits.next().is_none().then_some(first)
        };
        Ok(match (slot.kind, types, values) {
            (Kind::Node, Some(ts), Some(vs)) => {
                let mut out: Vec<Entry> = Vec::new();
                for t in &ts {
                    for v in &vs {
                        out.extend(self.node_lookup(t.as_bytes(), v.as_bytes(), None)?);
                    }
                }
                let mine: Vec<Entry> = part.slice(&out).to_vec();
                (Box::new(mine.into_iter().map(Ok)), None)
            }
            (_, Some(ts), _) => {
                let mut ids = Vec::new();
                for t in ts {
                    ids.extend(self.string_id(t.as_bytes())?);
                }
                let satisfied =
                    only(&|f| f.path.as_slice() == ["type"] && matches!(f.test, Test::In(_)));
                (
                    Box::new(
                        ids.into_iter()
                            .flat_map(move |ty| part.slice(pj.type_group(edges, ty)))
                            .filter_map(move |&(_, r)| pj.row(r))
                            .map(Ok),
                    ),
                    satisfied,
                )
            }
            (_, None, _) => {
                let by_prop = slot.filters.iter().enumerate().find_map(|(fi, f)| {
                    let ([k], Some(literals)) = (f.path.as_slice(), eq_strings(f)) else {
                        return None;
                    };
                    if NATIVE_STR.contains(&k.as_str()) || matches!(f.test, Test::NotIn(_)) {
                        return None;
                    }
                    let key = keys.keys.get(k).copied().flatten()?;
                    let vals: Vec<StrId> = literals
                        .iter()
                        .filter_map(Value::as_str)
                        .filter_map(|l| keys.literals.get(k)?.get(l).copied().flatten())
                        .collect();
                    Some((fi, key, vals))
                });
                match by_prop {
                    // One live value per (parent, key), so a parent appears once.
                    Some((fi, key, vals)) => (
                        Box::new(
                            vals.into_iter()
                                .flat_map(move |v| part.slice(pj.kv_range(key, v)))
                                .filter_map(move |&(_, _, parent)| pj.entry(parent))
                                .map(Ok),
                        ),
                        Some(fi),
                    ),
                    None => (
                        Box::new(
                            part.slice(pj.entries())
                                .iter()
                                .filter(move |e| matches!(e.record, Record::Edge { .. }) == edges)
                                .copied()
                                .map(Ok),
                        ),
                        None,
                    ),
                }
            }
        })
    }

    /// Completes chains around `seed` sitting in slot `s`. Returns `false` once `sink` does.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn expand(
        &self,
        w: &Walk<'_>,
        s: usize,
        seed: Entry,
        scratch: &mut Scratch,
        sink: &mut dyn FnMut(&[Entry]) -> bool,
    ) -> Result<bool> {
        let p = w.p;
        let n = p.slots.len();
        // Fill rightwards from the seed, then leftwards: (slot to fill, filled neighbor).
        let Scratch {
            order, chain, used, ..
        } = scratch;
        order.clear();
        order.extend((s + 1..n).map(|i| (i, i - 1)));
        order.extend((0..s).rev().map(|i| (i, i + 1)));
        chain.clear();
        chain.resize(n, seed);
        used.clear();
        if p.slots.get(s).is_some_and(|x| x.uniq) {
            used.push(seed.id);
        }
        let (order, mut chain, mut used) = (
            std::mem::take(order),
            std::mem::take(chain),
            std::mem::take(used),
        );
        let go = self.step(w, &order, (&mut chain, NO_ROW), &mut used, scratch, sink);
        scratch.order = order;
        scratch.chain = chain;
        scratch.used = used;
        go
    }

    /// # Errors
    /// Fails on storage errors.
    fn step(
        &self,
        w: &Walk<'_>,
        order: &[(usize, usize)],
        (chain, cur_row): (&mut [Entry], u32),
        seen: &mut Vec<LogId>,
        scratch: &mut Scratch,
        sink: &mut dyn FnMut(&[Entry]) -> bool,
    ) -> Result<bool> {
        let Walk {
            p,
            view,
            keys,
            proj,
        } = *w;
        let Some((&(pos, from), rest)) = order.split_first() else {
            scratch.kept.clear();
            scratch.kept.extend(
                p.slots
                    .iter()
                    .zip(chain.iter())
                    .filter(|(s, _)| s.keep)
                    .map(|(_, e)| *e),
            );
            return Ok(sink(&scratch.kept));
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
        let (mut cands, mut rows) = scratch.pool.pop().unwrap_or_default();
        cands.clear();
        rows.clear();
        self.neighbors(
            (&cur, cur_row),
            dir,
            types.as_deref(),
            view,
            keys,
            proj,
            (&mut cands, &mut rows),
        )?;
        let mut go = true;
        for i in 0..cands.len() {
            let Some(&cand) = cands.get(i) else { break };
            let cand_row = rows.get(i).copied().unwrap_or(NO_ROW);
            if slot.uniq && seen.contains(&cand.id)
                || !self.matches(&cand, slot, view, keys, proj)?
            {
                continue;
            }
            *chain.get_mut(pos).ok_or_else(bad)? = cand;
            if slot.uniq {
                seen.push(cand.id);
            }
            go = self.step(w, rest, (chain, cand_row), seen, scratch, sink)?;
            if slot.uniq {
                seen.pop();
            }
            if !go {
                break;
            }
        }
        scratch.pool.push((cands, rows));
        Ok(go)
    }

    /// Edges of a node, or endpoints of an edge (`Out` = target, `In` = source).
    ///
    /// # Errors
    /// Fails on storage errors.
    /// Edges of a node, or endpoints of an edge (`Out` = target, `In` = source), appended
    /// to `out`; with a projection, their rows go to `rows` so the next hop needs no
    /// ID → row lookup (`e_row` is this entry's row, or [`NO_ROW`]).
    ///
    /// # Errors
    /// Fails on storage errors.
    #[expect(clippy::too_many_arguments, reason = "one call site, hot path")]
    fn neighbors(
        &self,
        (e, e_row): (&Entry, u32),
        dir: Direction,
        types: Option<&[&str]>,
        view: Option<LogId>,
        keys: &Keys,
        proj: Option<&Projection>,
        (out, rows): (&mut Vec<Entry>, &mut Vec<u32>),
    ) -> Result<()> {
        match (e.record, types) {
            (Record::Node { .. }, None) => {
                if let Some(pj) = proj {
                    return pj.node_edges(e.id, dir, None, out, rows);
                }
                for edge in self.node_edges(e.id, dir, None, view)? {
                    out.push(edge?);
                }
                Ok(())
            }
            (Record::Node { .. }, Some(types)) => {
                for t in types {
                    // Type literals were interned once per query.
                    let ty = keys.literals.get("type").and_then(|m| m.get(*t).copied());
                    match (proj, ty) {
                        (Some(pj), Some(Some(ty))) => {
                            pj.node_edges(e.id, dir, Some(ty), out, rows)?;
                        }
                        (Some(_), _) => {}
                        (None, _) => {
                            for edge in self.node_edges(e.id, dir, Some(t.as_bytes()), view)? {
                                out.push(edge?);
                            }
                        }
                    }
                }
                Ok(())
            }
            (Record::Edge { src, tgt, .. }, _) => {
                // `In` leaves through the source, `Out` through the target.
                let dirs: &[Direction] = match dir {
                    Direction::In => &[Direction::In],
                    Direction::Out => &[Direction::Out],
                    Direction::Both if src == tgt => &[Direction::In],
                    Direction::Both => &[Direction::In, Direction::Out],
                };
                if let Some(pj) = proj {
                    let row = (e_row != NO_ROW).then_some(e_row);
                    return pj.edge_ends(e.id, row, dirs, out, rows);
                }
                for &d in dirs {
                    let id = if d == Direction::In { src } else { tgt };
                    out.push(self.entry(id)?.ok_or(GraphError::NotFound(id, "node"))?);
                }
                Ok(())
            }
            (Record::Prop { .. } | Record::Deletion { .. }, _) => Ok(()),
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
        proj: Option<&Projection>,
    ) -> Result<Option<Value>> {
        let id = match keys.keys.get(key) {
            Some(&id) => id,
            None => self.string_id(key.as_bytes())?,
        };
        let Some(id) = id else { return Ok(None) };
        match self.prop_id(parent, id, view, proj)? {
            Some(val) => Ok(value::decode(self.string(val)?).ok()),
            None => Ok(None),
        }
    }

    /// Value ID of live property `key` on `parent` in `view`.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn prop_id(
        &self,
        parent: LogId,
        key: StrId,
        view: Option<LogId>,
        proj: Option<&Projection>,
    ) -> Result<Option<StrId>> {
        if let Some(pj) = proj {
            return Ok(pj.prop(parent, key));
        }
        Ok(
            match self.lookup(self.g.t.prop_idx, &varint::pack(&[parent, key]), view)? {
                Some(Entry {
                    record: Record::Prop { val, .. },
                    ..
                }) => Some(val),
                _ => None,
            },
        )
    }

    /// Whether `e` satisfies every filter of `slot` in `view`.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn matches(
        &self,
        e: &Entry,
        slot: &Slot,
        view: Option<LogId>,
        keys: &Keys,
        proj: Option<&Projection>,
    ) -> Result<bool> {
        self.matches_except(e, slot, view, keys, proj, None)
    }

    /// [`matches`](Self::matches), trusting filter `skip` (one a seed source decided).
    ///
    /// # Errors
    /// Fails on storage errors.
    fn matches_except(
        &self,
        e: &Entry,
        slot: &Slot,
        view: Option<LogId>,
        keys: &Keys,
        proj: Option<&Projection>,
        skip: Option<usize>,
    ) -> Result<bool> {
        if kind(e) != Some(slot.kind) {
            return Ok(false);
        }
        for (fi, f) in slot.filters.iter().enumerate() {
            if skip == Some(fi) {
                continue;
            }
            let ok = match self.eq_by_id(e, f, view, keys, proj)? {
                Some(ok) => ok,
                None => self
                    .resolve(e, &f.path, view, keys, proj)?
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
        proj: Option<&Projection>,
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
                if self.native(e, k, view, proj)?.is_some() {
                    return Ok(None);
                }
                let Some(key) = keys.keys.get(k).copied().flatten() else {
                    return Ok(Some(false));
                };
                return Ok(Some(match self.prop_id(e.id, key, view, proj)? {
                    Some(val) => keys.any_is(k, literals, val) != negated,
                    None => false,
                }));
            }
        };
        Ok(Some(keys.any_is(k, literals, id) != negated))
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
        proj: Option<&Projection>,
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
            let node = match proj {
                Some(pj) => pj.entry(end),
                None => self.entry(end)?,
            };
            let Some(node) = node else {
                return Ok(None);
            };
            return self.resolve(&node, rest, view, keys, proj);
        }
        let base = match self.native(e, k0, view, proj)? {
            Some(v) => Some(v),
            None => self.prop_value(e.id, k0, view, keys, proj)?,
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
    fn native(
        &self,
        e: &Entry,
        k0: &str,
        view: Option<LogId>,
        proj: Option<&Projection>,
    ) -> Result<Option<Value>> {
        let s = |id| -> Result<Value> {
            Ok(Value::String(
                String::from_utf8_lossy(self.string(id)?).into_owned(),
            ))
        };
        let edges = |dir| -> Result<Vec<Entry>> {
            match proj {
                Some(pj) => {
                    let (mut out, mut rows) = (Vec::new(), Vec::new());
                    pj.node_edges(e.id, dir, None, &mut out, &mut rows)?;
                    Ok(out)
                }
                None => self.node_edges(e.id, dir, None, view)?.collect(),
            }
        };
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
                    let node = match proj {
                        Some(pj) => pj.entry(n),
                        None => self.entry(n)?,
                    };
                    if let Some(Entry {
                        record: Record::Node { ty, .. },
                        ..
                    }) = node
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
