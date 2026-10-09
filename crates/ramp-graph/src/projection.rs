//! A read-optimised projection of one committed view.
//!
//! Ad-hoc queries on LMDB pay a B-tree seek per candidate (the log read behind every
//! index hit, the index seek behind every hop). A [`Projection`] replaces those with array
//! walks: live nodes and edges in dense rows, node and edge rows grouped by type, CSR
//! adjacency per node in the order [`Txn::node_edges`] yields, and live properties sorted
//! both by `(key, value)` and by `(parent, key)`. It is built by one sequential scan of the
//! log and used by [`Txn::query`] when one exists for exactly the txn's view; every other
//! query, and `mquery`, keeps reading LMDB.
//!
//! Semantics are the executor's, unchanged: the projection only answers "which rows have this
//! type", "which edges touch this node", and "what is property `k` of this object".
//!
//! Everything a row holds is live at the view, so no row stores `next`; string IDs are
//! kept as `u32` (a graph with more interned strings than that cannot be projected).

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::query::Part;
use crate::{Direction, Entry, Graph, GraphError, IdHasher, LogId, Record, Result, StrId, Txn};

/// Maps keyed by integers: one multiply per word instead of `SipHash`.
type Fast = BuildHasherDefault<IdHasher>;

/// Row index into [`Projection::rows`].
type Row = u32;

/// A string ID as the projection stores it.
type Sid = u32;

/// A live node or edge: its log ID and strings. Edges keep their endpoints in `ends`.
#[derive(Debug, Clone, Copy)]
struct Obj {
    id: LogId,
    ty: Sid,
    val: Sid,
}

/// A node as an edge's endpoint: its row and enough to rebuild its [`Entry`].
#[derive(Debug, Clone, Copy, Default)]
struct End {
    row: Row,
    id: LogId,
    ty: Sid,
    val: Sid,
}

impl End {
    fn entry(self) -> Entry {
        Entry {
            id: self.id,
            next: 0,
            record: Record::Node {
                ty: StrId::from(self.ty),
                val: StrId::from(self.val),
            },
        }
    }
}

/// `end_idx` of a node row.
const NO_END: u32 = u32::MAX;

/// The immutable part of a [`Projection`]: one committed view, built from the log.
///
///
/// Ad-hoc queries on LMDB pay a B-tree seek per candidate: the log read behind every
/// index hit, the index seek behind every hop. A projection replaces those with array
/// walks: live nodes and edges in dense rows, rows grouped by type, CSR adjacency per
/// node in the order [`Txn::node_edges`] yields, and live properties sorted by
/// `(key, value)` and by `(parent, key)`. Built by [`Txn::projection`] from one
/// sequential scan of the log; used by [`Txn::query`] when one is cached for exactly
/// the txn's view (any commit drops it); historical views and `mquery` read LMDB.
/// Semantics are the executor's, unchanged.
#[derive(Debug)]
pub(crate) struct Base {
    /// The view: log entries `< end`.
    end: LogId,
    /// Log ID → row + 1 (0: not a live node or edge in this view). Indexed up to `end`.
    slot: Vec<Row>,
    /// Live nodes and edges, in log order.
    rows: Vec<Obj>,
    /// Node rows sorted by `(type, row)`.
    nodes_by_type: Vec<(Sid, Row)>,
    /// Edge rows sorted by `(type, row)`.
    edges_by_type: Vec<(Sid, Row)>,
    /// CSR: edge rows leaving node row `n` are `out_adj[out_off[n]..out_off[n + 1]]`,
    /// sorted by `(edge type, edge id)` like the `srcnode_idx` table.
    out_off: Vec<u32>,
    out_adj: Vec<Row>,
    /// CSR of edges entering each node row, as `out_*`.
    in_off: Vec<u32>,
    in_adj: Vec<Row>,
    /// Per row, an edge's index into `ends` (`NO_END` for nodes).
    end_idx: Vec<u32>,
    /// Per edge, both endpoints copied in, so a hop from an edge reads nothing cold.
    ends: Vec<[End; 2]>,
    /// Live properties as `(key, value, parent)`, sorted.
    by_kv: ByKv,
    /// Live properties as `(parent, key, value)`, sorted; one per `(parent, key)`.
    by_pk: ByPk,
    /// Per row, where its properties start in `by_pk` (one past the end at `rows.len()`).
    prop_off: Vec<u32>,
}

/// # Errors
/// Fails if the projection would need more rows than [`Row`] can index.
fn row(i: usize) -> Result<Row> {
    Row::try_from(i).map_err(|_| GraphError::Corrupt("projection has too many rows"))
}

/// # Errors
/// Fails on a string ID past what the projection stores.
fn sid(id: StrId) -> Result<Sid> {
    Sid::try_from(id).map_err(|_| GraphError::Corrupt("too many strings to project"))
}

fn index(x: u32) -> usize {
    usize::try_from(x).unwrap_or(usize::MAX)
}

/// The adjacency list of node row `n` in a CSR pair.
fn adjacency<'a>(n: Row, off: &[u32], adj: &'a [Row]) -> &'a [Row] {
    let (Some(&a), Some(&b)) = (off.get(index(n)), off.get(index(n) + 1)) else {
        return &[];
    };
    adj.get(index(a)..index(b)).unwrap_or_default()
}

/// Prefix sums turn per-row degrees into CSR offsets.
fn offsets(deg: &[u32]) -> Vec<u32> {
    let mut off = Vec::with_capacity(deg.len());
    let mut acc = 0;
    for &d in deg {
        off.push(acc);
        acc += d;
    }
    off
}

/// CSR adjacency `[out_off, out_adj, in_off, in_adj]` over `n_rows` rows.
///
/// `edges` is `(edge row, source row, target row)` in edge-row order; `by_type` lists
/// edge rows in `(edge type, edge id)` order, which is each node's list order.
fn csr(n_rows: usize, edges: &[(Row, Row, Row)], by_type: &[(Sid, Row)]) -> [Vec<u32>; 4] {
    let mut out_deg = vec![0_u32; n_rows + 1];
    let mut in_deg = vec![0_u32; n_rows + 1];
    for &(_, s, d) in edges {
        if let Some(c) = out_deg.get_mut(index(s)) {
            *c += 1;
        }
        if let Some(c) = in_deg.get_mut(index(d)) {
            *c += 1;
        }
    }
    let out_off = offsets(&out_deg);
    let in_off = offsets(&in_deg);
    let mut out_adj = vec![0; index(out_off.last().copied().unwrap_or(0))];
    let mut in_adj = vec![0; index(in_off.last().copied().unwrap_or(0))];
    let mut out_cur = out_off.clone();
    let mut in_cur = in_off.clone();
    for &(_, r) in by_type {
        let Ok(i) = edges.binary_search_by_key(&r, |&(er, _, _)| er) else {
            continue;
        };
        let Some(&(_, s, d)) = edges.get(i) else {
            continue;
        };
        for (n, cur, adj) in [
            (s, &mut out_cur, &mut out_adj),
            (d, &mut in_cur, &mut in_adj),
        ] {
            if let Some(c) = cur.get_mut(index(n)) {
                if let Some(a) = adj.get_mut(index(*c)) {
                    *a = r;
                }
                *c += 1;
            }
        }
    }
    [out_off, out_adj, in_off, in_adj]
}

/// Live properties as `(key, value, parent)`, sorted.
type ByKv = Vec<(Sid, Sid, LogId)>;
/// Live properties as `(parent, key, value)`, sorted; one per `(parent, key)`.
type ByPk = Vec<(LogId, Sid, Sid)>;

/// The two sorted property arrays and, per row, where its properties start in `by_pk`.
///
/// # Errors
/// Fails if there are more properties than [`Row`] can index.
fn props_index(rows: &[Obj], props: ByPk) -> Result<(ByKv, ByPk, Vec<u32>)> {
    let mut by_kv: Vec<(Sid, Sid, LogId)> = props.iter().map(|&(p, k, v)| (k, v, p)).collect();
    by_kv.sort_unstable();
    let mut by_pk = props;
    by_pk.sort_unstable();
    // Rows and `by_pk` are both in log-ID order; properties of non-rows (a property of
    // a property) sit between and are skipped past.
    let mut prop_off = Vec::with_capacity(rows.len() + 1);
    let mut at = 0_usize;
    for o in rows {
        at += by_pk
            .get(at..)
            .unwrap_or_default()
            .partition_point(|&(p, _, _)| p < o.id);
        prop_off.push(row(at)?);
    }
    prop_off.push(row(by_pk.len())?);
    Ok((by_kv, by_pk, prop_off))
}

impl Base {
    /// Projects the view of `t` (everything committed before it started, or `t`'s own
    /// writes too in a write txn).
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt log.
    pub(crate) fn build(t: &Txn<'_>) -> Result<Self> {
        let end = t.next_id()?;
        let view = Some(end);
        let mut slot =
            vec![0; usize::try_from(end).map_err(|_| GraphError::Corrupt("log too long"))?];
        let mut rows: Vec<Obj> = Vec::new();
        let mut end_idx: Vec<u32> = Vec::new();
        // Per edge, in row order: its row and endpoint IDs, until all rows are known.
        let mut edge_ids: Vec<(Row, LogId, LogId)> = Vec::new();
        let mut props: Vec<(LogId, Sid, Sid)> = Vec::new();
        for e in t.log(1, None)? {
            let e = e?;
            if !e.live_at(view) {
                continue;
            }
            match e.record {
                Record::Node { ty, val } | Record::Edge { ty, val, .. } => {
                    let r = row(rows.len())?;
                    if let Some(s) = slot.get_mut(usize::try_from(e.id).unwrap_or(usize::MAX)) {
                        *s = r + 1;
                    }
                    rows.push(Obj {
                        id: e.id,
                        ty: sid(ty)?,
                        val: sid(val)?,
                    });
                    if let Record::Edge { src, tgt, .. } = e.record {
                        end_idx.push(row(edge_ids.len())?);
                        edge_ids.push((r, src, tgt));
                    } else {
                        end_idx.push(NO_END);
                    }
                }
                Record::Prop { parent, key, val } => props.push((parent, sid(key)?, sid(val)?)),
                Record::Deletion { .. } => {}
            }
        }
        let by_row = |id: LogId| -> Option<Row> {
            let s = *slot.get(usize::try_from(id).ok()?)?;
            s.checked_sub(1)
        };
        let mut nodes_by_type = Vec::new();
        let mut edges_by_type = Vec::new();
        for (i, (o, &ei)) in rows.iter().zip(&end_idx).enumerate() {
            if ei == NO_END {
                nodes_by_type.push((o.ty, row(i)?));
            } else {
                edges_by_type.push((o.ty, row(i)?));
            }
        }
        nodes_by_type.sort_unstable();
        edges_by_type.sort_unstable();
        let end_of = |id: LogId| -> Result<End> {
            let r = by_row(id).ok_or(GraphError::Corrupt("live edge on a dead node"))?;
            match (rows.get(index(r)), end_idx.get(index(r))) {
                (Some(o), Some(&NO_END)) => Ok(End {
                    row: r,
                    id: o.id,
                    ty: o.ty,
                    val: o.val,
                }),
                _ => Err(GraphError::Corrupt("edge endpoint is not a node")),
            }
        };
        let mut ends = Vec::with_capacity(edge_ids.len());
        let mut edges = Vec::with_capacity(edge_ids.len());
        for &(r, src, tgt) in &edge_ids {
            let (s, d) = (end_of(src)?, end_of(tgt)?);
            edges.push((r, s.row, d.row));
            ends.push([s, d]);
        }
        drop(edge_ids);
        let [out_off, out_adj, in_off, in_adj] = csr(rows.len(), &edges, &edges_by_type);
        drop(edges);

        let (by_kv, by_pk, prop_off) = props_index(&rows, props)?;
        Ok(Self {
            end,
            slot,
            rows,
            nodes_by_type,
            edges_by_type,
            out_off,
            out_adj,
            in_off,
            in_adj,
            end_idx,
            ends,
            by_kv,
            by_pk,
            prop_off,
        })
    }

    /// The view this projection projects: log entries below it.
    #[must_use]
    pub(crate) const fn end(&self) -> LogId {
        self.end
    }

    /// Live nodes and edges in this view.
    #[must_use]
    pub(crate) const fn len(&self) -> usize {
        self.rows.len()
    }

    /// The entry in row `r`.
    #[must_use]
    pub(crate) fn row(&self, r: Row) -> Option<Entry> {
        let (o, &ei) = (self.rows.get(index(r))?, self.end_idx.get(index(r))?);
        let record = if ei == NO_END {
            Record::Node {
                ty: StrId::from(o.ty),
                val: StrId::from(o.val),
            }
        } else {
            let [s, d] = self.ends.get(index(ei))?;
            Record::Edge {
                ty: StrId::from(o.ty),
                val: StrId::from(o.val),
                src: s.id,
                tgt: d.id,
            }
        };
        Some(Entry {
            id: o.id,
            next: 0,
            record,
        })
    }

    /// Row `r` as an edge endpoint, if it is a node.
    fn node_end(&self, r: Row) -> Option<End> {
        let (o, &ei) = (self.rows.get(index(r))?, self.end_idx.get(index(r))?);
        (ei == NO_END).then_some(End {
            row: r,
            id: o.id,
            ty: o.ty,
            val: o.val,
        })
    }

    /// Rows of every edge into or out of node row `n`.
    fn edge_rows(&self, n: Row) -> impl Iterator<Item = Row> + '_ {
        adjacency(n, &self.in_off, &self.in_adj)
            .iter()
            .chain(adjacency(n, &self.out_off, &self.out_adj))
            .copied()
    }

    /// # Errors
    /// Fails on a row index the projection does not have.
    fn by_row(&self, r: Row) -> Result<Entry> {
        self.row(r)
            .ok_or(GraphError::Corrupt("projection row out of range"))
    }

    /// Rows of live nodes (`edges == false`) or edges of type `ty`, in log order.
    pub(crate) fn type_group(&self, edges: bool, ty: StrId) -> &[(Sid, Row)] {
        let Ok(ty) = sid(ty) else { return &[] };
        let groups = if edges {
            &self.edges_by_type
        } else {
            &self.nodes_by_type
        };
        let lo = groups.partition_point(|&(t, _)| t < ty);
        let hi = lo
            + groups
                .get(lo..)
                .unwrap_or_default()
                .partition_point(|&(t, _)| t == ty);
        groups.get(lo..hi).unwrap_or_default()
    }

    /// Row of a live node or edge.
    #[must_use]
    pub(crate) fn row_of(&self, id: LogId) -> Option<Row> {
        let s = *self.slot.get(usize::try_from(id).ok()?)?;
        s.checked_sub(1)
    }

    /// Appends the endpoints of edge `edge` (row `row` if the caller knows it) for
    /// `dirs` (`In`: source, `Out`: target) to `out`, with their rows to `rows`.
    ///
    /// # Errors
    /// Fails if the edge is not in the projection.
    pub(crate) fn edge_ends(
        &self,
        edge: LogId,
        row: Option<Row>,
        dirs: &[Direction],
        out: &mut Vec<Entry>,
        rows: &mut Vec<Row>,
    ) -> Result<()> {
        let r = row.or_else(|| self.row_of(edge));
        let [src, tgt] = r
            .and_then(|r| self.end_idx.get(index(r)).copied())
            .and_then(|i| self.ends.get(index(i)).copied())
            .ok_or(GraphError::NotFound(edge, "edge"))?;
        for &d in dirs {
            let n = if d == Direction::Out { tgt } else { src };
            out.push(n.entry());
            rows.push(n.row);
        }
        Ok(())
    }

    /// Appends the edges of `node` in direction `dir`, optionally of type `ty`, to
    /// `out` in the order [`Txn::node_edges`] yields them (inbound then outbound; a
    /// self-loop once).
    ///
    /// # Errors
    /// Fails on a corrupt projection.
    pub(crate) fn node_edges(
        &self,
        node: LogId,
        dir: Direction,
        ty: Option<StrId>,
        out: &mut Vec<Entry>,
        rows: &mut Vec<Row>,
    ) -> Result<()> {
        let Some(n) = self.row_of(node) else {
            return Ok(());
        };
        let ty = match ty.map(sid) {
            None => None,
            Some(Ok(t)) => Some(t),
            Some(Err(_)) => return Ok(()), // no edge can have that type
        };
        let has_type =
            |r: Row| ty.is_none_or(|t| self.rows.get(index(r)).is_some_and(|o| o.ty == t));
        let both = dir == Direction::Both;
        if dir != Direction::Out {
            for &r in adjacency(n, &self.in_off, &self.in_adj) {
                if has_type(r) {
                    out.push(self.by_row(r)?);
                    rows.push(r);
                }
            }
        }
        if dir != Direction::In {
            for &r in adjacency(n, &self.out_off, &self.out_adj) {
                let is_loop = both
                    && self
                        .end_idx
                        .get(index(r))
                        .and_then(|&ei| self.ends.get(index(ei)))
                        .is_some_and(|[s, d]| s.row == d.row);
                if !is_loop && has_type(r) {
                    out.push(self.by_row(r)?);
                    rows.push(r);
                }
            }
        }
        Ok(())
    }

    /// Value ID of live property `key` on `parent`.
    #[must_use]
    pub(crate) fn prop(&self, parent: LogId, key: StrId) -> Option<StrId> {
        let key = sid(key).ok()?;
        let i = self
            .by_pk
            .partition_point(|&(p, k, _)| (p, k) < (parent, key));
        self.by_pk
            .get(i)
            .filter(|&&(p, k, _)| p == parent && k == key)
            .map(|&(_, _, v)| StrId::from(v))
    }

    /// Live properties of `parent` as `(key, value)` string IDs, in key-ID order (the
    /// order [`Txn::props`] yields them).
    pub(crate) fn props_of(&self, parent: LogId) -> impl Iterator<Item = (StrId, StrId)> + '_ {
        let span = match self.row_of(parent) {
            // Not a node or edge (a property's properties): search.
            None => {
                let lo = self.by_pk.partition_point(|&(p, _, _)| p < parent);
                let hi = lo
                    + self
                        .by_pk
                        .get(lo..)
                        .unwrap_or_default()
                        .partition_point(|&(p, _, _)| p == parent);
                self.by_pk.get(lo..hi).unwrap_or_default()
            }
            Some(r) => match (self.prop_off.get(index(r)), self.prop_off.get(index(r) + 1)) {
                // The span runs to the next row's start, so it may end with properties
                // of non-rows that sort after this row; this row's own come first.
                (Some(&lo), Some(&hi)) => {
                    let span = self.by_pk.get(index(lo)..index(hi)).unwrap_or_default();
                    let len = span.partition_point(|&(p, _, _)| p == parent);
                    span.get(..len).unwrap_or_default()
                }
                _ => &[],
            },
        };
        span.iter()
            .map(|&(_, k, v)| (StrId::from(k), StrId::from(v)))
    }

    /// Live properties `key == val` as `(key, val, parent)`, in parent order.
    pub(crate) fn kv_range(&self, key: StrId, val: StrId) -> &[(Sid, Sid, LogId)] {
        let (Ok(key), Ok(val)) = (sid(key), sid(val)) else {
            return &[];
        };
        let lo = self.by_kv.partition_point(|&(k, v, _)| (k, v) < (key, val));
        let hi = lo
            + self
                .by_kv
                .get(lo..)
                .unwrap_or_default()
                .partition_point(|&(k, v, _)| (k, v) == (key, val));
        self.by_kv.get(lo..hi).unwrap_or_default()
    }

    /// Bytes this projection holds, for sizing.
    #[must_use]
    pub(crate) const fn bytes(&self) -> usize {
        self.slot.len() * size_of::<Row>()
            + self.rows.len() * size_of::<Obj>()
            + (self.nodes_by_type.len() + self.edges_by_type.len()) * size_of::<(Sid, Row)>()
            + (self.out_off.len() + self.in_off.len()) * size_of::<u32>()
            + (self.out_adj.len() + self.in_adj.len()) * size_of::<Row>()
            + self.end_idx.len() * size_of::<u32>()
            + self.ends.len() * size_of::<[End; 2]>()
            + self.by_kv.len() * size_of::<(Sid, Sid, LogId)>()
            + self.by_pk.len() * size_of::<(LogId, Sid, Sid)>()
            + self.prop_off.len() * size_of::<u32>()
    }
}

/// Changes logged after a projection's base was built.
#[derive(Debug, Clone, Default)]
struct Delta {
    /// Nodes and edges created since the base; delta row `i` is row `base.len() + i`.
    rows: Vec<DRow>,
    /// Log ID → delta row index.
    slot: HashMap<LogId, u32, Fast>,
    /// `(is edge, type)` → delta row indexes, in log order.
    by_type: HashMap<(bool, Sid), Vec<u32>, Fast>,
    /// Node ID → delta edges leaving it, and entering it, in log order.
    out: HashMap<LogId, Vec<u32>, Fast>,
    inb: HashMap<LogId, Vec<u32>, Fast>,
    /// Base rows ended since the base, one bit each; empty until one is.
    dead: Vec<u64>,
    /// Rows ended since the base, base and delta.
    deaths: usize,
    /// `(parent, key)` → the property's value now, `None` once removed.
    props: HashMap<(LogId, Sid), Option<Sid>, Fast>,
    /// Keys in `props`, per parent.
    keys: HashMap<LogId, Vec<Sid>, Fast>,
    /// `(key, value)` → parents given it since the base (some may have changed since).
    kv: HashMap<(Sid, Sid), Vec<LogId>, Fast>,
    kv_seen: HashSet<(Sid, Sid, LogId), Fast>,
    /// Log entries applied.
    ops: usize,
}

/// A node or edge created since the base.
#[derive(Debug, Clone, Copy)]
struct DRow {
    obj: Obj,
    /// An edge's endpoints; `None` for a node.
    ends: Option<[End; 2]>,
    dead: bool,
}

impl DRow {
    fn entry(&self) -> Entry {
        let (ty, val) = (StrId::from(self.obj.ty), StrId::from(self.obj.val));
        Entry {
            id: self.obj.id,
            next: 0,
            record: match self.ends {
                None => Record::Node { ty, val },
                Some([s, d]) => Record::Edge {
                    ty,
                    val,
                    src: s.id,
                    tgt: d.id,
                },
            },
        }
    }
}

/// The type of an edge entry, for ordering adjacency lists.
const fn edge_ty(e: &Entry) -> StrId {
    match e.record {
        Record::Edge { ty, .. } => ty,
        Record::Node { .. } | Record::Prop { .. } | Record::Deletion { .. } => 0,
    }
}

/// A read-optimised copy of one committed view, for [`Txn::query`].
///
/// Ad-hoc queries on LMDB pay a B-tree seek per candidate: the log read behind every
/// index hit, the index seek behind every hop. A projection replaces those with array
/// walks: live nodes and edges in dense rows, rows grouped by type, CSR adjacency per
/// node in the order [`Txn::node_edges`] yields, and live properties sorted by
/// `(key, value)` and by `(parent, key)`.
///
/// It is an immutable base, built by one sequential scan of the log, plus a delta of
/// what was logged since. [`Txn::projection`] advances a cached projection by replaying
/// only the new log entries, and rebuilds the base once the delta grows past a
/// fraction of it. [`Txn::query`] uses one when it projects exactly the txn's view;
/// historical views and `mquery` read LMDB. Semantics are the executor's, unchanged.
#[derive(Debug, Clone)]
pub struct Projection {
    base: Arc<Base>,
    delta: Delta,
    /// The view: log entries `< end`.
    end: LogId,
    /// The graph's reset count when the base was built; a reset reuses log IDs.
    epoch: u64,
}

impl Projection {
    /// Projects the view of `t` from scratch.
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt log.
    pub(crate) fn build(t: &Txn<'_>, epoch: u64) -> Result<Self> {
        let base = Base::build(t)?;
        Ok(Self {
            end: base.end(),
            base: Arc::new(base),
            delta: Delta::default(),
            epoch,
        })
    }

    /// The view this projection projects: log entries below it.
    #[must_use]
    pub const fn end(&self) -> LogId {
        self.end
    }

    pub(crate) const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Log entries applied on top of the base since it was built.
    #[must_use]
    pub const fn delta_len(&self) -> usize {
        self.delta.ops
    }

    /// Whether the delta has grown enough that a fresh base would serve queries better.
    pub(crate) fn wants_rebuild(&self) -> bool {
        self.delta.ops > (self.base.len() / 32).max(1 << 13)
    }

    /// Live nodes and edges in this view.
    #[must_use]
    pub fn len(&self) -> usize {
        (self.base.len() + self.delta.rows.len()).saturating_sub(self.delta.deaths)
    }

    /// Whether the view has no live node or edge.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes this projection holds, for sizing (the delta estimated).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.base.bytes() + self.delta.ops.saturating_mul(96) + self.delta.dead.len() * 8
    }

    fn base_rows(&self) -> Row {
        Row::try_from(self.base.len()).unwrap_or(Row::MAX)
    }

    fn base_dead(&self, r: Row) -> bool {
        self.delta
            .dead
            .get(index(r / 64))
            .is_some_and(|w| (w >> (r % 64)) & 1 == 1)
    }

    fn drow(&self, r: Row) -> Option<&DRow> {
        self.delta.rows.get(index(r.checked_sub(self.base_rows())?))
    }

    /// Row of a live node or edge.
    pub(crate) fn row_of(&self, id: LogId) -> Option<Row> {
        if let Some(r) = self.base.row_of(id) {
            return (!self.base_dead(r)).then_some(r);
        }
        if self.delta.slot.is_empty() {
            return None;
        }
        let r = self.base_rows().checked_add(*self.delta.slot.get(&id)?)?;
        self.drow(r).filter(|d| !d.dead).map(|_| r)
    }

    /// The live entry in row `r`.
    pub(crate) fn row(&self, r: Row) -> Option<Entry> {
        if r < self.base_rows() {
            return (!self.base_dead(r)).then(|| self.base.row(r)).flatten();
        }
        self.drow(r).filter(|d| !d.dead).map(DRow::entry)
    }

    /// The live node or edge with this ID.
    #[must_use]
    pub fn entry(&self, id: LogId) -> Option<Entry> {
        self.row(self.row_of(id)?)
    }

    /// Delta row indexes as rows, this `part` of them.
    fn delta_part<'a>(&self, list: &'a [u32], part: Part) -> impl Iterator<Item = Row> + 'a {
        let b = self.base_rows();
        list.iter()
            .enumerate()
            .filter(move |(i, _)| part.takes(*i))
            .map(move |(_, &i)| b.saturating_add(i))
    }

    /// Live nodes (`edges == false`) or edges of type `ty`, this `part` of them.
    pub(crate) fn type_rows(
        &self,
        edges: bool,
        ty: StrId,
        part: Part,
    ) -> Box<dyn Iterator<Item = Entry> + '_> {
        if self.delta.ops == 0 {
            let group = part.slice(self.base.type_group(edges, ty));
            return Box::new(group.iter().filter_map(|&(_, r)| self.base.row(r)));
        }
        let added = sid(ty)
            .ok()
            .and_then(|t| self.delta.by_type.get(&(edges, t)))
            .map_or(&[][..], Vec::as_slice);
        Box::new(
            part.slice(self.base.type_group(edges, ty))
                .iter()
                .map(|&(_, r)| r)
                .chain(self.delta_part(added, part))
                .filter_map(move |r| self.row(r)),
        )
    }

    /// Every live node or every live edge, this `part` of them.
    pub(crate) fn rows_of_kind(&self, edges: bool, part: Part) -> impl Iterator<Item = Entry> + '_ {
        let b = self.base_rows();
        let base = part
            .range(self.base.len())
            .filter_map(|i| Row::try_from(i).ok());
        let added = (0..self.delta.rows.len())
            .filter(move |i| part.takes(*i))
            .filter_map(|i| Row::try_from(i).ok())
            .map(move |i| b.saturating_add(i));
        base.chain(added)
            .filter_map(move |r| self.row(r))
            .filter(move |e| matches!(e.record, Record::Edge { .. }) == edges)
    }

    /// Live parents whose property `key` is `val`, this `part` of them.
    pub(crate) fn parents_with(
        &self,
        key: StrId,
        val: StrId,
        part: Part,
    ) -> Box<dyn Iterator<Item = Entry> + '_> {
        if self.delta.ops == 0 {
            let range = part.slice(self.base.kv_range(key, val));
            return Box::new(
                range
                    .iter()
                    .filter_map(|&(_, _, p)| self.base.row(self.base.row_of(p)?)),
            );
        }
        let (k, v) = (sid(key).ok(), sid(val).ok());
        // A base entry is current unless the delta has touched that (parent, key).
        let base = part
            .slice(self.base.kv_range(key, val))
            .iter()
            .filter(move |&&(_, _, p)| k.is_none_or(|k| !self.delta.props.contains_key(&(p, k))))
            .map(|&(_, _, p)| p);
        let added = k
            .zip(v)
            .and_then(|kv| self.delta.kv.get(&kv))
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .enumerate()
            .filter(move |(i, _)| part.takes(*i))
            .map(|(_, &p)| p)
            .filter(move |&p| k.is_some_and(|k| self.delta.props.get(&(p, k)) == Some(&v)));
        Box::new(base.chain(added).filter_map(move |p| self.entry(p)))
    }

    /// Value ID of live property `key` on `parent`.
    pub(crate) fn prop(&self, parent: LogId, key: StrId) -> Option<StrId> {
        if !self.delta.props.is_empty()
            && let Ok(k) = sid(key)
            && let Some(&v) = self.delta.props.get(&(parent, k))
        {
            return v.map(StrId::from);
        }
        self.base.prop(parent, key)
    }

    /// Live properties of `parent` as `(key, value)` string IDs, in key-ID order (the
    /// order [`Txn::props`] yields them).
    #[must_use]
    pub fn props_of(&self, parent: LogId) -> Vec<(StrId, StrId)> {
        let Some(keys) = (!self.delta.keys.is_empty())
            .then(|| self.delta.keys.get(&parent))
            .flatten()
        else {
            return self.base.props_of(parent).collect();
        };
        let mut out: Vec<(StrId, StrId)> = self
            .base
            .props_of(parent)
            .filter(|&(k, _)| Sid::try_from(k).is_ok_and(|k| !keys.contains(&k)))
            .collect();
        for &k in keys {
            if let Some(Some(v)) = self.delta.props.get(&(parent, k)) {
                out.push((StrId::from(k), StrId::from(*v)));
            }
        }
        out.sort_unstable_by_key(|&(k, _)| k);
        out
    }

    /// Appends the endpoints of edge `edge` (row `row` if the caller knows it) for
    /// `dirs` (`In`: source, `Out`: target) to `out`, with their rows to `rows`.
    ///
    /// # Errors
    /// Fails if the edge is not in the projection.
    pub(crate) fn edge_ends(
        &self,
        edge: LogId,
        row: Option<Row>,
        dirs: &[Direction],
        out: &mut Vec<Entry>,
        rows: &mut Vec<Row>,
    ) -> Result<()> {
        let r = row
            .or_else(|| self.row_of(edge))
            .ok_or(GraphError::NotFound(edge, "edge"))?;
        if r < self.base_rows() {
            return self.base.edge_ends(edge, Some(r), dirs, out, rows);
        }
        let [src, tgt] = self
            .drow(r)
            .and_then(|d| d.ends)
            .ok_or(GraphError::NotFound(edge, "edge"))?;
        for &d in dirs {
            let n = if d == Direction::Out { tgt } else { src };
            out.push(n.entry());
            rows.push(n.row);
        }
        Ok(())
    }

    /// Appends the edges of `node` in direction `dir`, optionally of type `ty`, to
    /// `out` in the order [`Txn::node_edges`] yields them (inbound then outbound, each
    /// by `(type, id)`; a self-loop once).
    ///
    /// # Errors
    /// Fails on a corrupt projection.
    pub(crate) fn node_edges(
        &self,
        node: LogId,
        dir: Direction,
        ty: Option<StrId>,
        out: &mut Vec<Entry>,
        rows: &mut Vec<Row>,
    ) -> Result<()> {
        if self.delta.ops == 0 {
            return self.base.node_edges(node, dir, ty, out, rows);
        }
        let Some(n) = self.row_of(node) else {
            return Ok(());
        };
        let ty = match ty.map(sid) {
            None => None,
            Some(Ok(t)) => Some(t),
            Some(Err(_)) => return Ok(()), // no edge can have that type
        };
        if dir != Direction::Out {
            self.side((node, n), Direction::In, false, ty, (out, rows))?;
        }
        if dir != Direction::In {
            let skip_loops = dir == Direction::Both;
            self.side((node, n), Direction::Out, skip_loops, ty, (out, rows))?;
        }
        Ok(())
    }

    /// One side of [`node_edges`](Self::node_edges): the base's edges less the dead,
    /// then the delta's, kept in `(type, id)` order.
    ///
    /// # Errors
    /// Fails on a corrupt projection.
    fn side(
        &self,
        (node, n): (LogId, Row),
        side: Direction,
        skip_loops: bool,
        ty: Option<Sid>,
        (out, rows): (&mut Vec<Entry>, &mut Vec<Row>),
    ) -> Result<()> {
        let start = out.len();
        let b = self.base_rows();
        if n < b {
            self.base
                .node_edges(node, side, ty.map(StrId::from), out, rows)?;
        }
        let map = if side == Direction::In {
            &self.delta.inb
        } else {
            &self.delta.out
        };
        let added = (!map.is_empty())
            .then(|| map.get(&node))
            .flatten()
            .map_or(&[][..], Vec::as_slice);
        for &i in added {
            let r = b.saturating_add(i);
            if let Some(d) = self
                .drow(r)
                .filter(|d| !d.dead && ty.is_none_or(|t| d.obj.ty == t))
            {
                out.push(d.entry());
                rows.push(r);
            }
        }
        // Drop dead base edges, and self-loops a `Both` walk already listed inbound.
        if self.delta.dead.is_empty() && !skip_loops && added.is_empty() {
            return Ok(());
        }
        let mut keep = start;
        for j in start..out.len() {
            let (Some(&e), Some(&r)) = (out.get(j), rows.get(j)) else {
                continue;
            };
            let dead = r < b && self.base_dead(r);
            let is_loop =
                skip_loops && matches!(e.record, Record::Edge { src, tgt, .. } if src == tgt);
            if !dead && !is_loop {
                if let (Some(o), Some(w)) = (out.get_mut(keep), rows.get_mut(keep)) {
                    *o = e;
                    *w = r;
                }
                keep += 1;
            }
        }
        out.truncate(keep);
        rows.truncate(keep);
        if !added.is_empty() {
            // Delta edges have the highest IDs; restore (type, id) order across both.
            let mut seg: Vec<(Entry, Row)> = out.drain(start..).zip(rows.drain(start..)).collect();
            seg.sort_by_key(|(e, _)| (edge_ty(e), e.id));
            for (e, r) in seg {
                out.push(e);
                rows.push(r);
            }
        }
        Ok(())
    }

    /// Applies the log entries after this projection's view, up to `t`'s view.
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt log.
    pub(crate) fn advance(&mut self, t: &Txn<'_>) -> Result<()> {
        let end = t.next_id()?;
        for e in t.log(self.end, Some(end))? {
            let e = e?;
            self.apply(t, &e)?;
            self.delta.ops += 1;
        }
        self.end = end;
        Ok(())
    }

    /// # Errors
    /// Fails on storage errors or a corrupt log.
    fn apply(&mut self, t: &Txn<'_>, e: &Entry) -> Result<()> {
        match e.record {
            Record::Node { ty, val } => {
                let obj = Obj {
                    id: e.id,
                    ty: sid(ty)?,
                    val: sid(val)?,
                };
                self.push_row(obj, None)?;
            }
            Record::Edge { ty, val, src, tgt } => {
                let obj = Obj {
                    id: e.id,
                    ty: sid(ty)?,
                    val: sid(val)?,
                };
                let ends = [self.node_end(src)?, self.node_end(tgt)?];
                let i = self.push_row(obj, Some(ends))?;
                self.delta.out.entry(src).or_default().push(i);
                self.delta.inb.entry(tgt).or_default().push(i);
            }
            Record::Prop { parent, key, val } => self.set_prop(parent, sid(key)?, Some(sid(val)?)),
            Record::Deletion { target } => self.kill(t, target)?,
        }
        Ok(())
    }

    /// # Errors
    /// Fails if the projection would need more rows than [`Row`] can index.
    fn push_row(&mut self, obj: Obj, ends: Option<[End; 2]>) -> Result<u32> {
        let i = row(self.delta.rows.len())?;
        self.base_rows()
            .checked_add(i)
            .ok_or(GraphError::Corrupt("projection has too many rows"))?;
        self.delta.slot.insert(obj.id, i);
        self.delta
            .by_type
            .entry((ends.is_some(), obj.ty))
            .or_default()
            .push(i);
        self.delta.rows.push(DRow {
            obj,
            ends,
            dead: false,
        });
        Ok(i)
    }

    /// # Errors
    /// Fails if `id` is not a live node.
    fn node_end(&self, id: LogId) -> Result<End> {
        let bad = || GraphError::Corrupt("live edge on a dead node");
        let r = self.row_of(id).ok_or_else(bad)?;
        if r < self.base_rows() {
            return self.base.node_end(r).ok_or_else(bad);
        }
        let d = self.drow(r).filter(|d| d.ends.is_none()).ok_or_else(bad)?;
        Ok(End {
            row: r,
            id: d.obj.id,
            ty: d.obj.ty,
            val: d.obj.val,
        })
    }

    fn set_prop(&mut self, parent: LogId, key: Sid, val: Option<Sid>) {
        self.delta.props.insert((parent, key), val);
        let keys = self.delta.keys.entry(parent).or_default();
        if !keys.contains(&key) {
            keys.push(key);
        }
        if let Some(v) = val
            && self.delta.kv_seen.insert((key, v, parent))
        {
            self.delta.kv.entry((key, v)).or_default().push(parent);
        }
    }

    /// Ends `target` and, for a node, its edges (their properties go with them).
    ///
    /// # Errors
    /// Fails on storage errors.
    fn kill(&mut self, t: &Txn<'_>, target: LogId) -> Result<()> {
        let Some(r) = self.row_of(target) else {
            // A property: forget its value.
            if let Some(Entry {
                record: Record::Prop { parent, key, .. },
                ..
            }) = t.entry(target)?
            {
                self.set_prop(parent, sid(key)?, None);
            }
            return Ok(());
        };
        if self
            .row(r)
            .is_some_and(|e| matches!(e.record, Record::Node { .. }))
        {
            let b = self.base_rows();
            let mut edges: Vec<Row> = Vec::new();
            if r < b {
                edges.extend(self.base.edge_rows(r));
            }
            for list in [self.delta.out.get(&target), self.delta.inb.get(&target)]
                .into_iter()
                .flatten()
            {
                edges.extend(list.iter().map(|&i| b.saturating_add(i)));
            }
            for e in edges {
                self.mark_dead(e);
            }
        }
        self.mark_dead(r);
        Ok(())
    }

    fn mark_dead(&mut self, r: Row) {
        let b = self.base_rows();
        if r < b {
            if self.delta.dead.is_empty() {
                self.delta.dead = vec![0; index(b) / 64 + 1];
            }
            if let Some(w) = self.delta.dead.get_mut(index(r / 64)) {
                let bit = 1_u64 << (r % 64);
                if *w & bit == 0 {
                    *w |= bit;
                    self.delta.deaths += 1;
                }
            }
        } else if let Some(d) = self.delta.rows.get_mut(index(r - b))
            && !d.dead
        {
            d.dead = true;
            self.delta.deaths += 1;
        }
    }

    /// Chains of `patterns` in this view, as [`Txn::query`] would yield them.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn query(
        &self,
        t: &Txn<'_>,
        patterns: &[crate::lgql::Pattern],
        sink: impl FnMut(usize, &[Entry]) -> bool,
    ) -> Result<()> {
        t.query_with(Some(self), patterns, None, sink)
    }

    /// [`query`](Self::query) over `threads` threads: seeds are dealt round-robin, each
    /// thread expands its share in its own read txn, and `sink(thread, pattern, chain)`
    /// is called from all of them (so chains arrive in no particular order; `thread`
    /// lets a caller keep per-thread state). Returning `false` stops every thread soon
    /// after.
    ///
    /// # Errors
    /// Fails on storage errors, or with [`GraphError::Corrupt`] if the graph has been
    /// written since this projection was built (a read txn would see a different view).
    pub fn query_par(
        &self,
        g: &Graph,
        patterns: &[crate::lgql::Pattern],
        threads: usize,
        sink: impl Fn(usize, usize, &[Entry]) -> bool + Sync,
    ) -> Result<()> {
        let threads = threads.max(1);
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|k| {
                    let (stop, sink) = (&stop, &sink);
                    scope.spawn(move || -> Result<()> {
                        let t = g.read()?;
                        if t.next_id()? != self.end {
                            return Err(GraphError::Corrupt("projection is stale"));
                        }
                        t.run(
                            Some(self),
                            patterns,
                            None,
                            Part { k, n: threads },
                            &mut |pi, chain| {
                                if stop.load(Ordering::Relaxed) {
                                    return false;
                                }
                                let go = sink(k, pi, chain);
                                if !go {
                                    stop.store(true, Ordering::Relaxed);
                                }
                                go
                            },
                        )
                    })
                })
                .collect();
            for w in workers {
                w.join()
                    .map_err(|_| GraphError::Corrupt("query thread panicked"))??;
            }
            Ok(())
        })
    }
}
