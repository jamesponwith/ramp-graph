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

use std::sync::atomic::{AtomicBool, Ordering};

use crate::query::Part;
use crate::{Direction, Entry, Graph, GraphError, LogId, Record, Result, StrId, Txn};

/// Row index into [`Projection::rows`].
type Row = u32;

/// A read-optimised copy of one committed view, for [`Txn::query`].
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
pub struct Projection {
    /// The view: log entries `< end`.
    end: LogId,
    /// Log ID → row + 1 (0: not a live node or edge in this view). Indexed up to `end`.
    slot: Vec<Row>,
    /// Live nodes and edges, in log order.
    rows: Vec<Entry>,
    /// Node rows sorted by `(type, row)`.
    nodes_by_type: Vec<(StrId, Row)>,
    /// Edge rows sorted by `(type, row)`.
    edges_by_type: Vec<(StrId, Row)>,
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
    by_kv: Vec<(StrId, StrId, LogId)>,
    /// Live properties as `(parent, key, value)`, sorted; one per `(parent, key)`.
    by_pk: Vec<(LogId, StrId, StrId)>,
}

/// # Errors
/// Fails if the projection would need more rows than [`Row`] can index.
/// A node as an edge's endpoint: its row and enough to rebuild its [`Entry`].
#[derive(Debug, Clone, Copy, Default)]
struct End {
    row: Row,
    id: LogId,
    next: LogId,
    ty: StrId,
    val: StrId,
}

impl End {
    const fn entry(self) -> Entry {
        Entry {
            id: self.id,
            next: self.next,
            record: Record::Node {
                ty: self.ty,
                val: self.val,
            },
        }
    }
}

/// `end_idx` of a node row.
const NO_END: u32 = u32::MAX;

/// # Errors
/// Fails if the projection would need more rows than [`Row`] can index.
fn row(i: usize) -> Result<Row> {
    Row::try_from(i).map_err(|_| GraphError::Corrupt("projection has too many rows"))
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

/// CSR adjacency `[out_off, out_adj, in_off, in_adj]` over `rows`, each node's list in
/// `(edge type, edge id)` order.
///
/// # Errors
/// Fails if a live edge references a node that is not live.
fn csr(
    rows: &[Entry],
    edges_by_type: &[(StrId, Row)],
    by_row: impl Fn(LogId) -> Option<Row>,
) -> Result<[Vec<u32>; 4]> {
    let mut out_deg = vec![0_u32; rows.len() + 1];
    let mut in_deg = vec![0_u32; rows.len() + 1];
    for e in rows {
        if let Record::Edge { src, tgt, .. } = e.record {
            let (Some(s), Some(d)) = (by_row(src), by_row(tgt)) else {
                return Err(GraphError::Corrupt("live edge on a dead node"));
            };
            if let Some(c) = out_deg.get_mut(index(s)) {
                *c += 1;
            }
            if let Some(c) = in_deg.get_mut(index(d)) {
                *c += 1;
            }
        }
    }
    let out_off = offsets(&out_deg);
    let in_off = offsets(&in_deg);
    let mut out_adj = vec![0; index(out_off.last().copied().unwrap_or(0))];
    let mut in_adj = vec![0; index(in_off.last().copied().unwrap_or(0))];
    let mut out_cur = out_off.clone();
    let mut in_cur = in_off.clone();
    // Walking edges in (type, id) order fills each node's list in that order.
    for &(_, r) in edges_by_type {
        let Some(Entry {
            record: Record::Edge { src, tgt, .. },
            ..
        }) = rows.get(index(r))
        else {
            continue;
        };
        for (end_id, cur, adj) in [
            (src, &mut out_cur, &mut out_adj),
            (tgt, &mut in_cur, &mut in_adj),
        ] {
            let Some(n) = by_row(*end_id) else { continue };
            if let Some(c) = cur.get_mut(index(n)) {
                if let Some(a) = adj.get_mut(index(*c)) {
                    *a = r;
                }
                *c += 1;
            }
        }
    }
    Ok([out_off, out_adj, in_off, in_adj])
}

impl Projection {
    /// Projects the view of `t` (everything committed before it started, or `t`'s own
    /// writes too in a write txn).
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt log.
    pub fn build(t: &Txn<'_>) -> Result<Self> {
        let end = t.next_id()?;
        let view = Some(end);
        let mut slot =
            vec![0; usize::try_from(end).map_err(|_| GraphError::Corrupt("log too long"))?];
        let mut rows: Vec<Entry> = Vec::new();
        let mut props: Vec<(LogId, StrId, StrId)> = Vec::new();
        for e in t.log(1, None)? {
            let e = e?;
            if !e.live_at(view) {
                continue;
            }
            match e.record {
                Record::Node { .. } | Record::Edge { .. } => {
                    let r = row(rows.len())?;
                    if let Some(s) = slot.get_mut(usize::try_from(e.id).unwrap_or(usize::MAX)) {
                        *s = r + 1;
                    }
                    rows.push(e);
                }
                Record::Prop { parent, key, val } => props.push((parent, key, val)),
                Record::Deletion { .. } => {}
            }
        }
        let by_row = |id: LogId| -> Option<Row> {
            let s = *slot.get(usize::try_from(id).ok()?)?;
            s.checked_sub(1)
        };
        let mut nodes_by_type = Vec::new();
        let mut edges_by_type = Vec::new();
        for (i, e) in rows.iter().enumerate() {
            match e.record {
                Record::Node { ty, .. } => nodes_by_type.push((ty, row(i)?)),
                Record::Edge { ty, .. } => edges_by_type.push((ty, row(i)?)),
                Record::Prop { .. } | Record::Deletion { .. } => {}
            }
        }
        nodes_by_type.sort_unstable();
        edges_by_type.sort_unstable();
        let [out_off, out_adj, in_off, in_adj] = csr(&rows, &edges_by_type, by_row)?;
        let end_of = |id: LogId| -> Result<End> {
            let r = by_row(id).ok_or(GraphError::Corrupt("live edge on a dead node"))?;
            match rows.get(index(r)) {
                Some(&Entry {
                    id,
                    next,
                    record: Record::Node { ty, val },
                }) => Ok(End {
                    row: r,
                    id,
                    next,
                    ty,
                    val,
                }),
                _ => Err(GraphError::Corrupt("edge endpoint is not a node")),
            }
        };
        let mut end_idx = Vec::with_capacity(rows.len());
        let mut ends = Vec::new();
        for e in &rows {
            if let Record::Edge { src, tgt, .. } = e.record {
                end_idx.push(row(ends.len())?);
                ends.push([end_of(src)?, end_of(tgt)?]);
            } else {
                end_idx.push(NO_END);
            }
        }

        let mut by_kv: Vec<(StrId, StrId, LogId)> =
            props.iter().map(|&(p, k, v)| (k, v, p)).collect();
        by_kv.sort_unstable();
        let mut by_pk = props;
        by_pk.sort_unstable();
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
        })
    }

    /// The view this projection projects: log entries below it.
    #[must_use]
    pub const fn end(&self) -> LogId {
        self.end
    }

    /// Live nodes and edges, in log order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.rows
    }

    /// The live node or edge with this ID.
    #[must_use]
    pub fn entry(&self, id: LogId) -> Option<Entry> {
        let s = *self.slot.get(usize::try_from(id).ok()?)?;
        self.rows.get(index(s.checked_sub(1)?)).copied()
    }

    /// # Errors
    /// Fails on a row index the projection does not have.
    fn by_row(&self, r: Row) -> Result<Entry> {
        self.rows
            .get(index(r))
            .copied()
            .ok_or(GraphError::Corrupt("projection row out of range"))
    }

    /// Rows of live nodes (`edges == false`) or edges of type `ty`, in log order.
    pub(crate) fn type_group(&self, edges: bool, ty: StrId) -> &[(StrId, Row)] {
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

    /// The entry in row `r`.
    #[must_use]
    pub(crate) fn row(&self, r: Row) -> Option<Entry> {
        self.rows.get(index(r)).copied()
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
        let Some(n) = self
            .slot
            .get(usize::try_from(node).ok().unwrap_or(usize::MAX))
            .and_then(|s| s.checked_sub(1))
        else {
            return Ok(());
        };
        let both = dir == Direction::Both;
        if dir != Direction::Out {
            for &r in adjacency(n, &self.in_off, &self.in_adj) {
                let e = self.by_row(r)?;
                if ty.is_none_or(|t| matches!(e.record, Record::Edge { ty, .. } if ty == t)) {
                    out.push(e);
                    rows.push(r);
                }
            }
        }
        if dir != Direction::In {
            for &r in adjacency(n, &self.out_off, &self.out_adj) {
                let e = self.by_row(r)?;
                let is_loop = matches!(e.record, Record::Edge { src, tgt, .. } if src == tgt);
                if !(both && is_loop)
                    && ty.is_none_or(|t| matches!(e.record, Record::Edge { ty, .. } if ty == t))
                {
                    out.push(e);
                    rows.push(r);
                }
            }
        }
        Ok(())
    }

    /// Value ID of live property `key` on `parent`.
    #[must_use]
    pub(crate) fn prop(&self, parent: LogId, key: StrId) -> Option<StrId> {
        let i = self
            .by_pk
            .partition_point(|&(p, k, _)| (p, k) < (parent, key));
        self.by_pk
            .get(i)
            .filter(|&&(p, k, _)| p == parent && k == key)
            .map(|&(_, _, v)| v)
    }

    /// Live properties `key == val` as `(key, val, parent)`, in parent order.
    pub(crate) fn kv_range(&self, key: StrId, val: StrId) -> &[(StrId, StrId, LogId)] {
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
    pub const fn bytes(&self) -> usize {
        self.slot.len() * size_of::<Row>()
            + self.rows.len() * size_of::<Entry>()
            + (self.nodes_by_type.len() + self.edges_by_type.len()) * size_of::<(StrId, Row)>()
            + (self.out_off.len() + self.in_off.len()) * size_of::<u32>()
            + (self.out_adj.len() + self.in_adj.len()) * size_of::<Row>()
            + self.end_idx.len() * size_of::<u32>()
            + self.ends.len() * size_of::<[End; 2]>()
            + self.by_kv.len() * size_of::<(StrId, StrId, LogId)>()
            + self.by_pk.len() * size_of::<(LogId, StrId, StrId)>()
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
