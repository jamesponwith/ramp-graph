//! Log-structured transactional graph database.
//!
//! Every change (new node, edge, property, or deletion) is appended to a log and gets
//! the next [`LogId`]. Records are never removed: deleting or superseding one stamps its
//! `next` field with the ID of the record that ended it. That makes every past state
//! queryable — pass `before: Some(id)` to see the graph as it was before log entry `id`.
//!
//! Types, values, and property keys/values are byte strings interned once into
//! [`StrId`]s; the empty string is always [`StrId`] 0.
//!
//! ```
//! # fn main() -> ramp_graph::Result<()> {
//! let dir = tempfile::tempdir().map_err(|e| ramp_graph::GraphError::Io(e.to_string()))?;
//! let g = ramp_graph::Graph::open(dir.path().join("g.db"))?;
//! let mut txn = g.write()?;
//! let a = txn.node(b"person", b"alice")?;
//! let b = txn.node(b"person", b"bob")?;
//! txn.edge(a.id, b.id, b"knows", b"")?;
//! txn.commit()?;
//! assert_eq!(g.read()?.counts(None)?, (2, 1));
//! # Ok(())
//! # }
//! ```

pub mod lgql;
mod query;
pub mod value;
mod varint;

use std::ops::Bound;
use std::path::Path;
use std::sync::{Condvar, Mutex};

use heed::types::Bytes;
use heed::{
    CompactionOption, Database, Env, EnvFlags, EnvOpenOptions, PutFlags, RoTxn, RwTxn, WithTls,
};

/// Position in the graph log. IDs start at 1; 0 means "none" (or "the graph itself"
/// when used as a property parent).
pub type LogId = u64;

/// ID of an interned byte string. 0 is the empty string.
pub type StrId = u64;

/// Everything that can go wrong.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GraphError {
    /// The storage layer failed.
    #[error(transparent)]
    Lmdb(#[from] heed::Error),
    /// Filesystem error outside LMDB.
    #[error("io: {0}")]
    Io(String),
    /// The file holds bytes this code did not write.
    #[error("corrupt database: {0}")]
    Corrupt(&'static str),
    /// No live entry of the expected kind has this ID.
    #[error("no live {1} with id {0}")]
    NotFound(LogId, &'static str),
    /// A property value could not be encoded or decoded.
    #[error("value: {0}")]
    Value(String),
    /// A write was attempted in a read transaction.
    #[error("transaction is read-only")]
    ReadOnly,
}

/// Result alias for this crate.
pub type Result<T, E = GraphError> = std::result::Result<T, E>;

/// What a log entry records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Record {
    /// Ends `target` and, by cascade, its properties (and a node's edges).
    Deletion {
        /// The entry that was deleted.
        target: LogId,
    },
    /// A node, unique among live nodes by `(ty, val)`.
    Node {
        /// Node type.
        ty: StrId,
        /// Node value.
        val: StrId,
    },
    /// A directed edge, unique among live edges by `(src, tgt, ty, val)`.
    Edge {
        /// Edge type.
        ty: StrId,
        /// Edge value.
        val: StrId,
        /// Source node.
        src: LogId,
        /// Target node.
        tgt: LogId,
    },
    /// A property on the graph (`parent == 0`), a node, an edge, or another property.
    Prop {
        /// Owning entry.
        parent: LogId,
        /// Property key.
        key: StrId,
        /// Property value.
        val: StrId,
    },
}

impl Record {
    const fn tag(&self) -> u8 {
        match self {
            Self::Deletion { .. } => 0,
            Self::Node { .. } => 1,
            Self::Edge { .. } => 2,
            Self::Prop { .. } => 3,
        }
    }

    const fn kind(&self) -> &'static str {
        match self {
            Self::Deletion { .. } => "deletion",
            Self::Node { .. } => "node",
            Self::Edge { .. } => "edge",
            Self::Prop { .. } => "property",
        }
    }
}

/// One log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// Position in the log.
    pub id: LogId,
    /// ID of the entry that deleted or superseded this one, or 0 while live.
    pub next: LogId,
    /// Payload.
    pub record: Record,
}

impl Entry {
    /// Whether this entry is visible in the view `before` (see [`Txn::counts`]).
    #[must_use]
    pub fn live_at(&self, before: Option<LogId>) -> bool {
        self.next == 0 || before.is_some_and(|b| self.next >= b)
    }

    fn encode(&self) -> Vec<u8> {
        let fields: &[u64] = match self.record {
            Record::Deletion { target } => &[target],
            Record::Node { ty, val } => &[ty, val],
            Record::Edge { ty, val, src, tgt } => &[ty, val, src, tgt],
            Record::Prop { parent, key, val } => &[parent, key, val],
        };
        let mut buf = vec![self.record.tag()];
        varint::push(&mut buf, self.next);
        for &f in fields {
            varint::push(&mut buf, f);
        }
        buf
    }

    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn decode(id: LogId, bytes: &[u8]) -> Result<Self> {
        let (&tag, mut rest) = bytes
            .split_first()
            .ok_or(GraphError::Corrupt("empty log record"))?;
        let next = varint::take(&mut rest)?;
        let mut f = || varint::take(&mut rest);
        let record = match tag {
            0 => Record::Deletion { target: f()? },
            1 => Record::Node {
                ty: f()?,
                val: f()?,
            },
            2 => Record::Edge {
                ty: f()?,
                val: f()?,
                src: f()?,
                tgt: f()?,
            },
            3 => Record::Prop {
                parent: f()?,
                key: f()?,
                val: f()?,
            },
            _ => return Err(GraphError::Corrupt("unknown log record type")),
        };
        Ok(Self { id, next, record })
    }
}

/// Which edges of a node to visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Edges whose target is the node.
    In,
    /// Edges whose source is the node.
    Out,
    /// Both; a self-loop is yielded once.
    Both,
}

type Table = Database<Bytes, Bytes>;

/// All tables. Keys are [`varint`] tuples unless noted; index values are empty.
#[derive(Debug, Clone, Copy)]
struct Tables {
    /// `id` → encoded [`Entry`] minus its id.
    log: Table,
    /// `StrId` (big-endian u64) → bytes.
    scalar: Table,
    /// `fnv64(bytes) ++ StrId` (both big-endian) → ''.
    scalar_idx: Table,
    /// `[ty, val, id]`.
    node_idx: Table,
    /// `[ty, val, src, tgt, id]`.
    edge_idx: Table,
    /// `[parent, key, id]`.
    prop_idx: Table,
    /// `[src, ty, edge]`.
    src_idx: Table,
    /// `[tgt, ty, edge]`.
    tgt_idx: Table,
    /// `[end]` → `[nodes, edges]`: live counts after each write txn, `end` = first ID after it.
    txnlog: Table,
    /// `[domain] ++ key` → value.
    kv: Table,
}

/// Address space mapped past the end of the file. The map grows before a write txn that
/// would have less, so one write txn can add at most this much data (as upstream).
const PAD: usize = if cfg!(test) { 1 << 20 } else { 1 << 30 };

/// Map size for a file of `len` bytes: whole pads, at least one pad past the end.
fn map_size(len: u64) -> usize {
    usize::try_from(len)
        .unwrap_or(usize::MAX)
        .div_ceil(PAD)
        .saturating_add(1)
        .saturating_mul(PAD)
}

/// A graph stored in one LMDB file (plus a `-lock` file beside it).
///
/// The file must only be modified through LMDB: truncating or rewriting it while it is
/// open is undefined behavior.
#[derive(Debug)]
pub struct Graph {
    env: Env,
    t: Tables,
    /// Txns active in this process, which LMDB requires to be zero to resize the map.
    txns: Mutex<Active>,
    idle: Condvar,
}

/// See [`Graph::txns`] (upstream `db_t.txns` in `lib/db.c`).
#[derive(Debug, Default)]
struct Active {
    count: usize,
    /// A resize is waiting: new txns hold off so it cannot starve.
    resizing: bool,
}

/// Counts one active txn; dropping it (after the LMDB txn has ended) uncounts it.
#[derive(Debug)]
struct Ticket<'a>(&'a Graph);

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        if let Ok(mut a) = self.0.txns.lock() {
            a.count = a.count.saturating_sub(1);
            if a.count == 0 {
                self.0.idle.notify_all();
            }
        }
    }
}

const POISONED: GraphError = GraphError::Corrupt("txn counter poisoned");

impl Graph {
    /// Opens the graph at `path`, creating it if missing.
    ///
    /// # Errors
    /// Fails if the file cannot be opened or is not an LMDB file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let len = std::fs::metadata(path.as_ref()).map_or(0, |m| m.len());
        let mut opts = EnvOpenOptions::new();
        opts.map_size(map_size(len)).max_dbs(10);
        // SAFETY: NO_SUB_DIR only selects the single-file layout.
        unsafe { opts.flags(EnvFlags::NO_SUB_DIR) };
        // SAFETY: the file is only modified through LMDB, whose lock file arbitrates between
        // processes; `Graph`'s docs forbid modifying it any other way.
        let env = unsafe { opts.open(path) }?;
        let mut w = env.write_txn()?;
        let mut table = |name| env.create_database::<Bytes, Bytes>(&mut w, Some(name));
        let t = Tables {
            log: table("log")?,
            scalar: table("scalar")?,
            scalar_idx: table("scalar_idx")?,
            node_idx: table("node_idx")?,
            edge_idx: table("edge_idx")?,
            prop_idx: table("prop_idx")?,
            src_idx: table("srcnode_idx")?,
            tgt_idx: table("tgtnode_idx")?,
            txnlog: table("txnlog")?,
            kv: table("kv")?,
        };
        w.commit()?;
        Ok(Self {
            env,
            t,
            txns: Mutex::default(),
            idle: Condvar::new(),
        })
    }

    /// Registers a txn about to start, waiting out any resize.
    ///
    /// # Errors
    /// Fails if a thread panicked while holding the counter.
    fn ticket(&self) -> Result<Ticket<'_>> {
        let mut a = self.txns.lock().map_err(|_| POISONED)?;
        while a.resizing {
            a = self.idle.wait(a).map_err(|_| POISONED)?;
        }
        a.count += 1;
        drop(a);
        Ok(Ticket(self))
    }

    /// Grows the map if less than [`PAD`] is left past the end of the file.
    ///
    /// # Errors
    /// Fails if the file cannot be stat'ed or the map cannot grow.
    fn reserve(&self) -> Result<()> {
        // `Some(new size)` when less than a pad is free past the end of the file.
        let short = |g: &Self| -> Result<Option<usize>> {
            let len = g.size()?;
            let end = usize::try_from(len)
                .unwrap_or(usize::MAX)
                .saturating_add(PAD);
            Ok((g.env.info().map_size < end).then(|| map_size(len)))
        };
        if short(self)?.is_none() {
            return Ok(());
        }
        let mut a = self.txns.lock().map_err(|_| POISONED)?;
        while a.resizing {
            a = self.idle.wait(a).map_err(|_| POISONED)?; // another thread is on it
        }
        a.resizing = true;
        while a.count > 0 {
            a = self.idle.wait(a).map_err(|_| POISONED)?;
        }
        let grown = short(self).and_then(|want| match want {
            // SAFETY: `count` is zero and `resizing` keeps new txns from starting, and every
            // txn holds a `Ticket` for its whole life: no txn of this env is active in the process.
            Some(want) => unsafe { self.env.resize(want) }.map_err(GraphError::from),
            None => Ok(()),
        });
        a.resizing = false;
        self.idle.notify_all();
        drop(a);
        grown
    }

    /// Starts a read transaction: a consistent snapshot that never blocks writers.
    ///
    /// # Errors
    /// Fails if LMDB is out of reader slots.
    pub fn read(&self) -> Result<Txn<'_>> {
        let ticket = self.ticket()?;
        Ok(Txn {
            g: self,
            inner: Inner::Ro(self.env.read_txn()?),
            state: State::default(),
            parent: None,
            _ticket: Some(ticket),
        })
    }

    /// Starts a write transaction. Blocks while another write transaction is open, and,
    /// when the map must grow first, until this process has no other txn on the graph
    /// (so do not call it while this thread holds one).
    ///
    /// # Errors
    /// Fails if LMDB cannot start the transaction or grow the map.
    pub fn write(&self) -> Result<Txn<'_>> {
        self.reserve()?;
        let ticket = self.ticket()?;
        let mut txn = Txn {
            g: self,
            inner: Inner::Rw(self.env.write_txn()?),
            state: State::default(),
            parent: None,
            _ticket: Some(ticket),
        };
        txn.state.begin = txn.next_id()?;
        txn.state.next_log = txn.state.begin;
        Ok(txn)
    }

    /// Flushes committed data to disk.
    ///
    /// # Errors
    /// Fails if the sync fails.
    pub fn sync(&self) -> Result<()> {
        Ok(self.env.force_sync()?)
    }

    /// Size of the database file in bytes.
    ///
    /// # Errors
    /// Fails if the file cannot be stat'ed.
    pub fn size(&self) -> Result<u64> {
        Ok(self.env.real_disk_size()?)
    }

    /// Writes a consistent, compacted copy of the graph to a new file at `path`.
    ///
    /// # Errors
    /// Fails if `path` exists or cannot be written.
    pub fn snapshot(&self, path: impl AsRef<Path>) -> Result<()> {
        let ticket = self.ticket()?; // the copy runs its own read txn
        self.env.copy_to_path(path, CompactionOption::Enabled)?;
        drop(ticket);
        Ok(())
    }
}

/// Per-write-txn bookkeeping, copied into the parent when a nested txn commits.
#[derive(Debug, Clone, Copy, Default)]
struct State {
    /// First log ID of the top-level txn.
    begin: LogId,
    node_delta: i64,
    edge_delta: i64,
    /// Next log ID (write txns only; 0 = not yet known).
    next_log: LogId,
    /// Next string ID (0 = not yet known).
    next_str: StrId,
}

enum Inner<'a> {
    Ro(RoTxn<'a, WithTls>),
    Rw(RwTxn<'a>),
}

/// A read or write transaction. Read methods take `&self`, writes `&mut self`.
///
/// Dropping a transaction without [`commit`](Self::commit) aborts it.
pub struct Txn<'a> {
    g: &'a Graph,
    inner: Inner<'a>,
    state: State,
    parent: Option<&'a mut State>,
    /// Held until the LMDB txn has ended (fields drop in order); nested txns ride on
    /// their parent's.
    _ticket: Option<Ticket<'a>>,
}

impl std::fmt::Debug for Txn<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Txn")
            .field("write", &matches!(self.inner, Inner::Rw(_)))
            .field("nested", &self.parent.is_some())
            .finish_non_exhaustive()
    }
}

/// An iterator of live entries.
pub trait Entries: Iterator<Item = Result<Entry>> {}
impl<I: Iterator<Item = Result<Entry>>> Entries for I {}

impl<'a> Txn<'a> {
    fn ro(&self) -> &RoTxn<'a> {
        match &self.inner {
            Inner::Ro(t) => t,
            Inner::Rw(t) => t,
        }
    }

    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    const fn rw(&mut self) -> Result<&mut RwTxn<'a>> {
        match &mut self.inner {
            Inner::Ro(_) => Err(GraphError::ReadOnly),
            Inner::Rw(t) => Ok(t),
        }
    }

    // ── Reads ──

    /// The ID the next log entry will get.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn next_id(&self) -> Result<LogId> {
        if self.state.next_log != 0 {
            return Ok(self.state.next_log);
        }
        match self.g.t.log.last(self.ro())? {
            Some((k, _)) => varint::last(k)?
                .checked_add(1)
                .ok_or(GraphError::Corrupt("log id overflow")),
            None => Ok(1),
        }
    }

    /// Normalizes a view bound: `None`, or one that covers the whole log, means "now".
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn view(&self, before: Option<LogId>) -> Result<Option<LogId>> {
        Ok(match before {
            Some(b) if b < self.next_id()? => Some(b),
            _ => None,
        })
    }

    /// Fetches any log entry by ID, live or not.
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt record.
    pub fn entry(&self, id: LogId) -> Result<Option<Entry>> {
        self.g
            .t
            .log
            .get(self.ro(), &varint::pack(&[id]))?
            .map(|b| Entry::decode(id, b))
            .transpose()
    }

    /// Fetches a live entry and checks its kind (`"node"`, `"entry"` for any non-deletion).
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn live(&self, id: LogId, kind: &'static str) -> Result<Entry> {
        self.entry(id)?
            .filter(|e| {
                e.next == 0
                    && match kind {
                        "entry" => !matches!(e.record, Record::Deletion { .. }),
                        k => e.record.kind() == k,
                    }
            })
            .ok_or(GraphError::NotFound(id, kind))
    }

    /// Bytes of an interned string.
    ///
    /// # Errors
    /// Fails if `id` was never interned.
    pub fn string(&self, id: StrId) -> Result<&[u8]> {
        if id == 0 {
            return Ok(b"");
        }
        self.g
            .t
            .scalar
            .get(self.ro(), &id.to_be_bytes())?
            .ok_or(GraphError::Corrupt("dangling string id"))
    }

    /// The ID of an already-interned string.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn string_id(&self, bytes: &[u8]) -> Result<Option<StrId>> {
        if bytes.is_empty() {
            return Ok(Some(0));
        }
        for kv in self
            .g
            .t
            .scalar_idx
            .prefix_iter(self.ro(), &fnv64(bytes).to_be_bytes())?
        {
            let (k, _) = kv?;
            let id = k
                .get(8..)
                .and_then(|s| <[u8; 8]>::try_from(s).ok())
                .map(u64::from_be_bytes)
                .ok_or(GraphError::Corrupt("bad string index key"))?;
            if self.string(id)? == bytes {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    /// Latest entry in `table` under `prefix` visible in view `before`.
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn lookup(&self, table: Table, prefix: &[u8], before: Option<LogId>) -> Result<Option<Entry>> {
        let before = self.view(before)?;
        let mut upper = prefix.to_vec();
        match before {
            Some(b) => varint::push(&mut upper, b),
            None => upper.push(0xff), // sorts after every length byte
        }
        let Some((k, _)) = table.get_lower_than(self.ro(), &upper)? else {
            return Ok(None);
        };
        if !k.starts_with(prefix) {
            return Ok(None);
        }
        let id = varint::last(k)?;
        let e = self
            .entry(id)?
            .ok_or(GraphError::Corrupt("index points past log"))?;
        Ok(e.live_at(before).then_some(e))
    }

    /// Live entries indexed in `table` under `prefix` in view `before`.
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn scan(
        &self,
        table: Table,
        prefix: Vec<u8>,
        before: Option<LogId>,
    ) -> Result<impl Entries + '_> {
        let before = self.view(before)?;
        // LMDB rejects empty seek keys, so an empty prefix scans from the start.
        let lo = if prefix.is_empty() {
            Bound::Unbounded
        } else {
            Bound::Included(prefix.as_slice())
        };
        let iter = table.range(self.ro(), &(lo, Bound::Unbounded))?;
        Ok(iter
            .take_while(move |kv| {
                kv.as_ref().is_ok_and(|(k, _)| k.starts_with(&prefix)) || kv.is_err()
            })
            .filter_map(move |kv| {
                let found: Result<Option<Entry>> = (|| {
                    let id = varint::last(kv?.0)?;
                    if before.is_some_and(|b| id >= b) {
                        return Ok(None);
                    }
                    let e = self
                        .entry(id)?
                        .ok_or(GraphError::Corrupt("index points past log"))?;
                    Ok(e.live_at(before).then_some(e))
                })();
                found.transpose()
            }))
    }

    /// The node `(ty, val)` in view `before`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn node_lookup(
        &self,
        ty: &[u8],
        val: &[u8],
        before: Option<LogId>,
    ) -> Result<Option<Entry>> {
        let (Some(ty), Some(val)) = (self.string_id(ty)?, self.string_id(val)?) else {
            return Ok(None);
        };
        self.lookup(self.g.t.node_idx, &varint::pack(&[ty, val]), before)
    }

    /// The edge `src -[ty, val]-> tgt` in view `before`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn edge_lookup(
        &self,
        src: LogId,
        tgt: LogId,
        ty: &[u8],
        val: &[u8],
        before: Option<LogId>,
    ) -> Result<Option<Entry>> {
        let (Some(ty), Some(val)) = (self.string_id(ty)?, self.string_id(val)?) else {
            return Ok(None);
        };
        self.lookup(
            self.g.t.edge_idx,
            &varint::pack(&[ty, val, src, tgt]),
            before,
        )
    }

    /// Property `key` of `parent` (0 = the graph) in view `before`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn prop(&self, parent: LogId, key: &[u8], before: Option<LogId>) -> Result<Option<Entry>> {
        let Some(key) = self.string_id(key)? else {
            return Ok(None);
        };
        self.lookup(self.g.t.prop_idx, &varint::pack(&[parent, key]), before)
    }

    /// Properties of `parent` (0 = the graph), ordered by key ID.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn props(&self, parent: LogId, before: Option<LogId>) -> Result<impl Entries + '_> {
        self.scan(self.g.t.prop_idx, varint::pack(&[parent]), before)
    }

    /// All nodes, or those of type `ty`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn nodes(&self, ty: Option<&[u8]>, before: Option<LogId>) -> Result<impl Entries + '_> {
        self.by_type(self.g.t.node_idx, ty, before)
    }

    /// All edges, or those of type `ty`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn edges(&self, ty: Option<&[u8]>, before: Option<LogId>) -> Result<impl Entries + '_> {
        self.by_type(self.g.t.edge_idx, ty, before)
    }

    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn by_type(
        &self,
        table: Table,
        ty: Option<&[u8]>,
        before: Option<LogId>,
    ) -> Result<impl Entries + '_> {
        let prefix = match ty {
            None => Some(Vec::new()),
            Some(ty) => self.string_id(ty)?.map(|t| varint::pack(&[t])),
        };
        Ok(prefix
            .map(|p| self.scan(table, p, before))
            .transpose()?
            .into_iter()
            .flatten())
    }

    /// Edges of `node` in direction `dir`, optionally only of type `ty`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn node_edges(
        &self,
        node: LogId,
        dir: Direction,
        ty: Option<&[u8]>,
        before: Option<LogId>,
    ) -> Result<impl Entries + '_> {
        let mut prefix = varint::pack(&[node]);
        if let Some(ty) = ty {
            match self.string_id(ty)? {
                Some(t) => varint::push(&mut prefix, t),
                None => prefix.clear(), // unknown type: scan nothing
            }
        }
        let side = |table, wanted| {
            (wanted && !prefix.is_empty())
                .then(|| self.scan(table, prefix.clone(), before))
                .transpose()
        };
        let ins = side(self.g.t.tgt_idx, dir != Direction::Out)?;
        let outs = side(self.g.t.src_idx, dir != Direction::In)?;
        let both = dir == Direction::Both;
        let is_loop = |e: &Result<Entry>| matches!(e, Ok(Entry { record: Record::Edge { src, tgt, .. }, .. }) if src == tgt);
        Ok(ins.into_iter().flatten().chain(
            outs.into_iter()
                .flatten()
                .filter(move |e| !(both && is_loop(e))),
        ))
    }

    /// Live `(nodes, edges)` in view `before`.
    ///
    /// Starts from the counts saved at the last write-txn boundary at or before the view
    /// and replays the log from there.
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt log.
    pub fn counts(&self, before: Option<LogId>) -> Result<(u64, u64)> {
        let end = match self.view(before)? {
            Some(b) => b,
            None => self.next_id()?,
        };
        let (mut pos, mut nodes, mut edges) = match self
            .g
            .t
            .txnlog
            .get_lower_than_or_equal_to(self.ro(), &varint::pack(&[end]))?
        {
            Some((k, mut v)) => (
                varint::last(k)?,
                varint::take(&mut v)?,
                varint::take(&mut v)?,
            ),
            None => (1, 0, 0),
        };
        // ponytail: replays the uncommitted part of a write txn on every call; track live deltas if hot.
        let (lo, hi) = (varint::pack(&[pos]), varint::pack(&[end]));
        for kv in self.g.t.log.range(
            self.ro(),
            &(
                Bound::Included(lo.as_slice()),
                Bound::Excluded(hi.as_slice()),
            ),
        )? {
            let (k, v) = kv?;
            pos = varint::last(k)?;
            match Entry::decode(pos, v)?.record {
                Record::Node { .. } => nodes += 1,
                Record::Edge { .. } => edges += 1,
                Record::Prop { .. } => {}
                Record::Deletion { target } => match self.entry(target)?.map(|e| e.record) {
                    Some(Record::Node { .. }) => {
                        nodes = nodes
                            .checked_sub(1)
                            .ok_or(GraphError::Corrupt("count underflow"))?;
                        let cascaded = self.edges_ended_by(target, pos)?;
                        edges = edges
                            .checked_sub(cascaded)
                            .ok_or(GraphError::Corrupt("count underflow"))?;
                    }
                    Some(Record::Edge { .. }) => {
                        edges = edges
                            .checked_sub(1)
                            .ok_or(GraphError::Corrupt("count underflow"))?;
                    }
                    Some(Record::Prop { .. } | Record::Deletion { .. }) => {}
                    None => return Err(GraphError::Corrupt("deletion of missing entry")),
                },
            }
        }
        Ok((nodes, edges))
    }

    /// Number of `node`'s edges that deletion `del` ended (self-loops counted once).
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn edges_ended_by(&self, node: LogId, del: LogId) -> Result<u64> {
        let mut n = 0;
        for (table, skip_loops) in [(self.g.t.src_idx, false), (self.g.t.tgt_idx, true)] {
            for kv in table.prefix_iter(self.ro(), &varint::pack(&[node]))? {
                let id = varint::last(kv?.0)?;
                if let Some(Entry {
                    next,
                    record: Record::Edge { src, tgt, .. },
                    ..
                }) = self.entry(id)?
                    && next == del
                    && !(skip_loops && src == tgt)
                {
                    n += 1;
                }
            }
        }
        Ok(n)
    }

    /// Value stored under `key` in `domain`.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn kv_get(&self, domain: &[u8], key: &[u8]) -> Result<Option<&[u8]>> {
        let Some(d) = self.string_id(domain)? else {
            return Ok(None);
        };
        let mut k = varint::pack(&[d]);
        k.extend_from_slice(key);
        Ok(self.g.t.kv.get(self.ro(), &k)?)
    }

    /// `(key, value)` pairs in `domain` whose key starts with `prefix`, in key order.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn kv_iter(
        &self,
        domain: &[u8],
        prefix: &[u8],
    ) -> Result<impl Iterator<Item = Result<(&[u8], &[u8])>> + '_> {
        let it = self.string_id(domain)?.map(|d| -> Result<_> {
            let mut k = varint::pack(&[d]);
            let skip = k.len();
            k.extend_from_slice(prefix);
            Ok(self.g.t.kv.prefix_iter(self.ro(), &k)?.map(move |kv| {
                let (k, v) = kv?;
                Ok((k.get(skip..).unwrap_or_default(), v))
            }))
        });
        Ok(it.transpose()?.into_iter().flatten())
    }

    /// All log entries in `start..end` (`None` = to the end), live or not, in order.
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt record.
    pub fn log(&self, start: LogId, end: Option<LogId>) -> Result<impl Entries + '_> {
        let (lo, hi) = (varint::pack(&[start]), end.map(|e| varint::pack(&[e])));
        let hi = hi.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
        Ok(self
            .g
            .t
            .log
            .range(self.ro(), &(Bound::Included(lo.as_slice()), hi))?
            .map(|kv| {
                let (k, v) = kv?;
                Entry::decode(varint::last(k)?, v)
            }))
    }

    /// Highest log ID that changed entry `id` (or the graph, for 0) in view `before`:
    /// its creation or end, or the creation or end of any of its direct properties.
    ///
    /// # Errors
    /// Fails on storage errors or a corrupt record.
    pub fn update_id(&self, id: LogId, before: Option<LogId>) -> Result<LogId> {
        let view = self.view(before)?;
        let latest = |e: &Entry| {
            if e.next != 0 && view.is_none_or(|b| e.next < b) {
                e.next
            } else {
                e.id
            }
        };
        let mut max = match self.entry(id)? {
            Some(e) if id != 0 && view.is_none_or(|b| id < b) => latest(&e),
            _ => 0,
        };
        for kv in self
            .g
            .t
            .prop_idx
            .prefix_iter(self.ro(), &varint::pack(&[id]))?
        {
            let pid = varint::last(kv?.0)?;
            if view.is_none_or(|b| pid < b) {
                let p = self
                    .entry(pid)?
                    .ok_or(GraphError::Corrupt("index points past log"))?;
                max = max.max(latest(&p));
            }
        }
        Ok(max)
    }

    /// Number of items queued in `domain` (see [`fifo_push`](Self::fifo_push)).
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn fifo_len(&self, domain: &[u8]) -> Result<u64> {
        self.kv_iter(domain, b"")?
            .try_fold(0, |n, kv| kv.map(|_| n + 1))
    }

    // ── Writes ──

    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn next_string_id(&self) -> Result<StrId> {
        if self.state.next_str != 0 {
            return Ok(self.state.next_str);
        }
        match self.g.t.scalar.last(self.ro())? {
            Some((k, _)) => <[u8; 8]>::try_from(k)
                .map(u64::from_be_bytes)
                .map_err(|_| GraphError::Corrupt("bad string id"))?
                .checked_add(1)
                .ok_or(GraphError::Corrupt("string id overflow")),
            None => Ok(1),
        }
    }

    /// Interns `bytes`, returning its ID.
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn intern(&mut self, bytes: &[u8]) -> Result<StrId> {
        let known = self.string_id(bytes)?;
        self.intern_known(bytes, known)
    }

    /// Interns `bytes` whose lookup already returned `known`.
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn intern_known(&mut self, bytes: &[u8], known: Option<StrId>) -> Result<StrId> {
        if let Some(id) = known {
            return Ok(id);
        }
        let id = self.next_string_id()?;
        self.state.next_str = id
            .checked_add(1)
            .ok_or(GraphError::Corrupt("string id overflow"))?;
        let t = self.g.t;
        let w = self.rw()?;
        t.scalar
            .put_with_flags(w, PutFlags::APPEND, &id.to_be_bytes(), bytes)?;
        let mut k = fnv64(bytes).to_be_bytes().to_vec();
        k.extend_from_slice(&id.to_be_bytes());
        t.scalar_idx.put(w, &k, &[])?;
        Ok(id)
    }

    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn put_idx(&mut self, table: Table, key: &[u64]) -> Result<()> {
        Ok(table.put(self.rw()?, &varint::pack(key), &[])?)
    }

    /// Appends `record`, first ending `supersedes` (and its dependents) if non-zero.
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn append(&mut self, record: Record, supersedes: LogId) -> Result<Entry> {
        let e = Entry {
            id: self.next_id()?,
            next: 0,
            record,
        };
        self.rw()?;
        self.state.next_log =
            e.id.checked_add(1)
                .ok_or(GraphError::Corrupt("log id overflow"))?;
        if supersedes != 0 {
            self.end(supersedes, e.id)?;
        }
        let t = self.g.t;
        t.log.put_with_flags(
            self.rw()?,
            PutFlags::APPEND,
            &varint::pack(&[e.id]),
            &e.encode(),
        )?;
        Ok(e)
    }

    /// Marks `root` and everything hanging off it as ended by log entry `by`.
    ///
    /// # Errors
    /// Fails on storage errors or corrupt data.
    fn end(&mut self, root: LogId, by: LogId) -> Result<()> {
        let t = self.g.t;
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            // Already-ended children are skipped: a self-loop is reachable twice.
            let Some(mut e) = self.entry(id)?.filter(|e| e.next == 0) else {
                continue;
            };
            e.next = by;
            t.log.put(self.rw()?, &varint::pack(&[id]), &e.encode())?;
            let key = varint::pack(&[id]);
            let mut children: Vec<LogId> = self
                .scan(t.prop_idx, key.clone(), None)?
                .map(|c| c.map(|c| c.id))
                .collect::<Result<_>>()?;
            match e.record {
                Record::Node { .. } => {
                    self.state.node_delta -= 1;
                    for table in [t.src_idx, t.tgt_idx] {
                        for c in self.scan(table, key.clone(), None)? {
                            children.push(c?.id);
                        }
                    }
                }
                Record::Edge { .. } => self.state.edge_delta -= 1,
                Record::Prop { .. } | Record::Deletion { .. } => {}
            }
            stack.extend(children);
        }
        Ok(())
    }

    /// Finds or creates node `(ty, val)`. Works in a read txn if the node exists.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] if it would have to be created in a read txn.
    pub fn node(&mut self, ty: &[u8], val: &[u8]) -> Result<Entry> {
        let (t0, v0) = (self.string_id(ty)?, self.string_id(val)?);
        if let (Some(t), Some(v)) = (t0, v0)
            && let Some(e) = self.lookup(self.g.t.node_idx, &varint::pack(&[t, v]), None)?
        {
            return Ok(e);
        }
        let (ty, val) = (self.intern_known(ty, t0)?, self.intern_known(val, v0)?);
        let e = self.append(Record::Node { ty, val }, 0)?;
        self.put_idx(self.g.t.node_idx, &[ty, val, e.id])?;
        self.state.node_delta += 1;
        Ok(e)
    }

    /// Finds or creates edge `src -[ty, val]-> tgt` between live nodes.
    ///
    /// # Errors
    /// [`GraphError::NotFound`] if `src` or `tgt` is not a live node;
    /// [`GraphError::ReadOnly`] if it would have to be created in a read txn.
    pub fn edge(&mut self, src: LogId, tgt: LogId, ty: &[u8], val: &[u8]) -> Result<Entry> {
        self.live(src, "node")?;
        self.live(tgt, "node")?;
        let (t0, v0) = (self.string_id(ty)?, self.string_id(val)?);
        if let (Some(t), Some(v)) = (t0, v0)
            && let Some(e) =
                self.lookup(self.g.t.edge_idx, &varint::pack(&[t, v, src, tgt]), None)?
        {
            return Ok(e);
        }
        let (ty, val) = (self.intern_known(ty, t0)?, self.intern_known(val, v0)?);
        let e = self.append(Record::Edge { ty, val, src, tgt }, 0)?;
        let t = self.g.t;
        self.put_idx(t.edge_idx, &[ty, val, src, tgt, e.id])?;
        self.put_idx(t.src_idx, &[src, ty, e.id])?;
        self.put_idx(t.tgt_idx, &[tgt, ty, e.id])?;
        self.state.edge_delta += 1;
        Ok(e)
    }

    /// Sets property `key` of `parent` (0 = the graph). Setting the current value is a
    /// no-op; a new value supersedes the old property, ending its sub-properties.
    ///
    /// # Errors
    /// [`GraphError::NotFound`] if `parent` is not live; [`GraphError::ReadOnly`] in a read txn.
    pub fn set(&mut self, parent: LogId, key: &[u8], val: &[u8]) -> Result<Entry> {
        if parent != 0 {
            self.live(parent, "entry")?;
        }
        let (k0, v0) = (self.string_id(key)?, self.string_id(val)?);
        let cur = match k0 {
            Some(k) => self.lookup(self.g.t.prop_idx, &varint::pack(&[parent, k]), None)?,
            None => None,
        };
        if let Some(
            c @ Entry {
                record: Record::Prop { val: cv, .. },
                ..
            },
        ) = cur
            && Some(cv) == v0
        {
            return Ok(c);
        }
        let (key, val) = (self.intern_known(key, k0)?, self.intern_known(val, v0)?);
        let e = self.append(Record::Prop { parent, key, val }, cur.map_or(0, |c| c.id))?;
        self.put_idx(self.g.t.prop_idx, &[parent, key, e.id])?;
        Ok(e)
    }

    /// Removes property `key` of `parent`. Returns whether it existed.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    pub fn unset(&mut self, parent: LogId, key: &[u8]) -> Result<bool> {
        match self.prop(parent, key, None)? {
            Some(p) => self.delete(p.id).map(|_| true),
            None => Ok(false),
        }
    }

    /// Deletes a live node, edge, or property, cascading to its properties and, for a
    /// node, its edges. Returns the deletion's log ID.
    ///
    /// # Errors
    /// [`GraphError::NotFound`] if `id` is not live; [`GraphError::ReadOnly`] in a read txn.
    pub fn delete(&mut self, id: LogId) -> Result<LogId> {
        self.live(id, "entry")?;
        Ok(self.append(Record::Deletion { target: id }, id)?.id)
    }

    /// Stores `value` under `key` in `domain`. Not logged: no history, not in views.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn; LMDB rejects keys over ~500 bytes.
    pub fn kv_put(&mut self, domain: &[u8], key: &[u8], value: &[u8]) -> Result<()> {
        let mut k = varint::pack(&[self.intern(domain)?]);
        k.extend_from_slice(key);
        let t = self.g.t;
        Ok(t.kv.put(self.rw()?, &k, value)?)
    }

    /// Appends `items` to the persistent queue `domain`, which lives in the kv store
    /// (do not also use `domain` with [`kv_put`](Self::kv_put)).
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    pub fn fifo_push(&mut self, domain: &[u8], items: &[&[u8]]) -> Result<()> {
        let prefix = varint::pack(&[self.intern(domain)?]);
        let t = self.g.t;
        let mut next = match t
            .kv
            .rev_prefix_iter(self.ro(), &prefix)?
            .next()
            .transpose()?
        {
            Some((k, _)) => varint::last(k)?
                .checked_add(1)
                .ok_or(GraphError::Corrupt("fifo index overflow"))?,
            None => 1,
        };
        for item in items {
            let mut k = prefix.clone();
            varint::push(&mut k, next);
            t.kv.put(self.rw()?, &k, item)?;
            next = next
                .checked_add(1)
                .ok_or(GraphError::Corrupt("fifo index overflow"))?;
        }
        Ok(())
    }

    /// Removes and returns up to `n` items from the front of queue `domain`.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    pub fn fifo_pop(&mut self, domain: &[u8], n: usize) -> Result<Vec<Vec<u8>>> {
        self.rw()?;
        let items: Vec<(Vec<u8>, Vec<u8>)> = self
            .kv_iter(domain, b"")?
            .take(n)
            .map(|kv| kv.map(|(k, v)| (k.to_vec(), v.to_vec())))
            .collect::<Result<_>>()?;
        for (k, _) in &items {
            self.kv_del(domain, k)?;
        }
        Ok(items.into_iter().map(|(_, v)| v).collect())
    }

    /// Removes `key` from `domain`. Returns whether it existed.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    pub fn kv_del(&mut self, domain: &[u8], key: &[u8]) -> Result<bool> {
        self.rw()?;
        let Some(d) = self.string_id(domain)? else {
            return Ok(false);
        };
        let mut k = varint::pack(&[d]);
        k.extend_from_slice(key);
        let t = self.g.t;
        Ok(t.kv.delete(self.rw()?, &k)?)
    }

    /// Erases everything: the graph is empty and the next log ID is 1.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    pub fn reset(&mut self) -> Result<()> {
        let t = self.g.t;
        let w = self.rw()?;
        for table in [
            t.log,
            t.scalar,
            t.scalar_idx,
            t.node_idx,
            t.edge_idx,
            t.prop_idx,
            t.src_idx,
            t.tgt_idx,
            t.txnlog,
            t.kv,
        ] {
            table.clear(w)?;
        }
        self.state = State {
            begin: 1,
            next_log: 1,
            ..State::default()
        };
        Ok(())
    }

    /// Starts a nested write transaction. Committing it folds its changes into this one;
    /// dropping it discards them.
    ///
    /// # Errors
    /// [`GraphError::ReadOnly`] in a read txn.
    pub fn nested(&mut self) -> Result<Txn<'_>> {
        let g = self.g;
        let Inner::Rw(parent) = &mut self.inner else {
            return Err(GraphError::ReadOnly);
        };
        Ok(Txn {
            g,
            inner: Inner::Rw(g.env.nested_write_txn(parent)?),
            state: self.state,
            parent: Some(&mut self.state),
            _ticket: None,
        })
    }

    /// Commits. A top-level write also records live counts for [`counts`](Self::counts).
    ///
    /// # Errors
    /// Fails if LMDB cannot commit; the changes are then discarded.
    pub fn commit(mut self) -> Result<()> {
        if self.parent.is_none() && matches!(self.inner, Inner::Rw(_)) {
            let end = self.next_id()?;
            if end > self.state.begin {
                let (nodes, edges) = match self.g.t.txnlog.last(self.ro())? {
                    Some((_, mut v)) => (varint::take(&mut v)?, varint::take(&mut v)?),
                    None => (0, 0),
                };
                let bad = || GraphError::Corrupt("count out of range");
                let nodes = nodes
                    .checked_add_signed(self.state.node_delta)
                    .ok_or_else(bad)?;
                let edges = edges
                    .checked_add_signed(self.state.edge_delta)
                    .ok_or_else(bad)?;
                let t = self.g.t;
                t.txnlog.put_with_flags(
                    self.rw()?,
                    PutFlags::APPEND,
                    &varint::pack(&[end]),
                    &varint::pack(&[nodes, edges]),
                )?;
            }
        }
        let Txn {
            inner,
            state,
            parent,
            _ticket: ticket,
            ..
        } = self;
        match inner {
            Inner::Ro(t) => t.commit()?,
            Inner::Rw(t) => t.commit()?,
        }
        drop(ticket);
        if let Some(p) = parent {
            *p = state;
        }
        Ok(())
    }
}

/// 64-bit FNV-1a: a stable hash for the string index (collisions are re-checked).
fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
