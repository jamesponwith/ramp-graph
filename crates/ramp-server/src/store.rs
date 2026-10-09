//! The collection: one graph file per UUID in a directory, an open handle per graph,
//! and an in-memory index of per-graph status used for listings and permissions.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use ramp_graph::Graph;
use serde_json::{Map, Value};

use crate::api::ApiError;
use crate::input::props_dict;

/// What a listing or `/status` shows, plus what permission checks read.
#[derive(Debug, Clone)]
pub(crate) struct Status {
    pub(crate) next_id: u64,
    pub(crate) size: u64,
    pub(crate) nodes: u64,
    pub(crate) edges: u64,
    /// Graph properties; `roles` and `enabled` live here.
    pub(crate) meta: Map<String, Value>,
}

/// How to open a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Open {
    Existing,
    /// Fail with 409 if it exists.
    New,
}

#[derive(Debug)]
pub(crate) struct Store {
    dir: PathBuf,
    // LMDB forbids opening one file twice in a process: every open goes through this map.
    open: Mutex<Handles>,
    // ponytail: rebuilt by opening every graph at startup; persist it if startup gets slow.
    index: RwLock<BTreeMap<String, Status>>,
    /// Bytes of projections kept across all open graphs.
    projection_budget: usize,
}

/// Open graphs, closed least-recently-used first once there are more than `max`.
#[derive(Debug)]
struct Handles {
    graphs: HashMap<String, (Arc<Graph>, u64)>,
    tick: u64,
    max: usize,
}

impl Handles {
    fn get(&mut self, uuid: &str) -> Option<Arc<Graph>> {
        self.tick += 1;
        let (g, used) = self.graphs.get_mut(uuid)?;
        *used = self.tick;
        Some(Arc::clone(g))
    }

    fn insert(&mut self, uuid: &str, g: &Arc<Graph>) {
        self.tick += 1;
        self.graphs
            .insert(uuid.to_owned(), (Arc::clone(g), self.tick));
        // Only close graphs no request holds: reopening one that is still open elsewhere
        // in the process is refused by LMDB. When all are busy, go over `max` for now.
        // ponytail: O(open graphs) scan per eviction; a linked LRU if `max` gets large.
        while self.graphs.len() > self.max {
            let idle = self
                .graphs
                .iter()
                .filter(|(_, (g, _))| Arc::strong_count(g) == 1)
                .min_by_key(|(_, (_, used))| *used);
            let Some(lru) = idle.map(|(u, _)| u.clone()) else {
                break;
            };
            self.graphs.remove(&lru);
        }
    }
}

fn poisoned<T>(_: PoisonError<T>) -> ApiError {
    ApiError::internal("lock poisoned")
}

/// Upstream's UUID shape: lower-case hex, 8-4-4-4-12.
pub(crate) fn is_uuid(s: &str) -> bool {
    s.len() == 36
        && s.char_indices().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => c == '-',
            _ => c.is_ascii_digit() || ('a'..='f').contains(&c),
        })
}

impl Store {
    /// Opens (creating if needed) the directory and indexes every `<uuid>.db` in it.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    /// At most `max_open` graphs stay open while idle (each holds three file descriptors
    /// and ~2 MiB of RAM).
    pub(crate) fn open(
        dir: impl Into<PathBuf>,
        max_open: usize,
        projection_budget: usize,
    ) -> Result<Self, ApiError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(ApiError::io)?;
        let store = Self {
            dir,
            open: Mutex::new(Handles {
                graphs: HashMap::new(),
                tick: 0,
                max: max_open,
            }),
            index: RwLock::default(),
            projection_budget,
        };
        let names: Vec<String> = std::fs::read_dir(&store.dir)
            .map_err(ApiError::io)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|n| {
                n.strip_suffix(".db")
                    .filter(|u| is_uuid(u))
                    .map(str::to_owned)
            })
            .collect();
        for uuid in names {
            let g = store.graph(&uuid, Open::Existing)?;
            store.refresh(&uuid, &g)?;
        }
        Ok(store)
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(crate) fn path(&self, uuid: &str) -> PathBuf {
        self.dir.join(format!("{uuid}.db"))
    }

    /// The graph's handle, opening (or creating) it as asked.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn graph(&self, uuid: &str, how: Open) -> Result<Arc<Graph>, ApiError> {
        let mut open = self.open.lock().map_err(poisoned)?;
        let exists = open.graphs.contains_key(uuid) || self.path(uuid).exists();
        match (how, exists) {
            (Open::Existing, false) => return Err(ApiError::new(404, format!("no graph {uuid}"))),
            (Open::New, true) => {
                return Err(ApiError::new(409, format!("graph {uuid} already exists")));
            }
            (Open::Existing, true) | (Open::New, false) => {}
        }
        if let Some(g) = open.get(uuid) {
            return Ok(g);
        }
        let g = Arc::new(Graph::open(self.path(uuid))?);
        open.insert(uuid, &g);
        drop(open);
        Ok(g)
    }

    /// Keeps the projections of open graphs within the budget, dropping least recently
    /// used graphs' projections first; `keep` (the one just built) goes last, and only
    /// if it alone is over budget. A dropped projection is rebuilt by the next scan.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn fit_projections(&self, keep: &str) -> Result<(), ApiError> {
        // ponytail: a graph larger than the budget rebuilds on every scan; estimate from the file and refuse if that bites
        let mut held: Vec<(bool, u64, Arc<Graph>, usize)> = self
            .open
            .lock()
            .map_err(poisoned)?
            .graphs
            .iter()
            .filter_map(|(u, (g, used))| {
                Some((
                    u == keep,
                    *used,
                    Arc::clone(g),
                    g.cached_projection()?.bytes(),
                ))
            })
            .collect();
        held.sort_by_key(|&(is_keep, used, _, _)| (is_keep, used));
        let mut total: usize = held.iter().map(|&(_, _, _, b)| b).sum();
        for (_, _, g, bytes) in held {
            if total <= self.projection_budget {
                break;
            }
            g.drop_projection();
            total = total.saturating_sub(bytes);
        }
        Ok(())
    }

    /// Recomputes a graph's index entry from its current state.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn refresh(&self, uuid: &str, g: &Graph) -> Result<(), ApiError> {
        let t = g.read()?;
        let (nodes, edges) = t.counts(None)?;
        let status = Status {
            next_id: t.next_id()?,
            size: g.size()?,
            nodes,
            edges,
            meta: props_dict(&t, 0, None)?,
        };
        self.index
            .write()
            .map_err(poisoned)?
            .insert(uuid.to_owned(), status);
        Ok(())
    }

    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn status(&self, uuid: &str) -> Result<Option<Status>, ApiError> {
        Ok(self.index.read().map_err(poisoned)?.get(uuid).cloned())
    }

    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn list(&self) -> Result<Vec<(String, Status)>, ApiError> {
        Ok(self
            .index
            .read()
            .map_err(poisoned)?
            .iter()
            .map(|(u, s)| (u.clone(), s.clone()))
            .collect())
    }

    /// Closes and removes a graph. Requests already holding its handle finish against
    /// the unlinked file.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn delete(&self, uuid: &str) -> Result<(), ApiError> {
        let mut open = self.open.lock().map_err(poisoned)?;
        open.graphs.remove(uuid);
        self.index.write().map_err(poisoned)?.remove(uuid);
        let removed = remove_lmdb(&self.path(uuid));
        drop(open); // held so a concurrent create cannot race the unlink
        removed
    }

    /// Moves a validated graph file into place as `uuid` and indexes it.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn adopt(&self, uuid: &str, file: &Path) -> Result<(), ApiError> {
        {
            let open = self.open.lock().map_err(poisoned)?;
            let target = self.path(uuid);
            if open.graphs.contains_key(uuid) || target.exists() {
                return Err(ApiError::new(409, format!("graph {uuid} already exists")));
            }
            // A lock file belongs to one data file: never carry it across a rename.
            std::fs::remove_file(lock_path(file)).unwrap_or_default();
            std::fs::rename(file, &target).map_err(ApiError::io)?;
            drop(open);
        }
        let g = self.graph(uuid, Open::Existing)?;
        self.refresh(uuid, &g)
    }
}

pub(crate) fn lock_path(db: &Path) -> PathBuf {
    let mut p = db.as_os_str().to_owned();
    p.push("-lock");
    PathBuf::from(p)
}

/// Removes an LMDB file and its lock file.
///
/// # Errors
/// The HTTP error to send instead.
pub(crate) fn remove_lmdb(path: &Path) -> Result<(), ApiError> {
    std::fs::remove_file(lock_path(path)).unwrap_or_default(); // may not exist
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(ApiError::io(e)),
        Ok(()) | Err(_) => Ok(()),
    }
}

/// Who is asking (from the `user` and `role` query parameters).
///
/// These are claims, not credentials: put the server behind something that sets them.
#[derive(Debug, Default)]
pub(crate) struct Creds {
    pub(crate) user: Option<String>,
    pub(crate) roles: Vec<String>,
}

/// Roles granted to `user` by a graph's `roles` property: `{user: {role: truthy}}` or
/// `{user: "role1 role2"}`.
fn user_roles(meta: &Map<String, Value>, user: &str) -> BTreeSet<String> {
    match meta.get("roles").and_then(|r| r.get(user)) {
        Some(Value::Object(m)) => m
            .iter()
            .filter(|(_, v)| crate::input::truthy(v))
            .map(|(k, _)| k.clone())
            .collect(),
        Some(Value::String(s)) => words(s),
        Some(v @ (Value::Number(_) | Value::Bool(_) | Value::Array(_))) => words(&v.to_string()),
        Some(Value::Null) | None => BTreeSet::new(),
    }
}

fn words(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// A graph without a `roles` property is open to all. Otherwise the caller must name a
/// user holding a role (one of `roles`, if any were asked for).
pub(crate) fn allowed(meta: &Map<String, Value>, creds: &Creds) -> bool {
    if !meta.contains_key("roles") {
        return true;
    }
    let Some(user) = &creds.user else {
        return false;
    };
    let have = user_roles(meta, user);
    if creds.roles.is_empty() {
        !have.is_empty()
    } else {
        creds.roles.iter().any(|r| have.contains(r))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn permissions() {
        let meta = |v: Value| v.as_object().cloned().unwrap();
        let creds = |u: Option<&str>, r: &[&str]| Creds {
            user: u.map(str::to_owned),
            roles: r.iter().map(|s| (*s).to_owned()).collect(),
        };
        let open = meta(json!({"name": "x"}));
        assert!(allowed(&open, &creds(None, &[])));
        let locked =
            meta(json!({"roles": {"amy": {"read": true, "write": false}, "bob": "read, write"}}));
        assert!(
            !allowed(&locked, &creds(None, &[])),
            "upstream skipped the check without a user"
        );
        assert!(allowed(&locked, &creds(Some("amy"), &[])));
        assert!(!allowed(&locked, &creds(Some("amy"), &["write"])));
        assert!(allowed(&locked, &creds(Some("bob"), &["write"])));
        assert!(!allowed(&locked, &creds(Some("eve"), &[])));
    }

    #[test]
    fn idle_graphs_close_lru_first() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), 2, usize::MAX).unwrap();
        let uuids = [
            "0b9e2f7c-5a1d-11ef-8d3e-0242ac120001",
            "0b9e2f7c-5a1d-11ef-8d3e-0242ac120002",
            "0b9e2f7c-5a1d-11ef-8d3e-0242ac120003",
        ];
        let open = || {
            let mut v: Vec<String> = store.open.lock().unwrap().graphs.keys().cloned().collect();
            v.sort();
            v
        };
        for u in &uuids {
            let g = store.graph(u, Open::New).unwrap();
            let mut t = g.write().unwrap();
            t.node(b"t", u.as_bytes()).unwrap();
            t.commit().unwrap();
        }
        assert_eq!(open(), [uuids[1], uuids[2]], "first one closed");

        // A graph a request still holds is never closed, even if least recently used.
        let held = store.graph(uuids[1], Open::Existing).unwrap();
        store.graph(uuids[2], Open::Existing).unwrap();
        let zero = store.graph(uuids[0], Open::Existing).unwrap();
        assert_eq!(open(), [uuids[0], uuids[1]]);
        let more = store.graph(uuids[2], Open::Existing).unwrap();
        assert_eq!(
            open().len(),
            3,
            "everything busy: over the cap rather than double-open"
        );
        drop((held, zero, more));

        // Reopened graphs still have their data.
        let g = store.graph(uuids[0], Open::Existing).unwrap();
        assert!(
            g.read()
                .unwrap()
                .node_lookup(b"t", uuids[0].as_bytes(), None)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn projections_fit_the_budget_lru_first() {
        let dir = tempfile::tempdir().unwrap();
        let uuids = [
            "0b9e2f7c-5a1d-11ef-8d3e-0242ac120001",
            "0b9e2f7c-5a1d-11ef-8d3e-0242ac120002",
            "0b9e2f7c-5a1d-11ef-8d3e-0242ac120003",
        ];
        let mut graphs = Vec::new();
        let mut one = 0;
        let store = Store::open(dir.path(), 8, 0).unwrap();
        for u in &uuids {
            let g = store.graph(u, Open::New).unwrap();
            let mut t = g.write().unwrap();
            for i in 0..50_u32 {
                t.node(b"t", i.to_string().as_bytes()).unwrap();
            }
            t.commit().unwrap();
            one = g.read().unwrap().projection().unwrap().bytes();
            graphs.push(g);
        }
        let cached = |i: usize| graphs[i].cached_projection().is_some();
        // Room for two: the least recently opened loses its projection.
        let store = Store {
            projection_budget: one * 2 + 1,
            ..store
        };
        store.fit_projections(uuids[2]).unwrap();
        assert_eq!([cached(0), cached(1), cached(2)], [false, true, true]);
        // Touching a graph makes it recent; the just-built one is dropped last.
        graphs[0].read().unwrap().projection().unwrap();
        store.graph(uuids[0], Open::Existing).unwrap();
        store.fit_projections(uuids[0]).unwrap();
        assert_eq!([cached(0), cached(1), cached(2)], [true, false, true]);
        let store = Store {
            projection_budget: one - 1,
            ..store
        };
        store.fit_projections(uuids[0]).unwrap();
        assert_eq!([cached(0), cached(1), cached(2)], [false, false, false]);
        // A commit drops the cache on its own; nothing to fit afterwards.
        graphs[2].read().unwrap().projection().unwrap();
        let mut t = graphs[2].write().unwrap();
        t.node(b"t", b"new").unwrap();
        t.commit().unwrap();
        assert!(!cached(2));
    }

    #[test]
    fn uuid_shape() {
        assert!(is_uuid("0b9e2f7c-5a1d-11ef-8d3e-0242ac120002"));
        assert!(!is_uuid("0B9E2F7C-5A1D-11EF-8D3E-0242AC120002"));
        assert!(!is_uuid("../../etc/passwd"));
    }
}
