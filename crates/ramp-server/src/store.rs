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
    // ponytail: every graph stays open; add an LRU if fd limits bite.
    open: Mutex<HashMap<String, Arc<Graph>>>,
    // ponytail: rebuilt by opening every graph at startup; persist it if startup gets slow.
    index: RwLock<BTreeMap<String, Status>>,
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
    pub(crate) fn open(dir: impl Into<PathBuf>) -> Result<Self, ApiError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(ApiError::io)?;
        let store = Self {
            dir,
            open: Mutex::default(),
            index: RwLock::default(),
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
        let exists = open.contains_key(uuid) || self.path(uuid).exists();
        match (how, exists) {
            (Open::Existing, false) => return Err(ApiError::new(404, format!("no graph {uuid}"))),
            (Open::New, true) => {
                return Err(ApiError::new(409, format!("graph {uuid} already exists")));
            }
            (Open::Existing, true) | (Open::New, false) => {}
        }
        if let Some(g) = open.get(uuid) {
            return Ok(Arc::clone(g));
        }
        let g = Arc::new(Graph::open(self.path(uuid))?);
        open.insert(uuid.to_owned(), Arc::clone(&g));
        drop(open);
        Ok(g)
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
        open.remove(uuid);
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
            if open.contains_key(uuid) || target.exists() {
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
    fn uuid_shape() {
        assert!(is_uuid("0b9e2f7c-5a1d-11ef-8d3e-0242ac120002"));
        assert!(!is_uuid("0B9E2F7C-5A1D-11EF-8D3E-0242AC120002"));
        assert!(!is_uuid("../../etc/passwd"));
    }
}
