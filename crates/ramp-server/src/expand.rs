//! `/expand`: expansion runs a client steers, for a System 1 model (Jev) behind its own
//! API.
//!
//! The client pulls work, so it needs no server of its own:
//! - `POST /expand` starts a run over a synthetic world with a planted scenario, writing
//!   into a new graph of this collection (queryable meanwhile, standing queries too);
//! - `GET /expand/<id>?wait=s` long-polls for the next batch of candidates to decide;
//! - `POST /expand/<id>` answers it, one decision per candidate;
//! - `GET /expand/<id>/trace` is the run's labelled decision trace, as JSON lines;
//! - `DELETE /expand/<id>` stops and forgets the run.
//!
//! The wire format is `ramp_expand::wire`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use ramp_expand::policy::{Bfs, Guided, Rules};
use ramp_expand::scenario::{self, Scenario};
use ramp_expand::world::Builder;
use ramp_expand::{Candidate, Context, Decision, Expander, Policy, Reads, Task, wire};
use ramp_graph::Graph;
use serde_json::{Value, json};

use crate::api::ApiError;
use crate::store::{Open, Store};

/// Most world entities a run may ask for.
const MAX_ENTITIES: u64 = 10_000_000;
/// Longest a `GET` waits for work.
const MAX_WAIT: Duration = Duration::from_secs(60);

type Res<T> = Result<T, ApiError>;

/// Runs by ID.
#[derive(Debug, Default)]
pub(crate) struct Sessions(Mutex<HashMap<String, Arc<Session>>>);

/// One run, shared between its thread and the requests about it.
#[derive(Debug)]
struct Session {
    graph: String,
    shared: Mutex<Shared>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct Shared {
    /// Numbers the batches asked, so a late answer cannot land on the next one.
    seq: u64,
    /// The batch waiting for decisions, and how many it needs.
    ask: Option<(Value, usize)>,
    answer: Option<Vec<Decision>>,
    cancelled: bool,
    /// Set when the run ends: the report, and the trace as JSON lines.
    done: Option<(Value, Vec<u8>)>,
}

impl Session {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A policy whose decisions come from the client.
#[derive(Debug)]
struct Remote {
    session: Arc<Session>,
    timeout: Duration,
}

impl Policy for Remote {
    fn name(&self) -> String {
        "remote".to_owned()
    }

    fn decide(
        &mut self,
        _: &Task,
        ctx: &Context<'_>,
        batch: &[Candidate],
        out: &mut Vec<Decision>,
    ) {
        let s = &self.session;
        let mut shared = s.lock();
        if !shared.cancelled {
            shared.seq += 1;
            let ask = json!({
                "seq": shared.seq,
                "context": wire::context(ctx),
                "candidates": batch.iter().map(wire::candidate).collect::<Vec<_>>(),
            });
            shared.ask = Some((ask, batch.len()));
            shared.answer = None;
            s.changed.notify_all();
            let deadline = Instant::now() + self.timeout;
            loop {
                if let Some(answer) = shared.answer.take() {
                    out.extend(answer);
                    return;
                }
                let left = deadline.saturating_duration_since(Instant::now());
                if shared.cancelled || left.is_zero() {
                    // An absent client ends the run: no decisions, and `stop` holds.
                    shared.cancelled = true;
                    break;
                }
                shared = s
                    .changed
                    .wait_timeout(shared, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            shared.ask = None;
        }
        drop(shared);
        out.extend(batch.iter().map(|_| Decision::default()));
    }
}

/// A body as JSON (`{}` when empty).
///
/// # Errors
/// 400 if it is not JSON.
fn body(bytes: &[u8]) -> Res<Value> {
    if bytes.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_slice(bytes).map_err(|e| ApiError::new(400, format!("bad JSON: {e}")))
}

/// `v[key]`, or `default` if absent.
///
/// # Errors
/// 400 if it is not a non-negative integer.
fn uint(v: &Value, key: &str, default: u64) -> Res<u64> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x
            .as_u64()
            .ok_or_else(|| ApiError::new(400, format!("{key} must be a non-negative integer"))),
    }
}

impl Sessions {
    /// # Errors
    /// 404 if there is no such run.
    fn get(&self, id: &str) -> Res<Arc<Session>> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError::new(404, format!("no expansion {id}")))
    }

    /// `POST /expand`: `{"scenario", "policy"?, "budget"?, "entities"?, "seed"?, "cap"?,
    /// "timeout"?}`. `policy` is `remote` (the default: the client decides), `guided`,
    /// `pushdown`, `bfs` or `rules` (with `cap`); `timeout` is how long a remote batch
    /// may go unanswered, in seconds, before the run stops.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn start(&self, store: &Store, id: &str, graph: &str, req: &[u8]) -> Res<Value> {
        let req = body(req)?;
        let entities = uint(&req, "entities", 200_000)?;
        if !(1_000..=MAX_ENTITIES).contains(&entities) {
            return Err(ApiError::new(
                400,
                format!("entities must be 1000..={MAX_ENTITIES}"),
            ));
        }
        let entities = u32::try_from(entities).map_err(|_| ApiError::new(400, "entities"))?;
        let seed = uint(&req, "seed", 42)?;
        let budget = uint(&req, "budget", 100_000)?;
        let timeout = Duration::from_secs(uint(&req, "timeout", 60)?);
        let cap = u32::try_from(uint(&req, "cap", 1000)?).unwrap_or(u32::MAX);
        let name = req
            .get("scenario")
            .and_then(Value::as_str)
            .unwrap_or("alert");
        let mut world = Builder::random(entities, seed);
        let s: Scenario = scenario::plant(&mut world)
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| ApiError::new(400, format!("no scenario {name}")))?;
        let world = world.build();
        let mut task = s.task.clone();
        task.budget = budget;
        let session = Arc::new(Session {
            graph: graph.to_owned(),
            shared: Mutex::default(),
            changed: Condvar::new(),
        });
        let policy: Box<dyn Policy + Send> = match req
            .get("policy")
            .and_then(Value::as_str)
            .unwrap_or("remote")
        {
            "remote" => Box::new(Remote {
                session: Arc::clone(&session),
                timeout,
            }),
            "guided" => Box::new(Guided::default()),
            "pushdown" => Box::new(Guided { pushdown: true }),
            "bfs" => Box::new(Bfs { depth: s.depth }),
            "rules" => Box::new(Rules {
                depth: s.depth,
                cap,
            }),
            other => return Err(ApiError::new(400, format!("no policy {other}"))),
        };
        let g = store.graph(graph, Open::New)?;
        store.refresh(graph, &g)?;
        let out = json!({"id": id, "graph": graph, "task": wire::task(&task, &world)});
        let run = Arc::clone(&session);
        std::thread::spawn(move || {
            let done = expand(&world, &g, policy, task, &s, &run);
            let mut shared = run.lock();
            shared.done = Some(done);
            shared.ask = None;
            drop(shared);
            run.changed.notify_all();
        });
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id.to_owned(), session);
        Ok(out)
    }

    /// `GET /expand/<id>`: the batch to decide, the final report once done, or
    /// `{"state": "running"}` if neither came within `wait`.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn poll(&self, store: &Store, id: &str, wait: Duration) -> Res<Value> {
        let session = self.get(id)?;
        let deadline = Instant::now() + wait.min(MAX_WAIT);
        let mut shared = session.lock();
        loop {
            if let Some((report, _)) = &shared.done {
                let report = report.clone();
                drop(shared);
                // The run wrote without going through the API: list it as it is now.
                let g = store.graph(&session.graph, Open::Existing)?;
                store.refresh(&session.graph, &g)?;
                return Ok(json!({"state": "done", "report": report}));
            }
            if let Some((ask, _)) = &shared.ask {
                let mut ask = ask.clone();
                if let Some(o) = ask.as_object_mut() {
                    o.insert("state".to_owned(), "ask".into());
                }
                return Ok(ask);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(json!({"state": "running"}));
            }
            shared = session
                .changed
                .wait_timeout(shared, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// `POST /expand/<id>`: `{"seq", "decisions": [..]}`, one per candidate asked.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn answer(&self, id: &str, req: &[u8]) -> Res<()> {
        let session = self.get(id)?;
        let req = body(req)?;
        let decisions = req
            .get("decisions")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::new(400, "decisions must be a list"))?
            .iter()
            .map(wire::parse_decision)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ApiError::new(400, e))?;
        let seq = uint(&req, "seq", 0)?;
        let mut shared = session.lock();
        let want = match &shared.ask {
            Some(_) if seq != shared.seq => {
                return Err(ApiError::new(
                    409,
                    format!("batch {seq} is not the open one"),
                ));
            }
            Some((_, want)) => *want,
            None => return Err(ApiError::new(409, "no batch is waiting")),
        };
        if decisions.len() != want {
            return Err(ApiError::new(
                400,
                format!("{want} candidates, {} decisions", decisions.len()),
            ));
        }
        shared.ask = None;
        shared.answer = Some(decisions);
        drop(shared);
        session.changed.notify_all();
        Ok(())
    }

    /// `GET /expand/<id>/trace`, once the run is done.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn trace(&self, id: &str) -> Res<Vec<u8>> {
        let session = self.get(id)?;
        let shared = session.lock();
        shared
            .done
            .as_ref()
            .map(|(_, trace)| trace.clone())
            .ok_or_else(|| ApiError::new(409, "the run is not done"))
    }

    /// `DELETE /expand/<id>`: stops the run; its graph stays.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn stop(&self, id: &str) -> Res<()> {
        let session = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(id)
            .ok_or_else(|| ApiError::new(404, format!("no expansion {id}")))?;
        session.lock().cancelled = true;
        session.changed.notify_all();
        Ok(())
    }
}

/// Runs `task` to the end; returns the report (with how much of the planted signal was
/// recovered) and the trace labelled with it.
fn expand(
    world: &ramp_expand::world::World,
    g: &Graph,
    policy: Box<dyn Policy + Send>,
    task: Task,
    s: &Scenario,
    session: &Session,
) -> (Value, Vec<u8>) {
    let mut recovered = 0;
    let run = Expander::new(world, g, policy, task, Reads::Projection).and_then(|ex| {
        ex.run(|ex| {
            recovered = s
                .signal
                .iter()
                .filter(|&&(a, b, r)| ex.has_edge(a, b, r))
                .count();
            session.lock().cancelled || (!s.signal.is_empty() && recovered == s.signal.len())
        })
    });
    match run {
        Err(e) => (json!({"error": e.to_string()}), Vec::new()),
        Ok(report) => {
            let truth: HashSet<_> = s
                .signal
                .iter()
                .map(|&(a, b, r)| (a.min(b), a.max(b), r))
                .collect();
            let mut out = wire::report(&report);
            if let Some(o) = out.as_object_mut() {
                o.insert("signal".to_owned(), json!([recovered, s.signal.len()]));
                o.insert("cancelled".to_owned(), session.lock().cancelled.into());
            }
            (out, wire::trace(&report, &truth))
        }
    }
}
