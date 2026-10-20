//! `/expand`: expansion runs a client steers, for a System 1 model (Jev) behind its own
//! API.
//!
//! The client pulls work, so it needs no server of its own:
//! - `POST /expand` starts a run, writing into a new graph of this collection (queryable
//!   meanwhile, standing queries too): over a synthetic world with a planted scenario,
//!   or over the ICIJ Offshore Leaks graph if the server was given its CSVs (`-x dir`);
//! - `GET /expand/<id>?wait=s` long-polls for the next batch of candidates to decide;
//! - `POST /expand/<id>` answers it, one decision per candidate;
//! - `GET /expand/<id>/trace` is the run's labelled decision trace, as JSON lines;
//! - `GET /expand/<id>/batches` is every batch it asked, oracle-labelled, if asked for;
//! - `DELETE /expand/<id>` stops and forgets the run.
//!
//! The wire format is `ramp_expand::wire`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use ramp_expand::policy::{Bfs, Guided, Rules};
use ramp_expand::scenario::{self, Scenario};
use ramp_expand::world::{Builder, World};
use ramp_expand::{
    Candidate, Context, Decision, Expander, Id, Policy, Reads, Rel, Source, Task, icij, wire,
};
use ramp_graph::Graph;
use serde_json::{Value, json};

use crate::api::ApiError;
use crate::store::{Open, Store};

/// Most world entities a run may ask for.
const MAX_ENTITIES: u64 = 10_000_000;
/// Longest a `GET` waits for work.
const MAX_WAIT: Duration = Duration::from_secs(60);

type Res<T> = Result<T, ApiError>;

/// Runs by ID, and the ICIJ world they share.
#[derive(Debug, Default)]
pub(crate) struct Sessions {
    runs: Mutex<HashMap<String, Arc<Session>>>,
    /// Where the ICIJ CSVs are, if the server has them.
    pub(crate) icij_dir: Option<PathBuf>,
    /// The ICIJ world, loaded by the first run that asks for it.
    icij: OnceLock<Result<Arc<World>, String>>,
}

/// How a run is scored and labelled.
#[derive(Debug)]
enum Check {
    /// A planted scenario: done once its signal edges are written.
    Signal(Vec<(Id, Id, Rel)>),
    /// ICIJ connect: done once the seeds are joined; the joining path is the hit, and
    /// the cheapest path in the whole graph costs `best`.
    Join(Id, Id, f64),
    /// Done when the goal says (an alert's first match) or the budget is spent.
    Goal,
}

/// One run, shared between its thread and the requests about it.
#[derive(Debug)]
struct Session {
    graph: String,
    /// What labels the batches, if they were asked for.
    oracle: Option<HashSet<Id>>,
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
    /// Set when the run ends: the report, the trace and the batches as JSON lines.
    done: Option<(Value, Vec<u8>, Vec<u8>)>,
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
    /// For each candidate's [`Source::text`].
    world: Arc<World>,
    timeout: Duration,
    spread: f64,
}

impl Policy for Remote {
    fn name(&self) -> String {
        "remote".to_owned()
    }

    fn spread(&self) -> f64 {
        self.spread
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
                "candidates": batch.iter().map(|c| {
                    let mut v = wire::candidate(c);
                    let text = self.world.text(c.id);
                    if let (false, Some(o)) = (text.is_empty(), v.as_object_mut()) {
                        o.insert("text".to_owned(), text.into());
                    }
                    v
                }).collect::<Vec<_>>(),
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
        self.runs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError::new(404, format!("no expansion {id}")))
    }

    /// The ICIJ world, loaded on first use.
    ///
    /// # Errors
    /// 400 if the server has no ICIJ data, 500 if it does not load.
    fn icij(&self) -> Res<Arc<World>> {
        let dir = self
            .icij_dir
            .as_ref()
            .ok_or_else(|| ApiError::new(400, "no ICIJ data: start the server with -x dir"))?;
        self.icij
            .get_or_init(|| icij::load(dir).map(Arc::new).map_err(|e| e.to_string()))
            .clone()
            .map_err(|e| ApiError::new(500, format!("ICIJ data: {e}")))
    }

    /// The world, task, check and BFS/rules depth a `POST /expand` asks for.
    ///
    /// # Errors
    /// 400 for a malformed or impossible request.
    fn world(&self, req: &Value, budget: u64) -> Res<(Arc<World>, Task, Check, u32)> {
        if req
            .get("world")
            .and_then(Value::as_str)
            .unwrap_or("synthetic")
            == "icij"
        {
            let w = self.icij()?;
            let seeds: Vec<Id> = req
                .get("seeds")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|s| {
                    s.as_u64()
                        .filter(|&s| usize::try_from(s).is_ok_and(|s| s < w.len()))
                })
                .collect::<Option<_>>()
                .ok_or_else(|| ApiError::new(400, "seeds must be entity IDs"))?;
            let goal = req.get("goal").and_then(Value::as_str).unwrap_or("connect");
            let (task, check, depth) = match (goal, seeds.as_slice()) {
                ("connect", &[a, b]) => {
                    let (task, (best, _)) = icij::connect_task(&w, a, b, budget)
                        .ok_or_else(|| ApiError::new(400, "the seeds are not connected"))?;
                    (task, Check::Join(a, b, best), 3)
                }
                ("alert", &[s]) => {
                    let task = icij::alert_task(&w, s, budget).ok_or_else(|| {
                        ApiError::new(400, "the seed has under 50 co-officers with a country")
                    })?;
                    (task, Check::Goal, 2)
                }
                _ => {
                    return Err(ApiError::new(
                        400,
                        "goal is connect with two seeds or alert with one",
                    ));
                }
            };
            return Ok((w, task, check, depth));
        }
        let entities = uint(req, "entities", 200_000)?;
        if !(1_000..=MAX_ENTITIES).contains(&entities) {
            return Err(ApiError::new(
                400,
                format!("entities must be 1000..={MAX_ENTITIES}"),
            ));
        }
        let entities = u32::try_from(entities).map_err(|_| ApiError::new(400, "entities"))?;
        let seed = uint(req, "seed", 42)?;
        let name = req
            .get("scenario")
            .and_then(Value::as_str)
            .unwrap_or("alert");
        let mut world = Builder::random(entities, seed);
        let s: Scenario = scenario::plant(&mut world)
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| ApiError::new(400, format!("no scenario {name}")))?;
        let mut task = s.task.clone();
        task.budget = budget;
        let depth = s.depth;
        Ok((
            Arc::new(world.build()),
            task,
            Check::Signal(s.signal),
            depth,
        ))
    }

    /// `POST /expand`: `{"world"?, "policy"?, "budget"?, "cap"?, "timeout"?, "parallel"?,
    /// "spread"?}` and what the world needs:
    /// - `synthetic` (the default): `{"scenario", "entities"?, "seed"?}`;
    /// - `icij`: `{"goal": "connect", "seeds": [a, b]}` or `{"goal": "alert", "seeds":
    ///   [s]}`, entity IDs as the ICIJ bench numbers them.
    ///
    /// `policy` is `remote` (the default: the client decides), `guided`, `pushdown`,
    /// `bfs` or `rules` (with `cap`); `timeout` is how long a remote batch may go
    /// unanswered, in seconds, before the run stops. `parallel` candidates are fetched
    /// per round (1); with a remote policy, a round takes those within `spread` of its
    /// best priority (unbounded). A remote policy's candidates carry their `text`.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn start(&self, store: &Store, id: &str, graph: &str, req: &[u8]) -> Res<Value> {
        let req = body(req)?;
        let budget = uint(&req, "budget", 100_000)?;
        let timeout = Duration::from_secs(uint(&req, "timeout", 60)?);
        let cap = u32::try_from(uint(&req, "cap", 1000)?).unwrap_or(u32::MAX);
        let parallel = usize::try_from(uint(&req, "parallel", 1)?.min(256)).unwrap_or(1);
        let spread = match req.get("spread") {
            None => f64::INFINITY,
            Some(v) => v
                .as_f64()
                .filter(|s| *s >= 0.0)
                .ok_or_else(|| ApiError::new(400, "spread must be a non-negative number"))?,
        };
        let (world, task, check, depth) = self.world(&req, budget)?;
        let oracle = req
            .get("batches")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            .then(|| oracle(&world, &task, &check));
        let session = Arc::new(Session {
            graph: graph.to_owned(),
            oracle,
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
                world: Arc::clone(&world),
                timeout,
                spread,
            }),
            "guided" => Box::new(Guided::default()),
            "pushdown" => Box::new(Guided { pushdown: true }),
            "bfs" => Box::new(Bfs { depth }),
            "rules" => Box::new(Rules { depth, cap }),
            other => return Err(ApiError::new(400, format!("no policy {other}"))),
        };
        let g = store.graph(graph, Open::New)?;
        store.refresh(graph, &g)?;
        let out = json!({"id": id, "graph": graph, "task": wire::task(&task, &*world)});
        let run = Arc::clone(&session);
        std::thread::spawn(move || {
            let done = expand(&world, &g, policy, task, parallel, &check, &run);
            let mut shared = run.lock();
            shared.done = Some(done);
            shared.ask = None;
            drop(shared);
            run.changed.notify_all();
        });
        self.runs
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
            if let Some((report, ..)) = &shared.done {
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
            .map(|(_, trace, _)| trace.clone())
            .ok_or_else(|| ApiError::new(409, "the run is not done"))
    }

    /// `GET /expand/<id>/batches`, once the run is done: empty unless it was started
    /// with `"batches": true`.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn batches(&self, id: &str) -> Res<Vec<u8>> {
        let session = self.get(id)?;
        let shared = session.lock();
        shared
            .done
            .as_ref()
            .map(|(.., batches)| batches.clone())
            .ok_or_else(|| ApiError::new(409, "the run is not done"))
    }

    /// `DELETE /expand/<id>`: stops the run; its graph stays.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    pub(crate) fn stop(&self, id: &str) -> Res<()> {
        let session = self
            .runs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(id)
            .ok_or_else(|| ApiError::new(404, format!("no expansion {id}")))?;
        session.lock().cancelled = true;
        session.changed.notify_all();
        Ok(())
    }
}

/// The entities a batch's candidate is labelled positive for, as the benches label them.
///
/// An endpoint of a planted signal edge; for ICIJ connect, on a path between
/// the seeds within ε of the cheapest; for an alert, sharing a company with an
/// officer from the target country.
fn oracle(world: &World, task: &Task, check: &Check) -> HashSet<Id> {
    match check {
        Check::Signal(signal) => signal.iter().flat_map(|&(a, b, _)| [a, b]).collect(),
        Check::Join(a, b, best) => icij::near_cheapest(world, *a, *b, *best),
        Check::Goal => icij::shared_with_target(world, task),
    }
}

/// Runs `task` to the end: the report, the trace and the batches.
///
/// The report says how much of a planted signal was recovered, or the joining path; the
/// trace is labelled with the hits; with the session's oracle, every batch is kept and
/// labelled by it.
fn expand(
    world: &World,
    g: &Graph,
    policy: Box<dyn Policy + Send>,
    task: Task,
    parallel: usize,
    check: &Check,
    session: &Session,
) -> (Value, Vec<u8>, Vec<u8>) {
    let mut extra = match check {
        Check::Signal(signal) => json!({"signal": [0, signal.len()]}),
        Check::Join(..) | Check::Goal => json!({}),
    };
    let mut truth = HashSet::new();
    let run = Expander::new(world, g, policy, task, Reads::Projection).and_then(|ex| {
        let ex = ex.parallel(parallel);
        let ex = if session.oracle.is_some() {
            ex.log_batches()
        } else {
            ex
        };
        ex.run(|ex| {
            let done = match check {
                Check::Signal(signal) => {
                    let recovered = signal
                        .iter()
                        .filter(|&&(a, b, r)| ex.has_edge(a, b, r))
                        .count();
                    extra = json!({"signal": [recovered, signal.len()]});
                    !signal.is_empty() && recovered == signal.len()
                }
                Check::Join(a, b, best) => {
                    let joined = ex.connected(*a, *b);
                    if let Some((cost, hub, edges)) = joined
                        .then(|| icij::joining(world, ex.edges(), *a, *b))
                        .flatten()
                    {
                        extra = json!({"path": {"cost": cost, "widest": hub, "best": best}});
                        truth = edges;
                    }
                    joined
                }
                Check::Goal => false,
            };
            session.lock().cancelled || done
        })
    });
    match run {
        Err(e) => (json!({"error": e.to_string()}), Vec::new(), Vec::new()),
        Ok(report) => {
            if let Check::Signal(signal) = check {
                truth = signal
                    .iter()
                    .map(|&(a, b, r)| (a.min(b), a.max(b), r))
                    .collect();
            }
            let mut out = wire::report(&report);
            if let (Some(o), Some(extra)) = (out.as_object_mut(), extra.as_object()) {
                o.extend(extra.clone());
                o.insert("cancelled".to_owned(), session.lock().cancelled.into());
            }
            let batches = session
                .oracle
                .as_ref()
                .map_or_else(Vec::new, |o| wire::batches(&report, |c| o.contains(&c.id)));
            (out, wire::trace(&report, &truth), batches)
        }
    }
}
