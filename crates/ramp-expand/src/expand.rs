//! The expansion loop.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::{Duration, Instant};

use ramp_graph::lgql::Pattern;
use ramp_graph::value::Value;
use ramp_graph::{EdgeSpec, Graph, GraphError, LogId, Result, Txn};

use crate::{
    Candidate, Context, Counts, Decision, Goal, Id, MAX_DOM, Policy, Rel, Source, Task, UNREACHED,
};

/// Where [`Task::probes`] read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reads {
    /// The graph's projection, advanced over each step's commit.
    Projection,
    /// LMDB, as an unprojected graph would.
    Lmdb,
}

/// What an expansion run did.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Expansion steps.
    pub steps: u64,
    /// Edges fetched from the source.
    pub fetched: u64,
    /// Edges fetched, by the domain of the entity expanded: where the budget went.
    pub fetched_by_domain: [u64; MAX_DOM],
    /// Source calls (peeks and fetches).
    pub calls: u64,
    /// Entities written to the graph.
    pub nodes: u64,
    /// Edges written to the graph.
    pub edges: u64,
    /// Edges fetched when the run's stop condition first held.
    pub done_at: Option<u64>,
    /// Edges fetched when the first standing-query match appeared.
    pub alert_at: Option<u64>,
    /// Edges fetched when the two sides of a [`Goal::Connect`] first joined.
    pub connected_at: Option<u64>,
    /// Standing-query matches.
    pub alerts: u64,
    /// Probe runs.
    pub probes: u64,
    /// Decisions the policy made.
    pub decisions: u64,
    /// Time in the whole run.
    pub wall: Duration,
    /// Time in the policy.
    pub policy: Duration,
    /// Time in the source.
    pub fetch: Duration,
    /// Time writing to the graph.
    pub write: Duration,
    /// Time in standing queries.
    pub standing: Duration,
    /// Time in probes.
    pub probe: Duration,
}

/// What the expander knows about a discovered entity.
#[derive(Debug, Clone)]
struct Known {
    log: LogId,
    dom: u8,
    hops: [u32; 2],
    cost: [f64; 2],
    peek: Counts,
    fetched: Counts,
    degree: u32,
    by_dom: [u32; MAX_DOM],
    watched: u32,
    is_watched: bool,
    alerts: u32,
    probes: Option<u32>,
    version: u32,
    decision: Decision,
    uf: usize,
}

impl Known {
    fn exhausted(&self) -> bool {
        self.peek.iter().zip(&self.fetched).all(|(p, f)| f >= p)
    }
}

/// An entity to expand and what to take of it.
type Next = (Id, Vec<(Rel, u32)>);

/// A frontier entry; stale once its entity's version moves on.
#[derive(Debug, Clone, Copy)]
struct Item {
    priority: f64,
    version: u32,
    id: Id,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            // Ties: the earlier-discovered entity first (lower ID on the max-heap).
            .then_with(|| other.id.cmp(&self.id))
            .then_with(|| self.version.cmp(&other.version))
    }
}

/// Runs one [`Task`] against a [`Source`], writing into a [`Graph`].
#[derive(Debug)]
pub struct Expander<'a, S, P> {
    src: &'a S,
    graph: &'a Graph,
    policy: P,
    task: Task,
    reads: Reads,
    known: HashMap<Id, Known>,
    by_log: HashMap<LogId, Id>,
    heap: BinaryHeap<Item>,
    /// Union-find over discovered entities, for [`Goal::Connect`], with each root's size.
    uf: Vec<usize>,
    size: Vec<u32>,
    /// Canonical `(min, max, rel)` of every edge written.
    written: HashSet<(Id, Id, Rel)>,
    standing: Vec<Pattern>,
    /// First log entry the standing queries have not seen.
    queried: LogId,
    ctx: Context,
    report: Report,
    started: Instant,
}

/// The root of `x` in `uf`. Union by size keeps trees `O(log n)` deep.
fn find(uf: &[usize], x: usize) -> usize {
    let mut root = x;
    while let Some(&p) = uf.get(root) {
        if p == root {
            break;
        }
        root = p;
    }
    root
}

impl<'a, S: Source, P: Policy> Expander<'a, S, P> {
    /// Writes the task's seeds into `graph` and scores them.
    ///
    /// # Errors
    /// Fails on storage errors or a standing query that does not parse.
    pub fn new(src: &'a S, graph: &'a Graph, policy: P, task: Task, reads: Reads) -> Result<Self> {
        let standing = task
            .standing
            .iter()
            .map(|q| Pattern::parse(q).map_err(|e| GraphError::Value(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let queried = graph.read()?.next_id()?;
        let mut ex = Self {
            src,
            graph,
            policy,
            ctx: Context {
                budget: task.budget,
                ..Context::default()
            },
            task,
            reads,
            known: HashMap::new(),
            by_log: HashMap::new(),
            heap: BinaryHeap::new(),
            uf: Vec::new(),
            size: Vec::new(),
            written: HashSet::new(),
            standing,
            queried,
            report: Report::default(),
            started: Instant::now(),
        };
        let seeds = ex.task.seeds.clone();
        let mut t = graph.write()?;
        for (i, &s) in seeds.iter().enumerate() {
            ex.discover(&mut t, s)?;
            let side = usize::from(i == 1 && ex.task.goal == Goal::Connect);
            if let Some(k) = ex.known.get_mut(&s) {
                if let Some(h) = k.hops.get_mut(side) {
                    *h = 0;
                }
                if let Some(c) = k.cost.get_mut(side) {
                    *c = 0.0;
                }
            }
        }
        t.commit()?;
        ex.rescore(seeds);
        Ok(ex)
    }

    /// The run so far.
    #[must_use]
    pub const fn report(&self) -> &Report {
        &self.report
    }

    /// Whether edge `a -rel- b` has been written.
    #[must_use]
    pub fn has_edge(&self, a: Id, b: Id, rel: Rel) -> bool {
        self.written.contains(&(a.min(b), a.max(b), rel))
    }

    /// Whether `a` and `b` are joined in the graph.
    #[must_use]
    pub fn connected(&self, a: Id, b: Id) -> bool {
        let (Some(x), Some(y)) = (self.known.get(&a), self.known.get(&b)) else {
            return false;
        };
        find(&self.uf, x.uf) == find(&self.uf, y.uf)
    }

    /// Expands until the budget is spent, the frontier is empty, the goal's natural end
    /// (the first match, for [`Goal::Alert`]), or `stop` holds.
    ///
    /// # Errors
    /// Fails on storage errors.
    pub fn run(mut self, mut stop: impl FnMut(&Self) -> bool) -> Result<Report> {
        while self.step()? {
            if stop(&self) {
                self.report.done_at = Some(self.report.fetched);
                break;
            }
            if self.task.goal == Goal::Alert && self.report.alert_at.is_some() {
                self.report.done_at = self.report.alert_at;
                break;
            }
        }
        self.report.wall = self.started.elapsed();
        Ok(self.report)
    }

    /// Writes `id` into the graph if new.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn discover(&mut self, t: &mut Txn<'_>, id: Id) -> Result<()> {
        if self.known.contains_key(&id) {
            return Ok(());
        }
        let dom = self.src.domain(id);
        let name = self
            .src
            .domains()
            .get(usize::from(dom))
            .copied()
            .unwrap_or("?");
        let log = t.node(name.as_bytes(), id.to_string().as_bytes())?.id;
        let mut attrs = Vec::new();
        self.src.attrs(id, &mut attrs);
        let is_watched = self
            .task
            .watch
            .is_some_and(|w| attrs.iter().any(|(k, _)| *k == w));
        for (k, v) in attrs {
            t.set_value(log, k, &Value::from(v))?;
        }
        let peek = self.src.peek(id);
        self.report.calls += 1;
        self.report.nodes += 1;
        let uf = self.uf.len();
        self.uf.push(uf);
        self.size.push(1);
        self.by_log.insert(log, id);
        self.known.insert(
            id,
            Known {
                log,
                dom,
                hops: [UNREACHED; 2],
                cost: [f64::INFINITY; 2],
                peek,
                fetched: [0; crate::MAX_REL],
                degree: 0,
                by_dom: [0; MAX_DOM],
                watched: 0,
                is_watched,
                alerts: 0,
                probes: None,
                version: 0,
                decision: Decision::default(),
                uf,
            },
        );
        Ok(())
    }

    /// Joins the sets of `a` and `b`, the smaller under the larger.
    fn union(&mut self, a: usize, b: usize) {
        let (x, y) = (find(&self.uf, a), find(&self.uf, b));
        if x == y {
            return;
        }
        let size = |r: usize| self.size.get(r).copied().unwrap_or(1);
        let (small, big) = if size(x) < size(y) { (x, y) } else { (y, x) };
        let joined = size(small) + size(big);
        if let Some(p) = self.uf.get_mut(small) {
            *p = big;
        }
        if let Some(s) = self.size.get_mut(big) {
            *s = joined;
        }
    }

    fn candidate(&self, id: Id) -> Option<Candidate> {
        let k = self.known.get(&id)?;
        Some(Candidate {
            id,
            domain: k.dom,
            peek: k.peek,
            fetched: k.fetched,
            hops: k.hops,
            cost: k.cost,
            known: k.degree,
            known_by_domain: k.by_dom,
            watched: k.watched,
            is_watched: k.is_watched,
            alerts: k.alerts,
            probes: k.probes,
        })
    }

    /// Asks the policy about every entity in `ids` that still has edges to fetch.
    fn rescore(&mut self, mut ids: Vec<Id>) {
        ids.sort_unstable();
        ids.dedup();
        let batch: Vec<Candidate> = ids
            .iter()
            .filter(|id| self.known.get(id).is_some_and(|k| !k.exhausted()))
            .filter_map(|&id| self.candidate(id))
            .collect();
        if batch.is_empty() {
            return;
        }
        let started = Instant::now();
        let mut out = Vec::with_capacity(batch.len());
        self.policy.decide(&self.task, &self.ctx, &batch, &mut out);
        self.report.policy += started.elapsed();
        self.report.decisions += u64::try_from(batch.len()).unwrap_or(u64::MAX);
        for (c, d) in batch.iter().zip(out) {
            if let Some(k) = self.known.get_mut(&c.id) {
                k.version += 1;
                k.probes = None;
                let item = Item {
                    priority: d.priority,
                    version: k.version,
                    id: c.id,
                };
                let take = !d.take.is_empty();
                k.decision = d;
                if take {
                    self.heap.push(item);
                }
            }
        }
    }

    /// Runs the probes rooted at `id`, returning their total matches.
    ///
    /// # Errors
    /// Fails on storage errors or a probe that does not parse.
    fn probe(&mut self, id: Id) -> Result<u32> {
        let Some(log) = self.known.get(&id).map(|k| k.log) else {
            return Ok(0);
        };
        let started = Instant::now();
        let r = self.graph.read()?;
        if self.reads == Reads::Projection {
            r.projection()?; // advanced over the commits since; queries below use it
        }
        let mut hits = 0_u32;
        for template in &self.task.probes {
            let src = template.replace("[id]", &log.to_string());
            let p = Pattern::parse(&src).map_err(|e| GraphError::Value(e.to_string()))?;
            r.query(std::slice::from_ref(&p), None, |_, _| {
                hits += 1;
                hits < 1 << 12
            })?;
        }
        drop(r);
        self.report.probes += 1;
        self.report.probe += started.elapsed();
        Ok(hits)
    }

    /// The next entity to expand and what to take of it.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn pop(&mut self) -> Result<Option<Next>> {
        while let Some(item) = self.heap.pop() {
            let Some(k) = self.known.get(&item.id) else {
                continue;
            };
            if k.version != item.version || k.decision.take.is_empty() {
                continue;
            }
            if !self.task.probes.is_empty() && k.probes.is_none() {
                // A live check before paying: the policy may revise its decision.
                let hits = self.probe(item.id)?;
                let Some(mut c) = self.candidate(item.id) else {
                    continue;
                };
                c.probes = Some(hits);
                let started = Instant::now();
                let mut out = Vec::with_capacity(1);
                self.policy
                    .decide(&self.task, &self.ctx, std::slice::from_ref(&c), &mut out);
                self.report.policy += started.elapsed();
                self.report.decisions += 1;
                let d = out.pop().unwrap_or_default();
                let next = self.heap.peek().map_or(f64::NEG_INFINITY, |i| i.priority);
                if let Some(k) = self.known.get_mut(&item.id) {
                    k.probes = Some(hits);
                    k.version += 1;
                    let defer = d.priority < next;
                    let item = Item {
                        priority: d.priority,
                        version: k.version,
                        id: item.id,
                    };
                    let take = d.take.clone();
                    k.decision = d;
                    if take.is_empty() {
                        continue;
                    }
                    if defer {
                        self.heap.push(item);
                        continue;
                    }
                    return Ok(Some((item.id, take)));
                }
                continue;
            }
            return Ok(Some((item.id, k.decision.take.clone())));
        }
        Ok(None)
    }

    /// One expansion step. `false` once the budget is spent or the frontier is empty.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn step(&mut self) -> Result<bool> {
        if self.report.fetched >= self.task.budget {
            return Ok(false);
        }
        let Some((id, take)) = self.pop()? else {
            return Ok(false);
        };
        let Some(me) = self.known.get(&id).cloned() else {
            return Ok(false);
        };
        let side = usize::from(me.hops[1] < me.hops[0]);

        // Fetch what the policy asked for, within the budget.
        let started = Instant::now();
        let mut got: Vec<(Id, Rel)> = Vec::new();
        let mut buf = Vec::new();
        let mut fetched = me.fetched;
        for (r, limit) in take {
            let left = u32::try_from(self.task.budget.saturating_sub(self.report.fetched))
                .unwrap_or(u32::MAX);
            let Some(f) = fetched.get_mut(usize::from(r)) else {
                continue;
            };
            buf.clear();
            self.src.fetch(id, r, *f, limit.min(left), &mut buf);
            self.report.calls += 1;
            let n = u32::try_from(buf.len()).unwrap_or(u32::MAX);
            *f += n;
            self.report.fetched += u64::from(n);
            got.extend(buf.iter().map(|&o| (o, r)));
        }
        self.report.fetch += started.elapsed();
        if let Some(k) = self.known.get_mut(&id) {
            k.fetched = fetched;
        }
        let n = u64::try_from(got.len()).unwrap_or(u64::MAX);
        if let Some(d) = self.report.fetched_by_domain.get_mut(usize::from(me.dom)) {
            *d += n;
        }
        if let Some(s) = self.ctx.by_side.get_mut(side) {
            *s += n;
        }
        self.ctx.fetched = self.report.fetched;
        self.ctx.steps += 1;
        self.report.steps += 1;

        let changed = self.write(id, &me, &got)?;
        let changed = self.watch(changed)?;
        if self.task.goal == Goal::Connect
            && self.report.connected_at.is_none()
            && let [a, b, ..] = self.task.seeds[..]
            && self.connected(a, b)
        {
            self.report.connected_at = Some(self.report.fetched);
        }
        self.rescore(changed);
        Ok(true)
    }

    /// Writes the fetched edges of `id` in one txn and updates what is known. Returns
    /// the entities to rescore.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn write(&mut self, id: Id, me: &Known, got: &[(Id, Rel)]) -> Result<Vec<Id>> {
        let started = Instant::now();
        let mut t = self.graph.write()?;
        let first_new = t.next_id()?;
        for &(o, _) in got {
            self.discover(&mut t, o)?;
        }
        let relations = self.src.relations();
        let mut specs = Vec::with_capacity(got.len());
        for &(o, r) in got {
            let (Some(rel), Some(other)) = (relations.get(usize::from(r)), self.known.get(&o))
            else {
                continue;
            };
            let (src, tgt) = if me.dom == rel.from {
                (me.log, other.log)
            } else {
                (other.log, me.log)
            };
            specs.push(EdgeSpec {
                src,
                tgt,
                ty: rel.name.as_bytes(),
                val: b"",
            });
        }
        let entries = t.edge_batch(&specs)?;
        t.commit()?;
        self.report.write += started.elapsed();

        let mut changed = Vec::with_capacity(got.len() + 1);
        for (e, &(o, r)) in entries.iter().zip(got) {
            let Some(other) = self.known.get(&o).cloned() else {
                continue;
            };
            if e.id >= first_new {
                self.report.edges += 1;
                self.written.insert((id.min(o), id.max(o), r));
                if let Some(k) = self.known.get_mut(&id) {
                    k.degree += 1;
                    if let Some(c) = k.by_dom.get_mut(usize::from(other.dom)) {
                        *c += 1;
                    }
                    k.watched += u32::from(other.is_watched);
                }
                if let Some(k) = self.known.get_mut(&o) {
                    k.degree += 1;
                    if let Some(c) = k.by_dom.get_mut(usize::from(me.dom)) {
                        *c += 1;
                    }
                    k.watched += u32::from(me.is_watched);
                }
                self.union(me.uf, other.uf);
            }
            // Distance from each side, through this entity.
            let step = f64::from(me.peek.iter().sum::<u32>()).ln_1p();
            if let Some(k) = self.known.get_mut(&o) {
                for (h, mine) in k.hops.iter_mut().zip(me.hops) {
                    *h = (*h).min(mine.saturating_add(1));
                }
                for (c, mine) in k.cost.iter_mut().zip(me.cost) {
                    *c = c.min(mine + step);
                }
            }
            changed.push(o);
        }
        changed.push(id);
        Ok(changed)
    }

    /// Runs the standing queries over the log entries since the last step; matches
    /// count against the entities in them, which are added to `changed`.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn watch(&mut self, mut changed: Vec<Id>) -> Result<Vec<Id>> {
        if self.standing.is_empty() {
            return Ok(changed);
        }
        let started = Instant::now();
        let r = self.graph.read()?;
        let end = r.next_id()?;
        let mut hits: Vec<LogId> = Vec::new();
        let mut matches = 0_u64;
        r.mquery(&self.standing, self.queried, None, |_, _, chain| {
            matches += 1;
            hits.extend(chain.iter().map(|e| e.id));
            matches < 1 << 14
        })?;
        drop(r);
        self.queried = end;
        if matches > 0 {
            self.report.alerts += matches;
            if self.report.alert_at.is_none() {
                self.report.alert_at = Some(self.report.fetched);
            }
        }
        for log in hits {
            if let Some(&id) = self.by_log.get(&log) {
                if let Some(k) = self.known.get_mut(&id) {
                    k.alerts += 1;
                }
                changed.push(id);
            }
        }
        self.report.standing += started.elapsed();
        Ok(changed)
    }
}
