//! The expansion loop.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::BuildHasher;
use std::time::{Duration, Instant};

use ramp_graph::lgql::Pattern;
use ramp_graph::value::Value;
use ramp_graph::{EdgeSpec, Graph, GraphError, LogId, Result, Txn};

use crate::{
    Candidate, Context, Counts, Decision, Goal, Id, MAX_DOM, MAX_REL, Policy, Rel, Source, Take,
    Task, UNREACHED,
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
    /// Rounds of concurrent fetches: the source round trips a run waits for.
    pub rounds: u64,
    /// Edges fetched from the source.
    pub fetched: u64,
    /// Edges fetched, by the domain of the entity expanded: where the budget went.
    pub fetched_by_domain: [u64; MAX_DOM],
    /// Source calls (peeks and fetches).
    pub calls: u64,
    /// Filtered fetches the source evaluated (part of `calls`).
    pub filtered: u64,
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
    /// Every expansion step, in order: the training data for a learned policy.
    pub trace: Vec<Step>,
    /// Edges, as `(min, max, rel)`, that are part of a standing-query match.
    pub matched: HashSet<(Id, Id, Rel)>,
}

/// One expansion step: what the policy saw, what it chose, and what came of it.
#[derive(Debug, Clone)]
pub struct Step {
    /// The expanded entity, as the policy saw it when choosing.
    pub candidate: Candidate,
    /// The decision that was carried out.
    pub decision: Decision,
    /// Edges fetched before this step.
    pub before: u64,
    /// Edges this step added to the graph, as `(min, max, rel)`.
    pub edges: Vec<(Id, Id, Rel)>,
}

impl Report {
    /// How many of `step`'s edges ended up in a hit: a standing-query match, or `truth`
    /// (edges known to matter, such as a scenario's planted signal).
    #[must_use]
    pub fn hits<H: BuildHasher>(&self, step: &Step, truth: &HashSet<(Id, Id, Rel), H>) -> usize {
        step.edges
            .iter()
            .filter(|e| self.matched.contains(e) || truth.contains(e))
            .count()
    }
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
    pushed: [bool; MAX_REL],
    version: u32,
    decision: Decision,
    uf: usize,
}

impl Known {
    fn exhausted(&self) -> bool {
        self.peek.iter().zip(&self.fetched).all(|(p, f)| f >= p)
    }

    /// What the policy sees about it.
    const fn candidate(&self, id: Id) -> Candidate {
        Candidate {
            id,
            domain: self.dom,
            peek: self.peek,
            fetched: self.fetched,
            hops: self.hops,
            cost: self.cost,
            known: self.degree,
            known_by_domain: self.by_dom,
            watched: self.watched,
            is_watched: self.is_watched,
            alerts: self.alerts,
            probes: self.probes,
            pushed: self.pushed,
        }
    }
}

/// An entity to expand and what to take of it.
type Next = (Id, Vec<Take>);

/// What one expansion step's fetches returned.
#[derive(Debug, Default)]
struct Pulled {
    /// Each neighbour with its relation and the cost of the hop to it.
    got: Vec<(Id, Rel, f64)>,
    /// The entity's edges per relation fetched, and filtered fetches made, after.
    fetched: Counts,
    pushed: [bool; MAX_REL],
    calls: u64,
    filtered: u64,
}

/// Fetches what `take` asks of `id` (`me`), each limit already within the budget.
fn fetch<S: Source>(src: &S, id: Id, me: &Known, take: &[Take]) -> Pulled {
    let mut out = Pulled {
        fetched: me.fetched,
        pushed: me.pushed,
        ..Pulled::default()
    };
    let degree = f64::from(me.peek.iter().sum::<u32>()).ln_1p();
    let mut buf = Vec::new();
    for t in take {
        let (Some(f), Some(p)) = (
            out.fetched.get_mut(usize::from(t.rel)),
            out.pushed.get_mut(usize::from(t.rel)),
        ) else {
            continue;
        };
        buf.clear();
        out.calls += 1;
        // A filtered fetch returns matches without paging the relation; a source that
        // cannot filter is fetched from unfiltered instead.
        let filtered = t
            .filter
            .as_ref()
            .and_then(|filter| src.fetch_where(id, t.rel, filter, t.limit, &mut buf));
        // Crossing a hub through a filter is as specific as what the filter returned.
        let hop = if filtered.is_some() {
            *p = true;
            out.filtered += 1;
            f64::from(u32::try_from(buf.len()).unwrap_or(u32::MAX)).ln_1p()
        } else {
            src.fetch(id, t.rel, *f, t.limit, &mut buf);
            *f += u32::try_from(buf.len()).unwrap_or(u32::MAX);
            degree
        };
        out.got.extend(buf.iter().map(|&o| (o, t.rel, hop)));
    }
    out
}

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
    /// Steps fetched concurrently per round.
    parallel: usize,
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
    ctx: Context<'a>,
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
                relations: src.relations(),
                filters: src.can_filter(),
                budget: task.budget,
                ..Context::default()
            },
            task,
            reads,
            parallel: 1,
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

    /// Fetches up to `k` (at least 1) of the best candidates concurrently per round,
    /// for a source whose latency dominates. The rest of each round's picks are made
    /// before the first one's results are known, so a round of `k` spends the budget a
    /// little less well than `k` rounds of one.
    #[must_use]
    pub fn parallel(mut self, k: usize) -> Self {
        self.parallel = k.max(1);
        self
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
                fetched: [0; MAX_REL],
                degree: 0,
                by_dom: [0; MAX_DOM],
                watched: 0,
                is_watched,
                alerts: 0,
                probes: None,
                pushed: [false; MAX_REL],
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
        Some(self.known.get(&id)?.candidate(id))
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

    /// One round: up to [`Expander::parallel`] expansion steps, their fetches made
    /// concurrently, then written and re-scored in order. `false` once the budget is
    /// spent or the frontier is empty.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn step(&mut self) -> Result<bool> {
        // Pop the best candidates, each with its share of the budget reserved, so
        // concurrent fetches cannot overspend it.
        let mut left = self.task.budget.saturating_sub(self.report.fetched);
        // A round takes only candidates near its best: one far behind would likely
        // not be chosen once the best's results are in.
        let mut plans: Vec<(Id, Known, Vec<Take>)> = Vec::new();
        let mut floor = f64::NEG_INFINITY;
        while plans.len() < self.parallel && left > 0 {
            let Some((id, mut take)) = self.pop()? else {
                break;
            };
            let Some(me) = self.known.get(&id).cloned() else {
                continue;
            };
            let priority = me.decision.priority;
            if plans.is_empty() {
                floor = priority - self.policy.spread();
            } else if priority < floor {
                self.heap.push(Item {
                    priority,
                    version: me.version,
                    id,
                });
                break;
            }
            for t in &mut take {
                t.limit = t.limit.min(u32::try_from(left).unwrap_or(u32::MAX));
                left -= u64::from(t.limit);
            }
            plans.push((id, me, take));
        }
        if plans.is_empty() {
            return Ok(false);
        }

        let started = Instant::now();
        let src = self.src;
        let results: Vec<Pulled> = if let [(id, me, take)] = &*plans {
            vec![fetch(src, *id, me, take)]
        } else {
            std::thread::scope(|scope| {
                #[expect(
                    clippy::needless_collect,
                    reason = "spawn every fetch before joining any"
                )]
                let running: Vec<_> = plans
                    .iter()
                    .map(|(id, me, take)| scope.spawn(move || fetch(src, *id, me, take)))
                    .collect();
                running
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                    .collect()
            })
        };
        self.report.fetch += started.elapsed();
        self.report.rounds += 1;

        let mut changed = Vec::new();
        for ((id, me, take), got) in plans.into_iter().zip(results) {
            changed.extend(self.apply(id, &me, take, &got)?);
        }
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

    /// Accounts for and writes one expansion step's fetch. Returns the entities to
    /// rescore.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn apply(&mut self, id: Id, me: &Known, take: Vec<Take>, f: &Pulled) -> Result<Vec<Id>> {
        let side = usize::from(me.hops[1] < me.hops[0]);
        let mut trace = Step {
            candidate: me.candidate(id),
            decision: Decision {
                priority: me.decision.priority,
                take,
            },
            before: self.report.fetched,
            edges: Vec::new(),
        };
        let n = u64::try_from(f.got.len()).unwrap_or(u64::MAX);
        self.report.calls += f.calls;
        self.report.filtered += f.filtered;
        self.report.fetched += n;
        if let Some(k) = self.known.get_mut(&id) {
            k.fetched = f.fetched;
            k.pushed = f.pushed;
        }
        if let Some(d) = self.report.fetched_by_domain.get_mut(usize::from(me.dom)) {
            *d += n;
        }
        if let Some(s) = self.ctx.by_side.get_mut(side) {
            *s += n;
        }
        self.ctx.fetched = self.report.fetched;
        self.ctx.steps += 1;
        self.report.steps += 1;
        let changed = self.write(id, me, &f.got, &mut trace.edges)?;
        self.report.trace.push(trace);
        Ok(changed)
    }

    /// Writes the fetched edges of `id` in one txn and updates what is known. Returns
    /// the entities to rescore.
    ///
    /// # Errors
    /// Fails on storage errors.
    fn write(
        &mut self,
        id: Id,
        me: &Known,
        got: &[(Id, Rel, f64)],
        new: &mut Vec<(Id, Id, Rel)>,
    ) -> Result<Vec<Id>> {
        let started = Instant::now();
        let mut t = self.graph.write()?;
        let first_new = t.next_id()?;
        for &(o, _, _) in got {
            self.discover(&mut t, o)?;
        }
        let relations = self.src.relations();
        let mut specs = Vec::with_capacity(got.len());
        for &(o, r, _) in got {
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
        for (e, &(o, r, step)) in entries.iter().zip(got) {
            let Some(other) = self.known.get(&o).cloned() else {
                continue;
            };
            if e.id >= first_new {
                self.report.edges += 1;
                let edge = (id.min(o), id.max(o), r);
                self.written.insert(edge);
                new.push(edge);
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
        // Each match's entities, in order, then `None`.
        let mut hits: Vec<Option<LogId>> = Vec::new();
        let mut matches = 0_u64;
        r.mquery(&self.standing, self.queried, None, |_, _, chain| {
            matches += 1;
            hits.extend(chain.iter().map(|e| Some(e.id)));
            hits.push(None);
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
        // The edges a match runs along: those written between consecutive entities.
        let ids: Vec<Option<Id>> = hits
            .iter()
            .map(|h| h.and_then(|log| self.by_log.get(&log).copied()))
            .collect();
        for pair in ids.windows(2) {
            if let [Some(a), Some(b)] = *pair {
                for r in (0..MAX_REL).filter_map(|r| Rel::try_from(r).ok()) {
                    let edge = (a.min(b), a.max(b), r);
                    if self.written.contains(&edge) {
                        self.report.matched.insert(edge);
                    }
                }
            }
        }
        for log in hits.into_iter().flatten() {
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
