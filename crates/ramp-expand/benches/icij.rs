//! Expansion on a real graph: the ICIJ Offshore Leaks database. Officers, entities,
//! intermediaries and addresses, ~2M nodes and ~3.3M relationships, with real hubs:
//! registered agents with 100k+ entities and addresses shared by thousands.
//!
//! `cargo bench -p ramp-expand --bench icij -- DIR [TASKS [BUDGET]]`, with DIR holding
//! the unzipped CSVs from <https://offshoreleaks-data.icij.org> (Open Database License; not in the repo).
//! TASKS = 10 sampled tasks per kind, BUDGET = 200,000 fetched edges per run.
//!
//! Nothing is planted, so the tasks come from the data:
//! - **connect:** two officers 4 to 6 hops apart. Scored by the edges fetched until
//!   they are joined, and by how specific the joining path is: its cost (the sum of
//!   `ln(1 + degree)` over the entities it passes through), against the cheapest path
//!   in the whole graph, and the widest hub it crosses.
//! - **alert:** from an officer, a co-officer (sharing an entity) from the country
//!   rarest among them, as a standing query. Scored by edges fetched until it fires.
//!
//! With `RAMP_TRACE=dir`, each run's decision trace goes to
//! `<kind>.<task>.<policy>.jsonl`, labelled by the joining path (connect) or the
//! standing query's match (alert). With `RAMP_BATCHES=dir`, every policy call goes to
//! the same name, each candidate labelled by an oracle that sees the whole graph:
//! - **connect:** a seed, or on a path between the seeds costing at most
//!   `1 + EPS` times the cheapest.
//! - **alert:** the seed, an entity it shares with an officer from the target country,
//!   or such an officer.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use ramp_expand::policy::{Bfs, Guided, Rules};
use ramp_expand::world::{Builder, World};
use ramp_expand::{
    Dom, Expander, Filter, Goal, Id, MAX_DOM, MAX_REL, Policy, Reads, Rel, Relation, Source, Task,
    wire,
};
use ramp_graph::{Graph, GraphError, Result};

const DOMAINS: [&str; 5] = ["entity", "officer", "intermediary", "address", "other"];
const ENTITY: Dom = 0;
const OFFICER: Dom = 1;
const INTERMEDIARY: Dom = 2;
const ADDRESS: Dom = 3;
const OTHER: Dom = 4;

const OFFICER_OF: Rel = 0;
const RELATIONS: [Relation; 6] = [
    Relation {
        name: "officer_of",
        from: OFFICER,
        to: ENTITY,
    },
    Relation {
        name: "intermediary_of",
        from: INTERMEDIARY,
        to: ENTITY,
    },
    Relation {
        name: "registered_address",
        from: ENTITY,
        to: ADDRESS,
    },
    Relation {
        name: "same_as",
        from: OFFICER,
        to: OFFICER,
    },
    Relation {
        name: "similar",
        from: ENTITY,
        to: ENTITY,
    },
    Relation {
        name: "connected_to",
        from: ENTITY,
        to: ENTITY,
    },
];

/// The relation group of an ICIJ `rel_type`.
fn relation(ty: &str) -> Option<Rel> {
    Some(match ty {
        "officer_of" => 0,
        "intermediary_of" => 1,
        "registered_address" => 2,
        "same_name_as"
        | "same_as"
        | "same_id_as"
        | "probably_same_officer_as"
        | "same_intermediary_as"
        | "same_address_as" => 3,
        "similar" | "similar_company_as" | "same_company_as" => 4,
        "connected_to" | "underlying" => 5,
        _ => return None,
    })
}

/// CSV records: quoted fields may hold commas, doubled quotes and newlines.
fn records(text: &str) -> impl Iterator<Item = Vec<String>> + '_ {
    let mut chars = text.chars().peekable();
    std::iter::from_fn(move || {
        chars.peek()?;
        let (mut rec, mut field, mut quoted) = (Vec::new(), String::new(), false);
        while let Some(c) = chars.next() {
            match (c, quoted) {
                ('"', true) if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                ('"', _) => quoted = !quoted,
                (',', false) => rec.push(std::mem::take(&mut field)),
                ('\n', false) => break,
                ('\r', false) => {}
                (c, _) => field.push(c),
            }
        }
        rec.push(field);
        Some(rec)
    })
}

fn io(e: impl std::fmt::Display) -> GraphError {
    GraphError::Io(e.to_string())
}

/// Loads the CSVs in `dir`; officers carry `country` (their `country_codes`).
///
/// # Errors
/// A file that cannot be read.
fn load(dir: &Path) -> Result<World> {
    let mut b = Builder::empty(&DOMAINS, &RELATIONS);
    let mut ids: HashMap<u64, Id> = HashMap::new();
    for (file, d) in [
        ("nodes-entities.csv", ENTITY),
        ("nodes-officers.csv", OFFICER),
        ("nodes-intermediaries.csv", INTERMEDIARY),
        ("nodes-addresses.csv", ADDRESS),
        ("nodes-others.csv", OTHER),
    ] {
        let text = std::fs::read_to_string(dir.join(file)).map_err(io)?;
        let mut rows = records(&text);
        let header = rows.next().unwrap_or_default();
        let country = header.iter().position(|h| h == "country_codes");
        for row in rows {
            let Some(n) = row.first().and_then(|n| n.parse::<u64>().ok()) else {
                continue;
            };
            if ids.contains_key(&n) {
                continue;
            }
            let id = b.entity(d);
            ids.insert(n, id);
            if let Some(c) = country
                .and_then(|i| row.get(i))
                .filter(|c| d == OFFICER && !c.is_empty())
            {
                b.attr(id, "country", c);
            }
        }
    }
    let text = std::fs::read_to_string(dir.join("relationships.csv")).map_err(io)?;
    for row in records(&text).skip(1) {
        let (Some(a), Some(z), Some(r)) = (row.first(), row.get(1), row.get(2)) else {
            continue;
        };
        let (Ok(a), Ok(z), Some(r)) = (a.parse::<u64>(), z.parse::<u64>(), relation(r)) else {
            continue;
        };
        if let (Some(&a), Some(&z)) = (ids.get(&a), ids.get(&z))
            && a != z
        {
            b.link(a, z, r);
        }
    }
    Ok(b.build())
}

/// Every neighbour of `id`.
fn neighbours(w: &World, id: Id, out: &mut Vec<Id>) {
    out.clear();
    for r in (0..RELATIONS.len()).filter_map(|r| Rel::try_from(r).ok()) {
        w.fetch(id, r, 0, u32::MAX, out);
    }
}

/// splitmix64.
struct Rng(u64);

impl Rng {
    const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n.max(1)).unwrap_or(1)).unwrap_or(0)
    }
}

fn officers(w: &World) -> Vec<Id> {
    (0..w.len())
        .filter_map(|i| Id::try_from(i).ok())
        .filter(|&i| w.domain(i) == OFFICER && w.degree(i) > 0)
        .collect()
}

/// Hops from `from` to everything within `depth`.
fn hops(w: &World, from: Id, depth: u8) -> HashMap<Id, u8> {
    let mut seen = HashMap::from([(from, 0_u8)]);
    let (mut layer, mut buf) = (vec![from], Vec::new());
    for d in 1..=depth {
        let mut next = Vec::new();
        for &x in &layer {
            neighbours(w, x, &mut buf);
            for &y in &buf {
                seen.entry(y).or_insert_with(|| {
                    next.push(y);
                    d
                });
            }
        }
        layer = next;
    }
    seen
}

/// The path from `a` to `b` minimising the sum of `ln(1 + degree)` over the entities
/// it passes through, over `adj` (all of `w` if `None`): its cost, its widest hub, and
/// its entities from `b` back to `a`.
fn cheapest(
    w: &World,
    adj: Option<&HashMap<Id, Vec<Id>>>,
    a: Id,
    b: Id,
) -> Option<(f64, u32, Vec<Id>)> {
    let weight = |x: Id| f64::from(w.degree(x)).ln_1p();
    let mut best: HashMap<Id, (f64, Id)> = HashMap::from([(a, (0.0, a))]);
    let mut heap = BinaryHeap::from([(Reverse(Ord64(0.0)), a)]);
    let mut buf = Vec::new();
    while let Some((Reverse(Ord64(c)), x)) = heap.pop() {
        if x == b {
            let mut widest = 0;
            let mut path = vec![b];
            let mut at = best.get(&b)?.1;
            while at != a {
                widest = widest.max(w.degree(at));
                path.push(at);
                at = best.get(&at)?.1;
            }
            path.push(a);
            return Some((c, widest, path));
        }
        if best.get(&x).is_some_and(|&(k, _)| k < c) {
            continue;
        }
        let step = if x == a { 0.0 } else { weight(x) };
        match adj {
            Some(adj) => {
                buf.clear();
                buf.extend(adj.get(&x).into_iter().flatten());
            }
            None => neighbours(w, x, &mut buf),
        }
        for &y in &buf {
            let k = c + step;
            if best.get(&y).is_none_or(|&(old, _)| k < old) {
                best.insert(y, (k, x));
                heap.push((Reverse(Ord64(k)), y));
            }
        }
    }
    None
}

/// How much dearer than the cheapest a path may be and still label its entities.
const EPS: f64 = 0.1;

/// The cost of the cheapest path from `from` to everything it reaches for at most
/// `bound`, counted as in [`cheapest`]: the entities passed through, not the ends.
fn within(w: &World, from: Id, bound: f64) -> HashMap<Id, f64> {
    let mut best: HashMap<Id, f64> = HashMap::from([(from, 0.0)]);
    let mut heap = BinaryHeap::from([(Reverse(Ord64(0.0)), from)]);
    let mut buf = Vec::new();
    while let Some((Reverse(Ord64(c)), x)) = heap.pop() {
        if best.get(&x).is_some_and(|&k| k < c) {
            continue;
        }
        let k = c + if x == from {
            0.0
        } else {
            f64::from(w.degree(x)).ln_1p()
        };
        if k > bound {
            continue;
        }
        neighbours(w, x, &mut buf);
        for &y in &buf {
            if best.get(&y).is_none_or(|&old| k < old) {
                best.insert(y, k);
                heap.push((Reverse(Ord64(k)), y));
            }
        }
    }
    best
}

/// The connect oracle: the seeds, and every entity on a path between them costing at
/// most `1 + EPS` times the cheapest, `best`.
fn near_cheapest(w: &World, a: Id, b: Id, best: f64) -> HashSet<Id> {
    let bound = best.mul_add(1.0 + EPS, 1e-9);
    let (from_a, from_b) = (within(w, a, bound), within(w, b, bound));
    let mut out: HashSet<Id> = from_a
        .iter()
        .filter(|&(&x, &ca)| {
            from_b
                .get(&x)
                .is_some_and(|&cb| ca + cb + f64::from(w.degree(x)).ln_1p() <= bound)
        })
        .map(|(&x, _)| x)
        .collect();
    out.extend([a, b]);
    out
}

/// The alert oracle: the seed, each entity it shares with an officer from the target
/// country, and those officers.
fn shared_with_target(w: &World, task: &Task) -> HashSet<Id> {
    let (Some(&s), Some(Filter::Attr { key, value })) = (task.seeds.first(), &task.target) else {
        return HashSet::new();
    };
    let mut out = HashSet::from([s]);
    let (mut entities, mut co, mut attrs) = (Vec::new(), Vec::new(), Vec::new());
    w.fetch(s, OFFICER_OF, 0, u32::MAX, &mut entities);
    for &e in &entities {
        co.clear();
        w.fetch(e, OFFICER_OF, 0, u32::MAX, &mut co);
        for &o in co.iter().filter(|&&o| o != s) {
            attrs.clear();
            w.attrs(o, &mut attrs);
            if attrs.iter().any(|(k, v)| k == key && v == value) {
                out.extend([e, o]);
            }
        }
    }
    out
}

/// An `f64` ordered by `total_cmp`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ord64(f64);

impl Eq for Ord64 {}

impl PartialOrd for Ord64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ord64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

const fn task(seeds: Vec<Id>, goal: Goal, relation_interest: [f64; MAX_REL], budget: u64) -> Task {
    Task {
        seeds,
        goal,
        relation_interest,
        domain_interest: [1.0; MAX_DOM],
        watch: None,
        standing: Vec::new(),
        target: None,
        shape: Vec::new(),
        probes: Vec::new(),
        budget,
    }
}

/// Connect tasks: officer pairs 4 to 6 hops apart, with their cheapest path.
fn connect_tasks(w: &World, n: usize, rng: &mut Rng, budget: u64) -> Vec<(Task, (f64, u32))> {
    let officers = officers(w);
    let mut out = Vec::new();
    while out.len() < n {
        let Some(&a) = officers.get(rng.below(officers.len())) else {
            break;
        };
        let mut far: Vec<Id> = hops(w, a, 6)
            .into_iter()
            .filter(|&(x, d)| d >= 4 && w.domain(x) == OFFICER)
            .map(|(x, _)| x)
            .collect();
        far.sort_unstable();
        let Some(&b) = far.get(rng.below(far.len())) else {
            continue;
        };
        if let Some((cost, hub, _)) = cheapest(w, None, a, b) {
            out.push((
                task(vec![a, b], Goal::Connect, [1.0; MAX_REL], budget),
                (cost, hub),
            ));
        }
    }
    out
}

/// Alert tasks: from an officer with at least 50 co-officers, one from the country
/// rarest among them (and not the seed's own).
fn alert_tasks(w: &World, n: usize, rng: &mut Rng, budget: u64) -> Vec<Task> {
    let officers = officers(w);
    let country = |x: Id| {
        let mut a = Vec::new();
        w.attrs(x, &mut a);
        a.into_iter().find(|(k, _)| *k == "country").map(|(_, v)| v)
    };
    let (mut out, mut entities, mut co) = (Vec::new(), Vec::new(), Vec::new());
    let mut tries = 0;
    while out.len() < n && tries < 1_000_000 {
        tries += 1;
        let Some(&s) = officers.get(rng.below(officers.len())) else {
            break;
        };
        entities.clear();
        w.fetch(s, OFFICER_OF, 0, u32::MAX, &mut entities);
        let mut counts: HashMap<String, u32> = HashMap::new();
        for &e in &entities {
            co.clear();
            w.fetch(e, OFFICER_OF, 0, u32::MAX, &mut co);
            for &o in co.iter().filter(|&&o| o != s) {
                if let Some(c) = country(o) {
                    *counts.entry(c).or_default() += 1;
                }
            }
        }
        if counts.values().sum::<u32>() < 50 {
            continue;
        }
        let own = country(s);
        let Some((rare, _)) = counts
            .into_iter()
            .filter(|(c, _)| Some(c) != own.as_ref())
            .min_by(|x, y| x.1.cmp(&y.1).then_with(|| x.0.cmp(&y.0)))
        else {
            continue;
        };
        let mut interest = [0.2; MAX_REL];
        if let Some(i) = interest.get_mut(usize::from(OFFICER_OF)) {
            *i = 1.0;
        }
        let mut t = task(vec![s], Goal::Alert, interest, budget);
        t.standing = vec![format!(
            "n(type='officer', value='{s}')-n(type='entity')-n(type='officer', country='{}')",
            rare.replace('\'', "\\'")
        )];
        t.shape = vec![vec![OFFICER, ENTITY, OFFICER]];
        t.target = Some(Filter::Attr {
            key: "country".to_owned(),
            value: rare,
        });
        out.push(t);
    }
    out
}

fn policies(depth: u32, pushdown: bool) -> Vec<Box<dyn Policy>> {
    let mut out: Vec<Box<dyn Policy>> = vec![Box::new(Bfs { depth })];
    for cap in [100, 1000, 10_000] {
        out.push(Box::new(Rules { depth, cap }));
    }
    out.push(Box::new(Guided::default()));
    if pushdown {
        out.push(Box::new(Guided { pushdown: true }));
    }
    out
}

/// `n` as a float (a count well under 2^32).
fn float(n: impl TryInto<u32>) -> f64 {
    f64::from(n.try_into().unwrap_or(u32::MAX))
}

/// The `pct` percentile of `v`.
fn percentile(v: &[f64], pct: usize) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(v.len().saturating_sub(1) * pct / 100)
        .copied()
        .unwrap_or(f64::NAN)
}

/// When a run's goal was met (edges fetched), and for a connect task, the joining
/// path's cost and widest hub.
type Outcome = (Option<u64>, Option<(f64, u32)>);

/// Runs `task` with policy `policy`, writing its trace as `<label>.jsonl` under
/// `RAMP_TRACE` if set, and its policy calls labelled by `oracle` under `RAMP_BATCHES`.
///
/// # Errors
/// Storage errors.
fn run(
    world: &World,
    task: &Task,
    policy: Box<dyn Policy>,
    label: &str,
    oracle: &HashSet<Id>,
) -> Result<Outcome> {
    let dir = tempfile::tempdir().map_err(io)?;
    let graph = Graph::open(dir.path().join("g.db"))?;
    let mut path = None;
    let mut truth = HashSet::new();
    let mut ex = Expander::new(world, &graph, policy, task.clone(), Reads::Projection)?;
    let batches = std::env::var_os("RAMP_BATCHES");
    if batches.is_some() {
        ex = ex.log_batches();
    }
    let report = ex.run(|ex| {
        let [from, to] = task.seeds[..] else {
            return false;
        };
        if !ex.connected(from, to) {
            return false;
        }
        let mut adj: HashMap<Id, Vec<Id>> = HashMap::new();
        for &(x, y, _) in ex.edges() {
            adj.entry(x).or_default().push(y);
            adj.entry(y).or_default().push(x);
        }
        path = cheapest(world, Some(&adj), from, to).map(|(cost, hub, on)| {
            let pairs: HashSet<(Id, Id)> = on
                .windows(2)
                .filter_map(|p| match *p {
                    [x, y] => Some((x.min(y), x.max(y))),
                    _ => None,
                })
                .collect();
            truth = ex
                .edges()
                .filter(|&&(x, y, _)| pairs.contains(&(x, y)))
                .copied()
                .collect();
            (cost, hub)
        });
        true
    })?;
    let file: String = format!("{label}.jsonl")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    if let Some(out) = std::env::var_os("RAMP_TRACE") {
        std::fs::write(Path::new(&out).join(&file), wire::trace(&report, &truth)).map_err(io)?;
    }
    if let Some(out) = batches {
        let rows = wire::batches(&report, |c| oracle.contains(&c.id));
        std::fs::write(Path::new(&out).join(&file), rows).map_err(io)?;
    }
    let done = if task.goal == Goal::Alert {
        report.alert_at
    } else {
        report.done_at
    };
    Ok((done, path))
}

/// Policy `i` of [`policies`].
///
/// # Errors
/// If there is none.
fn nth(i: usize, depth: u32, pushdown: bool) -> Result<Box<dyn Policy>> {
    policies(depth, pushdown)
        .into_iter()
        .nth(i)
        .ok_or_else(|| GraphError::Value(format!("no policy {i}")))
}

/// Every policy on connect tasks.
///
/// # Errors
/// Storage errors.
#[expect(clippy::print_stdout, reason = "benchmark report")]
fn connect(world: &World, tasks: &[(Task, (f64, u32))]) -> Result<()> {
    let costs: Vec<f64> = tasks.iter().map(|(_, (c, _))| *c).collect();
    let widest: Vec<f64> = tasks.iter().map(|(_, (_, h))| f64::from(*h)).collect();
    println!(
        "== connect — two officers 4–6 hops apart; cheapest path: median cost {:.1}, median widest hub {:.0}",
        percentile(&costs, 50),
        percentile(&widest, 50)
    );
    println!(
        "   {:<24} {:>7} {:>9} {:>9} {:>13} {:>13} {:>12}",
        "policy", "solved", "fetched", "p90", "cheapest", "mean cost/best", "widest hub"
    );
    let oracles: Vec<HashSet<Id>> = tasks
        .iter()
        .map(|(task, (best, _))| match task.seeds[..] {
            [a, b] if std::env::var_os("RAMP_BATCHES").is_some() => {
                near_cheapest(world, a, b, *best)
            }
            _ => HashSet::new(),
        })
        .collect();
    for (i, policy) in policies(3, false).iter().enumerate() {
        let (mut solved, mut fetched, mut ratio, mut widest) =
            (0_usize, Vec::new(), Vec::new(), Vec::new());
        for ((t, (task, (best, _))), oracle) in tasks.iter().enumerate().zip(&oracles) {
            let label = format!("connect.{t}.{}", policy.name());
            let (done, path) = run(world, task, nth(i, 3, false)?, &label, oracle)?;
            fetched.push(done.map_or(f64::INFINITY, float));
            if let (Some(_), Some((cost, hub))) = (done, path) {
                solved += 1;
                ratio.push(if *best > 0.0 { cost / best } else { 1.0 });
                widest.push(f64::from(hub));
            }
        }
        let cheapest = ratio.iter().filter(|&&r| r < 1.0 + 1e-9).count();
        let mean = ratio.iter().sum::<f64>() / float(ratio.len().max(1));
        println!(
            "   {:<24} {:>4}/{:<2} {:>9.0} {:>9.0} {:>9}/{:<3} {:>13.2} {:>12.0}",
            policy.name(),
            solved,
            tasks.len(),
            percentile(&fetched, 50),
            percentile(&fetched, 90),
            cheapest,
            solved,
            mean,
            percentile(&widest, 50)
        );
    }
    Ok(())
}

/// Every policy on alert tasks.
///
/// # Errors
/// Storage errors.
#[expect(clippy::print_stdout, reason = "benchmark report")]
fn alert(world: &World, tasks: &[Task]) -> Result<()> {
    println!("\n== alert — from an officer, a co-officer from the country rarest among them");
    println!(
        "   {:<24} {:>7} {:>9} {:>9}",
        "policy", "solved", "fetched", "p90"
    );
    let oracles: Vec<HashSet<Id>> = tasks
        .iter()
        .map(|task| shared_with_target(world, task))
        .collect();
    for (i, policy) in policies(2, true).iter().enumerate() {
        let (mut solved, mut fetched) = (0_usize, Vec::new());
        for ((t, task), oracle) in tasks.iter().enumerate().zip(&oracles) {
            let label = format!("alert.{t}.{}", policy.name());
            let (done, _) = run(world, task, nth(i, 2, true)?, &label, oracle)?;
            solved += usize::from(done.is_some());
            fetched.push(done.map_or(f64::INFINITY, float));
        }
        println!(
            "   {:<24} {:>4}/{:<2} {:>9.0} {:>9.0}",
            policy.name(),
            solved,
            tasks.len(),
            percentile(&fetched, 50),
            percentile(&fetched, 90)
        );
    }
    Ok(())
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).filter(|a| !a.starts_with('-'));
    let Some(dir) = args.next() else {
        println!("usage: icij DIR [TASKS [BUDGET]] (DIR: the unzipped Offshore Leaks CSVs)");
        return Ok(());
    };
    let n = args.next().and_then(|a| a.parse().ok()).unwrap_or(10);
    let budget = args.next().and_then(|a| a.parse().ok()).unwrap_or(200_000);
    let started = Instant::now();
    let world = load(Path::new(&dir))?;
    let mut by_degree: Vec<Id> = (0..world.len())
        .filter_map(|i| Id::try_from(i).ok())
        .collect();
    by_degree.sort_by_key(|&i| Reverse(world.degree(i)));
    let hubs: Vec<String> = by_degree
        .iter()
        .take(5)
        .map(|&i| {
            format!(
                "{} {}",
                DOMAINS.get(usize::from(world.domain(i))).unwrap_or(&"?"),
                world.degree(i)
            )
        })
        .collect();
    println!(
        "icij: {} nodes, {} edges, loaded in {:.1}s; biggest hubs: {}",
        world.len(),
        world.edges(),
        started.elapsed().as_secs_f64(),
        hubs.join(", ")
    );
    println!("budget: {budget} fetched edges per run, {n} tasks per kind\n");
    let mut rng = Rng(7);
    connect(&world, &connect_tasks(&world, n, &mut rng, budget))?;
    alert(&world, &alert_tasks(&world, n, &mut rng, budget))
}
