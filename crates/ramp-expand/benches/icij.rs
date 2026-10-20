//! Expansion on a real graph: the ICIJ Offshore Leaks database. Officers, entities,
//! intermediaries and addresses, ~2M nodes and ~3.3M relationships, with real hubs:
//! registered agents with 100k+ entities and addresses shared by thousands.
//!
//! `cargo bench -p ramp-expand --bench icij -- DIR [TASKS [BUDGET]]`, with DIR holding
//! the unzipped CSVs from <https://offshoreleaks-data.icij.org> (Open Database License; not in the repo).
//! TASKS = 10 sampled tasks per kind, BUDGET = 200,000 fetched edges per run.
//! `RAMP_SEED` (default 7) seeds the task sampler, for a disjoint set of tasks.
//!
//! Tasks are sampled by [`ramp_expand::icij`]. Connect runs are scored by the edges
//! fetched until the seeds are joined, and by how specific the joining path is: its
//! cost against the cheapest path in the whole graph, and the widest hub it crosses.
//! Alert runs are scored by the edges fetched until the standing query fires.
//!
//! With `RAMP_TRACE=dir`, each run's decision trace goes to
//! `<kind>.<task>.<policy>.jsonl`, labelled by the joining path (connect) or the
//! standing query's match (alert). With `RAMP_BATCHES=dir`, every policy call goes to
//! the same name, each candidate labelled by an oracle that sees the whole graph:
//! [`near_cheapest`] for connect, [`shared_with_target`] for alert. Every candidate's
//! text goes to `entities.jsonl` there, and every task (prompt and seeds, under its
//! `<kind>.<task>` run name) to `tasks.jsonl`.
//!
//! With `RAMP_SLACK=dir`, it writes only `dir/slack.jsonl`: each connect task's oracle,
//! every entity with its [`path_slack`], to weight positives by how near the cheapest
//! path they lie. No policy runs.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use ramp_expand::icij::{
    DOMAINS, alert_tasks, connect_tasks, joining, load, near_cheapest, path_slack,
    shared_with_target,
};
use ramp_expand::policy::{Bfs, Guided, Rules};
use ramp_expand::world::{Rng, World};
use ramp_expand::{Expander, Goal, Id, Policy, Reads, Source, Task, wire};
use ramp_graph::{Graph, GraphError, Result};

fn io(e: impl std::fmt::Display) -> GraphError {
    GraphError::Io(e.to_string())
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
    seen: &mut HashSet<Id>,
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
        path = joining(world, ex.edges(), from, to).map(|(cost, hub, edges)| {
            truth = edges;
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
        let rows = wire::batches(&report, world, |id| oracle.contains(&id));
        seen.extend(
            report
                .batches
                .iter()
                .flat_map(|b| &b.candidates)
                .map(|c| c.id),
        );
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
fn connect(world: &World, tasks: &[(Task, (f64, u32))], seen: &mut HashSet<Id>) -> Result<()> {
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
            let (done, path) = run(world, task, nth(i, 3, false)?, &label, oracle, seen)?;
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
fn alert(world: &World, tasks: &[Task], seen: &mut HashSet<Id>) -> Result<()> {
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
            let (done, _) = run(world, task, nth(i, 2, true)?, &label, oracle, seen)?;
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
    let sampler = std::env::var("RAMP_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let mut rng = Rng::new(sampler);
    let connects = connect_tasks(&world, n, &mut rng, budget);
    let alerts = alert_tasks(&world, n, &mut rng, budget);
    if let Some(out) = std::env::var_os("RAMP_BATCHES") {
        let all = (connects.iter().map(|(t, _)| ("connect", t)))
            .zip(0..)
            .chain(alerts.iter().map(|t| ("alert", t)).zip(0..));
        let mut rows = Vec::new();
        for ((kind, task), i) in all {
            let row =
                serde_json::json!({"run": format!("{kind}.{i}"), "task": wire::task(task, &world)});
            rows.extend(row.to_string().into_bytes());
            rows.push(b'\n');
        }
        std::fs::write(Path::new(&out).join("tasks.jsonl"), rows).map_err(io)?;
    }
    if let Some(out) = std::env::var_os("RAMP_SLACK") {
        // Only each connect task's oracle with its slack, for weighting positives.
        let mut rows = Vec::new();
        for (i, (task, (best, _))) in connects.iter().enumerate() {
            if let [a, b] = task.seeds[..] {
                let slack = path_slack(&world, a, b, *best);
                let row = serde_json::json!({"run": format!("connect.{i}"), "seeds": task.seeds, "slack": slack});
                rows.extend(row.to_string().into_bytes());
                rows.push(b'\n');
            }
        }
        std::fs::write(Path::new(&out).join("slack.jsonl"), rows).map_err(io)?;
        return Ok(());
    }
    let mut seen = HashSet::new();
    connect(&world, &connects, &mut seen)?;
    alert(&world, &alerts, &mut seen)?;
    if let Some(out) = std::env::var_os("RAMP_BATCHES") {
        let mut ids: Vec<Id> = seen.into_iter().collect();
        ids.sort_unstable();
        let rows = wire::entities(&world, &ids);
        std::fs::write(Path::new(&out).join("entities.jsonl"), rows).map_err(io)?;
    }
    Ok(())
}
