//! Expansion scenarios: every policy on every planted scenario, then the read path.
//!
//! `cargo bench -p ramp-expand --bench expand [-- N [BUDGET]]` (N = 2,000,000 world
//! entities, BUDGET = 400,000 fetched edges per run).

use std::time::{Duration, Instant};

use ramp_expand::policy::{Bfs, Guided, Rules};
use ramp_expand::scenario::{self, Scenario};
use ramp_expand::world::{Builder, World, dom};
use ramp_expand::{Expander, Policy, Reads, Report, Source};
use ramp_graph::{Graph, GraphError, Result};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Runs `s` with `policy`; returns the report and the share of signal edges recovered.
///
/// # Errors
/// Storage errors.
fn run(
    world: &World,
    s: &Scenario,
    policy: Box<dyn Policy>,
    budget: u64,
    reads: Reads,
) -> Result<(Report, usize)> {
    let dir = tempfile::tempdir().map_err(|e| GraphError::Io(e.to_string()))?;
    let g = Graph::open(dir.path().join("g.db"))?;
    let mut task = s.task.clone();
    task.budget = budget;
    let mut recovered = 0;
    let ex = Expander::new(world, &g, policy, task, reads)?;
    let report = ex.run(|ex| {
        recovered = s
            .signal
            .iter()
            .filter(|&&(a, b, r)| ex.has_edge(a, b, r))
            .count();
        !s.signal.is_empty() && recovered == s.signal.len()
    })?;
    Ok((report, recovered))
}

fn cost(at: Option<u64>) -> String {
    at.map_or_else(|| "—".to_owned(), |f| f.to_string())
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
fn main() -> Result<()> {
    let mut args = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse::<u64>().ok());
    let n = u32::try_from(args.next().unwrap_or(2_000_000)).unwrap_or(u32::MAX);
    let budget = args.next().unwrap_or(400_000);
    let started = Instant::now();
    let mut builder = Builder::random(n, 42);
    let scenarios = scenario::plant(&mut builder);
    let world = builder.build();
    println!(
        "world: {} entities, {} edges, built in {:.1}s; biggest org {} edges, biggest ip {}",
        world.len(),
        world.edges(),
        started.elapsed().as_secs_f64(),
        world.degree(builder_hub(&world, dom::ORG)),
        world.degree(builder_hub(&world, dom::IP)),
    );
    println!("budget: {budget} fetched edges per run\n");
    for s in scenarios.iter().filter(|s| !s.signal.is_empty()) {
        table(&world, s, budget)?;
    }
    if let Some(s) = scenarios.iter().find(|s| s.name == "explore") {
        read_path(&world, s, budget)?;
    }
    Ok(())
}

/// The heaviest entity of `d` (the first of its range).
fn builder_hub(world: &World, d: ramp_expand::Dom) -> u64 {
    (0..world.len())
        .filter_map(|i| u64::try_from(i).ok())
        .find(|&i| world.domain(i) == d)
        .unwrap_or(0)
}

/// Every policy on scenario `s`.
///
/// # Errors
/// Storage errors.
#[expect(clippy::print_stdout, reason = "benchmark report")]
fn table(world: &World, s: &Scenario, budget: u64) -> Result<()> {
    let degrees: Vec<String> = s
        .signal
        .iter()
        .map(|&(a, _, _)| world.degree(a).to_string())
        .collect();
    println!("== {} — {}", s.name, s.about);
    println!(
        "   signal: {} edges; degrees along it: {}",
        s.signal.len(),
        degrees.join(", ")
    );
    println!(
        "   {:<24} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7} {:>9} {:>9} {:>8} {:>8}",
        "policy",
        "found at",
        "fetched",
        "steps",
        "nodes",
        "edges",
        "signal",
        "edges/sig",
        "1st link",
        "wall ms",
        "decide"
    );
    let policies: Vec<Box<dyn Policy>> = vec![
        Box::new(Bfs { depth: s.depth }),
        Box::new(Rules {
            depth: s.depth,
            cap: 100,
        }),
        Box::new(Rules {
            depth: s.depth,
            cap: 1000,
        }),
        Box::new(Rules {
            depth: s.depth,
            cap: 10_000,
        }),
        Box::new(Guided),
    ];
    for p in policies {
        let name = p.name();
        let (r, got) = run(world, s, p, budget, Reads::Projection)?;
        let per = if got == 0 {
            "—".to_owned()
        } else {
            format!("{}", r.edges / u64::try_from(got).unwrap_or(1).max(1))
        };
        let first = if s.task.goal == ramp_expand::Goal::Alert {
            r.alert_at
        } else {
            r.connected_at
        };
        println!(
            "   {:<24} {:>9} {:>9} {:>8} {:>8} {:>8} {:>5}/{} {:>9} {:>9} {:>8.0} {:>8.1}",
            name,
            cost(r.done_at),
            r.fetched,
            r.steps,
            r.nodes,
            r.edges,
            got,
            s.signal.len(),
            per,
            cost(first),
            ms(r.wall),
            ms(r.policy),
        );
    }
    println!();
    Ok(())
}

/// `s` with probes reading the projection, then LMDB.
///
/// # Errors
/// Storage errors.
#[expect(clippy::print_stdout, reason = "benchmark report")]
fn read_path(world: &World, s: &Scenario, budget: u64) -> Result<()> {
    let budget = budget.min(150_000);
    println!("== {} — {} ({budget} fetched edges)", s.name, s.about);
    println!(
        "   {:<12} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "probes read",
        "steps",
        "nodes",
        "probes",
        "probe ms",
        "µs/probe",
        "write ms",
        "decide ms",
        "wall ms"
    );
    for reads in [Reads::Projection, Reads::Lmdb] {
        let (r, _) = run(world, s, Box::new(Guided), budget, reads)?;
        let per = ms(r.probe) * 1e3 / f64::from(u32::try_from(r.probes.max(1)).unwrap_or(u32::MAX));
        println!(
            "   {:<12} {:>8} {:>8} {:>8} {:>10.0} {:>10.1} {:>10.0} {:>10.0} {:>10.0}",
            format!("{reads:?}"),
            r.steps,
            r.nodes,
            r.probes,
            ms(r.probe),
            per,
            ms(r.write),
            ms(r.policy),
            ms(r.wall),
        );
    }
    Ok(())
}
