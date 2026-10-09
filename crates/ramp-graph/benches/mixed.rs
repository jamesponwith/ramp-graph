//! Queries while the graph is written: a writer commits small batches at a fixed rate
//! while a reader runs scan queries, taking a projection the way `ramp-server` does
//! (`projection_if_due(4)`, else the LMDB path).
//!
//! `cargo bench -p ramp-graph --bench mixed [-- N]` (N defaults to 1,000,000 nodes).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ramp_graph::lgql::Pattern;
use ramp_graph::value::Value;
use ramp_graph::{Graph, GraphError, Result};

/// Seconds each write rate runs for.
const RUN: Duration = Duration::from_secs(8);
/// Nodes (each with an edge to an existing node) per commit.
const BATCH: u64 = 10;
/// Builds stay under this fraction of a graph's time, as in the server.
const DUTY: u32 = 4;

/// Milliseconds at percentile `pct` of `sorted`.
fn pct(sorted: &[Duration], pct: usize) -> f64 {
    let i = (sorted.len() * pct / 100).min(sorted.len().saturating_sub(1));
    sorted.get(i).map_or(0.0, |d| d.as_secs_f64() * 1e3)
}

/// Builds the query-bench graph: `n` nodes with one property each, `n` edges.
///
/// # Errors
/// Storage errors.
fn load(g: &Graph, n: u64) -> Result<()> {
    let mut t = g.write()?;
    let mut nodes = Vec::new();
    for x in 0..n {
        let id = t
            .node(
                format!("node{}", x % 5).as_bytes(),
                x.to_string().as_bytes(),
            )?
            .id;
        t.set_value(
            id,
            &format!("prop{}", x % 5),
            &Value::from(format!("value{}", x % 5)),
        )?;
        nodes.push(id);
    }
    for (x, &src) in (0..n).zip(&nodes) {
        let y = (x * 7919 + 13) % n;
        let tgt = *nodes
            .get(usize::try_from(y).unwrap_or(usize::MAX))
            .ok_or(GraphError::Corrupt("bad index"))?;
        t.edge(
            src,
            tgt,
            format!("edge{}", y % 5).as_bytes(),
            x.to_string().as_bytes(),
        )?;
    }
    t.commit()
}

/// What one reader saw at one write rate.
struct Seen {
    latencies: Vec<Duration>,
    projected: usize,
    commits: u64,
    commit_p99: f64,
}

/// Runs `query` repeatedly for [`RUN`] while a writer commits `rate` times a second.
///
/// # Errors
/// Storage or parse errors.
fn run(g: &Graph, query: &Pattern, rate: u32, base: u64) -> Result<Seen> {
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let writer = scope.spawn(|| -> Result<(u64, f64)> {
            let mut commit_lat = Vec::new();
            let mut i = 0_u64;
            let period = (rate > 0).then(|| Duration::from_secs(1) / rate);
            let started = Instant::now();
            while let Some(p) = period {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let due = started + p * u32::try_from(i).unwrap_or(u32::MAX);
                // Pacing, not synchronisation: wait out the period (parking may wake early).
                while let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::park_timeout(wait);
                }
                let t0 = Instant::now();
                let mut t = g.write()?;
                for k in 0..BATCH {
                    let x = base + i * BATCH + k;
                    let id = t
                        .node(
                            format!("node{}", x % 5).as_bytes(),
                            x.to_string().as_bytes(),
                        )?
                        .id;
                    t.set_value(
                        id,
                        &format!("prop{}", x % 5),
                        &Value::from(format!("value{}", x % 5)),
                    )?;
                    if let Some(old) =
                        t.node_lookup(b"node0", (x % base).to_string().as_bytes(), None)?
                    {
                        t.edge(id, old.id, b"edge0", b"")?;
                    }
                }
                t.commit()?;
                commit_lat.push(t0.elapsed());
                i += 1;
            }
            commit_lat.sort_unstable();
            Ok((i, pct(&commit_lat, 99)))
        });
        let mut latencies = Vec::new();
        let mut projected = 0;
        let started = Instant::now();
        let read = (|| -> Result<()> {
            while started.elapsed() < RUN {
                let t0 = Instant::now();
                let t = g.read()?;
                if t.projection_if_due(DUTY)?.is_some() {
                    projected += 1;
                }
                let mut n = 0_u64;
                t.query(std::slice::from_ref(query), None, |_, _| {
                    n += 1;
                    true
                })?;
                latencies.push(t0.elapsed());
            }
            Ok(())
        })();
        stop.store(true, Ordering::Relaxed);
        let (commits, commit_p99) = writer
            .join()
            .map_err(|_| GraphError::Corrupt("writer panicked"))??;
        read?;
        latencies.sort_unstable();
        Ok(Seen {
            latencies,
            projected,
            commits,
            commit_p99,
        })
    })
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
fn main() -> Result<()> {
    let n: u64 = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let dir = tempfile::tempdir().map_err(|e| GraphError::Io(e.to_string()))?;
    let mut base = n;
    for src in ["n(type=\"node3\")", "n(type=\"node1\")->e()->n()"] {
        let g = Graph::open(dir.path().join(format!("mixed{}.db", src.len())))?;
        load(&g, n)?;
        let query = Pattern::parse(src).map_err(|e| GraphError::Value(e.to_string()))?;
        println!(
            "-- {src}, {BATCH}-node commits, {}s per rate",
            RUN.as_secs()
        );
        println!(
            "{:>12} {:>8} {:>9} {:>9} {:>9} {:>10} {:>12}",
            "commits/s", "queries", "p50 ms", "p90 ms", "p99 ms", "projected", "commit p99"
        );
        for rate in [0, 1, 5, 20, 100] {
            let s = run(&g, &query, rate, base)?;
            base += s.commits * BATCH;
            let share = s.projected * 100 / s.latencies.len().max(1);
            println!(
                "{rate:>12} {:>8} {:>9.2} {:>9.2} {:>9.2} {:>9}% {:>9.2} ms",
                s.latencies.len(),
                pct(&s.latencies, 50),
                pct(&s.latencies, 90),
                pct(&s.latencies, 99),
                share,
                s.commit_p99
            );
        }
    }
    Ok(())
}
