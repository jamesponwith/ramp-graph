//! Query benchmark, mirrored by a Python script against upstream for comparison.
//!
//! Builds N nodes (`node{x%5}`/`x`, property `prop{x%5}="value{x%5}"`) and N edges
//! `x -> (x*7919+13) % N` typed `edge{y%5}`, commits, then times ad-hoc queries in
//! a read txn. Result counts are printed so the two implementations can be checked
//! against each other.
//!
//! `cargo bench -p ramp-graph --bench query [-- N]` (N defaults to 1,000,000).

use std::time::Instant;

use ramp_graph::lgql::Pattern;
use ramp_graph::value::Value;
use ramp_graph::{Graph, GraphError, Result, Txn};

/// Runs `patterns` once each, returning the total number of chains.
///
/// # Errors
/// Fails on a bad pattern or storage error.
fn run(t: &Txn<'_>, patterns: &[String]) -> Result<u64> {
    let mut total = 0;
    for src in patterns {
        let p = Pattern::parse(src).map_err(|e| GraphError::Value(e.to_string()))?;
        t.query(&[p], None, |_, _| {
            total += 1;
            true
        })?;
    }
    Ok(total)
}

/// Times `f` and prints its result count.
///
/// # Errors
/// Whatever `f` fails with.
#[expect(clippy::print_stdout, reason = "benchmark report")]
fn timed(name: &str, f: impl FnOnce() -> Result<u64>) -> Result<()> {
    let start = Instant::now();
    let n = f()?;
    println!(
        "{name:<28} {:>9.3}s  {n:>8} results",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<()> {
    let n: u64 = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let dir = tempfile::tempdir().map_err(|e| GraphError::Io(e.to_string()))?;
    let g = Graph::open(dir.path().join("bench.db"))?;
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
    t.commit()?;

    let t = g.read()?;
    let points: Vec<String> = (0..10_000_u64)
        .map(|i| (i * 104_729) % n)
        .map(|k| format!("n(type=\"node{}\", value=\"{k}\")", k % 5))
        .collect();
    timed("10k point lookups", || run(&t, &points))?;
    timed("n(type=\"node3\")", || {
        run(&t, &["n(type=\"node3\")".to_owned()])
    })?;
    timed("n(prop2=\"value2\")", || {
        run(&t, &["n(prop2=\"value2\")".to_owned()])
    })?;
    timed("e(type=\"edge3\")", || {
        run(&t, &["e(type=\"edge3\")".to_owned()])
    })?;
    timed("n(type=\"node1\")->e()->n()", || {
        run(&t, &["n(type=\"node1\")->e()->n()".to_owned()])
    })?;
    timed("n(type=\"node1\")-n()-n()", || {
        run(&t, &["n(type=\"node1\")-n()-n()".to_owned()])
    })?;
    let pushdown = "n(type=\"node1\")-e(type=\"edge2\")-n()".to_owned();
    timed(&pushdown.clone(), || run(&t, &[pushdown]))?;
    timed("counts at mid", || {
        t.counts(Some(n / 2)).map(|(nodes, _)| nodes)
    })?;
    Ok(())
}
