//! Port of upstream `bench.py`: in one write txn, insert 1M nodes, then a property on
//! each, then 1M unique random edges. Prints per-phase rates and the file size.
//!
//! `cargo bench -p ramp-graph --bench insert [-- N]` (N defaults to 1,000,000).

use std::collections::BTreeSet;
use std::time::Instant;

use ramp_graph::value::Value;
use ramp_graph::{EdgeSpec, Graph, Result};

/// splitmix64: a dependency-free, reproducible stream of pseudo-random numbers.
fn rng(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
fn phase(name: &str, n: u64, start: Instant) {
    let secs = start.elapsed().as_secs_f64();
    let rate = f64::from(u32::try_from(n).unwrap_or(u32::MAX)) / secs;
    println!("{name:<10} {secs:>8.3}s  {rate:>10.0}/s");
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
fn main() -> Result<()> {
    let n: u64 = std::env::args()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let mut next = rng(42);
    let mut pairs = BTreeSet::new();
    while u64::try_from(pairs.len()).unwrap_or(u64::MAX) < n {
        pairs.insert((next() % n, next() % n));
    }

    let dir = tempfile::tempdir().map_err(|e| ramp_graph::GraphError::Io(e.to_string()))?;
    let g = Graph::open(dir.path().join("bench.db"))?;
    let mut t = g.write()?;

    let start = Instant::now();
    let mut nodes = Vec::new();
    for x in 0..n {
        nodes.push(
            t.node(
                format!("node{}", x % 5).as_bytes(),
                x.to_string().as_bytes(),
            )?
            .id,
        );
    }
    phase("nodes", n, start);

    let start = Instant::now();
    for (x, &id) in (0..n).zip(&nodes) {
        t.set_value(
            id,
            &format!("prop{}", x % 5),
            &Value::from(format!("value{}", x % 5)),
        )?;
    }
    phase("props", n, start);

    let start = Instant::now();
    for (i, &(x, y)) in (0_u64..).zip(&pairs) {
        let (Some(&src), Some(&tgt)) = (
            nodes.get(usize::try_from(x).unwrap_or(usize::MAX)),
            nodes.get(usize::try_from(y).unwrap_or(usize::MAX)),
        ) else {
            continue;
        };
        t.edge(
            src,
            tgt,
            format!("edge{}", (x + y) % 5).as_bytes(),
            i.to_string().as_bytes(),
        )?;
    }
    phase("edges", n, start);

    let start = Instant::now();
    t.commit()?;
    phase("commit", 1, start);
    println!("size       {} MiB", g.size()? >> 20);
    let start = Instant::now();
    g.snapshot(dir.path().join("compact.db"))?;
    let compact = std::fs::metadata(dir.path().join("compact.db"))
        .map_err(|e| ramp_graph::GraphError::Io(e.to_string()))?
        .len();
    println!(
        "compacted  {} MiB in {:.2}s",
        compact >> 20,
        start.elapsed().as_secs_f64()
    );

    bulk_edges(dir.path(), n, &pairs)
}

/// The same edges through `edge_batch`, in one batch and in REST-sized ones.
///
/// # Errors
/// Storage errors.
fn bulk_edges(dir: &std::path::Path, n: u64, pairs: &BTreeSet<(u64, u64)>) -> Result<()> {
    for batch in [n, 10_000] {
        let g = Graph::open(dir.join(format!("bulk{batch}.db")))?;
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
        let tys: Vec<String> = (0..5).map(|i| format!("edge{i}")).collect();
        let vals: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let specs: Vec<EdgeSpec<'_>> = (0_u64..)
            .zip(pairs)
            .filter_map(|(i, &(x, y))| {
                Some(EdgeSpec {
                    src: *nodes.get(usize::try_from(x).ok()?)?,
                    tgt: *nodes.get(usize::try_from(y).ok()?)?,
                    ty: tys.get(usize::try_from((x + y) % 5).ok()?)?.as_bytes(),
                    val: vals.get(usize::try_from(i).ok()?)?.as_bytes(),
                })
            })
            .collect();
        let start = Instant::now();
        for chunk in specs.chunks(usize::try_from(batch).unwrap_or(usize::MAX)) {
            t.edge_batch(chunk)?;
        }
        phase(&format!("edges/{batch}"), n, start);
        t.commit()?;
    }
    Ok(())
}
