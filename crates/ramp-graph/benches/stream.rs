//! Streaming queries, deletes, and historical views on the query-bench graph.
//!
//! `cargo bench -p ramp-graph --bench stream [-- N]` (N defaults to 1,000,000).

use std::time::Instant;

use ramp_graph::lgql::Pattern;
use ramp_graph::value::Value;
use ramp_graph::{Graph, GraphError, Result};

/// Times `f` and prints its result count.
///
/// # Errors
/// Whatever `f` fails with.
#[expect(clippy::print_stdout, reason = "benchmark report")]
fn timed(name: &str, f: impl FnOnce() -> Result<u64>) -> Result<()> {
    let start = Instant::now();
    let n = f()?;
    println!(
        "{name:<44} {:>8.3}s  {n:>8} results",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
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
    let end = g.read()?.next_id()?;

    let parse = |s: &str| Pattern::parse(s).map_err(|e| GraphError::Value(e.to_string()));
    let stream = |src: &str| -> Result<u64> {
        let t = g.read()?;
        let p = parse(src)?;
        let mut total = 0;
        t.mquery(std::slice::from_ref(&p), 1, None, |_, _, _| {
            total += 1;
            true
        })?;
        Ok(total)
    };
    println!("-- streaming over the whole log ({end} entries)");
    timed("mquery n(type=\"node3\")", || stream("n(type=\"node3\")"))?;
    timed("mquery n(prop2=\"value2\")", || {
        stream("n(prop2=\"value2\")")
    })?;
    timed("mquery n(type=\"node1\")->e()->n()", || {
        stream("n(type=\"node1\")->e()->n()")
    })?;
    timed("mquery e(type=\"edge3\")", || stream("e(type=\"edge3\")"))?;

    println!("-- historical view (log position {})", end / 2);
    let at = Some(end / 2);
    let hist = |src: &str| -> Result<u64> {
        let t = g.read()?;
        let p = parse(src)?;
        let mut total = 0;
        t.query(std::slice::from_ref(&p), at, |_, _| {
            total += 1;
            true
        })?;
        Ok(total)
    };
    timed("n(type=\"node3\") at mid", || hist("n(type=\"node3\")"))?;
    timed("n(type=\"node1\")->e()->n() at mid", || {
        hist("n(type=\"node1\")->e()->n()")
    })?;

    println!("-- deletes");
    timed("delete 200k nodes (cascading edges, props)", || {
        let mut t = g.write()?;
        for &id in nodes.iter().step_by(5) {
            t.delete(id)?;
        }
        let n = u64::try_from(nodes.len() / 5).unwrap_or(0);
        t.commit()?;
        Ok(n)
    })?;
    timed("counts after", || {
        g.read()?.counts(None).map(|(nodes, _)| nodes)
    })?;
    Ok(())
}
