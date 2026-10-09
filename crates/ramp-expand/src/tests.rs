use ramp_graph::Graph;

use super::*;
use crate::policy::{Bfs, Guided, Rules};
use crate::scenario::Scenario;
use crate::world::{Builder, World};

fn world() -> (World, Vec<Scenario>) {
    let mut b = Builder::random(20_000, 1);
    let scenarios = scenario::plant(&mut b);
    (b.build(), scenarios)
}

fn scenario<'s>(all: &'s [Scenario], name: &str) -> &'s Scenario {
    all.iter().find(|s| s.name == name).unwrap()
}

/// Runs `s` with `policy` on a fresh graph, stopping once its signal is written.
fn run<P: Policy>(
    w: &World,
    s: &Scenario,
    policy: P,
    budget: u64,
    reads: Reads,
) -> (Report, (u64, u64)) {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph::open(dir.path().join("g.db")).unwrap();
    let mut task = s.task.clone();
    task.budget = budget;
    let signal = s.signal.clone();
    let ex = Expander::new(w, &g, policy, task, reads).unwrap();
    let report = ex
        .run(|ex| !signal.is_empty() && signal.iter().all(|&(a, b, r)| ex.has_edge(a, b, r)))
        .unwrap();
    let counts = g.read().unwrap().counts(None).unwrap();
    (report, counts)
}

#[test]
fn world_is_consistent() {
    let (w, scenarios) = world();
    let mut buf = Vec::new();
    for id in (0..w.len()).step_by(97).map(|i| Id::try_from(i).unwrap()) {
        for r in 0..u8::try_from(world::RELATIONS.len()).unwrap() {
            buf.clear();
            w.fetch(id, r, 0, u32::MAX, &mut buf);
            assert_eq!(
                buf.len(),
                usize::try_from(w.peek(id)[usize::from(r)]).unwrap()
            );
            for &other in &buf {
                let mut back = Vec::new();
                w.fetch(other, r, 0, u32::MAX, &mut back);
                assert!(back.contains(&id), "{id} -{r}- {other} is not symmetric");
            }
        }
    }
    // Paging returns the same edges as one fetch.
    let hub = w.planted()[0].0;
    let (mut all, mut paged) = (Vec::new(), Vec::new());
    w.fetch(hub, world::rel::OWNS, 0, u32::MAX, &mut all);
    for off in (0..u32::try_from(all.len()).unwrap()).step_by(3) {
        w.fetch(hub, world::rel::OWNS, off, 3, &mut paged);
    }
    assert_eq!(all, paged);
    for s in &scenarios {
        for &(a, b, r) in &s.signal {
            buf.clear();
            w.fetch(a, r, 0, u32::MAX, &mut buf);
            assert!(buf.contains(&b), "{}: planted edge missing", s.name);
        }
    }
}

#[test]
fn guided_recovers_every_signal() {
    let (w, scenarios) = world();
    for name in ["path-5hop", "path-7hop", "pivot-busy", "alert"] {
        let s = scenario(&scenarios, name);
        let (report, (nodes, edges)) = run(&w, s, Guided::default(), 200_000, Reads::Projection);
        assert!(report.done_at.is_some(), "{name}: {report:?}");
        assert!(report.fetched <= 200_000);
        assert_eq!(
            (nodes, edges),
            (report.nodes, report.edges),
            "{name}: graph matches report"
        );
        if s.task.goal == Goal::Connect {
            assert!(
                report.connected_at.is_some_and(|c| c <= report.fetched),
                "{name}"
            );
        }
    }
    let (report, _) = run(
        &w,
        scenario(&scenarios, "alert"),
        Guided::default(),
        200_000,
        Reads::Projection,
    );
    assert!(report.alert_at.is_some() && report.alerts > 0);
}

#[test]
fn hardcoded_limits_hold() {
    let (w, scenarios) = world();
    let s = scenario(&scenarios, "path-5hop");
    // Depth 1: only the two seeds are expanded.
    let (report, _) = run(&w, s, Bfs { depth: 1 }, 1_000_000, Reads::Lmdb);
    assert_eq!(report.steps, 2);
    // A cap below every seed's degree: nothing is expanded.
    let (report, _) = run(&w, s, Rules { depth: 9, cap: 0 }, 1_000_000, Reads::Lmdb);
    assert_eq!(report.steps, 0);
    // The budget binds.
    let (report, _) = run(&w, s, Bfs { depth: 9 }, 1_000, Reads::Lmdb);
    assert!(report.fetched <= 1_000);
    assert!(report.done_at.is_none());
}

#[test]
fn reads_change_cost_not_decisions() {
    let (w, scenarios) = world();
    let s = scenario(&scenarios, "explore");
    let (a, _) = run(&w, s, Guided::default(), 5_000, Reads::Projection);
    let (b, _) = run(&w, s, Guided::default(), 5_000, Reads::Lmdb);
    assert!(a.probes > 10, "{a:?}");
    let key = |r: &Report| (r.steps, r.fetched, r.nodes, r.edges, r.probes, r.decisions);
    assert_eq!(key(&a), key(&b));
}

#[test]
fn shape_steers_alerts() {
    // The goal's shape is intent, not a limit: without it the alert is still found.
    let (w, scenarios) = world();
    let s = scenario(&scenarios, "pivot-busy");
    let mut shapeless = s.clone();
    shapeless.task.shape.clear();
    let (with, _) = run(&w, s, Guided::default(), 400_000, Reads::Projection);
    let (without, _) = run(
        &w,
        &shapeless,
        Guided::default(),
        400_000,
        Reads::Projection,
    );
    assert!(
        with.done_at.is_some() && without.done_at.is_some(),
        "{with:?} {without:?}"
    );
}

#[test]
fn every_fetch_is_accounted() {
    let (w, scenarios) = world();
    let (r, _) = run(
        &w,
        scenario(&scenarios, "path-7hop"),
        Guided::default(),
        50_000,
        Reads::Projection,
    );
    assert_eq!(r.fetched_by_domain.iter().sum::<u64>(), r.fetched);
}

#[test]
fn pushdown_cuts_through_hubs() {
    // With the target pushed to the source, both alerts are found by fetching little
    // more than the hit itself; the source pays in scanning instead.
    let (w, scenarios) = world();
    for name in ["pivot-busy", "alert"] {
        let s = scenario(&scenarios, name);
        let (plain, _) = run(&w, s, Guided::default(), 400_000, Reads::Projection);
        let before = w.scanned();
        let (pushed, _) = run(&w, s, Guided { pushdown: true }, 400_000, Reads::Projection);
        assert!(pushed.done_at.is_some(), "{name}: {pushed:?}");
        assert!(pushed.filtered > 0 && w.scanned() > before, "{name}");
        assert!(
            pushed.fetched * 4 < plain.fetched.max(1),
            "{name}: {} vs {}",
            pushed.fetched,
            plain.fetched
        );
    }
    // Without a target there is nothing to push: path tasks run as before.
    let s = scenario(&scenarios, "path-7hop");
    let (a, _) = run(&w, s, Guided::default(), 50_000, Reads::Projection);
    let (b, _) = run(&w, s, Guided { pushdown: true }, 50_000, Reads::Projection);
    assert_eq!((a.fetched, a.steps, b.filtered), (b.fetched, b.steps, 0));
}

/// A world whose source cannot filter.
struct Plain<'w>(&'w World);

impl Source for Plain<'_> {
    fn domains(&self) -> &[&'static str] {
        self.0.domains()
    }
    fn relations(&self) -> &[Relation] {
        self.0.relations()
    }
    fn domain(&self, id: Id) -> Dom {
        self.0.domain(id)
    }
    fn attrs(&self, id: Id, out: &mut Vec<(&'static str, String)>) {
        self.0.attrs(id, out);
    }
    fn peek(&self, id: Id) -> Counts {
        self.0.peek(id)
    }
    fn fetch(&self, id: Id, rel: Rel, offset: u32, limit: u32, out: &mut Vec<Id>) {
        self.0.fetch(id, rel, offset, limit, out);
    }
}

#[test]
fn pushdown_falls_back_without_source_support() {
    let (w, scenarios) = world();
    let s = scenario(&scenarios, "alert");
    let dir = tempfile::tempdir().unwrap();
    let g = Graph::open(dir.path().join("g.db")).unwrap();
    let mut task = s.task.clone();
    task.budget = 400_000;
    let plain = Plain(&w);
    let ex = Expander::new(
        &plain,
        &g,
        Guided { pushdown: true },
        task,
        Reads::Projection,
    )
    .unwrap();
    let r = ex.run(|_| false).unwrap();
    assert!(r.done_at.is_some(), "{r:?}");
    assert_eq!(r.filtered, 0);
}
