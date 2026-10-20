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
    run_k(w, s, policy, budget, reads, 1)
}

/// [`run`], fetching `k` candidates per round.
fn run_k<P: Policy>(
    w: &World,
    s: &Scenario,
    policy: P,
    budget: u64,
    reads: Reads,
    k: usize,
) -> (Report, (u64, u64)) {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph::open(dir.path().join("g.db")).unwrap();
    let mut task = s.task.clone();
    task.budget = budget;
    let signal = s.signal.clone();
    let ex = Expander::new(w, &g, policy, task, reads)
        .unwrap()
        .parallel(k);
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

#[test]
fn trace_records_every_step_and_credits_hits() {
    let (w, all) = world();
    let s = scenario(&all, "alert");
    let (r, _) = run(&w, s, Guided::default(), 100_000, Reads::Projection);
    assert_eq!(r.trace.len(), usize::try_from(r.steps).unwrap());
    let added: usize = r.trace.iter().map(|t| t.edges.len()).sum();
    assert_eq!(u64::try_from(added).unwrap(), r.edges);
    // The standing query fired, so some step's edges are credited with it, and the
    // step that wrote the planted edge is credited by the truth too.
    assert!(
        !r.matched.is_empty(),
        "alerts {} at {:?}",
        r.alerts,
        r.alert_at
    );
    let none = std::collections::HashSet::new();
    assert!(r.trace.iter().any(|t| r.hits(t, &none) > 0));
    let truth: std::collections::HashSet<_> = s
        .signal
        .iter()
        .map(|&(a, b, rel)| (a.min(b), a.max(b), rel))
        .collect();
    let rows = wire::trace(&r, &truth);
    let rows: Vec<serde_json::Value> = rows
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect();
    assert_eq!(rows.len(), r.trace.len());
    assert!(rows.iter().any(|row| row["hits"].as_u64().unwrap() > 0));
    // Every step with a hit is on the route (here the seed's step holds the hit).
    let route = r.route(&truth);
    for (row, t) in rows.iter().zip(&r.trace) {
        assert_eq!(row["route"], route.contains(&t.candidate.id));
        if r.hits(t, &truth) > 0 {
            assert!(row["route"].as_bool().unwrap());
        }
    }
}

#[test]
fn the_route_credits_steps_that_found_the_way() {
    // 1 (seed) finds 2, 2 finds 3, 3 fetches the hit (3, 4); 1 also finds 5, a dead end.
    let (w, all) = world();
    let (r, _) = run(
        &w,
        scenario(&all, "alert"),
        Guided::default(),
        100_000,
        Reads::Projection,
    );
    let template = r.trace.first().map(|t| t.candidate.clone());
    let step = |id: Id, edges: &[(Id, Id)]| Step {
        candidate: Candidate {
            id,
            ..template.clone().unwrap()
        },
        decision: Decision::default(),
        before: 0,
        edges: edges
            .iter()
            .map(|&(a, b)| (a.min(b), a.max(b), 0))
            .collect(),
    };
    let r = Report {
        trace: vec![
            step(1, &[(1, 2), (1, 5)]),
            step(5, &[(5, 6)]),
            step(2, &[(2, 3)]),
            step(3, &[(3, 4), (3, 1)]),
        ],
        ..Report::default()
    };
    let truth: std::collections::HashSet<_> = std::iter::once((3, 4, 0)).collect();
    let route = r.route(&truth);
    assert_eq!(route, [1, 2, 3, 4].into_iter().collect());
    let credited: Vec<(Id, usize)> = r
        .trace
        .iter()
        .filter(|t| route.contains(&t.candidate.id))
        .map(|t| (t.candidate.id, r.hits(t, &truth)))
        .collect();
    assert_eq!(credited, [(1, 0), (2, 0), (3, 1)]);
}

#[test]
fn decisions_round_trip_through_json() {
    let d = Decision {
        priority: -1.5,
        take: vec![
            Take::all(2, 64),
            Take {
                rel: 3,
                limit: 8,
                filter: Some(Filter::Via {
                    rel: 4,
                    then: Box::new(Filter::Attr {
                        key: "watch".to_owned(),
                        value: "T".to_owned(),
                    }),
                }),
            },
        ],
    };
    let back = wire::parse_decision(&wire::decision(&d)).unwrap();
    assert!((back.priority - d.priority).abs() < f64::EPSILON);
    assert_eq!(back.take, d.take);
    // A model may leave out what it does not want.
    let skip = wire::parse_decision(&serde_json::json!({"priority": 0})).unwrap();
    assert!(skip.take.is_empty());
    wire::parse_decision(&serde_json::json!({"take": []})).unwrap_err();
}

#[test]
fn concurrent_rounds_cut_round_trips_not_recall() {
    let (w, all) = world();
    for name in ["path-7hop", "pivot-busy"] {
        let s = scenario(&all, name);
        let policy = Guided {
            pushdown: s.task.target.is_some(),
        };
        let (one, _) = run_k(&w, s, policy, 100_000, Reads::Projection, 1);
        let (many, _) = run_k(&w, s, policy, 100_000, Reads::Projection, 16);
        assert!(many.done_at.is_some(), "{name}: k=16 recovers the signal");
        assert_eq!(one.rounds, one.steps);
        assert!(
            many.rounds * 2 < one.rounds,
            "{name}: {} vs {} rounds",
            many.rounds,
            one.rounds
        );
        // Speculative picks stay near the best, so the budget is spent about as well.
        assert!(
            many.fetched * 2 < one.fetched * 3,
            "{name}: {} vs {} fetched",
            many.fetched,
            one.fetched
        );
    }
    // Reserving each round's budget up front keeps concurrent fetches within it.
    let s = scenario(&all, "path-7hop");
    let (r, _) = run_k(&w, s, Bfs { depth: 9 }, 5_000, Reads::Projection, 16);
    assert!(r.fetched <= 5_000, "{}", r.fetched);
    assert_eq!(r.fetched_by_domain.iter().sum::<u64>(), r.fetched);
}

#[test]
fn a_loaded_world_serves_its_own_schema() {
    const DOMAINS: [&str; 2] = ["officer", "entity"];
    const RELATIONS: [Relation; 1] = [Relation {
        name: "officer_of",
        from: 0,
        to: 1,
    }];
    let mut builder = Builder::empty(&DOMAINS, &RELATIONS);
    let (officer, entity, other) = (builder.entity(0), builder.entity(1), builder.entity(0));
    builder.link(officer, entity, 0);
    builder.link(other, entity, 0);
    builder.attr(other, "country", "PAN");
    let w = builder.build();
    assert_eq!(w.domains(), DOMAINS);
    assert_eq!(w.relations().len(), 1);
    assert_eq!((w.domain(officer), w.domain(entity)), (0, 1));
    assert_eq!(w.peek(entity)[0], 2);
    let mut out = Vec::new();
    let filter = Filter::Attr {
        key: "country".to_owned(),
        value: "PAN".to_owned(),
    };
    w.fetch_where(entity, 0, &filter, u32::MAX, &mut out)
        .unwrap();
    assert_eq!(out, [other]);
}

#[test]
fn logged_batches_hold_every_decision_and_label_candidates() {
    let (w, all) = world();
    let s = scenario(&all, "alert");
    let dir = tempfile::tempdir().unwrap();
    let g = Graph::open(dir.path().join("g.db")).unwrap();
    let mut task = s.task.clone();
    task.budget = 100_000;
    let r = Expander::new(&w, &g, Guided::default(), task, Reads::Projection)
        .unwrap()
        .log_batches()
        .run(|_| false)
        .unwrap();
    let offered: usize = r.batches.iter().map(|b| b.candidates.len()).sum();
    assert_eq!(u64::try_from(offered).unwrap(), r.decisions);
    assert!(
        r.batches
            .iter()
            .all(|b| b.candidates.len() == b.decisions.len())
    );
    let ids: std::collections::HashSet<Id> = r
        .batches
        .iter()
        .flat_map(|b| b.candidates.iter().map(|c| c.id))
        .collect();
    assert!(r.trace.iter().all(|t| ids.contains(&t.candidate.id)));

    let seed = s.task.seeds[0];
    let lines = wire::batches(&r, &w, |id| id == seed);
    let rows: Vec<serde_json::Value> = lines
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect();
    assert_eq!(rows.len(), r.batches.len());
    let labels = |v: &serde_json::Value| v["label"].as_bool().unwrap();
    let first = rows[0]["candidates"].as_array().unwrap();
    assert!(first.iter().any(labels));
    assert_eq!(rows[0]["decisions"].as_array().unwrap().len(), first.len());
    // The seed's neighbours still have their edge back to it: a depth within what is left.
    let mut deep = 0;
    for c in rows
        .iter()
        .flat_map(|r| r["candidates"].as_array().unwrap())
    {
        for (r, d) in c["depth"].as_array().unwrap().iter().enumerate() {
            if let Some(d) = d.as_u64() {
                let left = c["peek"][r].as_u64().unwrap() - c["fetched"][r].as_u64().unwrap();
                assert!(d < left);
                deep += 1;
            }
        }
    }
    assert!(deep > 0);

    // Off by default.
    let (r, _) = run(&w, s, Guided::default(), 10_000, Reads::Projection);
    assert!(r.batches.is_empty());
}

#[test]
fn icij_entities_carry_their_text() {
    let dir = tempfile::tempdir().unwrap();
    let files = [
        (
            "nodes-entities.csv",
            "node_id,name,jurisdiction_description,note\n1,\"ACME, LTD.\",Samoa,boilerplate\n",
        ),
        (
            "nodes-officers.csv",
            "node_id,name,countries,country_codes\n2,KIM SOO IN,South Korea,KOR\n3,Tian Yuan,China,CHN\n",
        ),
        ("nodes-intermediaries.csv", "node_id,name\n"),
        ("nodes-addresses.csv", "node_id,address\n"),
        ("nodes-others.csv", "node_id,name\n"),
        (
            "relationships.csv",
            "node_id_start,node_id_end,rel_type,link\n2,1,officer_of,Nominee Director of\n3,1,officer_of,shareholder of\n",
        ),
    ];
    for (name, body) in files {
        std::fs::write(dir.path().join(name), body).unwrap();
    }
    let w = icij::load(dir.path()).unwrap();
    assert_eq!(
        w.text(0),
        "entity name: ACME, LTD.; jurisdiction_description: Samoa; linked by: nominee director of, shareholder of"
    );
    assert_eq!(
        w.text(1),
        "officer name: KIM SOO IN; countries: South Korea; links: nominee director of"
    );
    assert_eq!(w.peek(0)[usize::from(icij::OFFICER_OF)], 2);
    let mut attrs = Vec::new();
    w.attrs(2, &mut attrs);
    assert_eq!(attrs, [("country", "CHN".to_owned())]);
    let rows = wire::entities(&w, &[1]);
    let row: serde_json::Value = serde_json::from_slice(rows.trim_ascii()).unwrap();
    assert_eq!(row["domain"], 1);
}

#[test]
fn path_slack_grades_the_connect_oracle() {
    use std::fmt::Write as _;

    // Officers 3 and 4 share three companies, of degree 10, 11 and 20.
    let dir = tempfile::tempdir().unwrap();
    let mut officers = "node_id,name,countries,country_codes\n".to_owned();
    let mut rels = "node_id_start,node_id_end,rel_type,link\n".to_owned();
    let mut next = 100;
    for (company, degree) in [(1, 10), (2, 11), (3, 20)] {
        for officer in [10, 20] {
            writeln!(rels, "{officer},{company},officer_of,director").unwrap();
        }
        for _ in 2..degree {
            next += 1;
            writeln!(officers, "{next},filler,,").unwrap();
            writeln!(rels, "{next},{company},officer_of,director").unwrap();
        }
    }
    officers = officers.replacen('\n', "\n10,A,,\n20,B,,\n", 1);
    let files = [
        (
            "nodes-entities.csv",
            "node_id,name\n1,e1\n2,e2\n3,e3\n".to_owned(),
        ),
        ("nodes-officers.csv", officers),
        ("nodes-intermediaries.csv", "node_id,name\n".to_owned()),
        ("nodes-addresses.csv", "node_id,address\n".to_owned()),
        ("nodes-others.csv", "node_id,name\n".to_owned()),
        ("relationships.csv", rels),
    ];
    for (name, body) in files {
        std::fs::write(dir.path().join(name), body).unwrap();
    }
    let w = icij::load(dir.path()).unwrap();
    let (a, b) = (3, 4);
    assert_eq!((w.degree(0), w.degree(1), w.degree(2)), (10, 11, 20));
    let best = 11.0_f64.ln();
    let slack = icij::path_slack(&w, a, b, best);
    assert!(slack[&0].abs() < 1e-9, "{slack:?}");
    assert!((slack[&1] - (12.0_f64.ln() / best - 1.0)).abs() < 1e-9);
    assert!(
        !slack.contains_key(&2),
        "27% dearer than the cheapest: not near it"
    );
    assert_eq!((slack[&a], slack[&b]), (0.0, 0.0));
    let near = icij::near_cheapest(&w, a, b, best);
    assert_eq!(near, slack.keys().copied().collect());
}
