//! Conformance tests ported from upstream `test.py`, plus regressions for upstream bugs.

use super::*;
use std::sync::Mutex;

fn graph() -> (tempfile::TempDir, Graph) {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph::open(dir.path().join("g.db")).unwrap();
    (dir, g)
}

fn ids(it: Result<impl Entries>) -> Vec<LogId> {
    it.unwrap().map(|e| e.unwrap().id).collect()
}

/// Upstream fixture: foo/bar -edge:e1-> foo/baz -edge2:e2-> goo/gaz, one prop each.
fn load(t: &mut Txn<'_>) -> [LogId; 3] {
    let n = [(b"foo", b"bar"), (b"foo", b"baz"), (b"goo", b"gaz")]
        .map(|(ty, v)| t.node(ty, v).unwrap().id);
    for (i, &id) in n.iter().enumerate() {
        t.set(
            id,
            format!("np{i}k").as_bytes(),
            format!("np{i}v").as_bytes(),
        )
        .unwrap();
    }
    t.edge(n[0], n[1], b"edge", b"e1").unwrap();
    t.edge(n[1], n[2], b"edge2", b"e2").unwrap();
    n
}

#[test]
fn commit_and_abort() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    assert_eq!(t.next_id().unwrap(), 1);
    t.node(b"foo", b"bar").unwrap();
    assert_eq!(t.next_id().unwrap(), 2);
    drop(t); // abort
    assert_eq!(g.read().unwrap().next_id().unwrap(), 1);

    let mut t = g.write().unwrap();
    t.node(b"foo", b"bar").unwrap();
    t.commit().unwrap();
    assert_eq!(g.read().unwrap().next_id().unwrap(), 2);
}

#[test]
fn resolve_is_idempotent_and_works_read_only() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"foo", b"bar").unwrap();
    assert_eq!(t.node(b"foo", b"bar").unwrap(), a);
    t.commit().unwrap();
    let mut r = g.read().unwrap();
    assert_eq!(r.node(b"foo", b"bar").unwrap(), a);
    assert!(matches!(r.node(b"foo", b"new"), Err(GraphError::ReadOnly)));
}

#[test]
fn counts_and_history() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    assert_eq!(t.counts(None).unwrap(), (0, 0));
    let n1 = t.node(b"foo", b"bar").unwrap().id;
    let n2 = t.node(b"foo", b"baz").unwrap().id;
    assert_eq!(t.counts(None).unwrap(), (2, 0));
    assert_eq!(t.counts(Some(n2)).unwrap(), (1, 0));
    t.edge(n1, n2, b"e", b"").unwrap();
    assert_eq!(t.counts(None).unwrap(), (2, 1));
    t.commit().unwrap();

    let mut t = g.write().unwrap();
    assert_eq!(t.counts(Some(n2)).unwrap(), (1, 0));
    t.node(b"foo", b"blah").unwrap();
    assert_eq!(t.counts(None).unwrap().0, 3);
    let del = t.delete(n1).unwrap();
    assert_eq!(del, 5, "deletion is one log record");
    assert_eq!(t.counts(None).unwrap(), (2, 0), "edge cascaded");
    t.commit().unwrap();

    let r = g.read().unwrap();
    assert_eq!(r.counts(None).unwrap(), (2, 0));
    assert_eq!(r.counts(Some(del)).unwrap(), (3, 1));
    assert_eq!(r.node_lookup(b"foo", b"bar", None).unwrap(), None);
    assert_eq!(
        r.node_lookup(b"foo", b"bar", Some(del))
            .unwrap()
            .map(|e| e.id),
        Some(n1)
    );
    assert_eq!(ids(r.nodes(None, Some(del))).len(), 3);
    assert_eq!(ids(r.edges(None, None)), Vec::<LogId>::new());
}

#[test]
fn node_delete_only_uncounts_its_own_edges() {
    // Upstream `_nodes_edges_delta` subtracted every edge in the graph for a node deletion.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let [bar, baz, gaz] = load(&mut t);
    t.edge(gaz, gaz, b"loop", b"").unwrap();
    t.commit().unwrap();
    let mut t = g.write().unwrap();
    let del = t.delete(gaz).unwrap(); // takes edge2 and the self-loop
    t.node(b"x", b"y").unwrap(); // force a replay across the deletion
    t.commit().unwrap();
    let r = g.read().unwrap();
    assert_eq!(r.counts(None).unwrap(), (3, 1));
    assert_eq!(
        r.counts(Some(del + 1)).unwrap(),
        (2, 1),
        "replayed mid-txn view"
    );
    assert_eq!(ids(r.node_edges(bar, Direction::Both, None, None)).len(), 1);
    assert_eq!(ids(r.node_edges(baz, Direction::Both, None, None)).len(), 1);
}

#[test]
fn double_delete_is_rejected() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let n = t.node(b"a", b"b").unwrap().id;
    t.delete(n).unwrap();
    assert!(matches!(t.delete(n), Err(GraphError::NotFound(..))));
    assert_eq!(t.counts(None).unwrap(), (0, 0));
}

#[test]
fn edge_endpoints_must_be_live_nodes() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let n = t.node(b"a", b"b").unwrap().id;
    let p = t.set(n, b"k", b"v").unwrap().id;
    assert!(matches!(
        t.edge(n, p, b"e", b""),
        Err(GraphError::NotFound(_, "node"))
    ));
    assert!(matches!(
        t.edge(n, 99, b"e", b""),
        Err(GraphError::NotFound(99, "node"))
    ));
}

#[test]
fn edges_by_type_and_direction() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let [_, baz, _] = load(&mut t);
    let both = ids(t.node_edges(baz, Direction::Both, None, None));
    let ins = ids(t.node_edges(baz, Direction::In, None, None));
    let outs = ids(t.node_edges(baz, Direction::Out, None, None));
    assert_eq!((both.len(), ins.len(), outs.len()), (2, 1, 1));
    assert_ne!(ins, outs);
    assert_eq!(
        ids(t.node_edges(baz, Direction::Both, Some(b"edge2"), None)).len(),
        1
    );
    assert_eq!(
        ids(t.node_edges(baz, Direction::Both, Some(b"nope"), None)).len(),
        0
    );
    assert_eq!(ids(t.nodes(Some(b"foo"), None)).len(), 2);
    assert_eq!(ids(t.edges(Some(b"edge"), None)).len(), 1);
}

#[test]
fn self_loop_yielded_once() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let n = t.node(b"a", b"b").unwrap().id;
    t.edge(n, n, b"self", b"").unwrap();
    assert_eq!(ids(t.node_edges(n, Direction::Both, None, None)).len(), 1);
    assert_eq!(ids(t.node_edges(n, Direction::In, None, None)).len(), 1);
    assert_eq!(ids(t.node_edges(n, Direction::Out, None, None)).len(), 1);
}

#[test]
fn properties_supersede_and_cascade() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    assert_eq!(t.prop(0, b"foo", None).unwrap(), None);
    let p1 = t.set(0, b"foo", b"bar").unwrap();
    assert_eq!(
        t.set(0, b"foo", b"bar").unwrap(),
        p1,
        "same value is a no-op"
    );
    let sub = t.set(p1.id, b"sub", b"x").unwrap();
    let p2 = t.set(0, b"foo", b"baz").unwrap();
    let val = |e: Entry| {
        let Record::Prop { val, .. } = e.record else {
            panic!("not a prop")
        };
        val
    };
    assert_eq!(
        t.string(val(t.prop(0, b"foo", None).unwrap().unwrap()))
            .unwrap(),
        b"baz"
    );
    assert_eq!(
        t.prop(0, b"foo", Some(p2.id)).unwrap().map(|e| e.id),
        Some(p1.id)
    );
    assert_eq!(
        t.entry(sub.id).unwrap().unwrap().next,
        p2.id,
        "sub-property ended with its parent"
    );
    assert!(t.unset(0, b"foo").unwrap());
    assert!(!t.unset(0, b"foo").unwrap());
    assert_eq!(ids(t.props(0, None)), Vec::<LogId>::new());
}

#[test]
fn empty_strings_are_id_zero() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let n = t.node(b"t", b"").unwrap();
    assert_eq!(n.record, Record::Node { ty: 1, val: 0 });
    assert_eq!(t.string(0).unwrap(), b"");
}

#[test]
fn kv() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    for (i, k) in ["f", "fa", "foo", "foobar", "foobaz", "fz"]
        .iter()
        .enumerate()
    {
        t.kv_put(b"foo", k.as_bytes(), i.to_string().as_bytes())
            .unwrap();
    }
    t.kv_put(b"other", b"foo", b"x").unwrap();
    let keys: Vec<&[u8]> = t
        .kv_iter(b"foo", b"foo")
        .unwrap()
        .map(|kv| kv.unwrap().0)
        .collect();
    assert_eq!(keys, [&b"foo"[..], b"foobar", b"foobaz"]);
    assert_eq!(t.kv_get(b"foo", b"fa").unwrap(), Some(&b"1"[..]));
    assert!(t.kv_del(b"foo", b"fa").unwrap());
    assert!(!t.kv_del(b"foo", b"fa").unwrap());
    assert_eq!(t.kv_iter(b"missing", b"").unwrap().count(), 0);
}

#[test]
fn nested() {
    let (_d, g) = graph();
    let get = |t: &Txn<'_>| {
        let Some(Entry {
            record: Record::Prop { val, .. },
            ..
        }) = t.prop(0, b"foo", None).unwrap()
        else {
            panic!("no foo")
        };
        t.string(val).unwrap().to_vec()
    };
    let mut t0 = g.write().unwrap();
    t0.set(0, b"foo", b"t0").unwrap();
    {
        let mut t1 = t0.nested().unwrap();
        t1.set(0, b"foo", b"t1").unwrap();
        let mut t2 = t1.nested().unwrap();
        assert_eq!(get(&t2), b"t1");
        t2.set(0, b"foo", b"t2").unwrap();
        assert_eq!(get(&t2), b"t2");
        // t1 aborts: drop both
    }
    assert_eq!(get(&t0), b"t0");
    {
        let mut t1 = t0.nested().unwrap();
        t1.node(b"n", b"1").unwrap();
        t1.commit().unwrap();
    }
    assert_eq!(
        t0.counts(None).unwrap(),
        (1, 0),
        "child deltas folded into parent"
    );
    t0.commit().unwrap();
    let r = g.read().unwrap();
    assert_eq!(get(&r), b"t0");
    assert_eq!(r.counts(None).unwrap(), (1, 0));
}

#[test]
fn reset() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    t.node(b"a", b"b").unwrap();
    t.commit().unwrap();
    let mut t = g.write().unwrap();
    assert_eq!(t.next_id().unwrap(), 2);
    t.node(b"a", b"b").unwrap(); // warms the string cache, which reset must drop
    t.reset().unwrap();
    assert_eq!(t.next_id().unwrap(), 1);
    t.node(b"c", b"d").unwrap();
    let n = t.node(b"a", b"b").unwrap();
    assert_eq!(n.id, 2);
    t.commit().unwrap();
    let r = g.read().unwrap();
    assert_eq!(r.next_id().unwrap(), 3);
    assert_eq!(r.counts(None).unwrap(), (2, 0));
    assert_eq!(r.string(r.string_id(b"a").unwrap().unwrap()).unwrap(), b"a");
}

#[test]
fn edge_batch_matches_single_edges() {
    // The same entries as edge() one at a time: existing edges found, duplicates within
    // the batch shared, new ones created in spec order; a dead endpoint fails the batch.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let nodes: Vec<LogId> = (0..6_u32)
        .map(|i| t.node(b"t", i.to_string().as_bytes()).unwrap().id)
        .collect();
    let at = |i: usize| nodes[i];
    let existing = t.edge(at(0), at(1), b"e", b"x").unwrap();
    let specs = [
        EdgeSpec {
            src: at(0),
            tgt: at(1),
            ty: b"e",
            val: b"x",
        }, // exists
        EdgeSpec {
            src: at(2),
            tgt: at(3),
            ty: b"e",
            val: b"",
        },
        EdgeSpec {
            src: at(1),
            tgt: at(0),
            ty: b"e",
            val: b"x",
        }, // reverse: distinct
        EdgeSpec {
            src: at(2),
            tgt: at(3),
            ty: b"e",
            val: b"",
        }, // duplicate of #1
        EdgeSpec {
            src: at(4),
            tgt: at(4),
            ty: b"loop",
            val: b"",
        },
        EdgeSpec {
            src: at(5),
            tgt: at(3),
            ty: b"f",
            val: b"y",
        },
    ];
    let got = t.edge_batch(&specs).unwrap();
    assert_eq!(got[0], existing);
    assert_eq!(got[1], got[3]);
    assert!(got[1].id < got[2].id && got[2].id < got[4].id && got[4].id < got[5].id);
    // Running the batch again finds everything.
    assert_eq!(t.edge_batch(&specs).unwrap(), got);
    // The indexes agree with the single-edge path.
    for (i, s) in specs.iter().enumerate() {
        assert_eq!(t.edge(s.src, s.tgt, s.ty, s.val).unwrap(), got[i]);
    }
    t.commit().unwrap();
    let r = g.read().unwrap();
    assert_eq!(r.counts(None).unwrap(), (6, 5));
    // Index contents as sets: the batch writes the same keys as single inserts would.
    let sorted = |v: Vec<LogId>| {
        let mut v = v;
        v.sort_unstable();
        v
    };
    assert_eq!(
        sorted(ids(r.node_edges(at(3), Direction::In, None, None))),
        sorted(vec![got[1].id, got[5].id])
    );
    assert_eq!(
        sorted(ids(r.node_edges(at(0), Direction::Both, None, None))),
        sorted(vec![got[0].id, got[2].id])
    );
    assert_eq!(
        sorted(ids(r.edges(Some(b"e"), None))),
        sorted(vec![got[0].id, got[1].id, got[2].id])
    );
    assert_eq!(ids(r.edges(Some(b"loop"), None)), vec![got[4].id]);
    drop(r);
    let mut t = g.write().unwrap();
    t.delete(at(5)).unwrap();
    let err = t.edge_batch(&specs).unwrap_err();
    assert!(matches!(err, GraphError::NotFound(id, "node") if id == at(5)));
    assert_eq!(
        t.next_id().unwrap(),
        t.next_id().unwrap(),
        "nothing written"
    );
}

#[test]
fn edge_endpoint_liveness_is_rechecked_after_delete() {
    // Endpoints are cached as live per txn; a delete (direct, cascaded, or in a nested
    // txn) must drop them from the cache.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let na = t.node(b"t", b"a").unwrap().id;
    let nb = t.node(b"t", b"b").unwrap().id;
    let nc = t.node(b"t", b"c").unwrap().id;
    t.edge(na, nb, b"e", b"").unwrap();
    t.delete(nb).unwrap();
    assert!(
        matches!(t.edge(na, nb, b"e", b"2"), Err(GraphError::NotFound(id, "node")) if id == nb)
    );
    let mut n = t.nested().unwrap();
    n.edge(na, nc, b"e", b"").unwrap();
    n.delete(nc).unwrap();
    n.commit().unwrap();
    assert!(
        matches!(t.edge(na, nc, b"e", b"2"), Err(GraphError::NotFound(id, "node")) if id == nc)
    );
    t.edge(na, na, b"e", b"loop").unwrap();
    t.commit().unwrap();
    assert_eq!(g.read().unwrap().counts(None).unwrap(), (1, 1));
}

#[test]
fn map_grows_past_the_initial_size() {
    // Tests use a 1 MiB pad: ~6 MiB over many txns forces several resizes.
    let (_d, g) = graph();
    let initial = g.env.info().map_size;
    let blob = vec![b'x'; 64 << 10];
    let mut ids = Vec::new();
    for i in 0..96_u32 {
        let mut t = g.write().unwrap();
        let n = t.node(b"t", &i.to_be_bytes()).unwrap().id;
        t.set(n, b"blob", &[blob.as_slice(), &i.to_be_bytes()].concat())
            .unwrap();
        t.commit().unwrap();
        ids.push(n);
    }
    assert!(g.env.info().map_size > initial, "map grew");
    assert!(
        g.size().unwrap() > u64::try_from(4 * PAD).unwrap(),
        "file outgrew several pads"
    );
    let r = g.read().unwrap();
    assert_eq!(r.counts(None).unwrap().0, 96);
    assert!(r.prop(ids[95], b"blob", None).unwrap().is_some());
}

#[test]
fn growth_waits_for_readers_on_other_threads() {
    let (_d, g) = graph();
    let blob = vec![b'y'; 256 << 10];
    std::thread::scope(|s| {
        let (tx, rx) = std::sync::mpsc::channel();
        let g = &g;
        let reader = s.spawn(move || {
            let r = g.read().unwrap();
            tx.send(()).unwrap();
            std::thread::park_timeout(Duration::from_millis(200));
            r.counts(None).unwrap()
        });
        rx.recv().unwrap();
        // Enough data to need growth while the reader holds its txn: the writer must wait.
        for i in 0..12_u8 {
            let mut t = g.write().unwrap();
            let n = t.node(b"w", &[i]).unwrap().id;
            t.set(n, b"blob", &[blob.as_slice(), &[i]].concat())
                .unwrap();
            t.commit().unwrap();
        }
        assert_eq!(reader.join().unwrap(), (0, 0), "reader kept its snapshot");
    });
    assert_eq!(g.read().unwrap().counts(None).unwrap().0, 12);
}

#[test]
fn one_txn_larger_than_the_pad_fails_cleanly() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let n = t.node(b"t", b"big").unwrap().id;
    let blob = vec![b'z'; 3 * PAD];
    assert!(
        matches!(t.set(n, b"blob", &blob), Err(GraphError::Lmdb(_))),
        "map full"
    );
    drop(t);
    let mut t = g.write().unwrap();
    t.node(b"t", b"after").unwrap();
    t.commit().unwrap();
    assert_eq!(g.read().unwrap().counts(None).unwrap().0, 1);
}

#[test]
fn snapshot_and_reopen() {
    let (d, g) = graph();
    let mut t = g.write().unwrap();
    load(&mut t);
    t.commit().unwrap();
    // Out-of-order inserts and a delete, so pages are half full and history matters.
    let mut t = g.write().unwrap();
    for i in (0..3000_u32).rev() {
        let n = t.node(b"many", i.to_string().as_bytes()).unwrap().id;
        t.set_value(n, "k", &json!(i)).unwrap();
    }
    let gone = t.node(b"many", b"7").unwrap().id;
    let del = t.delete(gone).unwrap();
    t.kv_put(b"d", b"key", b"value").unwrap();
    t.commit().unwrap();
    let copy = d.path().join("copy.db");
    g.snapshot(&copy).unwrap();
    assert!(g.snapshot(&copy).is_err(), "refuses to overwrite");
    let dump = |g: &Graph| {
        let t = g.read().unwrap();
        let log: Vec<Entry> = t.log(1, None).unwrap().map(Result::unwrap).collect();
        let strings: Vec<Vec<u8>> = (1..=200)
            .map_while(|i| t.string(i).ok().map(<[u8]>::to_vec))
            .collect();
        (
            log,
            strings,
            t.counts(None).unwrap(),
            t.next_id().unwrap(),
            t.kv_get(b"d", b"key").unwrap().map(<[u8]>::to_vec),
        )
    };
    let before = dump(&g);
    let size = std::fs::metadata(d.path().join("g.db")).unwrap().len();
    drop(g);
    assert!(
        !d.path().join("copy.db-lock").exists(),
        "no lock file left behind"
    );
    assert!(
        std::fs::metadata(&copy).unwrap().len() < size,
        "packed is smaller"
    );
    let g = Graph::open(&copy).unwrap();
    assert_eq!(dump(&g), before);
    let t = g.read().unwrap();
    assert_eq!(t.counts(None).unwrap(), (3 + 2999, 2));
    assert_eq!(t.counts(Some(del)).unwrap().0, 3 + 3000, "history survives");
    drop(t);
    // Still writable, and the indexes still find what is there.
    let mut t = g.write().unwrap();
    assert_eq!(
        t.node(b"many", b"42").unwrap(),
        t.node(b"many", b"42").unwrap()
    );
    let id = t.node(b"many", b"new").unwrap().id;
    t.commit().unwrap();
    assert_eq!(
        g.read()
            .unwrap()
            .node_lookup(b"many", b"new", None)
            .unwrap()
            .unwrap()
            .id,
        id
    );
}

#[test]
fn fifo() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    t.fifo_push(b"q", &[b"1", b"2", b"3"]).unwrap();
    assert_eq!(
        t.fifo_pop(b"q", 4).unwrap(),
        [b"1", b"2", b"3"],
        "pop more than queued"
    );
    assert_eq!(t.fifo_len(b"q").unwrap(), 0);
    t.fifo_push(b"q", &[b"4", b"5", b"6"]).unwrap();
    assert_eq!(t.fifo_len(b"q").unwrap(), 3);
    assert_eq!(t.fifo_pop(b"q", 1).unwrap(), [b"4"]);
    t.fifo_push(b"q", &[b"7"]).unwrap();
    assert_eq!(
        t.fifo_pop(b"q", 9).unwrap(),
        [&b"5"[..], b"6", b"7"],
        "order kept across pushes"
    );
    assert_eq!(t.fifo_pop(b"missing", 1).unwrap(), Vec::<Vec<u8>>::new());
}

#[test]
fn log_scan() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"a", b"1").unwrap().id;
    let b = t.node(b"a", b"2").unwrap().id;
    t.edge(a, b, b"e", b"").unwrap();
    t.set(a, b"k", b"v").unwrap();
    t.delete(a).unwrap();
    let kinds: Vec<_> = t
        .log(1, None)
        .unwrap()
        .map(|e| e.unwrap().record.kind())
        .collect();
    assert_eq!(kinds, ["node", "node", "edge", "property", "deletion"]);
    let ends: Vec<_> = t
        .log(2, Some(4))
        .unwrap()
        .map(|e| e.unwrap().next)
        .collect();
    assert_eq!(ends, [0, 5], "ids 2..4; the cascaded edge ended at 5");
}

#[test]
fn update_id() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"a", b"1").unwrap().id; // 1
    assert_eq!(t.update_id(a, None).unwrap(), 1);
    t.node(b"a", b"2").unwrap(); // 2
    t.set(a, b"k", b"v").unwrap(); // 3
    t.set(a, b"k", b"w").unwrap(); // 4, ends 3
    assert_eq!(t.update_id(a, None).unwrap(), 4);
    assert_eq!(t.update_id(a, Some(4)).unwrap(), 3);
    assert_eq!(t.update_id(a, Some(3)).unwrap(), 1);
    t.set(0, b"g", b"x").unwrap(); // 5
    assert_eq!(t.update_id(0, None).unwrap(), 5);
    t.delete(a).unwrap(); // 6
    assert_eq!(t.update_id(a, None).unwrap(), 6);
}

// ── LGQL ──

use crate::lgql::Pattern;
use serde_json::{Value, json};

/// Runs one ad-hoc pattern and returns chains as `type:value` labels.
fn q(t: &Txn<'_>, src: &str, before: Option<LogId>) -> Vec<String> {
    let p = Pattern::parse(src).unwrap();
    let mut out = Vec::new();
    t.query(&[p], before, |_, c| {
        out.push(c.iter().map(|e| label(t, e)).collect::<Vec<_>>().join(" "));
        true
    })
    .unwrap();
    out.sort();
    out
}

fn label(t: &Txn<'_>, e: &Entry) -> String {
    let s = |id| String::from_utf8_lossy(t.string(id).unwrap()).into_owned();
    match e.record {
        Record::Node { ty, val } => format!("{}:{}", s(ty), s(val)),
        Record::Edge { ty, val, .. } => format!("[{}:{}]", s(ty), s(val)),
        Record::Prop { .. } | Record::Deletion { .. } => panic!("not a node or edge"),
    }
}

/// Runs streaming patterns over `start..` and returns `(pattern, chain)` labels in emit order.
fn mq(t: &Txn<'_>, pats: &[&str], start: LogId) -> Vec<String> {
    let ps: Vec<_> = pats.iter().map(|s| Pattern::parse(s).unwrap()).collect();
    let mut out = Vec::new();
    t.mquery(&ps, start, None, |pi, _, c| {
        out.push(format!(
            "{pi} {}",
            c.iter().map(|e| label(t, e)).collect::<Vec<_>>().join(" ")
        ));
        true
    })
    .unwrap();
    out
}

#[test]
fn query_basics() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    load(&mut t);
    assert_eq!(
        q(&t, "n(type='foo')->e()-n()", None),
        ["foo:bar [edge:e1] foo:baz", "foo:baz [edge2:e2] goo:gaz"]
    );
    assert_eq!(q(&t, "n()", None).len(), 3);
    assert_eq!(q(&t, "e()", None).len(), 2);
    assert_eq!(
        q(&t, "n()->n()", None),
        ["foo:bar foo:baz", "foo:baz goo:gaz"]
    );
    assert_eq!(
        q(&t, "n()<-n()", None),
        ["foo:baz foo:bar", "goo:gaz foo:baz"]
    );
    assert_eq!(
        q(&t, "n()-n()", None).len(),
        4,
        "undirected: each pair both ways"
    );
    assert_eq!(
        q(&t, "n(type='goo')<-e()<-@n()<-n(value='bar')", None),
        ["goo:gaz [edge2:e2] foo:bar"]
    );
    assert_eq!(q(&t, "n(value~/^ba/)", None), ["foo:bar", "foo:baz"]);
    assert_eq!(q(&t, "n(value!~/^ba/)", None), ["goo:gaz"]);
    assert_eq!(
        q(&t, "n(np0k)", None),
        Vec::<String>::new(),
        "props are msgpack; raw bytes do not resolve"
    );
    assert_eq!(q(&t, "e(src.value='baz')", None), ["[edge2:e2]"]);
    assert_eq!(q(&t, "n(edge_count=2)", None), ["foo:baz"]);
    assert_eq!(
        q(&t, "n(outbound_count=0, inbound_count>0)", None),
        ["goo:gaz"]
    );
    assert_eq!(
        q(&t, "n(type=['foo','goo'], value=['bar','gaz'])", None),
        ["foo:bar", "goo:gaz"]
    );
    let id = t.node_lookup(b"foo", b"baz", None).unwrap().unwrap().id;
    assert_eq!(
        q(&t, &format!("n(ID={id})-e()"), None),
        ["foo:baz [edge2:e2]", "foo:baz [edge:e1]"]
    );
    assert_eq!(q(&t, "n(nope=1)-n()", None), Vec::<String>::new());
}

#[test]
fn query_values_and_types() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let n = |t: &mut Txn<'_>, v: &[u8], props: Value| {
        let id = t.node(b"t", v).unwrap().id;
        for (k, v) in props.as_object().unwrap() {
            t.set_value(id, k, v).unwrap();
        }
    };
    n(
        &mut t,
        b"a",
        json!({"x": 1, "o": {"k": "deep"}, "f": 1.5, "b": true}),
    );
    n(&mut t, b"b", json!({"x": 2.0, "s": "str", "l": [1, 2]}));
    n(&mut t, b"c", json!({"x": "1"}));
    assert_eq!(q(&t, "n(x=1)", None), ["t:a"]);
    assert_eq!(q(&t, "n(x=2)", None), ["t:b"], "2 == 2.0");
    assert_eq!(
        q(&t, "n(x>=1)", None),
        ["t:a", "t:b"],
        "strings never compare with numbers"
    );
    assert_eq!(q(&t, "n(x>'0')", None), ["t:c"]);
    assert_eq!(q(&t, "n(x!=1)", None), ["t:b", "t:c"]);
    assert_eq!(q(&t, "n(o.k='deep')", None), ["t:a"]);
    assert_eq!(q(&t, "n(f=1.5)", None), ["t:a"]);
    assert_eq!(q(&t, "n(b=1)", None), Vec::<String>::new(), "true is not 1");
    assert_eq!(q(&t, "n(b=true)", None), ["t:a"]);
    assert_eq!(q(&t, "n(x:number)", None), ["t:a", "t:b"]);
    assert_eq!(q(&t, "n(x!:number)", None), ["t:c"]);
    assert_eq!(
        q(&t, "n(l:array, o:[object,array])", None),
        Vec::<String>::new()
    );
    assert_eq!(q(&t, "n(l:[array,object])", None), ["t:b"]);
    assert_eq!(
        q(&t, "n(s~/t/, s~/s/)", None),
        ["t:b"],
        "repeated ~ is AND (upstream: cannot match)"
    );
    assert_eq!(q(&t, "n(x=1, x=2)", None), Vec::<String>::new());
}

#[test]
fn projection_answers_like_lmdb() {
    // Every query shape the executor has, with and without a projection, on a graph
    // with deletions, a superseded property, a self-loop, edge properties, and a
    // property on a property.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let mut ids = Vec::new();
    for x in 0..40_u64 {
        let id = t
            .node(format!("t{}", x % 3).as_bytes(), x.to_string().as_bytes())
            .unwrap()
            .id;
        t.set_value(id, &format!("p{}", x % 2), &json!(format!("v{}", x % 4)))
            .unwrap();
        t.set_value(id, "n", &json!(x)).unwrap();
        ids.push(id);
    }
    for x in 0..40_u64 {
        let y = (x * 7 + 3) % 40;
        let at = |i: u64| ids[usize::try_from(i).unwrap()];
        let e = t
            .edge(at(x), at(y), format!("e{}", y % 3).as_bytes(), b"")
            .unwrap();
        t.set_value(e.id, "w", &json!(x % 5)).unwrap();
    }
    let lp = t.edge(ids[5], ids[5], b"loop", b"").unwrap();
    t.set_value(lp.id, "w", &json!(9)).unwrap();
    t.set_value(ids[1], "p1", &json!("v0")).unwrap(); // supersedes
    let p = t.set(ids[2], b"o", b"\xc0").unwrap();
    t.set_value(p.id, "k", &json!("deep")).unwrap();
    t.delete(ids[7]).unwrap(); // takes its edges and props
    t.unset(ids[8], b"n").unwrap();
    t.commit().unwrap();

    let patterns = [
        "n()",
        "e()",
        "n(type='t1')",
        "n(type='t2', value='8')",
        "n(p1='v0')",
        "n(p0=['v2','v3'])",
        "n(n>=30)",
        "n(n!=1)",
        "n(ID=[3,5,7,9])",
        "e(type='e2')",
        "e(w=4)",
        "e(src.type='t0')",
        "n(type='t1')->e()->n()",
        "n(type='t1')<-e()<-n()",
        "n(type='t0')-e(type=['e1','loop'])-n()",
        "n(value='5')-n()",
        "n(type='t1')-n()-n()",
        "n(edge_count>=3)",
        "n(neighbor_count=1)",
        "n(outbound_count=1)->e()->n(inbound_count>1)",
        // Seeded mid-chain (at the typed edge): the left hop must start from the edge.
        "n()-e(type='e2')-n()",
        "n()<-e(type=['e0','loop'])-n(type='t1')",
    ];
    let t = g.read().unwrap();
    let plain: Vec<_> = patterns.iter().map(|p| q(&t, p, None)).collect();
    assert!(g.cached_projection().is_none());
    let proj = t.projection().unwrap();
    assert_eq!(proj.end(), t.next_id().unwrap());
    assert!(proj.bytes() > 0);
    for (p, want) in patterns.iter().zip(&plain) {
        assert_eq!(&q(&t, p, None), want, "{p}");
        assert!(
            !want.is_empty() || p.contains("ID=") || p.contains("neighbor_count=1"),
            "{p} tests nothing"
        );
    }
    // The parallel runner yields the same chains, in some order. (Sinks run on other
    // threads, which cannot use this thread's txn, so `q_par` labels afterwards.)
    for (p, want) in patterns.iter().zip(&plain) {
        assert_eq!(&q_par(&g, &proj, &t, p), want, "{p} in parallel");
    }
    // Stopping from one thread stops the rest.
    let seen = std::sync::atomic::AtomicUsize::new(0);
    proj.query_par(&g, &[Pattern::parse("n()").unwrap()], 3, |_, _, _| {
        seen.fetch_add(1, Ordering::Relaxed) < 2
    })
    .unwrap();
    assert!(seen.into_inner() < 39);
    // A historical view never uses the projection: fewer nodes existed at log position 20.
    assert!(q(&t, "n(type='t1')", Some(20)).len() < plain[2].len());
    drop(t);
    // A commit leaves the cached projection behind; a query until it is advanced
    // reads LMDB and sees the change, and advancing replays only the new entry.
    let mut w = g.write().unwrap();
    w.delete(ids[0]).unwrap();
    w.commit().unwrap();
    let stale = g.cached_projection().unwrap();
    let t = g.read().unwrap();
    assert!(stale.end() < t.next_id().unwrap());
    assert_eq!(q(&t, "n(value='0')", None), Vec::<String>::new());
    let advanced = t.projection().unwrap();
    assert_eq!(advanced.end(), t.next_id().unwrap());
    assert_eq!(advanced.delta_len(), 1);
    assert_eq!(q(&t, "n(value='0')", None), Vec::<String>::new());
}

/// Runs `src` on LMDB only, as sorted labels.
fn q_lmdb(t: &Txn<'_>, src: &str) -> Vec<String> {
    let p = Pattern::parse(src).unwrap();
    let mut out = Vec::new();
    t.query_with(None, &[p], None, |_, c| {
        out.push(c.iter().map(|e| label(t, e)).collect::<Vec<_>>().join(" "));
        true
    })
    .unwrap();
    out.sort();
    out
}

/// splitmix64 values in `0..n`.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        usize::try_from(z ^ (z >> 31)).unwrap() % n.max(1)
    }

    fn pick(&mut self, v: &[LogId]) -> LogId {
        v[self.below(v.len())]
    }
}

/// Live IDs of the random workload.
#[derive(Default)]
struct Live {
    nodes: Vec<LogId>,
    edges: Vec<LogId>,
    next: u64,
}

/// One random write: a node, an edge (sometimes a self-loop), a property set,
/// overwritten, nested, or removed, or a delete of an edge or a node.
fn random_write(t: &mut Txn<'_>, rng: &mut Rng, live: &mut Live, round: usize) {
    let alive = |t: &Txn<'_>, id: &LogId| t.entry(*id).unwrap().is_some_and(|e| e.next == 0);
    live.nodes.retain(|id| alive(t, id));
    live.edges.retain(|id| alive(t, id));
    live.next += 1;
    let val = live.next.to_string();
    let have_nodes = live.nodes.len() > 5;
    match rng.below(10) {
        2 | 3 if have_nodes => {
            let a = rng.pick(&live.nodes);
            let b = if rng.below(5) == 0 {
                a
            } else {
                rng.pick(&live.nodes)
            };
            let ty = if a == b {
                "loop".to_owned()
            } else {
                format!("e{}", rng.below(3))
            };
            live.edges
                .push(t.edge(a, b, ty.as_bytes(), val.as_bytes()).unwrap().id);
        }
        4 | 5 if have_nodes => {
            let n = rng.pick(&live.nodes);
            t.set_value(n, "p", &json!(format!("v{}", rng.below(4))))
                .unwrap();
            if rng.below(3) == 0 {
                t.set_value(n, "q", &json!("x")).unwrap();
            }
            if rng.below(4) == 0 {
                let p = t.prop(n, b"p", None).unwrap().unwrap().id;
                t.set_value(p, "meta", &json!(round)).unwrap();
            }
        }
        6 if !live.edges.is_empty() => {
            let e = rng.pick(&live.edges);
            let w = if rng.below(2) == 0 { "y" } else { "z" };
            t.set_value(e, "w", &json!(w)).unwrap();
        }
        7 if have_nodes => {
            let key: &[u8] = if rng.below(2) == 0 { b"p" } else { b"q" };
            t.unset(rng.pick(&live.nodes), key).unwrap();
        }
        8 if !live.edges.is_empty() => {
            t.delete(rng.pick(&live.edges)).unwrap();
        }
        9 if have_nodes => {
            t.delete(rng.pick(&live.nodes)).unwrap();
        }
        _ => {
            let ty = format!("t{}", rng.below(3));
            live.nodes
                .push(t.node(ty.as_bytes(), val.as_bytes()).unwrap().id);
        }
    }
}

/// Every entry, its properties, and every node's adjacency in each direction, as the
/// projection answers and as LMDB does.
fn check_entries(r: &Txn<'_>, proj: &Projection, round: usize) {
    let (cn, ce) = r.counts(None).unwrap();
    assert_eq!(
        proj.len(),
        usize::try_from(cn + ce).unwrap(),
        "round {round}"
    );
    for id in 1..r.next_id().unwrap() {
        let live = r.entry(id).unwrap().filter(|e| e.next == 0);
        let object = live.filter(|e| matches!(e.record, Record::Node { .. } | Record::Edge { .. }));
        assert_eq!(proj.entry(id), object, "round {round}: entry {id}");
        if live.is_some_and(|e| !matches!(e.record, Record::Deletion { .. })) {
            let want: Vec<(StrId, StrId)> = r
                .props(id, None)
                .unwrap()
                .filter_map(|p| match p.unwrap().record {
                    Record::Prop { key, val, .. } => Some((key, val)),
                    Record::Node { .. } | Record::Edge { .. } | Record::Deletion { .. } => None,
                })
                .collect();
            assert_eq!(proj.props_of(id), want, "round {round}: props of {id}");
        }
        if object.is_some_and(|e| matches!(e.record, Record::Node { .. })) {
            for dir in [Direction::In, Direction::Out, Direction::Both] {
                let (mut out, mut rows) = (Vec::new(), Vec::new());
                proj.node_edges(id, dir, None, &mut out, &mut rows).unwrap();
                let got: Vec<LogId> = out.iter().map(|e| e.id).collect();
                assert_eq!(
                    got,
                    ids(r.node_edges(id, dir, None, None)),
                    "round {round}: {dir:?} {id}"
                );
            }
        }
    }
}

/// `p` through `query_par`, as sorted labels.
fn q_par(g: &Graph, proj: &Projection, r: &Txn<'_>, p: &str) -> Vec<String> {
    let got = Mutex::new(Vec::new());
    proj.query_par(g, &[Pattern::parse(p).unwrap()], 3, |_, _, c| {
        got.lock().unwrap().push(c.to_vec());
        true
    })
    .unwrap();
    let mut got: Vec<String> = got
        .into_inner()
        .unwrap()
        .iter()
        .map(|c| c.iter().map(|e| label(r, e)).collect::<Vec<_>>().join(" "))
        .collect();
    got.sort();
    got
}

#[test]
fn projection_advances_like_lmdb() {
    // A random workload; after every commit the advanced projection must answer
    // exactly as LMDB: entries, adjacency order, properties, queries, parallel runs.
    let (_d, g) = graph();
    let mut rng = Rng(7);
    let mut live = Live::default();
    let mut t = g.write().unwrap();
    for i in 0..30 {
        random_write(&mut t, &mut rng, &mut live, i);
    }
    t.commit().unwrap();
    g.read().unwrap().projection().unwrap(); // the base
    let patterns = [
        "n()",
        "e()",
        "n(type='t1')",
        "e(type='e2')",
        "n(p='v1')",
        "n(p=['v0','v3'])",
        "n(q='x')",
        "e(w='y')",
        "n(type='t0')->e()->n()",
        "n(type='t2')<-e()<-n()",
        "n()-e(type=['e0','loop'])-n()",
        "n(edge_count>=2)",
        "n(neighbor_count=1)",
        "n(p='v2')-n()-n()",
        "e(src.p='v1')",
    ];
    for round in 0..150 {
        let mut t = g.write().unwrap();
        for _ in 0..=rng.below(3) {
            random_write(&mut t, &mut rng, &mut live, round);
        }
        // A projection taken inside a write txn must not reach the cache.
        let before = g.cached_projection().map(|p| p.end());
        t.projection().unwrap();
        assert_eq!(g.cached_projection().map(|p| p.end()), before);
        t.commit().unwrap();

        let r = g.read().unwrap();
        let proj = r.projection().unwrap();
        assert_eq!(proj.end(), r.next_id().unwrap(), "round {round}");
        check_entries(&r, &proj, round);
        for p in patterns {
            assert_eq!(q(&r, p, None), q_lmdb(&r, p), "round {round}: {p}");
            if round % 25 == 0 {
                assert_eq!(
                    q_par(&g, &proj, &r, p),
                    q_lmdb(&r, p),
                    "round {round}: {p} ×3"
                );
            }
        }
    }
    let last = g.cached_projection().unwrap();
    assert!(last.delta_len() > 150, "advanced, never rebuilt");
}

#[test]
fn projection_rebuilds_within_a_duty_cycle() {
    let now = Instant::now();
    let cost = Duration::from_millis(100);
    assert!(rebuild_due(None, now, 4), "never built");
    assert!(!rebuild_due(Some((now, cost)), now, 4), "just built");
    assert!(
        !rebuild_due(Some((now, cost)), now + Duration::from_millis(399), 4),
        "too soon"
    );
    assert!(rebuild_due(
        Some((now, cost)),
        now + Duration::from_millis(400),
        4
    ));
    assert!(rebuild_due(Some((now, cost)), now, 0), "duty 0: always");

    // End to end: a cached one is returned whatever the timing; after a commit, a
    // rebuild is due only once the last build is old enough relative to its cost.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    t.node(b"t", b"a").unwrap();
    t.commit().unwrap();
    let t = g.read().unwrap();
    let first = t.projection_if_due(4).unwrap().expect("never built: due");
    assert!(Arc::ptr_eq(
        &first,
        &t.projection_if_due(4).unwrap().unwrap()
    ));
    drop(t);
    // An ordinary commit is advanced over whatever the timing.
    let mut w = g.write().unwrap();
    w.node(b"t", b"b").unwrap();
    w.commit().unwrap();
    *g.built.lock().unwrap() = Some((Instant::now(), Duration::from_secs(60)));
    let t = g.read().unwrap();
    assert_eq!(t.projection_if_due(4).unwrap().unwrap().delta_len(), 1);
    drop(t);
    // A reset forces a full build, which the duty cycle governs: a slow recent build
    // defers it, an old one allows it.
    let mut w = g.write().unwrap();
    w.reset().unwrap();
    w.node(b"t", b"c").unwrap();
    w.commit().unwrap();
    let t = g.read().unwrap();
    assert!(t.projection_if_due(4).unwrap().is_none());
    *g.built.lock().unwrap() = Some((
        Instant::now()
            .checked_sub(Duration::from_secs(300))
            .unwrap(),
        Duration::from_secs(60),
    ));
    assert_eq!(
        t.projection_if_due(4).unwrap().unwrap().end(),
        t.next_id().unwrap()
    );
}

#[test]
fn projection_props_of_matches_props() {
    // Properties per row come from an offset table; a property's own properties are
    // not rows and are found by search; both agree with the index in key order.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let na = t.node(b"t", b"a").unwrap().id;
    let nb = t.node(b"t", b"b").unwrap().id;
    let nc = t.node(b"t", b"c").unwrap().id;
    for (k, v) in [("z", 1), ("a", 2), ("m", 3)] {
        t.set_value(na, k, &json!(v)).unwrap();
    }
    t.set_value(na, "nested", &json!("x")).unwrap();
    let nested = t.prop(na, b"nested", None).unwrap().unwrap().id;
    t.set_value(nested, "inner", &json!("y")).unwrap();
    t.set_value(nested, "inner2", &json!("z")).unwrap();
    let edge = t.edge(na, nc, b"e", b"").unwrap().id;
    t.set_value(edge, "w", &json!(1)).unwrap();
    t.set_value(nc, "gone", &json!(1)).unwrap();
    t.unset(nc, b"gone").unwrap();
    t.commit().unwrap();
    let t = g.read().unwrap();
    let proj = t.projection().unwrap();
    for parent in [na, nb, nc, nested, edge] {
        let want: Vec<(StrId, StrId)> = t
            .props(parent, None)
            .unwrap()
            .filter_map(|p| match p.unwrap().record {
                Record::Prop { key, val, .. } => Some((key, val)),
                Record::Node { .. } | Record::Edge { .. } | Record::Deletion { .. } => None,
            })
            .collect();
        assert_eq!(proj.props_of(parent), want, "parent {parent}");
    }
    assert_eq!(proj.props_of(na).len(), 4);
    assert_eq!(proj.props_of(nested).len(), 2);
    assert_eq!(proj.props_of(nb).len() + proj.props_of(nc).len(), 0);
    assert_eq!(proj.props_of(999).len(), 0);
}

#[test]
fn query_string_equality_by_id() {
    // `=`/`!=` against string literals compares interned IDs: same answers as resolving.
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"t", b"a").unwrap().id;
    let b = t.node(b"t", b"b").unwrap().id;
    t.set_value(a, "s", &json!("str")).unwrap();
    t.set_value(b, "s", &json!(7)).unwrap();
    t.edge(a, b, b"e", b"").unwrap();
    let none = Vec::<String>::new();
    assert_eq!(q(&t, "n(s='str')", None), ["t:a"]);
    assert_eq!(q(&t, "n(s!='str')", None), ["t:b"], "7 is not 'str'");
    assert_eq!(q(&t, "n(s='7')", None), none, "'7' is not 7");
    assert_eq!(q(&t, "n(s=['none','str'])", None), ["t:a"]);
    assert_eq!(q(&t, "n(value='b')", None), ["t:b"]);
    assert_eq!(q(&t, "n(type!='t')", None), none);
    assert_eq!(
        q(&t, "n(type='nope')", None),
        none,
        "literal never interned"
    );
    assert_eq!(
        q(&t, "n(nope!='x')", None),
        none,
        "unresolved keys fail != too"
    );
    assert_eq!(
        q(&t, "e(src!='1')", None),
        ["[e:]"],
        "src is an ID, not a property"
    );
    assert_eq!(q(&t, "e(value='')", None), ["[e:]"]);
}

#[test]
fn edge_type_pushdown() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"t", b"a").unwrap().id;
    for (i, ty) in [b"e1", b"e2", b"e3"].iter().enumerate() {
        let b = t.node(b"t", format!("b{i}").as_bytes()).unwrap().id;
        t.edge(a, b, *ty, b"").unwrap();
        t.edge(b, a, *ty, b"back").unwrap();
    }
    assert_eq!(
        q(&t, "n(value='a')->e(type=['e1','e3'])->n()", None),
        ["t:a [e1:] t:b0", "t:a [e3:] t:b2"]
    );
    assert_eq!(
        q(&t, "n(value='a')<-e(type='e2')-n()", None),
        ["t:a [e2:back] t:b1"]
    );
    assert_eq!(q(&t, "n(value='a')-e(type='e2')-n()", None).len(), 2);
    assert_eq!(
        q(&t, "n(value='a')-e(type=1)-n()", None),
        Vec::<String>::new()
    );
    assert_eq!(
        q(&t, "n(value='a')-e(type='nope')-n()", None),
        Vec::<String>::new()
    );
}

#[test]
fn query_list_valued_natives() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let [bar, baz, gaz] = load(&mut t);
    t.edge(baz, baz, b"loop", b"").unwrap();
    t.edge(gaz, baz, b"back", b"").unwrap();
    assert_eq!(
        q(&t, "n(neighbor_count=2)", None),
        ["foo:baz"],
        "loop excluded, gaz counted once"
    );
    assert_eq!(
        q(&t, "n(neighbor_types.goo=1, neighbor_types.foo=1)", None),
        ["foo:baz"]
    );
    assert_eq!(
        q(&t, &format!("n(neighborIDs={bar})"), None),
        Vec::<String>::new(),
        "a list never equals a scalar"
    );
    assert_eq!(q(&t, "n(edges:array, inboundIDs:array)", None).len(), 3);
    assert_eq!(q(&t, "n(outbound_count=0)", None), Vec::<String>::new());
    let ids = |key: &str, n: LogId| {
        let e = t.entry(n).unwrap().unwrap();
        t.resolve(&e, &[key.to_owned()], None, &query::Keys::default(), None)
            .unwrap()
            .unwrap()
    };
    assert_eq!(ids("neighbors", baz), json!([bar, gaz]));
    assert_eq!(ids("inboundIDs", gaz).as_array().unwrap().len(), 1);
    assert_eq!(
        ids("edgeIDs", baz).as_array().unwrap().len(),
        4,
        "e1, e2, loop once, back"
    );
}

#[test]
fn query_uniqueness_and_history() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"t", b"a").unwrap().id;
    t.edge(a, a, b"loop", b"").unwrap();
    let b = t.node(b"t", b"b").unwrap().id;
    let ab = t.edge(a, b, b"ab", b"").unwrap().id;
    assert_eq!(
        q(&t, "n()-e()-n()", None),
        ["t:a [ab:] t:b", "t:b [ab:] t:a"],
        "self-loop needs N()"
    );
    assert_eq!(
        q(&t, "n(value='a')-e(type='loop')-N()", None),
        ["t:a [loop:] t:a"]
    );
    let del = t.delete(ab).unwrap();
    assert_eq!(q(&t, "n()->e(type='ab')->n()", None), Vec::<String>::new());
    assert_eq!(
        q(&t, "n()->e(type='ab')->n()", Some(del)),
        ["t:a [ab:] t:b"]
    );
}

#[test]
fn pattern_matches_plain_objects() {
    let v = json!({"graph": "x", "meta": {"name": "alpha"}, "nodes_count": 3});
    let m = |s: &str| {
        Pattern::parse(&format!("n({s})"))
            .unwrap()
            .matches_value(&v)
    };
    assert!(m("meta.name='alpha'"));
    assert!(m("nodes_count>2, graph"));
    assert!(!m("meta.name~/beta/"));
    assert!(!m("ID"));
}

#[test]
fn query_stops_when_sink_says() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    load(&mut t);
    let p = [
        Pattern::parse("n()").unwrap(),
        Pattern::parse("e()").unwrap(),
    ];
    let mut n = 0;
    t.query(&p, None, |_, _| {
        n += 1;
        n < 2
    })
    .unwrap();
    assert_eq!(n, 2);
}

#[test]
fn streaming() {
    let (_d, g) = graph();
    let mut t = g.write().unwrap();
    let a = t.node(b"t", b"a").unwrap().id; // 1
    let b = t.node(b"t", b"b").unwrap().id; // 2
    assert_eq!(mq(&t, &["n()"], 1), ["0 t:a", "0 t:b"]);
    assert_eq!(mq(&t, &["n()"], 2), ["0 t:b"]);
    let start = t.next_id().unwrap();
    t.edge(a, b, b"e", b"").unwrap();
    // Upstream suppressed the target node here (shared `seen` across src and tgt).
    assert_eq!(mq(&t, &["n(edge_count=1)"], start), ["0 t:a", "0 t:b"]);
    assert_eq!(
        mq(&t, &["n()->n()"], start),
        ["0 t:a t:b"],
        "reported once per entry"
    );

    let start = t.next_id().unwrap();
    t.set_value(a, "k", &json!(5)).unwrap();
    t.set_value(a, "k", &json!(6)).unwrap(); // still matches: no new report
    t.set_value(a, "k", &json!(0)).unwrap();
    t.set_value(a, "k", &json!(7)).unwrap(); // matches again
    assert_eq!(
        mq(&t, &["n(k>1, k<9)"], start),
        ["0 t:a", "0 t:a"],
        "two flips, no per-test duplicates"
    );
    assert_eq!(
        mq(&t, &["e(src.k=5)"], start),
        ["0 [e:]"],
        "endpoint property re-fires edge slots"
    );

    let start = t.next_id().unwrap();
    let ab = t.edge_lookup(a, b, b"e", b"", None).unwrap().unwrap().id;
    t.delete(ab).unwrap();
    assert_eq!(
        mq(&t, &["n(edge_count=0)"], start),
        ["0 t:a", "0 t:b"],
        "deletions trigger"
    );
    assert_eq!(mq(&t, &["n()"], 999), Vec::<String>::new());
}
