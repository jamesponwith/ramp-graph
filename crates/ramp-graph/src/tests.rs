//! Conformance tests ported from upstream `test.py`, plus regressions for upstream bugs.

use super::*;

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
            std::thread::park_timeout(std::time::Duration::from_millis(200));
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
    let copy = d.path().join("copy.db");
    g.snapshot(&copy).unwrap();
    drop(g);
    let g = Graph::open(&copy).unwrap();
    assert_eq!(g.read().unwrap().counts(None).unwrap(), (3, 2));
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
        t.resolve(&e, &[key.to_owned()], None, &query::Keys::default())
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
