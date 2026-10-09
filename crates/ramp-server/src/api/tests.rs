//! Endpoint tests: requests go straight to `handle`, no sockets.

use axum::http::{HeaderValue, header};
use serde_json::json;

use super::*;

struct T {
    _dir: tempfile::TempDir,
    store: Store,
}

struct Got {
    status: u16,
    headers: HeaderMap,
    body: Value,
}

fn server() -> T {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("graphs"), 256, usize::MAX).unwrap();
    T { _dir: dir, store }
}

/// Records a response. `stop_after` simulates a client that disconnects.
#[derive(Default)]
struct Buf {
    status: u16,
    headers: HeaderMap,
    chunks: Vec<Vec<u8>>,
    failed: bool,
    stop_after: Option<usize>,
}

impl Wire for Buf {
    fn start(&mut self, status: StatusCode, headers: Vec<(&'static str, String)>) {
        assert_eq!(self.status, 0, "started twice");
        self.status = status.as_u16();
        for (k, v) in headers {
            self.headers.append(
                axum::http::HeaderName::try_from(k).unwrap(),
                HeaderValue::from_str(&v).unwrap(),
            );
        }
    }

    fn write(&mut self, chunk: Vec<u8>) -> bool {
        self.chunks.push(chunk);
        self.stop_after.is_none_or(|n| self.chunks.len() < n)
    }

    fn fail(&mut self) {
        self.failed = true;
    }
}

impl Buf {
    fn body(&self) -> Vec<u8> {
        self.chunks.concat()
    }
}

fn run(store: &Store, req: &Req) -> Buf {
    let mut buf = Buf::default();
    handle(store, req, &mut buf);
    buf
}

impl T {
    fn call(
        &self,
        method: &str,
        uri: &str,
        body: Option<Value>,
        headers: &[(&'static str, &str)],
    ) -> Got {
        let (path, query) = uri.split_once('?').unwrap_or((uri, ""));
        let mut h = HeaderMap::new();
        for (k, v) in headers {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        let body = body.map_or_else(Bytes::new, |b| {
            h.entry("content-type")
                .or_insert(HeaderValue::from_static("application/json"));
            Bytes::from(serde_json::to_vec(&b).unwrap())
        });
        let req = Req {
            method: method.to_owned(),
            path: path.to_owned(),
            query: query.to_owned(),
            headers: h,
            body,
        };
        let res = run(&self.store, &req);
        let (status, bytes) = (res.status, res.body());
        let headers = res.headers;
        let body = if bytes.is_empty() {
            Value::Null
        } else if headers
            .get("content-type")
            .is_some_and(|c| c == "application/x-msgpack")
        {
            value::decode(&bytes)
                .unwrap_or_else(|_| Value::String(format!("{} msgpack bytes", bytes.len())))
        } else {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| Value::String(format!("{} raw bytes", bytes.len())))
        };
        Got {
            status,
            headers,
            body,
        }
    }

    fn get(&self, uri: &str) -> Got {
        self.call("GET", uri, None, &[])
    }

    fn post(&self, uri: &str, body: Value) -> Got {
        self.call("POST", uri, Some(body), &[])
    }

    /// Creates a graph from `body`, returning its UUID.
    fn create(&self, body: Value) -> String {
        let r = self.post("/graph", body);
        assert_eq!(r.status, 201, "{}", r.body);
        r.body["uuid"].as_str().unwrap().to_owned()
    }
}

#[test]
fn create_query_and_dump() {
    let s = server();
    let u = s.create(json!({
        "seed": true,
        "meta": {"name": "test", "tags": ["b"]},
        "chains": [[
            {"type": "foo", "value": "bar", "color": "red"},
            {"type": "e1", "value": ""},
            {"type": "foo", "value": "baz"},
            {"type": "e2"},
            {"type": "goo", "value": "gaz", "depth": 99},
        ]],
    }));
    assert!(is_uuid(&u));

    let dump = s.get(&format!("/graph/{u}"));
    assert_eq!(dump.status, 200);
    assert_eq!(dump.body["meta"], json!({"name": "test", "tags": ["b"]}));
    assert_eq!(dump.body["nodes"].as_array().unwrap().len(), 3);
    let edges = dump.body["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[0]["src"]["value"], "bar", "format_edge shape");
    assert!(edges[0].get("srcID").is_none());
    let max: u64 = dump.headers["X-lg-maxID"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(dump.body["maxID"], max);

    // Every node is a seed: depth 0 everywhere, and the client's depth=99 is ignored.
    let q = s.get(&format!("/graph/{u}?q=n(depth=0)&q=n(seed)&q=n(depth=0)"));
    let rows = q.body.as_array().unwrap();
    assert_eq!(
        rows[0],
        json!(["n(depth=0)", "n(seed)"]),
        "sorted unique queries first"
    );
    assert_eq!(rows[1..].iter().filter(|r| r[0] == 0).count(), 3);
    assert_eq!(rows[1..].iter().filter(|r| r[0] == 1).count(), 3);

    let chains = s.get(&format!("/graph/{u}?q=n(color='red')->e()->n()"));
    assert_eq!(chains.body[1][0], 0);
    assert_eq!(chains.body[1][1][2]["value"], "baz");

    assert_eq!(s.get(&format!("/graph/{u}?q=n(")).status, 400);
    assert_eq!(
        s.get(&format!("/graph/{u}?q=n()&limit=2"))
            .body
            .as_array()
            .unwrap()
            .len(),
        3,
        "header row + 2"
    );
}

#[test]
fn depth_and_cost() {
    let s = server();
    let u = s.create(json!({"seed": true, "nodes": [{"type": "t", "value": "root"}]}));
    let r = s.post(
        &format!("/graph/{u}"),
        json!({"chains": [[
            {"type": "t", "value": "root"}, {"type": "e"}, {"type": "t", "value": "a"}, {"type": "e"}, {"type": "t", "value": "b"}
        ]]}),
    );
    assert_eq!(r.status, 204);
    assert!(
        !r.headers.contains_key("content-length"),
        "204 has no length"
    );
    assert!(
        r.headers["X-lg-updates"]
            .to_str()
            .unwrap()
            .parse::<u64>()
            .unwrap()
            > 0
    );
    let depth =
        |v: &str| s.get(&format!("/graph/{u}?q=n(value='{v}')")).body[1][1][0]["depth"].clone();
    assert_eq!(
        (depth("root"), depth("a"), depth("b")),
        (json!(0), json!(1), json!(2))
    );

    // A cheaper edge pulls depth down and cascades; a higher cost is dropped.
    let first = s.get(&format!("/graph/{u}?q=e()")).body[1][1][0]["ID"]
        .as_u64()
        .unwrap();
    assert_eq!(
        s.call(
            "PUT",
            &format!("/graph/{u}/edge/{first}"),
            Some(json!({"cost": 0.5})),
            &[]
        )
        .status,
        204
    );
    assert_eq!((depth("a"), depth("b")), (json!(0.5), json!(1.5)));
    s.call(
        "PUT",
        &format!("/graph/{u}/edge/{first}"),
        Some(json!({"cost": 0.9})),
        &[],
    );
    let e = s.get(&format!("/graph/{u}/edge/{first}")).body;
    assert_eq!(e["cost"], 0.5);
    assert_eq!(e["src"]["value"], "root");
    assert_eq!(
        s.get(&format!("/graph/{u}/node/{first}")).status,
        404,
        "an edge is not a node"
    );
}

#[test]
fn streaming_with_start() {
    let s = server();
    let u = s.create(json!({"nodes": [{"type": "t", "value": "a"}]}));
    let after = s.get(&format!("/graph/{u}")).body["maxID"]
        .as_u64()
        .unwrap()
        + 1;
    s.post(
        &format!("/graph/{u}"),
        json!({"nodes": [{"type": "t", "value": "b", "x": 1}]}),
    );
    s.post(
        &format!("/graph/{u}"),
        json!({"nodes": [{"type": "t", "value": "b", "x": 2}]}),
    );
    let rows = s.get(&format!("/graph/{u}?q=n()&start={after}")).body;
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 2, "only b is new: {rows:?}");
    assert!(
        rows[1][1][0].get("x").is_none(),
        "frozen at the entry that matched: x came later"
    );
    assert_eq!(s.get(&format!("/graph/{u}?q=n()&start=0")).status, 400);
}

#[test]
fn listing_meta_status_and_permissions() {
    let s = server();
    let open = s.create(json!({"meta": {"name": "open"}}));
    let locked = s.create(json!({"meta": {"name": "locked", "roles": {"amy": "reader"}}}));

    let names = |uri: &str| {
        let mut v: Vec<String> = s
            .get(uri)
            .body
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["meta"]["name"].as_str().unwrap().to_owned())
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        names("/graph"),
        ["open"],
        "role-protected graphs need a user"
    );
    assert_eq!(names("/graph?user=amy"), ["locked", "open"]);
    assert_eq!(names("/graph?user=amy&filter=meta.name~/lock/"), ["locked"]);

    assert_eq!(s.get(&format!("/graph/{locked}")).status, 403);
    assert_eq!(
        s.get(&format!("/graph/{locked}/status")).status,
        403,
        "upstream skipped this check"
    );
    assert_eq!(
        s.get(&format!("/graph/{locked}?user=amy&role=writer"))
            .status,
        403
    );
    assert_eq!(s.get(&format!("/graph/{locked}?user=amy")).status, 200);

    let st = s.get(&format!("/graph/{open}/status")).body;
    assert_eq!(st["id"], open.as_str());
    assert!(st["created"].as_str().unwrap().ends_with('Z'));

    assert_eq!(
        s.call(
            "PUT",
            &format!("/graph/{open}/meta"),
            Some(json!({"tags": ["x"]})),
            &[]
        )
        .status,
        204
    );
    assert_eq!(
        s.get(&format!("/graph/{open}/meta")).body,
        json!({"name": "open", "tags": ["x"]})
    );

    assert_eq!(s.get("/graph?q=n()&user=amy").body[0], json!(["n()"]));
}

#[test]
fn kv_seeds_reset_delete() {
    let s = server();
    let u = s.create(json!({"seed": true, "nodes": [{"type": "t", "value": "a"}]}));
    s.post(
        &format!("/graph/{u}"),
        json!({"nodes": [{"type": "t", "value": "later"}]}),
    );
    s.post(&format!("/kv/{u}"), json!({"k": {"a": 1}}));
    s.post(&format!("/kv/{u}"), json!({"k": {"b": 2}}));
    assert_eq!(s.get(&format!("/kv/{u}/k")).body, json!({"a": 1, "b": 2}));
    assert_eq!(s.get(&format!("/kv/{u}/nope")).status, 404, "upstream: 500");
    assert_eq!(
        s.call("DELETE", &format!("/kv/{u}/nope"), None, &[]).status,
        404
    );

    assert_eq!(
        s.get(&format!("/graph/{u}/seeds"))
            .body
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(s.call("PUT", &format!("/reset/{u}"), None, &[]).status, 204);
    let dump = s.get(&format!("/graph/{u}")).body;
    assert_eq!(
        dump["nodes"].as_array().unwrap().len(),
        1,
        "only the seed is replayed"
    );
    assert_eq!(dump["nodes"][0]["depth"], 0);
    assert_eq!(
        s.get(&format!("/kv/{u}")).body,
        json!({"k": {"a": 1, "b": 2}})
    );

    assert_eq!(
        s.call("DELETE", &format!("/graph/{u}"), None, &[]).status,
        204
    );
    assert_eq!(s.get(&format!("/graph/{u}")).status, 404);
}

#[test]
fn formats_and_errors() {
    let s = server();
    let u = s.create(json!({"nodes": [{"type": "t", "value": "a"}]}));
    let mp = s.call(
        "GET",
        &format!("/graph/{u}?q=n()"),
        None,
        &[("accept", "application/x-msgpack")],
    );
    assert_eq!(mp.headers["content-type"], "application/x-msgpack");
    assert_eq!(
        s.call(
            "GET",
            &format!("/graph/{u}"),
            None,
            &[("accept", "application/x-msgpack")]
        )
        .status,
        406
    );

    let charset = s.call(
        "POST",
        &format!("/graph/{u}"),
        Some(json!({})),
        &[("content-type", "application/json; charset=utf-8")],
    );
    assert_eq!(
        charset.status, 204,
        "media type parameters are accepted (upstream: 409)"
    );
    let bad = s.call(
        "POST",
        &format!("/graph/{u}"),
        Some(json!({})),
        &[("content-type", "text/plain")],
    );
    assert_eq!(bad.status, 409);
    assert_eq!(bad.body["code"], 409);
    assert_eq!(
        s.post(&format!("/graph/{u}"), json!({"nodes": [{"type": "t"}]}))
            .status,
        409,
        "missing value (upstream: 500)"
    );
    assert_eq!(
        s.post(&format!("/graph/{u}"), json!({"chains": [[{}, {}]]}))
            .status,
        409
    );

    let other = "0b9e2f7c-5a1d-11ef-8d3e-0242ac120002";
    assert_eq!(
        s.call("POST", &format!("/graph/{other}"), None, &[]).status,
        404
    );
    assert_eq!(
        s.call(
            "POST",
            &format!("/graph/{other}?create"),
            Some(json!({"nodes": []})),
            &[]
        )
        .status,
        201
    );

    assert_eq!(s.call("PATCH", "/graph", None, &[]).status, 405);
    assert_eq!(s.get("/graph/not-a-uuid").status, 404);
    assert_eq!(s.get("/view/x").status, 404, "UI and /exec are gone");
    assert_eq!(
        s.get(&format!("/d3/{u}")).body["nodes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn upload_roundtrip() {
    let s = server();
    let u = s.create(json!({"nodes": [{"type": "t", "value": "a"}]}));
    let req = |method: &str, path: String, headers: HeaderMap, body: Vec<u8>| Req {
        method: method.to_owned(),
        path,
        query: String::new(),
        headers,
        body: Bytes::from(body),
    };
    let mut accept = HeaderMap::new();
    accept.insert(
        header::ACCEPT,
        HeaderValue::from_static("application/octet-stream"),
    );
    let snap = run(
        &s.store,
        &req("GET", format!("/graph/{u}"), accept, Vec::new()),
    )
    .body();
    let target = "0b9e2f7c-5a1d-11ef-8d3e-0242ac120003";
    let put = |body: Vec<u8>| {
        run(
            &s.store,
            &req("PUT", format!("/graph/{target}"), HeaderMap::new(), body),
        )
        .status
    };
    assert_eq!(put(b"garbage".to_vec()), 409);
    assert_eq!(put(snap.clone()), 204);
    assert_eq!(put(snap), 409, "exists");
    assert_eq!(
        s.get(&format!("/graph/{target}")).body["nodes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn reopen_reindexes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graphs");
    let u = {
        let s = T {
            _dir: tempfile::tempdir().unwrap(),
            store: Store::open(&path, 256, usize::MAX).unwrap(),
        };
        s.create(json!({"meta": {"name": "kept"}}))
    };
    let s = T {
        _dir: dir,
        store: Store::open(&path, 256, usize::MAX).unwrap(),
    };
    assert_eq!(
        s.get(&format!("/graph/{u}/status")).body["meta"]["name"],
        "kept"
    );
}

#[test]
fn large_responses_stream() {
    let s = server();
    let nodes: Vec<Value> = (0..20_000)
        .map(|i| json!({"type": "t", "value": i.to_string(), "pad": "x".repeat(64)}))
        .collect();
    let u = s.create(json!({"nodes": nodes}));
    let req = Req {
        method: "GET".to_owned(),
        path: format!("/graph/{u}"),
        query: String::new(),
        headers: HeaderMap::new(),
        body: Bytes::new(),
    };
    let res = run(&s.store, &req);
    assert_eq!(res.status, 200);
    assert!(
        res.chunks.len() > 2,
        "streamed in chunks: {}",
        res.chunks.len()
    );
    assert!(
        res.headers.get("content-length").is_none(),
        "length unknown when streaming"
    );
    assert!(res.headers.contains_key("x-lg-maxid"));
    let dump: Value = serde_json::from_slice(&res.body()).unwrap();
    assert_eq!(dump["nodes"].as_array().unwrap().len(), 20_000);

    let q = Req {
        query: "q=n()".to_owned(),
        ..req
    };
    let rows = run(&s.store, &q);
    let rows: Value = serde_json::from_slice(&rows.body()).unwrap();
    assert_eq!(
        rows.as_array().unwrap().len(),
        20_001,
        "header row + every chain"
    );

    // A client that goes away mid-stream stops the handler and cuts the connection.
    let mut gone = Buf {
        stop_after: Some(2),
        ..Buf::default()
    };
    handle(&s.store, &q, &mut gone);
    assert_eq!(gone.chunks.len(), 2);
    assert!(gone.failed);

    // Small responses still carry a length.
    let small = run(
        &s.store,
        &Req {
            query: "q=n(value='7')".to_owned(),
            ..q
        },
    );
    assert_eq!(small.chunks.len(), 1);
    assert!(small.headers.contains_key("content-length"));
}

#[test]
fn dates() {
    assert_eq!(parse_date("1970-01-01"), Some(0));
    assert_eq!(parse_date("2024-02-29T12:30:15.5Z"), Some(1_709_209_815));
    assert_eq!(parse_date("2024-13-01"), None);
    assert_eq!(civil(19_782), (2024, 2, 29));
    assert_eq!(unescape("a%20b+c%2"), "a b c%2");
}
