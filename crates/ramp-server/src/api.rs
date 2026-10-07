//! HTTP surface, wire-compatible with upstream's `RESTAPI`.
//!
//! `/exec`, `/view` and `/static` are gone. Everything here is synchronous; `main` runs
//! it on a blocking thread because LMDB transactions belong to the thread that opened them.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};

use axum::body::Bytes;
use axum::http::{HeaderMap, StatusCode};
use ramp_graph::lgql::{ParseError, Pattern};
use ramp_graph::value::{self, Value};
use ramp_graph::{Direction, Entry, Graph, GraphError, LogId, Record, Txn};
use serde_json::{Map, json};

use crate::input::{self, Input, SEEDS, as_dict, format_edge, props_dict, run_adapters};
use crate::store::{Creds, Open, Status, Store, allowed, is_uuid, remove_lmdb};

/// Kv domain behind `/kv/<uuid>`.
const RESTOBJS: &[u8] = b"lg.restobjs";

/// Bodies up to this size are buffered and sent with `Content-Length`; errors before it
/// still get a proper status. Larger ones stream (as upstream's `-b` buffer did).
const BUFFER: usize = 1 << 20;
/// Chunk size once streaming.
const CHUNK: usize = 64 << 10;

/// A response head: status and headers.
pub(crate) type Head = (StatusCode, Vec<(&'static str, String)>);

/// Where a response goes: `start` once, then body chunks.
pub(crate) trait Wire {
    fn start(&mut self, status: StatusCode, headers: Vec<(&'static str, String)>);
    /// Returns `false` once the client has gone away.
    fn write(&mut self, chunk: Vec<u8>) -> bool;
    /// Abandons a response whose headers were already sent (the connection is cut, so
    /// the client cannot mistake a truncated body for a complete one).
    fn fail(&mut self);
}

/// A request with its body already read.
#[derive(Debug)]
pub(crate) struct Req {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) query: String,
    pub(crate) headers: HeaderMap,
    pub(crate) body: Bytes,
}

/// An error response: `{"code", "reason", "message"}` as JSON, whatever was accepted.
#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    msg: String,
    allow: Option<&'static str>,
}

impl ApiError {
    pub(crate) fn new(code: u16, msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            msg: msg.into(),
            allow: None,
        }
    }

    pub(crate) fn internal(msg: impl Into<String>) -> Self {
        Self::new(500, msg)
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "used as `map_err(ApiError::io)`"
    )]
    pub(crate) fn io(e: std::io::Error) -> Self {
        let code = if e.kind() == std::io::ErrorKind::StorageFull {
            507
        } else {
            500
        };
        Self::new(code, e.to_string())
    }

    fn not_allowed(allow: &'static str) -> Self {
        Self {
            allow: Some(allow),
            ..Self::new(405, "method not allowed")
        }
    }

    fn send(&self, wire: &mut dyn Wire) {
        let body = json!({"code": self.status.as_u16(), "reason": self.status.canonical_reason(), "message": self.msg});
        let mut headers = vec![("Content-Type", "application/json".to_owned())];
        if let Some(a) = self.allow {
            headers.push(("Allow", a.to_owned()));
        }
        wire.start(self.status, headers);
        wire.write(json_line(&body));
    }
}

impl From<GraphError> for ApiError {
    fn from(e: GraphError) -> Self {
        match e {
            GraphError::NotFound(..) => Self::new(404, e.to_string()),
            GraphError::Lmdb(_)
            | GraphError::Io(_)
            | GraphError::Corrupt(_)
            | GraphError::Value(_)
            | GraphError::ReadOnly
            | _ => Self::internal(e.to_string()),
        }
    }
}

impl From<ParseError> for ApiError {
    fn from(e: ParseError) -> Self {
        Self::new(400, format!("query syntax error: {e}"))
    }
}

type Res<T> = Result<T, ApiError>;

/// A successful response before encoding.
#[derive(Debug)]
struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Out,
}

#[derive(Debug)]
enum Out {
    Empty,
    One(Value),
    Raw(Vec<u8>, &'static str),
    /// Already written to the wire by a [`Streamer`].
    Sent,
}

impl Reply {
    const fn new(body: Out) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body,
        }
    }

    fn header(mut self, k: &'static str, v: impl std::fmt::Display) -> Self {
        self.headers.push((k, format!("{v}")));
        self
    }
}

fn json_line(v: &Value) -> Vec<u8> {
    let mut out = serde_json::to_vec(v).unwrap_or_default();
    out.push(b'\n');
    out
}

/// Handles one request, writing the response to `wire`.
pub(crate) fn handle(store: &Store, req: &Req, wire: &mut dyn Wire) {
    let ctx = Ctx {
        store,
        req,
        params: parse_query(&req.query),
        wire: RefCell::new(wire),
        started: Cell::new(false),
        extra: RefCell::default(),
    };
    match ctx.route() {
        Ok(reply) => ctx.render(reply),
        Err(_) if ctx.started.get() => ctx.wire.borrow_mut().fail(),
        Err(e) => e.send(*ctx.wire.borrow_mut()),
    }
}

/// Writes a body incrementally: buffered up to [`BUFFER`], then streamed in [`CHUNK`]s.
struct Streamer<'c, 'r> {
    ctx: &'c Ctx<'r>,
    /// Status and headers, until they are sent.
    pending: Option<(StatusCode, Vec<(&'static str, String)>)>,
    ctype: &'static str,
    buf: Vec<u8>,
    msgpack: bool,
    items: usize,
}

impl Streamer<'_, '_> {
    /// # Errors
    /// The client went away.
    fn raw(&mut self, bytes: &[u8]) -> Res<()> {
        self.buf.extend_from_slice(bytes);
        if self.buf.len()
            >= if self.pending.is_some() {
                BUFFER
            } else {
                CHUNK
            }
        {
            self.flush()?;
        }
        Ok(())
    }

    /// # Errors
    /// The client went away.
    fn flush(&mut self) -> Res<()> {
        let mut wire = self.ctx.wire.borrow_mut();
        if let Some((status, mut headers)) = self.pending.take() {
            headers.push(("Content-Type", self.ctype.to_owned()));
            wire.start(status, headers);
            self.ctx.started.set(true);
        }
        if wire.write(std::mem::take(&mut self.buf)) {
            Ok(())
        } else {
            Err(ApiError::new(499, "client went away"))
        }
    }

    /// One element of a JSON array, or one concatenated msgpack value.
    ///
    /// # Errors
    /// The client went away.
    fn item(&mut self, v: &Value) -> Res<()> {
        if self.msgpack {
            return self.raw(&value::encode(v)?);
        }
        self.raw(if self.items == 0 { b"[" } else { b"," })?;
        self.items += 1;
        self.raw(&serde_json::to_vec(v).unwrap_or_default())
    }

    /// Closes an item stream.
    ///
    /// # Errors
    /// The client went away.
    fn end_items(mut self) -> Res<Reply> {
        if !self.msgpack {
            self.raw(if self.items == 0 { b"[]\n" } else { b"]\n" })?;
        }
        self.finish()
    }

    /// Sends what is left. A body that never outgrew the buffer goes out as a plain reply.
    ///
    /// # Errors
    /// The client went away.
    fn finish(mut self) -> Res<Reply> {
        if let Some((status, headers)) = self.pending.take() {
            return Ok(Reply {
                status: status.as_u16(),
                headers,
                body: Out::Raw(std::mem::take(&mut self.buf), self.ctype),
            });
        }
        self.flush()?;
        Ok(Reply::new(Out::Sent))
    }
}

/// `a=1&b=x%20y&b` → `[("a","1"), ("b","x y"), ("b","")]`.
fn parse_query(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (unescape(k), unescape(v))
        })
        .collect()
}

fn unescape(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let mut bytes = s.bytes();
    while let Some(b) = bytes.next() {
        match b {
            b'+' => out.push(b' '),
            b'%' => {
                let hex: Vec<u8> = bytes.clone().take(2).collect();
                match std::str::from_utf8(&hex)
                    .ok()
                    .and_then(|h| u8::from_str_radix(h, 16).ok())
                    .filter(|_| hex.len() == 2)
                {
                    Some(v) => {
                        out.push(v);
                        bytes.nth(1);
                    }
                    None => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

struct Ctx<'r> {
    store: &'r Store,
    req: &'r Req,
    params: Vec<(String, String)>,
    wire: RefCell<&'r mut dyn Wire>,
    /// Headers have gone out; errors can no longer change the status.
    started: Cell<bool>,
    /// Headers every response from here on carries (`X-lg-maxID` from `read`).
    extra: RefCell<Vec<(&'static str, String)>>,
}

impl<'r> Ctx<'r> {
    fn param(&self, name: &str) -> Option<&str> {
        self.params
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn params(&self, name: &str) -> Vec<&str> {
        self.params
            .iter()
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    fn header(&self, name: &str) -> &str {
        self.req
            .headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
    }

    fn msgpack(&self) -> bool {
        self.header("accept")
            .to_ascii_lowercase()
            .contains("application/x-msgpack")
    }

    fn creds(&self) -> Creds {
        Creds {
            user: self.param("user").map(str::to_owned),
            roles: self.params("role").into_iter().map(str::to_owned).collect(),
        }
    }

    /// A positive integer parameter (`start`, `stop`).
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn pos(&self, name: &str) -> Res<Option<u64>> {
        self.param(name)
            .map(|s| {
                s.parse::<u64>().ok().filter(|&n| n > 0).ok_or_else(|| {
                    ApiError::new(
                        400,
                        format!("Bad {name} parameter - if present, must be > 0"),
                    )
                })
            })
            .transpose()
    }

    /// The decoded body, if any (`application/json` or `application/x-msgpack`).
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn input(&self) -> Res<Option<Value>> {
        if self.req.body.is_empty() {
            return Ok(None);
        }
        let ctype = self.header("content-type");
        let media = ctype
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        match media.as_str() {
            "application/json" => serde_json::from_slice(&self.req.body)
                .map(Some)
                .map_err(|e| {
                    ApiError::new(400, format!("Decode failed for mime type: {media}: {e}"))
                }),
            "application/x-msgpack" => value::decode(&self.req.body).map(Some).map_err(|e| {
                ApiError::new(400, format!("Decode failed for mime type: {media}: {e}"))
            }),
            _ => Err(ApiError::new(
                409,
                format!(
                    "Bad/missing mime type ({ctype}) - use one of: application/json, application/x-msgpack"
                ),
            )),
        }
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn body_object(&self) -> Res<Map<String, Value>> {
        match self.input()? {
            Some(Value::Object(m)) => Ok(m),
            _ => Err(ApiError::new(400, "body must be an object")),
        }
    }

    fn render(&self, r: Reply) {
        let mp = self.msgpack();
        let (ctype, body) = match r.body {
            Out::Sent => return,
            Out::Empty => ("application/json", Vec::new()),
            Out::Raw(b, ctype) => (ctype, b),
            Out::One(v) if mp => (
                "application/x-msgpack",
                value::encode(&v).unwrap_or_default(),
            ),
            Out::One(v) => ("application/json", json_line(&v)),
        };
        let status = if body.is_empty() && r.status == 200 && self.req.method != "HEAD" {
            204
        } else {
            r.status
        };
        let mut headers = self.extra.take();
        headers.extend(r.headers);
        headers.push(("Content-Type", ctype.to_owned()));
        if status != 204 {
            // RFC 9110: a 204 carries no Content-Length.
            headers.push(("Content-Length", body.len().to_string()));
        }
        let mut wire = self.wire.borrow_mut();
        wire.start(
            StatusCode::from_u16(status).unwrap_or(StatusCode::OK),
            headers,
        );
        wire.write(body);
    }

    /// A streaming body for a 200 response.
    fn streamer(&self, ctype: &'static str, msgpack: bool) -> Streamer<'_, 'r> {
        Streamer {
            ctx: self,
            pending: Some((StatusCode::OK, self.extra.take())),
            ctype,
            buf: Vec::new(),
            msgpack,
            items: 0,
        }
    }

    /// Items as a JSON array, or concatenated msgpack values if accepted.
    fn items(&self) -> Streamer<'_, 'r> {
        let mp = self.msgpack();
        self.streamer(
            if mp {
                "application/x-msgpack"
            } else {
                "application/json"
            },
            mp,
        )
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn route(&self) -> Res<Reply> {
        let segs: Vec<&str> = self.req.path.trim_matches('/').split('/').collect();
        let m = self.req.method.as_str();
        match (segs.as_slice(), m) {
            (["graph"], "GET") => self.list(),
            (["graph"], "POST") => self.create(&new_uuid()),
            (["graph"], _) => Err(ApiError::not_allowed("GET, POST")),
            (["graph", u], "GET") if is_uuid(u) => self.get_graph(u),
            (["graph", u], "HEAD") if is_uuid(u) => self.read(u, |_, _| Ok(Reply::new(Out::Empty))),
            (["graph", u], "POST") if is_uuid(u) => self.post_graph(u),
            (["graph", u], "PUT") if is_uuid(u) => self.upload(u),
            (["graph", u], "DELETE") if is_uuid(u) => {
                self.open(u)?;
                self.store.delete(u)?;
                Ok(Reply::new(Out::Empty))
            }
            (["graph", u], _) if is_uuid(u) => {
                Err(ApiError::not_allowed("GET, HEAD, POST, PUT, DELETE"))
            }
            (["graph", u, "meta"], "GET") if is_uuid(u) => self.read(u, |t, _| {
                Ok(Reply::new(Out::One(props_dict(t, 0, None)?.into())))
            }),
            (["graph", u, "meta"], "PUT") if is_uuid(u) => {
                let meta = Value::Object(self.body_object()?);
                self.write(u, |t| {
                    Input {
                        t,
                        create: false,
                        seed: false,
                    }
                    .meta(&meta)
                })
            }
            (["graph", u, "meta"], _) if is_uuid(u) => Err(ApiError::not_allowed("GET, PUT")),
            (["graph", u, "status"], "GET" | "HEAD") if is_uuid(u) => {
                let status = self.status(u)?;
                Ok(Reply::new(if m == "HEAD" {
                    Out::Empty
                } else {
                    Out::One(enrich(u, &status))
                }))
            }
            (["graph", u, "status"], _) if is_uuid(u) => Err(ApiError::not_allowed("GET, HEAD")),
            (["graph", u, "seeds"], "GET") if is_uuid(u) => self.read(u, |t, _| {
                let mut out = self.items();
                for kv in t.kv_iter(SEEDS, b"")? {
                    out.item(&value::decode(kv?.1)?)?;
                }
                out.end_items()
            }),
            (["graph", u, kind @ ("node" | "edge"), n], _) if is_uuid(u) => {
                self.object(u, kind, n, m)
            }
            (["kv", u], _) if is_uuid(u) => self.kv_all(u, m),
            (["kv", u, key], _) if is_uuid(u) && !key.is_empty() => {
                self.kv_key(u, m, &unescape(key))
            }
            (["reset", u], "PUT") if is_uuid(u) => self.reset(u),
            (["reset", u], _) if is_uuid(u) => Err(ApiError::not_allowed("PUT")),
            (["d3", u], "GET") if is_uuid(u) => self.d3(u),
            (["d3", u], _) if is_uuid(u) => Err(ApiError::not_allowed("GET")),
            _ => Err(ApiError::new(
                404,
                format!("no handler for {}", self.req.path),
            )),
        }
    }

    /// `/graph/<uuid>/node/<id>` and `/graph/<uuid>/edge/<id>`.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn object(&self, u: &str, kind: &str, n: &str, m: &str) -> Res<Reply> {
        let n = n
            .parse::<LogId>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| ApiError::new(404, "bad ID"))?;
        match m {
            "GET" => self.read(u, |t, _| {
                let e = t.entry(n)?.filter(|e| e.next == 0);
                let found = match (e.map(|e| e.record), kind) {
                    (Some(Record::Node { .. }), "node") => e.map(|e| as_dict(t, &e, None)),
                    (Some(Record::Edge { .. }), "edge") => e.map(|e| format_edge(t, &e, None)),
                    _ => None,
                };
                Ok(Reply::new(Out::One(found.ok_or_else(|| {
                    ApiError::new(404, format!("not a {kind}"))
                })??)))
            }),
            "PUT" => {
                let mut data = self.body_object()?;
                data.insert("ID".to_owned(), n.into());
                let data = Value::Object(data);
                self.write(u, |t| {
                    let mut input = Input {
                        t,
                        create: false,
                        seed: false,
                    };
                    if kind == "node" {
                        input.node(&data)
                    } else {
                        input.edge(&data, None, None)
                    }
                    .map(drop)
                })
            }
            _ => Err(ApiError::not_allowed("GET, PUT")),
        }
    }

    // ── Access ──

    /// # Errors
    /// The HTTP error to send instead.
    fn status(&self, uuid: &str) -> Res<Status> {
        let status = self
            .store
            .status(uuid)?
            .ok_or_else(|| ApiError::new(404, format!("no graph {uuid}")))?;
        if !allowed(&status.meta, &self.creds()) {
            return Err(ApiError::new(403, format!("Permission denied: {uuid}")));
        }
        Ok(status)
    }

    /// An existing graph the caller may use.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn open(&self, uuid: &str) -> Res<std::sync::Arc<Graph>> {
        let g = self.store.graph(uuid, Open::Existing)?;
        self.status(uuid)?;
        Ok(g)
    }

    /// Runs `f` in a read txn, adding `X-lg-maxID`.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn read(&self, uuid: &str, f: impl FnOnce(&Txn<'_>, &Graph) -> Res<Reply>) -> Res<Reply> {
        let g = self.open(uuid)?;
        let t = g.read()?;
        let max = t.next_id()?.saturating_sub(1);
        self.extra
            .borrow_mut()
            .push(("X-lg-maxID", max.to_string()));
        f(&t, &g)
    }

    /// Runs `f` in a write txn, then the adapters; commits, re-indexes, adds `X-lg-updates`/`X-lg-maxID`.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn write(&self, uuid: &str, f: impl FnOnce(&mut Txn<'_>) -> Res<()>) -> Res<Reply> {
        let g = self.open(uuid)?;
        self.write_to(uuid, &g, f)
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn write_to(
        &self,
        uuid: &str,
        g: &Graph,
        f: impl FnOnce(&mut Txn<'_>) -> Res<()>,
    ) -> Res<Reply> {
        let mut t = g.write()?;
        let start = t.next_id()?;
        f(&mut t)?;
        run_adapters(&mut t, start)?;
        let end = t.next_id()?;
        t.commit()?;
        if self.req.headers.contains_key("x-lg-sync") {
            g.sync()?;
        }
        self.store.refresh(uuid, g)?;
        Ok(Reply::new(Out::Empty)
            .header("X-lg-updates", end.saturating_sub(start))
            .header("X-lg-maxID", end.saturating_sub(1)))
    }

    // ── /graph ──

    /// # Errors
    /// The HTTP error to send instead.
    fn list(&self) -> Res<Reply> {
        let creds = self.creds();
        let enabled = self.param("enabled").is_some();
        let date = |name| {
            self.param(name)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    parse_date(s).ok_or_else(|| {
                        ApiError::new(400, format!("Bad datetime string for {name}: {s}"))
                    })
                })
                .transpose()
        };
        let (after, before) = (date("created_after")?, date("created_before")?);
        let filters = self
            .params("filter")
            .into_iter()
            .map(|f| Pattern::parse(&format!("n({f})")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut graphs = Vec::new();
        for (uuid, status) in self.store.list()? {
            let created = created_secs(&uuid);
            if !allowed(&status.meta, &creds)
                || enabled && !status.meta.get("enabled").is_none_or(input::truthy)
                || after.is_some_and(|a| created.is_none_or(|c| c <= a))
                || before.is_some_and(|b| created.is_none_or(|c| c >= b))
            {
                continue;
            }
            let info = enrich(&uuid, &status);
            if filters.is_empty() || filters.iter().any(|p| p.matches_value(&info)) {
                graphs.push((uuid, info));
            }
        }
        let (uniq, patterns) = self.queries()?;
        let mut out = self.items();
        if patterns.is_empty() {
            for (_, info) in &graphs {
                out.item(info)?;
            }
            return out.end_items();
        }
        out.item(&uniq)?;
        for (uuid, _) in graphs {
            let Ok(g) = self.store.graph(&uuid, Open::Existing) else {
                continue;
            };
            let t = g.read()?;
            stream_chains(
                &t,
                &patterns,
                &mut out,
                0,
                |qi, chain| Ok(json!([uuid, qi, chain])),
                |sink| t.query(&patterns, None, |qi, c| sink(qi, None, c)),
            )?;
        }
        out.end_items()
    }

    /// Sorted unique `q` parameters, and their compiled patterns in that order.
    ///
    /// # Errors
    /// The HTTP error to send instead.
    fn queries(&self) -> Res<(Value, Vec<Pattern>)> {
        let mut qs: Vec<&str> = self
            .params("q")
            .into_iter()
            .filter(|q| !q.is_empty())
            .collect();
        qs.sort_unstable();
        qs.dedup();
        let patterns = qs
            .iter()
            .map(|q| Pattern::parse(q))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((Value::from(qs), patterns))
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn create(&self, uuid: &str) -> Res<Reply> {
        let data = self.input()?;
        let g = self.store.graph(uuid, Open::New)?;
        let made = self.write_to(uuid, &g, |t| {
            data.as_ref().map_or(Ok(()), |d| Input::apply(t, d, true))
        });
        drop(g);
        match made {
            Err(e) => {
                self.store.delete(uuid)?;
                Err(e)
            }
            Ok(mut r) => {
                r.status = 201;
                r.body = Out::One(json!({"uuid": uuid, "id": uuid}));
                Ok(r.header("Location", format!("/graph/{uuid}")))
            }
        }
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn post_graph(&self, uuid: &str) -> Res<Reply> {
        if self.store.status(uuid)?.is_none() {
            return if self.param("create").is_some() {
                self.create(uuid)
            } else {
                Err(ApiError::new(404, format!("no graph {uuid}")))
            };
        }
        let data = self.input()?;
        self.write(uuid, |t| {
            data.as_ref().map_or(Ok(()), |d| Input::apply(t, d, false))
        })
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn upload(&self, uuid: &str) -> Res<Reply> {
        if self.req.body.is_empty() {
            return Err(ApiError::new(400, "empty upload"));
        }
        let tmp = self
            .store
            .dir()
            .join(format!("tmp_{uuid}_{}.db", new_uuid()));
        let adopted = (|| {
            std::fs::write(&tmp, &self.req.body).map_err(ApiError::io)?;
            drop(
                Graph::open(&tmp)
                    .map_err(|e| ApiError::new(409, format!("Upload for {uuid} failed: {e}")))?,
            );
            self.store.adopt(uuid, &tmp)
        })();
        if adopted.is_err() {
            remove_lmdb(&tmp)?;
        }
        adopted.map(|()| Reply::new(Out::Empty))
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn get_graph(&self, uuid: &str) -> Res<Reply> {
        if self.header("accept").contains("application/octet-stream") {
            return self.snapshot(uuid);
        }
        let (start, stop) = (self.pos("start")?, self.pos("stop")?);
        let limit = match self.param("limit") {
            Some(s) => s
                .parse::<usize>()
                .map_err(|_| ApiError::new(400, "Bad limit parameter"))?,
            None => 0,
        };
        let view = stop.map(|s| s.saturating_add(1));
        let (uniq, patterns) = self.queries()?;
        let crawl = self.param("crawl").is_some_and(|c| c != "0");
        if (patterns.is_empty() || crawl) && self.msgpack() {
            return Err(ApiError::new(
                406,
                "Format for graph dump has not been determined for non-json output",
            ));
        }
        self.read(uuid, |t, g| {
            if patterns.is_empty() {
                let nodes = t.nodes(None, view)?;
                let edges = t.edges(None, view)?;
                return dump(
                    t,
                    g,
                    uuid,
                    view,
                    nodes,
                    edges,
                    self.streamer("application/json", false),
                );
            }
            if crawl {
                let mut found = Vec::new();
                t.query(&patterns, view, |_, c| {
                    found.push(c);
                    limit == 0 || found.len() < limit
                })?;
                let (nodes, edges) = spider(t, found.into_iter(), view)?;
                let out = self.streamer("application/json", false);
                return dump(
                    t,
                    g,
                    uuid,
                    view,
                    nodes.into_iter().map(Ok),
                    edges.into_iter().map(Ok),
                    out,
                );
            }
            let mut out = self.items();
            out.item(&uniq)?;
            stream_chains(
                t,
                &patterns,
                &mut out,
                limit,
                |qi, chain| Ok(json!([qi, chain])),
                |sink| match start {
                    Some(start) => {
                        t.mquery(&patterns, start, stop, |qi, x, c| sink(qi, Some(x + 1), c))
                    }
                    None => t.query(&patterns, view, |qi, c| sink(qi, view, c)),
                },
            )?;
            out.end_items()
        })
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn snapshot(&self, uuid: &str) -> Res<Reply> {
        let g = self.open(uuid)?;
        let tmp = self
            .store
            .dir()
            .join(format!("tmp_{uuid}_{}.snap", new_uuid()));
        let bytes = g
            .snapshot(&tmp)
            .map_err(ApiError::from)
            .and_then(|()| std::fs::read(&tmp).map_err(ApiError::io));
        std::fs::remove_file(&tmp).unwrap_or_default(); // best effort
        Ok(
            Reply::new(Out::Raw(bytes?, "application/octet-stream")).header(
                "Content-Disposition",
                format!("attachment; filename=\"{uuid}.db\""),
            ),
        )
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn reset(&self, uuid: &str) -> Res<Reply> {
        let keep = self
            .input()?
            .unwrap_or_else(|| json!({"seeds": 1, "kv": true}));
        let keep_kv = keep.get("kv").is_some_and(input::truthy);
        let seeds = match keep.get("seeds") {
            Some(Value::Bool(true)) => usize::MAX,
            Some(Value::Number(n)) => n
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(0),
            _ => 0,
        };
        self.write(uuid, |t| {
            let kv: Vec<(Vec<u8>, Vec<u8>)> = if keep_kv {
                t.kv_iter(RESTOBJS, b"")?
                    .map(|kv| kv.map(|(k, v)| (k.to_vec(), v.to_vec())))
                    .collect::<Result<_, _>>()?
            } else {
                Vec::new()
            };
            let seeds = t
                .kv_iter(SEEDS, b"")?
                .take(seeds)
                .map(|kv| Ok(value::decode(kv?.1)?))
                .collect::<Res<Vec<_>>>()?;
            // In place: one txn, so a failure leaves the graph untouched. The file keeps its size.
            t.reset()?;
            for (k, v) in kv {
                t.kv_put(RESTOBJS, &k, &v)?;
            }
            for s in &seeds {
                Input::apply(t, s, false)?;
            }
            run_adapters(t, 1)
        })
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn d3(&self, uuid: &str) -> Res<Reply> {
        let view = match self.param("stop") {
            Some(s) => Some(
                s.parse::<u64>()
                    .map_err(|_| ApiError::new(400, "Bad stop parameter"))?
                    .saturating_add(1),
            ),
            None => None,
        }
        .filter(|&v| v > 1);
        self.read(uuid, |t, _| {
            let mut out = self.streamer("application/json", false);
            let mut index = HashMap::new();
            out.raw(b"{\"nodes\":[")?;
            for n in t.nodes(None, view)? {
                let n = n?;
                out.raw(if index.is_empty() { b"" } else { b"," })?;
                index.insert(n.id, index.len());
                out.raw(&json_bytes(&json!({"data": as_dict(t, &n, view)?})))?;
            }
            out.raw(b"],\"edges\":[")?;
            let mut first = true;
            for e in t.edges(None, view)? {
                let e = e?;
                if let Record::Edge { src, tgt, .. } = e.record {
                    out.raw(if first { b"" } else { b"," })?;
                    first = false;
                    let d = json!({"data": as_dict(t, &e, view)?, "source": index.get(&src), "target": index.get(&tgt)});
                    out.raw(&json_bytes(&d))?;
                }
            }
            out.raw(b"]}\n")?;
            out.finish()
        })
    }

    // ── /kv ──

    /// # Errors
    /// The HTTP error to send instead.
    fn kv_all(&self, uuid: &str, m: &str) -> Res<Reply> {
        match m {
            "GET" => self.read(uuid, |t, _| {
                let mut out = Map::new();
                for kv in t.kv_iter(RESTOBJS, b"")? {
                    let (k, v) = kv?;
                    out.insert(String::from_utf8_lossy(k).into_owned(), value::decode(v)?);
                }
                Ok(Reply::new(Out::One(out.into())))
            }),
            "POST" | "PUT" => {
                let data = self.body_object()?;
                self.write(uuid, |t| {
                    if m == "PUT" {
                        clear_kv(t)?;
                    }
                    for (k, v) in &data {
                        let old = t
                            .kv_get(RESTOBJS, k.as_bytes())?
                            .map(value::decode)
                            .transpose()?;
                        let new = if m == "POST" {
                            input::merge(old, v)
                        } else {
                            v.clone()
                        };
                        t.kv_put(RESTOBJS, k.as_bytes(), &value::encode(&new)?)?;
                    }
                    Ok(())
                })
            }
            "DELETE" => self.write(uuid, clear_kv),
            _ => Err(ApiError::not_allowed("GET, POST, PUT, DELETE")),
        }
    }

    /// # Errors
    /// The HTTP error to send instead.
    fn kv_key(&self, uuid: &str, m: &str, key: &str) -> Res<Reply> {
        let k = key.as_bytes();
        let missing = || ApiError::new(404, "key not found");
        match m {
            "GET" => self.read(uuid, |t, _| {
                Ok(Reply::new(Out::One(value::decode(
                    t.kv_get(RESTOBJS, k)?.ok_or_else(missing)?,
                )?)))
            }),
            "POST" | "PUT" => {
                let data = self.input()?.unwrap_or(Value::Null);
                self.write(uuid, |t| {
                    let old = t.kv_get(RESTOBJS, k)?.map(value::decode).transpose()?;
                    let new = if m == "POST" {
                        input::merge(old, &data)
                    } else {
                        data
                    };
                    Ok(t.kv_put(RESTOBJS, k, &value::encode(&new)?)?)
                })
            }
            "DELETE" => self.write(uuid, |t| {
                if t.kv_del(RESTOBJS, k)? {
                    Ok(())
                } else {
                    Err(missing())
                }
            }),
            _ => Err(ApiError::not_allowed("GET, POST, PUT, DELETE")),
        }
    }
}

/// # Errors
/// The HTTP error to send instead.
fn clear_kv(t: &mut Txn<'_>) -> Res<()> {
    let keys: Vec<Vec<u8>> = t
        .kv_iter(RESTOBJS, b"")?
        .map(|kv| kv.map(|(k, _)| k.to_vec()))
        .collect::<Result<_, _>>()?;
    for k in keys {
        t.kv_del(RESTOBJS, &k)?;
    }
    Ok(())
}

/// Upstream's full-graph JSON dump, streamed node by node.
///
/// # Errors
/// The HTTP error to send instead.
fn dump(
    t: &Txn<'_>,
    g: &Graph,
    uuid: &str,
    view: Option<LogId>,
    nodes: impl Iterator<Item = Result<Entry, GraphError>>,
    edges: impl Iterator<Item = Result<Entry, GraphError>>,
    mut out: Streamer<'_, '_>,
) -> Res<Reply> {
    let head = json!({
        "graph": uuid,
        "id": uuid,
        "maxID": t.next_id()?.saturating_sub(1),
        "size": g.size()?,
        "created": created(uuid),
        "meta": props_dict(t, 0, view)?,
    });
    // `{...head` without its closing brace, then the arrays.
    let head = json_bytes(&head);
    out.raw(head.get(..head.len().saturating_sub(1)).unwrap_or_default())?;
    out.raw(b",\"nodes\":[")?;
    for (i, n) in nodes.enumerate() {
        out.raw(if i == 0 { b"" } else { b"," })?;
        out.raw(&json_bytes(&as_dict(t, &n?, view)?))?;
    }
    out.raw(b"],\"edges\":[")?;
    for (i, e) in edges.enumerate() {
        out.raw(if i == 0 { b"" } else { b"," })?;
        out.raw(&json_bytes(&format_edge(t, &e?, view)?))?;
    }
    out.raw(b"]}\n")?;
    out.finish()
}

/// Runs a query via `run`, writing `row(pattern index, chain dicts)` per match, at most
/// `limit` (0 = all).
///
/// # Errors
/// The HTTP error to send instead.
fn stream_chains(
    t: &Txn<'_>,
    patterns: &[Pattern],
    out: &mut Streamer<'_, '_>,
    limit: usize,
    row: impl Fn(usize, Vec<Value>) -> Res<Value>,
    run: impl FnOnce(&mut dyn FnMut(usize, Option<LogId>, Vec<Entry>) -> bool) -> Result<(), GraphError>,
) -> Res<()> {
    debug_assert!(!patterns.is_empty(), "callers handle the no-query case");
    let (mut n, mut failed) = (0, None);
    run(&mut |qi, at, chain| {
        let written = chain
            .iter()
            .map(|e| as_dict(t, e, at))
            .collect::<Res<Vec<_>>>()
            .and_then(|c| row(qi, c))
            .and_then(|r| out.item(&r));
        if let Err(e) = written {
            failed = Some(e);
            return false;
        }
        n += 1;
        limit == 0 || n < limit
    })?;
    failed.map_or(Ok(()), Err)
}

fn json_bytes(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

/// Everything connected to the query results: nodes, then edges, each once.
///
/// # Errors
/// The HTTP error to send instead.
fn spider(
    t: &Txn<'_>,
    chains: impl Iterator<Item = Vec<Entry>>,
    view: Option<LogId>,
) -> Res<(Vec<Entry>, Vec<Entry>)> {
    let (mut seen, mut nodes, mut edges) = (HashSet::new(), Vec::new(), Vec::new());
    for chain in chains {
        let mut todo: VecDeque<Entry> = chain.into_iter().filter(|e| seen.insert(e.id)).collect();
        while let Some(e) = todo.pop_front() {
            match e.record {
                Record::Node { .. } => {
                    for edge in t.node_edges(e.id, Direction::Both, None, view)? {
                        let edge = edge?;
                        if seen.insert(edge.id) {
                            todo.push_back(edge);
                        }
                    }
                    nodes.push(e);
                }
                Record::Edge { src, tgt, .. } => {
                    for id in [src, tgt] {
                        if seen.insert(id)
                            && let Some(n) = t.entry(id)?
                        {
                            todo.push_back(n);
                        }
                    }
                    edges.push(e);
                }
                Record::Prop { .. } | Record::Deletion { .. } => {}
            }
        }
    }
    Ok((nodes, edges))
}

/// A graph's listing entry (upstream `_status_enrich`).
fn enrich(uuid: &str, s: &Status) -> Value {
    json!({
        "graph": uuid,
        "id": uuid,
        "meta": s.meta,
        "size": s.size,
        "nodes_count": s.nodes,
        "edges_count": s.edges,
        "maxID": s.next_id.saturating_sub(1),
        "created": created(uuid),
    })
}

/// A fresh time-based (v1) UUID, as upstream assigns graph IDs.
fn new_uuid() -> String {
    let random = uuid::Uuid::new_v4().into_bytes();
    let node = random.first_chunk::<6>().copied().unwrap_or_default();
    uuid::Uuid::now_v1(&node).to_string()
}

/// Creation time of a v1 UUID, in Unix seconds.
fn created_secs(uuid: &str) -> Option<i64> {
    let (secs, _) = uuid::Uuid::parse_str(uuid).ok()?.get_timestamp()?.to_unix();
    i64::try_from(secs).ok()
}

/// Creation time of a v1 UUID as `YYYY-MM-DDTHH:MM:SS.ffffffZ`.
fn created(uuid: &str) -> Option<String> {
    let (secs, nanos) = uuid::Uuid::parse_str(uuid).ok()?.get_timestamp()?.to_unix();
    let secs = i64::try_from(secs).ok()?;
    let (y, mo, d) = civil(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    Some(format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}.{:06}Z",
        s / 3600,
        s % 3600 / 60,
        s % 60,
        nanos / 1000
    ))
}

/// Days since 1970-01-01 → (year, month, day). (Howard Hinnant's algorithm.)
const fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if m <= 2 {
            yoe + era * 400 + 1
        } else {
            yoe + era * 400
        },
        m,
        d,
    )
}

/// (year, month, day) → days since 1970-01-01.
const fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// `YYYY-MM-DD[(T| )HH:MM[:SS[.fff]]][Z]` (UTC) → Unix seconds.
// ponytail: a strict subset of upstream's dateutil parsing; widen if clients send other forms.
fn parse_date(s: &str) -> Option<i64> {
    let s = s.trim().trim_end_matches('Z');
    let (date, time) = s.split_once(['T', ' ']).unwrap_or((s, "00:00"));
    let num = |p: Option<&str>| p?.parse::<i64>().ok();
    let mut dp = date.split('-');
    let (y, mo, d) = (num(dp.next())?, num(dp.next())?, num(dp.next())?);
    let mut tp = time.split(':');
    let (h, mi) = (num(tp.next())?, num(tp.next())?);
    let sec = tp.next().map_or(Some(0), |s| num(s.split('.').next()))?;
    let ok = dp.next().is_none()
        && (1..=12).contains(&mo)
        && (1..=31).contains(&d)
        && h < 24
        && mi < 60
        && sec < 61;
    ok.then(|| days(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
