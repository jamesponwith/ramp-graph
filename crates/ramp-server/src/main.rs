//! `ramp-server [-i ip] [-p port] [-n max_open] [-m projection_mib] [dir]`: serves the graphs in `dir` (default `graphs`)
//! over upstream `LemonGraph`'s REST API.

mod api;
mod input;
mod store;

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::store::Store;

/// Largest accepted request body (uploads are whole graph files).
const MAX_BODY: usize = 1 << 30;
/// Time allowed to receive a request body.
const BODY_TIMEOUT: Duration = Duration::from_secs(60);

const USAGE: &str = "usage: ramp-server [-i ip] [-p port] [-n max_open] [-m projection_mib] [dir]";

#[expect(clippy::print_stderr, reason = "CLI diagnostics")]
fn main() -> ExitCode {
    let (mut ip, mut port, mut dir) = ("127.0.0.1".to_owned(), 8000_u16, "graphs".to_owned());
    // Idle graphs kept open. Each costs 3 file descriptors and ~2 MiB of RAM (LMDB
    // preallocates its write-txn dirty list per environment).
    let mut max_open = 64_usize;
    // Projections kept across open graphs (≈300 MiB per million nodes, edges, and properties).
    let mut projection_mib = 1024_usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let ok = match a.as_str() {
            "-i" => args.next().map(|v| ip = v).is_some(),
            "-p" => args
                .next()
                .and_then(|v| v.parse().ok())
                .map(|v| port = v)
                .is_some(),
            "-n" => args
                .next()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n > 0)
                .map(|v| max_open = v)
                .is_some(),
            "-m" => args
                .next()
                .and_then(|v| v.parse().ok())
                .map(|v| projection_mib = v)
                .is_some(),
            "-h" | "--help" => false,
            _ if !a.starts_with('-') => {
                dir = a;
                true
            }
            _ => false,
        };
        if !ok {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    }
    let store = match Store::open(&dir, max_open, projection_mib << 20) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("cannot open {dir}: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("cannot start runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let served = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind((ip.as_str(), port)).await?;
        eprintln!("serving {dir} on http://{ip}:{port}");
        let app = Router::new().fallback(serve).with_state(store);
        let shutdown = async { tokio::signal::ctrl_c().await.unwrap_or_default() };
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
    });
    if let Err(e) = served {
        eprintln!("server error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn serve(State(store): State<Arc<Store>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = match tokio::time::timeout(BODY_TIMEOUT, axum::body::to_bytes(body, MAX_BODY)).await
    {
        Ok(Ok(b)) => b,
        Ok(Err(_)) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large or broken\n",
            )
                .into_response();
        }
        Err(_) => return (StatusCode::REQUEST_TIMEOUT, "request body timed out\n").into_response(),
    };
    let req = api::Req {
        method: parts.method.as_str().to_owned(),
        path: parts.uri.path().to_owned(),
        query: parts.uri.query().unwrap_or_default().to_owned(),
        headers: parts.headers,
        body,
    };
    let (head_tx, head_rx) = oneshot::channel();
    let (body_tx, body_rx) = mpsc::channel(4);
    let task = tokio::task::spawn_blocking(move || {
        api::handle(
            &store,
            &req,
            &mut Channel {
                head: Some(head_tx),
                body: body_tx,
            },
        );
    });
    let Ok((status, headers)) = head_rx.await else {
        // The handler panicked before responding.
        drop(task.await);
        return (StatusCode::INTERNAL_SERVER_ERROR, "handler panicked\n").into_response();
    };
    let mut res = Response::new(Body::from_stream(ReceiverStream::new(body_rx)));
    *res.status_mut() = status;
    for (k, v) in headers {
        if let (Ok(k), Ok(v)) = (HeaderName::try_from(k), HeaderValue::try_from(v)) {
            res.headers_mut().append(k, v);
        }
    }
    res
}

/// Carries a response from the blocking handler to the async side: the head through a
/// oneshot, body chunks through a small bounded channel, so a slow client stalls the
/// handler instead of growing memory.
struct Channel {
    head: Option<oneshot::Sender<api::Head>>,
    body: mpsc::Sender<Result<Bytes, std::io::Error>>,
}

impl api::Wire for Channel {
    fn start(&mut self, status: StatusCode, headers: Vec<(&'static str, String)>) {
        if let Some(head) = self.head.take() {
            // If the connection is gone, the following writes report it.
            head.send((status, headers)).unwrap_or_default();
        }
    }

    fn write(&mut self, chunk: Vec<u8>) -> bool {
        self.body.blocking_send(Ok(Bytes::from(chunk))).is_ok()
    }

    fn fail(&mut self) {
        // An error item makes hyper abort the connection mid-body.
        self.body
            .blocking_send(Err(std::io::Error::other("response aborted")))
            .unwrap_or_default();
    }
}
