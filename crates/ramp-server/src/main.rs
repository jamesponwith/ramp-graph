//! `ramp-server [-i ip] [-p port] [dir]`: serves the graphs in `dir` (default `graphs`)
//! over upstream `LemonGraph`'s REST API.

mod api;
mod input;
mod store;

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::store::Store;

/// Largest accepted request body (uploads are whole graph files).
const MAX_BODY: usize = 1 << 30;
/// Time allowed to receive a request body.
const BODY_TIMEOUT: Duration = Duration::from_secs(60);

const USAGE: &str = "usage: ramp-server [-i ip] [-p port] [dir]";

#[expect(clippy::print_stderr, reason = "CLI diagnostics")]
fn main() -> ExitCode {
    let (mut ip, mut port, mut dir) = ("127.0.0.1".to_owned(), 8000_u16, "graphs".to_owned());
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let ok = match a.as_str() {
            "-i" => args.next().map(|v| ip = v).is_some(),
            "-p" => args
                .next()
                .and_then(|v| v.parse().ok())
                .map(|v| port = v)
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
    let store = match Store::open(&dir) {
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
    tokio::task::spawn_blocking(move || api::handle(&store, &req))
        .await
        .unwrap_or_else(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, "handler panicked\n").into_response()
        })
}
