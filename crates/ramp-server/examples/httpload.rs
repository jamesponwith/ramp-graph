//! A keep-alive HTTP load generator for the REST numbers in `docs/spec.md`.
//!
//! `cargo run --release -p ramp-server --example httpload -- HOST:PORT PATH CONNS PER_CONN [N]`
//!
//! `PATH` may contain `[k]`, replaced by `(i * 104729) % N` for request `i`, and `[m]`,
//! replaced by `k % 5`, so point lookups spread over the query-bench graph. Prints
//! throughput and latency percentiles over all requests.

use std::io::{Error, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

fn arg<T: std::str::FromStr>(args: &[String], i: usize) -> Option<T> {
    args.get(i)?.parse().ok()
}

/// One connection: `per` sequential requests, returning each latency and the bytes read.
///
/// # Errors
/// A connection or protocol failure, or a non-200 status.
fn worker(
    addr: &str,
    template: &str,
    conn: usize,
    per: usize,
    modulus: u64,
) -> std::io::Result<(Vec<Duration>, usize)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_nodelay(true)?;
    let mut buf = vec![0_u8; 1 << 16];
    let (mut latencies, mut bytes) = (Vec::with_capacity(per), 0);
    for i in 0..per {
        let key = u64::try_from(conn * per + i)
            .unwrap_or(u64::MAX)
            .wrapping_mul(104_729)
            % modulus.max(1);
        let path = template
            .replace("[k]", &key.to_string())
            .replace("[m]", &(key % 5).to_string());
        let started = Instant::now();
        stream.write_all(
            format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n").as_bytes(),
        )?;
        let mut head = Vec::new();
        loop {
            let got = stream.read(&mut buf)?;
            if got == 0 {
                return Err(Error::other("connection closed"));
            }
            head.extend_from_slice(buf.get(..got).unwrap_or_default());
            let Some(split) = head.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let header =
                String::from_utf8_lossy(head.get(..split).unwrap_or_default()).to_ascii_lowercase();
            if !header.starts_with("http/1.1 200") {
                return Err(Error::other(
                    header.lines().next().unwrap_or("bad status").to_owned(),
                ));
            }
            let len: usize = header
                .lines()
                .find_map(|l| l.strip_prefix("content-length: "))
                .and_then(|v| v.trim().parse().ok())
                .ok_or_else(|| Error::other("no content-length"))?;
            let mut have = head.len().saturating_sub(split + 4);
            while have < len {
                let got = stream.read(&mut buf)?;
                if got == 0 {
                    return Err(Error::other("connection closed in body"));
                }
                have += got;
            }
            bytes += len;
            break;
        }
        latencies.push(started.elapsed());
    }
    Ok((latencies, bytes))
}

/// Milliseconds at percentile `pct` of sorted `sorted`.
fn percentile(sorted: &[Duration], pct: usize) -> f64 {
    let i = (sorted.len() * pct / 100).min(sorted.len().saturating_sub(1));
    sorted.get(i).map_or(0.0, |d| d.as_secs_f64() * 1e3)
}

#[expect(clippy::print_stdout, reason = "benchmark report")]
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(addr), Some(template), Some(conns), Some(per)) = (
        args.get(1),
        args.get(2),
        arg::<usize>(&args, 3),
        arg::<usize>(&args, 4),
    ) else {
        return Err(Error::other(
            "usage: httpload HOST:PORT PATH CONNS PER_CONN [N]",
        ));
    };
    let modulus: u64 = arg(&args, 5).unwrap_or(1_000_000);
    let started = Instant::now();
    let results: Vec<std::io::Result<(Vec<Duration>, usize)>> = std::thread::scope(|scope| {
        #[expect(
            clippy::needless_collect,
            reason = "spawn every connection before joining any"
        )]
        let handles: Vec<_> = (0..conns)
            .map(|conn| scope.spawn(move || worker(addr, template, conn, per, modulus)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::other("worker panicked")))
            })
            .collect()
    });
    let mut all = Vec::new();
    let mut bytes = 0_usize;
    for r in results {
        let (lat, b) = r?;
        all.extend(lat);
        bytes += b;
    }
    let elapsed = started.elapsed().as_secs_f64();
    all.sort_unstable();
    let count = u32::try_from(all.len()).map_or(f64::MAX, f64::from);
    let mib = u32::try_from(bytes / 1024).map_or(f64::MAX, f64::from) / 1024.0;
    println!(
        "{} req in {elapsed:.2}s: {:.0} req/s, {:.1} MiB/s, p50 {:.3}ms p99 {:.3}ms",
        all.len(),
        count / elapsed,
        mib / elapsed,
        percentile(&all, 50),
        percentile(&all, 99)
    );
    Ok(())
}
