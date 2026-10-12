//! JSON for what crosses to a policy outside the process (a System 1 model behind an
//! API) and for the decision trace it learns from.
//!
//! IDs, domains and relations are numbers: [`task`] names them once. Infinite costs
//! and [`UNREACHED`] hops are `null`.

use std::collections::HashSet;
use std::hash::BuildHasher;

use serde_json::{Value, json};

use crate::{
    Candidate, Context, Decision, Filter, Goal, Id, Rel, Report, Source, Step, Take, Task,
    UNREACHED,
};

/// The task, with the source's domain and relation names.
#[must_use]
pub fn task(task: &Task, src: &impl Source) -> Value {
    let relations: Vec<Value> = src
        .relations()
        .iter()
        .map(|r| json!({"name": r.name, "from": r.from, "to": r.to}))
        .collect();
    json!({
        "seeds": task.seeds,
        "goal": match task.goal {
            Goal::Connect => "connect",
            Goal::Alert => "alert",
            Goal::Explore => "explore",
        },
        "domains": src.domains(),
        "relations": relations,
        "relation_interest": task.relation_interest,
        "domain_interest": task.domain_interest,
        "watch": task.watch,
        "standing": task.standing,
        "target": task.target.as_ref().map(filter),
        "shape": task.shape,
        "probes": task.probes,
        "budget": task.budget,
        "filters": src.can_filter(),
    })
}

/// What has been spent.
#[must_use]
pub fn context(ctx: &Context<'_>) -> Value {
    json!({
        "fetched": ctx.fetched,
        "by_side": ctx.by_side,
        "budget": ctx.budget,
        "steps": ctx.steps,
    })
}

/// One candidate.
#[must_use]
pub fn candidate(c: &Candidate) -> Value {
    let hops: Vec<Option<u32>> = c
        .hops
        .iter()
        .map(|&h| (h != UNREACHED).then_some(h))
        .collect();
    json!({
        "id": c.id,
        "domain": c.domain,
        "peek": c.peek,
        "fetched": c.fetched,
        "hops": hops,
        "cost": c.cost,
        "known": c.known,
        "known_by_domain": c.known_by_domain,
        "watched": c.watched,
        "is_watched": c.is_watched,
        "alerts": c.alerts,
        "probes": c.probes,
        "pushed": c.pushed,
    })
}

/// `{"attr": key, "value": v}` or `{"via": rel, "then": filter}`.
#[must_use]
pub fn filter(f: &Filter) -> Value {
    match f {
        Filter::Attr { key, value } => json!({"attr": key, "value": value}),
        Filter::Via { rel, then } => json!({"via": rel, "then": filter(then)}),
    }
}

/// The inverse of [`filter`].
#[must_use]
pub fn parse_filter(v: &Value) -> Option<Filter> {
    if let Some(key) = v.get("attr") {
        return Some(Filter::Attr {
            key: key.as_str()?.to_owned(),
            value: v.get("value")?.as_str()?.to_owned(),
        });
    }
    Some(Filter::Via {
        rel: Rel::try_from(v.get("via")?.as_u64()?).ok()?,
        then: Box::new(parse_filter(v.get("then")?)?),
    })
}

/// `{"priority": p, "take": [{"rel": r, "limit": n, "filter": f?}]}`.
#[must_use]
pub fn decision(d: &Decision) -> Value {
    let take: Vec<Value> = d
        .take
        .iter()
        .map(|t| json!({"rel": t.rel, "limit": t.limit, "filter": t.filter.as_ref().map(filter)}))
        .collect();
    json!({"priority": d.priority, "take": take})
}

/// The inverse of [`decision`]. A missing `take` means do not expand.
///
/// # Errors
/// What is malformed.
pub fn parse_decision(v: &Value) -> Result<Decision, String> {
    let priority = v
        .get("priority")
        .and_then(Value::as_f64)
        .ok_or("decision needs a numeric priority")?;
    let mut take = Vec::new();
    for t in v
        .get("take")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let rel = t
            .get("rel")
            .and_then(Value::as_u64)
            .and_then(|r| Rel::try_from(r).ok())
            .ok_or("take needs a relation index")?;
        let limit = t
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(u32::MAX, |l| u32::try_from(l).unwrap_or(u32::MAX));
        let filter = match t.get("filter") {
            None | Some(Value::Null) => None,
            Some(f) => Some(parse_filter(f).ok_or("malformed filter")?),
        };
        take.push(Take { rel, limit, filter });
    }
    Ok(Decision { priority, take })
}

/// A run's totals.
#[must_use]
pub fn report(r: &Report) -> Value {
    json!({
        "steps": r.steps,
        "rounds": r.rounds,
        "fetched": r.fetched,
        "calls": r.calls,
        "filtered": r.filtered,
        "nodes": r.nodes,
        "edges": r.edges,
        "done_at": r.done_at,
        "alert_at": r.alert_at,
        "connected_at": r.connected_at,
        "alerts": r.alerts,
        "probes": r.probes,
        "decisions": r.decisions,
        "wall_ms": r.wall.as_secs_f64() * 1e3,
        "policy_ms": r.policy.as_secs_f64() * 1e3,
    })
}

/// One trace row: the step's candidate and decision, the edges it added, and how many
/// of them ended up in a hit (see [`Report::hits`]).
#[must_use]
pub fn step<H: BuildHasher>(r: &Report, s: &Step, truth: &HashSet<(Id, Id, Rel), H>) -> Value {
    json!({
        "before": s.before,
        "candidate": candidate(&s.candidate),
        "decision": decision(&s.decision),
        "edges": s.edges.len(),
        "hits": r.hits(s, truth),
    })
}

/// The whole trace as JSON lines.
#[must_use]
pub fn trace<H: BuildHasher>(r: &Report, truth: &HashSet<(Id, Id, Rel), H>) -> Vec<u8> {
    let mut out = Vec::new();
    for s in &r.trace {
        out.extend(step(r, s, truth).to_string().into_bytes());
        out.push(b'\n');
    }
    out
}
