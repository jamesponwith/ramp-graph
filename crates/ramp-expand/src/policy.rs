//! Policies: two hardcoded baselines, and a stand-in for a System 1 model.
//!
//! [`Bfs`] and [`Rules`] are what expansion looks like without a model: a depth limit,
//! and for [`Rules`] the task's relations and a degree cap, all fixed in advance and
//! tuned per task. [`Guided`] has no depth or breadth constants; it scores what a
//! model would see. It is a baseline for a learned policy, not a substitute: a model
//! conditioned on the task's prompt replaces it behind the same [`Policy`] trait.

use crate::{
    Candidate, Context, Decision, Dom, Filter, MAX_REL, Policy, Rel, Relation, Take, Task,
    UNREACHED,
};

/// Breadth-first to a fixed depth: every relation, every edge.
#[derive(Debug, Clone, Copy)]
pub struct Bfs {
    /// Entities this many hops out are not expanded.
    pub depth: u32,
}

impl Policy for Bfs {
    fn name(&self) -> String {
        format!("bfs(depth={})", self.depth)
    }

    fn decide(&mut self, _: &Task, _: &Context<'_>, batch: &[Candidate], out: &mut Vec<Decision>) {
        out.extend(batch.iter().map(|c| {
            let h = c.hops();
            if h >= self.depth || h == UNREACHED {
                return Decision::default();
            }
            Decision {
                priority: -f64::from(h),
                take: take_all(c, |_| true),
            }
        }));
    }
}

/// Hand-tuned rules: the task's relations only, a fixed depth, and entities with more
/// than `cap` edges skipped.
#[derive(Debug, Clone, Copy)]
pub struct Rules {
    /// Entities this many hops out are not expanded.
    pub depth: u32,
    /// Entities with more edges than this are not expanded.
    pub cap: u32,
}

impl Policy for Rules {
    fn name(&self) -> String {
        format!("rules(depth={}, cap={})", self.depth, self.cap)
    }

    fn decide(
        &mut self,
        task: &Task,
        _: &Context<'_>,
        batch: &[Candidate],
        out: &mut Vec<Decision>,
    ) {
        out.extend(batch.iter().map(|c| {
            let h = c.hops();
            let degree: u32 = c.peek.iter().sum();
            if h >= self.depth || h == UNREACHED || degree > self.cap {
                return Decision::default();
            }
            Decision {
                priority: -f64::from(h),
                take: take_all(c, |r| interest(task, r) > 0.0),
            }
        }));
    }
}

/// A stand-in for a System 1 model: scores the features a model would see, with no
/// depth or breadth constants.
///
/// - Cost: passing through an entity with `d` edges costs `ln(1 + d)`, so a hub is
///   expensive and a specific entity cheap. A candidate's priority is the cheapest known
///   route to it from either side ([`Candidate::cost`]) plus the cost of expanding it,
///   `ln(1 + relevant edges left to fetch)`: uniform-cost search in that metric, which
///   finds the least hub-heavy paths first and never needs a depth limit. With two sides
///   it is bidirectional, meeting in the middle.
/// - Relevance: only relations the task cares about are fetched, and only they count
///   towards the cost; the cost is divided by the task's interest in the entity's domain
///   and by signal (watched neighbours, standing-query matches, probe hits), so what the
///   task cares about is cheap.
/// - Shape: if the task says what a hit looks like ([`Task::shape`]), an entity whose
///   domain fits its distance from the seed, with more of the shape beyond it, keeps
///   its weight; anything else is discounted 20-fold, not excluded.
/// - Breadth: it takes all of a relation's remaining edges up to 1/64 of the remaining
///   budget; a hub that is reached at all is taken in chunks, and as it shrinks it gets
///   cheaper to finish.
/// - Pushdown (if enabled): when the task names a target and the candidate sits on the
///   shape with more of it beyond, it fetches only the next step's relation, filtered by
///   the semi-join chain to the target, and prices that as a few edges, whatever the
///   relation's size. A hub on the way to a hit costs what it returns.
#[derive(Debug, Clone, Copy, Default)]
pub struct Guided {
    /// Push the task's target down to the source as filters.
    pub pushdown: bool,
}

fn interest(task: &Task, r: usize) -> f64 {
    task.relation_interest.get(r).copied().unwrap_or(0.0)
}

/// 1 if expanding `c` can extend a hit of the task's shape (its domain sits at its
/// distance from the seed, with more of the shape beyond it), 1/20 if not; 1 with no
/// shape.
fn fit(task: &Task, c: &Candidate) -> f64 {
    if task.shape.is_empty() {
        return 1.0;
    }
    let at = usize::try_from(c.hops()).unwrap_or(usize::MAX);
    let extends = task
        .shape
        .iter()
        .any(|s| s.get(at) == Some(&c.domain) && at + 1 < s.len());
    if extends { 1.0 } else { 0.05 }
}

/// Every remaining edge of each relation `want` accepts.
fn take_all(c: &Candidate, want: impl Fn(usize) -> bool) -> Vec<Take> {
    (0_u8..)
        .zip(c.remaining())
        .take(MAX_REL)
        .filter(|&(r, n)| n > 0 && want(usize::from(r)))
        .map(|(r, n)| Take::all(r, n))
        .collect()
}

/// The relation joining domains `a` and `b`, either way round.
fn joining(relations: &[Relation], a: Dom, b: Dom) -> Option<Rel> {
    (0_u8..)
        .zip(relations)
        .find(|(_, r)| (r.from, r.to) == (a, b) || (r.from, r.to) == (b, a))
        .map(|(i, _)| i)
}

/// The filtered fetch that leads from `c` towards the task's target, if there is one.
///
/// For a candidate on the task's shape with more of it beyond: the relation to its next
/// step, and a filter that holds only on neighbours leading to the target (the target
/// wrapped in one semi-join per remaining step of the shape). `None` if the task names
/// no target, the candidate is off-shape, or the relation was already fetched filtered.
fn pushdown(task: &Task, ctx: &Context<'_>, c: &Candidate) -> Option<(Rel, Filter)> {
    if !ctx.filters {
        return None;
    }
    let target = task.target.as_ref()?;
    let at = usize::try_from(c.hops()).ok()?;
    task.shape.iter().find_map(|shape| {
        if shape.get(at) != Some(&c.domain) || at + 1 >= shape.len() {
            return None;
        }
        let mut filter = target.clone();
        for i in (at + 1..shape.len() - 1).rev() {
            let (a, b) = (*shape.get(i)?, *shape.get(i + 1)?);
            filter = Filter::Via {
                rel: joining(ctx.relations, a, b)?,
                then: Box::new(filter),
            };
        }
        let rel = joining(ctx.relations, c.domain, *shape.get(at + 1)?)?;
        let fresh = !c.pushed.get(usize::from(rel)).copied().unwrap_or(true);
        fresh.then_some((rel, filter))
    })
}

impl Policy for Guided {
    fn name(&self) -> String {
        if self.pushdown {
            "guided+pushdown".to_owned()
        } else {
            "guided".to_owned()
        }
    }

    fn decide(
        &mut self,
        task: &Task,
        ctx: &Context<'_>,
        batch: &[Candidate],
        out: &mut Vec<Decision>,
    ) {
        let left = ctx.budget.saturating_sub(ctx.fetched);
        let breadth = u32::try_from(left / 64).unwrap_or(u32::MAX).max(16);
        for c in batch {
            let rem = c.remaining();
            let total: u32 = rem.iter().sum();
            let relevant: f64 = rem
                .iter()
                .enumerate()
                .map(|(r, &n)| interest(task, r) * f64::from(n))
                .sum();
            if total == 0 || relevant <= 0.0 {
                out.push(Decision::default());
                continue;
            }
            let domain = task
                .domain_interest
                .get(usize::from(c.domain))
                .copied()
                .unwrap_or(0.0);
            let probes = c.probes.map_or(0.0, |p| f64::from(p).ln_1p());
            let signal = 2.0_f64.mul_add(
                probes,
                8.0_f64.mul_add(
                    f64::from(c.alerts),
                    4.0_f64.mul_add(
                        f64::from(c.watched) + f64::from(u8::from(c.is_watched)),
                        1.0,
                    ),
                ),
            );
            let weight = domain * signal * fit(task, c);
            if weight <= 0.0 {
                out.push(Decision::default());
                continue;
            }
            let route = c.cost[0].min(c.cost[1]);
            if let Some((rel, filter)) = self.pushdown.then(|| pushdown(task, ctx, c)).flatten() {
                // What comes back is what can complete a hit: a few edges, not the hub.
                out.push(Decision {
                    priority: -(route + 2.0_f64.ln()) / weight,
                    take: vec![Take {
                        rel,
                        limit: breadth,
                        filter: Some(filter),
                    }],
                });
                continue;
            }
            // What expanding it costs is what is left to fetch of what the task wants: a
            // fresh hub is expensive, a nearly finished one cheap to finish.
            let left: u32 = rem
                .iter()
                .enumerate()
                .filter(|&(r, _)| interest(task, r) > 0.0)
                .map(|(_, &n)| n)
                .sum();
            let cost = route + f64::from(left).ln_1p();
            out.push(Decision {
                priority: -cost / weight,
                take: (0_u8..)
                    .zip(rem)
                    .take(MAX_REL)
                    .filter(|&(r, n)| n > 0 && interest(task, usize::from(r)) > 0.0)
                    .map(|(r, n)| Take::all(r, n.min(breadth)))
                    .collect(),
            });
        }
    }
}
