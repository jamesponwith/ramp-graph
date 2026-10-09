//! Budgeted graph expansion, steered by a policy that can be a System 1 model.
//!
//! Expansion is a search over a world too large to copy: each step pays to fetch some
//! of one entity's edges from a [`Source`] and writes them into a [`ramp_graph::Graph`].
//! What to expand next, and how much of it to take, is the whole problem. A hub (an
//! employer of 100k people, a VPN exit shared by 100k accounts) is expensive to expand
//! and says little about any one of its neighbours, the way a common word says little
//! about a document; expanding hubs blindly floods the graph with noise.
//!
//! Nothing here hardcodes depth or breadth. The [`Expander`] keeps a frontier of
//! discovered entities and, for every one whose situation changes, hands the policy a
//! [`Candidate`]: what the source says is behind it ([`Source::peek`], per relation,
//! before paying for it), how far it is from each seed, what is already known about it
//! in the graph, how many watched entities it touches, and how many standing-query
//! matches it is part of. The policy answers with a [`Decision`]: a priority and which
//! relations to take how much of. A [`Task`] carries the intent behind the expansion
//! (relation and domain interest compiled from a prompt, standing queries, live-query
//! probes, a budget) and is the same object a model would be conditioned on.
//!
//! Decisions are made in batches, so a model's inference can be batched. When a
//! candidate reaches the top of the frontier, the task's probes (LGQL queries rooted at
//! it) run against the live graph and the policy may revise its decision: a cheap score
//! for everything, a live check for what is about to be paid for. Standing queries run
//! over every step's new log entries (`mquery`), and their matches feed back into the
//! candidates they touch.
//!
//! The graph is written constantly and read constantly: writes go to the log (durable,
//! with the history of why each entity was expanded), probes read the incrementally
//! maintained projection, so a probe costs what it would on a quiet graph. [`Reads`]
//! switches probes to LMDB to measure what that saves.

mod expand;
pub mod policy;
pub mod scenario;
pub mod world;

pub use expand::{Expander, Reads, Report};

/// A world entity's ID in a [`Source`].
pub type Id = u64;
/// A domain index into [`Source::domains`].
pub type Dom = u8;
/// A relation index into [`Source::relations`].
pub type Rel = u8;
/// Most domains a source may have.
pub const MAX_DOM: usize = 8;
/// Most relations a source may have.
pub const MAX_REL: usize = 8;
/// A count per relation.
pub type Counts = [u32; MAX_REL];
/// Hops from a seed that has not reached an entity.
pub const UNREACHED: u32 = u32::MAX;

/// A kind of edge, between two domains.
#[derive(Debug, Clone, Copy)]
pub struct Relation {
    /// Edge type in the graph.
    pub name: &'static str,
    /// Domain of the edge's source.
    pub from: Dom,
    /// Domain of the edge's target.
    pub to: Dom,
}

/// The world being expanded into: an external API, a corpus, a feed.
pub trait Source {
    /// Domain names, by [`Dom`].
    fn domains(&self) -> &[&'static str];
    /// Relations, by [`Rel`].
    fn relations(&self) -> &[Relation];
    /// The domain of `id`.
    fn domain(&self, id: Id) -> Dom;
    /// Properties to record on `id` when it is first written.
    fn attrs(&self, id: Id, out: &mut Vec<(&'static str, String)>);
    /// How many edges of each relation `id` has, without fetching them.
    fn peek(&self, id: Id) -> Counts;
    /// Appends up to `limit` neighbours of `id` by `rel`, from position `offset`.
    fn fetch(&self, id: Id, rel: Rel, offset: u32, limit: u32, out: &mut Vec<Id>);
}

/// What an expansion is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goal {
    /// Link `seeds[0]` and `seeds[1]` through meaningful paths.
    Connect,
    /// Find matches of the standing queries.
    Alert,
    /// Map the neighbourhood of the seeds.
    Explore,
}

/// The intent behind an expansion, compiled from a prompt or an analyst's request.
#[derive(Debug, Clone)]
pub struct Task {
    /// Where expansion starts. For [`Goal::Connect`], `seeds[0]` is side 0 and
    /// `seeds[1]` side 1; any others join side 0.
    pub seeds: Vec<Id>,
    /// What counts as progress.
    pub goal: Goal,
    /// How much each relation matters, by [`Rel`] (0 = irrelevant).
    pub relation_interest: [f64; MAX_REL],
    /// How much each domain matters, by [`Dom`] (0 = irrelevant).
    pub domain_interest: [f64; MAX_DOM],
    /// A property whose presence on an entity is signal (a watchlist flag).
    pub watch: Option<&'static str>,
    /// LGQL patterns watched while expanding; matches are reported and fed back.
    pub standing: Vec<String>,
    /// The shapes a goal match takes, as domains outward from a seed (`[person,
    /// account, ip, account, person]`): what the prompt says a hit looks like. Empty if
    /// it says nothing.
    pub shape: Vec<Vec<Dom>>,
    /// LGQL patterns run against the live graph for a candidate about to be expanded,
    /// with `[id]` replaced by its graph ID; their match counts go to the policy.
    pub probes: Vec<String>,
    /// Edges to fetch from the source, at most.
    pub budget: u64,
}

/// What a policy sees about one frontier entity.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The entity.
    pub id: Id,
    /// Its domain.
    pub domain: Dom,
    /// Its edges per relation at the source.
    pub peek: Counts,
    /// Its edges per relation already fetched.
    pub fetched: Counts,
    /// Hops from each side's seeds ([`UNREACHED`] if not reached from that side).
    pub hops: [u32; 2],
    /// The cheapest known route from each side's seeds, as the sum of `ln(1 + degree)`
    /// over the entities passed through (infinite if not reached from that side): a
    /// path through hubs is expensive, a path through specific entities cheap.
    pub cost: [f64; 2],
    /// Its edges in the graph so far.
    pub known: u32,
    /// Its neighbours in the graph, per domain.
    pub known_by_domain: [u32; MAX_DOM],
    /// Its neighbours in the graph that carry [`Task::watch`].
    pub watched: u32,
    /// Whether it carries [`Task::watch`] itself.
    pub is_watched: bool,
    /// Standing-query matches it is part of.
    pub alerts: u32,
    /// Matches of [`Task::probes`] rooted at it, once probed.
    pub probes: Option<u32>,
}

impl Candidate {
    /// Edges per relation still behind it.
    #[must_use]
    pub fn remaining(&self) -> Counts {
        let mut out = [0; MAX_REL];
        for ((o, p), f) in out.iter_mut().zip(&self.peek).zip(&self.fetched) {
            *o = p.saturating_sub(*f);
        }
        out
    }

    /// Hops from the nearer side.
    #[must_use]
    pub fn hops(&self) -> u32 {
        self.hops[0].min(self.hops[1])
    }

    /// The side it was reached from first (0 on a tie).
    #[must_use]
    pub const fn side(&self) -> usize {
        if self.hops[1] < self.hops[0] { 1 } else { 0 }
    }
}

/// What the expansion has spent, for the policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct Context {
    /// Edges fetched so far.
    pub fetched: u64,
    /// Edges fetched expanding each side.
    pub by_side: [u64; 2],
    /// The budget.
    pub budget: u64,
    /// Expansion steps so far.
    pub steps: u64,
}

/// A policy's answer for one candidate.
#[derive(Debug, Clone, Default)]
pub struct Decision {
    /// Higher is expanded sooner.
    pub priority: f64,
    /// Relations to take, and how many more edges of each. Empty: do not expand.
    pub take: Vec<(Rel, u32)>,
}

/// Chooses what to expand. A System 1 model plugs in here.
pub trait Policy {
    /// A label for reports.
    fn name(&self) -> String;
    /// One decision per candidate, in order, appended to `out`.
    fn decide(&mut self, task: &Task, ctx: &Context, batch: &[Candidate], out: &mut Vec<Decision>);
}

impl<P: Policy + ?Sized> Policy for &mut P {
    fn name(&self) -> String {
        (**self).name()
    }

    fn decide(&mut self, task: &Task, ctx: &Context, batch: &[Candidate], out: &mut Vec<Decision>) {
        (**self).decide(task, ctx, batch, out);
    }
}

impl<P: Policy + ?Sized> Policy for Box<P> {
    fn name(&self) -> String {
        (**self).name()
    }

    fn decide(&mut self, task: &Task, ctx: &Context, batch: &[Candidate], out: &mut Vec<Decision>) {
        (**self).decide(task, ctx, batch, out);
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
