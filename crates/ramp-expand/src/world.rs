//! A synthetic multi-domain world to expand into.
//!
//! Seven domains joined by seven typed relations, with heavy-tailed degrees per domain:
//! organisations and IP addresses have the heaviest tails (an employer of 100k people,
//! a VPN exit shared by 100k accounts), the way real selectors do. The world is built
//! once and held as CSR, grouped by relation per entity, so [`Source::peek`] is free
//! and [`Source::fetch`] is a slice copy. Planted edges are added before the build and
//! remembered, so a run can be scored on whether it recovered them.

use std::collections::HashMap;

use crate::{Counts, Dom, Id, MAX_REL, Rel, Relation, Source};

/// Domain indexes.
pub mod dom {
    use crate::Dom;
    /// People.
    pub const PERSON: Dom = 0;
    /// Employers.
    pub const ORG: Dom = 1;
    /// Phone numbers.
    pub const PHONE: Dom = 2;
    /// Email addresses.
    pub const EMAIL: Dom = 3;
    /// Online accounts.
    pub const ACCOUNT: Dom = 4;
    /// IP addresses.
    pub const IP: Dom = 5;
    /// Devices.
    pub const DEVICE: Dom = 6;
}

/// Relation indexes.
pub mod rel {
    use crate::Rel;
    /// person → org
    pub const WORKS_AT: Rel = 0;
    /// person → phone
    pub const HAS_PHONE: Rel = 1;
    /// person → email
    pub const HAS_EMAIL: Rel = 2;
    /// person → account
    pub const OWNS: Rel = 3;
    /// account → ip
    pub const LOGIN_FROM: Rel = 4;
    /// account → device
    pub const USES_DEVICE: Rel = 5;
    /// account → email
    pub const REGISTERED_WITH: Rel = 6;
}

/// Domain names, by index.
pub const DOMAINS: [&str; 7] = ["person", "org", "phone", "email", "account", "ip", "device"];

/// The relations, by index.
pub const RELATIONS: [Relation; 7] = [
    Relation {
        name: "works_at",
        from: dom::PERSON,
        to: dom::ORG,
    },
    Relation {
        name: "has_phone",
        from: dom::PERSON,
        to: dom::PHONE,
    },
    Relation {
        name: "has_email",
        from: dom::PERSON,
        to: dom::EMAIL,
    },
    Relation {
        name: "owns",
        from: dom::PERSON,
        to: dom::ACCOUNT,
    },
    Relation {
        name: "login_from",
        from: dom::ACCOUNT,
        to: dom::IP,
    },
    Relation {
        name: "uses_device",
        from: dom::ACCOUNT,
        to: dom::DEVICE,
    },
    Relation {
        name: "registered_with",
        from: dom::ACCOUNT,
        to: dom::EMAIL,
    },
];

/// Share of entities per domain, and the Zipf exponent of its degree distribution.
const SHAPE: [(f64, f64); 7] = [
    (0.40, 0.2), // person
    (0.02, 1.1), // org: a few employers with ~100k people
    (0.10, 0.5), // phone
    (0.10, 0.3), // email
    (0.20, 0.4), // account
    (0.08, 1.0), // ip: VPN and NAT exits shared by ~100k accounts
    (0.10, 0.6), // device
];

/// Edges per relation, per entity in the world.
const DENSITY: [f64; 7] = [0.40, 0.24, 0.32, 0.48, 0.80, 0.30, 0.20];

/// splitmix64.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        // 53 random bits make an exact double in [0, 1).
        let bits = u32::try_from(self.next_u64() >> 43).unwrap_or(0);
        f64::from(bits) / f64::from(1_u32 << 21)
    }
}

/// Entities of one domain: a contiguous ID range, sampled by Zipf weight.
#[derive(Debug, Clone)]
struct Range {
    start: u32,
    /// Cumulative weights, rising to 1.
    cum: Vec<f64>,
}

impl Range {
    fn new(start: u32, count: u32, alpha: f64) -> Self {
        let mut cum = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        let mut acc = 0.0;
        for i in 0..count {
            acc += (f64::from(i) + 1.0).powf(-alpha);
            cum.push(acc);
        }
        for c in &mut cum {
            *c /= acc.max(f64::MIN_POSITIVE);
        }
        Self { start, cum }
    }

    fn len(&self) -> u32 {
        u32::try_from(self.cum.len()).unwrap_or(u32::MAX)
    }

    /// An entity drawn with probability proportional to its weight.
    fn sample(&self, rng: &mut Rng) -> u32 {
        let u = rng.unit();
        let i = self.cum.partition_point(|&c| c < u);
        self.start
            + u32::try_from(i)
                .unwrap_or(0)
                .min(self.len().saturating_sub(1))
    }

    /// The `k`th entity of the range (0 is the heaviest, the last the lightest).
    fn nth(&self, k: u32) -> Id {
        Id::from(self.start + k.min(self.len().saturating_sub(1)))
    }
}

/// Edges and planted structure, before the CSR is built.
#[derive(Debug, Clone)]
pub struct Builder {
    n: f64,
    domain: Vec<Dom>,
    ranges: Vec<Range>,
    edges: Vec<(u32, u32, Rel)>,
    planted: Vec<(Id, Id, Rel)>,
    attrs: HashMap<u32, Vec<(&'static str, String)>>,
    /// Picks entities for planted structure, so scenarios never collide.
    used: std::collections::HashSet<u32>,
    rng: Rng,
}

impl Builder {
    /// A random world of about `n` entities, from `seed`.
    #[must_use]
    pub fn random(n: u32, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut domain = Vec::new();
        let mut ranges = Vec::new();
        let total = f64::from(n);
        for (d, &(share, alpha)) in (0_u8..).zip(&SHAPE) {
            let count = (total * share).round().max(1.0);
            let count = format!("{count:.0}").parse::<u32>().unwrap_or(1);
            let start = u32::try_from(domain.len()).unwrap_or(u32::MAX);
            domain.extend(std::iter::repeat_n(d, usize::try_from(count).unwrap_or(0)));
            ranges.push(Range::new(start, count, alpha));
        }
        let mut edges = Vec::new();
        for (r, (relation, &density)) in (0_u8..).zip(RELATIONS.iter().zip(&DENSITY)) {
            let m = format!("{:.0}", (total * density).round())
                .parse::<u32>()
                .unwrap_or(0);
            let (Some(a), Some(b)) = (
                ranges.get(usize::from(relation.from)),
                ranges.get(usize::from(relation.to)),
            ) else {
                continue;
            };
            for _ in 0..m {
                edges.push((a.sample(&mut rng), b.sample(&mut rng), r));
            }
        }
        Self {
            n: total,
            domain,
            ranges,
            edges,
            planted: Vec::new(),
            attrs: HashMap::new(),
            used: std::collections::HashSet::new(),
            rng,
        }
    }

    /// The heaviest entity of domain `d` (rank 0) or a lighter one.
    #[must_use]
    pub fn hub(&self, d: Dom, rank: u32) -> Id {
        self.ranges.get(usize::from(d)).map_or(0, |r| r.nth(rank))
    }

    /// A fresh entity of domain `d` whose expected degree is about `degree`.
    pub fn around(&mut self, d: Dom, degree: f64) -> Id {
        let Some(r) = self.ranges.get(usize::from(d)) else {
            return 0;
        };
        // Expected degree of rank i: the edges of every relation touching `d`, times its
        // share of the domain's weight.
        let edges: f64 = (0_u8..)
            .zip(RELATIONS.iter().zip(&DENSITY))
            .filter(|(_, (rel, _))| rel.from == d || rel.to == d)
            .map(|(_, (_, &density))| self.n * density)
            .sum();
        let mut prev = 0.0;
        let mut pick = r.len().saturating_sub(1);
        for (i, &c) in (0_u32..).zip(&r.cum) {
            if edges * (c - prev) <= degree && !self.used.contains(&(r.start + i)) {
                pick = i;
                break;
            }
            prev = c;
        }
        let id = r.start + pick;
        self.used.insert(id);
        Id::from(id)
    }

    /// A fresh light entity of domain `d`: from the light tail, never handed out before.
    pub fn quiet(&mut self, d: Dom) -> Id {
        let Some(r) = self.ranges.get(usize::from(d)) else {
            return 0;
        };
        let (len, start) = (r.len(), r.start);
        loop {
            // The lightest half of the domain.
            let k = len / 2
                + u32::try_from(self.rng.next_u64() % u64::from((len / 2).max(1))).unwrap_or(0);
            let id = start + k.min(len.saturating_sub(1));
            if self.used.insert(id) {
                return Id::from(id);
            }
        }
    }

    /// Adds edge `a -rel- b` to the world (not as planted signal).
    pub fn link(&mut self, a: Id, b: Id, r: Rel) {
        if let (Ok(a), Ok(b)) = (u32::try_from(a), u32::try_from(b)) {
            self.edges.push((a, b, r));
        }
    }

    /// Adds edge `a -rel- b` and remembers it as planted signal.
    pub fn plant(&mut self, a: Id, b: Id, r: Rel) {
        self.link(a, b, r);
        self.planted.push((a, b, r));
    }

    /// Sets property `key = val` on entity `id`.
    pub fn attr(&mut self, id: Id, key: &'static str, val: &str) {
        if let Ok(id) = u32::try_from(id) {
            self.attrs
                .entry(id)
                .or_default()
                .push((key, val.to_owned()));
        }
    }

    /// Sets `key = val` on a random share `p` of domain `d` (decoys).
    pub fn sprinkle(&mut self, d: Dom, p: f64, key: &'static str, val: &str) {
        let Some(r) = self.ranges.get(usize::from(d)) else {
            return;
        };
        let (start, len) = (r.start, r.len());
        for i in 0..len {
            if self.rng.unit() < p {
                self.attr(Id::from(start + i), key, val);
            }
        }
    }

    /// Builds the CSR.
    #[must_use]
    pub fn build(self) -> World {
        let n = self.domain.len();
        let mut counts = vec![[0_u32; MAX_REL]; n];
        let mut edges = self.edges;
        edges.sort_unstable();
        edges.dedup();
        for &(a, b, r) in &edges {
            for x in [a, b] {
                if let Some(c) = counts
                    .get_mut(usize::try_from(x).unwrap_or(usize::MAX))
                    .and_then(|c| c.get_mut(usize::from(r)))
                {
                    *c += 1;
                }
            }
        }
        // Offsets per (entity, relation), in entity-then-relation order.
        let mut off = Vec::with_capacity(n * MAX_REL + 1);
        let mut acc = 0_u32;
        for c in &counts {
            for &k in c {
                off.push(acc);
                acc += k;
            }
        }
        off.push(acc);
        let mut cur = off.clone();
        let mut adj = vec![0_u32; usize::try_from(acc).unwrap_or(0)];
        for &(a, b, r) in &edges {
            for (x, y) in [(a, b), (b, a)] {
                let slot = usize::try_from(x).unwrap_or(usize::MAX) * MAX_REL + usize::from(r);
                if let Some(c) = cur.get_mut(slot) {
                    if let Some(s) = adj.get_mut(usize::try_from(*c).unwrap_or(usize::MAX)) {
                        *s = y;
                    }
                    *c += 1;
                }
            }
        }
        World {
            domain: self.domain,
            counts,
            off,
            adj,
            planted: self.planted,
            attrs: self.attrs,
        }
    }
}

/// The built world, a [`Source`].
#[derive(Debug)]
pub struct World {
    domain: Vec<Dom>,
    counts: Vec<Counts>,
    off: Vec<u32>,
    adj: Vec<u32>,
    planted: Vec<(Id, Id, Rel)>,
    attrs: HashMap<u32, Vec<(&'static str, String)>>,
}

impl World {
    /// Entities in the world.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.domain.len()
    }

    /// Whether the world is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.domain.is_empty()
    }

    /// Undirected edges in the world.
    #[must_use]
    pub const fn edges(&self) -> usize {
        self.adj.len() / 2
    }

    /// The planted edges, as given to [`Builder::plant`].
    #[must_use]
    pub fn planted(&self) -> &[(Id, Id, Rel)] {
        &self.planted
    }

    /// Total degree of `id`.
    #[must_use]
    pub fn degree(&self, id: Id) -> u32 {
        self.peek(id).iter().sum()
    }
}

impl Source for World {
    fn domains(&self) -> &[&'static str] {
        &DOMAINS
    }

    fn relations(&self) -> &[Relation] {
        &RELATIONS
    }

    fn domain(&self, id: Id) -> Dom {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.domain.get(i))
            .copied()
            .unwrap_or(0)
    }

    fn attrs(&self, id: Id, out: &mut Vec<(&'static str, String)>) {
        if let Some(a) = u32::try_from(id).ok().and_then(|i| self.attrs.get(&i)) {
            out.extend(a.iter().cloned());
        }
    }

    fn peek(&self, id: Id) -> Counts {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.counts.get(i))
            .copied()
            .unwrap_or_default()
    }

    fn fetch(&self, id: Id, rel: Rel, offset: u32, limit: u32, out: &mut Vec<Id>) {
        let Ok(i) = usize::try_from(id) else { return };
        let slot = i * MAX_REL + usize::from(rel);
        let (Some(&lo), Some(&hi)) = (self.off.get(slot), self.off.get(slot + 1)) else {
            return;
        };
        let from = lo.saturating_add(offset).min(hi);
        let to = from.saturating_add(limit).min(hi);
        let (Ok(from), Ok(to)) = (usize::try_from(from), usize::try_from(to)) else {
            return;
        };
        out.extend(
            self.adj
                .get(from..to)
                .unwrap_or_default()
                .iter()
                .map(|&x| Id::from(x)),
        );
    }
}
