//! The ICIJ Offshore Leaks database as a [`World`].
//!
//! Officers, entities, intermediaries and addresses, ~2M nodes and ~3.3M
//! relationships, with real hubs (registered agents with 100k+ entities, addresses
//! shared by thousands). Load the unzipped CSVs from
//! <https://offshoreleaks-data.icij.org> (Open Database License; not in the repo) with
//! [`load`].
//!
//! Nothing is planted, so tasks are sampled from the data:
//! - [`connect_tasks`]: two officers 4 to 6 hops apart, with the cheapest path between
//!   them (the sum of `ln(1 + degree)` over the entities it passes through) and its
//!   widest hub.
//! - [`alert_tasks`]: from an officer, a co-officer (sharing an entity) from the country
//!   rarest among them, as a standing query.
//!
//! [`near_cheapest`] and [`shared_with_target`] are their oracles: what a policy that
//! saw the whole graph would expand.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;

use ramp_graph::{GraphError, Result};

use crate::world::{Builder, Rng, World};
use crate::{Dom, Filter, Goal, Id, MAX_DOM, MAX_REL, Rel, Relation, Source, Task};

/// Domain names, by [`Dom`].
pub const DOMAINS: [&str; 5] = ["entity", "officer", "intermediary", "address", "other"];
/// Companies, trusts and foundations.
pub const ENTITY: Dom = 0;
/// People and companies that hold a role in an entity.
pub const OFFICER: Dom = 1;
/// Agents that set entities up (law firms, banks, registered agents).
pub const INTERMEDIARY: Dom = 2;
/// Registered addresses.
pub const ADDRESS: Dom = 3;
/// Everything else (registry records without a type).
pub const OTHER: Dom = 4;

/// Officer to entity: director, shareholder, beneficiary and the like.
pub const OFFICER_OF: Rel = 0;
/// The 13 ICIJ relationship types, grouped into six relations.
pub const RELATIONS: [Relation; 6] = [
    Relation {
        name: "officer_of",
        from: OFFICER,
        to: ENTITY,
    },
    Relation {
        name: "intermediary_of",
        from: INTERMEDIARY,
        to: ENTITY,
    },
    Relation {
        name: "registered_address",
        from: ENTITY,
        to: ADDRESS,
    },
    Relation {
        name: "same_as",
        from: OFFICER,
        to: OFFICER,
    },
    Relation {
        name: "similar",
        from: ENTITY,
        to: ENTITY,
    },
    Relation {
        name: "connected_to",
        from: ENTITY,
        to: ENTITY,
    },
];

/// The relation group of an ICIJ `rel_type`.
fn relation(ty: &str) -> Option<Rel> {
    Some(match ty {
        "officer_of" => 0,
        "intermediary_of" => 1,
        "registered_address" => 2,
        "same_name_as"
        | "same_as"
        | "same_id_as"
        | "probably_same_officer_as"
        | "same_intermediary_as"
        | "same_address_as" => 3,
        "similar" | "similar_company_as" | "same_company_as" => 4,
        "connected_to" | "underlying" => 5,
        _ => return None,
    })
}

/// CSV records: quoted fields may hold commas, doubled quotes and newlines.
fn records(text: &str) -> impl Iterator<Item = Vec<String>> + '_ {
    let mut chars = text.chars().peekable();
    std::iter::from_fn(move || {
        chars.peek()?;
        let (mut rec, mut field, mut quoted) = (Vec::new(), String::new(), false);
        while let Some(c) = chars.next() {
            match (c, quoted) {
                ('"', true) if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                ('"', _) => quoted = !quoted,
                (',', false) => rec.push(std::mem::take(&mut field)),
                ('\n', false) => break,
                ('\r', false) => {}
                (c, _) => field.push(c),
            }
        }
        rec.push(field);
        Some(rec)
    })
}

fn io(e: impl std::fmt::Display) -> GraphError {
    GraphError::Io(e.to_string())
}

/// Columns left out of an entity's text: identifiers and provenance boilerplate.
const UNSAID: [&str; 6] = [
    "node_id",
    "internal_id",
    "ibcRUC",
    "country_codes",
    "valid_until",
    "note",
];

/// Most distinct relationship labels in an entity's text.
const LINKS: usize = 6;

/// Loads the CSVs in `dir`.
///
/// Officers carry `country` (their `country_codes`). Every entity's [`Source::text`]
/// is its domain, its non-empty columns (`name: ...; address: ...`), and up to six
/// distinct labels of its relationships out (`links: nominee director of`) and in
/// (`linked by: shareholder of`).
///
/// # Errors
/// A file that cannot be read.
pub fn load(dir: &Path) -> Result<World> {
    let mut b = Builder::empty(&DOMAINS, &RELATIONS);
    let mut ids: HashMap<u64, Id> = HashMap::new();
    let mut texts: Vec<String> = Vec::new();
    for (file, d) in [
        ("nodes-entities.csv", ENTITY),
        ("nodes-officers.csv", OFFICER),
        ("nodes-intermediaries.csv", INTERMEDIARY),
        ("nodes-addresses.csv", ADDRESS),
        ("nodes-others.csv", OTHER),
    ] {
        let text = std::fs::read_to_string(dir.join(file)).map_err(io)?;
        let mut rows = records(&text);
        let header = rows.next().unwrap_or_default();
        let country = header.iter().position(|h| h == "country_codes");
        for row in rows {
            let Some(n) = row.first().and_then(|n| n.parse::<u64>().ok()) else {
                continue;
            };
            if ids.contains_key(&n) {
                continue;
            }
            let id = b.entity(d);
            ids.insert(n, id);
            let mut text = DOMAINS
                .get(usize::from(d))
                .copied()
                .unwrap_or("?")
                .to_owned();
            for (h, v) in header.iter().zip(&row) {
                if !v.is_empty() && !UNSAID.contains(&h.as_str()) {
                    text.push_str(if text.contains(':') { "; " } else { " " });
                    text.push_str(h);
                    text.push_str(": ");
                    text.push_str(v.trim());
                }
            }
            texts.push(text);
            if let Some(c) = country
                .and_then(|i| row.get(i))
                .filter(|c| d == OFFICER && !c.is_empty())
            {
                b.attr(id, "country", c);
            }
        }
    }
    let text = std::fs::read_to_string(dir.join("relationships.csv")).map_err(io)?;
    // Per entity, the labels of its relationships out and in.
    let mut links: HashMap<Id, [Vec<String>; 2]> = HashMap::new();
    for row in records(&text).skip(1) {
        let (Some(a), Some(z), Some(r)) = (row.first(), row.get(1), row.get(2)) else {
            continue;
        };
        let (Ok(a), Ok(z), Some(r)) = (a.parse::<u64>(), z.parse::<u64>(), relation(r)) else {
            continue;
        };
        if let (Some(&a), Some(&z)) = (ids.get(&a), ids.get(&z))
            && a != z
        {
            b.link(a, z, r);
            let label = row
                .get(3)
                .map(|l| l.trim().to_lowercase())
                .unwrap_or_default();
            for (x, end) in [(a, 0), (z, 1)].into_iter().filter(|_| !label.is_empty()) {
                if let Some(seen) = links.entry(x).or_default().get_mut(end)
                    && seen.len() < LINKS
                    && !seen.contains(&label)
                {
                    seen.push(label.clone());
                }
            }
        }
    }
    for (id, mut text) in (0..).zip(texts) {
        for (end, labels) in ["; links: ", "; linked by: "]
            .into_iter()
            .zip(links.get(&id).into_iter().flatten())
            .filter(|(_, l)| !l.is_empty())
        {
            text.push_str(end);
            text.push_str(&labels.join(", "));
        }
        b.text(id, text);
    }
    Ok(b.build())
}

/// Every neighbour of `id`.
pub fn neighbours(w: &World, id: Id, out: &mut Vec<Id>) {
    out.clear();
    for r in (0..RELATIONS.len()).filter_map(|r| Rel::try_from(r).ok()) {
        w.fetch(id, r, 0, u32::MAX, out);
    }
}

fn officers(w: &World) -> Vec<Id> {
    (0..w.len())
        .filter_map(|i| Id::try_from(i).ok())
        .filter(|&i| w.domain(i) == OFFICER && w.degree(i) > 0)
        .collect()
}

/// Hops from `from` to everything within `depth`.
fn hops(w: &World, from: Id, depth: u8) -> HashMap<Id, u8> {
    let mut seen = HashMap::from([(from, 0_u8)]);
    let (mut layer, mut buf) = (vec![from], Vec::new());
    for d in 1..=depth {
        let mut next = Vec::new();
        for &x in &layer {
            neighbours(w, x, &mut buf);
            for &y in &buf {
                seen.entry(y).or_insert_with(|| {
                    next.push(y);
                    d
                });
            }
        }
        layer = next;
    }
    seen
}

/// The least hub-heavy path from `a` to `b`: its cost, its widest hub, and its
/// entities from `b` back to `a`.
///
/// It minimises the sum of `ln(1 + degree)` (in `w`) over the entities it passes
/// through, over the edges `adj` appends for an entity ([`neighbours`] for all of `w`).
pub fn cheapest(
    w: &World,
    adj: impl Fn(Id, &mut Vec<Id>),
    a: Id,
    b: Id,
) -> Option<(f64, u32, Vec<Id>)> {
    let weight = |x: Id| f64::from(w.degree(x)).ln_1p();
    let mut best: HashMap<Id, (f64, Id)> = HashMap::from([(a, (0.0, a))]);
    let mut heap = BinaryHeap::from([(Reverse(Ord64(0.0)), a)]);
    let mut buf = Vec::new();
    while let Some((Reverse(Ord64(c)), x)) = heap.pop() {
        if x == b {
            let mut widest = 0;
            let mut path = vec![b];
            let mut at = best.get(&b)?.1;
            while at != a {
                widest = widest.max(w.degree(at));
                path.push(at);
                at = best.get(&at)?.1;
            }
            path.push(a);
            return Some((c, widest, path));
        }
        if best.get(&x).is_some_and(|&(k, _)| k < c) {
            continue;
        }
        let step = if x == a { 0.0 } else { weight(x) };
        buf.clear();
        adj(x, &mut buf);
        for &y in &buf {
            let k = c + step;
            if best.get(&y).is_none_or(|&(old, _)| k < old) {
                best.insert(y, (k, x));
                heap.push((Reverse(Ord64(k)), y));
            }
        }
    }
    None
}

/// How much dearer than the cheapest a path may be and still label its entities.
pub const EPS: f64 = 0.1;

/// The cost of the cheapest path from `from` to everything it reaches for at most
/// `bound`, counted as in [`cheapest`]: the entities passed through, not the ends.
fn within(w: &World, from: Id, bound: f64) -> HashMap<Id, f64> {
    let mut best: HashMap<Id, f64> = HashMap::from([(from, 0.0)]);
    let mut heap = BinaryHeap::from([(Reverse(Ord64(0.0)), from)]);
    let mut buf = Vec::new();
    while let Some((Reverse(Ord64(c)), x)) = heap.pop() {
        if best.get(&x).is_some_and(|&k| k < c) {
            continue;
        }
        let k = c + if x == from {
            0.0
        } else {
            f64::from(w.degree(x)).ln_1p()
        };
        if k > bound {
            continue;
        }
        neighbours(w, x, &mut buf);
        for &y in &buf {
            if best.get(&y).is_none_or(|&old| k < old) {
                best.insert(y, k);
                heap.push((Reverse(Ord64(k)), y));
            }
        }
    }
    best
}

/// The connect oracle: the seeds, and every entity on a path between them costing at
/// most `1 + EPS` times the cheapest, `best`.
pub fn near_cheapest(w: &World, a: Id, b: Id, best: f64) -> HashSet<Id> {
    let bound = best.mul_add(1.0 + EPS, 1e-9);
    let (from_a, from_b) = (within(w, a, bound), within(w, b, bound));
    let mut out: HashSet<Id> = from_a
        .iter()
        .filter(|&(&x, &ca)| {
            from_b
                .get(&x)
                .is_some_and(|&cb| ca + cb + f64::from(w.degree(x)).ln_1p() <= bound)
        })
        .map(|(&x, _)| x)
        .collect();
    out.extend([a, b]);
    out
}

/// The alert oracle: the seed, each entity it shares with an officer from the target
/// country, and those officers.
pub fn shared_with_target(w: &World, task: &Task) -> HashSet<Id> {
    let (Some(&s), Some(Filter::Attr { key, value })) = (task.seeds.first(), &task.target) else {
        return HashSet::new();
    };
    let mut out = HashSet::from([s]);
    let (mut entities, mut co, mut attrs) = (Vec::new(), Vec::new(), Vec::new());
    w.fetch(s, OFFICER_OF, 0, u32::MAX, &mut entities);
    for &e in &entities {
        co.clear();
        w.fetch(e, OFFICER_OF, 0, u32::MAX, &mut co);
        for &o in co.iter().filter(|&&o| o != s) {
            attrs.clear();
            w.attrs(o, &mut attrs);
            if attrs.iter().any(|(k, v)| k == key && v == value) {
                out.extend([e, o]);
            }
        }
    }
    out
}

/// An `f64` ordered by `total_cmp`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ord64(f64);

impl Eq for Ord64 {}

impl PartialOrd for Ord64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ord64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The name in `id`'s text, or its ID.
fn name(w: &World, id: Id) -> String {
    w.text(id)
        .split("; ")
        .find_map(|f| f.split_once("name: ").map(|(_, n)| n.to_owned()))
        .unwrap_or_else(|| id.to_string())
}

const fn task(
    prompt: String,
    seeds: Vec<Id>,
    goal: Goal,
    relation_interest: [f64; MAX_REL],
    budget: u64,
) -> Task {
    Task {
        prompt,
        seeds,
        goal,
        relation_interest,
        domain_interest: [1.0; MAX_DOM],
        watch: None,
        standing: Vec::new(),
        target: None,
        shape: Vec::new(),
        probes: Vec::new(),
        budget,
    }
}

/// The connect task between officers `a` and `b`, with the cost and widest hub of the
/// cheapest path between them; `None` if there is none.
#[must_use]
pub fn connect_task(w: &World, a: Id, b: Id, budget: u64) -> Option<(Task, (f64, u32))> {
    let (cost, hub, _) = cheapest(w, |x, out| neighbours(w, x, out), a, b)?;
    let prompt = format!(
        "How is the officer {} connected to the officer {}? Find the most specific link, \
         through companies, people and addresses particular to them rather than mass \
         registration agents or shared addresses.",
        name(w, a),
        name(w, b)
    );
    let t = task(prompt, vec![a, b], Goal::Connect, [1.0; MAX_REL], budget);
    Some((t, (cost, hub)))
}

/// Connect tasks: officer pairs 4 to 6 hops apart, with their cheapest path.
pub fn connect_tasks(w: &World, n: usize, rng: &mut Rng, budget: u64) -> Vec<(Task, (f64, u32))> {
    let officers = officers(w);
    let mut out = Vec::new();
    while out.len() < n {
        let Some(&a) = officers.get(rng.below(officers.len())) else {
            break;
        };
        let mut far: Vec<Id> = hops(w, a, 6)
            .into_iter()
            .filter(|&(x, d)| d >= 4 && w.domain(x) == OFFICER)
            .map(|(x, _)| x)
            .collect();
        far.sort_unstable();
        let Some(&b) = far.get(rng.below(far.len())) else {
            continue;
        };
        out.extend(connect_task(w, a, b, budget));
    }
    out
}

/// The alert task from officer `s`: a co-officer (sharing an entity) from the country
/// rarest among them, not `s`'s own. `None` if `s` has fewer than 50 co-officers with
/// a country.
#[must_use]
pub fn alert_task(w: &World, s: Id, budget: u64) -> Option<Task> {
    let country = |x: Id| {
        let mut a = Vec::new();
        w.attrs(x, &mut a);
        a.into_iter().find(|(k, _)| *k == "country").map(|(_, v)| v)
    };
    let (mut entities, mut co) = (Vec::new(), Vec::new());
    w.fetch(s, OFFICER_OF, 0, u32::MAX, &mut entities);
    let mut counts: HashMap<String, u32> = HashMap::new();
    for &e in &entities {
        co.clear();
        w.fetch(e, OFFICER_OF, 0, u32::MAX, &mut co);
        for &o in co.iter().filter(|&&o| o != s) {
            if let Some(c) = country(o) {
                *counts.entry(c).or_default() += 1;
            }
        }
    }
    if counts.values().sum::<u32>() < 50 {
        return None;
    }
    let own = country(s);
    let (rare, _) = counts
        .into_iter()
        .filter(|(c, _)| Some(c) != own.as_ref())
        .min_by(|x, y| x.1.cmp(&y.1).then_with(|| x.0.cmp(&y.0)))?;
    let mut interest = [0.2; MAX_REL];
    if let Some(i) = interest.get_mut(usize::from(OFFICER_OF)) {
        *i = 1.0;
    }
    let prompt = format!(
        "Alert when the officer {} shares a company with an officer from {rare}.",
        name(w, s)
    );
    let mut t = task(prompt, vec![s], Goal::Alert, interest, budget);
    t.standing = vec![format!(
        "n(type='officer', value='{s}')-n(type='entity')-n(type='officer', country='{}')",
        rare.replace('\'', "\\'")
    )];
    t.shape = vec![vec![OFFICER, ENTITY, OFFICER]];
    t.target = Some(Filter::Attr {
        key: "country".to_owned(),
        value: rare,
    });
    Some(t)
}

/// Alert tasks ([`alert_task`]) from randomly chosen officers.
pub fn alert_tasks(w: &World, n: usize, rng: &mut Rng, budget: u64) -> Vec<Task> {
    let officers = officers(w);
    let mut out = Vec::new();
    let mut tries = 0;
    while out.len() < n && tries < 1_000_000 {
        tries += 1;
        let Some(&s) = officers.get(rng.below(officers.len())) else {
            break;
        };
        out.extend(alert_task(w, s, budget));
    }
    out
}

/// A path's cost and widest hub (as in [`cheapest`]), and its edges as `(min, max,
/// rel)`.
pub type Joined = (f64, u32, HashSet<(Id, Id, Rel)>);

/// The cheapest path between `from` and `to` over the edges written so far (degrees
/// in `w`).
pub fn joining<'e>(
    w: &World,
    written: impl Iterator<Item = &'e (Id, Id, Rel)>,
    from: Id,
    to: Id,
) -> Option<Joined> {
    let written: Vec<(Id, Id, Rel)> = written.copied().collect();
    let mut adj: HashMap<Id, Vec<Id>> = HashMap::new();
    for &(x, y, _) in &written {
        adj.entry(x).or_default().push(y);
        adj.entry(y).or_default().push(x);
    }
    let known = |x: Id, out: &mut Vec<Id>| out.extend(adj.get(&x).into_iter().flatten());
    let (cost, hub, on) = cheapest(w, known, from, to)?;
    let pairs: HashSet<(Id, Id)> = on
        .windows(2)
        .filter_map(|p| match *p {
            [x, y] => Some((x.min(y), x.max(y))),
            _ => None,
        })
        .collect();
    let edges = written
        .into_iter()
        .filter(|&(x, y, _)| pairs.contains(&(x, y)))
        .collect();
    Some((cost, hub, edges))
}
