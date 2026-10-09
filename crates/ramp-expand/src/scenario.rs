//! Benchmark scenarios, planted into a [`Builder`] world.
//!
//! Each plants a signal (edges a good expansion should recover) among noise that a
//! blind one walks into: hubs adjacent to the seeds, decoy watchlist entries, and
//! paths that only exist through hubs.

use crate::world::{Builder, dom, rel};
use crate::{Goal, Id, MAX_DOM, MAX_REL, Rel, Task};

/// A planted task.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Short name.
    pub name: &'static str,
    /// What it tests.
    pub about: &'static str,
    /// The task (its budget is set by the caller).
    pub task: Task,
    /// The planted edges a good expansion recovers.
    pub signal: Vec<(Id, Id, Rel)>,
    /// The depth a hardcoded expansion needs to reach the signal at all.
    pub depth: u32,
}

/// "How are these two people linked financially?": accounts, logins, emails matter;
/// employers do not.
const MONEY: [f64; MAX_REL] = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
const MONEY_DOMAINS: [f64; MAX_DOM] = [1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0];

const fn task(seeds: Vec<Id>, goal: Goal, rels: [f64; MAX_REL], doms: [f64; MAX_DOM]) -> Task {
    Task {
        seeds,
        goal,
        relation_interest: rels,
        domain_interest: doms,
        watch: None,
        standing: Vec::new(),
        shape: Vec::new(),
        probes: Vec::new(),
        budget: 0,
    }
}

/// Plants a chain `ends.0 - via... - ends.1` of `(entity, relation)` hops.
fn chain(world: &mut Builder, start: Id, hops: &[(Id, Rel)]) -> Vec<(Id, Id, Rel)> {
    let mut out = Vec::new();
    let mut at = start;
    for &(next, r) in hops {
        world.plant(at, next, r);
        out.push((at, next, r));
        at = next;
    }
    out
}

/// The heaviest entities, which every scenario surrounds its seeds with.
#[derive(Debug, Clone, Copy)]
struct Hubs {
    org0: Id,
    org1: Id,
    ip0: Id,
    ip1: Id,
}

/// Plants every scenario into `world`.
pub fn plant(world: &mut Builder) -> Vec<Scenario> {
    let hubs = Hubs {
        org0: world.hub(dom::ORG, 0),
        org1: world.hub(dom::ORG, 1),
        ip0: world.hub(dom::IP, 0),
        ip1: world.hub(dom::IP, 1),
    };
    vec![
        path5(world, hubs),
        path7(world, hubs),
        pivot(world, hubs),
        alert(world, hubs),
        explore(world),
    ]
}

/// Two people, five hops apart through a moderately shared IP; both work at the
/// biggest employer, and one account also logs in from the biggest VPN exit.
fn path5(world: &mut Builder, hubs: Hubs) -> Scenario {
    let (left, right) = (world.quiet(dom::PERSON), world.quiet(dom::PERSON));
    let (first, second, decoy) = (
        world.quiet(dom::ACCOUNT),
        world.quiet(dom::ACCOUNT),
        world.quiet(dom::ACCOUNT),
    );
    let (ip, email) = (world.around(dom::IP, 200.0), world.quiet(dom::EMAIL));
    let signal = chain(
        world,
        left,
        &[
            (first, rel::OWNS),
            (ip, rel::LOGIN_FROM),
            (second, rel::LOGIN_FROM),
            (email, rel::REGISTERED_WITH),
            (right, rel::HAS_EMAIL),
        ],
    );
    world.link(left, hubs.org0, rel::WORKS_AT);
    world.link(right, hubs.org0, rel::WORKS_AT);
    world.link(first, hubs.ip0, rel::LOGIN_FROM);
    world.link(right, decoy, rel::OWNS);
    world.link(decoy, hubs.ip0, rel::LOGIN_FROM);
    Scenario {
        name: "path-5hop",
        about: "link two people via account-ip-account-email; both at the top employer",
        task: task(vec![left, right], Goal::Connect, MONEY, MONEY_DOMAINS),
        signal,
        depth: 3,
    }
}

/// Seven hops across five domains, through a device shared by ~500 accounts.
fn path7(world: &mut Builder, hubs: Hubs) -> Scenario {
    let (left, right) = (world.quiet(dom::PERSON), world.quiet(dom::PERSON));
    let email = world.quiet(dom::EMAIL);
    let (first, second, third) = (
        world.quiet(dom::ACCOUNT),
        world.quiet(dom::ACCOUNT),
        world.quiet(dom::ACCOUNT),
    );
    let (device, ip) = (
        world.around(dom::DEVICE, 500.0),
        world.around(dom::IP, 50.0),
    );
    let signal = chain(
        world,
        left,
        &[
            (email, rel::HAS_EMAIL),
            (first, rel::REGISTERED_WITH),
            (device, rel::USES_DEVICE),
            (second, rel::USES_DEVICE),
            (ip, rel::LOGIN_FROM),
            (third, rel::LOGIN_FROM),
            (right, rel::OWNS),
        ],
    );
    world.link(left, hubs.org1, rel::WORKS_AT);
    world.link(right, hubs.org0, rel::WORKS_AT);
    for (who, hub) in [(left, hubs.ip0), (right, hubs.ip1)] {
        let decoy = world.quiet(dom::ACCOUNT);
        world.link(who, decoy, rel::OWNS);
        world.link(decoy, hub, rel::LOGIN_FROM);
    }
    Scenario {
        name: "path-7hop",
        about: "seven hops, email-account-device(~500)-account-ip-account",
        task: task(vec![left, right], Goal::Connect, MONEY, MONEY_DOMAINS),
        signal,
        depth: 4,
    }
}

/// One-sided: from a watched person, find a watched counterpart two accounts away.
///
/// The only route runs through an IP shared by ~5000 accounts (a café, an office), so
/// that IP must be expanded, not just met; the seed's other accounts sit behind VPN hubs.
fn pivot(world: &mut Builder, hubs: Hubs) -> Scenario {
    let (seed, target) = (world.quiet(dom::PERSON), world.quiet(dom::PERSON));
    world.attr(seed, "watch", "S");
    world.attr(target, "watch", "T");
    let (mine, theirs) = (world.quiet(dom::ACCOUNT), world.quiet(dom::ACCOUNT));
    let ip = world.around(dom::IP, 5000.0);
    world.link(seed, mine, rel::OWNS);
    world.link(mine, ip, rel::LOGIN_FROM);
    world.plant(theirs, ip, rel::LOGIN_FROM);
    world.plant(target, theirs, rel::OWNS);
    world.link(seed, hubs.org0, rel::WORKS_AT);
    for hub in [hubs.ip0, hubs.ip1] {
        let decoy = world.quiet(dom::ACCOUNT);
        world.link(seed, decoy, rel::OWNS);
        world.link(decoy, hub, rel::LOGIN_FROM);
    }
    let mut task = task(
        vec![seed],
        Goal::Alert,
        [0.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 0.0],
        [1.0, 0.0, 0.2, 0.2, 1.0, 1.0, 0.2, 0.0],
    );
    task.watch = Some("watch");
    task.standing = vec![
        "n(type='person', watch='S')-n(type='account')-n(type='ip')-n(type='account')-n(type='person', watch='T')"
            .to_owned(),
    ];
    task.shape = vec![vec![
        dom::PERSON,
        dom::ACCOUNT,
        dom::IP,
        dom::ACCOUNT,
        dom::PERSON,
    ]];
    Scenario {
        name: "pivot-busy",
        about: "from a watched person through an ip shared by ~5000 accounts to a watched one",
        task,
        signal: vec![(theirs, ip, rel::LOGIN_FROM), (target, theirs, rel::OWNS)],
        depth: 4,
    }
}

/// A watchlist-A person owns 30 accounts, most logging in from the biggest VPN exits;
/// one is co-owned by a watchlist-B person. 0.3% of people are watchlist-B decoys.
fn alert(world: &mut Builder, hubs: Hubs) -> Scenario {
    let (watched, partner) = (world.quiet(dom::PERSON), world.quiet(dom::PERSON));
    world.attr(watched, "watch", "A");
    world.attr(partner, "watch", "B");
    world.sprinkle(dom::PERSON, 0.003, "watch", "B");
    world.link(watched, hubs.org0, rel::WORKS_AT);
    let mut shared = 0;
    for i in 0..30 {
        let account = world.quiet(dom::ACCOUNT);
        world.link(watched, account, rel::OWNS);
        world.link(
            account,
            if i % 3 == 0 { hubs.ip1 } else { hubs.ip0 },
            rel::LOGIN_FROM,
        );
        if i == 17 {
            shared = account;
        }
    }
    world.plant(partner, shared, rel::OWNS);
    let mut task = task(
        vec![watched],
        Goal::Alert,
        [0.0, 0.2, 0.2, 1.0, 0.1, 0.2, 0.2, 0.0],
        [1.0, 0.0, 0.2, 0.2, 1.0, 0.2, 0.2, 0.0],
    );
    task.watch = Some("watch");
    task.standing = vec![
        "n(type='person', watch='A')-n(type='account')-n(type='person', watch='B')".to_owned(),
    ];
    task.shape = vec![vec![dom::PERSON, dom::ACCOUNT, dom::PERSON]];
    Scenario {
        name: "alert",
        about: "standing query: a watch-A person shares an account with a watch-B person",
        task,
        signal: vec![(partner, shared, rel::OWNS)],
        depth: 2,
    }
}

/// Open-ended mapping with a live-query probe per expansion (for the read-path
/// comparison): how many people sit two hops from the candidate in the graph.
fn explore(world: &mut Builder) -> Scenario {
    let seed = world.around(dom::PERSON, 20.0);
    let mut task = task(
        vec![seed],
        Goal::Explore,
        [0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    );
    task.probes = vec!["n(ID=[id])-n()-n(type='person')".to_owned()];
    Scenario {
        name: "explore",
        about: "map a neighbourhood, probing the live graph before each expansion",
        task,
        signal: Vec::new(),
        depth: 3,
    }
}
