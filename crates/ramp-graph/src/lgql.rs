//! LGQL: the `LemonGraph` query language.
//!
//! A pattern is a chain of node and edge slots, e.g. `n(type="foo")->e()-n(depth<=1)`:
//!
//! - `n`/`e` match a node/edge; upper case (`N`/`E`) lets the slot repeat an object
//!   already used elsewhere in the chain (lower-case slots must all be distinct).
//! - `@` omits the slot from results. Adjacent slots of the same kind get an omitted
//!   slot of the other kind inferred between them (`n()->n()` is `n()->@e()->n()`).
//! - Links: `-` (either way), `->`, `<-`, `<->` (either way).
//! - Filters inside the parens must all hold. `key` alone tests existence; nested keys are
//!   dotted (`a.b`). Operators: `=` `!=` (value or `[list]`), `~` `!~` (`/regex/imsx`
//!   or list; search, not anchored), `:` `!:` (`boolean|string|number|array|object`
//!   or list), `<` `<=` `>` `>=` (number or string).
//! - Values: `"str"`/`'str'`, numbers (decimal, float, `0x` hex, `0`-prefixed octal),
//!   `true`/`false`, `null`/`none`.
//! - Trailing filters merge into slots by 1-based position (counting only written
//!   slots) or by alias: `n:a()-n(), a(x=1), 2(y)`. An alias or trailer containing
//!   upper case must have its counterpart.

use std::collections::HashMap;

use regex::Regex;
use serde_json::{Number, Value};

use crate::Direction;

/// A malformed pattern.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{msg} at offset {pos}")]
pub struct ParseError {
    /// Byte offset into the pattern.
    pub pos: usize,
    /// What went wrong.
    pub msg: &'static str,
}

/// Node or edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A node slot.
    Node,
    /// An edge slot.
    Edge,
}

/// Value types for `:` tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ty {
    Boolean,
    String,
    Number,
    Array,
    Object,
}

impl Ty {
    pub(crate) const fn has(self, v: &Value) -> bool {
        matches!(
            (self, v),
            (Self::Boolean, Value::Bool(_))
                | (Self::String, Value::String(_))
                | (Self::Number, Value::Number(_))
                | (Self::Array, Value::Array(_))
                | (Self::Object, Value::Object(_))
        )
    }
}

/// Range comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// One test applied to the value found at a key path.
#[derive(Debug, Clone)]
pub(crate) enum Test {
    Exists,
    In(Vec<Value>),
    NotIn(Vec<Value>),
    Re(Vec<Regex>),
    NotRe(Vec<Regex>),
    Is(Vec<Ty>),
    IsNot(Vec<Ty>),
    Cmp(Cmp, Value),
}

/// A test on a key path.
#[derive(Debug, Clone)]
pub(crate) struct Filter {
    pub(crate) path: Vec<String>,
    pub(crate) test: Test,
}

/// One position in a pattern.
#[derive(Debug, Clone)]
pub(crate) struct Slot {
    pub(crate) kind: Kind,
    /// Included in results.
    pub(crate) keep: bool,
    /// Must differ from objects in other unique slots.
    pub(crate) uniq: bool,
    pub(crate) filters: Vec<Filter>,
    /// Direction to walk toward the next slot.
    pub(crate) fwd: Direction,
    /// Direction to walk toward the previous slot.
    pub(crate) bwd: Direction,
}

/// A compiled pattern.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub(crate) src: String,
    pub(crate) slots: Vec<Slot>,
    /// Slot that queries start from (the most selective one).
    pub(crate) seed: usize,
}

impl Pattern {
    /// Whether running this pattern scans objects (as opposed to starting from an `ID=`
    /// or a node `type=`/`value=` pair, which the indexes answer directly). A scan is
    /// what a [`Projection`](crate::Projection) speeds up.
    #[must_use]
    pub fn scans(&self) -> bool {
        let Some(seed) = self.slots.get(self.seed) else {
            return false;
        };
        if seed.eq_set("ID").is_some() {
            return false;
        }
        !(seed.kind == Kind::Node
            && seed.eq_set("type").is_some()
            && seed.eq_set("value").is_some())
    }

    /// Compiles a pattern.
    ///
    /// # Errors
    /// Returns where and why the pattern is malformed.
    pub fn parse(src: &str) -> Result<Self, ParseError> {
        let mut slots = Parser { src, pos: 0 }.pattern()?;
        // ponytail: seed choice ignores ID/type values that cannot exist; fine until stats-based planning.
        let seed = slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.rank())
            .map_or(0, |(i, _)| i);
        slots.shrink_to_fit();
        Ok(Self {
            src: src.to_owned(),
            slots,
            seed,
        })
    }

    /// The source text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.src
    }
}

impl Slot {
    /// Values a slot requires for a native key via `key=…` (intersected across tests).
    pub(crate) fn eq_set(&self, key: &str) -> Option<Vec<&Value>> {
        let mut set: Option<Vec<&Value>> = None;
        for f in &self.filters {
            if let (Test::In(vals), [k]) = (&f.test, f.path.as_slice())
                && k == key
            {
                set = Some(match set {
                    None => vals.iter().collect(),
                    Some(s) => s.into_iter().filter(|v| vals.contains(v)).collect(),
                });
            }
        }
        set
    }

    /// Seed preference, lowest first (upstream `MatchLGQL.py:509-530`).
    fn rank(&self) -> (u8, usize) {
        let edge = u8::from(self.kind == Kind::Edge);
        if self.eq_set("ID").is_some() {
            return (1 - edge, 0);
        }
        match (self.eq_set("type"), self.eq_set("value")) {
            (Some(t), Some(v)) => (2 + edge, t.len().saturating_mul(v.len())),
            (Some(t), None) => (4 + edge, t.len()),
            _ => (6 + edge, 0),
        }
    }
}

struct Parser<'s> {
    src: &'s str,
    pos: usize,
}

type Parsed<T> = Result<T, ParseError>;

impl<'s> Parser<'s> {
    fn rest(&self) -> &'s str {
        self.src.get(self.pos..).unwrap_or_default()
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    const fn err<T>(&self, msg: &'static str) -> Parsed<T> {
        Err(ParseError { pos: self.pos, msg })
    }

    fn ws(&mut self) {
        let r = self.rest();
        self.pos += r.len() - r.trim_start().len();
    }

    fn eat(&mut self, s: &str) -> bool {
        let hit = self.rest().starts_with(s);
        if hit {
            self.pos += s.len();
        }
        hit
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn take_while(&mut self, f: impl Fn(char) -> bool) -> &'s str {
        let r = self.rest();
        let n = r.find(|c| !f(c)).unwrap_or(r.len());
        self.pos += n;
        r.get(..n).unwrap_or_default()
    }

    fn bareword(&mut self) -> Option<&'s str> {
        self.peek().filter(|c| c.is_alphabetic() || *c == '_')?;
        Some(self.take_while(|c| c.is_alphanumeric() || c == '_'))
    }

    /// Skips commas and whitespace; returns whether there was at least one comma.
    fn commas(&mut self) -> bool {
        self.ws();
        let mut any = false;
        while self.eat(",") {
            any = true;
            self.ws();
        }
        any
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn pattern(&mut self) -> Parsed<Vec<Slot>> {
        let mut objs = Vec::new(); // (slot, aliases)
        let mut links = Vec::new();
        loop {
            self.ws();
            objs.push(self.obj()?);
            self.ws();
            let link = if self.eat("<->") {
                (Direction::Both, Direction::Both)
            } else if self.eat("->") {
                (Direction::Out, Direction::In)
            } else if self.eat("<-") {
                (Direction::In, Direction::Out)
            } else if self.eat("-") {
                (Direction::Both, Direction::Both)
            } else {
                break;
            };
            links.push(link);
        }

        // Aliases: lower-cased name -> (written-slot indexes, required).
        let mut aliases: HashMap<String, (Vec<usize>, bool, bool)> = HashMap::new();
        for (i, (_, names)) in objs.iter().enumerate() {
            for name in names {
                let e = aliases.entry(name.to_lowercase()).or_default();
                e.0.push(i);
                e.1 |= name.chars().any(char::is_uppercase);
            }
        }
        if self.commas() {
            while self.peek().is_some() {
                let at = self.pos;
                let name = self.take_while(|c| c.is_alphanumeric() || c == '_');
                if name.is_empty() || !self.eat("(") {
                    return self.err("expected trailing filter");
                }
                let filters = self.guts()?;
                let targets = if name.starts_with(|c: char| c.is_ascii_digit()) {
                    let idx = name
                        .parse::<usize>()
                        .ok()
                        .and_then(|n| n.checked_sub(1))
                        .filter(|&n| n < objs.len());
                    vec![idx.ok_or(ParseError {
                        pos: at,
                        msg: "slot index out of range",
                    })?]
                } else if let Some(e) = aliases.get_mut(&name.to_lowercase()) {
                    e.2 = true;
                    e.0.clone()
                } else if name.chars().any(char::is_uppercase) {
                    return Err(ParseError {
                        pos: at,
                        msg: "filter names a missing alias",
                    });
                } else {
                    Vec::new()
                };
                for t in targets {
                    if let Some((slot, _)) = objs.get_mut(t) {
                        slot.filters.extend(filters.iter().cloned());
                    }
                }
                self.commas();
            }
        }
        if aliases
            .values()
            .any(|(_, required, used)| *required && !used)
        {
            return self.err("alias with upper case has no filter");
        }
        if self.peek().is_some() {
            return self.err("unexpected input");
        }

        let mut slots: Vec<Slot> = Vec::with_capacity(objs.len() * 2);
        let mut objs = objs.into_iter().map(|(s, _)| s);
        slots.extend(objs.next());
        for ((fwd, bwd), mut next) in links.into_iter().zip(objs) {
            if let Some(prev) = slots.last_mut() {
                prev.fwd = fwd;
                if prev.kind == next.kind {
                    let kind = if next.kind == Kind::Node {
                        Kind::Edge
                    } else {
                        Kind::Node
                    };
                    let uniq = prev.uniq || next.uniq;
                    slots.push(Slot {
                        kind,
                        keep: false,
                        uniq,
                        filters: Vec::new(),
                        fwd,
                        bwd,
                    });
                }
            }
            next.bwd = bwd;
            slots.push(next);
        }
        Ok(slots)
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn obj(&mut self) -> Parsed<(Slot, Vec<&'s str>)> {
        let keep = self.take_while(|c| c == '@').is_empty();
        let (kind, uniq) = match self.bump() {
            Some('n') => (Kind::Node, true),
            Some('N') => (Kind::Node, false),
            Some('e') => (Kind::Edge, true),
            Some('E') => (Kind::Edge, false),
            _ => return self.err("expected n(), N(), e() or E()"),
        };
        let mut names = Vec::new();
        if self.eat(":") {
            loop {
                names.push(
                    self.bareword()
                        .map_or_else(|| self.err("expected alias"), Ok)?,
                );
                if !self.eat(",") {
                    break;
                }
            }
        }
        if !self.eat("(") {
            return self.err("expected '('");
        }
        let filters = self.guts()?;
        let slot = Slot {
            kind,
            keep,
            uniq,
            filters,
            fwd: Direction::Both,
            bwd: Direction::Both,
        };
        Ok((slot, names))
    }

    /// Filters up to and including the closing paren.
    ///
    /// # Errors
    /// Returns where and why the input is malformed.
    fn guts(&mut self) -> Parsed<Vec<Filter>> {
        let mut out = Vec::new();
        loop {
            self.commas();
            if self.eat(")") {
                return Ok(out);
            }
            if !out.is_empty()
                && !self
                    .src
                    .get(..self.pos)
                    .unwrap_or_default()
                    .trim_end()
                    .ends_with(',')
            {
                return self.err("expected ',' or ')'");
            }
            out.push(self.filter()?);
        }
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn filter(&mut self) -> Parsed<Filter> {
        let mut path = vec![self.key()?];
        loop {
            self.ws();
            if !self.eat(".") {
                break;
            }
            self.ws();
            path.push(self.key()?);
        }
        let ops = [
            ("<=", 0),
            (">=", 1),
            ("<", 2),
            (">", 3),
            ("!=", 4),
            ("!~", 5),
            ("!:", 6),
            ("=", 7),
            ("~", 8),
            (":", 9),
        ];
        let Some(op) = ops.iter().find(|(s, _)| self.eat(s)).map(|(_, op)| *op) else {
            return Ok(Filter {
                path,
                test: Test::Exists,
            });
        };
        self.ws();
        let test = match op {
            0..=3 => {
                let cmp = [Cmp::Le, Cmp::Ge, Cmp::Lt, Cmp::Gt]
                    .get(op)
                    .copied()
                    .unwrap_or(Cmp::Lt);
                match self.literal()? {
                    v @ (Value::Number(_) | Value::String(_)) => Test::Cmp(cmp, v),
                    Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => {
                        return self.err("range needs a number or string");
                    }
                }
            }
            4 => Test::NotIn(self.list(Self::literal)?),
            7 => Test::In(self.list(Self::literal)?),
            5 => Test::NotRe(self.list(Self::regex)?),
            8 => Test::Re(self.list(Self::regex)?),
            6 => Test::IsNot(self.list(Self::ty)?),
            _ => Test::Is(self.list(Self::ty)?),
        };
        Ok(Filter { path, test })
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn key(&mut self) -> Parsed<String> {
        if matches!(self.peek(), Some('"' | '\'')) {
            return self.quoted();
        }
        self.bareword()
            .map(str::to_owned)
            .map_or_else(|| self.err("expected key"), Ok)
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn quoted(&mut self) -> Parsed<String> {
        let q = self.bump();
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return self.err("unterminated string"),
                Some('\\') => match self.bump() {
                    Some(c @ ('\\' | '"' | '\'')) => s.push(c),
                    _ => return self.err("bad escape"),
                },
                c if c == q => return Ok(s),
                Some(c) => s.push(c),
            }
        }
    }

    /// One item, or `[item, …]`.
    ///
    /// # Errors
    /// Returns where and why the input is malformed.
    fn list<T>(&mut self, item: fn(&mut Self) -> Parsed<T>) -> Parsed<Vec<T>> {
        if !self.eat("[") {
            return Ok(vec![item(self)?]);
        }
        let mut out = Vec::new();
        loop {
            self.commas();
            if self.eat("]") {
                return Ok(out);
            }
            out.push(item(self)?);
            self.ws();
            if !self.rest().starts_with([',', ']']) {
                return self.err("expected ',' or ']'");
            }
        }
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn literal(&mut self) -> Parsed<Value> {
        if matches!(self.peek(), Some('"' | '\'')) {
            return self.quoted().map(Value::String);
        }
        let at = self.pos;
        if let Some(w) = self.bareword() {
            return match w.to_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                "null" | "none" => Ok(Value::Null),
                _ => Err(ParseError {
                    pos: at,
                    msg: "expected a value",
                }),
            };
        }
        let neg = self.eat("-");
        let n = if self.eat("0x") || self.eat("0X") {
            let digits = self.take_while(|c| c.is_ascii_hexdigit());
            i64::from_str_radix(digits, 16).ok().map(Number::from)
        } else {
            let tok =
                self.take_while(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'));
            if tok.len() > 1
                && tok.starts_with('0')
                && tok.bytes().all(|b| (b'0'..=b'7').contains(&b))
            {
                i64::from_str_radix(tok, 8).ok().map(Number::from)
            } else if !tok.is_empty() && tok.bytes().all(|b| b.is_ascii_digit()) {
                tok.parse::<i64>().ok().map(Number::from)
            } else {
                tok.parse::<f64>().ok().and_then(Number::from_f64)
            }
        };
        let n = n.ok_or(ParseError {
            pos: at,
            msg: "bad number",
        })?;
        Ok(Value::Number(if neg {
            negate(&n).ok_or(ParseError {
                pos: at,
                msg: "bad number",
            })?
        } else {
            n
        }))
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn regex(&mut self) -> Parsed<Regex> {
        let at = self.pos;
        if !self.eat("/") {
            return self.err("expected /regex/");
        }
        let mut pat = String::new();
        loop {
            match self.bump() {
                None => return self.err("unterminated regex"),
                Some('/') => break,
                Some('\\') if self.eat("/") => pat.push('/'),
                Some('\\') => {
                    pat.push('\\');
                    pat.extend(self.bump());
                }
                Some(c) => pat.push(c),
            }
        }
        let flags = self.take_while(|c| matches!(c, 'i' | 'm' | 's' | 'x'));
        let full = if flags.is_empty() {
            pat
        } else {
            format!("(?{flags}){pat}")
        };
        Regex::new(&full).map_err(|_| ParseError {
            pos: at,
            msg: "bad regex",
        })
    }

    /// # Errors
    /// Returns where and why the input is malformed.
    fn ty(&mut self) -> Parsed<Ty> {
        match self.bareword() {
            Some("boolean") => Ok(Ty::Boolean),
            Some("string") => Ok(Ty::String),
            Some("number") => Ok(Ty::Number),
            Some("array") => Ok(Ty::Array),
            Some("object") => Ok(Ty::Object),
            _ => self.err("expected boolean, string, number, array or object"),
        }
    }
}

fn negate(n: &Number) -> Option<Number> {
    n.as_i64()
        .and_then(i64::checked_neg)
        .map(Number::from)
        .or_else(|| Number::from_f64(-n.as_f64()?))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    fn shape(src: &str) -> String {
        let p = Pattern::parse(src).unwrap();
        p.slots
            .iter()
            .map(|s| {
                let k = if s.kind == Kind::Node { 'n' } else { 'e' };
                let k = if s.uniq { k } else { k.to_ascii_uppercase() };
                format!("{}{k}{}", if s.keep { "" } else { "@" }, s.filters.len())
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn chains_and_inference() {
        assert_eq!(shape("n()"), "n0");
        assert_eq!(shape("n()->n()"), "n0 @e0 n0");
        assert_eq!(shape("n()-N()<-e(a)"), "n0 @e0 N0 e1");
        assert_eq!(shape("@e() - e()"), "@e0 @n0 e0");
        assert_eq!(shape("n(a, b=1,, c~/x/i ,)"), "n3");
    }

    #[test]
    fn directions() {
        let p = Pattern::parse("n()->e()<-n()").unwrap();
        let d: Vec<_> = p.slots.iter().map(|s| (s.bwd, s.fwd)).collect();
        assert_eq!(d[0].1, Direction::Out);
        assert_eq!(d[1], (Direction::In, Direction::In));
        assert_eq!(d[2].0, Direction::Out);
    }

    #[test]
    fn literals() {
        let lit = |src: &str| {
            let p = Pattern::parse(&format!("n(a={src})")).unwrap();
            let Test::In(v) = &p.slots[0].filters[0].test else {
                panic!("not an = test")
            };
            v.clone()
        };
        assert_eq!(lit("0x10"), [Value::from(16)], "hex parses (upstream bug)");
        assert_eq!(
            lit("1.5"),
            [Value::from(1.5)],
            "floats kept (upstream truncated)"
        );
        assert_eq!(lit("010"), [Value::from(8)], "octal");
        assert_eq!(lit("-3"), [Value::from(-3)]);
        assert_eq!(
            lit("[1, 'a', TRUE, None,]"),
            [
                Value::from(1),
                Value::from("a"),
                Value::Bool(true),
                Value::Null
            ]
        );
        assert_eq!(lit(r#""q\"x""#), [Value::from("q\"x")]);
    }

    #[test]
    fn regex_escaped_slash() {
        let p = Pattern::parse(r"n(a~/x\/y/i)").unwrap();
        let Test::Re(r) = &p.slots[0].filters[0].test else {
            panic!()
        };
        assert!(r[0].is_match("aX/Yb"));
    }

    #[test]
    fn trailers_and_aliases() {
        assert_eq!(
            shape("n()-n()-n(), 3(x), 1(y, z)"),
            "n2 @e0 n0 @e0 n1",
            "indexes count written slots"
        );
        assert_eq!(shape("n:a()-n:a(), a(x)"), "n1 @e0 n1");
        assert_eq!(shape("n:A()-n(), a(x)"), "n1 @e0 n0");
        assert_eq!(shape("n:a()"), "n0");
        assert_eq!(shape("n(), b(x)"), "n0");
        Pattern::parse("n:Blah()").unwrap_err();
        Pattern::parse("n(), Blah()").unwrap_err();
        Pattern::parse("n(), 2()").unwrap_err();
    }

    #[test]
    fn seed_choice() {
        assert_eq!(Pattern::parse("n()-e(type='x')").unwrap().seed, 1);
        assert_eq!(
            Pattern::parse("e(type='x')-n(type='y', value='z')")
                .unwrap()
                .seed,
            1
        );
        assert_eq!(
            Pattern::parse("n(type='y',value=['a','b'])-n(ID=3)")
                .unwrap()
                .seed,
            2
        );
    }

    #[test]
    fn errors() {
        for bad in [
            "",
            "x()",
            "n(",
            "n(a=)",
            "n(a b)",
            "n()->",
            "n(a</x/)",
            "n(a~/(/)",
            "n(a:thing)",
            "n()) ",
        ] {
            assert!(Pattern::parse(bad).is_err(), "{bad:?} should fail");
        }
    }
}
