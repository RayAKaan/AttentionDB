//! Typed metadata filter AST + evaluator (Phase 2 §11–13).
//!
//! Design:
//! - Filters are a typed AST (`FilterExpr`), never string matching.
//! - Values are typed: string, integer, float, bool, null.
//! - Evaluation is TOTAL and never panics on missing fields or wrong types.
//!
//! ## Null / missing semantics (explicit, two-valued logic)
//!
//! - `field op value` where the field is MISSING or NULL evaluates to FALSE
//!   for every op except `IsNull`.
//! - `field != value` on a missing/null field is therefore FALSE as well
//!   (SQL-style: an absent value is not "different from value").
//! - Use `IsNull`/`IsNotNull` to test presence explicitly.
//! - NOT(...) composes the two-valued result: `NOT (a = 5)` DOES match
//!   documents missing `a` (they don't have a = 5). This is a deliberate,
//!   documented departure from SQL three-valued logic — predictable and total.
//!
//! ## Type coercion rules
//!
//! - Int and Float compare numerically with each other.
//! - Str compares lexicographically with Str.
//! - Bool compares only with Bool (equality ops; ordering → false).
//! - Any other cross-type comparison → false (never an error).

use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl FilterOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            FilterOp::Eq => "=",
            FilterOp::Ne => "!=",
            FilterOp::Gt => ">",
            FilterOp::Gte => ">=",
            FilterOp::Lt => "<",
            FilterOp::Lte => "<=",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    Comparison {
        field: String,
        op: FilterOp,
        value: FilterValue,
    },
    In {
        field: String,
        values: Vec<FilterValue>,
        negated: bool,
    },
    IsNull {
        field: String,
        negated: bool,
    },
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
    Not(Box<FilterExpr>),
}

/// Maximum AST depth accepted from untrusted input (§42).
pub const MAX_FILTER_DEPTH: usize = 32;
/// Maximum total nodes accepted from untrusted input (§42).
pub const MAX_FILTER_NODES: usize = 256;

impl FilterExpr {
    pub fn node_count(&self) -> usize {
        match self {
            FilterExpr::Comparison { .. } | FilterExpr::IsNull { .. } => 1,
            FilterExpr::In { .. } => 1,
            FilterExpr::And(a, b) | FilterExpr::Or(a, b) => 1 + a.node_count() + b.node_count(),
            FilterExpr::Not(a) => 1 + a.node_count(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            FilterExpr::Comparison { .. } | FilterExpr::In { .. } | FilterExpr::IsNull { .. } => 1,
            FilterExpr::And(a, b) | FilterExpr::Or(a, b) => 1 + a.depth().max(b.depth()),
            FilterExpr::Not(a) => 1 + a.depth(),
        }
    }

    /// Reject unbounded filters early (§42).
    pub fn validate(&self) -> Result<(), String> {
        if self.depth() > MAX_FILTER_DEPTH {
            return Err(format!("filter exceeds max depth {}", MAX_FILTER_DEPTH));
        }
        if self.node_count() > MAX_FILTER_NODES {
            return Err(format!("filter exceeds max nodes {}", MAX_FILTER_NODES));
        }
        Ok(())
    }

    /// Evaluate against a document's fields. Total: never panics, never NaN.
    pub fn eval(&self, fields: &HashMap<String, Value>) -> bool {
        match self {
            FilterExpr::Comparison { field, op, value } => {
                compare(get_typed(fields, field).as_ref(), *op, value)
            }
            FilterExpr::In {
                field,
                values,
                negated,
            } => {
                let missing_or_null = fields.get(field).map(Value::is_null).unwrap_or(true);
                let in_list = match get_typed(fields, field) {
                    None => false,
                    Some(actual) => values.iter().any(|v| typed_eq(&actual, v)),
                };
                if *negated {
                    // NOT IN: present AND not in the list. Missing/null never
                    // matches, even negated (see module docs).
                    !in_list && !missing_or_null
                } else {
                    in_list
                }
            }
            FilterExpr::IsNull { field, negated } => {
                let is_null = fields.get(field).map(Value::is_null).unwrap_or(true);
                if *negated {
                    !is_null
                } else {
                    is_null
                }
            }
            FilterExpr::And(a, b) => a.eval(fields) && b.eval(fields),
            FilterExpr::Or(a, b) => a.eval(fields) || b.eval(fields),
            FilterExpr::Not(a) => !a.eval(fields),
        }
    }
}

/// Extract a typed value from the field map. Returns None for missing/null
/// and for values that are not scalar (arrays/objects never match).
fn get_typed(fields: &HashMap<String, Value>, field: &str) -> Option<FilterValue> {
    match fields.get(field) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(FilterValue::Str(s.clone())),
        Some(Value::Bool(b)) => Some(FilterValue::Bool(*b)),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Some(FilterValue::Int(i))
            } else {
                n.as_f64().map(FilterValue::Float)
            }
        }
        _ => None,
    }
}

fn typed_eq(actual: &FilterValue, expected: &FilterValue) -> bool {
    match (actual, expected) {
        (FilterValue::Str(a), FilterValue::Str(b)) => a == b,
        (FilterValue::Bool(a), FilterValue::Bool(b)) => a == b,
        (FilterValue::Null, FilterValue::Null) => true,
        (FilterValue::Int(a), FilterValue::Int(b)) => a == b,
        (a, b) => match (num_of(a), num_of(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

fn num_of(v: &FilterValue) -> Option<f64> {
    match v {
        FilterValue::Int(i) => Some(*i as f64),
        FilterValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// Total comparison: None (missing/null) never matches anything but IsNull.
fn compare(actual: Option<&FilterValue>, op: FilterOp, expected: &FilterValue) -> bool {
    let Some(a) = actual else {
        return false;
    };
    if matches!(expected, FilterValue::Null) {
        // `field = NULL` is spelled IsNull; a Comparison against Null is
        // defined as false for every op (never silently equal).
        return false;
    }
    match op {
        FilterOp::Eq => typed_eq(a, expected),
        FilterOp::Ne => {
            // != is the negation of == ONLY when types are comparable;
            // cross-type (e.g. "5" != 5) is false, not true.
            match comparable(a, expected) {
                Some(eq) => !eq,
                None => false,
            }
        }
        FilterOp::Gt | FilterOp::Gte | FilterOp::Lt | FilterOp::Lte => match order(a, expected) {
            Some(o) => match op {
                FilterOp::Gt => o == std::cmp::Ordering::Greater,
                FilterOp::Gte => o != std::cmp::Ordering::Less,
                FilterOp::Lt => o == std::cmp::Ordering::Less,
                FilterOp::Lte => o != std::cmp::Ordering::Greater,
                _ => unreachable!(),
            },
            None => false,
        },
    }
}

fn comparable(a: &FilterValue, b: &FilterValue) -> Option<bool> {
    match (a, b) {
        (FilterValue::Str(x), FilterValue::Str(y)) => Some(x == y),
        (FilterValue::Bool(x), FilterValue::Bool(y)) => Some(x == y),
        _ => match (num_of(a), num_of(b)) {
            (Some(x), Some(y)) => Some(x == y),
            _ => None,
        },
    }
}

fn order(a: &FilterValue, b: &FilterValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (FilterValue::Str(x), FilterValue::Str(y)) => Some(x.cmp(y)),
        _ => match (num_of(a), num_of(b)) {
            (Some(x), Some(y)) => x.partial_cmp(&y),
            _ => None,
        },
    }
}

// ---------------------------------------------------------------------------
// Structured (JSON) parsing for untrusted input — typed, bounded (§42)
// ---------------------------------------------------------------------------

/// Parse the wire format:
/// `{"field":"category","op":"=","value":"book"}`
/// `{"field":"tag","op":"in","values":[...],"negated":true}`
/// `{"field":"deleted_at","op":"is_null"}` / `{"op":"is_not_null","field":...}`
/// `{"and":[...,...]}`, `{"or":[...,...]}`, `{"not":{...}}`
pub fn parse_filter_json(v: &Value) -> Result<FilterExpr, String> {
    let expr = parse_node(v)?;
    expr.validate()?;
    Ok(expr)
}

fn parse_node(v: &Value) -> Result<FilterExpr, String> {
    let obj = v.as_object().ok_or("filter node must be an object")?;
    // logical combinators first
    for key in ["and", "or"] {
        if let Some(items) = obj.get(key) {
            let arr = items.as_array().ok_or(format!("{key} must be an array"))?;
            if arr.len() < 2 {
                return Err(format!("{key} requires at least 2 clauses"));
            }
            let parsed: Vec<FilterExpr> = arr.iter().map(parse_node).collect::<Result<_, _>>()?;
            let mut it = parsed.into_iter();
            let first = it.next().unwrap();
            let mut acc = first;
            let comb = if key == "and" {
                |a: Box<FilterExpr>, b: Box<FilterExpr>| FilterExpr::And(a, b)
            } else {
                |a: Box<FilterExpr>, b: Box<FilterExpr>| FilterExpr::Or(a, b)
            };
            for next in it {
                acc = comb(Box::new(acc), Box::new(next));
            }
            return Ok(acc);
        }
    }
    if let Some(inner) = obj.get("not") {
        return Ok(FilterExpr::Not(Box::new(parse_node(inner)?)));
    }
    // comparisons
    let field = obj
        .get("field")
        .and_then(|f| f.as_str())
        .ok_or("filter comparison requires a string 'field'")?
        .to_string();
    let op = obj
        .get("op")
        .and_then(|o| o.as_str())
        .ok_or("filter comparison requires a string 'op'")?;
    match op {
        "is_null" => Ok(FilterExpr::IsNull {
            field,
            negated: false,
        }),
        "is_not_null" => Ok(FilterExpr::IsNull {
            field,
            negated: true,
        }),
        "in" | "not_in" => {
            let vals = obj
                .get("values")
                .and_then(|x| x.as_array())
                .ok_or("'in' requires 'values' array")?;
            if vals.len() > 1024 {
                return Err("IN list too long (max 1024)".into());
            }
            Ok(FilterExpr::In {
                field,
                values: vals.iter().map(parse_value).collect::<Result<_, _>>()?,
                negated: op == "not_in",
            })
        }
        "=" | "==" | "!=" | ">" | ">=" | "<" | "<=" => {
            let raw = obj.get("value").ok_or("comparison requires 'value'")?;
            let value = parse_value(raw)?;
            let op = match op {
                "=" | "==" => FilterOp::Eq,
                "!=" => FilterOp::Ne,
                ">" => FilterOp::Gt,
                ">=" => FilterOp::Gte,
                "<" => FilterOp::Lt,
                _ => FilterOp::Lte,
            };
            Ok(FilterExpr::Comparison { field, op, value })
        }
        other => Err(format!("unsupported filter op '{other}'")),
    }
}

fn parse_value(v: &Value) -> Result<FilterValue, String> {
    Ok(match v {
        Value::Null => FilterValue::Null,
        Value::String(s) => FilterValue::Str(s.clone()),
        Value::Bool(b) => FilterValue::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                FilterValue::Int(i)
            } else {
                FilterValue::Float(n.as_f64().ok_or("number out of range")?)
            }
        }
        _ => return Err("filter values must be scalars".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fields(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn cmp(field: &str, op: FilterOp, value: FilterValue) -> FilterExpr {
        FilterExpr::Comparison {
            field: field.into(),
            op,
            value,
        }
    }

    #[test]
    fn string_equality_and_inequality() {
        let f = fields(&[("category", json!("book")), ("other", json!("toy"))]);
        assert!(cmp("category", FilterOp::Eq, FilterValue::Str("book".into())).eval(&f));
        assert!(!cmp("category", FilterOp::Eq, FilterValue::Str("to".into())).eval(&f));
        assert!(cmp("category", FilterOp::Ne, FilterValue::Str("toy".into())).eval(&f));
        // cross-type != is false, not true
        assert!(!cmp("category", FilterOp::Ne, FilterValue::Int(5)).eval(&f));
    }

    #[test]
    fn numeric_ranges_and_cross_type_number_compare() {
        let f = fields(&[("year", json!(2021)), ("price", json!(9.5))]);
        assert!(cmp("year", FilterOp::Gte, FilterValue::Int(2020)).eval(&f));
        assert!(!cmp("year", FilterOp::Lt, FilterValue::Int(2021)).eval(&f));
        // int field vs float literal
        assert!(cmp("year", FilterOp::Eq, FilterValue::Float(2021.0)).eval(&f));
        assert!(cmp("price", FilterOp::Gt, FilterValue::Int(9)).eval(&f));
        assert!(!cmp("price", FilterOp::Gt, FilterValue::Int(10)).eval(&f));
    }

    #[test]
    fn booleans_and_wrong_types() {
        let f = fields(&[("active", json!(true))]);
        assert!(cmp("active", FilterOp::Eq, FilterValue::Bool(true)).eval(&f));
        assert!(
            !cmp("active", FilterOp::Eq, FilterValue::Int(1)).eval(&f),
            "bool vs int never matches"
        );
        assert!(
            !cmp("active", FilterOp::Gt, FilterValue::Bool(false)).eval(&f),
            "bool ordering never matches"
        );
    }

    #[test]
    fn null_and_missing_semantics_are_explicit() {
        let with_null = fields(&[("deleted_at", json!(null))]);
        let missing = fields(&[("a", json!(1))]);
        let present = fields(&[("deleted_at", json!("2024-01-01"))]);

        // = null is spelled IsNull
        assert!(FilterExpr::IsNull {
            field: "deleted_at".into(),
            negated: false
        }
        .eval(&with_null));
        assert!(FilterExpr::IsNull {
            field: "deleted_at".into(),
            negated: false
        }
        .eval(&missing));
        assert!(!FilterExpr::IsNull {
            field: "deleted_at".into(),
            negated: false
        }
        .eval(&present));

        // comparisons on missing/null → false (including !=)
        for op in [FilterOp::Eq, FilterOp::Ne, FilterOp::Gt, FilterOp::Lte] {
            assert!(
                !cmp("deleted_at", op, FilterValue::Str("x".into())).eval(&with_null),
                "op {op:?}"
            );
            assert!(
                !cmp("deleted_at", op, FilterValue::Str("x".into())).eval(&missing),
                "op {op:?}"
            );
        }

        // NOT IN on missing/null → false (documented)
        let not_in = FilterExpr::In {
            field: "deleted_at".into(),
            values: vec![FilterValue::Str("x".into())],
            negated: true,
        };
        assert!(!not_in.eval(&with_null));
        assert!(!not_in.eval(&missing));
        assert!(not_in.eval(&present));

        // NOT(a = x) is TRUE for missing a (documented two-valued logic)
        let not_eq = FilterExpr::Not(Box::new(cmp("a", FilterOp::Eq, FilterValue::Int(5))));
        assert!(not_eq.eval(&missing));
    }

    #[test]
    fn and_or_not_nested() {
        let f = fields(&[
            ("category", json!("book")),
            ("year", json!(2022)),
            ("active", json!(true)),
        ]);
        let q = FilterExpr::And(
            Box::new(cmp(
                "category",
                FilterOp::Eq,
                FilterValue::Str("book".into()),
            )),
            Box::new(FilterExpr::Or(
                Box::new(cmp("year", FilterOp::Gte, FilterValue::Int(2023))),
                Box::new(FilterExpr::And(
                    Box::new(cmp("year", FilterOp::Eq, FilterValue::Int(2022))),
                    Box::new(FilterExpr::Not(Box::new(cmp(
                        "active",
                        FilterOp::Eq,
                        FilterValue::Bool(false),
                    )))),
                )),
            )),
        );
        assert!(q.eval(&f));
        let q2 = FilterExpr::And(
            Box::new(cmp(
                "category",
                FilterOp::Eq,
                FilterValue::Str("book".into()),
            )),
            Box::new(cmp("year", FilterOp::Gte, FilterValue::Int(2023))),
        );
        assert!(!q2.eval(&f));
    }

    #[test]
    fn arrays_and_objects_never_match() {
        let f = fields(&[("tags", json!(["a", "b"]))]);
        assert!(!cmp("tags", FilterOp::Eq, FilterValue::Str("a".into())).eval(&f));
    }

    #[test]
    fn json_parsing_end_to_end() {
        let v = json!({
            "and": [
                {"field": "category", "op": "=", "value": "book"},
                {"or": [
                    {"field": "year", "op": ">=", "value": 2020},
                    {"field": "tag", "op": "in", "values": ["new", "sale"]}
                ]},
                {"not": {"field": "deleted_at", "op": "is_not_null"}}
            ]
        });
        let expr = parse_filter_json(&v).unwrap();
        let doc = fields(&[
            ("category", json!("book")),
            ("year", json!(2019)),
            ("tag", json!("sale")),
            ("deleted_at", json!(null)),
        ]);
        assert!(expr.eval(&doc));
    }

    #[test]
    fn json_parsing_rejects_unsupported_and_overbounded() {
        assert!(parse_filter_json(&json!({"field": "a", "op": "~=", "value": 1})).is_err());
        let mut deep = json!({"field": "a", "op": "=", "value": 1});
        for _ in 0..40 {
            deep = json!({ "not": deep });
        }
        assert!(parse_filter_json(&deep).is_err(), "depth bound must reject");
    }

    #[test]
    fn empty_results_and_empty_fields() {
        let empty = fields(&[]);
        assert!(!cmp("any", FilterOp::Eq, FilterValue::Int(0)).eval(&empty));
        assert!(!FilterExpr::IsNull {
            field: "any".into(),
            negated: true
        }
        .eval(&empty));
    }

    /// §34 fuzz: filter construction + validate + eval must be total — no
    /// panics on deterministic pseudo-random expression trees; validate()
    /// must reject over-deep trees.
    #[test]
    fn filter_validate_eval_fuzz_no_panic() {
        let fields = fields(&[
            ("cat", json!("sport")),
            ("num", json!(42)),
            ("f", json!(1.5)),
            ("flag", json!(true)),
            ("empty", Value::Null),
        ]);
        let leaf_fields = ["cat", "num", "f", "flag", "empty", "missing"];
        let mut x = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mk_value = |n: u64| match n % 5 {
            0 => FilterValue::Str("sport".into()),
            1 => FilterValue::Int((n % 200) as i64),
            2 => FilterValue::Float((n % 97) as f64),
            3 => FilterValue::Bool(n % 2 == 0),
            _ => FilterValue::Null,
        };
        for _ in 0..256u32 {
            let expr = build_random(next(), &leaf_fields, &mk_value, 0);
            let _ = expr.node_count();
            let v = expr.validate();
            assert!(v.is_ok(), "bounded random tree must validate");
            let _ = expr.eval(&fields); // total: no panic
            let neg = FilterExpr::Not(Box::new(expr));
            assert!(neg.validate().is_ok());
            let _ = neg.eval(&fields);
        }
        // over-deep chain must be rejected by validate
        let mut deep = FilterExpr::Comparison {
            field: "num".into(),
            op: FilterOp::Eq,
            value: FilterValue::Int(1),
        };
        for _ in 0..(MAX_FILTER_DEPTH + 2) {
            deep = FilterExpr::Not(Box::new(deep));
        }
        assert!(deep.validate().is_err(), "over-deep filter must fail validation");
    }

    fn build_random(
        mut n: u64,
        leaf_fields: &[&str],
        mk_value: &impl Fn(u64) -> FilterValue,
        depth: usize,
    ) -> FilterExpr {
        n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if depth >= 3 {
            return FilterExpr::Comparison {
                field: leaf_fields[(n % leaf_fields.len() as u64) as usize].to_string(),
                op: match n % 6 {
                    0 => FilterOp::Eq,
                    1 => FilterOp::Ne,
                    2 => FilterOp::Gt,
                    3 => FilterOp::Gte,
                    4 => FilterOp::Lt,
                    _ => FilterOp::Lte,
                },
                value: mk_value(n >> 8),
            };
        }
        match n % 5 {
            0 => FilterExpr::And(
                Box::new(build_random(n ^ 1, leaf_fields, mk_value, depth + 1)),
                Box::new(build_random(n ^ 2, leaf_fields, mk_value, depth + 1)),
            ),
            1 => FilterExpr::Or(
                Box::new(build_random(n ^ 3, leaf_fields, mk_value, depth + 1)),
                Box::new(build_random(n ^ 4, leaf_fields, mk_value, depth + 1)),
            ),
            2 => FilterExpr::IsNull {
                field: leaf_fields[(n % leaf_fields.len() as u64) as usize].to_string(),
                negated: n % 2 == 0,
            },
            3 => FilterExpr::In {
                field: leaf_fields[(n % leaf_fields.len() as u64) as usize].to_string(),
                values: (0..3).map(|k| mk_value(n.wrapping_add(k))).collect(),
                negated: n % 2 == 1,
            },
            _ => FilterExpr::Comparison {
                field: leaf_fields[(n % leaf_fields.len() as u64) as usize].to_string(),
                op: match n % 6 {
                    0 => FilterOp::Eq,
                    1 => FilterOp::Ne,
                    2 => FilterOp::Gt,
                    3 => FilterOp::Gte,
                    4 => FilterOp::Lt,
                    _ => FilterOp::Lte,
                },
                value: mk_value(n >> 8),
            },
        }
    }
}
