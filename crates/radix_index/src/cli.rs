//! Flag parsing shared by the service and bridge binaries.
//!
//! Deliberately strict: an unknown flag, a missing value, or an
//! unparsable value is a startup error, never a silent fallback to a
//! default. The bridge's `--block-size` decides the keyspace key, so a
//! typo that fell back to the default would feed the whole event stream
//! into a keyspace no gateway queries — a quietly degraded hit rate
//! with no other symptom (audit finding: a typo'd `--bootstrap-from`
//! once meant silently-cold restarts forever).
//!
//! Both `--flag value` and `--flag=value` are accepted; the latter is
//! the form Kubernetes manifests conventionally use (see
//! `deploy/statefulset.yaml`).

/// Split `--flag=value` into `("--flag", Some("value"))`; a bare flag
/// keeps its `None`.
fn split_eq(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((flag, value)) => (flag, Some(value)),
        None => (arg, None),
    }
}

/// Every `--` argument must be a known flag, and every flag must carry a
/// value (inline or as the next argument). Call before any `parse_flag`.
pub fn validate_flags(args: &[String], known: &[&str]) {
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        assert!(arg.starts_with("--"), "unexpected argument {arg}");
        let (flag, inline) = split_eq(arg);
        assert!(
            known.contains(&flag),
            "unknown flag {flag}; known flags: {known:?}"
        );
        if inline.is_some() {
            i += 1;
        } else {
            assert!(
                i + 1 < args.len() && !args[i + 1].starts_with("--"),
                "flag {flag} is missing its value"
            );
            i += 2;
        }
    }
}

/// The value of `flag`, parsed; `None` when the flag is absent. An
/// unparsable value panics with the flag name and the offending text.
pub fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let mut i = 1;
    while i < args.len() {
        let (name, inline) = split_eq(&args[i]);
        if name == flag {
            let value = match inline {
                Some(v) => v.to_string(),
                None => args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| panic!("flag {flag} is missing its value")),
            };
            return Some(
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("flag {flag} has an unparsable value: {value:?}")),
            );
        }
        i += if inline.is_some() { 1 } else { 2 };
    }
    None
}

/// Comma-separated list value (trimmed, empties dropped); empty when
/// the flag is absent.
pub fn parse_list(args: &[String], flag: &str) -> Vec<String> {
    parse_flag::<String>(args, flag)
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(parts: &[&str]) -> Vec<String> {
        std::iter::once("bin")
            .chain(parts.iter().copied())
            .map(str::to_string)
            .collect()
    }

    const KNOWN: &[&str] = &["--port", "--peers", "--block-size"];

    #[test]
    fn both_value_forms_parse_identically() {
        let spaced = argv(&["--port", "40000", "--peers", "a,b"]);
        let inline = argv(&["--port=40000", "--peers=a,b"]);
        validate_flags(&spaced, KNOWN);
        validate_flags(&inline, KNOWN);
        assert_eq!(parse_flag::<u16>(&spaced, "--port"), Some(40000));
        assert_eq!(parse_flag::<u16>(&inline, "--port"), Some(40000));
        assert_eq!(parse_list(&spaced, "--peers"), vec!["a", "b"]);
        assert_eq!(parse_list(&inline, "--peers"), vec!["a", "b"]);
        assert_eq!(parse_flag::<u32>(&inline, "--block-size"), None);
    }

    #[test]
    #[should_panic(expected = "unparsable value")]
    fn unparsable_value_is_a_startup_error_not_a_default() {
        parse_flag::<u32>(&argv(&["--block-size", "128x"]), "--block-size");
    }

    #[test]
    #[should_panic(expected = "unknown flag --blocksize")]
    fn misspelled_flag_is_rejected() {
        validate_flags(&argv(&["--blocksize", "256"]), KNOWN);
    }

    #[test]
    #[should_panic(expected = "missing its value")]
    fn trailing_flag_without_value_is_rejected() {
        validate_flags(&argv(&["--port"]), KNOWN);
    }
}
