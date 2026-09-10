//! The process denylist: which programs Mixlyzer must never attach to.
//!
//! `process_denylist.json` lists process names (`lsass`, `1password`,
//! `easyanticheat`, …), executable path keywords (`\windows\system32\`,
//! `\crowdstrike\`, …) and company keywords. Matching is case-insensitive and
//! ignores a `.exe` suffix, so `LSASS.EXE` and `lsass` are the same target.
//!
//! The fail-closed behaviour of the Python version is kept exactly: if the
//! denylist cannot be read or parsed, *every* process is blocked. A missing
//! denylist must never mean "attach to anything". The failure is not cached, so
//! a file that was briefly locked recovers on the next poll — that part of
//! `_load_process_denylist_payload` was already right and is pinned by a test
//! here.
//!
//! `blocked_company_keywords` is dead weight in the Python code: nothing ever
//! reads the key. It is implemented here as [`Denylist::denies_company`] and is
//! consulted whenever a [`ProcessIdentity`] carries a company name. The Windows
//! backend in this crate does not read the executable's version resource, so in
//! practice the field is only used by callers that supply one; the alternative
//! was to delete the third of the denylist file that exists to catch renamed
//! security software.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::error::SyncError;
use crate::reader::ProcessIdentity;

/// Normalise a process name for comparison: trimmed, lowercased, `.exe` gone.
pub fn normalize_process_name(name: &str) -> String {
    let text = name.trim().to_lowercase();
    text.strip_suffix(".exe").unwrap_or(&text).to_string()
}

/// Normalise an executable path for keyword matching.
///
/// Keywords in the file are written Windows-style (`\windows\system32\`), so
/// forward slashes are folded onto backslashes.
pub fn normalize_image_path(path: &str) -> String {
    path.trim().to_lowercase().replace('/', "\\")
}

/// Why a process was blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenyReason {
    /// The process name is listed in `blocked_process_names`.
    Name(String),
    /// The executable path contains a listed keyword.
    PathKeyword(String),
    /// The executable's company contains a listed keyword.
    CompanyKeyword(String),
}

impl std::fmt::Display for DenyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenyReason::Name(name) => write!(f, "process name {name:?} is on the denylist"),
            DenyReason::PathKeyword(kw) => {
                write!(f, "executable path contains the blocked keyword {kw:?}")
            }
            DenyReason::CompanyKeyword(kw) => {
                write!(f, "executable company contains the blocked keyword {kw:?}")
            }
        }
    }
}

/// A parsed `process_denylist.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Denylist {
    names: BTreeSet<String>,
    path_keywords: Vec<String>,
    company_keywords: Vec<String>,
}

impl Denylist {
    /// A denylist that blocks nothing. Only for tests and for callers that
    /// deliberately supply their own list.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a list from already-known entries.
    pub fn from_entries<I, J, K>(names: I, path_keywords: J, company_keywords: K) -> Self
    where
        I: IntoIterator<Item = String>,
        J: IntoIterator<Item = String>,
        K: IntoIterator<Item = String>,
    {
        Self {
            names: names
                .into_iter()
                .map(|n| normalize_process_name(&n))
                .filter(|n| !n.is_empty())
                .collect(),
            path_keywords: path_keywords
                .into_iter()
                .map(|k| normalize_image_path(&k))
                .filter(|k| !k.is_empty())
                .collect(),
            company_keywords: company_keywords
                .into_iter()
                .map(|k| k.trim().to_lowercase())
                .filter(|k| !k.is_empty())
                .collect(),
        }
    }

    /// Parse the JSON document.
    ///
    /// Leniently, as Python does: a payload that is not an object, or a list
    /// that holds a stray number, yields the entries that *are* usable rather
    /// than blocking everything. Only unreadable or syntactically invalid JSON
    /// is a failure, and that failure is what fails closed.
    pub fn from_json(text: &str) -> Result<Self, SyncError> {
        let payload: serde_json::Value = serde_json::from_str(text)
            .map_err(|err| SyncError::DenylistUnavailable(err.to_string()))?;
        let strings = |key: &str| -> Vec<String> {
            payload
                .get(key)
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(Self::from_entries(
            strings("blocked_process_names"),
            strings("blocked_path_keywords"),
            strings("blocked_company_keywords"),
        ))
    }

    /// Read and parse the denylist file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SyncError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|err| SyncError::DenylistUnavailable(format!("{}: {err}", path.display())))?;
        Self::from_json(&text)
    }

    /// Whether a process name is listed.
    ///
    /// An empty name matches nothing, as in Python: the name is unknown, and
    /// the path and company checks still apply.
    pub fn denies_process_name(&self, name: &str) -> bool {
        let target = normalize_process_name(name);
        !target.is_empty() && self.names.contains(&target)
    }

    /// Whether an executable path contains a blocked keyword.
    pub fn denies_image_path(&self, image_path: &str) -> Option<&str> {
        let target = normalize_image_path(image_path);
        if target.is_empty() {
            return None;
        }
        self.path_keywords
            .iter()
            .find(|kw| target.contains(kw.as_str()))
            .map(String::as_str)
    }

    /// Whether a company name contains a blocked keyword.
    ///
    /// This is `blocked_company_keywords`, which the Python code never reads.
    pub fn denies_company(&self, company: &str) -> Option<&str> {
        let target = company.trim().to_lowercase();
        if target.is_empty() {
            return None;
        }
        self.company_keywords
            .iter()
            .find(|kw| target.contains(kw.as_str()))
            .map(String::as_str)
    }

    /// Check a whole identity, name first.
    pub fn check(&self, identity: &ProcessIdentity) -> Option<DenyReason> {
        if self.denies_process_name(&identity.name) {
            return Some(DenyReason::Name(normalize_process_name(&identity.name)));
        }
        if let Some(kw) = self.denies_image_path(&identity.image_path) {
            return Some(DenyReason::PathKeyword(kw.to_string()));
        }
        self.denies_company(&identity.company)
            .map(|kw| DenyReason::CompanyKeyword(kw.to_string()))
    }

    /// Number of blocked process names, for diagnostics.
    pub fn name_count(&self) -> usize {
        self.names.len()
    }
}

/// Where a [`DenylistGuard`] gets its list from.
#[derive(Debug, Clone)]
enum DenylistSource {
    File(PathBuf),
    Fixed,
}

/// Caches a successfully loaded denylist and re-reads after a failure.
///
/// The cache is one-way on purpose: a *successful* load is remembered for the
/// life of the guard (the file does not change while the app runs), a
/// *failure* is not, so a transient read error recovers on the next poll
/// instead of blocking the feature forever.
#[derive(Debug, Clone)]
pub struct DenylistGuard {
    source: DenylistSource,
    cached: Option<Denylist>,
}

impl DenylistGuard {
    /// Load lazily from `process_denylist.json`.
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            source: DenylistSource::File(path.into()),
            cached: None,
        }
    }

    /// Use a list that is already in hand.
    pub fn fixed(list: Denylist) -> Self {
        Self {
            source: DenylistSource::Fixed,
            cached: Some(list),
        }
    }

    /// The list, loading it if this is the first successful attempt.
    pub fn denylist(&mut self) -> Result<&Denylist, SyncError> {
        if self.cached.is_none() {
            match &self.source {
                DenylistSource::File(path) => self.cached = Some(Denylist::load(path)?),
                DenylistSource::Fixed => {
                    return Err(SyncError::DenylistUnavailable(
                        "no denylist was supplied".into(),
                    ))
                }
            }
        }
        Ok(self.cached.as_ref().expect("just loaded"))
    }

    /// Fail-closed check of a process identity.
    ///
    /// `Err(SyncError::DenylistUnavailable)` when the list cannot be read: the
    /// caller must treat that as "blocked", which is what the engine does.
    pub fn check(&mut self, identity: &ProcessIdentity) -> Result<(), SyncError> {
        let list = self.denylist()?;
        match list.check(identity) {
            None => Ok(()),
            Some(reason) => Err(SyncError::ProcessDenied {
                name: if identity.name.trim().is_empty() {
                    identity.image_path.clone()
                } else {
                    identity.name.clone()
                },
                reason: reason.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mixlyzer-denylist-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample() -> Denylist {
        Denylist::from_entries(
            ["lsass".to_string(), "1Password.exe".to_string()],
            ["\\windows\\system32\\".to_string()],
            ["crowdstrike".to_string()],
        )
    }

    #[test]
    fn names_match_case_insensitively_and_ignore_the_exe_suffix() {
        let list = sample();
        assert!(list.denies_process_name("LSASS.EXE"));
        assert!(list.denies_process_name("  lsass  "));
        assert!(list.denies_process_name("1password"));
        assert!(!list.denies_process_name("rekordbox.exe"));
    }

    #[test]
    fn an_unknown_name_is_allowed() {
        assert!(!sample().denies_process_name(""));
        assert!(!sample().denies_process_name("serato.exe"));
    }

    #[test]
    fn path_keywords_match_anywhere_and_fold_slashes() {
        let list = sample();
        assert_eq!(
            list.denies_image_path("C:/Windows/System32/lsass.exe"),
            Some("\\windows\\system32\\")
        );
        assert_eq!(list.denies_image_path("D:\\DJ\\rekordbox.exe"), None);
        assert_eq!(list.denies_image_path(""), None);
    }

    /// `blocked_company_keywords` is present in the JSON and read by nothing in
    /// the Python code. Here it works.
    #[test]
    fn company_keywords_are_implemented_not_ignored() {
        let list = sample();
        assert_eq!(
            list.denies_company("CrowdStrike, Inc."),
            Some("crowdstrike")
        );
        assert_eq!(list.denies_company("AlphaTheta Corporation"), None);
    }

    #[test]
    fn check_reports_the_first_matching_rule() {
        let list = sample();
        let denied = ProcessIdentity {
            pid: 4,
            name: "lsass.exe".into(),
            image_path: "C:\\Windows\\System32\\lsass.exe".into(),
            company: String::new(),
        };
        assert_eq!(list.check(&denied), Some(DenyReason::Name("lsass".into())));

        let by_path = ProcessIdentity {
            name: "renamed.exe".into(),
            image_path: "C:\\Windows\\System32\\renamed.exe".into(),
            ..ProcessIdentity::default()
        };
        assert!(matches!(
            list.check(&by_path),
            Some(DenyReason::PathKeyword(_))
        ));

        let by_company = ProcessIdentity {
            name: "sensor.exe".into(),
            image_path: "D:\\tools\\sensor.exe".into(),
            company: "CrowdStrike".into(),
            ..ProcessIdentity::default()
        };
        assert!(matches!(
            list.check(&by_company),
            Some(DenyReason::CompanyKeyword(_))
        ));

        let allowed = ProcessIdentity {
            name: "rekordbox.exe".into(),
            image_path: "D:\\Program Files\\rekordbox\\rekordbox.exe".into(),
            company: "AlphaTheta".into(),
            ..ProcessIdentity::default()
        };
        assert_eq!(list.check(&allowed), None);
    }

    #[test]
    fn the_shipped_denylist_parses() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../process_denylist.json");
        let list = Denylist::load(&path).expect("the shipped denylist must load");
        assert!(list.name_count() > 100, "got {}", list.name_count());
        assert!(list.denies_process_name("lsass"));
        assert!(list
            .denies_image_path("c:\\windows\\system32\\x.exe")
            .is_some());
        assert!(list.denies_company("Kaspersky Lab").is_some());
    }

    #[test]
    fn a_payload_that_is_not_an_object_yields_an_empty_list() {
        let list = Denylist::from_json("[1, 2, 3]").unwrap();
        assert_eq!(list.name_count(), 0);
    }

    #[test]
    fn stray_non_string_entries_are_skipped_not_fatal() {
        let list =
            Denylist::from_json(r#"{"blocked_process_names":["lsass", 7, null, ""]}"#).unwrap();
        assert_eq!(list.name_count(), 1);
        assert!(list.denies_process_name("lsass"));
    }

    #[test]
    fn a_missing_denylist_file_blocks_everything() {
        let dir = temp_dir("missing");
        let mut guard = DenylistGuard::from_file(dir.join("nope.json"));
        let err = guard
            .check(&ProcessIdentity {
                name: "rekordbox.exe".into(),
                ..ProcessIdentity::default()
            })
            .unwrap_err();
        assert!(
            matches!(err, SyncError::DenylistUnavailable(_)),
            "fail closed: {err:?}"
        );
    }

    #[test]
    fn malformed_json_blocks_everything() {
        let dir = temp_dir("malformed");
        let path = dir.join("process_denylist.json");
        std::fs::write(&path, "{ not json").unwrap();
        let mut guard = DenylistGuard::from_file(&path);
        assert!(matches!(
            guard.denylist().unwrap_err(),
            SyncError::DenylistUnavailable(_)
        ));
    }

    /// A read failure must not be cached, or a file that was locked for one
    /// poll would block the feature for the rest of the session.
    #[test]
    fn a_transient_read_failure_recovers_on_the_next_poll() {
        let dir = temp_dir("transient");
        let path = dir.join("process_denylist.json");
        let mut guard = DenylistGuard::from_file(&path);
        assert!(guard.denylist().is_err(), "not written yet");

        std::fs::write(&path, r#"{"blocked_process_names":["lsass"]}"#).unwrap();
        assert!(guard.denylist().is_ok(), "the retry must load the file");
        assert!(guard
            .check(&ProcessIdentity {
                name: "rekordbox.exe".into(),
                ..ProcessIdentity::default()
            })
            .is_ok());
    }

    /// The reverse: a list that loaded is not re-read on every poll.
    #[test]
    fn a_successful_load_is_cached() {
        let dir = temp_dir("cached");
        let path = dir.join("process_denylist.json");
        std::fs::write(&path, r#"{"blocked_process_names":["lsass"]}"#).unwrap();
        let mut guard = DenylistGuard::from_file(&path);
        assert_eq!(guard.denylist().unwrap().name_count(), 1);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            guard.denylist().unwrap().name_count(),
            1,
            "the cached list must survive the file going away"
        );
    }

    #[test]
    fn a_denied_process_names_the_rule_in_the_error() {
        let mut guard = DenylistGuard::fixed(sample());
        let err = guard
            .check(&ProcessIdentity {
                name: "lsass.exe".into(),
                ..ProcessIdentity::default()
            })
            .unwrap_err();
        assert!(err.to_string().contains("denylist"), "{err}");
        assert!(err.is_permanent());
    }
}
