//! Track metadata: the row stored in the library database.

use crate::error::DomainError;
use crate::key::Key;

/// Normalise a filesystem path for use as the library's primary key.
///
/// Comparisons must survive separator and case differences on Windows, where
/// the same file can arrive as `D:\Music\a.flac` or `d:/music/a.flac`.
pub fn normalize_track_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let unified = trimmed.replace('\\', "/");
    // Collapse repeated separators and resolve `.` components, without
    // touching the filesystem (the file may not exist yet).
    let mut parts: Vec<&str> = Vec::new();
    for part in unified.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if !matches!(parts.last(), None | Some(&"..")) {
                    parts.pop();
                } else {
                    parts.push("..");
                }
            }
            other => parts.push(other),
        }
    }
    let joined = parts.join("/");
    let absolute = unified.starts_with('/');
    let normalized = if absolute {
        format!("/{joined}")
    } else {
        joined
    };
    if cfg!(windows) {
        normalized.to_lowercase()
    } else {
        normalized
    }
}

/// Check that a uid is a canonical UUIDv4, as the library requires.
pub fn canonical_uid(value: &str) -> Result<String, DomainError> {
    let text = value.trim();
    let parsed = uuid::Uuid::parse_str(text)
        .map_err(|_| DomainError::InvalidUid(text.to_string()))?;
    if parsed.get_version_num() != 4 {
        return Err(DomainError::InvalidUid(text.to_string()));
    }
    Ok(parsed.hyphenated().to_string())
}

/// Generate a fresh track uid.
pub fn new_uid() -> String {
    uuid::Uuid::new_v4().hyphenated().to_string()
}

/// One row of the `tracks` table.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Track {
    /// Normalised path. The primary key.
    pub path: String,
    /// Links the row to its feature file. `None` for rows written before uids.
    pub uid: Option<String>,
    pub title: String,
    pub artist: String,
    pub album: String,
    pub bpm: Option<f64>,
    pub key: Option<Key>,
    pub duration: Option<f64>,
    pub total_samples: Option<i64>,
    pub rating: i32,
    /// Epoch seconds.
    pub added_ts: i64,
    pub comment: String,
    pub file_mtime: f64,
    pub file_size: i64,
}

impl Track {
    /// A new row for `path`, with a fresh uid and everything else empty.
    pub fn new(path: &str) -> Self {
        Self {
            path: normalize_track_path(path),
            uid: Some(new_uid()),
            title: String::new(),
            artist: String::new(),
            album: String::new(),
            bpm: None,
            key: None,
            duration: None,
            total_samples: None,
            rating: 0,
            added_ts: 0,
            comment: String::new(),
            file_mtime: 0.0,
            file_size: 0,
        }
    }

    /// The uid, or an error when the row has none.
    ///
    /// Rekordbox export derives its `TrackID` from the uid, and Python raises a
    /// bare `ValueError` from deep inside XML generation when it is missing,
    /// which aborts a whole library sync over one bad row.
    pub fn require_uid(&self) -> Result<&str, DomainError> {
        self.uid.as_deref().ok_or(DomainError::MissingUid)
    }

    /// Camelot label for the track key, or an empty string.
    pub fn key_label(&self) -> String {
        self.key.map(|k| k.camelot().to_string()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_separators_are_unified() {
        assert_eq!(normalize_track_path("a\\b\\c.flac"), "a/b/c.flac");
    }

    #[test]
    fn redundant_components_are_collapsed() {
        assert_eq!(normalize_track_path("a//b/./c.flac"), "a/b/c.flac");
        assert_eq!(normalize_track_path("a/b/../c.flac"), "a/c.flac");
        assert_eq!(normalize_track_path("/music/./x.wav"), "/music/x.wav");
    }

    #[test]
    fn leading_parent_components_are_kept() {
        assert_eq!(normalize_track_path("../a.flac"), "../a.flac");
        assert_eq!(normalize_track_path("../../a.flac"), "../../a.flac");
    }

    #[test]
    fn absolute_paths_keep_their_root() {
        assert!(normalize_track_path("/music/a.flac").starts_with('/'));
        assert!(!normalize_track_path("music/a.flac").starts_with('/'));
    }

    #[test]
    fn empty_and_blank_paths_normalize_to_empty() {
        assert_eq!(normalize_track_path(""), "");
        assert_eq!(normalize_track_path("   "), "");
    }

    #[test]
    fn normalization_is_idempotent() {
        let once = normalize_track_path("a\\b/../c//d.flac");
        assert_eq!(normalize_track_path(&once), once);
    }

    #[test]
    fn uid_round_trips_and_rejects_other_versions() {
        let uid = new_uid();
        assert_eq!(canonical_uid(&uid).unwrap(), uid);
        assert_eq!(canonical_uid(&format!("  {uid}  ")).unwrap(), uid);

        // A v1 UUID must be refused: the library assumes v4 everywhere.
        assert!(matches!(
            canonical_uid("2c5ea4c0-4067-11e9-8bad-9b1deb4d3b7d"),
            Err(DomainError::InvalidUid(_))
        ));
        assert!(canonical_uid("not-a-uuid").is_err());
        assert!(canonical_uid("").is_err());
    }

    #[test]
    fn generated_uids_are_canonical_and_distinct() {
        let a = new_uid();
        let b = new_uid();
        assert_ne!(a, b);
        assert!(canonical_uid(&a).is_ok());
    }

    #[test]
    fn a_new_track_normalizes_its_path_and_gets_a_uid() {
        let track = Track::new("Music\\Song.flac");
        assert_eq!(track.path, "Music/Song.flac");
        assert!(track.require_uid().is_ok());
    }

    #[test]
    fn a_row_without_a_uid_reports_a_typed_error() {
        let mut track = Track::new("a.flac");
        track.uid = None;
        assert_eq!(track.require_uid(), Err(DomainError::MissingUid));
    }

    #[test]
    fn key_label_is_camelot_or_empty() {
        let mut track = Track::new("a.flac");
        assert_eq!(track.key_label(), "");
        track.key = Some(Key::from_index(21));
        assert_eq!(track.key_label(), "8A");
    }
}
