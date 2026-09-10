//! Validating the track path the external program reports.
//!
//! The path comes out of another process's memory, so it is treated as
//! untrusted input: it must be absolute, must not be a UNC share, and must name
//! a file that exists before Mixlyzer will act on it. Comparison uses
//! [`mixlyzer_core::track::normalize_track_path`], the same normalisation the
//! library database uses as its primary key, so "is this the track we already
//! have loaded?" gives the same answer here as it does there.
//!
//! One deliberate difference from Python: results are not cached.
//! `_validate_external_track_path` memoises every answer, including the
//! failures, for the life of the controller — so a track that had not finished
//! copying when it was first seen can never be followed, and a file that has
//! since been deleted or unplugged keeps being loaded. Validation here is a
//! `stat` per poll, which is cheap next to reading another process's memory.

use std::path::Path;

/// A path that passed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPath {
    /// The path as reported, trimmed. What gets handed to the loader.
    pub raw: String,
    /// The normalised form, for comparison against the library.
    pub normalized: String,
}

/// Why a reported path was not usable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathRejection {
    /// The deck reports nothing at all.
    #[error("no path is reported")]
    Empty,

    /// `\\server\share\...`: reading it can hang on a dead host, and it is
    /// never a local DJ library path.
    #[error("{0:?} is a UNC network path")]
    Unc(String),

    /// A relative path has no meaning outside the other program's working
    /// directory.
    #[error("{0:?} is not an absolute path")]
    Relative(String),

    /// Nothing is at that path right now.
    #[error("{0:?} does not exist")]
    Missing(String),

    /// The path names a directory or a device, not a track.
    #[error("{0:?} is not a regular file")]
    NotAFile(String),
}

/// Whether a path is absolute in either Windows or POSIX form.
///
/// `std::path::Path::is_absolute` answers for the *host*, so a Windows path
/// read out of a DJ program looks relative when the tests run on Linux, and a
/// POSIX path would look relative on Windows. Both forms are accepted here, and
/// the caller's filesystem decides whether the file is really there.
pub fn is_absolute_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    if path.starts_with('/') || path.starts_with('\\') {
        return true;
    }
    // `C:\...` or `C:/...`
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Whether a path names a UNC share.
pub fn is_unc_path(path: &str) -> bool {
    path.starts_with("\\\\") || path.starts_with("//")
}

/// Check a path reported by the external program.
pub fn validate_track_path(reported: &str) -> Result<ValidatedPath, PathRejection> {
    let raw = reported.trim();
    if raw.is_empty() {
        return Err(PathRejection::Empty);
    }
    if is_unc_path(raw) {
        return Err(PathRejection::Unc(raw.to_string()));
    }
    if !is_absolute_path(raw) {
        return Err(PathRejection::Relative(raw.to_string()));
    }
    let on_disk = Path::new(raw);
    let meta = std::fs::metadata(on_disk).map_err(|_| PathRejection::Missing(raw.to_string()))?;
    if !meta.is_file() {
        return Err(PathRejection::NotAFile(raw.to_string()));
    }
    Ok(ValidatedPath {
        raw: raw.to_string(),
        normalized: mixlyzer_core::track::normalize_track_path(raw),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mixlyzer-syncpath-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_existing_file_validates_and_normalises() {
        let dir = temp_dir("ok");
        let file = dir.join("track.flac");
        std::fs::write(&file, b"x").unwrap();
        let reported = file.to_string_lossy().to_string();

        let validated = validate_track_path(&reported).unwrap();
        assert_eq!(validated.raw, reported);
        assert_eq!(
            validated.normalized,
            mixlyzer_core::track::normalize_track_path(&reported)
        );
    }

    #[test]
    fn surrounding_whitespace_is_trimmed() {
        let dir = temp_dir("trim");
        let file = dir.join("track.flac");
        std::fs::write(&file, b"x").unwrap();
        let padded = format!("  {}  ", file.to_string_lossy());
        assert_eq!(
            validate_track_path(&padded).unwrap().raw,
            file.to_string_lossy()
        );
    }

    #[test]
    fn an_empty_path_is_rejected() {
        assert_eq!(validate_track_path("   "), Err(PathRejection::Empty));
    }

    #[test]
    fn unc_paths_are_rejected_in_both_spellings() {
        assert!(matches!(
            validate_track_path("\\\\nas\\music\\a.flac"),
            Err(PathRejection::Unc(_))
        ));
        assert!(matches!(
            validate_track_path("//nas/music/a.flac"),
            Err(PathRejection::Unc(_))
        ));
    }

    #[test]
    fn relative_paths_are_rejected() {
        assert!(matches!(
            validate_track_path("music/a.flac"),
            Err(PathRejection::Relative(_))
        ));
        assert!(matches!(
            validate_track_path("..\\a.flac"),
            Err(PathRejection::Relative(_))
        ));
    }

    #[test]
    fn a_windows_path_is_absolute_even_when_the_tests_run_on_linux() {
        assert!(is_absolute_path("D:\\Music\\a.flac"));
        assert!(is_absolute_path("d:/Music/a.flac"));
        assert!(is_absolute_path("/home/dj/a.flac"));
        assert!(!is_absolute_path("D:a.flac"));
        assert!(!is_absolute_path("a.flac"));
        // Absolute, but caught earlier as UNC.
        assert!(is_absolute_path("\\\\nas\\a.flac"));
    }

    #[test]
    fn a_path_that_does_not_exist_is_rejected() {
        let dir = temp_dir("missing");
        let file = dir.join("nope.flac");
        assert!(matches!(
            validate_track_path(&file.to_string_lossy()),
            Err(PathRejection::Missing(_))
        ));
    }

    #[test]
    fn a_directory_is_not_a_track() {
        let dir = temp_dir("dir");
        assert!(matches!(
            validate_track_path(&dir.to_string_lossy()),
            Err(PathRejection::NotAFile(_))
        ));
    }

    /// Python caches the answer forever, so a file that appears later can never
    /// be followed and one that vanishes keeps being loaded.
    #[test]
    fn validation_follows_the_filesystem_rather_than_a_cache() {
        let dir = temp_dir("vanish");
        let file = dir.join("track.flac");
        let reported = file.to_string_lossy().to_string();

        assert!(validate_track_path(&reported).is_err(), "not there yet");
        std::fs::write(&file, b"x").unwrap();
        assert!(validate_track_path(&reported).is_ok(), "now it is");
        std::fs::remove_file(&file).unwrap();
        assert!(validate_track_path(&reported).is_err(), "and now it is not");
    }
}
