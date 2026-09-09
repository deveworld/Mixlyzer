//! Typed errors for the core layer.
//!
//! The Python implementation lets `OSError` / `sqlite3.Error` escape from
//! `load_cfg()` and `LibraryDB.connect()` all the way past `app.main`, which
//! makes an unreachable library path or a damaged `library.db` an
//! unrecoverable startup crash. Every fallible operation here returns a typed
//! error instead, so callers can report the problem and offer a way out.

use std::path::PathBuf;

/// Failure modes when loading or materialising configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read config file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("config file {path} is not valid JSON: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("could not write config file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The library directory could not be created or is not usable.
    ///
    /// Python raises a bare `OSError` from `_ensure_library_dir` outside the
    /// try/except in `load_cfg`, which bricks startup with no way to pick a
    /// different folder. Callers of [`crate::config::Config::ensure_library_dir`]
    /// get this instead and can prompt for a new path.
    #[error("library path {path} is not usable: {source}")]
    LibraryPath {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Failure modes for domain-level validation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DomainError {
    #[error("selection end ({end}) must be greater than start ({start})")]
    EmptySelection { start: String, end: String },

    #[error("phrase label must not be empty")]
    EmptyPhraseLabel,

    #[error("jump cue label {0:?} is not a single letter A-Z")]
    InvalidJumpCueLabel(String),

    #[error("duplicate jump cue label {0:?}")]
    DuplicateJumpCueLabel(String),

    #[error("track uid is required for this operation")]
    MissingUid,

    #[error("{0:?} is not a canonical UUIDv4")]
    InvalidUid(String),
}
