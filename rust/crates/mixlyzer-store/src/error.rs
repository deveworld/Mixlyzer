//! Typed errors for every fallible persistence operation.
//!
//! The Python implementation never catches a single `sqlite3` exception: a
//! corrupt, locked or unreadable `library.db` propagates out of
//! `LibraryDB.connect()` past `app.main` and kills startup with a traceback and
//! no way to pick another library. The feature store is the same story with
//! `zipfile.BadZipFile` / `EOFError` / `zlib.error` — one damaged `.npz` makes a
//! track permanently unloadable rather than merely un-analysed.
//!
//! Everything in this crate returns [`StoreError`] instead. Every variant names
//! the file it is about, so a caller can put the offending path in front of the
//! user.

use std::path::{Path, PathBuf};

/// Anything that can go wrong while reading or writing the library.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    /// The database file could not be opened at all (missing directory,
    /// permissions, unsupported URI).
    #[error("could not open library database {path}: {source}")]
    Open {
        /// The database file that could not be opened.
        path: PathBuf,
        /// What SQLite reported.
        #[source]
        source: rusqlite::Error,
    },

    /// SQLite rejected the file as not a database, or reported corruption.
    ///
    /// This is the variant a garbage or truncated `library.db` produces, and
    /// the one a caller should offer "choose another library" for.
    #[error("library database {path} is damaged or is not a database: {detail}")]
    Corrupt {
        /// The damaged database.
        path: PathBuf,
        /// SQLite's description of the damage.
        detail: String,
    },

    /// Another process (or another connection) holds the database.
    #[error("library database {path} is locked by another process: {detail}")]
    Busy {
        /// The locked database.
        path: PathBuf,
        /// SQLite's description of the lock.
        detail: String,
    },

    /// A statement failed for a reason that is neither corruption nor a lock.
    #[error("library query failed ({context}): {source}")]
    Query {
        /// What the crate was doing, e.g. `"upsert track"`.
        context: String,
        /// What SQLite reported.
        #[source]
        source: rusqlite::Error,
    },

    /// A schema change against an existing library failed.
    #[error("schema migration failed for {path} ({context}): {source}")]
    Schema {
        /// The database whose schema was being changed.
        path: PathBuf,
        /// The change being attempted, e.g. `"add tracks.total_samples"`.
        context: String,
        /// What SQLite reported.
        #[source]
        source: rusqlite::Error,
    },

    /// Plain filesystem failure: reading, writing or listing a file.
    #[error("could not access {path}: {source}")]
    Io {
        /// The file or directory involved.
        path: PathBuf,
        /// What the operating system reported.
        #[source]
        source: std::io::Error,
    },

    /// A feature file exists but does not parse: bad magic, truncation, a
    /// checksum mismatch, an unknown value kind, invalid UTF-8.
    #[error("feature file {path} is corrupt: {detail}")]
    FeatureFormat {
        /// The unparseable feature file.
        path: PathBuf,
        /// Where and how parsing gave up.
        detail: String,
    },

    /// A feature file was written by a newer build of Mixlyzer.
    #[error(
        "feature file {path} has format version {found}, but this build reads at most {supported}"
    )]
    FeatureVersion {
        /// The feature file from the future.
        path: PathBuf,
        /// The format version the file declares.
        found: u16,
        /// The newest format version this build understands.
        supported: u16,
    },

    /// A feature file was expected but is not there.
    #[error("no feature file for track {uid} at {path}")]
    FeatureMissing {
        /// The track that has no features stored.
        uid: String,
        /// Where the file would have been.
        path: PathBuf,
    },

    /// The `VERSION` file exists but cannot be read or is not valid text.
    ///
    /// Python maps *any* read failure to the oldest version, so a 0.3.0 library
    /// with an unreadable `VERSION` silently re-runs the whole chain. See
    /// [`crate::migration::read_library_version`].
    #[error("library version file {path} is unreadable: {detail}")]
    UnreadableVersion {
        /// The `VERSION` file.
        path: PathBuf,
        /// Why it could not be read.
        detail: String,
    },

    /// No chain of migration steps connects two versions, or the chain loops.
    #[error("cannot migrate library from {from} to {to}: {detail}")]
    NoMigrationPath {
        /// The version the library is at.
        from: String,
        /// The version it was asked to reach.
        to: String,
        /// Where the chain broke, e.g. the version with no outgoing step.
        detail: String,
    },

    /// A migration step hit something it cannot work around.
    ///
    /// Per-track problems are *not* this: they are reported as skips in
    /// [`crate::migration::MigrationOutcome`].
    #[error("migration {from} -> {to} failed: {detail}")]
    MigrationFailed {
        /// The step's source version.
        from: String,
        /// The step's target version.
        to: String,
        /// What made further work impossible.
        detail: String,
    },

    /// A uid was not a canonical UUIDv4, or a row lacked one entirely.
    #[error(transparent)]
    Domain(#[from] mixlyzer_core::DomainError),
}

impl StoreError {
    /// Classify a `rusqlite` error against the database it came from.
    ///
    /// Corruption and lock contention are the two failures a user can act on,
    /// so they get their own variants instead of being flattened into one
    /// opaque "database error" the way Python flattens everything into an
    /// uncaught `sqlite3.DatabaseError`.
    pub(crate) fn from_sqlite(path: &Path, context: &str, source: rusqlite::Error) -> Self {
        use rusqlite::ErrorCode;
        if let rusqlite::Error::SqliteFailure(inner, ref message) = source {
            let detail = message
                .clone()
                .unwrap_or_else(|| format!("sqlite error code {:?}", inner.code));
            match inner.code {
                ErrorCode::NotADatabase | ErrorCode::DatabaseCorrupt => {
                    return StoreError::Corrupt {
                        path: path.to_path_buf(),
                        detail,
                    }
                }
                ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked => {
                    return StoreError::Busy {
                        path: path.to_path_buf(),
                        detail,
                    }
                }
                _ => {}
            }
        }
        StoreError::Query {
            context: context.to_string(),
            source,
        }
    }

    /// Build an [`StoreError::Io`] for `path`.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        StoreError::Io {
            path: path.into(),
            source,
        }
    }

    /// The file this error is about, when it is about one.
    ///
    /// Lets a caller say "library.db is damaged" without matching every
    /// variant.
    pub fn path(&self) -> Option<&Path> {
        match self {
            StoreError::Open { path, .. }
            | StoreError::Corrupt { path, .. }
            | StoreError::Busy { path, .. }
            | StoreError::Schema { path, .. }
            | StoreError::Io { path, .. }
            | StoreError::FeatureFormat { path, .. }
            | StoreError::FeatureVersion { path, .. }
            | StoreError::FeatureMissing { path, .. }
            | StoreError::UnreadableVersion { path, .. } => Some(path),
            StoreError::Query { .. }
            | StoreError::NoMigrationPath { .. }
            | StoreError::MigrationFailed { .. }
            | StoreError::Domain(_) => None,
        }
    }

    /// Whether the underlying file is damaged, as opposed to merely absent,
    /// busy or rejected.
    pub fn is_corruption(&self) -> bool {
        matches!(
            self,
            StoreError::Corrupt { .. }
                | StoreError::FeatureFormat { .. }
                | StoreError::FeatureVersion { .. }
        )
    }
}
