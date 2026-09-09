//! Mixlyzer persistence: the SQLite track library, the per-track feature store
//! and the schema migration runner.
//!
//! A library on disk is a directory holding:
//!
//! ```text
//! library/
//!   VERSION            the schema version this directory is in
//!   library.db         tracks and their BPM / key segments
//!   <uid>.mxf          one feature file per track
//! ```
//!
//! This crate is a reimplementation of `core/library_handler.py`,
//! `core/analysis_lib_handler.py` and `migration/`. It reads and writes the
//! same SQLite schema, so an existing `library.db` opens unchanged; the feature
//! container is replaced by a format of our own ([`features`]) because the
//! Python one is an NPZ zip whose failure modes cannot be reported.
//!
//! What is different, in one line each, with the detail in each module:
//!
//! * Every fallible operation returns [`StoreError`]. Python catches no
//!   `sqlite3` or `zipfile` exception anywhere, so a damaged library kills
//!   startup.
//! * A migration step returns a *count inside a success value*, never an int
//!   that the runner might read as a status code — the bug that makes every
//!   non-empty Python library fail migration on every launch.
//! * A library version that cannot be read is an error, not a silent
//!   "assume the oldest version and re-run everything".

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

pub mod error;
pub mod features;
pub mod library;
pub mod migration;

pub use error::StoreError;
pub use features::{FeatureFile, FeatureStore, FeatureValue};
pub use library::{safe_order_by, Library, Transition, TransitionSide};
pub use migration::{MigrationOutcome, MigrationReport, SkippedTrack, Step};

#[cfg(test)]
mod testutil;
