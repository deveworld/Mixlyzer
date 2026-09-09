//! Mixlyzer domain types and pure analysis logic.
//!
//! This crate holds everything that describes a track's musical structure —
//! beatgrid, key, phrases, cue points, JumpCUEs — plus the configuration
//! schema. It has no audio, no database and no user interface, so it can be
//! exercised entirely from tests.
//!
//! It is a reimplementation of the Python `core/`, `utils/` and
//! `analyzer_core/` pure-logic modules. Where the two deliberately differ, the
//! module documentation says so and a test pins the new behaviour.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod beatgrid;
pub mod config;
pub mod cue;
pub mod error;
pub mod jumpcue;
pub mod key;
pub mod linear;
pub mod phrase;
pub mod segments;
pub mod track;

pub use beatgrid::{BarBeat, Beatgrid};
pub use config::{AnalysisConfig, ChromaMethod, Config, KeyConfig, LibraryConfig};
pub use cue::CuePoint;
pub use error::{ConfigError, DomainError};
pub use jumpcue::{Direction, JumpCue, JumpCueGraph, JumpLink};
pub use key::{Key, Mode};
pub use linear::{BpmSegmentRow, KeySegmentRow};
pub use phrase::{FillDirection, FillMarker, Phrase};
pub use segments::{KeySegment, TempoSegment};
pub use track::Track;

/// The library schema version this build reads and writes.
pub const LIBRARY_VERSION: &str = "0.3.0";
