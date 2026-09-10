//! Audio decoding and the Mixlyzer analysis pipeline.
//!
//! This crate turns an audio file into the musical description the rest of the
//! app works with: where the beats are, how the tempo moves, and what key the
//! music is in. It reimplements the Python `analyzer_core` pipeline, with two
//! structural differences.
//!
//! Decoding is in-process rather than a pipe from an FFmpeg subprocess, so no
//! external binary has to be installed or found, and failures arrive as typed
//! errors instead of a printed traceback. And every stage reports what it could
//! not do: silence, an over-short track and an unusable tempo range are named
//! conditions rather than an `IndexError` from somewhere inside the numerics.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod decode;
pub mod envelope;
pub mod error;
pub mod jumpcue_detect;
pub mod key;
pub mod onset;
pub mod pipeline;
pub mod tempo;

pub use decode::{decode_file, decode_for_analysis, resample, Audio};
pub use envelope::{Band, Envelopes};
pub use error::{AnalysisError, DecodeError};
pub use jumpcue_detect::{JumpCueOptions, SimilarLink};
pub use key::{Chroma, ChromaOptions, KeyOptions};
pub use onset::{OnsetEnvelope, OnsetOptions};
pub use mixlyzer_phrase::{PhraseError, PhraseModel};
pub use pipeline::{
    analyze_file, analyze_file_with, analyze_samples, analyze_samples_with, Analysis,
    AnalysisOptions,
};
pub use tempo::{TempoEstimate, TempoOptions};
