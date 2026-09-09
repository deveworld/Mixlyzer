//! Errors from decoding and analysis.

use std::path::PathBuf;

/// Why an audio file could not be turned into samples.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("could not open {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not in a format this build can read: {reason}")]
    UnsupportedFormat { path: PathBuf, reason: String },

    #[error("no audio track found in {path}")]
    NoAudioTrack { path: PathBuf },

    #[error("{path} is damaged or truncated: {reason}")]
    Corrupt { path: PathBuf, reason: String },

    #[error("{path} decoded to no audio at all")]
    Empty { path: PathBuf },

    #[error("cannot resample {from} Hz to {to} Hz")]
    BadSampleRate { from: u32, to: u32 },
}

/// Why analysis could not produce a result.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error(transparent)]
    Decode(#[from] DecodeError),

    /// The audio is too short for the requested analysis.
    ///
    /// Python raises `IndexError` from inside the tempo refinement for this
    /// case, which aborts the whole track with a message that says nothing
    /// about the real cause.
    #[error("track is {duration_sec:.2}s, which is shorter than the {needed_sec:.2}s this analysis needs")]
    TooShort { duration_sec: f64, needed_sec: f64 },

    /// No onset energy anywhere: digital silence, or a file that decoded to
    /// zeros. Python's eigendecomposition raises an opaque ArpackError here.
    #[error("no onset energy found, so there is no tempo to estimate")]
    Silent,

    #[error("tempo search range {lo}-{hi} BPM is too narrow at hop {hop} to hold a candidate")]
    TempoRangeTooNarrow { lo: f64, hi: f64, hop: usize },
}
