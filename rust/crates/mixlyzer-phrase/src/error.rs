//! Failure modes of phrase detection, each naming what the caller got wrong.

use thiserror::Error;

/// Everything that can stop the phrase detector producing a result.
#[derive(Debug, Error)]
pub enum PhraseError {
    #[error("beat times must be a strictly increasing array of at least 8 beats")]
    BeatGridTooShort,

    #[error("beat times must be strictly increasing, but beat {index} is not after beat {}", index - 1)]
    BeatsNotIncreasing { index: usize },

    #[error("beat times must be finite and non-negative")]
    BeatTimesNotFinite,

    #[error("no usable tempo segment: each needs a positive span and a meter of at least 1")]
    NoTempoSegments,

    #[error(
        "tempo segment downbeat at {inizio:.6}s is {error:.3}s from the nearest beat, \
         more than the {tolerance:.3}s tolerance"
    )]
    DownbeatNotOnBeat {
        inizio: f64,
        error: f64,
        tolerance: f64,
    },

    #[error("the tempo segments do not overlap the beat grid at all")]
    SegmentsMissTheGrid,

    #[error("audio is empty")]
    EmptyAudio,

    #[error(
        "audio is at {found} Hz but the model expects {expected} Hz; decode at the \
         model's rate (e.g. mixlyzer_dsp::decode_for_analysis) rather than resampling \
         here, which would shift every feature value away from the training data"
    )]
    SampleRateMismatch { found: u32, expected: u32 },

    #[error(
        "the beat grid ends at {beats_end:.3}s but the audio ends at {audio_end:.3}s; \
         these are probably not the same track"
    )]
    GridPastAudio { beats_end: f64, audio_end: f64 },

    #[error("could not read the phrase model at {path}: {source}")]
    ModelIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("the phrase model is not a readable NumPy archive: {0}")]
    MalformedModel(String),

    #[error("the phrase model is missing the field '{0}'")]
    MissingModelField(String),

    #[error("the phrase model field '{field}' is unusable: {reason}")]
    ModelField { field: String, reason: String },

    #[error("unsupported phrase model format '{found}', expected '{expected}'")]
    UnsupportedModelFormat { found: String, expected: String },
}
