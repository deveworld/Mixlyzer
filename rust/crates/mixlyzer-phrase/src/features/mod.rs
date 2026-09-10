//! Beat-synchronous acoustic features, ported from `structure.py`.
//!
//! Every submodule here mirrors a librosa routine. Matching librosa is not a
//! stylistic preference: the gradient-boosted models split on raw feature
//! values, so a feature that is merely *similar* moves samples across
//! thresholds and turns the output into plausible-looking noise.

pub mod chroma;
pub mod hpss;
pub mod matrix;
pub mod mel;
pub mod onset;
pub mod song;
pub mod spectral;
pub mod stft;
