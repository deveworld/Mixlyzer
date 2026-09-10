//! Turning an audio file into samples, without an external binary.
//!
//! The Python implementation shells out to FFmpeg, reads the whole PCM stream
//! through a pipe, and calls the result a memory map (it is not one). That
//! makes FFmpeg an undocumented install requirement, gives no way to cancel a
//! decode, and holds the entire file in memory twice while `communicate()`
//! accumulates it.
//!
//! Here decoding is in-process through Symphonia. Nothing external is
//! required, failures are typed rather than printed, and the caller decides
//! how much of the file to read.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::DecodeError;

/// Decoded audio, always interleaved `f32` in `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct Audio {
    /// Interleaved samples, `channels` per frame.
    samples: Vec<f32>,
    channels: usize,
    sample_rate: u32,
}

impl Audio {
    /// Wrap raw interleaved samples.
    pub fn new(samples: Vec<f32>, channels: usize, sample_rate: u32) -> Self {
        debug_assert!(channels >= 1);
        Self {
            samples,
            channels: channels.max(1),
            sample_rate,
        }
    }

    /// Build from one mono channel.
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(samples, 1, sample_rate)
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Interleaved sample data.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Number of frames, i.e. samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    pub fn duration_sec(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames() as f64 / f64::from(self.sample_rate)
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Average the channels down to mono.
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }
        let scale = 1.0 / self.channels as f32;
        self.samples
            .chunks_exact(self.channels)
            .map(|frame| frame.iter().sum::<f32>() * scale)
            .collect()
    }

    /// Resample to `target_rate`, returning mono samples.
    ///
    /// Analysis runs at a fixed low rate (22.05 kHz by default), so this is
    /// almost always downsampling. A windowed-sinc kernel keeps the aliasing
    /// out of the onset detector; linear interpolation would fold high
    /// frequencies down onto the very transients being measured.
    pub fn to_mono_resampled(&self, target_rate: u32) -> Result<Vec<f32>, DecodeError> {
        if target_rate == 0 || self.sample_rate == 0 {
            return Err(DecodeError::BadSampleRate {
                from: self.sample_rate,
                to: target_rate,
            });
        }
        let mono = self.to_mono();
        if self.sample_rate == target_rate {
            return Ok(mono);
        }
        Ok(resample_sinc(&mono, self.sample_rate, target_rate))
    }
}

/// Decode a whole file.
pub fn decode_file(path: impl AsRef<Path>) -> Result<Audio, DecodeError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|source| DecodeError::Open {
        path: path.to_path_buf(),
        source,
    })?;

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .map_err(|err| DecodeError::UnsupportedFormat {
            path: path.to_path_buf(),
            reason: err.to_string(),
        })?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| DecodeError::NoAudioTrack {
            path: path.to_path_buf(),
        })?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| DecodeError::UnsupportedFormat {
            path: path.to_path_buf(),
            reason: err.to_string(),
        })?;

    let mut samples: Vec<f32> = Vec::new();
    let mut channels = 0usize;
    let mut sample_rate = 0u32;
    let mut buffer: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // A clean end of stream, however Symphonia chooses to report it.
            Err(SymphoniaError::IoError(ref err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(err) => {
                // Keep whatever decoded cleanly: a truncated file is still
                // worth analysing, and refusing it outright is what makes a
                // half-synced library unusable.
                if samples.is_empty() {
                    return Err(DecodeError::Corrupt {
                        path: path.to_path_buf(),
                        reason: err.to_string(),
                    });
                }
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if buffer.is_none() {
                    let spec = *decoded.spec();
                    channels = spec.channels.count();
                    sample_rate = spec.rate;
                    buffer = Some(SampleBuffer::new(decoded.capacity() as u64, spec));
                }
                if let Some(buf) = buffer.as_mut() {
                    buf.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(buf.samples());
                }
            }
            // Decode errors on individual packets are recoverable; skip them.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => {
                if samples.is_empty() {
                    return Err(DecodeError::Corrupt {
                        path: path.to_path_buf(),
                        reason: err.to_string(),
                    });
                }
                break;
            }
        }
    }

    if samples.is_empty() || channels == 0 {
        return Err(DecodeError::Empty {
            path: path.to_path_buf(),
        });
    }

    Ok(Audio::new(samples, channels, sample_rate))
}

/// Decode a file straight to mono at `target_rate`, which is what analysis wants.
pub fn decode_for_analysis(
    path: impl AsRef<Path>,
    target_rate: u32,
) -> Result<Vec<f32>, DecodeError> {
    decode_file(path)?.to_mono_resampled(target_rate)
}

/// Half-width of the resampling kernel, in output samples.
///
/// Sixteen lobes either side puts the stopband below measurement noise while
/// keeping the passband flat to within a few percent up to the new Nyquist.
/// Eight lobes also kills the stopband but sags to 0.79 gain at 10 kHz, which
/// would quietly dull the hi-hats the onset detector keys on.
const SINC_HALF_WIDTH: isize = 16;

/// Band-limited resampling with a Blackman-windowed sinc kernel.
///
/// Returns `input` unchanged when the rates already match, so a caller can ask
/// for a rate without first checking whether it has one.
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || from_rate == 0 || to_rate == 0 {
        return input.to_vec();
    }
    resample_sinc(input, from_rate, to_rate)
}

/// Band-limited resampling with a Blackman-windowed sinc kernel.
fn resample_sinc(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }

    // When downsampling, the kernel has to be stretched to the *output* band
    // so it filters before it decimates. Upsampling keeps the input band.
    let cutoff = ratio.min(1.0);
    let step = 1.0 / ratio;

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i as f64 * step;
        let base = center.floor() as isize;
        let width = (SINC_HALF_WIDTH as f64 / cutoff).ceil() as isize;

        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for tap in -width..=width {
            let idx = base + tap;
            if idx < 0 || idx as usize >= input.len() {
                continue;
            }
            let dist = (idx as f64 - center) * cutoff;
            let weight = sinc(dist) * blackman(dist, SINC_HALF_WIDTH as f64);
            acc += f64::from(input[idx as usize]) * weight;
            norm += weight;
        }
        // Normalising by the realised window keeps the gain flat where part of
        // the kernel hangs off the end of the signal. The trade is that the
        // first and last few samples are filtered less sharply than the
        // interior; over a whole track that is a handful of samples and no
        // analysis stage looks at them in isolation.
        let value = if norm.abs() > 1e-12 { acc / norm } else { 0.0 };
        out.push(value as f32);
    }
    out
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

/// Blackman window evaluated at `x`, zero outside `[-half_width, half_width]`.
fn blackman(x: f64, half_width: f64) -> f64 {
    if x.abs() > half_width {
        return 0.0;
    }
    let t = (x / half_width + 1.0) * 0.5; // map to [0, 1]
    let two_pi_t = 2.0 * std::f64::consts::PI * t;
    0.42 - 0.5 * two_pi_t.cos() + 0.08 * (2.0 * two_pi_t).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, seconds: f64, rate: u32) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn mono_audio_reports_its_shape() {
        let audio = Audio::mono(vec![0.0; 4410], 44_100);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.frames(), 4410);
        assert!((audio.duration_sec() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn stereo_frames_count_pairs_not_samples() {
        let audio = Audio::new(vec![0.0; 200], 2, 44_100);
        assert_eq!(audio.frames(), 100);
    }

    #[test]
    fn to_mono_averages_the_channels() {
        // Left is 1.0, right is -1.0, so the mono mix cancels to zero.
        let interleaved: Vec<f32> = (0..10).flat_map(|_| [1.0f32, -1.0f32]).collect();
        let audio = Audio::new(interleaved, 2, 44_100);
        let mono = audio.to_mono();
        assert_eq!(mono.len(), 10);
        assert!(mono.iter().all(|s| s.abs() < 1e-6));
    }

    #[test]
    fn to_mono_on_mono_audio_is_a_copy() {
        let audio = Audio::mono(vec![0.25; 8], 22_050);
        assert_eq!(audio.to_mono(), vec![0.25; 8]);
    }

    #[test]
    fn resampling_to_the_same_rate_changes_nothing() {
        let audio = Audio::mono(sine(440.0, 0.05, 44_100), 44_100);
        let out = audio.to_mono_resampled(44_100).unwrap();
        assert_eq!(out, audio.samples());
    }

    #[test]
    fn downsampling_produces_the_expected_length() {
        let audio = Audio::mono(sine(440.0, 1.0, 44_100), 44_100);
        let out = audio.to_mono_resampled(22_050).unwrap();
        assert!(
            (out.len() as i64 - 22_050).abs() <= 1,
            "expected about 22050 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn downsampling_preserves_a_tone_below_the_new_nyquist() {
        // 440 Hz survives a trip to 22.05 kHz; measure by zero crossings.
        let audio = Audio::mono(sine(440.0, 1.0, 44_100), 44_100);
        let out = audio.to_mono_resampled(22_050).unwrap();
        let crossings = out
            .windows(2)
            .filter(|w| (w[0] <= 0.0) != (w[1] <= 0.0))
            .count();
        // A 440 Hz sine crosses zero 880 times per second.
        assert!(
            (crossings as i64 - 880).abs() < 12,
            "expected ~880 zero crossings, saw {crossings}"
        );
    }

    /// Peak magnitude away from the signal edges, where the kernel is only
    /// partly covered and the gain correction is deliberately looser.
    fn interior_peak(signal: &[f32]) -> f32 {
        let margin = signal.len() / 8;
        signal[margin..signal.len() - margin]
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()))
    }

    #[test]
    fn downsampling_removes_content_above_the_new_nyquist() {
        // 15 kHz cannot exist at 22.05 kHz. Left in, it would fold down to
        // 7 kHz and land right on top of the transients the onset detector
        // keys on, so it has to be filtered out rather than decimated.
        for freq in [15_000.0, 18_000.0, 20_000.0] {
            let audio = Audio::mono(sine(freq, 0.5, 44_100), 44_100);
            let out = audio.to_mono_resampled(22_050).unwrap();
            let peak = interior_peak(&out);
            assert!(peak < 0.01, "{freq} Hz survived downsampling at {peak}");
        }
    }

    #[test]
    fn downsampling_keeps_the_passband_flat() {
        // Everything below the new Nyquist should come through at full level;
        // a sagging passband would quietly dull the high end.
        for freq in [440.0, 2_000.0, 5_000.0, 9_000.0] {
            let audio = Audio::mono(sine(freq, 0.5, 44_100), 44_100);
            let out = audio.to_mono_resampled(22_050).unwrap();
            let peak = interior_peak(&out);
            assert!(
                (peak - 1.0).abs() < 0.1,
                "{freq} Hz came through at {peak}, expected about 1.0"
            );
        }
    }

    #[test]
    fn upsampling_preserves_amplitude() {
        let audio = Audio::mono(sine(200.0, 0.2, 22_050), 22_050);
        let out = audio.to_mono_resampled(44_100).unwrap();
        let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak drifted to {peak}");
        assert!((out.len() as i64 - 8820).abs() <= 2);
    }

    #[test]
    fn empty_audio_resamples_to_empty() {
        let audio = Audio::mono(Vec::new(), 44_100);
        assert!(audio.to_mono_resampled(22_050).unwrap().is_empty());
    }

    #[test]
    fn a_zero_target_rate_is_an_error_not_a_panic() {
        let audio = Audio::mono(vec![0.0; 10], 44_100);
        assert!(matches!(
            audio.to_mono_resampled(0),
            Err(DecodeError::BadSampleRate { .. })
        ));
    }

    #[test]
    fn a_missing_file_is_a_typed_open_error() {
        let err = decode_file("/nonexistent/definitely/not/here.flac").unwrap_err();
        assert!(matches!(err, DecodeError::Open { .. }));
        assert!(err.to_string().contains("not/here.flac"));
    }

    #[test]
    fn a_non_audio_file_is_rejected_without_panicking() {
        let dir = std::env::temp_dir().join(format!("mixlyzer-dsp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("notaudio.flac");
        std::fs::write(&path, b"this is definitely not a FLAC file").unwrap();
        assert!(decode_file(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_sinc_kernel_is_one_at_zero_and_zero_at_integers() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-12);
        for k in 1..8 {
            assert!(sinc(k as f64).abs() < 1e-12, "sinc({k}) should vanish");
        }
    }

    #[test]
    fn the_window_vanishes_outside_its_support() {
        assert!(blackman(9.0, 8.0).abs() < 1e-12);
        assert!(blackman(0.0, 8.0) > 0.9);
    }
}
