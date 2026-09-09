//! Band envelopes: the low / mid / high energy curves the waveform is drawn from.
//!
//! Each band is a cascaded biquad bandpass applied forwards and then backwards,
//! which cancels the phase response so a transient stays where it belongs. The
//! result is framed into RMS values, plus the per-frame sample extremes used
//! for the waveform outline.

/// One second-order section of a filter.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    /// Constant-skirt bandpass with unity peak gain, from the audio EQ cookbook.
    fn bandpass(center_hz: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * center_hz / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    /// Filter in place, forwards, carrying state across the whole signal.
    fn run(&self, signal: &mut [f64]) {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for sample in signal.iter_mut() {
            let x0 = *sample;
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *sample = y0;
        }
    }
}

/// A band of the spectrum, expressed as a frequency range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub low_hz: f64,
    pub high_hz: f64,
}

impl Band {
    pub fn new(low_hz: f64, high_hz: f64) -> Self {
        Self { low_hz, high_hz }
    }

    /// Geometric centre, which is where a bandpass should sit for a log-spaced band.
    fn center(&self, nyquist: f64) -> f64 {
        let low = self.low_hz.max(1.0);
        let high = self.high_hz.min(nyquist * 0.98).max(low + 1.0);
        (low * high).sqrt()
    }

    /// Q that makes the -3 dB points land on the band edges.
    fn q(&self, nyquist: f64) -> f64 {
        let low = self.low_hz.max(1.0);
        let high = self.high_hz.min(nyquist * 0.98).max(low + 1.0);
        let center = (low * high).sqrt();
        let bandwidth = high - low;
        if bandwidth <= 0.0 {
            1.0
        } else {
            (center / bandwidth).clamp(0.2, 20.0)
        }
    }
}

/// The three envelopes plus the waveform outline, all on the same frame grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelopes {
    pub low: Vec<f32>,
    pub mid: Vec<f32>,
    pub high: Vec<f32>,
    /// Most negative sample in each frame.
    pub min: Vec<f32>,
    /// Most positive sample in each frame.
    pub max: Vec<f32>,
    /// Frame hop in samples; frames do not overlap.
    pub hop: usize,
    pub sample_rate: u32,
}

impl Envelopes {
    pub fn len(&self) -> usize {
        self.low.len()
    }

    pub fn is_empty(&self) -> bool {
        self.low.is_empty()
    }

    /// Seconds covered by each frame.
    pub fn frame_duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.hop as f64 / f64::from(self.sample_rate)
        }
    }

    /// Time at the start of frame `index`.
    pub fn frame_time(&self, index: usize) -> f64 {
        index as f64 * self.frame_duration()
    }
}

/// Compute the band envelopes and waveform outline for `signal`.
///
/// `hop` is both the frame length and the step, so frames tile the signal
/// without overlap. `order` is rounded up to an even number of biquad passes.
pub fn compute(
    signal: &[f32],
    sample_rate: u32,
    hop: usize,
    bands: [Band; 3],
    order: u32,
) -> Envelopes {
    let hop = hop.max(1);
    let frames = signal.len() / hop;
    let mut out = Envelopes {
        low: Vec::with_capacity(frames),
        mid: Vec::with_capacity(frames),
        high: Vec::with_capacity(frames),
        min: Vec::with_capacity(frames),
        max: Vec::with_capacity(frames),
        hop,
        sample_rate,
    };
    if frames == 0 {
        return out;
    }

    for chunk in signal.chunks_exact(hop) {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &sample in chunk {
            lo = lo.min(sample);
            hi = hi.max(sample);
        }
        out.min.push(lo);
        out.max.push(hi);
    }

    let nyquist = f64::from(sample_rate) * 0.5;
    let sections = order.div_ceil(2).max(1);
    for (band, target) in bands
        .iter()
        .zip([&mut out.low, &mut out.mid, &mut out.high])
    {
        let filtered = filter_band(signal, *band, nyquist, sample_rate, sections);
        *target = frame_rms(&filtered, hop, frames);
    }
    out
}

/// Zero-phase bandpass: forward pass, reverse, forward again, reverse back.
fn filter_band(
    signal: &[f32],
    band: Band,
    nyquist: f64,
    sample_rate: u32,
    sections: u32,
) -> Vec<f64> {
    let mut buffer: Vec<f64> = signal.iter().map(|s| f64::from(*s)).collect();
    if buffer.is_empty() {
        return buffer;
    }
    let filter = Biquad::bandpass(
        band.center(nyquist),
        band.q(nyquist),
        f64::from(sample_rate),
    );
    for _ in 0..sections {
        filter.run(&mut buffer);
        buffer.reverse();
        filter.run(&mut buffer);
        buffer.reverse();
    }
    buffer
}

/// Root-mean-square of each non-overlapping frame.
fn frame_rms(signal: &[f64], hop: usize, frames: usize) -> Vec<f32> {
    signal
        .chunks_exact(hop)
        .take(frames)
        .map(|chunk| {
            let sum: f64 = chunk.iter().map(|s| s * s).sum();
            ((sum / chunk.len() as f64).sqrt()) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BANDS: [Band; 3] = [
        Band {
            low_hz: 20.0,
            high_hz: 200.0,
        },
        Band {
            low_hz: 200.0,
            high_hz: 3_000.0,
        },
        Band {
            low_hz: 3_000.0,
            high_hz: 11_025.0,
        },
    ];

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
    fn frame_count_follows_the_hop() {
        let signal = vec![0.0f32; 1000];
        let env = compute(&signal, 22_050, 100, BANDS, 4);
        assert_eq!(env.len(), 10);
        assert_eq!(env.min.len(), 10);
        assert_eq!(env.max.len(), 10);
    }

    #[test]
    fn a_signal_shorter_than_one_frame_yields_nothing() {
        let env = compute(&[0.1, 0.2], 22_050, 100, BANDS, 4);
        assert!(env.is_empty());
        assert_eq!(env.len(), 0);
    }

    #[test]
    fn silence_gives_zero_energy_in_every_band() {
        let env = compute(&vec![0.0f32; 4410], 22_050, 441, BANDS, 4);
        assert!(env.low.iter().all(|v| *v < 1e-9));
        assert!(env.mid.iter().all(|v| *v < 1e-9));
        assert!(env.high.iter().all(|v| *v < 1e-9));
    }

    #[test]
    fn a_bass_tone_lands_in_the_low_band() {
        let signal = sine(80.0, 1.0, 22_050);
        let env = compute(&signal, 22_050, 441, BANDS, 4);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (low, mid, high) = (mean(&env.low), mean(&env.mid), mean(&env.high));
        assert!(low > mid * 4.0, "low {low} should dominate mid {mid}");
        assert!(low > high * 4.0, "low {low} should dominate high {high}");
    }

    #[test]
    fn a_mid_tone_lands_in_the_mid_band() {
        let signal = sine(1_000.0, 1.0, 22_050);
        let env = compute(&signal, 22_050, 441, BANDS, 4);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (low, mid, high) = (mean(&env.low), mean(&env.mid), mean(&env.high));
        assert!(mid > low * 4.0, "mid {mid} should dominate low {low}");
        assert!(mid > high * 4.0, "mid {mid} should dominate high {high}");
    }

    #[test]
    fn a_treble_tone_lands_in_the_high_band() {
        let signal = sine(6_000.0, 1.0, 22_050);
        let env = compute(&signal, 22_050, 441, BANDS, 4);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (low, mid, high) = (mean(&env.low), mean(&env.mid), mean(&env.high));
        assert!(high > low * 4.0, "high {high} should dominate low {low}");
        assert!(high > mid * 2.0, "high {high} should dominate mid {mid}");
    }

    #[test]
    fn the_outline_tracks_the_sample_extremes() {
        let mut signal = vec![0.0f32; 300];
        signal[50] = 0.9;
        signal[150] = -0.7;
        let env = compute(&signal, 22_050, 100, BANDS, 4);
        assert!((env.max[0] - 0.9).abs() < 1e-6);
        assert!((env.min[1] + 0.7).abs() < 1e-6);
        assert_eq!(env.max[2], 0.0);
    }

    /// Forward-and-back filtering must not move a transient in time, which is
    /// the whole reason the filter is run in both directions.
    #[test]
    fn filtering_does_not_shift_a_transient() {
        let mut signal = vec![0.0f32; 4410];
        let spike = 2205;
        for offset in 0..20 {
            signal[spike + offset] = 1.0;
        }
        let env = compute(&signal, 22_050, 64, BANDS, 4);
        let peak_frame = env
            .mid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let expected = spike / 64;
        assert!(
            (peak_frame as i64 - expected as i64).abs() <= 2,
            "peak landed at frame {peak_frame}, expected near {expected}"
        );
    }

    #[test]
    fn frame_timing_matches_the_hop() {
        let env = compute(&vec![0.0f32; 22_050], 22_050, 441, BANDS, 4);
        assert!((env.frame_duration() - 0.02).abs() < 1e-9);
        assert!((env.frame_time(10) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn a_zero_hop_is_treated_as_one_sample() {
        let env = compute(&[0.5, -0.5], 22_050, 0, BANDS, 4);
        assert_eq!(env.hop, 1);
        assert_eq!(env.len(), 2);
    }

    #[test]
    fn rms_of_a_constant_frame_is_its_magnitude() {
        let values = vec![0.5f64; 100];
        let rms = frame_rms(&values, 100, 1);
        assert!((rms[0] - 0.5).abs() < 1e-6);
    }
}
