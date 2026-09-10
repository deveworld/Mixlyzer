//! Dump every intermediate feature as JSON, for `rust/parity/phrase_parity.py`.
//!
//! Usage: `cargo run -p mixlyzer-phrase --example parity_dump -- <out.json>`
//!
//! The dump carries the input signal alongside the outputs so the Python side
//! feeds librosa exactly the samples Rust saw. Anything computed here that the
//! Python script does not know how to check is simply ignored, so adding a new
//! feature to the dump is always safe.

use mixlyzer_phrase::features::matrix::Mat;
use mixlyzer_phrase::features::{chroma, hpss, mel, onset, spectral, stft};
use mixlyzer_phrase::testsig;
use serde_json::{json, Map, Value};

fn mat(m: &Mat) -> Value {
    json!({ "shape": [m.rows(), m.cols()], "data": m.as_slice() })
}

fn vector(v: &[f64]) -> Value {
    json!({ "shape": [1, v.len()], "data": v })
}

fn main() {
    let sample_rate = 22_050u32;
    let n_fft = 2048usize;
    let hop = 512usize;
    let n_mels = 48usize;
    let n_mfcc = 20usize;
    let variant = std::env::args().nth(2).unwrap_or_else(|| "music".to_string());
    let signal = match variant.as_str() {
        // Broadband noise: every bin is populated, so nothing hides behind a
        // zero and the quantile-based features are fully exercised.
        "noise" => {
            let mut rng = testsig::Xorshift::new(0x5EED);
            (0..(4.0 * f64::from(sample_rate)) as usize)
                .map(|_| (0.4 * rng.next_unit()) as f32)
                .collect()
        }
        // Near-silence: the clamping paths (amin, top_db, the normalisation
        // thresholds) all engage here and nowhere else.
        "quiet" => (0..(4.0 * f64::from(sample_rate)) as usize)
            .map(|i| (1e-7 * (i as f64 * 0.01).sin()) as f32)
            .collect(),
        "structured" => testsig::structured_signal(sample_rate, 4.0, 1.0),
        _ => testsig::parity_signal(sample_rate, 4.0),
    };
    let sr = f64::from(sample_rate);

    let magnitude = stft::stft_magnitude(&signal, n_fft, hop);
    let power = magnitude.map(|v| v * v);
    let (harmonic_mag, percussive_mag) = hpss::hpss(&magnitude);
    let harmonic_power = harmonic_mag.map(|v| v * v);
    let percussive_power = percussive_mag.map(|v| v * v);

    let basis = mel::mel_filterbank(sr, n_fft, n_mels, 30.0, 11_025.0);
    let mel_power = mel::melspectrogram(&power, &basis);
    let mel_db = mel::power_to_db(&mel_power, mel::DbRef::Max);
    let mfcc = mel::dct_ortho_rows(&mel_db, n_mfcc);

    let tuning = chroma::estimate_tuning(&harmonic_power, sr, n_fft);
    let chromagram = chroma::chroma_stft(&harmonic_power, sr, n_fft, tuning);
    let tonnetz = chroma::tonnetz(&chromagram);

    let contrast = spectral::spectral_contrast(&magnitude, sr, n_fft, 4);
    let centroid = spectral::spectral_centroid(&magnitude, sr, n_fft);
    let bandwidth = spectral::spectral_bandwidth(&magnitude, sr, n_fft);
    let flatness = spectral::spectral_flatness(&power);
    let rolloff = spectral::spectral_rolloff(&magnitude, sr, n_fft, 0.85);
    let rms_from_spectrum = spectral::rms_from_spectrum(&magnitude, n_fft);
    let rms_from_signal = spectral::rms_from_signal(&signal, n_fft, hop);

    let percussive_mel = mel::melspectrogram(&percussive_power, &basis);
    let onset_full = onset::onset_strength(&mel_db);
    let onset_percussive =
        onset::onset_strength(&mel::power_to_db(&percussive_mel, mel::DbRef::Max));

    let mut out = Map::new();
    out.insert("sample_rate".into(), json!(sample_rate));
    out.insert("n_fft".into(), json!(n_fft));
    out.insert("hop_length".into(), json!(hop));
    out.insert("n_mels".into(), json!(n_mels));
    out.insert("n_mfcc".into(), json!(n_mfcc));
    out.insert("signal".into(), json!(signal));
    out.insert("tuning".into(), json!(tuning));

    out.insert("stft_magnitude".into(), mat(&magnitude));
    out.insert("mel_power".into(), mat(&mel_power));
    out.insert("mel_db".into(), mat(&mel_db));
    out.insert("mfcc".into(), mat(&mfcc));
    out.insert("harmonic_magnitude".into(), mat(&harmonic_mag));
    out.insert("percussive_magnitude".into(), mat(&percussive_mag));
    out.insert("chroma".into(), mat(&chromagram));
    out.insert("tonnetz".into(), mat(&tonnetz));
    out.insert("spectral_contrast".into(), mat(&contrast));
    out.insert("spectral_centroid".into(), vector(&centroid));
    out.insert("spectral_bandwidth".into(), vector(&bandwidth));
    out.insert("spectral_flatness".into(), vector(&flatness));
    out.insert("spectral_rolloff".into(), vector(&rolloff));
    out.insert("rms".into(), vector(&rms_from_spectrum));
    out.insert("rms_signal".into(), vector(&rms_from_signal));
    out.insert("onset_full".into(), vector(&onset_full));
    out.insert("onset_percussive".into(), vector(&onset_percussive));

    let text = serde_json::to_string(&Value::Object(out)).expect("serialise dump");
    match std::env::args().nth(1) {
        Some(path) => std::fs::write(&path, text).expect("write dump"),
        None => println!("{text}"),
    }
}
