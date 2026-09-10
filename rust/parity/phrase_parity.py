#!/usr/bin/env python3
"""Check the Rust phrase features against librosa, feature by feature.

Run from the repository root:

    .venv/bin/python rust/parity/phrase_parity.py

The Rust example ``parity_dump`` generates a fixed synthetic signal, computes
every intermediate feature, and writes both to JSON. This script feeds the very
same samples through librosa and reports, per feature, the largest absolute and
relative disagreement.

The floor on agreement is not our arithmetic: librosa runs the STFT and most of
what follows in float32, while the Rust port uses float64 throughout. So a
relative error around 1e-6 on a magnitude-derived quantity is librosa's own
rounding, not a porting bug. Anything materially larger is.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Per-feature tolerance, as (absolute, relative), applied the way numpy's
# allclose does: a cell passes when |rust - librosa| <= abs + rel * |librosa|.
# The absolute term is what matters for coefficients that sit near zero (high
# MFCCs, quiet onset frames), where a relative test is meaningless.
TOLERANCE = {
    "stft_magnitude": (1e-4, 1e-5, "librosa's STFT is float32; we use float64"),
    "mel_power": (1e-4, 1e-5, "float32 filterbank applied to a float32 spectrogram"),
    "mel_db": (1e-4, 1e-5, "log of the above, so the error is compressed"),
    "mfcc": (5e-4, 1e-5, "orthonormal DCT-II summing 48 float32 mel-dB values"),
    "harmonic_magnitude": (1e-4, 1e-5, "median filter is exact; the mask is float32"),
    "percussive_magnitude": (1e-4, 1e-5, "median filter is exact; the mask is float32"),
    "chroma": (1e-5, 1e-5, "float32 chroma basis, then an L2 column normalisation"),
    "tonnetz": (1e-5, 1e-5, "linear map of an L1-normalised chroma"),
    "spectral_contrast": (1e-4, 1e-6, "peak/valley means of float32 magnitudes"),
    "spectral_centroid": (1e-3, 1e-6, "L1-normalised magnitude weighted by frequency (hertz)"),
    "spectral_bandwidth": (1e-3, 1e-6, "second moment about the centroid (hertz)"),
    "spectral_flatness": (1e-10, 1e-5, "geometric mean over 1025 bins of |S|^4"),
    "spectral_rolloff": (0.0, 0.0, "a bin frequency, so it matches exactly or not at all"),
    "rms": (1e-7, 1e-6, "sum of float32 squares"),
    "rms_signal": (1e-7, 1e-6, "framed float32 signal"),
    "onset_full": (1e-5, 1e-5, "difference of two nearby decibel values"),
    "onset_percussive": (1e-5, 1e-5, "difference of two nearby decibel values"),
}


VARIANTS = ("music", "noise", "quiet", "structured")


def rust_dump(variant: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dump.json"
        subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "--release",
                "-p",
                "mixlyzer-phrase",
                "--example",
                "parity_dump",
                "--",
                str(out),
                variant,
            ],
            cwd=REPO / "rust",
            check=True,
        )
        return json.loads(out.read_text())


def as_matrix(entry: dict) -> np.ndarray:
    rows, cols = entry["shape"]
    return np.asarray(entry["data"], dtype=np.float64).reshape(rows, cols)


def python_features(dump: dict) -> dict[str, np.ndarray]:
    import librosa

    sr = int(dump["sample_rate"])
    n_fft = int(dump["n_fft"])
    hop = int(dump["hop_length"])
    n_mels = int(dump["n_mels"])
    n_mfcc = int(dump["n_mfcc"])
    y = np.asarray(dump["signal"], dtype=np.float32)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    magnitude = np.abs(stft).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(magnitude)
    harmonic_power = np.square(harmonic_mag, dtype=np.float32)
    percussive_power = np.square(percussive_mag, dtype=np.float32)

    mel_power = librosa.feature.melspectrogram(
        S=power, sr=sr, n_mels=n_mels, fmin=30.0, fmax=11025.0
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(
        S=harmonic_power, sr=sr, n_fft=n_fft, hop_length=hop, norm=2
    )
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr, n_fft=n_fft, n_bands=4)

    percussive_mel = librosa.feature.melspectrogram(
        S=percussive_power, sr=sr, n_mels=n_mels, fmin=30.0, fmax=11025.0
    )

    return {
        "stft_magnitude": magnitude,
        "mel_power": mel_power,
        "mel_db": mel_db,
        "mfcc": mfcc,
        "harmonic_magnitude": harmonic_mag,
        "percussive_magnitude": percussive_mag,
        "chroma": chroma,
        "tonnetz": tonnetz,
        "spectral_contrast": contrast,
        "spectral_centroid": librosa.feature.spectral_centroid(S=magnitude, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=magnitude, sr=sr),
        "spectral_flatness": librosa.feature.spectral_flatness(S=power),
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            S=magnitude, sr=sr, roll_percent=0.85
        ),
        "rms": librosa.feature.rms(S=magnitude, frame_length=n_fft),
        "rms_signal": librosa.feature.rms(
            y=y, frame_length=n_fft, hop_length=hop, center=True
        ),
        "onset_full": librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop),
        "onset_percussive": librosa.onset.onset_strength(
            S=librosa.power_to_db(percussive_mel, ref=np.max), sr=sr, hop_length=hop
        ),
        "_tuning": librosa.estimate_tuning(S=harmonic_power, sr=sr, bins_per_octave=12),
    }


def compare(name: str, rust: np.ndarray, expected: np.ndarray) -> tuple[bool, str]:
    expected = np.atleast_2d(np.asarray(expected, dtype=np.float64))
    if rust.shape != expected.shape:
        return False, f"shape {rust.shape} != {expected.shape}"
    abs_tol, rel_tol, reason = TOLERANCE.get(name, (0.0, 0.0, ""))
    diff = np.abs(rust - expected)
    allowed = abs_tol + rel_tol * np.abs(expected)
    bad = diff > allowed
    max_abs = float(np.max(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(expected), 1e-300)
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    detail = f"max_abs={max_abs:.2e} max_rel={max_rel:.2e} tol=({abs_tol:.0e},{rel_tol:.0e})"
    ok = not bool(np.any(bad))
    if not ok:
        worst = np.unravel_index(int(np.argmax(diff - allowed)), diff.shape)
        detail += f" fails={int(bad.sum())}/{bad.size} worst@{tuple(int(x) for x in worst)}"
        detail += f" rust={rust[worst]:.9g} librosa={expected[worst]:.9g}"
    if reason:
        detail += f"  [{reason}]"
    return ok, detail


def main() -> int:
    failures = 0
    for variant in VARIANTS:
        dump = rust_dump(variant)
        expected = python_features(dump)
        print(f"\n=== signal: {variant} ===")
        print(f"  tuning: rust={dump['tuning']!r} librosa={float(expected['_tuning'])!r}")
        for name in TOLERANCE:
            if name not in dump:
                print(f"  SKIP {name:24s} (not in the Rust dump)")
                continue
            ok, detail = compare(name, as_matrix(dump[name]), expected[name])
            status = "ok  " if ok else "FAIL"
            print(f"  {status} {name:24s} {detail}")
            failures += 0 if ok else 1
        if not math.isclose(
            float(dump["tuning"]), float(expected["_tuning"]), abs_tol=1e-12
        ):
            print("  FAIL tuning                   estimates disagree")
            failures += 1

    print(f"\n{failures} feature(s) outside tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
