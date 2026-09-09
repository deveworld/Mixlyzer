//! Configuration: schema, defaults, and loading that reports failure instead of
//! crashing or silently discarding settings.
//!
//! Three behaviours differ deliberately from the Python implementation.
//!
//! * Creating the library directory can fail. Python calls `mkdir` outside the
//!   try/except in `load_cfg`, so an unreachable Library Path raises before any
//!   window exists and the app cannot start again until `config.json` is edited
//!   by hand. Here it is [`Config::ensure_library_dir`], a separate fallible
//!   step the caller can report and recover from.
//! * A malformed config file is an error, not a reason to overwrite. Python
//!   rewrites `config.json` with defaults on any parse error, silently losing
//!   every setting including the library path.
//! * Types are checked. Python coerces with `bool(value)`, so the JSON string
//!   `"false"` becomes `true`, and a non-numeric BPM bound becomes `0`.

use std::path::{Path, PathBuf};

use crate::error::ConfigError;

/// Where the library lives and how it is logged.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LibraryConfig {
    pub libpath: String,
    pub write_log: bool,
    pub logpath: String,
    pub rekordbox_sync_enabled: bool,
    pub rekordbox_xml_path: String,
}

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            libpath: "library".into(),
            write_log: false,
            logpath: "mixlyzer.log".into(),
            rekordbox_sync_enabled: false,
            rekordbox_xml_path: String::new(),
        }
    }
}

/// How chroma is extracted for key detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChromaMethod {
    Cqt,
    Cens,
}

/// Parameters of the analysis pipeline.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AnalysisConfig {
    pub use_hpss: bool,
    pub analysis_samp_rate: u32,
    pub chroma_method: ChromaMethod,
    pub chroma_hop_length: usize,
    pub chroma_cqt_bins_per_octave: usize,
    pub chroma_cqt_octaves: usize,
    pub bpm_hop_length: usize,
    /// Analysis window for dynamic tempo tracking, in milliseconds.
    pub bpm_win_length: usize,
    pub bpm_min: u32,
    pub bpm_max: u32,
    pub bpm_dynamic: bool,
    pub bpm_adaptive_window: bool,
    pub dynamic_downbeat: bool,
    /// Shifts the whole grid, in milliseconds. Positive moves beats later.
    pub beatgrid_offset_msec: f64,
    /// Envelope frame length in milliseconds.
    pub env_frame_ms: f64,
    pub env_lo: (f64, f64),
    pub env_mid: (f64, f64),
    pub env_hi: (f64, f64),
    pub env_order: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            use_hpss: true,
            analysis_samp_rate: 22_050,
            chroma_method: ChromaMethod::Cens,
            chroma_hop_length: 512,
            chroma_cqt_bins_per_octave: 36,
            chroma_cqt_octaves: 6,
            bpm_hop_length: 128,
            bpm_win_length: 5_000,
            bpm_min: 110,
            bpm_max: 220,
            bpm_dynamic: true,
            bpm_adaptive_window: true,
            dynamic_downbeat: false,
            beatgrid_offset_msec: 0.0,
            env_frame_ms: 4.0,
            env_lo: (20.0, 200.0),
            env_mid: (200.0, 3_000.0),
            env_hi: (3_000.0, 11_025.0),
            env_order: 4,
        }
    }
}

impl AnalysisConfig {
    /// The tempo search range, ordered and clamped to something usable.
    ///
    /// Python passes the raw values through, so `bpm_min > bpm_max` or a range
    /// narrower than the autocorrelation window produces either an `IndexError`
    /// or a silently wrong tempo.
    pub fn bpm_range(&self) -> (f64, f64) {
        let lo = f64::from(self.bpm_min.min(self.bpm_max)).max(20.0);
        let hi = f64::from(self.bpm_max.max(self.bpm_min)).min(400.0);
        if hi - lo < 1.0 {
            (lo, lo + 1.0)
        } else {
            (lo, hi)
        }
    }

    /// Grid offset in seconds.
    pub fn beatgrid_offset_sec(&self) -> f64 {
        self.beatgrid_offset_msec / 1000.0
    }

    /// Envelope hop in samples at the analysis rate.
    pub fn env_hop_samples(&self) -> usize {
        let samples = self.env_frame_ms * 1e-3 * f64::from(self.analysis_samp_rate);
        (samples.round() as usize).max(1)
    }
}

/// Weights of the key-detection transition model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KeyConfig {
    pub min_offset: f64,
    pub pitch_self: f64,
    pub pitch_semitone: f64,
    pub pitch_fifth: f64,
    pub pitch_others: f64,
}

impl Default for KeyConfig {
    fn default() -> Self {
        Self {
            min_offset: 0.4,
            pitch_self: 0.9,
            pitch_semitone: 0.02,
            pitch_fifth: 0.001,
            pitch_others: 0.01,
        }
    }
}

impl KeyConfig {
    /// Whether the transition weights can form a probability distribution.
    ///
    /// `analyzer_core/key/viterbi_key.py:51` has a bare `raise` for this case,
    /// which surfaces as `RuntimeError: No active exception to reraise` from
    /// inside the analysis subprocess, with nothing pointing at the setting
    /// that caused it.
    pub fn weights_are_usable(&self) -> bool {
        let total =
            self.pitch_self + self.pitch_semitone + self.pitch_fifth + self.pitch_others;
        total.is_finite() && total > 0.0
    }
}

/// The whole configuration file.
///
/// Sections the Rust tools do not use (the Qt view settings, playback, and the
/// Windows-only external deck sync) are preserved verbatim in `extra`, so
/// reading and writing `config.json` here never destroys the desktop app's
/// settings.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Config {
    #[serde(default)]
    pub libconfig: LibraryConfig,
    #[serde(default)]
    pub analysisconfig: AnalysisConfig,
    #[serde(default)]
    pub keyconfig: KeyConfig,
    /// Sections owned by the GUI, round-tripped untouched.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            libconfig: LibraryConfig::default(),
            analysisconfig: AnalysisConfig::default(),
            keyconfig: KeyConfig::default(),
            extra: serde_json::Map::new(),
        }
    }
}

impl Config {
    /// Read a config file, falling back to defaults only when it is absent.
    ///
    /// A file that exists but cannot be read or parsed is an error. Python
    /// overwrites it with defaults instead, discarding the user's library path
    /// along with everything else.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default())
            }
            Err(source) => {
                return Err(ConfigError::Read {
                    path: path.to_path_buf(),
                    source,
                })
            }
        };
        serde_json::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Write the config back out, pretty-printed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let text = serde_json::to_string_pretty(self).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        std::fs::write(path, text).map_err(|source| ConfigError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    /// The configured library directory.
    pub fn library_path(&self) -> PathBuf {
        PathBuf::from(&self.libconfig.libpath)
    }

    /// Create the library directory, reporting failure instead of panicking.
    ///
    /// Call this once at startup and show the user the error; the path is
    /// theirs to fix, and they cannot fix it if the process dies first.
    pub fn ensure_library_dir(&self) -> Result<PathBuf, ConfigError> {
        let path = self.library_path();
        std::fs::create_dir_all(&path).map_err(|source| ConfigError::LibraryPath {
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mixlyzer-config-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn defaults_match_the_python_schema() {
        let cfg = Config::default();
        assert_eq!(cfg.libconfig.libpath, "library");
        assert_eq!(cfg.analysisconfig.analysis_samp_rate, 22_050);
        assert_eq!(cfg.analysisconfig.bpm_hop_length, 128);
        assert_eq!(cfg.analysisconfig.bpm_min, 110);
        assert_eq!(cfg.analysisconfig.bpm_max, 220);
        assert_eq!(cfg.analysisconfig.chroma_method, ChromaMethod::Cens);
        assert!(cfg.analysisconfig.bpm_dynamic);
        assert_eq!(cfg.keyconfig.pitch_self, 0.9);
    }

    #[test]
    fn a_missing_file_yields_defaults_without_writing_anything() {
        let dir = temp_dir("missing");
        let path = dir.join("config.json");
        assert_eq!(Config::load(&path).unwrap(), Config::default());
        assert!(!path.exists(), "load must not create the file");
    }

    /// Python's `except json.JSONDecodeError` branch overwrites the file with
    /// defaults, so a truncated config silently loses the library path.
    #[test]
    fn a_malformed_file_is_an_error_and_is_left_alone() {
        let dir = temp_dir("malformed");
        let path = dir.join("config.json");
        std::fs::write(&path, "{ this is not json").unwrap();

        let err = Config::load(&path).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{ this is not json",
            "the original file must survive"
        );
    }

    /// `bool("false")` is `True` in Python, so this string silently flips the
    /// setting on.
    #[test]
    fn a_string_where_a_bool_belongs_is_rejected() {
        let dir = temp_dir("bool");
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"libconfig":{"libpath":"lib","write_log":"false","logpath":"l","rekordbox_sync_enabled":false,"rekordbox_xml_path":""}}"#).unwrap();
        assert!(matches!(Config::load(&path), Err(ConfigError::Parse { .. })));
    }

    #[test]
    fn unknown_sections_survive_a_round_trip() {
        let dir = temp_dir("extra");
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{"viewconfig":{"fps":60,"display_phrase":true},"playbackconfig":{"enable_metronome":true}}"#,
        )
        .unwrap();

        let cfg = Config::load(&path).unwrap();
        assert!(cfg.extra.contains_key("viewconfig"));
        assert!(cfg.extra.contains_key("playbackconfig"));

        let out = dir.join("out.json");
        cfg.save(&out).unwrap();
        let reloaded = Config::load(&out).unwrap();
        assert_eq!(reloaded.extra, cfg.extra, "GUI settings must not be dropped");
        assert_eq!(
            reloaded.extra["viewconfig"]["fps"],
            serde_json::json!(60)
        );
    }

    #[test]
    fn config_round_trips_through_save_and_load() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("config.json");
        let mut cfg = Config::default();
        cfg.libconfig.libpath = "my-library".into();
        cfg.analysisconfig.bpm_min = 90;
        cfg.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), cfg);
    }

    /// The critical startup failure: Python raises `FileNotFoundError` out of
    /// `load_cfg` for this path and the app never opens a window.
    #[test]
    fn an_unreachable_library_path_is_a_typed_error_not_a_crash() {
        let cfg = Config {
            libconfig: LibraryConfig {
                libpath: "/proc/nonexistent/deep/lib".into(),
                ..LibraryConfig::default()
            },
            ..Config::default()
        };
        let err = cfg.ensure_library_dir().unwrap_err();
        assert!(
            matches!(err, ConfigError::LibraryPath { .. }),
            "got {err:?}"
        );
        // The message names the path, so the caller can tell the user which
        // setting to change.
        assert!(err.to_string().contains("/proc/nonexistent"));
    }

    #[test]
    fn a_usable_library_path_is_created() {
        let dir = temp_dir("libdir");
        let mut cfg = Config::default();
        cfg.libconfig.libpath = dir.join("nested/library").to_string_lossy().into();
        let created = cfg.ensure_library_dir().unwrap();
        assert!(created.is_dir());
        // Idempotent.
        assert!(cfg.ensure_library_dir().is_ok());
    }

    #[test]
    fn bpm_range_is_ordered_and_never_degenerate() {
        let mut cfg = AnalysisConfig::default();
        assert_eq!(cfg.bpm_range(), (110.0, 220.0));

        cfg.bpm_min = 220;
        cfg.bpm_max = 110;
        assert_eq!(cfg.bpm_range(), (110.0, 220.0), "reversed bounds are sorted");

        cfg.bpm_min = 128;
        cfg.bpm_max = 128;
        let (lo, hi) = cfg.bpm_range();
        assert!(hi > lo, "a zero-width range would break the tempo search");
    }

    #[test]
    fn env_hop_is_at_least_one_sample() {
        let mut cfg = AnalysisConfig::default();
        assert_eq!(cfg.env_hop_samples(), 88); // 4 ms at 22050 Hz
        cfg.env_frame_ms = 0.0;
        assert_eq!(cfg.env_hop_samples(), 1);
    }

    #[test]
    fn key_weights_are_checked_before_use() {
        let mut cfg = KeyConfig::default();
        assert!(cfg.weights_are_usable());
        cfg.pitch_self = 0.0;
        cfg.pitch_semitone = 0.0;
        cfg.pitch_fifth = 0.0;
        cfg.pitch_others = 0.0;
        assert!(!cfg.weights_are_usable(), "all-zero weights must be caught");
    }

    #[test]
    fn beatgrid_offset_converts_to_seconds() {
        let cfg = AnalysisConfig {
            beatgrid_offset_msec: 25.0,
            ..AnalysisConfig::default()
        };
        assert!((cfg.beatgrid_offset_sec() - 0.025).abs() < 1e-12);
    }
}
