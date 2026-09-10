//! The `externalsyncconfig` section of `config.json`.
//!
//! Field names and defaults match the Python dataclasses exactly, so an
//! existing config file round-trips. [`mixlyzer_core::Config`] keeps the
//! section verbatim in its `extra` map; [`SyncConfig::from_app_config`] pulls
//! it out.

use crate::timing::TotalSampleSource;
use crate::value::{ValueSpec, ValueType};

/// Which memory field the playhead is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMode {
    /// The deck reports seconds directly.
    Time,
    /// The deck reports a sample index; see [`crate::timing`].
    SampleIndex,
}

/// The five values that describe one deck.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeckConfig {
    /// Playhead in seconds, scaled by its multiplier.
    pub time: ValueSpec,
    /// Playhead as a sample index.
    pub sample_index: ValueSpec,
    /// Full path of the loaded file.
    pub path: ValueSpec,
    /// Whether this deck is the one the DJ is playing.
    pub active: ValueSpec,
    /// Whether this deck has a track in it at all.
    pub loaded: ValueSpec,
}

impl Default for DeckConfig {
    fn default() -> Self {
        Self {
            time: ValueSpec::new("0", ValueType::Float),
            sample_index: ValueSpec::new("0", ValueType::Int),
            path: ValueSpec {
                length: 2048,
                ..ValueSpec::new("0", ValueType::Str)
            },
            active: ValueSpec::new("0", ValueType::Bool),
            loaded: ValueSpec::new("0", ValueType::Bool),
        }
    }
}

/// Everything the engine needs from the config file.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SyncConfig {
    /// Whether the user has turned external sync on.
    pub enabled: bool,
    /// Whether the playhead is read as a time or as a sample index.
    pub mode: SyncMode,
    /// Where the total sample count comes from in `sample_index` mode.
    pub total_sample_count_source: TotalSampleSource,
    /// Sample rate assumed by [`TotalSampleSource::ReferenceSampleRate`].
    pub reference_sample_rate: u32,
    /// Image name of the DJ program, e.g. `rekordbox.exe`.
    pub memory_process_name: String,
    /// Pin the target by pid instead of by name. `0` means "by name".
    pub memory_process_pid: u32,
    /// Deck 1, preferred when both decks are active.
    pub memory_deck1: DeckConfig,
    /// Deck 2.
    pub memory_deck2: DeckConfig,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: SyncMode::Time,
            total_sample_count_source: TotalSampleSource::ReferenceSampleRate,
            reference_sample_rate: 44_100,
            memory_process_name: String::new(),
            memory_process_pid: 0,
            memory_deck1: DeckConfig::default(),
            memory_deck2: DeckConfig::default(),
        }
    }
}

/// The key the section lives under in `config.json`.
pub const CONFIG_SECTION: &str = "externalsyncconfig";

impl SyncConfig {
    /// Read the section out of a parsed `config.json`.
    ///
    /// A file with no external sync section yields the defaults (disabled),
    /// which is what a first run looks like. A section that is present but
    /// malformed is an error rather than a silent fallback to defaults —
    /// Python's `_coerce_value` turns an unknown `mode` into `"time"` and a
    /// non-numeric sample rate into `0`, so a typo changes what the feature
    /// does without saying anything.
    pub fn from_app_config(cfg: &mixlyzer_core::Config) -> Result<Self, crate::SyncError> {
        match cfg.extra.get(CONFIG_SECTION) {
            None => Ok(Self::default()),
            Some(value) => Self::from_json_value(value.clone()),
        }
    }

    /// Read the section from its JSON value.
    pub fn from_json_value(value: serde_json::Value) -> Result<Self, crate::SyncError> {
        serde_json::from_value(value).map_err(|err| crate::SyncError::Config(err.to_string()))
    }

    /// The deck configs, in the order they are considered.
    pub fn decks(&self) -> [(u8, &DeckConfig); 2] {
        [(1, &self.memory_deck1), (2, &self.memory_deck2)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn python_default_json() -> serde_json::Value {
        serde_json::json!({
            "enabled": false,
            "mode": "time",
            "total_sample_count_source": "reference_sample_rate",
            "reference_sample_rate": 44100,
            "memory_process_name": "",
            "memory_process_pid": 0,
            "memory_deck1": deck_json(),
            "memory_deck2": deck_json(),
        })
    }

    fn deck_json() -> serde_json::Value {
        serde_json::json!({
            "time": {"offsets": "0", "value_type": "float", "length": 0, "encoding": "utf-8", "bit_pos": 0, "multiplier": 1.0},
            "sample_index": {"offsets": "0", "value_type": "int", "length": 0, "encoding": "utf-8", "bit_pos": 0, "multiplier": 1.0},
            "path": {"offsets": "0", "value_type": "str", "length": 2048, "encoding": "utf-8", "bit_pos": 0, "multiplier": 1.0},
            "active": {"offsets": "0", "value_type": "bool", "length": 0, "encoding": "utf-8", "bit_pos": 0, "multiplier": 1.0},
            "loaded": {"offsets": "0", "value_type": "bool", "length": 0, "encoding": "utf-8", "bit_pos": 0, "multiplier": 1.0},
        })
    }

    #[test]
    fn the_defaults_match_the_python_dataclass() {
        let parsed = SyncConfig::from_json_value(python_default_json()).unwrap();
        assert_eq!(parsed, SyncConfig::default());
        assert_eq!(parsed.memory_deck1.path.length, 2048);
        assert_eq!(parsed.reference_sample_rate, 44_100);
        assert!(!parsed.enabled);
    }

    #[test]
    fn a_config_round_trips_through_json() {
        let cfg = SyncConfig {
            enabled: true,
            mode: SyncMode::SampleIndex,
            memory_process_name: "rekordbox.exe".into(),
            ..SyncConfig::default()
        };
        let text = serde_json::to_string(&cfg).unwrap();
        assert!(text.contains("\"sample_index\""), "{text}");
        assert_eq!(
            SyncConfig::from_json_value(serde_json::from_str(&text).unwrap()).unwrap(),
            cfg
        );
    }

    #[test]
    fn the_section_is_read_out_of_a_whole_config_file() {
        let mut app = mixlyzer_core::Config::default();
        app.extra.insert(
            CONFIG_SECTION.to_string(),
            serde_json::json!({
                "enabled": true,
                "mode": "sample_index",
                "total_sample_count_source": "file",
                "reference_sample_rate": 48000,
                "memory_process_name": "serato.exe",
                "memory_process_pid": 0,
                "memory_deck1": deck_json(),
                "memory_deck2": deck_json(),
            }),
        );
        let cfg = SyncConfig::from_app_config(&app).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.mode, SyncMode::SampleIndex);
        assert_eq!(cfg.total_sample_count_source, TotalSampleSource::File);
        assert_eq!(cfg.reference_sample_rate, 48_000);
    }

    #[test]
    fn a_config_without_the_section_is_simply_disabled() {
        let app = mixlyzer_core::Config::default();
        assert_eq!(
            SyncConfig::from_app_config(&app).unwrap(),
            SyncConfig::default()
        );
    }

    /// Python coerces an unknown `mode` to `"time"`, so a typo quietly changes
    /// which memory field is followed.
    #[test]
    fn an_unknown_mode_is_an_error_not_a_silent_fallback() {
        let mut value = python_default_json();
        value["mode"] = serde_json::json!("timecode");
        let err = SyncConfig::from_json_value(value).unwrap_err();
        assert!(matches!(err, crate::SyncError::Config(_)), "{err:?}");
        assert!(err.is_permanent());
    }

    #[test]
    fn an_unknown_value_type_is_an_error() {
        let mut value = python_default_json();
        value["memory_deck1"]["time"]["value_type"] = serde_json::json!("double");
        assert!(SyncConfig::from_json_value(value).is_err());
    }

    #[test]
    fn decks_are_offered_lowest_first() {
        let cfg = SyncConfig::default();
        let decks = cfg.decks();
        assert_eq!(decks[0].0, 1);
        assert_eq!(decks[1].0, 2);
    }
}
