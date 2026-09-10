//! The poll loop: read both decks, pick one, report where it is.
//!
//! [`SyncEngine::poll`] is the whole feature in one call. It is driven by the
//! host at the UI frame rate and returns the state of the deck being followed,
//! or `None` when there is nothing to follow. Everything it needs from the
//! operating system arrives through [`MemoryReader`], so the entire loop —
//! deck selection, chain dereferencing, value decoding, path validation and the
//! failure policy — is exercised in this module's tests without Windows.
//!
//! Differences from `ExternalSyncController`, beyond the failure policy in
//! [`crate::failure`]:
//!
//! * A deck whose state cannot be read is skipped, not fatal. Python's
//!   `_read_deck_state` disables the whole feature if either deck's `loaded`
//!   flag cannot be read — and an empty deck commonly has a null pointer chain,
//!   so an idle deck 2 takes deck 1 down with it. Here the poll only fails when
//!   *no* deck could be read.
//! * The engine holds no Qt objects, emits no signals and never writes
//!   `config.json`. It answers a question; the host decides what to do about
//!   the answer.

use crate::address::AddressChain;
use crate::config::{DeckConfig, SyncConfig, SyncMode};
use crate::denylist::DenylistGuard;
use crate::error::SyncError;
use crate::failure::{DisableReason, FailurePolicy, FailureTracker, SyncStatus};
use crate::path::{validate_track_path, PathRejection, ValidatedPath};
use crate::reader::MemoryReader;
use crate::timing::{sample_index_to_time, total_samples_for, TrackInfo};
use crate::value::{MemoryValue, ValueSpec};

/// What the followed deck is doing.
#[derive(Debug, Clone, PartialEq)]
pub struct DeckState {
    /// Which deck this is, 1 or 2.
    pub deck: u8,
    /// The track the deck reports, once validated. `None` when the deck
    /// reports a path Mixlyzer will not act on (see [`DeckState::path_rejection`]).
    pub path: Option<String>,
    /// [`DeckState::path`] normalised for comparison with the library.
    pub normalized_path: Option<String>,
    /// Playhead in seconds, never negative and never NaN.
    pub time_sec: f64,
    /// Whether the deck is loaded and active.
    pub playing: bool,
    /// Whether this is a different track from the one the last poll saw.
    ///
    /// The host loads a track when this is true and only seeks when it is not,
    /// which is the distinction Python spreads across `_pending_external_path`,
    /// `_load_request_inflight` and `_analysis_request_inflight`.
    pub path_changed: bool,
    /// Why the reported path was not usable, when it was not.
    pub path_rejection: Option<PathRejection>,
}

/// A value spec with its chain already parsed.
#[derive(Debug, Clone)]
struct CompiledValue {
    chain: AddressChain,
    spec: ValueSpec,
}

impl CompiledValue {
    fn compile(spec: &ValueSpec) -> Result<Self, SyncError> {
        // Both the chain and the length are checked once, when the engine is
        // built, rather than on every poll inside a `try/except`.
        let chain = AddressChain::parse(&spec.offsets)?;
        spec.read_len()?;
        Ok(Self {
            chain,
            spec: spec.clone(),
        })
    }

    fn read<R: MemoryReader>(&self, reader: &R) -> Result<MemoryValue, SyncError> {
        let address = self.chain.resolve(reader)?;
        let bytes = reader.read(address, self.spec.read_len()?)?;
        self.spec.decode(&bytes)
    }
}

/// One deck's compiled specs.
#[derive(Debug, Clone)]
struct CompiledDeck {
    number: u8,
    loaded: CompiledValue,
    active: CompiledValue,
    path: CompiledValue,
    playhead: CompiledValue,
}

impl CompiledDeck {
    fn compile(number: u8, cfg: &DeckConfig, mode: SyncMode) -> Result<Self, SyncError> {
        Ok(Self {
            number,
            loaded: CompiledValue::compile(&cfg.loaded)?,
            active: CompiledValue::compile(&cfg.active)?,
            path: CompiledValue::compile(&cfg.path)?,
            // Only the field the configured mode actually reads is compiled, so
            // a stale `sample_index` chain cannot break a time-mode setup.
            playhead: CompiledValue::compile(match mode {
                SyncMode::Time => &cfg.time,
                SyncMode::SampleIndex => &cfg.sample_index,
            })?,
        })
    }
}

/// What the previous poll saw, so a change can be detected.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Followed {
    deck: u8,
    normalized_path: Option<String>,
}

/// Follows an external DJ program's decks.
pub struct SyncEngine<R: MemoryReader> {
    config: SyncConfig,
    reader: R,
    denylist: DenylistGuard,
    track_info: Option<Box<dyn TrackInfo>>,
    decks: Vec<CompiledDeck>,
    failures: FailureTracker,
    followed: Option<Followed>,
}

impl<R: MemoryReader> std::fmt::Debug for SyncEngine<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncEngine")
            .field("config", &self.config)
            .field("status", &self.status())
            .field("followed", &self.followed)
            .field("has_track_info", &self.track_info.is_some())
            .finish()
    }
}

impl<R: MemoryReader> SyncEngine<R> {
    /// Build an engine, compiling every offset chain the configuration uses.
    ///
    /// A malformed chain is reported here, once, instead of failing inside
    /// every poll. The denylist is mandatory: there is no constructor that
    /// attaches to a process without one.
    pub fn new(config: SyncConfig, reader: R, denylist: DenylistGuard) -> Result<Self, SyncError> {
        let decks = compile_decks(&config)?;
        Ok(Self {
            config,
            reader,
            denylist,
            track_info: None,
            decks,
            failures: FailureTracker::default(),
            followed: None,
        })
    }

    /// Supply the library lookups needed by `sample_index` mode.
    pub fn with_track_info(mut self, info: Box<dyn TrackInfo>) -> Self {
        self.track_info = Some(info);
        self
    }

    /// Override the default failure policy.
    pub fn with_failure_policy(mut self, policy: FailurePolicy) -> Self {
        self.failures = FailureTracker::new(policy);
        self
    }

    /// The configuration in force.
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    /// The reader being used.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Replace the reader, e.g. after reattaching to a process that restarted.
    ///
    /// The followed deck is kept: if the same track is still loaded, the host
    /// should not reload it just because the handle is new.
    pub fn set_reader(&mut self, reader: R) {
        self.reader = reader;
    }

    /// Replace the configuration, recompiling the chains and forgetting the
    /// followed deck. Any disable is cleared: new settings deserve a new try.
    pub fn set_config(&mut self, config: SyncConfig) -> Result<(), SyncError> {
        self.decks = compile_decks(&config)?;
        self.config = config;
        self.followed = None;
        self.failures.reset();
        Ok(())
    }

    /// What the engine is doing: active, struggling, backing off, or stopped.
    pub fn status(&self) -> SyncStatus {
        self.failures.status()
    }

    /// Why the engine stopped, if it did.
    pub fn disable_reason(&self) -> Option<&DisableReason> {
        self.failures.disable_reason()
    }

    /// Stop polling, for a reason of the host's own.
    pub fn disable(&mut self, reason: DisableReason) {
        self.failures.disable(reason);
    }

    /// Clear a disable and the failure streak — "turn it back on".
    pub fn reset(&mut self) {
        self.failures.reset();
        self.followed = None;
    }

    /// The deck being followed, if any.
    pub fn followed_deck(&self) -> Option<u8> {
        self.followed.as_ref().map(|f| f.deck)
    }

    /// Read the external program once.
    ///
    /// `Ok(None)` means there is nothing to follow — sync is off, the engine is
    /// backing off or disabled, or neither deck is loaded and active.
    /// `Err` is a failure this poll; consult [`SyncEngine::status`] to find out
    /// whether the engine has given up on it.
    pub fn poll(&mut self) -> Result<Option<DeckState>, SyncError> {
        if self.failures.is_disabled() || !self.config.enabled {
            return Ok(None);
        }
        if self.failures.should_skip() {
            return Ok(None);
        }
        match self.poll_inner() {
            Ok(state) => {
                self.failures.record_success();
                Ok(state)
            }
            Err(err) => {
                self.failures.record_failure(&err);
                Err(err)
            }
        }
    }

    fn poll_inner(&mut self) -> Result<Option<DeckState>, SyncError> {
        self.check_denylist()?;

        let Some(index) = self.select_deck()? else {
            // No deck is loaded and active. That is a healthy answer, not a
            // failure, and it clears whatever we were following.
            self.followed = None;
            return Ok(None);
        };

        let deck = &self.decks[index];
        let number = deck.number;
        let reported = deck
            .path
            .read(&self.reader)?
            .as_str()
            .ok_or_else(|| {
                // Not a transient read problem: no poll will ever produce a
                // path from a spec that is not a `str`.
                SyncError::ValueSpec("the deck path must be a str value".into())
            })?
            .trim()
            .to_string();
        let (path, normalized, rejection) = match validate_track_path(&reported) {
            Ok(ValidatedPath { raw, normalized }) => (Some(raw), Some(normalized), None),
            Err(rejection) => (None, None, Some(rejection)),
        };
        let time_sec = self.read_playhead(deck, normalized.as_deref())?;

        let path_changed = match &self.followed {
            Some(prev) => prev.deck != number || prev.normalized_path != normalized,
            None => true,
        };
        self.followed = Some(Followed {
            deck: number,
            normalized_path: normalized.clone(),
        });

        Ok(Some(DeckState {
            deck: number,
            path,
            normalized_path: normalized,
            time_sec,
            playing: true,
            path_changed,
            path_rejection: rejection,
        }))
    }

    /// Fail-closed denylist check of the configured name and, when the backend
    /// knows it, the real process identity.
    fn check_denylist(&mut self) -> Result<(), SyncError> {
        let mut identity = self.reader.identity();
        if identity.name.trim().is_empty() {
            identity.name = self.config.memory_process_name.clone();
        }
        self.denylist.check(&identity)
    }

    /// The lowest-numbered deck that reports both `loaded` and `active`.
    ///
    /// A deck that cannot be read is skipped; its error is only reported if no
    /// deck at all could be read.
    fn select_deck(&self) -> Result<Option<usize>, SyncError> {
        let mut first_error: Option<SyncError> = None;
        let mut any_read = false;
        for (index, deck) in self.decks.iter().enumerate() {
            match self.read_deck_flags(deck) {
                Ok((loaded, active)) => {
                    any_read = true;
                    if loaded && active {
                        return Ok(Some(index));
                    }
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        match (any_read, first_error) {
            (false, Some(err)) => Err(err),
            _ => Ok(None),
        }
    }

    fn read_deck_flags(&self, deck: &CompiledDeck) -> Result<(bool, bool), SyncError> {
        let loaded = deck.loaded.read(&self.reader)?.truthy();
        let active = deck.active.read(&self.reader)?.truthy();
        Ok((loaded, active))
    }

    fn read_playhead(
        &self,
        deck: &CompiledDeck,
        normalized_path: Option<&str>,
    ) -> Result<f64, SyncError> {
        let value = deck.playhead.read(&self.reader)?;
        let raw = value.as_f64().ok_or_else(|| {
            SyncError::ValueSpec("the playhead must be a float, int or bool value".into())
        })?;
        let seconds = match self.config.mode {
            SyncMode::Time => raw,
            SyncMode::SampleIndex => {
                let Some(path) = normalized_path else {
                    // Without a usable path there is no duration to scale by.
                    return Ok(0.0);
                };
                let duration = self
                    .track_info
                    .as_ref()
                    .and_then(|info| info.duration_sec(path))
                    .unwrap_or(0.0);
                let total = total_samples_for(
                    self.config.total_sample_count_source,
                    duration,
                    self.config.reference_sample_rate,
                    self.track_info
                        .as_ref()
                        .and_then(|info| info.total_samples(path)),
                );
                sample_index_to_time(raw, total, duration)
            }
        };
        Ok(if seconds.is_finite() {
            seconds.max(0.0)
        } else {
            0.0
        })
    }
}

fn compile_decks(config: &SyncConfig) -> Result<Vec<CompiledDeck>, SyncError> {
    config
        .decks()
        .iter()
        .map(|(number, deck)| CompiledDeck::compile(*number, deck, config.mode))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeckConfig;
    use crate::denylist::Denylist;
    use crate::reader::fake::FakeReader;
    use crate::reader::ProcessIdentity;
    use crate::timing::TotalSampleSource;
    use crate::value::ValueType;
    use std::path::PathBuf;

    const BASE: u64 = 0x1000;
    // Deck 1 lives at module+0x100, deck 2 at module+0x300.
    const D1: u64 = 0x100;
    const D2: u64 = 0x300;
    const PATH_LEN: usize = 128;

    fn deck_cfg(base: u64) -> DeckConfig {
        DeckConfig {
            loaded: ValueSpec::new(format!("{:#x}", base), ValueType::Bool),
            active: ValueSpec::new(format!("{:#x}", base + 4), ValueType::Bool),
            time: ValueSpec::new(format!("{:#x}", base + 8), ValueType::Float),
            sample_index: ValueSpec::new(format!("{:#x}", base + 0xC), ValueType::Int),
            path: ValueSpec {
                length: PATH_LEN,
                ..ValueSpec::new(format!("{:#x}", base + 0x10), ValueType::Str)
            },
        }
    }

    fn config() -> SyncConfig {
        SyncConfig {
            enabled: true,
            memory_process_name: "rekordbox.exe".into(),
            memory_deck1: deck_cfg(D1),
            memory_deck2: deck_cfg(D2),
            ..SyncConfig::default()
        }
    }

    fn temp_track(tag: &str) -> (PathBuf, String) {
        let dir = std::env::temp_dir().join(format!(
            "mixlyzer-syncengine-{tag}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("track.flac");
        std::fs::write(&file, b"audio").unwrap();
        let reported = file.to_string_lossy().to_string();
        (file, reported)
    }

    struct Deck {
        loaded: bool,
        active: bool,
        path: String,
        time: f32,
        sample_index: i32,
    }

    impl Deck {
        fn idle() -> Self {
            Self {
                loaded: false,
                active: false,
                path: String::new(),
                time: 0.0,
                sample_index: 0,
            }
        }

        fn playing(path: &str, time: f32) -> Self {
            Self {
                loaded: true,
                active: true,
                path: path.to_string(),
                time,
                sample_index: 0,
            }
        }
    }

    fn target(deck1: &Deck, deck2: &Deck) -> FakeReader {
        let mut reader = FakeReader::new().with_module_base(BASE);
        for (offset, deck) in [(D1, deck1), (D2, deck2)] {
            let at = BASE + offset;
            reader.write_u8(at, u8::from(deck.loaded));
            reader.write_u8(at + 4, u8::from(deck.active));
            reader.write_f32(at + 8, deck.time);
            reader.write_i32(at + 0xC, deck.sample_index);
            reader.write_utf8_buffer(at + 0x10, &deck.path, PATH_LEN);
        }
        reader
    }

    fn permissive() -> DenylistGuard {
        DenylistGuard::fixed(Denylist::empty())
    }

    fn engine(reader: FakeReader) -> SyncEngine<FakeReader> {
        SyncEngine::new(config(), reader, permissive()).unwrap()
    }

    #[test]
    fn a_playing_deck_is_reported_with_its_path_and_time() {
        let (_file, path) = temp_track("basic");
        let mut engine = engine(target(&Deck::playing(&path, 12.5), &Deck::idle()));

        let state = engine.poll().unwrap().expect("deck 1 is playing");
        assert_eq!(state.deck, 1);
        assert_eq!(state.path.as_deref(), Some(path.as_str()));
        assert!((state.time_sec - 12.5).abs() < 1e-6);
        assert!(state.playing);
        assert!(state.path_changed, "the first sighting is a change");
        assert_eq!(engine.followed_deck(), Some(1));
        assert_eq!(engine.status(), SyncStatus::Active);
    }

    #[test]
    fn the_lowest_numbered_active_deck_wins() {
        let (_a, path_a) = temp_track("both-a");
        let (_b, path_b) = temp_track("both-b");
        let mut engine = engine(target(
            &Deck::playing(&path_a, 1.0),
            &Deck::playing(&path_b, 2.0),
        ));
        assert_eq!(engine.poll().unwrap().unwrap().deck, 1);
    }

    #[test]
    fn deck_two_is_followed_when_deck_one_is_idle() {
        let (_file, path) = temp_track("deck2");
        let mut engine = engine(target(&Deck::idle(), &Deck::playing(&path, 3.0)));
        let state = engine.poll().unwrap().unwrap();
        assert_eq!(state.deck, 2);
        assert_eq!(state.path.as_deref(), Some(path.as_str()));
    }

    #[test]
    fn a_deck_that_is_loaded_but_not_active_is_not_followed() {
        let (_file, path) = temp_track("loaded-only");
        let mut deck1 = Deck::playing(&path, 1.0);
        deck1.active = false;
        let mut engine = engine(target(&deck1, &Deck::idle()));
        assert_eq!(engine.poll().unwrap(), None);
        assert_eq!(engine.followed_deck(), None);
    }

    #[test]
    fn no_active_deck_is_not_a_failure() {
        let mut engine = engine(target(&Deck::idle(), &Deck::idle()));
        assert_eq!(engine.poll().unwrap(), None);
        assert_eq!(engine.status(), SyncStatus::Active);
    }

    /// Python takes the whole feature down when either deck's state read
    /// fails, and an empty deck routinely has a null pointer chain.
    #[test]
    fn an_unreadable_deck_does_not_hide_a_readable_one() {
        let (_file, path) = temp_track("one-broken");
        let mut cfg = config();
        // Deck 1's chain dereferences a pointer that is not there.
        cfg.memory_deck1.loaded = ValueSpec::new("0x900, 0x0", ValueType::Bool);
        let reader = target(&Deck::idle(), &Deck::playing(&path, 4.0));
        let mut engine = SyncEngine::new(cfg, reader, permissive()).unwrap();

        let state = engine.poll().unwrap().expect("deck 2 is fine");
        assert_eq!(state.deck, 2);
        assert_eq!(engine.status(), SyncStatus::Active);
    }

    #[test]
    fn a_poll_fails_only_when_no_deck_can_be_read() {
        let mut cfg = config();
        cfg.memory_deck1.loaded = ValueSpec::new("0x900, 0x0", ValueType::Bool);
        cfg.memory_deck2.loaded = ValueSpec::new("0x900, 0x0", ValueType::Bool);
        let reader = target(&Deck::idle(), &Deck::idle());
        let mut engine = SyncEngine::new(cfg, reader, permissive()).unwrap();
        assert!(engine.poll().is_err());
        assert_ne!(engine.status(), SyncStatus::Active);
        assert!(engine.disable_reason().is_none(), "still transient");
    }

    #[test]
    fn the_path_change_flag_tracks_the_previous_poll() {
        let (_a, path_a) = temp_track("change-a");
        let (_b, path_b) = temp_track("change-b");
        let mut engine = engine(target(&Deck::playing(&path_a, 0.0), &Deck::idle()));

        assert!(engine.poll().unwrap().unwrap().path_changed);
        assert!(
            !engine.poll().unwrap().unwrap().path_changed,
            "the same track is not a change"
        );

        let mut cfg = config();
        cfg.memory_deck1 = deck_cfg(D1);
        let reader = target(&Deck::playing(&path_b, 0.0), &Deck::idle());
        // Same engine, new contents in the target.
        engine.set_reader(reader);
        assert!(
            engine.poll().unwrap().unwrap().path_changed,
            "a different track is a change"
        );
    }

    #[test]
    fn switching_decks_counts_as_a_change() {
        let (_a, path) = temp_track("switch");
        let mut engine = engine(target(&Deck::playing(&path, 0.0), &Deck::idle()));
        assert_eq!(engine.poll().unwrap().unwrap().deck, 1);

        engine.set_reader(target(&Deck::idle(), &Deck::playing(&path, 0.0)));
        let state = engine.poll().unwrap().unwrap();
        assert_eq!(state.deck, 2);
        assert!(state.path_changed, "the same file on another deck reloads");
    }

    #[test]
    fn a_path_that_vanishes_stops_being_followed() {
        let (file, path) = temp_track("vanish");
        let mut engine = engine(target(&Deck::playing(&path, 5.0), &Deck::idle()));
        assert!(engine.poll().unwrap().unwrap().path.is_some());

        std::fs::remove_file(&file).unwrap();
        let state = engine.poll().unwrap().expect("the deck is still playing");
        assert_eq!(state.path, None);
        assert!(matches!(
            state.path_rejection,
            Some(PathRejection::Missing(_))
        ));
        assert!(
            (state.time_sec - 5.0).abs() < 1e-6,
            "the playhead still reads"
        );
    }

    #[test]
    fn a_unc_path_is_ignored_but_the_deck_is_still_reported() {
        let mut engine = engine(target(
            &Deck::playing("\\\\nas\\music\\a.flac", 1.0),
            &Deck::idle(),
        ));
        let state = engine.poll().unwrap().unwrap();
        assert_eq!(state.path, None);
        assert!(matches!(state.path_rejection, Some(PathRejection::Unc(_))));
    }

    #[test]
    fn a_relative_path_is_ignored() {
        let mut engine = engine(target(&Deck::playing("music/a.flac", 1.0), &Deck::idle()));
        assert!(matches!(
            engine.poll().unwrap().unwrap().path_rejection,
            Some(PathRejection::Relative(_))
        ));
    }

    #[test]
    fn a_utf16_path_is_read_end_to_end() {
        let (_file, path) = temp_track("utf16");
        let mut cfg = config();
        cfg.memory_deck1.path = ValueSpec {
            length: 256,
            encoding: "utf-16".into(),
            ..ValueSpec::new(format!("{:#x}", D1 + 0x10), ValueType::Str)
        };
        let mut reader = target(&Deck::playing("", 1.0), &Deck::idle());
        reader.write_utf16_buffer(BASE + D1 + 0x10, &path, 256);

        let mut engine = SyncEngine::new(cfg, reader, permissive()).unwrap();
        assert_eq!(
            engine.poll().unwrap().unwrap().path.as_deref(),
            Some(path.as_str()),
            "the Python UTF-16 read would return only the first character"
        );
    }

    #[test]
    fn the_time_multiplier_is_applied() {
        let (_file, path) = temp_track("multiplier");
        let mut cfg = config();
        cfg.memory_deck1.time.multiplier = 0.001; // the deck reports milliseconds
        let mut deck = Deck::playing(&path, 0.0);
        deck.time = 90_000.0;
        let mut engine = SyncEngine::new(cfg, target(&deck, &Deck::idle()), permissive()).unwrap();
        assert!((engine.poll().unwrap().unwrap().time_sec - 90.0).abs() < 1e-6);
    }

    #[test]
    fn a_negative_playhead_is_clamped_to_the_start() {
        let (_file, path) = temp_track("negative");
        let mut deck = Deck::playing(&path, -3.0);
        deck.loaded = true;
        let mut engine = engine(target(&deck, &Deck::idle()));
        assert_eq!(engine.poll().unwrap().unwrap().time_sec, 0.0);
    }

    struct Library {
        duration: f64,
        total_samples: Option<i64>,
    }

    impl TrackInfo for Library {
        fn duration_sec(&self, _normalized_path: &str) -> Option<f64> {
            Some(self.duration)
        }

        fn total_samples(&self, _normalized_path: &str) -> Option<i64> {
            self.total_samples
        }
    }

    fn sample_index_engine(
        source: TotalSampleSource,
        rate: u32,
        library: Library,
        index: i32,
        path: &str,
    ) -> SyncEngine<FakeReader> {
        let mut cfg = config();
        cfg.mode = SyncMode::SampleIndex;
        cfg.total_sample_count_source = source;
        cfg.reference_sample_rate = rate;
        let mut deck = Deck::playing(path, 0.0);
        deck.sample_index = index;
        SyncEngine::new(cfg, target(&deck, &Deck::idle()), permissive())
            .unwrap()
            .with_track_info(Box::new(library))
    }

    #[test]
    fn a_sample_index_becomes_a_time_from_the_reference_rate() {
        let (_file, path) = temp_track("sample-rate");
        let mut engine = sample_index_engine(
            TotalSampleSource::ReferenceSampleRate,
            44_100,
            Library {
                duration: 200.0,
                total_samples: None,
            },
            44_100 * 30,
            &path,
        );
        assert!((engine.poll().unwrap().unwrap().time_sec - 30.0).abs() < 1e-6);
    }

    #[test]
    fn a_sample_index_becomes_a_time_from_the_stored_total() {
        let (_file, path) = temp_track("sample-file");
        let mut engine = sample_index_engine(
            TotalSampleSource::File,
            44_100,
            Library {
                duration: 100.0,
                total_samples: Some(4_800_000),
            },
            2_400_000,
            &path,
        );
        assert!((engine.poll().unwrap().unwrap().time_sec - 50.0).abs() < 1e-6);
    }

    #[test]
    fn a_zero_reference_rate_parks_the_playhead_instead_of_dividing_by_zero() {
        let (_file, path) = temp_track("zero-rate");
        let mut engine = sample_index_engine(
            TotalSampleSource::ReferenceSampleRate,
            0,
            Library {
                duration: 200.0,
                total_samples: None,
            },
            44_100,
            &path,
        );
        assert_eq!(engine.poll().unwrap().unwrap().time_sec, 0.0);
    }

    #[test]
    fn a_sample_index_without_library_information_is_zero() {
        let (_file, path) = temp_track("no-library");
        let mut cfg = config();
        cfg.mode = SyncMode::SampleIndex;
        let mut deck = Deck::playing(&path, 0.0);
        deck.sample_index = 44_100;
        let mut engine = SyncEngine::new(cfg, target(&deck, &Deck::idle()), permissive()).unwrap();
        assert_eq!(engine.poll().unwrap().unwrap().time_sec, 0.0);
    }

    #[test]
    fn a_disabled_configuration_polls_nothing() {
        let (_file, path) = temp_track("disabled");
        let mut cfg = config();
        cfg.enabled = false;
        let mut engine = SyncEngine::new(
            cfg,
            target(&Deck::playing(&path, 1.0), &Deck::idle()),
            permissive(),
        )
        .unwrap();
        assert_eq!(engine.poll().unwrap(), None);
    }

    #[test]
    fn a_denied_process_name_stops_the_engine() {
        let (_file, path) = temp_track("denied");
        let mut cfg = config();
        cfg.memory_process_name = "lsass.exe".into();
        let denylist = DenylistGuard::fixed(Denylist::from_entries(["lsass".to_string()], [], []));
        let mut engine = SyncEngine::new(
            cfg,
            target(&Deck::playing(&path, 1.0), &Deck::idle()),
            denylist,
        )
        .unwrap();

        let err = engine.poll().unwrap_err();
        assert!(matches!(err, SyncError::ProcessDenied { .. }), "{err:?}");
        assert!(matches!(
            engine.status(),
            SyncStatus::Disabled(DisableReason::Denied(_))
        ));
        assert_eq!(engine.poll().unwrap(), None, "and it stays stopped");
    }

    #[test]
    fn a_denied_image_path_from_the_backend_stops_the_engine() {
        let (_file, path) = temp_track("denied-path");
        let reader =
            target(&Deck::playing(&path, 1.0), &Deck::idle()).with_identity(ProcessIdentity {
                pid: 40,
                name: "renamed.exe".into(),
                image_path: "C:\\Windows\\System32\\renamed.exe".into(),
                company: String::new(),
            });
        let denylist = DenylistGuard::fixed(Denylist::from_entries(
            [],
            ["\\windows\\system32\\".to_string()],
            [],
        ));
        let mut engine = SyncEngine::new(config(), reader, denylist).unwrap();
        assert!(matches!(
            engine.poll().unwrap_err(),
            SyncError::ProcessDenied { .. }
        ));
    }

    #[test]
    fn an_unreadable_denylist_blocks_the_poll_without_disabling_it() {
        let (_file, path) = temp_track("denylist-missing");
        let dir = std::env::temp_dir().join(format!(
            "mixlyzer-syncdeny-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let list_path = dir.join("process_denylist.json");
        let mut engine = SyncEngine::new(
            config(),
            target(&Deck::playing(&path, 1.0), &Deck::idle()),
            DenylistGuard::from_file(&list_path),
        )
        .unwrap();

        assert!(matches!(
            engine.poll().unwrap_err(),
            SyncError::DenylistUnavailable(_)
        ));
        assert_ne!(engine.status(), SyncStatus::Active);
        assert!(
            engine.disable_reason().is_none(),
            "fail closed, not fail over"
        );

        std::fs::write(&list_path, r#"{"blocked_process_names":["lsass"]}"#).unwrap();
        assert!(
            engine.poll().unwrap().is_some(),
            "a denylist that becomes readable must let sync resume"
        );
        assert_eq!(engine.status(), SyncStatus::Active);
    }

    /// The behaviour this crate exists to fix: Python's
    /// `_disable_due_to_failure` turns Memory Sync off on the first read
    /// failure and writes `enabled=false` to `config.json`.
    #[test]
    fn a_transient_failure_recovers_and_a_permanent_one_disables() {
        let (_file, path) = temp_track("policy");
        let mut engine = engine(target(&Deck::playing(&path, 7.0), &Deck::idle()));
        assert!(engine.poll().unwrap().is_some());

        // A read fails the way a track load makes it fail.
        engine
            .reader()
            .fail_with(SyncError::NullPointer { step: 1 });
        assert!(engine.poll().is_err());
        assert!(
            engine.disable_reason().is_none(),
            "one null pointer must not end the session"
        );

        engine.reader().clear_error();
        let state = engine.poll().unwrap().expect("recovered");
        assert!((state.time_sec - 7.0).abs() < 1e-6);
        assert_eq!(engine.status(), SyncStatus::Active);

        // The process exiting is not something to wait out.
        engine.reader().fail_with(SyncError::ProcessGone);
        assert_eq!(engine.poll().unwrap_err(), SyncError::ProcessGone);
        assert_eq!(
            engine.status(),
            SyncStatus::Disabled(DisableReason::ProcessGone)
        );
        assert_eq!(engine.poll().unwrap(), None);

        // Only an explicit reset brings it back — nothing was written to disk.
        engine.reader().clear_error();
        engine.reset();
        assert!(engine.poll().unwrap().is_some());
    }

    #[test]
    fn repeated_transient_failures_back_off_and_skip_polls() {
        let (_file, path) = temp_track("backoff");
        let mut engine = engine(target(&Deck::playing(&path, 1.0), &Deck::idle()))
            .with_failure_policy(FailurePolicy {
                tolerated_failures: 1,
                backoff_polls: 2,
                max_backoff_polls: 4,
                disable_after: None,
            });
        engine.reader().fail_with(SyncError::Read {
            address: 0x10,
            len: 4,
            detail: "unmapped".into(),
        });
        assert!(engine.poll().is_err());
        assert!(engine.poll().is_err());
        assert!(matches!(engine.status(), SyncStatus::BackingOff { .. }));

        // Backed-off polls do nothing and cost nothing.
        engine.reader().clear_error();
        assert_eq!(engine.poll().unwrap(), None);
        assert_eq!(engine.poll().unwrap(), None);
        assert!(engine.poll().unwrap().is_some(), "then it tries again");
        assert!(engine.disable_reason().is_none());
    }

    /// A field configured as the wrong type can never work, so it stops the
    /// engine rather than failing every poll for the rest of the session.
    #[test]
    fn a_field_of_the_wrong_type_is_a_configuration_error() {
        let (_file, path) = temp_track("wrong-type");
        let mut cfg = config();
        cfg.memory_deck1.path = ValueSpec::new(format!("{:#x}", D1 + 0x10), ValueType::Int);
        let mut engine = SyncEngine::new(
            cfg,
            target(&Deck::playing(&path, 1.0), &Deck::idle()),
            permissive(),
        )
        .unwrap();
        assert!(matches!(
            engine.poll().unwrap_err(),
            SyncError::ValueSpec(_)
        ));
        assert!(matches!(
            engine.status(),
            SyncStatus::Disabled(DisableReason::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn a_malformed_chain_is_reported_when_the_engine_is_built() {
        let mut cfg = config();
        cfg.memory_deck2.path.offsets = "0x10,,0x20".into();
        let err =
            SyncEngine::new(cfg, target(&Deck::idle(), &Deck::idle()), permissive()).unwrap_err();
        assert!(matches!(err, SyncError::Address(_)), "{err:?}");
    }

    /// A chain that the configured mode never reads must not block startup.
    #[test]
    fn only_the_playhead_field_in_use_is_compiled() {
        let mut cfg = config();
        cfg.mode = SyncMode::Time;
        cfg.memory_deck1.sample_index.offsets = "nonsense".into();
        assert!(SyncEngine::new(
            cfg.clone(),
            target(&Deck::idle(), &Deck::idle()),
            permissive()
        )
        .is_ok());

        cfg.mode = SyncMode::SampleIndex;
        assert!(SyncEngine::new(cfg, target(&Deck::idle(), &Deck::idle()), permissive()).is_err());
    }

    #[test]
    fn a_new_configuration_recompiles_and_clears_the_state() {
        let (_file, path) = temp_track("reconfig");
        let mut engine = engine(target(&Deck::playing(&path, 1.0), &Deck::idle()));
        assert!(engine.poll().unwrap().is_some());
        assert_eq!(engine.followed_deck(), Some(1));

        let mut cfg = config();
        cfg.enabled = false;
        engine.set_config(cfg).unwrap();
        assert_eq!(engine.followed_deck(), None);
        assert_eq!(engine.poll().unwrap(), None);

        let mut bad = config();
        bad.memory_deck1.loaded.offsets = "zz".into();
        assert!(engine.set_config(bad).is_err());
    }

    #[test]
    fn the_host_can_stop_and_restart_the_engine() {
        let (_file, path) = temp_track("stop");
        let mut engine = engine(target(&Deck::playing(&path, 1.0), &Deck::idle()));
        engine.disable(DisableReason::Stopped);
        assert_eq!(engine.poll().unwrap(), None);
        engine.reset();
        assert!(engine.poll().unwrap().is_some());
    }

    #[test]
    fn a_multi_step_chain_is_followed_to_the_deck() {
        let (_file, path) = temp_track("chain");
        let mut cfg = config();
        // deck 1's `loaded` flag now lives behind two pointers.
        cfg.memory_deck1.loaded = ValueSpec::new("0x800, 0x8, 0x2", ValueType::Bool);
        let mut reader = target(&Deck::playing(&path, 1.0), &Deck::idle());
        reader.write_u64(BASE + 0x800, 0x5000);
        reader.write_u64(0x5008, 0x6000);
        reader.write_bytes(0x6000, &[0, 0, 1, 0]);

        let mut engine = SyncEngine::new(cfg, reader, permissive()).unwrap();
        assert_eq!(engine.poll().unwrap().unwrap().deck, 1);
    }

    #[test]
    fn the_engine_prints_its_state_for_a_log_line() {
        let engine = engine(target(&Deck::idle(), &Deck::idle()));
        let text = format!("{engine:?}");
        assert!(text.contains("SyncEngine"), "{text}");
    }
}
