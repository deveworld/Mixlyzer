//! The library schema migration runner: 0.1.0 -> 0.1.1 -> 0.2.0 -> 0.3.0.
//!
//! A port of `migration/migration.py` and its three step modules, with the
//! failure semantics rebuilt. Three things in the Python version are fixed
//! here, each pinned by a test whose name states the behaviour.
//!
//! **1. A count is not an error code.** Python's steps return an `int` and the
//! runner does `if int(result or 0) != 0: raise`. The 0.2.0 -> 0.3.0 step ends
//! with `return converted` — *the number of files it converted*. So every
//! library holding at least one track reports failure, the `VERSION` file is
//! never stamped, and the same doomed migration runs again on the next launch.
//! Here a step returns [`MigrationOutcome`], whose [`MigrationOutcome::tracks_converted`]
//! is a count and whose success is the `Ok` variant. A count cannot be mistaken
//! for a status because the two live in different types.
//!
//! **2. Absent is not unreadable.** Python's `read_library_version` catches
//! every exception and returns the *oldest* version, so a 0.3.0 library whose
//! `VERSION` file is unreadable (permissions, a bad mount, non-text bytes)
//! quietly re-runs the entire chain. [`read_library_version`] returns `None`
//! only when the file is genuinely not there, and errors otherwise.
//!
//! **3. One bad track is not a failed migration.** Python's 0.1.1 -> 0.2.0 step
//! returns non-zero if any single track could not be converted, which — via
//! bug 1 — aborts the chain and the app. Per-track problems land in
//! [`MigrationOutcome::tracks_skipped`] and the step still succeeds; only
//! errors that make further work impossible (an unopenable database, a failed
//! write) return `Err`.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use mixlyzer_core::linear::{build_bpm_segments, build_key_segments};
use mixlyzer_core::LIBRARY_VERSION;

use crate::error::StoreError;
use crate::features::{write_atomic, FeatureStore};
use crate::library::Library;

/// The version assumed for a library that has never been stamped.
pub const DEFAULT_LIBRARY_VERSION: &str = "0.1.0";

/// Name of the file holding the library's schema version.
pub const VERSION_FILENAME: &str = "VERSION";

/// Name of the library database inside the library directory.
pub const LIBRARY_DB_FILENAME: &str = "library.db";

/// Legacy feature block dropped by the 0.1.1 -> 0.2.0 step.
///
/// Waveform images were cached alongside the analysis and are regenerated on
/// demand; keeping them made every feature file an order of magnitude larger.
const LEGACY_WAVE_IMAGE_PREFIX: &str = "wave_img_np";

/// Where the `VERSION` file lives for a library directory.
pub fn version_file_path(lib_path: impl AsRef<Path>) -> PathBuf {
    lib_path.as_ref().join(VERSION_FILENAME)
}

/// Where the database lives for a library directory.
pub fn database_path(lib_path: impl AsRef<Path>) -> PathBuf {
    lib_path.as_ref().join(LIBRARY_DB_FILENAME)
}

/// Read the stamped library version.
///
/// Returns `Ok(None)` when no version has been stamped — the file is absent, or
/// is present but empty, which is what an interrupted write leaves behind. Any
/// other failure (permissions, a directory in its place, bytes that are not
/// UTF-8) is [`StoreError::UnreadableVersion`], because guessing "oldest" for
/// those would re-run every migration against an already-current library.
pub fn read_library_version(lib_path: impl AsRef<Path>) -> Result<Option<String>, StoreError> {
    let path = version_file_path(lib_path);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(StoreError::UnreadableVersion {
                path,
                detail: err.to_string(),
            })
        }
    };
    let text = String::from_utf8(bytes).map_err(|err| StoreError::UnreadableVersion {
        path: path.clone(),
        detail: format!("file is not UTF-8 text: {err}"),
    })?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed.to_string()))
}

/// The version to migrate *from*: the stamped one, or [`DEFAULT_LIBRARY_VERSION`].
pub fn effective_library_version(lib_path: impl AsRef<Path>) -> Result<String, StoreError> {
    Ok(read_library_version(lib_path)?.unwrap_or_else(|| DEFAULT_LIBRARY_VERSION.to_string()))
}

/// Stamp the library with `version`.
///
/// Written atomically: a half-written `VERSION` is exactly the state that makes
/// the next launch's decision unsound.
pub fn write_library_version(
    lib_path: impl AsRef<Path>,
    version: &str,
) -> Result<(), StoreError> {
    let lib_path = lib_path.as_ref();
    fs::create_dir_all(lib_path).map_err(|e| StoreError::io(lib_path, e))?;
    let path = version_file_path(lib_path);
    write_atomic(&path, format!("{}\n", version.trim()).as_bytes())
}

/// Stamp the current version if the library has never been stamped.
pub fn ensure_version_file(lib_path: impl AsRef<Path>) -> Result<(), StoreError> {
    if read_library_version(&lib_path)?.is_none() {
        write_library_version(&lib_path, LIBRARY_VERSION)?;
    }
    Ok(())
}

/// One track a step could not convert, and why.
///
/// A skip is information, not failure: the track keeps working at its current
/// fidelity and can be re-analysed later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedTrack {
    /// Something a user can recognise: the title, else the path.
    pub label: String,
    /// The track uid, when it had one.
    pub uid: Option<String>,
    /// Why the track was skipped.
    pub reason: String,
}

/// What one migration step did.
///
/// The count lives here, in the success value, precisely so that it can never
/// be read as a status code the way Python's `return converted` is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationOutcome {
    /// The version this step migrated from.
    pub from: &'static str,
    /// The version this step migrated to.
    pub to: &'static str,
    /// Tracks (or feature files) actually brought to the new shape.
    pub tracks_converted: usize,
    /// Tracks the step could not convert, each with a reason.
    pub tracks_skipped: Vec<SkippedTrack>,
}

impl MigrationOutcome {
    fn new(from: &'static str, to: &'static str) -> Self {
        Self {
            from,
            to,
            tracks_converted: 0,
            tracks_skipped: Vec::new(),
        }
    }

    fn skip(&mut self, label: impl Into<String>, uid: Option<String>, reason: impl fmt::Display) {
        self.tracks_skipped.push(SkippedTrack {
            label: label.into(),
            uid,
            reason: reason.to_string(),
        });
    }

    /// Whether every track was converted.
    pub fn is_clean(&self) -> bool {
        self.tracks_skipped.is_empty()
    }
}

/// One version-to-version migration.
#[derive(Clone, Copy)]
pub struct Step {
    from: &'static str,
    to: &'static str,
    apply: fn(&Path) -> Result<MigrationOutcome, StoreError>,
}

impl Step {
    /// The version this step migrates from.
    pub fn from_version(&self) -> &'static str {
        self.from
    }

    /// The version this step migrates to.
    pub fn to_version(&self) -> &'static str {
        self.to
    }

    /// Run the step against a library directory.
    ///
    /// Does not stamp `VERSION`; [`run`] does that after each step succeeds, so
    /// an interrupted chain resumes from the last completed step.
    pub fn run(&self, lib_path: impl AsRef<Path>) -> Result<MigrationOutcome, StoreError> {
        (self.apply)(lib_path.as_ref())
    }
}

impl fmt::Debug for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Step({} -> {})", self.from, self.to)
    }
}

/// The migration chain, in order.
static STEPS: [Step; 3] = [
    Step {
        from: "0.1.0",
        to: "0.1.1",
        apply: step_0_1_0_to_0_1_1,
    },
    Step {
        from: "0.1.1",
        to: "0.2.0",
        apply: step_0_1_1_to_0_2_0,
    },
    Step {
        from: "0.2.0",
        to: "0.3.0",
        apply: step_0_2_0_to_0_3_0,
    },
];

/// Every known migration step, oldest first.
pub fn steps() -> &'static [Step] {
    &STEPS
}

/// The steps that take a library from `from` to `to`.
///
/// An empty plan means the library is already at `to`. Errors when no chain
/// connects the two versions, or when the chain would loop.
pub fn plan(from: &str, to: &str) -> Result<Vec<&'static Step>, StoreError> {
    let from = from.trim();
    let to = to.trim();
    let mut current = from;
    let mut chain = Vec::new();
    let mut visited: Vec<&str> = Vec::new();
    while current != to {
        if visited.contains(&current) {
            return Err(StoreError::NoMigrationPath {
                from: from.to_string(),
                to: to.to_string(),
                detail: format!("migration chain loops at {current}"),
            });
        }
        visited.push(current);
        let Some(step) = STEPS.iter().find(|s| s.from == current) else {
            return Err(StoreError::NoMigrationPath {
                from: from.to_string(),
                to: to.to_string(),
                detail: format!("no step leaves version {current}"),
            });
        };
        chain.push(step);
        current = step.to;
    }
    Ok(chain)
}

/// What a whole [`run`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// The version the library was at before the run.
    pub from: String,
    /// The version it is at now.
    pub to: String,
    /// One entry per step executed, in order. Empty when nothing was needed.
    pub outcomes: Vec<MigrationOutcome>,
}

impl MigrationReport {
    /// Total tracks converted across every step.
    pub fn tracks_converted(&self) -> usize {
        self.outcomes.iter().map(|o| o.tracks_converted).sum()
    }

    /// Every track any step had to skip.
    pub fn skipped(&self) -> impl Iterator<Item = &SkippedTrack> {
        self.outcomes.iter().flat_map(|o| o.tracks_skipped.iter())
    }

    /// Whether the run needed no steps at all.
    pub fn was_already_current(&self) -> bool {
        self.outcomes.is_empty()
    }
}

/// Bring a library directory up to [`mixlyzer_core::LIBRARY_VERSION`].
///
/// `VERSION` is stamped after each successful step, so a run interrupted
/// halfway resumes from where it stopped rather than starting over. A library
/// that is already current is stamped (if it never was) and returns an empty
/// report.
pub fn run(lib_path: impl AsRef<Path>) -> Result<MigrationReport, StoreError> {
    run_to(lib_path, LIBRARY_VERSION)
}

/// [`run`], but stopping at `target` instead of the current version. For tests
/// and for stepping a library forward one version at a time.
pub fn run_to(lib_path: impl AsRef<Path>, target: &str) -> Result<MigrationReport, StoreError> {
    let lib_path = lib_path.as_ref();
    let from = effective_library_version(lib_path)?;
    let chain = plan(&from, target)?;
    let mut outcomes = Vec::with_capacity(chain.len());
    for step in chain {
        let outcome = step.run(lib_path)?;
        // Stamping after each step is what makes an interrupted chain
        // resumable. Python stamps too, but never reaches this line for the
        // last step because of the count-as-error-code bug.
        write_library_version(lib_path, step.to)?;
        outcomes.push(outcome);
    }
    if outcomes.is_empty() {
        ensure_version_file(lib_path)?;
    }
    Ok(MigrationReport {
        from,
        to: target.to_string(),
        outcomes,
    })
}

/// Open the library database for a step, or `None` when there is no database.
///
/// Python raises `FileNotFoundError` here, so migrating a library directory
/// that has no `library.db` yet — a brand-new library, or one whose tracks were
/// never scanned — aborts the chain. Nothing to migrate is not an error.
fn open_library_if_present(lib_path: &Path) -> Result<Option<Library>, StoreError> {
    let db_path = database_path(lib_path);
    if !db_path.exists() {
        return Ok(None);
    }
    Library::open(&db_path).map(Some)
}

/// 0.1.0 -> 0.1.1: add `tracks.total_samples`.
///
/// Python also probes each track's sample count with a bundled `ffmpeg.exe` and
/// fails the step when any probe fails. This crate has no decoder — that is the
/// audio crate's job — so the column is added and every track still missing a
/// sample count is reported as a skip for the analyzer to fill in later. The
/// column is what the schema needs; the values are not load-bearing until a
/// track is re-analysed.
fn step_0_1_0_to_0_1_1(lib_path: &Path) -> Result<MigrationOutcome, StoreError> {
    let mut outcome = MigrationOutcome::new("0.1.0", "0.1.1");
    let Some(library) = open_library_if_present(lib_path)? else {
        return Ok(outcome);
    };
    if !library.column_exists("tracks", "total_samples")? {
        library
            .connection()
            .execute("ALTER TABLE tracks ADD COLUMN total_samples INTEGER;", [])
            .map_err(|source| StoreError::Schema {
                path: database_path(lib_path),
                context: "add tracks.total_samples".to_string(),
                source,
            })?;
    }
    for track in library.list_all()? {
        if track.total_samples.unwrap_or(0) <= 0 {
            let label = if track.title.is_empty() {
                track.path.clone()
            } else {
                track.title.clone()
            };
            outcome.skip(
                label,
                track.uid.clone(),
                "total_samples unknown; will be filled in by the next analysis",
            );
        }
    }
    Ok(outcome)
}

/// 0.1.1 -> 0.2.0: populate the segment tables from each track's feature file
/// and drop the cached waveform image.
///
/// A track without a uid, without a feature file, or with a damaged one is
/// skipped. Python treats each of those as a step failure.
fn step_0_1_1_to_0_2_0(lib_path: &Path) -> Result<MigrationOutcome, StoreError> {
    let mut outcome = MigrationOutcome::new("0.1.1", "0.2.0");
    let Some(library) = open_library_if_present(lib_path)? else {
        return Ok(outcome);
    };
    let store = FeatureStore::new(lib_path);

    for track in library.list_all()? {
        let label = if track.title.is_empty() {
            track.path.clone()
        } else {
            track.title.clone()
        };
        let Some(uid) = track.uid.clone() else {
            outcome.skip(label, None, "track has no uid, so its features cannot be found");
            continue;
        };
        let mut features = match store.load_optional(&uid) {
            Ok(Some(features)) => features,
            Ok(None) => {
                outcome.skip(label, Some(uid), "no feature file; re-analyse this track");
                continue;
            }
            Err(err) => {
                outcome.skip(label, Some(uid), err);
                continue;
            }
        };

        let bpm_rows = build_bpm_segments(&features.tempo_segments());
        let key_rows = build_key_segments(&features.key_segments());
        // A database write failure is not a per-track problem: the next track
        // would fail the same way, so it stops the step.
        library.replace_bpm_segments(&uid, &bpm_rows)?;
        library.replace_key_segments(&uid, &key_rows)?;

        if features.remove_prefixed(LEGACY_WAVE_IMAGE_PREFIX) > 0 {
            if let Err(err) = store.save(&uid, &features) {
                // The segment rows are already in, so the track is migrated;
                // only the size saving was lost.
                outcome.skip(label, Some(uid), err);
                continue;
            }
        }
        outcome.tracks_converted += 1;
    }
    Ok(outcome)
}

/// 0.2.0 -> 0.3.0: give every feature file the canonical, empty cue-point
/// block. Phrase data stays optional, so there is nothing to add for it.
///
/// Files that already carry the block are left alone, which makes the step
/// idempotent: re-running it converts nothing and rewrites nothing. Python
/// rewrites every file every time and returns the count, which is the value the
/// runner then reads as an error code.
fn step_0_2_0_to_0_3_0(lib_path: &Path) -> Result<MigrationOutcome, StoreError> {
    let mut outcome = MigrationOutcome::new("0.2.0", "0.3.0");
    let store = FeatureStore::new(lib_path);
    for uid in store.list_uids()? {
        let mut features = match store.load_optional(&uid) {
            Ok(Some(features)) => features,
            Ok(None) => continue, // removed between listing and loading
            Err(err) => {
                outcome.skip(uid.clone(), Some(uid), err);
                continue;
            }
        };
        if features.has_cue_point_block() {
            continue;
        }
        features.set_cue_points(&[]);
        if let Err(err) = store.save(&uid, &features) {
            outcome.skip(uid.clone(), Some(uid), err);
            continue;
        }
        outcome.tracks_converted += 1;
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FeatureFile, FeatureValue};
    use crate::testutil::TempDir;
    use mixlyzer_core::key::{Key, Mode};
    use mixlyzer_core::segments::{KeySegment, TempoSegment};
    use mixlyzer_core::track::new_uid;
    use mixlyzer_core::Track;

    /// Build a library directory holding `count` analysed tracks, stamped at
    /// `version`. This is the shape of a real 0.2.0 library: a database with
    /// tracks and one feature file each, no cue-point block anywhere.
    fn library_at(dir: &TempDir, version: &str, count: usize) -> Vec<String> {
        let library = Library::open(database_path(dir.path())).unwrap();
        let store = FeatureStore::new(dir.path());
        let mut uids = Vec::new();
        for i in 0..count {
            let uid = new_uid();
            let mut track = Track::new(&format!("/music/track {i}.flac"));
            track.uid = Some(uid.clone());
            track.title = format!("트랙 {i}");
            track.added_ts = 1_700_000_000 + i as i64;
            track.total_samples = Some(1_000_000);
            library.upsert(&track).unwrap();

            let mut features = FeatureFile::new();
            features.set_beats_time_sec(&[0.0, 0.5, 1.0, 1.5]);
            features.set_tempo_segments(&[
                TempoSegment::new(0.0, 30.0, 128.0, 0.0, 4),
                TempoSegment::new(30.0, 60.0, 140.0, 30.0, 4),
            ]);
            features.set_key_segments(&[
                KeySegment::new(0.0, 30.0, Key::new(9, Mode::Minor)),
                KeySegment::new(30.0, 60.0, Key::new(0, Mode::Major)),
            ]);
            store.save(&uid, &features).unwrap();
            uids.push(uid);
        }
        write_library_version(dir.path(), version).unwrap();
        uids
    }

    // ----- the VERSION file ----------------------------------------------

    #[test]
    fn an_absent_version_file_reads_as_unstamped() {
        let dir = TempDir::new("noversion");
        assert_eq!(read_library_version(dir.path()).unwrap(), None);
        assert_eq!(
            effective_library_version(dir.path()).unwrap(),
            DEFAULT_LIBRARY_VERSION
        );
    }

    #[test]
    fn a_version_file_round_trips_and_ignores_surrounding_whitespace() {
        let dir = TempDir::new("version");
        write_library_version(dir.path(), "  0.2.0\n").unwrap();
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.2.0")
        );
    }

    #[test]
    fn an_empty_version_file_reads_as_unstamped() {
        let dir = TempDir::new("emptyversion");
        std::fs::write(version_file_path(dir.path()), b"   \n").unwrap();
        assert_eq!(read_library_version(dir.path()).unwrap(), None);
    }

    /// Python maps *every* read failure to the oldest version, so a current
    /// library with an unreadable `VERSION` re-runs the whole chain.
    #[test]
    fn an_unreadable_version_file_is_an_error_not_the_oldest_version() {
        let dir = TempDir::new("badversion");
        let path = version_file_path(dir.path());
        std::fs::write(&path, [0xFF, 0xFE, 0x00, 0x9C]).unwrap();

        let err = read_library_version(dir.path()).unwrap_err();
        assert!(matches!(err, StoreError::UnreadableVersion { .. }), "{err:?}");
        assert_eq!(err.path(), Some(path.as_path()));
        assert!(effective_library_version(dir.path()).is_err());
        assert!(
            run(dir.path()).is_err(),
            "an unreadable version must stop the run, not restart the chain"
        );
    }

    #[test]
    fn a_directory_where_the_version_file_should_be_is_an_error() {
        let dir = TempDir::new("versiondir");
        std::fs::create_dir_all(version_file_path(dir.path())).unwrap();
        assert!(matches!(
            read_library_version(dir.path()),
            Err(StoreError::UnreadableVersion { .. })
        ));
    }

    #[test]
    fn ensure_version_file_stamps_only_when_unstamped() {
        let dir = TempDir::new("ensure");
        ensure_version_file(dir.path()).unwrap();
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some(LIBRARY_VERSION)
        );
        write_library_version(dir.path(), "0.1.1").unwrap();
        ensure_version_file(dir.path()).unwrap();
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.1.1")
        );
    }

    // ----- planning -------------------------------------------------------

    #[test]
    fn the_chain_covers_every_known_version_in_order() {
        let chain = plan(DEFAULT_LIBRARY_VERSION, LIBRARY_VERSION).unwrap();
        let hops: Vec<(&str, &str)> = chain
            .iter()
            .map(|s| (s.from_version(), s.to_version()))
            .collect();
        assert_eq!(
            hops,
            vec![("0.1.0", "0.1.1"), ("0.1.1", "0.2.0"), ("0.2.0", "0.3.0")]
        );
        assert_eq!(steps().len(), 3);
    }

    #[test]
    fn planning_from_the_current_version_needs_no_steps() {
        assert!(plan(LIBRARY_VERSION, LIBRARY_VERSION).unwrap().is_empty());
    }

    #[test]
    fn planning_from_an_intermediate_version_starts_there() {
        let chain = plan("0.2.0", LIBRARY_VERSION).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].from_version(), "0.2.0");
    }

    #[test]
    fn planning_from_an_unknown_version_is_a_typed_error() {
        let err = plan("9.9.9", LIBRARY_VERSION).unwrap_err();
        assert!(matches!(err, StoreError::NoMigrationPath { .. }), "{err:?}");
        assert!(err.to_string().contains("9.9.9"));
    }

    #[test]
    fn planning_backwards_is_refused_rather_than_looping() {
        let err = plan(LIBRARY_VERSION, "0.1.0").unwrap_err();
        assert!(matches!(err, StoreError::NoMigrationPath { .. }), "{err:?}");
    }

    // ----- the whole chain ------------------------------------------------

    /// The scenario that fails on every launch in Python: a 0.2.0 library with
    /// tracks in it. Its 0.2.0 -> 0.3.0 step ends with `return converted`, the
    /// runner reads any non-zero return as failure, and the library is never
    /// stamped 0.3.0.
    #[test]
    fn a_zero_two_zero_library_with_three_tracks_migrates_and_stamps_zero_three_zero() {
        let dir = TempDir::new("chain");
        let uids = library_at(&dir, "0.2.0", 3);

        let report = run(dir.path()).expect("a library with tracks must migrate");

        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.3.0"),
            "the VERSION file must be stamped"
        );
        assert_eq!(report.from, "0.2.0");
        assert_eq!(report.to, "0.3.0");
        assert_eq!(report.outcomes.len(), 1);
        assert_eq!(report.tracks_converted(), 3);
        assert_eq!(report.skipped().count(), 0);

        // Every track came out with the canonical cue-point block.
        let store = FeatureStore::new(dir.path());
        for uid in &uids {
            let features = store.load(uid).unwrap();
            assert!(features.has_cue_point_block(), "{uid} kept no cue block");
            assert!(features.cue_points().is_empty());
            assert_eq!(features.beats_time_sec().len(), 4, "{uid} lost its beats");
            assert_eq!(features.tempo_segments().len(), 2);
        }
    }

    /// Named for the bug it guards: the number of converted tracks lives in the
    /// success value, so no amount of work can be read as a failure.
    #[test]
    fn a_converted_count_is_never_read_as_an_error_code() {
        let stepwise = TempDir::new("count-step");
        library_at(&stepwise, "0.2.0", 5);
        let outcome = steps()[2].run(stepwise.path()).expect("step must succeed");
        assert_eq!(outcome.tracks_converted, 5);
        assert!(outcome.is_clean());

        // The same work through the runner, on its own library so the step has
        // something left to do: success, five conversions, no error.
        let whole = TempDir::new("count-run");
        library_at(&whole, "0.2.0", 5);
        let report = run(whole.path()).expect("five conversions is not a failure");
        assert_eq!(report.tracks_converted(), 5);
    }

    #[test]
    fn migrating_twice_is_a_no_op_the_second_time() {
        let dir = TempDir::new("idempotent");
        library_at(&dir, "0.2.0", 2);
        assert_eq!(run(dir.path()).unwrap().tracks_converted(), 2);

        let second = run(dir.path()).unwrap();
        assert!(second.was_already_current());
        assert_eq!(second.tracks_converted(), 0);
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.3.0")
        );
    }

    #[test]
    fn the_full_chain_runs_from_the_oldest_version() {
        let dir = TempDir::new("full");
        library_at(&dir, "0.1.0", 2);
        let report = run(dir.path()).unwrap();
        assert_eq!(report.outcomes.len(), 3);
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.3.0")
        );
    }

    /// Each step stamps as it completes, so an interrupted chain picks up where
    /// it stopped instead of redoing finished work.
    #[test]
    fn an_interrupted_chain_resumes_from_the_last_stamped_version() {
        let dir = TempDir::new("resume");
        library_at(&dir, "0.1.1", 2);

        run_to(dir.path(), "0.2.0").unwrap();
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.2.0")
        );

        let rest = run(dir.path()).unwrap();
        assert_eq!(rest.from, "0.2.0");
        assert_eq!(rest.outcomes.len(), 1, "only the remaining step re-runs");
    }

    #[test]
    fn a_fresh_empty_library_is_stamped_at_the_current_version() {
        let dir = TempDir::new("fresh");
        write_library_version(dir.path(), LIBRARY_VERSION).unwrap();
        let report = run(dir.path()).unwrap();
        assert!(report.was_already_current());
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some(LIBRARY_VERSION)
        );
    }

    /// Python raises `FileNotFoundError` from the first two steps when there is
    /// no `library.db`, so a library directory that has never been scanned
    /// cannot be migrated at all.
    #[test]
    fn a_library_directory_with_no_database_migrates_cleanly() {
        let dir = TempDir::new("nodb");
        let report = run(dir.path()).unwrap();
        assert_eq!(report.outcomes.len(), 3);
        assert_eq!(report.tracks_converted(), 0);
        assert_eq!(report.skipped().count(), 0);
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some(LIBRARY_VERSION)
        );
    }

    // ----- per-track resilience ------------------------------------------

    /// Python aborts the chain — and the app — when one track cannot be
    /// converted. The rest of the library must not be held hostage.
    #[test]
    fn a_damaged_feature_file_is_skipped_and_the_rest_still_migrate() {
        let dir = TempDir::new("damaged");
        let uids = library_at(&dir, "0.2.0", 3);
        let store = FeatureStore::new(dir.path());
        std::fs::write(store.path_for(&uids[1]).unwrap(), b"not a feature file").unwrap();

        let report = run(dir.path()).expect("one damaged file must not fail the run");
        assert_eq!(report.tracks_converted(), 2);
        let skipped: Vec<&SkippedTrack> = report.skipped().collect();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].uid.as_deref(), Some(uids[1].as_str()));
        assert!(
            skipped[0].reason.contains("corrupt"),
            "the reason must say what happened: {}",
            skipped[0].reason
        );
        assert_eq!(
            read_library_version(dir.path()).unwrap().as_deref(),
            Some("0.3.0"),
            "the library is still stamped"
        );
    }

    #[test]
    fn a_track_without_a_uid_is_skipped_not_fatal() {
        let dir = TempDir::new("nouid");
        library_at(&dir, "0.1.1", 1);
        {
            let library = Library::open(database_path(dir.path())).unwrap();
            library
                .connection()
                .execute(
                    "INSERT INTO tracks(path, uid, title, added_ts) VALUES('/x.flac', NULL, 'No UID', 1);",
                    [],
                )
                .unwrap();
        }
        let report = run(dir.path()).unwrap();
        let skipped: Vec<&SkippedTrack> = report.skipped().collect();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].label, "No UID");
        assert!(skipped[0].reason.contains("no uid"));
    }

    #[test]
    fn a_track_with_no_feature_file_is_skipped_not_fatal() {
        let dir = TempDir::new("nofeatures");
        let uids = library_at(&dir, "0.1.1", 2);
        let store = FeatureStore::new(dir.path());
        store.delete(&uids[0]).unwrap();

        let report = run(dir.path()).unwrap();
        let skipped: Vec<&SkippedTrack> = report.skipped().collect();
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].reason.contains("re-analyse"));
    }

    // ----- individual steps ----------------------------------------------

    #[test]
    fn the_zero_one_one_step_fills_the_segment_tables_from_the_feature_files() {
        let dir = TempDir::new("segments");
        let uids = library_at(&dir, "0.1.1", 1);
        let outcome = steps()[1].run(dir.path()).unwrap();
        assert_eq!(outcome.tracks_converted, 1);

        let library = Library::open(database_path(dir.path())).unwrap();
        let bpm = library.bpm_segments(&uids[0]).unwrap();
        assert_eq!(bpm.len(), 2);
        assert_eq!(bpm[0].bpm, 128.0);
        assert_eq!(bpm[1].bpm_rounded, 140);
        let keys = library.key_segments(&uids[0]).unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].key_label, "8A");
    }

    #[test]
    fn the_zero_one_one_step_drops_the_cached_waveform_image() {
        let dir = TempDir::new("waveform");
        let uids = library_at(&dir, "0.1.1", 1);
        let store = FeatureStore::new(dir.path());
        let mut features = store.load(&uids[0]).unwrap();
        features.insert(
            "wave_img_np.rgba",
            FeatureValue::I32(vec![1, 2, 3, 4]),
        );
        store.save(&uids[0], &features).unwrap();

        steps()[1].run(dir.path()).unwrap();
        let after = store.load(&uids[0]).unwrap();
        assert!(!after.contains("wave_img_np.rgba"));
        assert_eq!(after.beats_time_sec().len(), 4, "the rest is untouched");
    }

    #[test]
    fn the_zero_one_zero_step_adds_total_samples_and_reports_tracks_still_missing_it() {
        let dir = TempDir::new("totalsamples");
        let db_path = database_path(dir.path());
        {
            // A pre-0.1.1 tracks table: no total_samples column at all.
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE tracks (path TEXT PRIMARY KEY, uid TEXT, title TEXT NOT NULL DEFAULT '',
                    artist TEXT DEFAULT '', album TEXT DEFAULT '', bpm REAL, key INTEGER,
                    duration REAL, rating INTEGER DEFAULT 0, added_ts INTEGER NOT NULL,
                    comment TEXT DEFAULT '', file_mtime REAL DEFAULT 0, file_size INTEGER DEFAULT 0);
                 INSERT INTO tracks(path, title, added_ts) VALUES('/a.flac', 'A', 1);",
            )
            .unwrap();
        }
        let outcome = steps()[0].run(dir.path()).unwrap();
        assert_eq!(outcome.tracks_converted, 0);
        assert_eq!(outcome.tracks_skipped.len(), 1);
        assert_eq!(outcome.tracks_skipped[0].label, "A");

        let library = Library::open(&db_path).unwrap();
        assert!(library.column_exists("tracks", "total_samples").unwrap());
    }

    #[test]
    fn the_zero_two_zero_step_leaves_files_that_already_have_a_cue_block_alone() {
        let dir = TempDir::new("alreadydone");
        let uids = library_at(&dir, "0.2.0", 2);
        let store = FeatureStore::new(dir.path());
        let mut features = store.load(&uids[0]).unwrap();
        features.set_cue_points(&[]);
        store.save(&uids[0], &features).unwrap();

        let outcome = steps()[2].run(dir.path()).unwrap();
        assert_eq!(outcome.tracks_converted, 1, "only the untouched file needs work");
    }

    #[test]
    fn an_outcome_reports_whether_it_was_clean() {
        let mut outcome = MigrationOutcome::new("0.2.0", "0.3.0");
        assert!(outcome.is_clean());
        outcome.skip("A", None, "because");
        assert!(!outcome.is_clean());
        assert_eq!(outcome.tracks_skipped[0].reason, "because");
    }

    #[test]
    fn a_step_prints_its_hop_when_debugged() {
        assert_eq!(format!("{:?}", steps()[0]), "Step(0.1.0 -> 0.1.1)");
    }
}
