//! The per-track feature store: named numeric and text arrays, one file per
//! track uid.
//!
//! Python writes `{libpath}/{uid}.npz` — a zip of `.npy` members — and flattens
//! nested dictionaries into dotted keys such as `jump_cues_np.cue_start`. This
//! module keeps the *naming* convention (so the two layouts describe the same
//! features) but replaces the container with a small self-describing format,
//! `{libpath}/{uid}.mxf`, described in [`mod@self`]'s "On-disk format" section
//! below.
//!
//! Two things it fixes:
//!
//! * **Corruption is typed.** Python raises `zipfile.BadZipFile`, `EOFError` or
//!   `zlib.error` straight out of `np.load`, and no caller catches any of them,
//!   so a single damaged file makes a track permanently unloadable. Here every
//!   malformed file yields [`StoreError::FeatureFormat`] naming the path, and a
//!   file from a future build yields [`StoreError::FeatureVersion`].
//! * **Writes are atomic and leave nothing behind.** Python's atomic write uses
//!   a fixed temp-name prefix in the same directory and, in the 0.3.0
//!   migration, `unlink(missing_ok=True)` *after* the rename — but on failure
//!   before the rename the temp file survives under a predictable name that the
//!   next writer can collide with. Here the temp name is unique per attempt and
//!   removed on every failure path.
//!
//! # On-disk format
//!
//! All integers are little-endian.
//!
//! ```text
//! header   magic  8 bytes  "MXFEAT\r\n"
//!          u16              format version
//!          u16              flags (reserved, must be 0)
//!          u32              entry count
//! entry*   u32              name length in bytes
//!          ..               name, UTF-8
//!          u8               kind tag
//!          u64              element count
//!          ..               payload
//! trailer  u32              CRC-32 of every preceding byte
//! ```
//!
//! The `\r\n` inside the magic is deliberate: a file mangled by a text-mode
//! copy fails the magic check instead of parsing as something plausible.
//! Numeric payloads are `count` fixed-width elements; text payloads are `count`
//! records of `u32` byte length followed by UTF-8. The version prefix means a
//! later layout change is detected rather than misread.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use mixlyzer_core::key::Key;
use mixlyzer_core::segments::{KeySegment, TempoSegment};
use mixlyzer_core::track::canonical_uid;
use mixlyzer_core::{CuePoint, Phrase};

use crate::error::StoreError;

/// File magic. Never changes; the version field carries layout changes.
pub const MAGIC: [u8; 8] = *b"MXFEAT\r\n";

/// Layout version this build writes.
pub const FORMAT_VERSION: u16 = 1;

/// Extension of a feature file.
pub const FEATURE_EXTENSION: &str = "mxf";

/// Name of the beat-grid array, unchanged from Python.
pub const BEATS_TIME_SEC: &str = "beats_time_sec";
/// Dotted prefix for the tempo-segment arrays.
pub const TEMPO_SEGMENT_PREFIX: &str = "tempo_segments.";
/// Dotted prefix for the key-segment arrays.
pub const KEY_SEGMENT_PREFIX: &str = "key_segments.";
/// Dotted prefix for the phrase arrays, matching Python's `phrase_segments_np`.
pub const PHRASE_PREFIX: &str = "phrase_segments_np.";
/// Dotted prefix for the cue-point arrays, matching Python's `cue_points_np`.
pub const CUE_POINT_PREFIX: &str = "cue_points_np.";

const KIND_F32: u8 = 1;
const KIND_F64: u8 = 2;
const KIND_I32: u8 = 3;
const KIND_I64: u8 = 4;
const KIND_TEXT: u8 = 5;
const KIND_SCALAR_F64: u8 = 6;
const KIND_SCALAR_I64: u8 = 7;
const KIND_SCALAR_TEXT: u8 = 8;

/// One stored value: an array of one of four numeric types, an array of
/// strings, or a single scalar.
///
/// Arrays are one-dimensional. Structured data (a tempo segment, a cue point)
/// is stored as one array per field under a shared dotted prefix, which is both
/// Python's convention and what makes each array independently readable.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureValue {
    /// Single-precision floats, for bulk data where precision is not critical.
    F32(Vec<f32>),
    /// Double-precision floats: times, tempi, anything the domain types use.
    F64(Vec<f64>),
    /// 32-bit integers, for small counts and enumerations.
    I32(Vec<i32>),
    /// 64-bit integers, for sample offsets and identifiers.
    I64(Vec<i64>),
    /// UTF-8 strings, one per element.
    Text(Vec<String>),
    /// A single float, e.g. a track-wide confidence.
    ScalarF64(f64),
    /// A single integer, e.g. a sample rate.
    ScalarI64(i64),
    /// A single string, e.g. the analyser version that wrote the file.
    ScalarText(String),
}

impl FeatureValue {
    /// Number of elements. Scalars count as one.
    pub fn len(&self) -> usize {
        match self {
            FeatureValue::F32(v) => v.len(),
            FeatureValue::F64(v) => v.len(),
            FeatureValue::I32(v) => v.len(),
            FeatureValue::I64(v) => v.len(),
            FeatureValue::Text(v) => v.len(),
            FeatureValue::ScalarF64(_) | FeatureValue::ScalarI64(_) | FeatureValue::ScalarText(_) => 1,
        }
    }

    /// Whether the value holds no elements. Scalars are never empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read as `f64`, widening whatever numeric type was stored.
    ///
    /// `None` for text values, so a caller asking for numbers never gets a
    /// silent zero.
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            FeatureValue::F32(v) => Some(v.iter().map(|x| f64::from(*x)).collect()),
            FeatureValue::F64(v) => Some(v.clone()),
            FeatureValue::I32(v) => Some(v.iter().map(|x| f64::from(*x)).collect()),
            FeatureValue::I64(v) => Some(v.iter().map(|x| *x as f64).collect()),
            FeatureValue::ScalarF64(x) => Some(vec![*x]),
            FeatureValue::ScalarI64(x) => Some(vec![*x as f64]),
            FeatureValue::Text(_) | FeatureValue::ScalarText(_) => None,
        }
    }

    /// Read as `i64`, rounding stored floats to nearest.
    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            FeatureValue::I32(v) => Some(v.iter().map(|x| i64::from(*x)).collect()),
            FeatureValue::I64(v) => Some(v.clone()),
            FeatureValue::ScalarI64(x) => Some(vec![*x]),
            FeatureValue::F32(_) | FeatureValue::F64(_) | FeatureValue::ScalarF64(_) => self
                .as_f64_vec()
                .map(|v| v.iter().map(|x| x.round() as i64).collect()),
            FeatureValue::Text(_) | FeatureValue::ScalarText(_) => None,
        }
    }

    /// Read as strings. `None` for numeric values.
    pub fn as_str_vec(&self) -> Option<Vec<String>> {
        match self {
            FeatureValue::Text(v) => Some(v.clone()),
            FeatureValue::ScalarText(s) => Some(vec![s.clone()]),
            _ => None,
        }
    }

    /// The single value of a scalar, or of a one-element array.
    pub fn as_f64(&self) -> Option<f64> {
        self.as_f64_vec().and_then(|v| v.first().copied())
    }

    fn kind(&self) -> u8 {
        match self {
            FeatureValue::F32(_) => KIND_F32,
            FeatureValue::F64(_) => KIND_F64,
            FeatureValue::I32(_) => KIND_I32,
            FeatureValue::I64(_) => KIND_I64,
            FeatureValue::Text(_) => KIND_TEXT,
            FeatureValue::ScalarF64(_) => KIND_SCALAR_F64,
            FeatureValue::ScalarI64(_) => KIND_SCALAR_I64,
            FeatureValue::ScalarText(_) => KIND_SCALAR_TEXT,
        }
    }
}

/// The contents of one track's feature file.
///
/// Entries are kept sorted by name so that writing the same features twice
/// produces identical bytes, which makes "did the analysis change anything?"
/// answerable by comparing files.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeatureFile {
    values: BTreeMap<String, FeatureValue>,
}

impl FeatureFile {
    /// An empty feature file.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `value` under `name`, returning whatever it replaced.
    pub fn insert(&mut self, name: impl Into<String>, value: FeatureValue) -> Option<FeatureValue> {
        self.values.insert(name.into(), value)
    }

    /// Look up a value.
    pub fn get(&self, name: &str) -> Option<&FeatureValue> {
        self.values.get(name)
    }

    /// Remove a value, returning it.
    pub fn remove(&mut self, name: &str) -> Option<FeatureValue> {
        self.values.remove(name)
    }

    /// Whether `name` is present.
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Every stored name, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Remove every entry whose name starts with `prefix`.
    ///
    /// Used by migrations to drop a superseded block wholesale, the way the
    /// Python 0.2.0 step pops every `jump_cues_np.*` key.
    pub fn remove_prefixed(&mut self, prefix: &str) -> usize {
        let doomed: Vec<String> = self
            .values
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        for name in &doomed {
            self.values.remove(name);
        }
        doomed.len()
    }

    // ----- encoding -------------------------------------------------------

    /// Serialise to the on-disk representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&(self.values.len() as u32).to_le_bytes());
        for (name, value) in &self.values {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.push(value.kind());
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            match value {
                FeatureValue::F32(v) => {
                    for x in v {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                FeatureValue::F64(v) => {
                    for x in v {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                FeatureValue::I32(v) => {
                    for x in v {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                FeatureValue::I64(v) => {
                    for x in v {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
                FeatureValue::Text(v) => {
                    for s in v {
                        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                        out.extend_from_slice(s.as_bytes());
                    }
                }
                FeatureValue::ScalarF64(x) => out.extend_from_slice(&x.to_le_bytes()),
                FeatureValue::ScalarI64(x) => out.extend_from_slice(&x.to_le_bytes()),
                FeatureValue::ScalarText(s) => {
                    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
            }
        }
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse the on-disk representation.
    ///
    /// `origin` is only used to name the file in any error, so an in-memory
    /// buffer can pass a placeholder path.
    pub fn from_bytes(bytes: &[u8], origin: &Path) -> Result<Self, StoreError> {
        let bad = |detail: String| StoreError::FeatureFormat {
            path: origin.to_path_buf(),
            detail,
        };
        if bytes.len() < MAGIC.len() + 8 + 4 {
            return Err(bad(format!("file is only {} bytes", bytes.len())));
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(bad("bad magic: not a Mixlyzer feature file".to_string()));
        }
        let body_len = bytes.len() - 4;
        let stored_crc = u32::from_le_bytes([
            bytes[body_len],
            bytes[body_len + 1],
            bytes[body_len + 2],
            bytes[body_len + 3],
        ]);
        let actual_crc = crc32(&bytes[..body_len]);
        if stored_crc != actual_crc {
            return Err(bad(format!(
                "checksum mismatch: stored {stored_crc:#010x}, computed {actual_crc:#010x}"
            )));
        }

        let mut cursor = Cursor::new(&bytes[..body_len], origin);
        cursor.skip(MAGIC.len())?;
        let version = cursor.u16()?;
        if version > FORMAT_VERSION {
            return Err(StoreError::FeatureVersion {
                path: origin.to_path_buf(),
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let flags = cursor.u16()?;
        if flags != 0 {
            return Err(bad(format!("unknown flags {flags:#06x}")));
        }
        let entry_count = cursor.u32()? as usize;

        let mut values = BTreeMap::new();
        for _ in 0..entry_count {
            let name_len = cursor.u32()? as usize;
            let name = cursor.utf8(name_len)?;
            let kind = cursor.u8()?;
            let count = cursor.u64()?;
            let value = cursor.value(kind, count)?;
            values.insert(name, value);
        }
        if cursor.remaining() != 0 {
            return Err(bad(format!(
                "{} trailing bytes after {entry_count} entries",
                cursor.remaining()
            )));
        }
        Ok(FeatureFile { values })
    }

    // ----- convenience accessors -----------------------------------------

    /// The beat grid, in seconds.
    pub fn beats_time_sec(&self) -> Vec<f64> {
        self.get(BEATS_TIME_SEC)
            .and_then(FeatureValue::as_f64_vec)
            .unwrap_or_default()
    }

    /// Replace the beat grid.
    pub fn set_beats_time_sec(&mut self, beats: &[f64]) {
        self.insert(BEATS_TIME_SEC, FeatureValue::F64(beats.to_vec()));
    }

    /// The tempo segments.
    ///
    /// Rows that fail [`TempoSegment::from_row`] validation are dropped, and
    /// ragged arrays are truncated to the shortest, so a partially written
    /// block yields fewer segments rather than an index panic.
    pub fn tempo_segments(&self) -> Vec<TempoSegment> {
        let start = self.numeric(TEMPO_SEGMENT_PREFIX, "start");
        let end = self.numeric(TEMPO_SEGMENT_PREFIX, "end");
        let bpm = self.numeric(TEMPO_SEGMENT_PREFIX, "bpm");
        let inizio = self.numeric(TEMPO_SEGMENT_PREFIX, "inizio");
        let ts = self.numeric(TEMPO_SEGMENT_PREFIX, "time_signature");
        let n = [start.len(), end.len(), bpm.len()]
            .into_iter()
            .min()
            .unwrap_or(0);
        (0..n)
            .filter_map(|i| {
                TempoSegment::from_row(&[
                    start[i],
                    end[i],
                    bpm[i],
                    inizio.get(i).copied().unwrap_or(start[i]),
                    ts.get(i).copied().unwrap_or(4.0),
                ])
            })
            .collect()
    }

    /// Replace the tempo segments.
    pub fn set_tempo_segments(&mut self, segments: &[TempoSegment]) {
        self.insert(
            format!("{TEMPO_SEGMENT_PREFIX}start"),
            FeatureValue::F64(segments.iter().map(|s| s.start).collect()),
        );
        self.insert(
            format!("{TEMPO_SEGMENT_PREFIX}end"),
            FeatureValue::F64(segments.iter().map(|s| s.end).collect()),
        );
        self.insert(
            format!("{TEMPO_SEGMENT_PREFIX}bpm"),
            FeatureValue::F64(segments.iter().map(|s| s.bpm).collect()),
        );
        self.insert(
            format!("{TEMPO_SEGMENT_PREFIX}inizio"),
            FeatureValue::F64(segments.iter().map(|s| s.inizio).collect()),
        );
        self.insert(
            format!("{TEMPO_SEGMENT_PREFIX}time_signature"),
            FeatureValue::I32(segments.iter().map(|s| i32::from(s.time_signature)).collect()),
        );
    }

    /// The key segments.
    ///
    /// Stored as `start`, `end` and a single `key` index rather than Python's
    /// `[pitch, mode, start, end]` row: the index already encodes both, and
    /// keeping them together makes an out-of-range value impossible to build
    /// from half-valid columns.
    pub fn key_segments(&self) -> Vec<KeySegment> {
        let start = self.numeric(KEY_SEGMENT_PREFIX, "start");
        let end = self.numeric(KEY_SEGMENT_PREFIX, "end");
        let key = self.numeric(KEY_SEGMENT_PREFIX, "key");
        let n = [start.len(), end.len(), key.len()]
            .into_iter()
            .min()
            .unwrap_or(0);
        (0..n)
            .filter(|i| start[*i].is_finite() && end[*i].is_finite() && key[*i].is_finite())
            .map(|i| KeySegment::new(start[i], end[i], Key::from_index(key[i].round() as i64)))
            .collect()
    }

    /// Replace the key segments.
    pub fn set_key_segments(&mut self, segments: &[KeySegment]) {
        self.insert(
            format!("{KEY_SEGMENT_PREFIX}start"),
            FeatureValue::F64(segments.iter().map(|s| s.start).collect()),
        );
        self.insert(
            format!("{KEY_SEGMENT_PREFIX}end"),
            FeatureValue::F64(segments.iter().map(|s| s.end).collect()),
        );
        self.insert(
            format!("{KEY_SEGMENT_PREFIX}key"),
            FeatureValue::I32(segments.iter().map(|s| i32::from(s.key.index())).collect()),
        );
    }

    /// The song-structure phrases.
    pub fn phrases(&self) -> Vec<Phrase> {
        let start = self.numeric(PHRASE_PREFIX, "start");
        let end = self.numeric(PHRASE_PREFIX, "end");
        let label = self.text(PHRASE_PREFIX, "label");
        let n = [start.len(), end.len(), label.len()]
            .into_iter()
            .min()
            .unwrap_or(0);
        (0..n)
            .filter(|i| start[*i].is_finite() && end[*i].is_finite())
            .map(|i| Phrase::new(start[i], end[i], label[i].clone()))
            .collect()
    }

    /// Replace the phrases.
    pub fn set_phrases(&mut self, phrases: &[Phrase]) {
        self.insert(
            format!("{PHRASE_PREFIX}start"),
            FeatureValue::F64(phrases.iter().map(|p| p.start).collect()),
        );
        self.insert(
            format!("{PHRASE_PREFIX}end"),
            FeatureValue::F64(phrases.iter().map(|p| p.end).collect()),
        );
        self.insert(
            format!("{PHRASE_PREFIX}label"),
            FeatureValue::Text(phrases.iter().map(|p| p.label.clone()).collect()),
        );
    }

    /// The cue points.
    pub fn cue_points(&self) -> Vec<CuePoint> {
        let id = self.numeric(CUE_POINT_PREFIX, "cue_id");
        let time = self.numeric(CUE_POINT_PREFIX, "cue_time_sec");
        let label = self.text(CUE_POINT_PREFIX, "cue_label");
        let comment = self.text(CUE_POINT_PREFIX, "cue_comment");
        let n = [id.len(), time.len(), label.len(), comment.len()]
            .into_iter()
            .min()
            .unwrap_or(0);
        (0..n)
            .filter(|i| time[*i].is_finite() && time[*i] >= 0.0)
            .map(|i| {
                CuePoint::new(
                    id[i].max(0.0).round() as usize,
                    time[i],
                    label[i].clone(),
                    comment[i].clone(),
                )
            })
            .collect()
    }

    /// Replace the cue points.
    pub fn set_cue_points(&mut self, points: &[CuePoint]) {
        self.insert(
            format!("{CUE_POINT_PREFIX}cue_id"),
            FeatureValue::I32(points.iter().map(|p| p.id as i32).collect()),
        );
        self.insert(
            format!("{CUE_POINT_PREFIX}cue_time_sec"),
            FeatureValue::F64(points.iter().map(|p| p.time_sec).collect()),
        );
        self.insert(
            format!("{CUE_POINT_PREFIX}cue_label"),
            FeatureValue::Text(points.iter().map(|p| p.label.clone()).collect()),
        );
        self.insert(
            format!("{CUE_POINT_PREFIX}cue_comment"),
            FeatureValue::Text(points.iter().map(|p| p.comment.clone()).collect()),
        );
    }

    /// Whether the canonical cue-point block is present.
    ///
    /// The 0.2.0 -> 0.3.0 migration uses this to decide whether a file needs
    /// rewriting.
    pub fn has_cue_point_block(&self) -> bool {
        ["cue_id", "cue_time_sec", "cue_label", "cue_comment"]
            .iter()
            .all(|field| self.contains(&format!("{CUE_POINT_PREFIX}{field}")))
    }

    fn numeric(&self, prefix: &str, field: &str) -> Vec<f64> {
        self.get(&format!("{prefix}{field}"))
            .and_then(FeatureValue::as_f64_vec)
            .unwrap_or_default()
    }

    fn text(&self, prefix: &str, field: &str) -> Vec<String> {
        self.get(&format!("{prefix}{field}"))
            .and_then(FeatureValue::as_str_vec)
            .unwrap_or_default()
    }
}

/// A directory of feature files, addressed by track uid.
#[derive(Debug, Clone)]
pub struct FeatureStore {
    base_dir: PathBuf,
}

/// Counter that makes concurrent temp-file names unique within a process.
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

impl FeatureStore {
    /// A store rooted at `base_dir`. The directory is created on first write.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// The directory this store writes to.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// The file a uid maps to.
    ///
    /// Errors when the uid is not a canonical UUIDv4, so a caller can never
    /// build a path from attacker-shaped text.
    pub fn path_for(&self, uid: &str) -> Result<PathBuf, StoreError> {
        let uid = canonical_uid(uid)?;
        Ok(self.base_dir.join(format!("{uid}.{FEATURE_EXTENSION}")))
    }

    /// Whether a feature file exists for `uid`.
    pub fn exists(&self, uid: &str) -> Result<bool, StoreError> {
        Ok(self.path_for(uid)?.exists())
    }

    /// Load a track's features, erroring when the file is absent.
    pub fn load(&self, uid: &str) -> Result<FeatureFile, StoreError> {
        let path = self.path_for(uid)?;
        match self.load_path(&path) {
            Ok(Some(file)) => Ok(file),
            Ok(None) => Err(StoreError::FeatureMissing {
                uid: uid.to_string(),
                path,
            }),
            Err(err) => Err(err),
        }
    }

    /// Load a track's features, or `None` when there is no file yet.
    ///
    /// A file that exists but is damaged is still an error: "not analysed" and
    /// "analysed, then corrupted" must not look the same to a caller.
    pub fn load_optional(&self, uid: &str) -> Result<Option<FeatureFile>, StoreError> {
        let path = self.path_for(uid)?;
        self.load_path(&path)
    }

    fn load_path(&self, path: &Path) -> Result<Option<FeatureFile>, StoreError> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(StoreError::io(path, err)),
        };
        FeatureFile::from_bytes(&bytes, path).map(Some)
    }

    /// Write a track's features, replacing any previous file atomically.
    ///
    /// The bytes go to a uniquely named temp file in the same directory (so the
    /// rename cannot cross a filesystem), are flushed and synced, then renamed
    /// over the target. Every failure path removes the temp file.
    pub fn save(&self, uid: &str, features: &FeatureFile) -> Result<PathBuf, StoreError> {
        let path = self.path_for(uid)?;
        write_atomic(&path, &features.to_bytes())?;
        Ok(path)
    }

    /// Delete a track's feature file. Returns whether one was there.
    pub fn delete(&self, uid: &str) -> Result<bool, StoreError> {
        let path = self.path_for(uid)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(StoreError::io(&path, err)),
        }
    }

    /// Every uid that has a feature file, sorted.
    ///
    /// Files whose stem is not a canonical uid are ignored: the directory also
    /// holds `library.db`, `VERSION` and whatever else the app keeps there.
    pub fn list_uids(&self) -> Result<Vec<String>, StoreError> {
        let entries = match fs::read_dir(&self.base_dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(StoreError::io(&self.base_dir, err)),
        };
        let mut uids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| StoreError::io(&self.base_dir, e))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(FEATURE_EXTENSION) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Ok(uid) = canonical_uid(stem) {
                uids.push(uid);
            }
        }
        uids.sort();
        Ok(uids)
    }
}

/// Write `bytes` to `path` so that a reader sees either the old file or the
/// whole new one, never a partial write.
///
/// The temp file is created in the destination directory (a rename across
/// filesystems is not atomic) under a name unique to this attempt, and is
/// removed on every failure path.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| StoreError::io(dir, e))?;
    let temp_path = dir.join(temp_name(path));

    let write = || -> std::io::Result<()> {
        let mut file = fs::File::create(&temp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    };
    if let Err(err) = write() {
        let _ = fs::remove_file(&temp_path);
        return Err(StoreError::io(&temp_path, err));
    }
    if let Err(err) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(StoreError::io(path, err));
    }
    Ok(())
}

/// A temp-file name that no concurrent writer can collide with.
///
/// Python uses a fixed prefix plus `mkstemp`, which is unique, but its 0.3.0
/// migration then unlinks the temp path unconditionally after the rename and
/// leaves it behind when the write fails earlier. A per-attempt name plus
/// removal on every failure path is what makes a crashed write invisible to the
/// next one.
fn temp_name(target: &Path) -> String {
    let stem = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("feature");
    let seq = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!(".{stem}.{}.{seq}.{nanos}.tmp", std::process::id())
}

/// Bounds-checked reader over the file body.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    origin: &'a Path,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], origin: &'a Path) -> Self {
        Self {
            bytes,
            pos: 0,
            origin,
        }
    }

    fn bad(&self, detail: String) -> StoreError {
        StoreError::FeatureFormat {
            path: self.origin.to_path_buf(),
            detail,
        }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StoreError> {
        if n > self.remaining() {
            return Err(self.bad(format!(
                "truncated at byte {}: wanted {n} more, {} left",
                self.pos,
                self.remaining()
            )));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn skip(&mut self, n: usize) -> Result<(), StoreError> {
        self.take(n).map(|_| ())
    }

    fn u8(&mut self) -> Result<u8, StoreError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, StoreError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, StoreError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, StoreError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn utf8(&mut self, len: usize) -> Result<String, StoreError> {
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|e| self.bad(format!("invalid UTF-8: {e}")))
    }

    /// Read `count` elements of `kind`.
    ///
    /// The element count is validated against the bytes actually left before
    /// anything is allocated, so a corrupt length cannot ask for a terabyte.
    fn value(&mut self, kind: u8, count: u64) -> Result<FeatureValue, StoreError> {
        let scalar = matches!(kind, KIND_SCALAR_F64 | KIND_SCALAR_I64 | KIND_SCALAR_TEXT);
        if scalar && count != 1 {
            return Err(self.bad(format!("scalar entry declares {count} elements")));
        }
        let width = match kind {
            KIND_F32 | KIND_I32 => 4,
            KIND_F64 | KIND_I64 | KIND_SCALAR_F64 | KIND_SCALAR_I64 => 8,
            KIND_TEXT | KIND_SCALAR_TEXT => 4, // minimum: the per-string length
            other => return Err(self.bad(format!("unknown value kind {other}"))),
        };
        let needed = (count as u128) * (width as u128);
        if needed > self.remaining() as u128 {
            return Err(self.bad(format!(
                "entry declares {count} elements ({needed} bytes) but only {} bytes remain",
                self.remaining()
            )));
        }
        let count = count as usize;
        Ok(match kind {
            KIND_F32 => {
                let b = self.take(count * 4)?;
                FeatureValue::F32(
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            KIND_F64 => {
                let b = self.take(count * 8)?;
                FeatureValue::F64(
                    b.chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect(),
                )
            }
            KIND_I32 => {
                let b = self.take(count * 4)?;
                FeatureValue::I32(
                    b.chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            KIND_I64 => {
                let b = self.take(count * 8)?;
                FeatureValue::I64(
                    b.chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect(),
                )
            }
            KIND_TEXT => {
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    let len = self.u32()? as usize;
                    out.push(self.utf8(len)?);
                }
                FeatureValue::Text(out)
            }
            KIND_SCALAR_F64 => {
                let b = self.take(8)?;
                FeatureValue::ScalarF64(f64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
            KIND_SCALAR_I64 => {
                let b = self.take(8)?;
                FeatureValue::ScalarI64(i64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
            KIND_SCALAR_TEXT => {
                let len = self.u32()? as usize;
                FeatureValue::ScalarText(self.utf8(len)?)
            }
            other => return Err(self.bad(format!("unknown value kind {other}"))),
        })
    }
}

/// CRC-32 (IEEE 802.3, the zlib polynomial), so a truncated or bit-rotted file
/// is rejected rather than parsed into plausible-looking numbers.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        let index = ((crc ^ u32::from(*byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

const CRC32_TABLE: [u32; 256] = build_crc32_table();

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut value = i as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 != 0 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[i] = value;
        i += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use mixlyzer_core::key::Mode;
    use mixlyzer_core::track::new_uid;

    fn origin() -> PathBuf {
        PathBuf::from("/library/test.mxf")
    }

    fn round_trip(file: &FeatureFile) -> FeatureFile {
        FeatureFile::from_bytes(&file.to_bytes(), &origin()).expect("valid file must parse")
    }

    // ----- the container --------------------------------------------------

    #[test]
    fn an_empty_file_round_trips() {
        let file = FeatureFile::new();
        let back = round_trip(&file);
        assert!(back.is_empty());
        assert_eq!(back.len(), 0);
        assert_eq!(back, file);
    }

    #[test]
    fn every_value_kind_round_trips() {
        let mut file = FeatureFile::new();
        file.insert("f32", FeatureValue::F32(vec![1.5, -2.25, f32::MAX]));
        file.insert("f64", FeatureValue::F64(vec![1.5, -2.25, f64::MAX]));
        file.insert("i32", FeatureValue::I32(vec![i32::MIN, 0, i32::MAX]));
        file.insert("i64", FeatureValue::I64(vec![i64::MIN, 0, i64::MAX]));
        file.insert(
            "text",
            FeatureValue::Text(vec!["a".into(), String::new(), "ünïcødé 🎛".into()]),
        );
        file.insert("scalar_f64", FeatureValue::ScalarF64(0.125));
        file.insert("scalar_i64", FeatureValue::ScalarI64(-7));
        file.insert("scalar_text", FeatureValue::ScalarText("hello".into()));

        assert_eq!(round_trip(&file), file);
    }

    #[test]
    fn empty_arrays_round_trip_as_empty_not_absent() {
        let mut file = FeatureFile::new();
        file.insert("nothing", FeatureValue::F64(Vec::new()));
        file.insert("no_text", FeatureValue::Text(Vec::new()));
        let back = round_trip(&file);
        assert!(back.contains("nothing"));
        assert_eq!(back.get("nothing").unwrap().len(), 0);
        assert!(back.get("no_text").unwrap().is_empty());
    }

    #[test]
    fn unicode_names_round_trip() {
        let mut file = FeatureFile::new();
        file.insert("키_세그먼트.시작", FeatureValue::F64(vec![1.0]));
        let back = round_trip(&file);
        assert_eq!(back.get("키_세그먼트.시작").unwrap().as_f64(), Some(1.0));
    }

    #[test]
    fn non_finite_values_survive_the_round_trip() {
        let mut file = FeatureFile::new();
        file.insert(
            "odd",
            FeatureValue::F64(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        );
        let values = round_trip(&file).get("odd").unwrap().as_f64_vec().unwrap();
        assert!(values[0].is_nan());
        assert_eq!(values[1], f64::INFINITY);
        assert_eq!(values[2], f64::NEG_INFINITY);
    }

    /// Identical features must produce identical bytes, so "did anything
    /// change?" is answerable by comparing files.
    #[test]
    fn encoding_is_deterministic_regardless_of_insertion_order() {
        let mut a = FeatureFile::new();
        a.insert("z", FeatureValue::I32(vec![1]));
        a.insert("a", FeatureValue::F64(vec![2.0]));
        let mut b = FeatureFile::new();
        b.insert("a", FeatureValue::F64(vec![2.0]));
        b.insert("z", FeatureValue::I32(vec![1]));
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn the_header_carries_the_magic_and_version() {
        let bytes = FeatureFile::new().to_bytes();
        assert_eq!(&bytes[..8], &MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), FORMAT_VERSION);
    }

    // ----- corruption -----------------------------------------------------

    #[test]
    fn a_file_with_bad_magic_is_typed_corruption() {
        let mut bytes = FeatureFile::new().to_bytes();
        bytes[0] = b'X';
        let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
        assert!(matches!(err, StoreError::FeatureFormat { .. }), "{err:?}");
        assert!(err.to_string().contains("test.mxf"));
        assert!(err.is_corruption());
    }

    #[test]
    fn a_random_blob_is_typed_corruption_rather_than_a_panic() {
        let bytes = vec![0xABu8; 512];
        let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
        assert!(matches!(err, StoreError::FeatureFormat { .. }), "{err:?}");
    }

    #[test]
    fn a_file_that_is_too_short_is_typed_corruption() {
        for len in [0usize, 1, 8, 12, 19] {
            let bytes = vec![0u8; len];
            let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
            assert!(matches!(err, StoreError::FeatureFormat { .. }), "len {len}");
        }
    }

    #[test]
    fn a_truncated_file_is_typed_corruption() {
        let mut file = FeatureFile::new();
        file.insert("beats", FeatureValue::F64(vec![1.0; 64]));
        let bytes = file.to_bytes();
        let err = FeatureFile::from_bytes(&bytes[..bytes.len() / 2], &origin()).unwrap_err();
        assert!(matches!(err, StoreError::FeatureFormat { .. }), "{err:?}");
    }

    #[test]
    fn a_single_flipped_bit_fails_the_checksum() {
        let mut file = FeatureFile::new();
        file.insert("beats", FeatureValue::F64(vec![1.0, 2.0, 3.0]));
        let mut bytes = file.to_bytes();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x01;
        let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
        assert!(
            err.to_string().contains("checksum"),
            "expected a checksum complaint, got {err}"
        );
    }

    #[test]
    fn a_file_from_a_future_build_reports_its_version() {
        let mut bytes = FeatureFile::new().to_bytes();
        let next = FORMAT_VERSION + 1;
        bytes[8..10].copy_from_slice(&next.to_le_bytes());
        // The checksum covers the header, so refresh it before parsing.
        let body_len = bytes.len() - 4;
        let crc = crc32(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&crc.to_le_bytes());

        let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
        match err {
            StoreError::FeatureVersion { found, supported, .. } => {
                assert_eq!(found, next);
                assert_eq!(supported, FORMAT_VERSION);
            }
            other => panic!("expected FeatureVersion, got {other:?}"),
        }
    }

    /// A corrupt element count must be rejected against the bytes actually
    /// present, never used to size an allocation.
    #[test]
    fn an_absurd_element_count_is_rejected_without_allocating() {
        let mut file = FeatureFile::new();
        file.insert("a", FeatureValue::F64(vec![1.0]));
        let mut bytes = file.to_bytes();
        // Entry layout: header(16) + name_len(4) + name(1) + kind(1) + count(8).
        let count_at = 16 + 4 + 1 + 1;
        bytes[count_at..count_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let body_len = bytes.len() - 4;
        let crc = crc32(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&crc.to_le_bytes());

        let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
        assert!(matches!(err, StoreError::FeatureFormat { .. }), "{err:?}");
        assert!(err.to_string().contains("bytes remain"), "{err}");
    }

    #[test]
    fn an_unknown_value_kind_is_rejected() {
        let mut file = FeatureFile::new();
        file.insert("a", FeatureValue::F64(vec![1.0]));
        let mut bytes = file.to_bytes();
        bytes[16 + 4 + 1] = 200; // the kind tag
        let body_len = bytes.len() - 4;
        let crc = crc32(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&crc.to_le_bytes());

        let err = FeatureFile::from_bytes(&bytes, &origin()).unwrap_err();
        assert!(err.to_string().contains("unknown value kind"), "{err}");
    }

    // ----- value accessors ------------------------------------------------

    #[test]
    fn numeric_values_widen_to_f64_and_text_does_not() {
        assert_eq!(
            FeatureValue::I32(vec![1, 2]).as_f64_vec(),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(FeatureValue::ScalarI64(3).as_f64(), Some(3.0));
        assert_eq!(FeatureValue::Text(vec!["1".into()]).as_f64_vec(), None);
        assert_eq!(FeatureValue::F64(vec![1.6]).as_i64_vec(), Some(vec![2]));
        assert_eq!(
            FeatureValue::ScalarText("x".into()).as_str_vec(),
            Some(vec!["x".to_string()])
        );
        assert_eq!(FeatureValue::F64(vec![1.0]).as_str_vec(), None);
    }

    #[test]
    fn remove_prefixed_drops_a_whole_block() {
        let mut file = FeatureFile::new();
        file.insert("jump_cues_np.cue_start", FeatureValue::F64(vec![1.0]));
        file.insert("jump_cues_np.cue_end", FeatureValue::F64(vec![2.0]));
        file.insert("beats_time_sec", FeatureValue::F64(vec![0.5]));
        assert_eq!(file.remove_prefixed("jump_cues_np"), 2);
        assert_eq!(file.names().collect::<Vec<_>>(), vec!["beats_time_sec"]);
        assert_eq!(file.remove_prefixed("jump_cues_np"), 0);
    }

    #[test]
    fn insert_replaces_and_remove_returns_the_old_value() {
        let mut file = FeatureFile::new();
        assert!(file.insert("a", FeatureValue::ScalarI64(1)).is_none());
        assert_eq!(
            file.insert("a", FeatureValue::ScalarI64(2)),
            Some(FeatureValue::ScalarI64(1))
        );
        assert_eq!(file.remove("a"), Some(FeatureValue::ScalarI64(2)));
        assert!(file.remove("a").is_none());
    }

    // ----- domain accessors ----------------------------------------------

    #[test]
    fn beats_round_trip() {
        let mut file = FeatureFile::new();
        let beats: Vec<f64> = (0..64).map(|i| f64::from(i) * 0.46875).collect();
        file.set_beats_time_sec(&beats);
        assert_eq!(round_trip(&file).beats_time_sec(), beats);
    }

    #[test]
    fn missing_blocks_read_as_empty_rather_than_failing() {
        let file = FeatureFile::new();
        assert!(file.beats_time_sec().is_empty());
        assert!(file.tempo_segments().is_empty());
        assert!(file.key_segments().is_empty());
        assert!(file.phrases().is_empty());
        assert!(file.cue_points().is_empty());
        assert!(!file.has_cue_point_block());
    }

    #[test]
    fn tempo_segments_round_trip() {
        let segments = vec![
            TempoSegment::new(0.0, 30.0, 128.0, 0.25, 4),
            TempoSegment::new(30.0, 60.0, 140.5, 30.125, 3),
        ];
        let mut file = FeatureFile::new();
        file.set_tempo_segments(&segments);
        assert_eq!(round_trip(&file).tempo_segments(), segments);
    }

    #[test]
    fn zero_length_tempo_segments_round_trip() {
        let segments = vec![TempoSegment::new(5.0, 5.0, 128.0, 5.0, 4)];
        let mut file = FeatureFile::new();
        file.set_tempo_segments(&segments);
        let back = round_trip(&file).tempo_segments();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].duration(), 0.0);
    }

    #[test]
    fn key_segments_round_trip_through_the_key_index() {
        let segments = vec![
            KeySegment::new(0.0, 30.0, Key::new(9, Mode::Minor)),
            KeySegment::new(30.0, 60.0, Key::new(0, Mode::Major)),
        ];
        let mut file = FeatureFile::new();
        file.set_key_segments(&segments);
        let back = round_trip(&file).key_segments();
        assert_eq!(back, segments);
        assert_eq!(back[0].key.camelot(), "8A");
    }

    #[test]
    fn phrases_round_trip_including_unicode_labels() {
        let phrases = vec![
            Phrase::new(0.0, 16.0, "INTRO"),
            Phrase::new(16.0, 48.0, "코러스"),
        ];
        let mut file = FeatureFile::new();
        file.set_phrases(&phrases);
        assert_eq!(round_trip(&file).phrases(), phrases);
    }

    #[test]
    fn cue_points_round_trip() {
        let points = vec![
            CuePoint::new(0, 12.5, "CHORUS_IN", "first chorus"),
            CuePoint::new(1, 96.0, "OUTRO_IN", "그 아웃트로"),
        ];
        let mut file = FeatureFile::new();
        file.set_cue_points(&points);
        let back = round_trip(&file);
        assert!(back.has_cue_point_block());
        assert_eq!(back.cue_points(), points);
    }

    #[test]
    fn an_empty_cue_block_is_present_but_yields_no_points() {
        let mut file = FeatureFile::new();
        file.set_cue_points(&[]);
        let back = round_trip(&file);
        assert!(back.has_cue_point_block());
        assert!(back.cue_points().is_empty());
    }

    /// A block whose columns disagree in length is truncated to the shortest,
    /// which is what Python's `min(array.size ...)` does — and is why an index
    /// panic is impossible here.
    #[test]
    fn a_ragged_block_truncates_to_its_shortest_column() {
        let mut file = FeatureFile::new();
        file.insert(
            format!("{TEMPO_SEGMENT_PREFIX}start"),
            FeatureValue::F64(vec![0.0, 10.0, 20.0]),
        );
        file.insert(
            format!("{TEMPO_SEGMENT_PREFIX}end"),
            FeatureValue::F64(vec![10.0, 20.0]),
        );
        file.insert(
            format!("{TEMPO_SEGMENT_PREFIX}bpm"),
            FeatureValue::F64(vec![128.0, 128.0, 128.0]),
        );
        assert_eq!(file.tempo_segments().len(), 2);
    }

    #[test]
    fn tempo_rows_that_fail_validation_are_dropped() {
        let mut file = FeatureFile::new();
        file.insert(
            format!("{TEMPO_SEGMENT_PREFIX}start"),
            FeatureValue::F64(vec![0.0, 10.0]),
        );
        file.insert(
            format!("{TEMPO_SEGMENT_PREFIX}end"),
            FeatureValue::F64(vec![10.0, 20.0]),
        );
        file.insert(
            format!("{TEMPO_SEGMENT_PREFIX}bpm"),
            FeatureValue::F64(vec![0.0, 128.0]),
        );
        let segments = file.tempo_segments();
        assert_eq!(segments.len(), 1, "a zero-BPM row is not a segment");
        assert_eq!(segments[0].bpm, 128.0);
    }

    #[test]
    fn cue_points_with_impossible_times_are_dropped() {
        let mut file = FeatureFile::new();
        file.insert(
            format!("{CUE_POINT_PREFIX}cue_id"),
            FeatureValue::I32(vec![0, 1, 2]),
        );
        file.insert(
            format!("{CUE_POINT_PREFIX}cue_time_sec"),
            FeatureValue::F64(vec![-1.0, f64::NAN, 10.0]),
        );
        file.insert(
            format!("{CUE_POINT_PREFIX}cue_label"),
            FeatureValue::Text(vec!["A".into(), "B".into(), "C".into()]),
        );
        file.insert(
            format!("{CUE_POINT_PREFIX}cue_comment"),
            FeatureValue::Text(vec![String::new(); 3]),
        );
        let points = file.cue_points();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].label, "C");
    }

    // ----- the store ------------------------------------------------------

    #[test]
    fn save_then_load_round_trips_through_the_filesystem() {
        let dir = TempDir::new("store");
        let store = FeatureStore::new(dir.path());
        let uid = new_uid();

        let mut file = FeatureFile::new();
        file.set_beats_time_sec(&[0.0, 0.5, 1.0]);
        file.set_phrases(&[Phrase::new(0.0, 16.0, "INTRO")]);
        let path = store.save(&uid, &file).unwrap();

        assert!(path.exists());
        assert_eq!(path.extension().unwrap(), FEATURE_EXTENSION);
        assert_eq!(store.load(&uid).unwrap(), file);
        assert!(store.exists(&uid).unwrap());
    }

    #[test]
    fn saving_creates_the_library_directory() {
        let dir = TempDir::new("mkdir");
        let store = FeatureStore::new(dir.join("not-yet"));
        let uid = new_uid();
        store.save(&uid, &FeatureFile::new()).unwrap();
        assert!(store.load(&uid).is_ok());
    }

    #[test]
    fn a_missing_track_is_none_from_load_optional_and_an_error_from_load() {
        let dir = TempDir::new("missing");
        let store = FeatureStore::new(dir.path());
        let uid = new_uid();
        assert!(store.load_optional(&uid).unwrap().is_none());
        assert!(!store.exists(&uid).unwrap());
        assert!(matches!(
            store.load(&uid),
            Err(StoreError::FeatureMissing { .. })
        ));
    }

    /// The failure Python leaves uncaught: `np.load` raises `BadZipFile`,
    /// nothing catches it, and the track becomes permanently unloadable. Here
    /// "damaged" is distinguishable from "not analysed yet".
    #[test]
    fn a_damaged_feature_file_is_an_error_not_an_empty_result() {
        let dir = TempDir::new("damaged");
        let store = FeatureStore::new(dir.path());
        let uid = new_uid();
        let path = store.path_for(&uid).unwrap();
        std::fs::write(&path, b"PK\x03\x04 not actually a feature file at all").unwrap();

        let err = store.load_optional(&uid).unwrap_err();
        assert!(matches!(err, StoreError::FeatureFormat { .. }), "{err:?}");
        assert_eq!(err.path(), Some(path.as_path()));
        assert!(err.to_string().contains(&uid), "the message must name the file: {err}");
    }

    #[test]
    fn a_uid_that_is_not_a_uuid_never_becomes_a_path() {
        let dir = TempDir::new("baduid");
        let store = FeatureStore::new(dir.path());
        for bad in ["", "../../etc/passwd", "not-a-uuid", "2c5ea4c0-4067-11e9-8bad-9b1deb4d3b7d"] {
            assert!(store.path_for(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn saving_replaces_the_previous_file_and_leaves_no_temp_behind() {
        let dir = TempDir::new("atomic");
        let store = FeatureStore::new(dir.path());
        let uid = new_uid();

        let mut first = FeatureFile::new();
        first.set_beats_time_sec(&[1.0]);
        store.save(&uid, &first).unwrap();
        let mut second = FeatureFile::new();
        second.set_beats_time_sec(&[2.0, 3.0]);
        store.save(&uid, &second).unwrap();

        assert_eq!(store.load(&uid).unwrap().beats_time_sec(), vec![2.0, 3.0]);
        let leftovers: Vec<PathBuf> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn every_temp_name_is_unique() {
        let target = Path::new("/library/x.mxf");
        let names: std::collections::HashSet<String> =
            (0..100).map(|_| temp_name(target)).collect();
        assert_eq!(names.len(), 100);
    }

    #[test]
    fn delete_reports_whether_a_file_was_there() {
        let dir = TempDir::new("delete");
        let store = FeatureStore::new(dir.path());
        let uid = new_uid();
        store.save(&uid, &FeatureFile::new()).unwrap();
        assert!(store.delete(&uid).unwrap());
        assert!(!store.delete(&uid).unwrap());
    }

    #[test]
    fn list_uids_ignores_everything_that_is_not_a_feature_file() {
        let dir = TempDir::new("list");
        let store = FeatureStore::new(dir.path());
        let mut uids: Vec<String> = (0..3).map(|_| new_uid()).collect();
        for uid in &uids {
            store.save(uid, &FeatureFile::new()).unwrap();
        }
        std::fs::write(dir.join("library.db"), b"x").unwrap();
        std::fs::write(dir.join("VERSION"), b"0.3.0\n").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::write(dir.join("not-a-uid.mxf"), b"x").unwrap();

        uids.sort();
        assert_eq!(store.list_uids().unwrap(), uids);
    }

    #[test]
    fn listing_a_directory_that_does_not_exist_yields_nothing() {
        let dir = TempDir::new("nodir");
        let store = FeatureStore::new(dir.join("absent"));
        assert!(store.list_uids().unwrap().is_empty());
    }

    #[test]
    fn crc32_matches_known_vectors() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }
}
