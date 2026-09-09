//! The SQLite track library.
//!
//! A port of `core/library_handler.py`. The schema is byte-for-byte the one the
//! Python app creates, so an existing `library.db` opens unchanged and a
//! library written here is still readable by the Python build.
//!
//! The behavioural difference is error handling: every operation returns
//! [`StoreError`] rather than letting a `sqlite3` exception escape. Python
//! catches none of them anywhere in the repository, so a corrupt or locked
//! `library.db` aborts startup.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mixlyzer_core::linear::{BpmSegmentRow, KeySegmentRow};
use mixlyzer_core::track::{canonical_uid, normalize_track_path};
use mixlyzer_core::{Key, Track};
use rusqlite::{named_params, Connection, OpenFlags, Row};

use crate::error::StoreError;

/// Columns that may appear in an `ORDER BY` clause.
///
/// `order_by` is interpolated into SQL rather than bound, so it must never
/// carry untrusted text. Kept identical to the Python `_ORDERABLE_COLUMNS`.
pub const ORDERABLE_COLUMNS: [&str; 14] = [
    "path",
    "uid",
    "title",
    "artist",
    "album",
    "bpm",
    "key",
    "duration",
    "total_samples",
    "rating",
    "added_ts",
    "comment",
    "file_mtime",
    "file_size",
];

/// The ordering used when a caller does not ask for one.
pub const DEFAULT_ORDER_BY: &str = "added_ts DESC";

/// How long to wait for a competing writer before reporting [`StoreError::Busy`].
const BUSY_TIMEOUT_MS: u32 = 5_000;

/// Validate an `ORDER BY` clause against the column whitelist.
///
/// Accepts `<column>` or `<column> ASC|DESC`, case-insensitively for the
/// direction. Anything else — an unknown column, extra tokens, a trailing
/// semicolon, an injected `UNION` — falls back to `default`. This is a direct
/// port of the Python `_safe_order_by`, including its fallback-rather-than-fail
/// behaviour, so a stale saved sort order in the UI degrades to the default
/// instead of erroring.
pub fn safe_order_by(order_by: &str, default: &str) -> String {
    let tokens: Vec<&str> = order_by.split_whitespace().collect();
    match tokens.as_slice() {
        [column] if ORDERABLE_COLUMNS.contains(column) => (*column).to_string(),
        [column, direction]
            if ORDERABLE_COLUMNS.contains(column)
                && (direction.eq_ignore_ascii_case("ASC")
                    || direction.eq_ignore_ascii_case("DESC")) =>
        {
            format!("{column} {direction}")
        }
        _ => default.to_string(),
    }
}

/// One side of a transition: the segment a mix moves out of, or into.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionSide {
    /// Position of the segment within its track.
    pub seq_index: i64,
    /// Segment start, in seconds from the beginning of the track.
    pub start_sec: f64,
    /// Segment end, in seconds.
    pub end_sec: f64,
    /// Segment length in seconds, as stored.
    pub duration_sec: f64,
    /// Set for BPM searches, `None` for key searches.
    pub bpm: Option<f64>,
    /// Set for key searches, `None` for BPM searches.
    pub key: Option<Key>,
    /// The label as stored, which may be empty for rows written before labels.
    pub key_label: String,
}

/// A pair of segments in one track that a DJ could mix across.
///
/// Python returns a flat 18-field `LinearTransitionRow` with `from_`/`to_`
/// prefixes on every column; grouping the two sides makes the symmetry explicit
/// and stops callers reading `from_bpm` against `to_start_sec`.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// The track both segments belong to.
    pub track_uid: String,
    /// The track's stored path, for opening the file.
    pub path: String,
    /// The track's title, for display.
    pub title: String,
    /// The track's artist, for display.
    pub artist: String,
    /// The segment being mixed out of.
    pub from: TransitionSide,
    /// The segment being mixed into.
    pub to: TransitionSide,
}

/// An open library database.
#[derive(Debug)]
pub struct Library {
    conn: Connection,
    path: PathBuf,
}

impl Library {
    /// Open `path`, creating the file and schema if they are not there.
    ///
    /// Returns [`StoreError::Corrupt`] for a file that is not a database and
    /// [`StoreError::Busy`] when another process holds it, rather than
    /// propagating an untyped `sqlite3.DatabaseError` the way Python does.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_URI
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|source| StoreError::Open {
            path: path.clone(),
            source,
        })?;
        let library = Library { conn, path };
        library.apply_pragmas()?;
        library.ensure_schema()?;
        Ok(library)
    }

    /// Open a private in-memory library. Used by tests and dry runs.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory().map_err(|source| StoreError::Open {
            path: PathBuf::from(":memory:"),
            source,
        })?;
        let library = Library {
            conn,
            path: PathBuf::from(":memory:"),
        };
        library.apply_pragmas()?;
        library.ensure_schema()?;
        Ok(library)
    }

    /// The file this library was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Borrow the underlying connection, for callers that need a query this
    /// module does not wrap.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    fn fail(&self, context: &str, source: rusqlite::Error) -> StoreError {
        StoreError::from_sqlite(&self.path, context, source)
    }

    fn apply_pragmas(&self) -> Result<(), StoreError> {
        // The first pragma touches the file header, so this is where a garbage
        // `library.db` is caught — before any caller can act on a half-open
        // connection.
        self.conn
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| self.fail("enable WAL journal", e))?;
        for (name, value) in [
            ("synchronous", "NORMAL"),
            ("foreign_keys", "ON"),
            ("temp_store", "MEMORY"),
            ("mmap_size", "3000000000"),
        ] {
            self.conn
                .pragma_update(None, name, value)
                .map_err(|e| self.fail("apply pragmas", e))?;
        }
        // Python sets no busy timeout, so two windows analysing at once raise
        // "database is locked" immediately. Wait a little first.
        self.conn
            .busy_timeout(std::time::Duration::from_millis(u64::from(BUSY_TIMEOUT_MS)))
            .map_err(|e| self.fail("set busy timeout", e))?;
        Ok(())
    }

    fn ensure_schema(&self) -> Result<(), StoreError> {
        let schema = r#"
        CREATE TABLE IF NOT EXISTS tracks (
            path        TEXT PRIMARY KEY,
            uid         TEXT,
            title       TEXT NOT NULL DEFAULT '',
            artist      TEXT DEFAULT '',
            album       TEXT DEFAULT '',
            bpm         REAL,
            key         INTEGER,
            duration    REAL,
            total_samples INTEGER,
            rating      INTEGER DEFAULT 0,
            added_ts    INTEGER NOT NULL,
            comment     TEXT DEFAULT '',
            file_mtime  REAL DEFAULT 0,
            file_size   INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS track_bpm_segments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            track_uid     TEXT NOT NULL,
            seq_index     INTEGER NOT NULL,
            start_sec     REAL NOT NULL,
            end_sec       REAL NOT NULL,
            duration_sec  REAL NOT NULL,
            bpm           REAL,
            bpm_rounded   INTEGER,
            time_signature INTEGER DEFAULT 4,
            UNIQUE(track_uid, seq_index)
        );
        CREATE TABLE IF NOT EXISTS track_key_segments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            track_uid     TEXT NOT NULL,
            seq_index     INTEGER NOT NULL,
            start_sec     REAL NOT NULL,
            end_sec       REAL NOT NULL,
            duration_sec  REAL NOT NULL,
            key_value     INTEGER,
            key_label     TEXT DEFAULT '',
            UNIQUE(track_uid, seq_index)
        );
        "#;
        self.conn.execute_batch(schema).map_err(|source| {
            // A file that is not a database fails here rather than at open().
            StoreError::from_sqlite(&self.path, "create schema", source)
        })?;

        // Self-healing upgrade for libraries created before per-segment time
        // signatures existed. Kept from Python: it avoids a migration step for
        // a purely additive column.
        if !self.column_exists("track_bpm_segments", "time_signature")? {
            self.conn
                .execute(
                    "ALTER TABLE track_bpm_segments ADD COLUMN time_signature INTEGER DEFAULT 4;",
                    [],
                )
                .map_err(|source| StoreError::Schema {
                    path: self.path.clone(),
                    context: "add track_bpm_segments.time_signature".to_string(),
                    source,
                })?;
        }

        let indexes = r#"
        CREATE INDEX IF NOT EXISTS idx_tbs_track_uid ON track_bpm_segments(track_uid);
        CREATE INDEX IF NOT EXISTS idx_tbs_bpm_duration ON track_bpm_segments(bpm_rounded, duration_sec);
        CREATE INDEX IF NOT EXISTS idx_tks_track_uid ON track_key_segments(track_uid);
        CREATE INDEX IF NOT EXISTS idx_tks_key_duration ON track_key_segments(key_value, duration_sec);
        "#;
        self.conn
            .execute_batch(indexes)
            .map_err(|e| self.fail("create indexes", e))?;
        Ok(())
    }

    /// Whether `table` has a column named `column`.
    pub fn column_exists(&self, table: &str, column: &str) -> Result<bool, StoreError> {
        // `table` is never caller-supplied; the two call sites pass literals.
        let mut stmt = self
            .conn
            .prepare(&format!("PRAGMA table_info({table});"))
            .map_err(|e| self.fail("inspect table", e))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| self.fail("inspect table", e))?;
        while let Some(row) = rows.next().map_err(|e| self.fail("inspect table", e))? {
            let name: String = row.get(1).map_err(|e| self.fail("inspect table", e))?;
            if name == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // ----- tracks ---------------------------------------------------------

    /// Insert `track`, or update the existing row for its path.
    ///
    /// Preserves the two fields a re-analysis must not clobber, exactly as the
    /// Python `COALESCE`/`CASE` clauses do: an existing `added_ts` survives, and
    /// an existing `uid` survives when the incoming row has none (losing it
    /// would orphan the track's feature file).
    ///
    /// Differs from Python in two places, both deliberate:
    /// * A row arriving with `added_ts <= 0` is stamped with the current time
    ///   rather than stored as 0, so "recently added" ordering is never
    ///   degenerate. Python only does this in `TrackRow.from_meta`, so rows
    ///   built any other way sort as epoch zero forever.
    /// * The legacy-row adoption below is looked up by the caller's *original*
    ///   path. Python looks it up by the already-normalised path, so its
    ///   adoption branch can never fire and a re-scan of a library written
    ///   before path normalisation silently duplicates every track. Adopting
    ///   also carries `added_ts` across, which the `ON CONFLICT` clause cannot
    ///   do because the row is being re-keyed rather than updated in place.
    pub fn upsert(&self, track: &Track) -> Result<(), StoreError> {
        let mut row = track.clone();
        let normalized = normalize_track_path(&track.path);

        if let Some(existing) = self.get(&track.path)? {
            if existing.path != normalized {
                if row.uid.is_none() {
                    row.uid = existing.uid.clone();
                }
                // Re-keying a row is still an update of that row, so the
                // fields the ON CONFLICT clause below would have preserved
                // must be carried across the delete/insert by hand.
                if existing.added_ts > 0 {
                    row.added_ts = existing.added_ts;
                }
                self.conn
                    .execute("DELETE FROM tracks WHERE path = ?1;", [&existing.path])
                    .map_err(|e| self.fail("drop legacy track row", e))?;
            }
        }
        row.path = normalized;
        if row.added_ts <= 0 {
            row.added_ts = now_epoch_secs();
        }

        let sql = r#"
        INSERT INTO tracks(path, uid, title, artist, album, bpm, key, duration, total_samples,
                           rating, added_ts, comment, file_mtime, file_size)
        VALUES(:path, :uid, :title, :artist, :album, :bpm, :key, :duration, :total_samples,
               :rating, :added_ts, :comment, :file_mtime, :file_size)
        ON CONFLICT(path) DO UPDATE SET
            uid           = COALESCE(excluded.uid, tracks.uid),
            title         = COALESCE(excluded.title, tracks.title),
            artist        = COALESCE(excluded.artist, tracks.artist),
            album         = COALESCE(excluded.album, tracks.album),
            bpm           = COALESCE(excluded.bpm, tracks.bpm),
            key           = COALESCE(excluded.key, tracks.key),
            duration      = COALESCE(excluded.duration, tracks.duration),
            total_samples = COALESCE(excluded.total_samples, tracks.total_samples),
            rating        = COALESCE(excluded.rating, tracks.rating),
            added_ts      = CASE WHEN tracks.added_ts IS NULL OR tracks.added_ts = 0
                                 THEN excluded.added_ts ELSE tracks.added_ts END,
            comment       = COALESCE(excluded.comment, tracks.comment),
            file_mtime    = COALESCE(excluded.file_mtime, tracks.file_mtime),
            file_size     = COALESCE(excluded.file_size, tracks.file_size);
        "#;
        self.conn
            .execute(
                sql,
                named_params! {
                    ":path": row.path,
                    ":uid": row.uid,
                    ":title": row.title,
                    ":artist": row.artist,
                    ":album": row.album,
                    ":bpm": row.bpm,
                    ":key": row.key.map(|k| i64::from(k.index())),
                    ":duration": row.duration,
                    ":total_samples": row.total_samples,
                    ":rating": row.rating,
                    ":added_ts": row.added_ts,
                    ":comment": row.comment,
                    ":file_mtime": row.file_mtime,
                    ":file_size": row.file_size,
                },
            )
            .map_err(|e| self.fail("upsert track", e))?;
        Ok(())
    }

    /// Insert or update many tracks in one transaction.
    pub fn upsert_many<'a>(
        &self,
        tracks: impl IntoIterator<Item = &'a Track>,
    ) -> Result<usize, StoreError> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| self.fail("begin transaction", e))?;
        let mut count = 0;
        for track in tracks {
            self.upsert(track)?;
            count += 1;
        }
        tx.commit().map_err(|e| self.fail("commit tracks", e))?;
        Ok(count)
    }

    /// Look a track up by path.
    ///
    /// Tries the normalised path first, then the raw string, which is what
    /// finds rows written before path normalisation existed.
    ///
    /// Python has a third fallback comparing `REPLACE(LOWER(path), '/', '\')`
    /// against the *forward-slashed* normalised path. Those two forms can only
    /// be equal for a path with no separator at all, and then only on Windows
    /// where normalisation already lower-cases — so the clause is dead code and
    /// is not ported.
    pub fn get(&self, path: &str) -> Result<Option<Track>, StoreError> {
        let normalized = normalize_track_path(path);
        if let Some(track) = self.get_exact(&normalized)? {
            return Ok(Some(track));
        }
        let raw = path.trim();
        if !raw.is_empty() && raw != normalized {
            return self.get_exact(raw);
        }
        Ok(None)
    }

    fn get_exact(&self, path: &str) -> Result<Option<Track>, StoreError> {
        self.conn
            .query_row("SELECT * FROM tracks WHERE path = ?1;", [path], |row| {
                track_from_row(row)
            })
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(self.fail("get track", other)),
            })
    }

    /// Look a track up by uid.
    ///
    /// A uid that is not a canonical UUIDv4 matches nothing rather than
    /// erroring, matching the Python behaviour of returning `None`.
    pub fn get_by_uid(&self, uid: &str) -> Result<Option<Track>, StoreError> {
        let Ok(uid) = canonical_uid(uid) else {
            return Ok(None);
        };
        self.conn
            .query_row("SELECT * FROM tracks WHERE uid = ?1;", [&uid], |row| {
                track_from_row(row)
            })
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(self.fail("get track by uid", other)),
            })
    }

    /// Every track, newest first.
    pub fn list_all(&self) -> Result<Vec<Track>, StoreError> {
        self.list_ordered(DEFAULT_ORDER_BY, None, 0)
    }

    /// Every track in a caller-chosen order, optionally paged.
    ///
    /// `order_by` passes through [`safe_order_by`], so an unrecognised clause
    /// silently becomes [`DEFAULT_ORDER_BY`].
    pub fn list_ordered(
        &self,
        order_by: &str,
        limit: Option<i64>,
        offset: i64,
    ) -> Result<Vec<Track>, StoreError> {
        let order = safe_order_by(order_by, DEFAULT_ORDER_BY);
        let mut sql = format!("SELECT * FROM tracks ORDER BY {order}");
        if limit.is_some() {
            sql.push_str(" LIMIT :limit OFFSET :offset");
        }
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| self.fail("list tracks", e))?;
        let rows = match limit {
            Some(limit) => stmt.query(named_params! { ":limit": limit, ":offset": offset }),
            None => stmt.query([]),
        }
        .map_err(|e| self.fail("list tracks", e))?;
        collect_rows(rows, track_from_row).map_err(|e| self.fail("list tracks", e))
    }

    /// Number of rows in `tracks`.
    pub fn count(&self) -> Result<u64, StoreError> {
        self.conn
            .query_row("SELECT COUNT(*) FROM tracks;", [], |row| row.get::<_, i64>(0))
            .map(|n| n as u64)
            .map_err(|e| self.fail("count tracks", e))
    }

    /// Delete a track and its segment rows. Returns whether a row was removed.
    pub fn delete(&self, path: &str) -> Result<bool, StoreError> {
        let Some(existing) = self.get(path)? else {
            return Ok(false);
        };
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| self.fail("begin transaction", e))?;
        if let Some(uid) = existing.uid.as_deref() {
            for table in ["track_bpm_segments", "track_key_segments"] {
                tx.execute(
                    &format!("DELETE FROM {table} WHERE track_uid = ?1;"),
                    [uid],
                )
                .map_err(|e| self.fail("delete segments", e))?;
            }
        }
        let removed = tx
            .execute("DELETE FROM tracks WHERE path = ?1;", [&existing.path])
            .map_err(|e| self.fail("delete track", e))?;
        tx.commit().map_err(|e| self.fail("commit delete", e))?;
        Ok(removed > 0)
    }

    // ----- segments -------------------------------------------------------

    /// Replace a track's BPM segments with `rows`, atomically.
    ///
    /// An empty `rows` clears the track's segments, which is how a re-analysis
    /// that found nothing is recorded.
    pub fn replace_bpm_segments(
        &self,
        track_uid: &str,
        rows: &[BpmSegmentRow],
    ) -> Result<(), StoreError> {
        let uid = canonical_uid(track_uid)?;
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| self.fail("begin transaction", e))?;
        tx.execute("DELETE FROM track_bpm_segments WHERE track_uid = ?1;", [&uid])
            .map_err(|e| self.fail("clear bpm segments", e))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO track_bpm_segments(track_uid, seq_index, start_sec, end_sec,
                         duration_sec, bpm, bpm_rounded, time_signature)
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
                )
                .map_err(|e| self.fail("insert bpm segments", e))?;
            for row in rows {
                stmt.execute(rusqlite::params![
                    &uid,
                    row.seq_index as i64,
                    row.start_sec,
                    row.end_sec,
                    row.duration_sec,
                    row.bpm,
                    row.bpm_rounded,
                    i64::from(row.time_signature),
                ])
                .map_err(|e| self.fail("insert bpm segments", e))?;
            }
        }
        tx.commit().map_err(|e| self.fail("commit bpm segments", e))
    }

    /// Replace a track's key segments with `rows`, atomically.
    pub fn replace_key_segments(
        &self,
        track_uid: &str,
        rows: &[KeySegmentRow],
    ) -> Result<(), StoreError> {
        let uid = canonical_uid(track_uid)?;
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| self.fail("begin transaction", e))?;
        tx.execute("DELETE FROM track_key_segments WHERE track_uid = ?1;", [&uid])
            .map_err(|e| self.fail("clear key segments", e))?;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO track_key_segments(track_uid, seq_index, start_sec, end_sec,
                         duration_sec, key_value, key_label)
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7);",
                )
                .map_err(|e| self.fail("insert key segments", e))?;
            for row in rows {
                stmt.execute(rusqlite::params![
                    &uid,
                    row.seq_index as i64,
                    row.start_sec,
                    row.end_sec,
                    row.duration_sec,
                    i64::from(row.key_value),
                    &row.key_label,
                ])
                .map_err(|e| self.fail("insert key segments", e))?;
            }
        }
        tx.commit().map_err(|e| self.fail("commit key segments", e))
    }

    /// A track's BPM segments, in sequence order.
    ///
    /// Rows whose `bpm` is NULL are skipped: they carry no usable tempo and
    /// [`BpmSegmentRow`] has nowhere to put the absence. Only libraries written
    /// by hand or by a pre-0.2.0 build contain them.
    pub fn bpm_segments(&self, track_uid: &str) -> Result<Vec<BpmSegmentRow>, StoreError> {
        let uid = canonical_uid(track_uid)?;
        let mut stmt = self
            .conn
            .prepare(
                "SELECT seq_index, start_sec, end_sec, duration_sec, bpm, bpm_rounded,
                        COALESCE(time_signature, 4) AS time_signature
                 FROM track_bpm_segments WHERE track_uid = ?1 ORDER BY seq_index ASC;",
            )
            .map_err(|e| self.fail("read bpm segments", e))?;
        let mut rows = stmt
            .query([&uid])
            .map_err(|e| self.fail("read bpm segments", e))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| self.fail("read bpm segments", e))?
        {
            let bpm: Option<f64> = row.get("bpm").map_err(|e| self.fail("read bpm segments", e))?;
            let Some(bpm) = bpm else { continue };
            out.push(BpmSegmentRow {
                seq_index: row
                    .get::<_, i64>("seq_index")
                    .map_err(|e| self.fail("read bpm segments", e))? as usize,
                start_sec: row.get("start_sec").map_err(|e| self.fail("read bpm segments", e))?,
                end_sec: row.get("end_sec").map_err(|e| self.fail("read bpm segments", e))?,
                duration_sec: row
                    .get("duration_sec")
                    .map_err(|e| self.fail("read bpm segments", e))?,
                bpm,
                bpm_rounded: row
                    .get::<_, Option<i64>>("bpm_rounded")
                    .map_err(|e| self.fail("read bpm segments", e))?
                    .unwrap_or_else(|| bpm.round() as i64),
                time_signature: row
                    .get::<_, i64>("time_signature")
                    .map_err(|e| self.fail("read bpm segments", e))?
                    .clamp(1, 255) as u8,
            });
        }
        Ok(out)
    }

    /// A track's key segments, in sequence order.
    ///
    /// Rows whose `key_value` is NULL or outside `0..24` are skipped, for the
    /// same reason as [`Library::bpm_segments`].
    pub fn key_segments(&self, track_uid: &str) -> Result<Vec<KeySegmentRow>, StoreError> {
        let uid = canonical_uid(track_uid)?;
        let mut stmt = self
            .conn
            .prepare(
                "SELECT seq_index, start_sec, end_sec, duration_sec, key_value,
                        COALESCE(key_label, '') AS key_label
                 FROM track_key_segments WHERE track_uid = ?1 ORDER BY seq_index ASC;",
            )
            .map_err(|e| self.fail("read key segments", e))?;
        let mut rows = stmt
            .query([&uid])
            .map_err(|e| self.fail("read key segments", e))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| self.fail("read key segments", e))?
        {
            let key_value: Option<i64> = row
                .get("key_value")
                .map_err(|e| self.fail("read key segments", e))?;
            let Some(key_value) = key_value.filter(|v| (0..24).contains(v)) else {
                continue;
            };
            out.push(KeySegmentRow {
                seq_index: row
                    .get::<_, i64>("seq_index")
                    .map_err(|e| self.fail("read key segments", e))? as usize,
                start_sec: row.get("start_sec").map_err(|e| self.fail("read key segments", e))?,
                end_sec: row.get("end_sec").map_err(|e| self.fail("read key segments", e))?,
                duration_sec: row
                    .get("duration_sec")
                    .map_err(|e| self.fail("read key segments", e))?,
                key_value: key_value as u8,
                key_label: row.get("key_label").map_err(|e| self.fail("read key segments", e))?,
            });
        }
        Ok(out)
    }

    // ----- transition search ---------------------------------------------

    /// Find tracks that move from around `from_bpm` to around `to_bpm` across
    /// two *adjacent* segments.
    ///
    /// `tolerance_percent` widens each anchor into a range: 128 BPM at 2%
    /// matches 125.44..130.56. A negative tolerance is treated as zero, and a
    /// non-finite or non-positive anchor matches nothing (Python builds a
    /// `BETWEEN nan AND nan` clause instead, which silently returns no rows but
    /// looks like an empty library).
    pub fn search_bpm_transitions(
        &self,
        from_bpm: f64,
        to_bpm: f64,
        tolerance_percent: f64,
        min_duration_sec: f64,
        require_first_segment: bool,
    ) -> Result<Vec<Transition>, StoreError> {
        let (Some((from_min, from_max)), Some((to_min, to_max))) = (
            bpm_range(from_bpm, tolerance_percent),
            bpm_range(to_bpm, tolerance_percent),
        ) else {
            return Ok(Vec::new());
        };
        let first_clause = if require_first_segment {
            "AND a.seq_index = 0"
        } else {
            ""
        };
        let sql = format!(
            r#"
            SELECT t.uid AS track_uid, t.path, t.title, t.artist,
                   a.seq_index AS from_seq_index, a.start_sec AS from_start_sec,
                   a.end_sec AS from_end_sec, a.duration_sec AS from_duration_sec, a.bpm AS from_bpm,
                   b.seq_index AS to_seq_index, b.start_sec AS to_start_sec,
                   b.end_sec AS to_end_sec, b.duration_sec AS to_duration_sec, b.bpm AS to_bpm
            FROM track_bpm_segments a
            JOIN track_bpm_segments b
              ON b.track_uid = a.track_uid AND b.seq_index = a.seq_index + 1
            JOIN tracks t ON t.uid = a.track_uid
            WHERE a.duration_sec >= :min_duration
              AND b.duration_sec >= :min_duration
              AND a.bpm IS NOT NULL AND b.bpm IS NOT NULL
              AND a.bpm BETWEEN :from_min AND :from_max
              AND b.bpm BETWEEN :to_min AND :to_max
              {first_clause}
            ORDER BY t.added_ts DESC, t.title COLLATE NOCASE ASC, a.seq_index ASC;
            "#
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| self.fail("search bpm transitions", e))?;
        let rows = stmt
            .query(named_params! {
                ":min_duration": min_duration_sec.max(0.0),
                ":from_min": from_min,
                ":from_max": from_max,
                ":to_min": to_min,
                ":to_max": to_max,
            })
            .map_err(|e| self.fail("search bpm transitions", e))?;
        collect_rows(rows, bpm_transition_from_row)
            .map_err(|e| self.fail("search bpm transitions", e))
    }

    /// Find tracks that move from a key compatible with `from_key` to one
    /// compatible with `to_key`, across any two segments in order.
    ///
    /// Unlike the BPM search the second segment need only come *later*, not
    /// immediately after: a harmonic move is still a harmonic move with a
    /// bridge in between. Both anchors expand through
    /// [`Key::harmonic_neighbours`].
    pub fn search_harmonic_key_transitions(
        &self,
        from_key: Key,
        to_key: Key,
        min_duration_sec: f64,
        require_first_segment: bool,
    ) -> Result<Vec<Transition>, StoreError> {
        let from_keys = from_key.harmonic_neighbours();
        let to_keys = to_key.harmonic_neighbours();
        let first_clause = if require_first_segment {
            "AND a.seq_index = 0"
        } else {
            ""
        };
        // One placeholder per key: the counts are fixed by the wheel, but
        // generating them keeps the binding honest if that ever changes.
        let from_placeholders = placeholders(2, from_keys.len());
        let to_placeholders = placeholders(2 + from_keys.len(), to_keys.len());
        let sql = format!(
            r#"
            SELECT t.uid AS track_uid, t.path, t.title, t.artist,
                   a.seq_index AS from_seq_index, a.start_sec AS from_start_sec,
                   a.end_sec AS from_end_sec, a.duration_sec AS from_duration_sec,
                   a.key_value AS from_key_value, a.key_label AS from_key_label,
                   b.seq_index AS to_seq_index, b.start_sec AS to_start_sec,
                   b.end_sec AS to_end_sec, b.duration_sec AS to_duration_sec,
                   b.key_value AS to_key_value, b.key_label AS to_key_label
            FROM track_key_segments a
            JOIN track_key_segments b
              ON b.track_uid = a.track_uid AND b.seq_index > a.seq_index
            JOIN tracks t ON t.uid = a.track_uid
            WHERE a.duration_sec >= ?1
              AND b.duration_sec >= ?1
              AND a.key_value IN ({from_placeholders})
              AND b.key_value IN ({to_placeholders})
              {first_clause}
            ORDER BY t.added_ts DESC, t.title COLLATE NOCASE ASC, a.seq_index ASC;
            "#
        );
        let mut params: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(min_duration_sec.max(0.0))];
        for key in from_keys.iter().chain(to_keys.iter()) {
            params.push(Box::new(i64::from(key.index())));
        }
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| self.fail("search key transitions", e))?;
        let rows = stmt
            .query(rusqlite::params_from_iter(params.iter()))
            .map_err(|e| self.fail("search key transitions", e))?;
        collect_rows(rows, key_transition_from_row)
            .map_err(|e| self.fail("search key transitions", e))
    }
}

/// Epoch seconds, or 0 if the clock is before the epoch.
fn now_epoch_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `?n, ?n+1, ...` for `count` positional parameters starting at `start`.
fn placeholders(start: usize, count: usize) -> String {
    (0..count)
        .map(|i| format!("?{}", start + i))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The inclusive BPM range for an anchor, or `None` if the anchor is unusable.
fn bpm_range(bpm: f64, tolerance_percent: f64) -> Option<(f64, f64)> {
    if !bpm.is_finite() || bpm <= 0.0 {
        return None;
    }
    let tolerance = if tolerance_percent.is_finite() {
        tolerance_percent.max(0.0)
    } else {
        0.0
    };
    let margin = bpm * tolerance / 100.0;
    Some((bpm - margin, bpm + margin))
}

/// Drain a `Rows` cursor through `map`.
fn collect_rows<T>(
    mut rows: rusqlite::Rows<'_>,
    map: impl Fn(&Row<'_>) -> rusqlite::Result<T>,
) -> rusqlite::Result<Vec<T>> {
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(map(row)?);
    }
    Ok(out)
}

/// Map a `tracks` row.
///
/// A uid that is not a canonical UUIDv4 and a key outside `0..24` both become
/// `None` rather than failing the whole listing: one bad row must not hide the
/// rest of the library.
fn track_from_row(row: &Row<'_>) -> rusqlite::Result<Track> {
    let uid: Option<String> = row.get("uid")?;
    let key: Option<i64> = row.get("key")?;
    Ok(Track {
        path: row.get("path")?,
        uid: uid.and_then(|u| canonical_uid(&u).ok()),
        title: row.get::<_, Option<String>>("title")?.unwrap_or_default(),
        artist: row.get::<_, Option<String>>("artist")?.unwrap_or_default(),
        album: row.get::<_, Option<String>>("album")?.unwrap_or_default(),
        bpm: row.get("bpm")?,
        key: key
            .filter(|v| (0..24).contains(v))
            .map(Key::from_index),
        duration: row.get("duration")?,
        total_samples: row.get("total_samples")?,
        rating: row.get::<_, Option<i32>>("rating")?.unwrap_or(0),
        added_ts: row.get::<_, Option<i64>>("added_ts")?.unwrap_or(0),
        comment: row.get::<_, Option<String>>("comment")?.unwrap_or_default(),
        file_mtime: row.get::<_, Option<f64>>("file_mtime")?.unwrap_or(0.0),
        file_size: row.get::<_, Option<i64>>("file_size")?.unwrap_or(0),
    })
}

fn bpm_transition_from_row(row: &Row<'_>) -> rusqlite::Result<Transition> {
    Ok(Transition {
        track_uid: row.get("track_uid")?,
        path: row.get("path")?,
        title: row.get::<_, Option<String>>("title")?.unwrap_or_default(),
        artist: row.get::<_, Option<String>>("artist")?.unwrap_or_default(),
        from: TransitionSide {
            seq_index: row.get("from_seq_index")?,
            start_sec: row.get("from_start_sec")?,
            end_sec: row.get("from_end_sec")?,
            duration_sec: row.get("from_duration_sec")?,
            bpm: row.get("from_bpm")?,
            key: None,
            key_label: String::new(),
        },
        to: TransitionSide {
            seq_index: row.get("to_seq_index")?,
            start_sec: row.get("to_start_sec")?,
            end_sec: row.get("to_end_sec")?,
            duration_sec: row.get("to_duration_sec")?,
            bpm: row.get("to_bpm")?,
            key: None,
            key_label: String::new(),
        },
    })
}

fn key_transition_from_row(row: &Row<'_>) -> rusqlite::Result<Transition> {
    let from_key: Option<i64> = row.get("from_key_value")?;
    let to_key: Option<i64> = row.get("to_key_value")?;
    Ok(Transition {
        track_uid: row.get("track_uid")?,
        path: row.get("path")?,
        title: row.get::<_, Option<String>>("title")?.unwrap_or_default(),
        artist: row.get::<_, Option<String>>("artist")?.unwrap_or_default(),
        from: TransitionSide {
            seq_index: row.get("from_seq_index")?,
            start_sec: row.get("from_start_sec")?,
            end_sec: row.get("from_end_sec")?,
            duration_sec: row.get("from_duration_sec")?,
            bpm: None,
            key: from_key.filter(|v| (0..24).contains(v)).map(Key::from_index),
            key_label: row
                .get::<_, Option<String>>("from_key_label")?
                .unwrap_or_default(),
        },
        to: TransitionSide {
            seq_index: row.get("to_seq_index")?,
            start_sec: row.get("to_start_sec")?,
            end_sec: row.get("to_end_sec")?,
            duration_sec: row.get("to_duration_sec")?,
            bpm: None,
            key: to_key.filter(|v| (0..24).contains(v)).map(Key::from_index),
            key_label: row
                .get::<_, Option<String>>("to_key_label")?
                .unwrap_or_default(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use mixlyzer_core::key::Mode;
    use mixlyzer_core::track::new_uid;

    fn track(path: &str, uid: &str) -> Track {
        Track {
            path: path.to_string(),
            uid: Some(uid.to_string()),
            title: String::new(),
            artist: String::new(),
            album: String::new(),
            bpm: None,
            key: None,
            duration: None,
            total_samples: None,
            rating: 0,
            added_ts: 1_700_000_000,
            comment: String::new(),
            file_mtime: 0.0,
            file_size: 0,
        }
    }

    fn bpm_row(seq: usize, start: f64, end: f64, bpm: f64) -> BpmSegmentRow {
        BpmSegmentRow {
            seq_index: seq,
            start_sec: start,
            end_sec: end,
            duration_sec: end - start,
            bpm,
            bpm_rounded: bpm.round() as i64,
            time_signature: 4,
        }
    }

    fn key_row(seq: usize, start: f64, end: f64, key: Key) -> KeySegmentRow {
        KeySegmentRow {
            seq_index: seq,
            start_sec: start,
            end_sec: end,
            duration_sec: end - start,
            key_value: key.index(),
            key_label: key.camelot().to_string(),
        }
    }

    // ----- order-by whitelist --------------------------------------------

    #[test]
    fn safe_order_by_accepts_every_whitelisted_column() {
        for column in ORDERABLE_COLUMNS {
            assert_eq!(safe_order_by(column, "added_ts DESC"), column);
            assert_eq!(
                safe_order_by(&format!("{column} DESC"), "added_ts DESC"),
                format!("{column} DESC")
            );
        }
    }

    #[test]
    fn safe_order_by_direction_is_case_insensitive() {
        assert_eq!(safe_order_by("title asc", "x"), "title asc");
        assert_eq!(safe_order_by("title Desc", "x"), "title Desc");
    }

    #[test]
    fn safe_order_by_falls_back_for_unknown_columns_and_directions() {
        assert_eq!(safe_order_by("nonexistent", "added_ts DESC"), "added_ts DESC");
        assert_eq!(safe_order_by("title SIDEWAYS", "added_ts DESC"), "added_ts DESC");
        assert_eq!(safe_order_by("", "added_ts DESC"), "added_ts DESC");
        assert_eq!(safe_order_by("   ", "added_ts DESC"), "added_ts DESC");
    }

    /// `order_by` is the one value interpolated into SQL rather than bound.
    #[test]
    fn safe_order_by_rejects_injection_attempts() {
        let attacks = [
            "path; DROP TABLE tracks",
            "path;--",
            "title DESC; DELETE FROM tracks",
            "1 UNION SELECT * FROM tracks",
            "title, (SELECT uid FROM tracks)",
            "title/**/DESC",
            "(SELECT 1)",
            "title DESC LIMIT 1",
            "path ASC, uid DESC",
        ];
        for attack in attacks {
            assert_eq!(
                safe_order_by(attack, "added_ts DESC"),
                "added_ts DESC",
                "{attack:?} must not reach the SQL"
            );
        }
    }

    /// A whitelisted single token with a trailing semicolon is still not a
    /// whitelisted column, because the semicolon is part of the token.
    #[test]
    fn safe_order_by_does_not_strip_punctuation_to_find_a_match() {
        assert_eq!(safe_order_by("path;", "added_ts DESC"), "added_ts DESC");
        assert_eq!(safe_order_by("'path'", "added_ts DESC"), "added_ts DESC");
    }

    // ----- open / schema --------------------------------------------------

    #[test]
    fn a_fresh_library_is_empty() {
        let db = Library::open_in_memory().unwrap();
        assert_eq!(db.count().unwrap(), 0);
        assert!(db.list_all().unwrap().is_empty());
        assert!(db.get("anything").unwrap().is_none());
    }

    #[test]
    fn opening_creates_the_file_and_schema() {
        let dir = TempDir::new("open");
        let path = dir.join("library.db");
        {
            let db = Library::open(&path).unwrap();
            assert!(db.column_exists("tracks", "total_samples").unwrap());
            assert!(db
                .column_exists("track_bpm_segments", "time_signature")
                .unwrap());
        }
        assert!(path.exists());
        // Re-opening an existing library must not disturb it.
        Library::open(&path).unwrap();
    }

    /// A library created before per-segment time signatures gets the column
    /// added on open, with no migration step.
    #[test]
    fn an_old_segment_table_gains_time_signature_on_open() {
        let dir = TempDir::new("selfheal");
        let path = dir.join("library.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE track_bpm_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, track_uid TEXT NOT NULL,
                    seq_index INTEGER NOT NULL, start_sec REAL NOT NULL, end_sec REAL NOT NULL,
                    duration_sec REAL NOT NULL, bpm REAL, bpm_rounded INTEGER,
                    UNIQUE(track_uid, seq_index));",
            )
            .unwrap();
        }
        let db = Library::open(&path).unwrap();
        assert!(db
            .column_exists("track_bpm_segments", "time_signature")
            .unwrap());
    }

    /// The failure Python leaves uncaught: a damaged `library.db` must arrive
    /// as a typed error naming the file, not as a panic or a raw sqlite error.
    #[test]
    fn a_garbage_database_file_reports_typed_corruption_naming_the_path() {
        let dir = TempDir::new("corrupt");
        let path = dir.join("library.db");
        std::fs::write(&path, vec![b'\x7f'; 4096]).unwrap();

        let err = Library::open(&path).expect_err("garbage must not open");
        assert!(
            matches!(err, StoreError::Corrupt { .. }),
            "expected Corrupt, got {err:?}"
        );
        assert_eq!(err.path(), Some(path.as_path()));
        assert!(err.is_corruption());
        assert!(
            err.to_string().contains("library.db"),
            "the message must name the file: {err}"
        );
    }

    #[test]
    fn a_truncated_database_header_reports_corruption() {
        let dir = TempDir::new("truncated");
        let path = dir.join("library.db");
        // A real header cut short: enough to look like SQLite, not enough to be.
        std::fs::write(&path, b"SQLite format 3\0\x04\x00\x01\x01").unwrap();
        let err = Library::open(&path).expect_err("truncated must not open");
        assert!(err.is_corruption(), "got {err:?}");
    }

    // ----- tracks ---------------------------------------------------------

    #[test]
    fn upsert_then_get_round_trips_a_track() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let mut row = track("/music/a.flac", &uid);
        row.title = "Track".to_string();
        row.artist = "Artist".to_string();
        row.album = "Album".to_string();
        row.bpm = Some(128.5);
        row.key = Some(Key::from_index(21));
        row.duration = Some(301.25);
        row.total_samples = Some(13_283_000);
        row.rating = 4;
        row.comment = "note".to_string();
        row.file_mtime = 1_700_000_001.5;
        row.file_size = 42;
        db.upsert(&row).unwrap();

        let got = db.get("/music/a.flac").unwrap().unwrap();
        assert_eq!(got, row);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn unicode_paths_and_titles_survive_a_round_trip() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let path = "/음악/Björk – Jóga (리믹스)/트랙 01.flac";
        let mut row = track(path, &uid);
        row.title = "Jóga — 리믹스 🎧".to_string();
        row.artist = "Björk".to_string();
        db.upsert(&row).unwrap();

        let got = db.get(path).unwrap().unwrap();
        assert_eq!(got.title, "Jóga — 리믹스 🎧");
        assert_eq!(got.artist, "Björk");
        assert_eq!(got.path, path);
    }

    #[test]
    fn upsert_preserves_the_original_added_ts_and_uid() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let mut first = track("/music/a.flac", &uid);
        first.added_ts = 1_000;
        db.upsert(&first).unwrap();

        let mut second = track("/music/a.flac", &uid);
        second.added_ts = 9_999;
        second.uid = None;
        second.title = "Renamed".to_string();
        db.upsert(&second).unwrap();

        let got = db.get("/music/a.flac").unwrap().unwrap();
        assert_eq!(got.added_ts, 1_000, "added_ts must survive re-analysis");
        assert_eq!(
            got.uid.as_deref(),
            Some(uid.as_str()),
            "uid must survive: losing it orphans the feature file"
        );
        assert_eq!(got.title, "Renamed");
    }

    /// Python only stamps `added_ts` inside `TrackRow.from_meta`, so a row built
    /// any other way sorts as epoch zero forever.
    #[test]
    fn upsert_stamps_added_ts_when_the_caller_left_it_unset() {
        let db = Library::open_in_memory().unwrap();
        let mut row = track("/music/a.flac", &new_uid());
        row.added_ts = 0;
        db.upsert(&row).unwrap();
        assert!(db.get("/music/a.flac").unwrap().unwrap().added_ts > 0);
    }

    #[test]
    fn upsert_normalizes_the_stored_path() {
        let db = Library::open_in_memory().unwrap();
        db.upsert(&track("Music\\Sub\\..\\a.flac", &new_uid())).unwrap();
        let got = db.get("Music/a.flac").unwrap().unwrap();
        assert_eq!(got.path, "Music/a.flac");
    }

    /// Python looks the existing row up by the *already normalised* path, so
    /// this adoption branch can never fire and re-scanning a library written
    /// before normalisation duplicates every track under a second key.
    #[test]
    fn upsert_adopts_the_uid_of_a_legacy_unnormalized_row_and_removes_it() {
        let db = Library::open_in_memory().unwrap();
        let legacy_uid = new_uid();
        db.connection()
            .execute(
                "INSERT INTO tracks(path, uid, title, added_ts) VALUES(?1, ?2, 'Legacy', 5);",
                rusqlite::params!["D:\\Music\\Song.flac", &legacy_uid],
            )
            .unwrap();

        let mut incoming = track("D:\\Music\\Song.flac", "");
        incoming.uid = None;
        db.upsert(&incoming).unwrap();

        assert_eq!(db.count().unwrap(), 1, "the legacy row must not survive");
        let got = db.get("D:/Music/Song.flac").unwrap().unwrap();
        assert_eq!(got.path, "D:/Music/Song.flac");
        assert_eq!(got.uid.as_deref(), Some(legacy_uid.as_str()));
        assert_eq!(got.added_ts, 5, "the original added_ts is carried over");
    }

    #[test]
    fn get_falls_back_to_the_raw_path_for_legacy_rows() {
        let db = Library::open_in_memory().unwrap();
        db.connection()
            .execute(
                "INSERT INTO tracks(path, title, added_ts) VALUES('D:\\Music\\x.flac', 'x', 1);",
                [],
            )
            .unwrap();
        let got = db.get("D:\\Music\\x.flac").unwrap().unwrap();
        assert_eq!(got.path, "D:\\Music\\x.flac");
    }

    #[test]
    fn get_by_uid_finds_the_track_and_rejects_non_uuids() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        db.upsert(&track("/music/a.flac", &uid)).unwrap();
        assert_eq!(
            db.get_by_uid(&uid).unwrap().unwrap().path,
            "/music/a.flac"
        );
        assert!(db.get_by_uid("not-a-uuid").unwrap().is_none());
        assert!(db.get_by_uid(&new_uid()).unwrap().is_none());
    }

    /// One row with a hand-edited uid must not hide the rest of the library.
    #[test]
    fn a_row_with_an_invalid_uid_reads_back_with_none_rather_than_failing() {
        let db = Library::open_in_memory().unwrap();
        db.connection()
            .execute(
                "INSERT INTO tracks(path, uid, title, added_ts) VALUES('/a.flac', 'garbage', 'A', 1);",
                [],
            )
            .unwrap();
        db.upsert(&track("/b.flac", &new_uid())).unwrap();

        let all = db.list_all().unwrap();
        assert_eq!(all.len(), 2);
        let bad = all.iter().find(|t| t.path == "/a.flac").unwrap();
        assert_eq!(bad.uid, None);
    }

    #[test]
    fn an_out_of_range_key_reads_back_as_none() {
        let db = Library::open_in_memory().unwrap();
        db.connection()
            .execute(
                "INSERT INTO tracks(path, key, added_ts) VALUES('/a.flac', 99, 1);",
                [],
            )
            .unwrap();
        assert_eq!(db.get("/a.flac").unwrap().unwrap().key, None);
    }

    #[test]
    fn list_ordered_sorts_and_pages() {
        let db = Library::open_in_memory().unwrap();
        for (i, name) in ["c", "a", "b"].iter().enumerate() {
            let mut row = track(&format!("/music/{name}.flac"), &new_uid());
            row.title = name.to_string();
            row.added_ts = 100 + i as i64;
            db.upsert(&row).unwrap();
        }
        let titles: Vec<String> = db
            .list_ordered("title ASC", None, 0)
            .unwrap()
            .into_iter()
            .map(|t| t.title)
            .collect();
        assert_eq!(titles, vec!["a", "b", "c"]);

        let page: Vec<String> = db
            .list_ordered("title ASC", Some(1), 1)
            .unwrap()
            .into_iter()
            .map(|t| t.title)
            .collect();
        assert_eq!(page, vec!["b"]);

        // An injected clause degrades to the default rather than erroring.
        assert_eq!(db.list_ordered("title; DROP TABLE tracks", None, 0).unwrap().len(), 3);
        assert_eq!(db.count().unwrap(), 3);
    }

    #[test]
    fn upsert_many_writes_every_row() {
        let db = Library::open_in_memory().unwrap();
        let rows: Vec<Track> = (0..5)
            .map(|i| track(&format!("/music/{i}.flac"), &new_uid()))
            .collect();
        assert_eq!(db.upsert_many(rows.iter()).unwrap(), 5);
        assert_eq!(db.count().unwrap(), 5);
    }

    #[test]
    fn delete_removes_the_track_and_its_segments() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        db.upsert(&track("/music/a.flac", &uid)).unwrap();
        db.replace_bpm_segments(&uid, &[bpm_row(0, 0.0, 10.0, 128.0)])
            .unwrap();
        db.replace_key_segments(&uid, &[key_row(0, 0.0, 10.0, Key::from_index(0))])
            .unwrap();

        assert!(db.delete("/music/a.flac").unwrap());
        assert!(db.get("/music/a.flac").unwrap().is_none());
        assert!(db.bpm_segments(&uid).unwrap().is_empty());
        assert!(db.key_segments(&uid).unwrap().is_empty());
    }

    #[test]
    fn deleting_a_missing_track_reports_false_rather_than_failing() {
        let db = Library::open_in_memory().unwrap();
        assert!(!db.delete("/music/nope.flac").unwrap());
    }

    // ----- segments -------------------------------------------------------

    #[test]
    fn bpm_segments_round_trip_and_replace_wholesale() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let first = vec![bpm_row(0, 0.0, 10.0, 128.0), bpm_row(1, 10.0, 30.0, 140.0)];
        db.replace_bpm_segments(&uid, &first).unwrap();
        assert_eq!(db.bpm_segments(&uid).unwrap(), first);

        let second = vec![bpm_row(0, 0.0, 5.0, 90.0)];
        db.replace_bpm_segments(&uid, &second).unwrap();
        assert_eq!(db.bpm_segments(&uid).unwrap(), second);
    }

    #[test]
    fn key_segments_round_trip_and_replace_wholesale() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let rows = vec![
            key_row(0, 0.0, 10.0, Key::new(9, Mode::Minor)),
            key_row(1, 10.0, 20.0, Key::new(0, Mode::Major)),
        ];
        db.replace_key_segments(&uid, &rows).unwrap();
        assert_eq!(db.key_segments(&uid).unwrap(), rows);
    }

    #[test]
    fn replacing_with_no_rows_clears_the_segments() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        db.replace_bpm_segments(&uid, &[bpm_row(0, 0.0, 10.0, 128.0)])
            .unwrap();
        db.replace_bpm_segments(&uid, &[]).unwrap();
        assert!(db.bpm_segments(&uid).unwrap().is_empty());
    }

    #[test]
    fn zero_length_segments_round_trip_unchanged() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let rows = vec![bpm_row(0, 5.0, 5.0, 128.0)];
        db.replace_bpm_segments(&uid, &rows).unwrap();
        let got = db.bpm_segments(&uid).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].duration_sec, 0.0);
    }

    #[test]
    fn segments_for_an_unknown_uid_are_empty() {
        let db = Library::open_in_memory().unwrap();
        assert!(db.bpm_segments(&new_uid()).unwrap().is_empty());
        assert!(db.key_segments(&new_uid()).unwrap().is_empty());
    }

    #[test]
    fn segment_operations_reject_a_non_canonical_uid() {
        let db = Library::open_in_memory().unwrap();
        assert!(matches!(
            db.replace_bpm_segments("not-a-uuid", &[]),
            Err(StoreError::Domain(_))
        ));
        assert!(matches!(
            db.key_segments(""),
            Err(StoreError::Domain(_))
        ));
    }

    /// Legacy rows with no tempo are skipped rather than read as 0 BPM.
    #[test]
    fn segment_rows_with_null_values_are_skipped_on_read() {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        db.connection()
            .execute(
                "INSERT INTO track_bpm_segments(track_uid, seq_index, start_sec, end_sec, duration_sec, bpm)
                 VALUES(?1, 0, 0.0, 10.0, 10.0, NULL);",
                [&uid],
            )
            .unwrap();
        db.connection()
            .execute(
                "INSERT INTO track_key_segments(track_uid, seq_index, start_sec, end_sec, duration_sec, key_value)
                 VALUES(?1, 0, 0.0, 10.0, 10.0, NULL);",
                [&uid],
            )
            .unwrap();
        assert!(db.bpm_segments(&uid).unwrap().is_empty());
        assert!(db.key_segments(&uid).unwrap().is_empty());
    }

    // ----- transition search ---------------------------------------------

    fn library_with_bpm_track(bpms: &[f64], durations: &[f64]) -> (Library, String) {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let mut row = track("/music/a.flac", &uid);
        row.title = "A".to_string();
        db.upsert(&row).unwrap();
        let mut start = 0.0;
        let rows: Vec<BpmSegmentRow> = bpms
            .iter()
            .zip(durations)
            .enumerate()
            .map(|(i, (bpm, dur))| {
                let r = bpm_row(i, start, start + dur, *bpm);
                start += dur;
                r
            })
            .collect();
        db.replace_bpm_segments(&uid, &rows).unwrap();
        (db, uid)
    }

    #[test]
    fn bpm_transitions_match_only_adjacent_segments() {
        let (db, _uid) = library_with_bpm_track(&[128.0, 100.0, 140.0], &[30.0, 30.0, 30.0]);
        // 128 -> 140 exists in the track, but not as neighbours.
        assert!(db
            .search_bpm_transitions(128.0, 140.0, 1.0, 4.0, false)
            .unwrap()
            .is_empty());

        let hits = db
            .search_bpm_transitions(128.0, 100.0, 1.0, 4.0, false)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].from.seq_index, 0);
        assert_eq!(hits[0].to.seq_index, 1);
        assert_eq!(hits[0].from.bpm, Some(128.0));
        assert_eq!(hits[0].to.bpm, Some(100.0));
        assert_eq!(hits[0].from.key, None);
        assert_eq!(hits[0].title, "A");
    }

    #[test]
    fn bpm_tolerance_widens_the_match_window() {
        let (db, _uid) = library_with_bpm_track(&[128.0, 130.0], &[30.0, 30.0]);
        assert!(db
            .search_bpm_transitions(128.0, 128.0, 0.0, 4.0, false)
            .unwrap()
            .is_empty());
        // 2% of 128 is 2.56, which reaches 130.
        assert_eq!(
            db.search_bpm_transitions(128.0, 128.0, 2.0, 4.0, false)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn bpm_transitions_drop_segments_shorter_than_the_minimum() {
        let (db, _uid) = library_with_bpm_track(&[128.0, 130.0], &[30.0, 2.0]);
        assert!(db
            .search_bpm_transitions(128.0, 130.0, 1.0, 4.0, false)
            .unwrap()
            .is_empty());
        assert_eq!(
            db.search_bpm_transitions(128.0, 130.0, 1.0, 1.0, false)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn require_first_segment_anchors_the_search_at_the_track_start() {
        let (db, _uid) = library_with_bpm_track(&[100.0, 128.0, 130.0], &[30.0, 30.0, 30.0]);
        assert_eq!(
            db.search_bpm_transitions(128.0, 130.0, 1.0, 4.0, false)
                .unwrap()
                .len(),
            1
        );
        assert!(db
            .search_bpm_transitions(128.0, 130.0, 1.0, 4.0, true)
            .unwrap()
            .is_empty());
        assert_eq!(
            db.search_bpm_transitions(100.0, 128.0, 1.0, 4.0, true)
                .unwrap()
                .len(),
            1
        );
    }

    /// Python builds `BETWEEN nan AND nan`, which returns nothing but is
    /// indistinguishable from an empty library.
    #[test]
    fn an_unusable_bpm_anchor_matches_nothing_without_reaching_sql() {
        let (db, _uid) = library_with_bpm_track(&[128.0, 130.0], &[30.0, 30.0]);
        for bad in [f64::NAN, f64::INFINITY, 0.0, -128.0] {
            assert!(db
                .search_bpm_transitions(bad, 130.0, 1.0, 4.0, false)
                .unwrap()
                .is_empty());
            assert!(db
                .search_bpm_transitions(128.0, bad, 1.0, 4.0, false)
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn a_negative_tolerance_is_treated_as_exact() {
        let (db, _uid) = library_with_bpm_track(&[128.0, 130.0], &[30.0, 30.0]);
        assert!(db
            .search_bpm_transitions(128.0, 128.0, -50.0, 4.0, false)
            .unwrap()
            .is_empty());
    }

    fn library_with_key_track(keys: &[Key]) -> (Library, String) {
        let db = Library::open_in_memory().unwrap();
        let uid = new_uid();
        let mut row = track("/music/a.flac", &uid);
        row.title = "A".to_string();
        db.upsert(&row).unwrap();
        let rows: Vec<KeySegmentRow> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| key_row(i, i as f64 * 30.0, (i as f64 + 1.0) * 30.0, *k))
            .collect();
        db.replace_key_segments(&uid, &rows).unwrap();
        (db, uid)
    }

    /// Unlike the BPM search, a harmonic move counts across a bridge segment.
    #[test]
    fn key_transitions_match_any_later_segment_not_only_the_next() {
        let a_minor = Key::new(9, Mode::Minor); // 8A
        let bridge = Key::new(2, Mode::Major); // 2B, unrelated
        let c_major = Key::new(0, Mode::Major); // 8B, relative of A minor
        let (db, _uid) = library_with_key_track(&[a_minor, bridge, c_major]);

        let hits = db
            .search_harmonic_key_transitions(a_minor, c_major, 4.0, false)
            .unwrap();
        let pairs: Vec<(i64, i64)> = hits.iter().map(|h| (h.from.seq_index, h.to.seq_index)).collect();
        assert!(pairs.contains(&(0, 2)), "0 -> 2 across the bridge: {pairs:?}");
        assert_eq!(hits[0].from.bpm, None);
        assert_eq!(hits[0].from.key, Some(a_minor));
        assert_eq!(hits[0].from.key_label, "8A");
    }

    #[test]
    fn key_transitions_expand_each_anchor_through_its_harmonic_neighbours() {
        let a_minor = Key::new(9, Mode::Minor); // 8A
        let e_minor = Key::new(4, Mode::Minor); // 9A, a neighbour of 8A
        let (db, _uid) = library_with_key_track(&[a_minor, e_minor]);

        // Anchoring on A minor at both ends still finds 8A -> 9A, because 9A is
        // in A minor's neighbour set.
        let hits = db
            .search_harmonic_key_transitions(a_minor, a_minor, 4.0, false)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].to.key, Some(e_minor));

        // A key two steps away on the wheel is not in the set.
        let d_minor = Key::new(2, Mode::Minor); // 7A
        let far = Key::new(11, Mode::Minor);
        assert!(!d_minor.harmonic_neighbours().contains(&far));
    }

    #[test]
    fn key_transitions_respect_min_duration_and_first_segment() {
        let a_minor = Key::new(9, Mode::Minor);
        let (db, uid) = library_with_key_track(&[a_minor, a_minor]);
        assert_eq!(
            db.search_harmonic_key_transitions(a_minor, a_minor, 4.0, true)
                .unwrap()
                .len(),
            1
        );
        // Shrink both segments below the floor.
        let rows = vec![
            key_row(0, 0.0, 1.0, a_minor),
            key_row(1, 1.0, 2.0, a_minor),
        ];
        db.replace_key_segments(&uid, &rows).unwrap();
        assert!(db
            .search_harmonic_key_transitions(a_minor, a_minor, 4.0, false)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn transition_searches_on_an_empty_library_return_nothing() {
        let db = Library::open_in_memory().unwrap();
        assert!(db
            .search_bpm_transitions(128.0, 130.0, 2.0, 4.0, false)
            .unwrap()
            .is_empty());
        assert!(db
            .search_harmonic_key_transitions(Key::from_index(0), Key::from_index(0), 4.0, false)
            .unwrap()
            .is_empty());
    }
}
