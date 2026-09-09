//! Rekordbox XML export.
//!
//! Builds the `DJ_PLAYLISTS` document Rekordbox imports: one `COLLECTION` with
//! a single `TRACK`, carrying the beatgrid as [`TEMPO`](tempo) children and the
//! cues as [`POSITION_MARK`](marks) children.
//!
//! This is a port of the Python `third_party/rekordbox.py`. It is pure: it
//! reads nothing from disk and returns the document as text, so the caller
//! decides where it goes. Where the behaviour deliberately departs from the
//! Python original, the item's documentation says so and a test named after the
//! new behaviour pins it. The departures are:
//!
//! * Metadata is sanitised for XML instead of being handed straight to a parser
//!   that rejects it (see [`xml`]).
//! * Hot cues are allocated rather than assigned modulo eight, and structural
//!   cue points are exported at all (see [`marks`]).
//! * `AverageBpm` comes from the beatgrid rather than the stored column (see
//!   [`average_bpm`]).
//! * A missing or malformed uid returns [`ExportError`] instead of raising from
//!   deep inside XML generation.
//! * `DateAdded` is formatted in UTC (see [`format_date`]).

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod marks;
pub mod tempo;
pub mod xml;

use std::path::{Path, PathBuf};

use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};

use mixlyzer_core::{CuePoint, JumpCue, TempoSegment, Track};

use crate::xml::Element;

/// Schema version written to `DJ_PLAYLISTS` and to `PRODUCT` by default.
pub const DJ_PLAYLISTS_VERSION: &str = "1.0.0";

/// Sample rate written when the caller does not override it.
pub const DEFAULT_SAMPLE_RATE: u32 = 44_100;

/// Weight given to a zero-length segment when averaging tempo, so that a
/// degenerate segment still contributes rather than dividing by zero.
const MIN_SEGMENT_WEIGHT: f64 = 1e-3;

/// Everything that can stop an export.
///
/// Python raises a bare `ValueError` from inside XML generation for a missing
/// uid, which aborts a whole library sync over one bad row. These are typed so
/// a caller can skip the track and carry on.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The track has no uid, so there is nothing to derive a `TrackID` from.
    #[error("track {path:?} has no uid, so it has no Rekordbox TrackID")]
    MissingUid { path: String },

    /// The uid is not a UUID, so it has no integer form.
    #[error("track uid {uid:?} is not a UUID: {source}")]
    InvalidUid {
        uid: String,
        #[source]
        source: uuid::Error,
    },
}

/// Knobs the caller can turn; [`Default`] matches the Python export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOptions {
    /// Written to `PRODUCT/@Name`.
    pub product_name: String,
    /// Written to `PRODUCT/@Version`.
    pub product_version: String,
    /// Written to `TRACK/@SampleRate`; the export never inspects the audio.
    pub sample_rate: u32,
    /// Absolute path to the audio file, when the caller has resolved one.
    ///
    /// Used for `Location`, `Kind` and the title fallback in place of the
    /// track's stored (possibly relative) path.
    pub audio_path: Option<PathBuf>,
    /// File size in bytes, when the caller has stat'ed the file. Falls back to
    /// the size stored on the track.
    pub file_size: Option<u64>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            product_name: "Mixlyzer".to_string(),
            product_version: DJ_PLAYLISTS_VERSION.to_string(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            audio_path: None,
            file_size: None,
        }
    }
}

/// A finished export: the document text and the name to save it under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RekordboxXml {
    /// The complete `DJ_PLAYLISTS` document, including the XML declaration.
    pub document: String,
    /// Suggested file name, `{sanitised title}_rekordbox.xml`.
    pub filename: String,
}

/// Build the Rekordbox XML document for one track.
///
/// `tempo_segments` is the beatgrid, `jump_cues` the JumpCUEs competing for the
/// eight hot-cue buttons, and `cue_points` the structural cues, which are
/// exported as memory cues.
pub fn build_rekordbox_xml(
    track: &Track,
    tempo_segments: &[TempoSegment],
    jump_cues: &[JumpCue],
    cue_points: &[CuePoint],
    options: &ExportOptions,
) -> Result<RekordboxXml, ExportError> {
    let track_id = track_id(track)?;

    let location_path = location_path(track, options);
    let title = song_title(track, &location_path);
    let duration = duration(track, tempo_segments);
    let average = average_bpm(track, tempo_segments);
    let file_size = options.file_size.unwrap_or(track.file_size.max(0) as u64);

    let mut track_elem = Element::new("TRACK")
        .attr("TrackID", track_id)
        .attr("Name", title.clone())
        .attr("Artist", track.artist.clone())
        .attr("Composer", "")
        .attr("Album", track.album.clone())
        .attr("Grouping", "")
        .attr("Genre", "")
        .attr("Kind", kind_from_path(&location_path))
        .attr("Size", file_size.to_string())
        .attr("TotalTime", total_time(duration).to_string())
        .attr("DiscNumber", "1")
        .attr("TrackNumber", "1")
        .attr("Year", "0")
        .attr("AverageBpm", format!("{:.2}", average.max(0.0)))
        .attr("DateAdded", format_date(track.added_ts))
        .attr("BitRate", bit_rate(file_size, duration).to_string())
        .attr("SampleRate", options.sample_rate.to_string())
        .attr("Comments", track.comment.clone())
        .attr("PlayCount", "0")
        .attr("Rating", track.rating.to_string())
        .attr("Location", file_url(&location_path))
        .attr("Remixer", "")
        .attr("Tonality", track.key_label())
        .attr("Label", "")
        .attr("Mix", "");

    for entry in tempo::tempo_entries(tempo_segments, average) {
        track_elem.push(entry.to_element());
    }
    for mark in marks::position_marks(jump_cues, cue_points) {
        track_elem.push(mark.to_element());
    }

    let root = Element::new("DJ_PLAYLISTS")
        .attr("Version", DJ_PLAYLISTS_VERSION)
        .child(
            Element::new("PRODUCT")
                .attr("Name", options.product_name.clone())
                .attr("Version", options.product_version.clone())
                .attr("Company", options.product_name.clone()),
        )
        .child(Element::new("COLLECTION").attr("Entries", "1").child(track_elem));

    Ok(RekordboxXml {
        document: root.to_document(),
        filename: format!("{}_rekordbox.xml", sanitize_filename(&title)),
    })
}

/// Make `name` safe to use as a file name.
///
/// Runs of the characters Windows forbids collapse into a single `_`, as in the
/// Python `sanitize_filename`. Control characters are removed as well: they are
/// legal in a POSIX file name but make the resulting file awkward to handle
/// everywhere else, and tag data really does contain them.
pub fn sanitize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_run = false;
    for c in name.chars().filter(|c| !c.is_control()) {
        if matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
            if !in_run {
                out.push('_');
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    let trimmed = out.trim();
    if trimmed.is_empty() {
        "track".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The track's tempo, as a duration-weighted mean of the beatgrid segments.
///
/// Python returns the stored `bpm` column whenever it is set, so a track whose
/// beatgrid has since been edited exports the stale value and Rekordbox shows a
/// tempo the grid contradicts. The segments are the beatgrid, so they win here;
/// the stored column is only the fallback when there is no grid.
pub fn average_bpm(track: &Track, segments: &[TempoSegment]) -> f64 {
    let mut weighted = 0.0;
    let mut weight = 0.0;
    for segment in segments
        .iter()
        .filter(|s| s.bpm.is_finite() && s.bpm > 0.0 && s.start.is_finite() && s.end.is_finite())
    {
        let w = segment.duration().max(MIN_SEGMENT_WEIGHT);
        weighted += segment.bpm * w;
        weight += w;
    }
    if weight > 0.0 {
        return weighted / weight;
    }
    track
        .bpm
        .filter(|bpm| bpm.is_finite() && *bpm > 0.0)
        .unwrap_or(0.0)
}

/// Playing time in seconds: the track's own duration, else the end of the last
/// beatgrid segment, else zero.
pub fn duration(track: &Track, segments: &[TempoSegment]) -> f64 {
    if let Some(duration) = track.duration.filter(|d| d.is_finite() && *d > 0.0) {
        return duration;
    }
    segments
        .iter()
        .map(|s| s.end)
        .filter(|end| end.is_finite() && *end > 0.0)
        .fold(0.0, f64::max)
}

/// `TrackID`: the uid's 128-bit integer value in decimal, as Rekordbox wants a
/// numeric id and Python writes `str(UUID(uid).int)`.
pub fn track_id(track: &Track) -> Result<String, ExportError> {
    let uid = track.require_uid().map_err(|_| ExportError::MissingUid {
        path: track.path.clone(),
    })?;
    let parsed = uuid::Uuid::parse_str(uid).map_err(|source| ExportError::InvalidUid {
        uid: uid.to_string(),
        source,
    })?;
    Ok(parsed.as_u128().to_string())
}

/// Format an epoch timestamp as `YYYY-MM-DD` in UTC, or `""` for no timestamp.
///
/// Python uses `datetime.fromtimestamp`, i.e. the exporting machine's local
/// zone, so the same library exports different dates on different machines.
/// UTC keeps the document reproducible.
pub fn format_date(added_ts: i64) -> String {
    if added_ts == 0 {
        return String::new();
    }
    let (year, month, day) = civil_from_days(added_ts.div_euclid(86_400));
    format!("{year:04}-{month:02}-{day:02}")
}

/// A `file://localhost/...` URL for `path`, percent-encoded the way Rekordbox
/// expects. An empty path yields an empty string.
pub fn file_url(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let encoded = utf8_percent_encode(path, PATH_ENCODE_SET).to_string();
    if encoded.starts_with('/') {
        format!("file://localhost{encoded}")
    } else {
        format!("file://localhost/{encoded}")
    }
}

/// Characters left unencoded in a `Location`, matching Python's
/// `quote(path, safe="/:.")`: the URL unreserved set plus `/` and `:`.
const PATH_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~')
    .remove(b'/')
    .remove(b':');

/// `Kind`, e.g. `"FLAC File"`. Empty when the path has no extension.
fn kind_from_path(path: &str) -> String {
    match Path::new(path).extension().and_then(|ext| ext.to_str()) {
        Some(ext) if !ext.is_empty() => format!("{} File", ext.to_uppercase()),
        _ => String::new(),
    }
}

/// The path used for `Location`, `Kind` and the title fallback: the resolved
/// audio path when the caller supplied one, else the track's stored path.
fn location_path(track: &Track, options: &ExportOptions) -> String {
    match options.audio_path.as_deref() {
        Some(path) => path.to_string_lossy().replace('\\', "/"),
        None => track.path.clone(),
    }
}

/// `Name`: the stored title, else the file stem, else `"Untitled"`.
fn song_title(track: &Track, location_path: &str) -> String {
    let title = track.title.trim();
    if !title.is_empty() {
        return title.to_string();
    }
    Path::new(location_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("Untitled")
        .to_string()
}

/// `TotalTime`: whole seconds, rounded.
fn total_time(duration: f64) -> i64 {
    if duration.is_finite() && duration > 0.0 {
        duration.round() as i64
    } else {
        0
    }
}

/// `BitRate` in kbit/s, inferred from size and duration; 0 when either is
/// unknown, which is how Rekordbox reads "not stated".
fn bit_rate(file_size: u64, duration: f64) -> i64 {
    if file_size == 0 || !duration.is_finite() || duration <= 0.0 {
        return 0;
    }
    ((file_size as f64 * 8.0) / duration / 1000.0).round() as i64
}

/// Civil date from a count of days since the Unix epoch (Howard Hinnant's
/// `civil_from_days`), valid for any day the proleptic Gregorian calendar
/// covers.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests;
