//! Tests for the assembled document.
//!
//! The module-level pieces are tested next to the code they cover; these
//! exercise `build_rekordbox_xml` end to end.

use super::*;

use mixlyzer_core::{Key, Mode};

const UID: &str = "3f2504e0-4f89-41d3-9a0c-0305e82c3301";

fn track() -> Track {
    let mut track = Track::new("/music/Artist/Song.flac");
    track.uid = Some(UID.to_string());
    track.title = "Song".to_string();
    track.artist = "Artist".to_string();
    track.album = "Album".to_string();
    track.bpm = Some(128.0);
    track.key = Some(Key::new(9, Mode::Minor)); // Am, Camelot 8A
    track.duration = Some(200.0);
    track.rating = 4;
    track.added_ts = 1_700_000_000; // 2023-11-14 UTC
    track.comment = "notes".to_string();
    track.file_size = 8_000_000;
    track
}

fn segments() -> Vec<TempoSegment> {
    vec![
        TempoSegment::new(0.0, 100.0, 128.0, 0.0, 4),
        TempoSegment::new(100.0, 200.0, 130.0, 100.0, 4),
    ]
}

fn build(
    track: &Track,
    segments: &[TempoSegment],
    jump_cues: &[JumpCue],
    cue_points: &[CuePoint],
) -> RekordboxXml {
    build_rekordbox_xml(track, segments, jump_cues, cue_points, &ExportOptions::default())
        .expect("export succeeds")
}

/// The whole document for a small known track, so any change to layout,
/// attribute order or formatting has to be made on purpose.
#[test]
fn a_known_track_renders_the_expected_document() {
    let jump_cues = vec![JumpCue::new(0, "A", 30.0, 38.0, 32.0, 0)];
    let cue_points = vec![CuePoint::new(0, 64.0, "CHORUS_IN", "chorus 1")];
    let export = build(&track(), &segments(), &jump_cues, &cue_points);

    assert_eq!(export.filename, "Song_rekordbox.xml");
    assert_eq!(
        export.document,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="Mixlyzer" Version="1.0.0" Company="Mixlyzer"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="83933577619577555887446891116005962497" Name="Song" Artist="Artist" Composer="" Album="Album" Grouping="" Genre="" Kind="FLAC File" Size="8000000" TotalTime="200" DiscNumber="1" TrackNumber="1" Year="0" AverageBpm="129.00" DateAdded="2023-11-14" BitRate="320" SampleRate="44100" Comments="notes" PlayCount="0" Rating="4" Location="file://localhost/music/Artist/Song.flac" Remixer="" Tonality="8A" Label="" Mix="">
      <TEMPO Inizio="0.000" Bpm="128.00" Metro="4/4" Battito="1"/>
      <TEMPO Inizio="100.000" Bpm="130.00" Metro="4/4" Battito="1"/>
      <POSITION_MARK Name="A" Type="0" Start="32.000" Num="0" Red="34" Green="139" Blue="34"/>
      <POSITION_MARK Name="CHORUS_IN" Type="0" Start="64.000" Num="-1" Red="40" Green="226" Blue="20"/>
    </TRACK>
  </COLLECTION>
</DJ_PLAYLISTS>
"#
    );
}

#[test]
fn the_track_id_is_the_uid_as_a_decimal_integer() {
    let export = build(&track(), &[], &[], &[]);
    let expected = u128::from_str_radix(&UID.replace('-', ""), 16).unwrap().to_string();
    assert!(export.document.contains(&format!("TrackID=\"{expected}\"")));
}

/// Python raises a bare `ValueError` here and takes the whole sync with it.
#[test]
fn a_track_without_a_uid_returns_a_typed_error() {
    let mut track = track();
    track.uid = None;
    let error = build_rekordbox_xml(&track, &[], &[], &[], &ExportOptions::default()).unwrap_err();
    assert!(matches!(error, ExportError::MissingUid { .. }));
    assert!(error.to_string().contains("Song.flac"));
}

#[test]
fn a_track_with_a_malformed_uid_returns_a_typed_error() {
    let mut track = track();
    track.uid = Some("not-a-uuid".to_string());
    let error = build_rekordbox_xml(&track, &[], &[], &[], &ExportOptions::default()).unwrap_err();
    assert!(matches!(error, ExportError::InvalidUid { .. }));
}

/// Python prefers the stored `bpm` column, so a re-analysed grid exports a
/// tempo the grid itself contradicts.
#[test]
fn average_bpm_comes_from_the_beatgrid_not_the_stored_column() {
    let mut track = track();
    track.bpm = Some(100.0); // stale
    let export = build(&track, &segments(), &[], &[]);
    assert!(export.document.contains(r#"AverageBpm="129.00""#));
}

#[test]
fn average_bpm_is_weighted_by_segment_length() {
    let mut track = track();
    track.bpm = None;
    let segments = [
        TempoSegment::new(0.0, 180.0, 120.0, 0.0, 4),
        TempoSegment::new(180.0, 200.0, 140.0, 180.0, 4),
    ];
    // (120*180 + 140*20) / 200 = 122.0
    assert!((average_bpm(&track, &segments) - 122.0).abs() < 1e-9);
}

#[test]
fn average_bpm_falls_back_to_the_stored_column_without_a_grid() {
    assert_eq!(average_bpm(&track(), &[]), 128.0);
    let mut track = track();
    track.bpm = None;
    assert_eq!(average_bpm(&track, &[]), 0.0);
    track.bpm = Some(f64::NAN);
    assert_eq!(average_bpm(&track, &[]), 0.0);
}

#[test]
fn a_track_with_no_segments_still_gets_a_flat_grid_from_its_stored_tempo() {
    let export = build(&track(), &[], &[], &[]);
    assert!(export.document.contains(r#"<TEMPO Inizio="0.000" Bpm="128.00" Metro="4/4" Battito="1"/>"#));
}

#[test]
fn a_track_with_neither_segments_nor_tempo_exports_without_a_grid() {
    let mut track = track();
    track.bpm = None;
    let export = build(&track, &[], &[], &[]);
    assert!(!export.document.contains("<TEMPO"));
    assert!(export.document.contains(r#"AverageBpm="0.00""#));
}

#[test]
fn a_track_with_no_cues_has_no_position_marks() {
    let export = build(&track(), &segments(), &[], &[]);
    assert!(!export.document.contains("POSITION_MARK"));
}

#[test]
fn a_zero_duration_track_reports_no_time_and_no_bit_rate() {
    let mut track = track();
    track.duration = Some(0.0);
    let export = build(&track, &[], &[], &[]);
    assert!(export.document.contains(r#"TotalTime="0""#));
    assert!(export.document.contains(r#"BitRate="0""#));
}

#[test]
fn duration_falls_back_to_the_end_of_the_grid() {
    let mut track = track();
    track.duration = None;
    assert_eq!(duration(&track, &segments()), 200.0);
    assert_eq!(duration(&track, &[]), 0.0);
    track.duration = Some(f64::NAN);
    assert_eq!(duration(&track, &segments()), 200.0);
}

/// A control character in a title makes the Python export raise `ExpatError`
/// from `minidom` and abort the library sync.
#[test]
fn hostile_metadata_produces_a_parseable_document() {
    let mut track = track();
    track.title = "Bad\u{0}\u{1} \"Title\" & <tag> 🎧 中文".to_string();
    track.artist = "A\u{7}rtist".to_string();
    track.comment = "line1\nline2\u{c}".to_string();
    let export = build(&track, &segments(), &[], &[]);

    assert!(!export.document.chars().any(|c| !xml::is_xml_char(c)));
    assert!(export
        .document
        .contains(r#"Name="Bad &quot;Title&quot; &amp; &lt;tag&gt; 🎧 中文""#));
    assert!(export.document.contains(r#"Artist="Artist""#));
    assert!(export.document.contains(r#"Comments="line1&#10;line2""#));
    // The file name loses the control characters and the quotes stay put.
    assert_eq!(export.filename, "Bad _Title_ & _tag_ 🎧 中文_rekordbox.xml");
}

#[test]
fn more_than_eight_jump_cues_export_without_overwriting_a_hot_cue() {
    let jump_cues: Vec<JumpCue> = (0..12)
        .map(|i| {
            let label = char::from(b'A' + i as u8).to_string();
            JumpCue::new(i, label, 10.0 * i as f64, 10.0 * i as f64 + 5.0, 10.0 * i as f64, i % 3)
        })
        .collect();
    let export = build(&track(), &segments(), &jump_cues, &[]);

    let nums: Vec<&str> = export
        .document
        .lines()
        .filter(|line| line.contains("POSITION_MARK"))
        .map(|line| {
            let start = line.find("Num=\"").unwrap() + 5;
            let rest = &line[start..];
            &rest[..rest.find('"').unwrap()]
        })
        .collect();
    assert_eq!(nums.len(), 12);
    let mut hot: Vec<&str> = nums.iter().copied().filter(|n| *n != "-1").collect();
    let hot_count = hot.len();
    hot.sort_unstable();
    hot.dedup();
    assert_eq!(hot.len(), hot_count, "a hot cue was overwritten: {nums:?}");
    assert_eq!(hot_count, marks::HOT_CUE_SLOTS);
}

#[test]
fn a_non_four_four_grid_writes_its_own_metro() {
    let segments = [TempoSegment::new(0.0, 100.0, 150.0, 0.0, 3)];
    let export = build(&track(), &segments, &[], &[]);
    assert!(export.document.contains(r#"Metro="3/4""#));
}

#[test]
fn a_track_without_a_key_exports_an_empty_tonality() {
    let mut track = track();
    track.key = None;
    assert!(build(&track, &[], &[], &[]).document.contains(r#"Tonality="""#));
}

#[test]
fn the_location_is_a_percent_encoded_file_url() {
    assert_eq!(
        file_url("/music/Bad Bunny/Tití Me Preguntó.flac"),
        "file://localhost/music/Bad%20Bunny/Tit%C3%AD%20Me%20Pregunt%C3%B3.flac"
    );
    // Windows drive letters keep their colon, and a relative path still gets a
    // root, as in Python.
    assert_eq!(file_url("C:/Music/a.mp3"), "file://localhost/C:/Music/a.mp3");
    assert_eq!(file_url("music/a.mp3"), "file://localhost/music/a.mp3");
    assert_eq!(file_url(""), "");
    // `&` in a path must not leak into the attribute unescaped.
    let mut track = track();
    track.path = "/music/AC&DC/x.mp3".to_string();
    assert!(build(&track, &[], &[], &[])
        .document
        .contains(r#"Location="file://localhost/music/AC%26DC/x.mp3""#));
}

#[test]
fn a_resolved_audio_path_overrides_the_stored_one() {
    let options = ExportOptions {
        audio_path: Some(PathBuf::from("/mnt/usb/Song.aiff")),
        file_size: Some(1_000),
        ..ExportOptions::default()
    };
    let export = build_rekordbox_xml(&track(), &[], &[], &[], &options).unwrap();
    assert!(export.document.contains(r#"Location="file://localhost/mnt/usb/Song.aiff""#));
    assert!(export.document.contains(r#"Kind="AIFF File""#));
    assert!(export.document.contains(r#"Size="1000""#));
}

#[test]
fn a_track_without_a_title_is_named_after_its_file() {
    let mut track = track();
    track.title = "   ".to_string();
    let export = build(&track, &[], &[], &[]);
    assert!(export.document.contains(r#"Name="Song""#));
    assert_eq!(export.filename, "Song_rekordbox.xml");

    track.path = String::new();
    let export = build(&track, &[], &[], &[]);
    assert!(export.document.contains(r#"Name="Untitled""#));
    assert!(export.document.contains(r#"Kind="""#));
    assert!(export.document.contains(r#"Location="""#));
}

#[test]
fn file_names_lose_the_characters_a_filesystem_refuses() {
    assert_eq!(sanitize_filename("AC/DC: Back?"), "AC_DC_ Back_");
    assert_eq!(sanitize_filename("a///b"), "a_b");
    assert_eq!(sanitize_filename("  spaced  "), "spaced");
    assert_eq!(sanitize_filename("bad\u{0}name"), "badname");
    assert_eq!(sanitize_filename("///"), "_");
    assert_eq!(sanitize_filename(""), "track");
    assert_eq!(sanitize_filename("🎧 中文"), "🎧 中文");
}

#[test]
fn dates_are_formatted_in_utc_and_omitted_when_unset() {
    assert_eq!(format_date(0), "");
    assert_eq!(format_date(1), "1970-01-01");
    assert_eq!(format_date(1_700_000_000), "2023-11-14");
    assert_eq!(format_date(951_782_400), "2000-02-29"); // leap day
    assert_eq!(format_date(-1), "1969-12-31");
}

#[test]
fn the_bit_rate_is_derived_from_size_and_duration() {
    // 8 MB over 200 s is 320 kbit/s.
    assert!(build(&track(), &[], &[], &[]).document.contains(r#"BitRate="320""#));
    let mut track = track();
    track.file_size = 0;
    assert!(build(&track, &[], &[], &[]).document.contains(r#"BitRate="0""#));
}

#[test]
fn the_document_has_exactly_one_track_and_says_so() {
    let export = build(&track(), &segments(), &[], &[]);
    assert_eq!(export.document.matches("<TRACK ").count(), 1);
    assert!(export.document.contains(r#"<COLLECTION Entries="1">"#));
    assert!(export.document.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
    assert!(export.document.ends_with("</DJ_PLAYLISTS>\n"));
}
