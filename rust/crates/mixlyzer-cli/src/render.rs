//! Turning analysis results into something readable in a terminal.

use std::path::Path;

use mixlyzer_core::Track;
use mixlyzer_dsp::Analysis;
use mixlyzer_store::Transition;
use serde_json::json;

/// A track's title, falling back to its path when there is none.
pub fn display_title(track: &Track) -> String {
    if !track.title.is_empty() {
        if track.artist.is_empty() {
            track.title.clone()
        } else {
            format!("{} - {}", track.artist, track.title)
        }
    } else {
        track.path.clone()
    }
}

/// `m:ss`, the form a DJ reads track lengths in.
pub fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--:--".to_string();
    }
    let total = seconds.round() as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// A one-screen summary of an analysis.
pub fn analysis_text(path: &Path, analysis: &Analysis) -> String {
    let mut out = String::new();
    let key = analysis
        .overall_key
        .map(|k| k.display())
        .unwrap_or_else(|| "unknown".to_string());

    out.push_str(&format!("file       {}\n", path.display()));
    out.push_str(&format!(
        "duration   {} ({:.2}s)\n",
        format_duration(analysis.duration_sec),
        analysis.duration_sec
    ));
    out.push_str(&format!("tempo      {:.2} BPM\n", analysis.tempo_global));
    out.push_str(&format!("key        {key}\n"));
    out.push_str(&format!(
        "beats      {} ({} bars)\n",
        analysis.beats().len(),
        analysis.downbeats().len()
    ));
    out.push_str(&format!(
        "confidence {:.0}%\n",
        analysis.beat_confidence * 100.0
    ));

    let segments = analysis.tempo_segments();
    out.push_str(&format!("\ntempo segments ({})\n", segments.len()));
    for (index, segment) in segments.iter().enumerate() {
        out.push_str(&format!(
            "  {index:>3}  {:>8} - {:<8}  {:>7.2} BPM  {}/4  bar starts {:.3}s\n",
            format_duration(segment.start),
            format_duration(segment.end),
            segment.bpm,
            segment.time_signature,
            segment.inizio
        ));
    }

    let cues = analysis.jump_cues.cues();
    if !cues.is_empty() {
        out.push_str(&format!("\njump cues ({})\n", cues.len()));
        for cue in cues {
            out.push_str(&format!(
                "  {}  {:>8} - {:<8}  jump at {:>8}  pair {}\n",
                cue.label,
                format_duration(cue.start),
                format_duration(cue.end),
                format_duration(cue.point),
                cue.component + 1
            ));
        }
    }

    if !analysis.phrases.is_empty() {
        out.push_str(&format!("\nphrases ({})\n", analysis.phrases.len()));
        for phrase in &analysis.phrases {
            out.push_str(&format!(
                "  {:>8} - {:<8}  {}\n",
                format_duration(phrase.start),
                format_duration(phrase.end),
                phrase.label
            ));
        }
    }

    if !analysis.cue_points.is_empty() {
        out.push_str(&format!("\ncue points ({})\n", analysis.cue_points.len()));
        for cue in &analysis.cue_points {
            out.push_str(&format!(
                "  {:>2}  {:>8}  {}\n",
                cue.id,
                format_duration(cue.time_sec),
                cue.label
            ));
        }
    }

    if !analysis.key_segments.is_empty() {
        out.push_str(&format!("\nkey segments ({})\n", analysis.key_segments.len()));
        // A long track can hold dozens; the first few show the shape.
        for segment in analysis.key_segments.iter().take(12) {
            out.push_str(&format!(
                "  {:>8} - {:<8}  {}\n",
                format_duration(segment.start),
                format_duration(segment.end),
                segment.key.display()
            ));
        }
        if analysis.key_segments.len() > 12 {
            out.push_str(&format!(
                "  ... {} more\n",
                analysis.key_segments.len() - 12
            ));
        }
    }
    out
}

/// The same analysis as machine-readable JSON.
pub fn analysis_json(path: &Path, analysis: &Analysis) -> String {
    let document = json!({
        "file": path.display().to_string(),
        "duration_sec": analysis.duration_sec,
        "analysis_sample_rate": analysis.analysis_sample_rate,
        "tempo_global": analysis.tempo_global,
        "beat_confidence": analysis.beat_confidence,
        "key": analysis.overall_key.map(|k| json!({
            "index": k.index(),
            "camelot": k.camelot(),
            "classical": k.classical(),
        })),
        "beats": analysis.beats(),
        "downbeats": analysis.downbeats(),
        "tempo_segments": analysis.tempo_segments().iter().map(|s| json!({
            "start": s.start,
            "end": s.end,
            "bpm": s.bpm,
            "inizio": s.inizio,
            "time_signature": s.time_signature,
        })).collect::<Vec<_>>(),
        "jump_cues": analysis.jump_cues.cues().iter().map(|c| json!({
            "label": c.label,
            "start": c.start,
            "end": c.end,
            "point": c.point,
            "component": c.component,
        })).collect::<Vec<_>>(),
        "phrases": analysis.phrases.iter().map(|p| json!({
            "start": p.start,
            "end": p.end,
            "label": p.label,
        })).collect::<Vec<_>>(),
        "cue_points": analysis.cue_points.iter().map(|c| json!({
            "id": c.id,
            "time_sec": c.time_sec,
            "label": c.label,
            "comment": c.comment,
        })).collect::<Vec<_>>(),
        "key_segments": analysis.key_segments.iter().map(|s| json!({
            "start": s.start,
            "end": s.end,
            "key": s.key.index(),
            "camelot": s.key.camelot(),
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string_pretty(&document).unwrap_or_else(|_| "{}".to_string())
}

/// A fixed-width listing of library tracks.
pub fn track_table(tracks: &[Track]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<40}  {:>7}  {:>5}  {:>8}\n",
        "TITLE", "BPM", "KEY", "LENGTH"
    ));
    out.push_str(&format!("{}\n", "-".repeat(65)));
    for track in tracks {
        let bpm = track
            .bpm
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "--".to_string());
        let key = track.key.map(|k| k.camelot().to_string()).unwrap_or_default();
        out.push_str(&format!(
            "{:<40}  {:>7}  {:>5}  {:>8}\n",
            truncate(&display_title(track), 40),
            bpm,
            key,
            format_duration(track.duration.unwrap_or(0.0))
        ));
    }
    out.push_str(&format!("\n{} track(s)\n", tracks.len()));
    out
}

/// A listing of tempo transitions found in the library.
pub fn transition_table(transitions: &[Transition]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<40}  {:>16}  {:>16}\n",
        "TITLE", "FROM", "TO"
    ));
    out.push_str(&format!("{}\n", "-".repeat(78)));
    for item in transitions {
        let title = if item.title.is_empty() {
            item.path.clone()
        } else {
            item.title.clone()
        };
        out.push_str(&format!(
            "{:<40}  {:>7} @ {:>5}  {:>7} @ {:>5}\n",
            truncate(&title, 40),
            side_label(&item.from),
            format_duration(item.from.start_sec),
            side_label(&item.to),
            format_duration(item.to.start_sec),
        ));
    }
    out.push_str(&format!("\n{} transition(s)\n", transitions.len()));
    out
}

/// What a transition side is matched on: a tempo for BPM searches, a key for
/// harmonic ones. Exactly one of the two is set.
fn side_label(side: &mixlyzer_store::TransitionSide) -> String {
    if let Some(bpm) = side.bpm {
        format!("{bpm:.2}")
    } else if let Some(key) = side.key {
        key.camelot().to_string()
    } else {
        "--".to_string()
    }
}

/// Cut a string to `width`, marking the cut so nothing looks complete when it is not.
fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let kept: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mixlyzer_core::Key;

    fn track(title: &str, artist: &str) -> Track {
        let mut track = Track::new("/music/song.flac");
        track.title = title.into();
        track.artist = artist.into();
        track
    }

    #[test]
    fn a_title_with_an_artist_reads_as_artist_dash_title() {
        assert_eq!(display_title(&track("Song", "Artist")), "Artist - Song");
    }

    #[test]
    fn a_title_without_an_artist_stands_alone() {
        assert_eq!(display_title(&track("Song", "")), "Song");
    }

    #[test]
    fn a_track_with_no_title_falls_back_to_its_path() {
        assert_eq!(display_title(&track("", "")), "/music/song.flac");
    }

    #[test]
    fn durations_read_as_minutes_and_seconds() {
        assert_eq!(format_duration(0.0), "0:00");
        assert_eq!(format_duration(9.4), "0:09");
        assert_eq!(format_duration(61.0), "1:01");
        assert_eq!(format_duration(3599.0), "59:59");
        assert_eq!(format_duration(3600.0), "60:00");
    }

    #[test]
    fn an_unknown_duration_is_marked_rather_than_shown_as_zero() {
        assert_eq!(format_duration(f64::NAN), "--:--");
        assert_eq!(format_duration(-1.0), "--:--");
    }

    #[test]
    fn truncation_marks_where_it_cut() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("exactlyten", 10), "exactlyten");
        assert_eq!(truncate("this is far too long", 10), "this is f…");
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        // Multi-byte titles must not be cut mid-character.
        let title = "한국어 제목이 아주 깁니다";
        let cut = truncate(title, 5);
        assert_eq!(cut.chars().count(), 5);
    }

    #[test]
    fn the_track_table_has_a_row_per_track_and_a_total() {
        let mut first = track("One", "A");
        first.bpm = Some(128.0);
        first.key = Some(Key::from_index(0));
        first.duration = Some(200.0);
        let second = track("", "");
        let rendered = track_table(&[first, second]);
        assert!(rendered.contains("A - One"));
        assert!(rendered.contains("128.00"));
        assert!(rendered.contains("8B"));
        assert!(rendered.contains("3:20"));
        assert!(rendered.contains("--"), "a missing BPM should be marked");
        assert!(rendered.contains("2 track(s)"));
    }

    #[test]
    fn an_empty_table_still_has_headings() {
        let rendered = track_table(&[]);
        assert!(rendered.contains("TITLE"));
        assert!(rendered.contains("0 track(s)"));
    }
}
