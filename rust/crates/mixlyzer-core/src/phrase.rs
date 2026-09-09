//! Song-structure phrases: intro, verse, chorus and friends.
//!
//! Phrases come out of the two-stage gradient-boosting detector and can be
//! edited by hand afterwards. A phrase carries a *base* label; display labels
//! (numbering, abbreviation) are derived on demand and never stored.

use crate::error::DomainError;

/// Phrases shorter than this are treated as zero-length and dropped.
pub const MIN_PHRASE_DURATION: f64 = 1e-6;

/// The labels the detector can emit, plus the two fill markers.
pub const PHRASE_LABELS: [&str; 10] = [
    "INTRO",
    "VERSE",
    "CHORUS",
    "BREAK_CHORUS",
    "BRIDGE",
    "OUTRO",
    "INTERLUDE",
    "SILENCE",
    "FILL_IN",
    "FILL_OUT",
];

/// Colours for the known labels, as RGB.
const PHRASE_COLORS: [(&str, (u8, u8, u8)); 11] = [
    ("INTRO", (70, 130, 180)),
    ("VERSE", (46, 139, 87)),
    ("CHORUS", (220, 60, 90)),
    ("BREAK_CHORUS", (235, 140, 120)),
    ("BRIDGE", (148, 0, 211)),
    ("OUTRO", (95, 95, 110)),
    ("INTERLUDE", (218, 165, 32)),
    ("SILENCE", (110, 110, 110)),
    ("FILL_IN", (0, 180, 200)),
    ("FILL_OUT", (0, 120, 150)),
    // Emitted by older models and by pre-0.3.0 libraries.
    ("FILL", (0, 153, 204)),
];

/// Short forms used on the overview strip, where there is no room for a word.
///
/// These are an explicit table rather than "take the first letter", which is
/// what the Python code does. First letters collide twice over: INTRO and
/// INTERLUDE both become "I", and BRIDGE and BREAK_CHORUS both become "B", so
/// four of the ten labels are ambiguous on the strip.
const PHRASE_ABBREVIATIONS: [(&str, &str); 11] = [
    ("INTRO", "I"),
    ("VERSE", "V"),
    ("CHORUS", "C"),
    ("BREAK_CHORUS", "BC"),
    ("BRIDGE", "B"),
    ("OUTRO", "O"),
    ("INTERLUDE", "IL"),
    ("SILENCE", "S"),
    ("FILL_IN", "FI"),
    ("FILL_OUT", "FO"),
    ("FILL", "F"),
];

const FALLBACK_PALETTE: [(u8, u8, u8); 6] = [
    (200, 80, 40),
    (40, 160, 160),
    (160, 100, 200),
    (120, 160, 40),
    (200, 120, 160),
    (80, 120, 200),
];

/// One labelled stretch of the track.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Phrase {
    pub start: f64,
    pub end: f64,
    /// The base label, exactly as detected or typed. Never carries a number.
    pub label: String,
}

impl Phrase {
    pub fn new(start: f64, end: f64, label: impl Into<String>) -> Self {
        Self {
            start,
            end,
            label: label.into().trim().to_string(),
        }
    }

    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    /// Whether this phrase is one of the fill markers, which are drawn as part
    /// of a neighbour rather than as a section of their own.
    pub fn is_fill(&self) -> bool {
        matches!(
            self.label.to_ascii_uppercase().as_str(),
            "FILL_IN" | "FILL_OUT" | "FILL"
        )
    }

    /// The colour to draw this phrase in.
    pub fn color(&self) -> (u8, u8, u8) {
        phrase_color(&self.label)
    }
}

/// Colour for a base label, falling back to a palette entry for custom labels.
///
/// The fallback is chosen with a fixed FNV-1a hash. Python indexes the palette
/// with the builtin `hash()`, which is randomised per process, so a custom
/// label gets a different colour every time the app restarts.
pub fn phrase_color(label: &str) -> (u8, u8, u8) {
    let key = label.trim().to_ascii_uppercase();
    if key.is_empty() {
        return (110, 110, 110);
    }
    if let Some((_, rgb)) = PHRASE_COLORS.iter().find(|(name, _)| *name == key) {
        return *rgb;
    }
    let idx = (fnv1a(&key) % FALLBACK_PALETTE.len() as u64) as usize;
    FALLBACK_PALETTE[idx]
}

/// 64-bit FNV-1a. Stable across processes and releases, unlike `hash()`.
fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Sort phrases by start time and drop degenerate ones.
pub fn normalize(phrases: &[Phrase]) -> Vec<Phrase> {
    let mut rows: Vec<Phrase> = phrases
        .iter()
        .filter(|p| {
            p.start.is_finite() && p.end.is_finite() && p.end - p.start > MIN_PHRASE_DURATION
        })
        .cloned()
        .collect();
    rows.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

/// Display labels: a label used more than once is numbered from 1 in time order.
///
/// A label that occurs exactly once is left alone, so a track with one bridge
/// shows "BRIDGE" rather than "BRIDGE1".
pub fn numbered_labels(phrases: &[Phrase]) -> Vec<String> {
    let rows = normalize(phrases);
    let mut totals: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for row in &rows {
        *totals.entry(row.label.as_str()).or_insert(0) += 1;
    }
    let mut running: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    rows.iter()
        .map(|row| {
            let base = row.label.as_str();
            if totals.get(base).copied().unwrap_or(0) >= 2 {
                let n = running.entry(base).or_insert(0);
                *n += 1;
                format!("{base}{n}")
            } else {
                base.to_string()
            }
        })
        .collect()
}

/// Short display labels for the overview strip, e.g. `CHORUS2` becomes `C2`.
///
/// Custom labels keep their full numbered form.
pub fn abbreviated_labels(phrases: &[Phrase]) -> Vec<String> {
    let rows = normalize(phrases);
    let numbered = numbered_labels(&rows);
    rows.iter()
        .zip(numbered)
        .map(|(row, display)| {
            let key = row.label.trim().to_ascii_uppercase();
            match PHRASE_ABBREVIATIONS.iter().find(|(name, _)| *name == key) {
                Some((_, abbrev)) => {
                    let suffix = &display[row.label.len()..];
                    format!("{abbrev}{suffix}")
                }
                None => display,
            }
        })
        .collect()
}

/// Overwrite `[start, end)` with `label`, splitting whatever was there.
///
/// Adjacent phrases sharing a label are deliberately left unmerged so that
/// repeated sections keep their individual numbers.
pub fn assign_to_selection(
    phrases: &[Phrase],
    start: f64,
    end: f64,
    label: &str,
) -> Result<Vec<Phrase>, DomainError> {
    if !matches!(start.partial_cmp(&end), Some(std::cmp::Ordering::Less)) {
        return Err(DomainError::EmptySelection {
            start: start.to_string(),
            end: end.to_string(),
        });
    }
    let base = label.trim();
    if base.is_empty() {
        return Err(DomainError::EmptyPhraseLabel);
    }

    let new_seg = Phrase::new(start, end, base);
    let mut result: Vec<Phrase> = Vec::new();
    let mut inserted = false;

    for seg in normalize(phrases) {
        if seg.end <= start {
            result.push(seg);
            continue;
        }
        if seg.start >= end {
            if !inserted {
                result.push(new_seg.clone());
                inserted = true;
            }
            result.push(seg);
            continue;
        }
        // Overlaps the selection: keep whatever sticks out on either side.
        if seg.start < start {
            let left = Phrase::new(seg.start, start, seg.label.clone());
            if left.duration() > 1e-9 {
                result.push(left);
            }
        }
        if !inserted {
            result.push(new_seg.clone());
            inserted = true;
        }
        if seg.end > end {
            let right = Phrase::new(end, seg.end, seg.label.clone());
            if right.duration() > 1e-9 {
                result.push(right);
            }
        }
    }
    if !inserted {
        result.push(new_seg);
    }
    result.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// Remove all phrase coverage inside `[start, end)`, splitting overlaps.
pub fn clear_selection(phrases: &[Phrase], start: f64, end: f64) -> Vec<Phrase> {
    if !matches!(start.partial_cmp(&end), Some(std::cmp::Ordering::Less)) {
        return normalize(phrases);
    }
    let mut result: Vec<Phrase> = Vec::new();
    for seg in normalize(phrases) {
        if seg.end <= start || seg.start >= end {
            result.push(seg);
            continue;
        }
        if seg.start < start {
            let left = Phrase::new(seg.start, start, seg.label.clone());
            if left.duration() > 1e-9 {
                result.push(left);
            }
        }
        if seg.end > end {
            let right = Phrase::new(end, seg.end, seg.label.clone());
            if right.duration() > 1e-9 {
                result.push(right);
            }
        }
    }
    result
}

/// Display-only view with fills absorbed into the section they belong to.
///
/// `FILL_IN` is shown as part of the phrase that follows it, `FILL_OUT` as
/// part of the one before. A fill with no non-fill neighbour is dropped.
pub fn merge_fills_for_display(phrases: &[Phrase]) -> Vec<Phrase> {
    let rows = normalize(phrases);
    if rows.is_empty() {
        return Vec::new();
    }

    // Resolve each fill to a neighbour's label, remembering which entries came
    // from a fill so only those get glued to their neighbour below.
    let mut display: Vec<(Phrase, bool)> = Vec::new();
    for (index, seg) in rows.iter().enumerate() {
        let upper = seg.label.trim().to_ascii_uppercase();
        let (label, from_fill) = match upper.as_str() {
            "FILL_IN" => (neighbour_label(&rows, index, 1), true),
            "FILL_OUT" | "FILL" => (neighbour_label(&rows, index, -1), true),
            _ => (seg.label.clone(), false),
        };
        if label.is_empty() {
            continue;
        }
        display.push((Phrase::new(seg.start, seg.end, label), from_fill));
    }

    let mut merged: Vec<(Phrase, bool)> = Vec::new();
    for (seg, from_fill) in display {
        if let Some((last, last_from_fill)) = merged.last_mut() {
            let touching = (last.end - seg.start).abs() <= 1e-6;
            if last.label == seg.label && touching && (*last_from_fill || from_fill) {
                last.end = seg.end;
                *last_from_fill = *last_from_fill || from_fill;
                continue;
            }
        }
        merged.push((seg, from_fill));
    }
    merged.into_iter().map(|(seg, _)| seg).collect()
}

/// Which way a fill marker points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillDirection {
    In,
    Out,
}

/// A fill marker to draw over the phrase strip.
#[derive(Debug, Clone, PartialEq)]
pub struct FillMarker {
    pub start: f64,
    pub end: f64,
    pub direction: FillDirection,
}

/// Extract the fill markers, for drawing arrows on the phrase strip.
pub fn fill_markers(phrases: &[Phrase]) -> Vec<FillMarker> {
    normalize(phrases)
        .into_iter()
        .filter_map(|seg| {
            let direction = match seg.label.trim().to_ascii_uppercase().as_str() {
                "FILL_IN" => FillDirection::In,
                "FILL_OUT" | "FILL" => FillDirection::Out,
                _ => return None,
            };
            Some(FillMarker {
                start: seg.start,
                end: seg.end,
                direction,
            })
        })
        .collect()
}

/// Walk outwards from `index` in `step` direction for the first non-fill label.
fn neighbour_label(rows: &[Phrase], index: usize, step: isize) -> String {
    let mut pos = index as isize + step;
    while pos >= 0 && (pos as usize) < rows.len() {
        let candidate = &rows[pos as usize];
        if !candidate.is_fill() {
            return candidate.label.clone();
        }
        pos += step;
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(start: f64, end: f64, label: &str) -> Phrase {
        Phrase::new(start, end, label)
    }

    #[test]
    fn normalize_sorts_and_drops_degenerate_phrases() {
        let rows = normalize(&[
            p(10.0, 20.0, "CHORUS"),
            p(0.0, 0.0, "VERSE"),
            p(0.0, 10.0, "INTRO"),
            p(30.0, 20.0, "OUTRO"),
        ]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "INTRO");
        assert_eq!(rows[1].label, "CHORUS");
    }

    #[test]
    fn repeated_labels_are_numbered_and_singletons_are_not() {
        let rows = vec![
            p(0.0, 10.0, "INTRO"),
            p(10.0, 20.0, "VERSE"),
            p(20.0, 30.0, "CHORUS"),
            p(30.0, 40.0, "VERSE"),
        ];
        assert_eq!(
            numbered_labels(&rows),
            vec!["INTRO", "VERSE1", "CHORUS", "VERSE2"]
        );
    }

    /// Python abbreviates by first letter, so INTRO/INTERLUDE and
    /// BRIDGE/BREAK_CHORUS are indistinguishable on the overview strip.
    #[test]
    fn abbreviations_do_not_collide() {
        let rows = vec![
            p(0.0, 10.0, "INTRO"),
            p(10.0, 20.0, "INTERLUDE"),
            p(20.0, 30.0, "BRIDGE"),
            p(30.0, 40.0, "BREAK_CHORUS"),
        ];
        let abbrevs = abbreviated_labels(&rows);
        assert_eq!(abbrevs, vec!["I", "IL", "B", "BC"]);
        let mut unique = abbrevs.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), abbrevs.len(), "abbreviations must stay distinct");
    }

    #[test]
    fn abbreviations_keep_the_number_suffix() {
        let rows = vec![p(0.0, 10.0, "CHORUS"), p(10.0, 20.0, "CHORUS")];
        assert_eq!(abbreviated_labels(&rows), vec!["C1", "C2"]);
    }

    #[test]
    fn custom_labels_keep_their_full_form() {
        let rows = vec![p(0.0, 10.0, "DROP"), p(10.0, 20.0, "DROP")];
        assert_eq!(abbreviated_labels(&rows), vec!["DROP1", "DROP2"]);
    }

    /// Python picks the fallback colour with the randomised builtin `hash()`,
    /// so a custom label changes colour on every restart.
    #[test]
    fn custom_label_colors_are_stable_across_runs() {
        let first = phrase_color("MY SECTION");
        for _ in 0..100 {
            assert_eq!(phrase_color("MY SECTION"), first);
        }
        assert!(FALLBACK_PALETTE.contains(&first));
    }

    #[test]
    fn known_labels_have_fixed_colors_and_are_case_insensitive() {
        assert_eq!(phrase_color("CHORUS"), (220, 60, 90));
        assert_eq!(phrase_color("chorus"), (220, 60, 90));
        assert_eq!(phrase_color("  Chorus  "), (220, 60, 90));
        assert_eq!(phrase_color(""), (110, 110, 110));
    }

    #[test]
    fn assign_splits_an_overlapping_phrase_on_both_sides() {
        let rows = vec![p(0.0, 100.0, "VERSE")];
        let out = assign_to_selection(&rows, 40.0, 60.0, "CHORUS").unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!((out[0].start, out[0].end, out[0].label.as_str()), (0.0, 40.0, "VERSE"));
        assert_eq!((out[1].start, out[1].end, out[1].label.as_str()), (40.0, 60.0, "CHORUS"));
        assert_eq!((out[2].start, out[2].end, out[2].label.as_str()), (60.0, 100.0, "VERSE"));
    }

    #[test]
    fn assign_into_a_gap_inserts_in_time_order() {
        let rows = vec![p(0.0, 10.0, "INTRO"), p(50.0, 60.0, "OUTRO")];
        let out = assign_to_selection(&rows, 20.0, 30.0, "VERSE").unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].label, "VERSE");
        assert!(out.windows(2).all(|w| w[0].start <= w[1].start));
    }

    #[test]
    fn assign_rejects_an_empty_selection_or_label() {
        let rows = vec![p(0.0, 10.0, "INTRO")];
        assert!(matches!(
            assign_to_selection(&rows, 10.0, 10.0, "VERSE"),
            Err(DomainError::EmptySelection { .. })
        ));
        assert!(matches!(
            assign_to_selection(&rows, 0.0, 5.0, "   "),
            Err(DomainError::EmptyPhraseLabel)
        ));
    }

    #[test]
    fn adjacent_same_label_phrases_stay_separate_so_numbering_survives() {
        let rows = vec![p(0.0, 10.0, "CHORUS")];
        let out = assign_to_selection(&rows, 10.0, 20.0, "CHORUS").unwrap();
        assert_eq!(out.len(), 2, "merging here would collapse CHORUS1/CHORUS2");
        assert_eq!(numbered_labels(&out), vec!["CHORUS1", "CHORUS2"]);
    }

    #[test]
    fn clear_punches_a_hole_and_keeps_both_sides() {
        let rows = vec![p(0.0, 100.0, "VERSE")];
        let out = clear_selection(&rows, 40.0, 60.0);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].start, out[0].end), (0.0, 40.0));
        assert_eq!((out[1].start, out[1].end), (60.0, 100.0));
    }

    #[test]
    fn clear_with_an_empty_range_changes_nothing() {
        let rows = vec![p(0.0, 10.0, "INTRO")];
        assert_eq!(clear_selection(&rows, 5.0, 5.0), normalize(&rows));
    }

    #[test]
    fn fill_in_joins_the_following_phrase() {
        let rows = vec![
            p(0.0, 10.0, "VERSE"),
            p(10.0, 12.0, "FILL_IN"),
            p(12.0, 30.0, "CHORUS"),
        ];
        let display = merge_fills_for_display(&rows);
        assert_eq!(display.len(), 2);
        assert_eq!(display[0].label, "VERSE");
        assert_eq!(display[1].label, "CHORUS");
        assert_eq!(display[1].start, 10.0, "the fill is drawn as part of the chorus");
        assert_eq!(display[1].end, 30.0);
    }

    #[test]
    fn fill_out_joins_the_preceding_phrase() {
        let rows = vec![
            p(0.0, 10.0, "CHORUS"),
            p(10.0, 12.0, "FILL_OUT"),
            p(12.0, 30.0, "VERSE"),
        ];
        let display = merge_fills_for_display(&rows);
        assert_eq!(display.len(), 2);
        assert_eq!((display[0].label.as_str(), display[0].end), ("CHORUS", 12.0));
        assert_eq!(display[1].start, 12.0);
    }

    #[test]
    fn an_orphan_fill_is_dropped() {
        let rows = vec![p(0.0, 2.0, "FILL_IN")];
        assert!(merge_fills_for_display(&rows).is_empty());
    }

    #[test]
    fn legacy_fill_label_behaves_like_fill_out() {
        let rows = vec![p(0.0, 10.0, "CHORUS"), p(10.0, 12.0, "FILL")];
        let display = merge_fills_for_display(&rows);
        assert_eq!(display.len(), 1);
        assert_eq!(display[0].end, 12.0);
    }

    #[test]
    fn fill_markers_report_direction() {
        let rows = vec![
            p(0.0, 10.0, "VERSE"),
            p(10.0, 12.0, "FILL_IN"),
            p(20.0, 22.0, "FILL_OUT"),
            p(30.0, 32.0, "FILL"),
        ];
        let markers = fill_markers(&rows);
        assert_eq!(markers.len(), 3);
        assert_eq!(markers[0].direction, FillDirection::In);
        assert_eq!(markers[1].direction, FillDirection::Out);
        assert_eq!(markers[2].direction, FillDirection::Out);
    }

    #[test]
    fn every_known_label_has_a_color_and_an_abbreviation() {
        for label in PHRASE_LABELS {
            assert!(
                PHRASE_COLORS.iter().any(|(name, _)| *name == label),
                "{label} has no colour"
            );
            assert!(
                PHRASE_ABBREVIATIONS.iter().any(|(name, _)| *name == label),
                "{label} has no abbreviation"
            );
        }
    }
}
