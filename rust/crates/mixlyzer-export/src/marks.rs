//! `POSITION_MARK` elements: hot cues and memory cues.
//!
//! Rekordbox stores both kinds in the same element. `Num` distinguishes them:
//! `0..=7` are the eight hot-cue buttons A-H, and `-1` is a memory cue, an
//! unlimited list of markers the player can cycle through.
//!
//! Two things differ from the Python export, both deliberate:
//!
//! * Python maps a cue whose label is a letter A-H onto that button and falls
//!   back to `idx % 8` for everything else, so the ninth cue on a track lands on
//!   button A again and silently overwrites the cue already there. Here each cue
//!   takes the button its label names when that button is still free, the
//!   remaining cues fill the buttons still unused, and anything left over is
//!   written as a memory cue instead of colliding.
//! * Python drops structural [`CuePoint`]s entirely, so the phrase analysis
//!   never reaches Rekordbox. They are exported here as memory cues.

use mixlyzer_core::{CuePoint, JumpCue};

use crate::xml::Element;

/// Number of hot-cue buttons a Rekordbox player has.
pub const HOT_CUE_SLOTS: usize = 8;

/// The button letters, in `Num` order.
pub const HOT_CUE_LETTERS: [char; HOT_CUE_SLOTS] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

/// The `Num` value Rekordbox uses for a memory cue.
pub const MEMORY_CUE_NUM: i32 = -1;

/// Colour for marks that carry none of their own, matching Python's default.
pub const DEFAULT_MARK_COLOR: (u8, u8, u8) = (40, 226, 20);

/// One `POSITION_MARK`.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionMark {
    /// Label shown on the player.
    pub name: String,
    /// Position in seconds from the start of the track.
    pub start: f64,
    /// Hot-cue button index `0..8`, or [`MEMORY_CUE_NUM`] for a memory cue.
    pub num: i32,
    /// RGB triple drawn on the waveform.
    pub color: (u8, u8, u8),
}

impl PositionMark {
    /// Whether this mark occupies a hot-cue button.
    pub fn is_hot_cue(&self) -> bool {
        self.num >= 0
    }

    /// Render as a `POSITION_MARK` element.
    ///
    /// `Type="0"` is Rekordbox's plain cue; loops and fades use other values,
    /// which Mixlyzer never produces.
    pub fn to_element(&self) -> Element {
        Element::new("POSITION_MARK")
            .attr("Name", self.name.clone())
            .attr("Type", "0")
            .attr("Start", format!("{:.3}", self.start))
            .attr("Num", self.num.to_string())
            .attr("Red", self.color.0.to_string())
            .attr("Green", self.color.1.to_string())
            .attr("Blue", self.color.2.to_string())
    }
}

/// The hot-cue button a label asks for, if it names one.
///
/// A label is usually just `"A"`, but `"Drop A"` and `"A (intro)"` name a
/// button too. Mirrors the Python `_slot_index`: a standalone letter A-H
/// anywhere in the label, else a letter A-H the label starts with.
pub fn hot_cue_slot(label: &str) -> Option<usize> {
    let upper = label.trim().to_uppercase();
    let chars: Vec<char> = upper.chars().collect();
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    for (i, c) in chars.iter().enumerate() {
        if !HOT_CUE_LETTERS.contains(c) {
            continue;
        }
        let before_is_word = i > 0 && is_word(chars[i - 1]);
        let after_is_word = chars.get(i + 1).is_some_and(|next| is_word(*next));
        if !before_is_word && !after_is_word {
            return Some(*c as usize - 'A' as usize);
        }
    }
    // Fall back to a leading letter, so `"Ahead"` still means button A.
    chars
        .first()
        .and_then(|c| HOT_CUE_LETTERS.iter().position(|letter| letter == c))
}

/// Build every `POSITION_MARK` for a track, in time order.
///
/// JumpCUEs compete for the eight hot-cue buttons (see the module docs for the
/// allocation rule); structural cue points are always memory cues. Marks with a
/// non-finite or negative time are dropped — Rekordbox refuses the whole file
/// over one bad `Start`.
pub fn position_marks(jump_cues: &[JumpCue], cue_points: &[CuePoint]) -> Vec<PositionMark> {
    let mut cues: Vec<&JumpCue> = jump_cues
        .iter()
        .filter(|cue| cue.point.is_finite() && cue.point >= 0.0)
        .collect();
    cues.sort_by(|a, b| a.point.total_cmp(&b.point).then(a.id.cmp(&b.id)));

    let mut slot_of: Vec<Option<usize>> = vec![None; cues.len()];
    let mut taken: [bool; HOT_CUE_SLOTS] = [false; HOT_CUE_SLOTS];

    // First pass: honour the button each label names, first cue to ask wins.
    for (index, cue) in cues.iter().enumerate() {
        if let Some(slot) = hot_cue_slot(&cue.label) {
            if !taken[slot] {
                taken[slot] = true;
                slot_of[index] = Some(slot);
            }
        }
    }
    // Second pass: the cues that asked for nothing, or for a taken button, get
    // whatever is still free. Anything past the eighth becomes a memory cue.
    let mut next_free = 0;
    for slot in slot_of.iter_mut() {
        if slot.is_some() {
            continue;
        }
        while next_free < HOT_CUE_SLOTS && taken[next_free] {
            next_free += 1;
        }
        if next_free < HOT_CUE_SLOTS {
            taken[next_free] = true;
            *slot = Some(next_free);
        }
    }

    let mut marks: Vec<PositionMark> = cues
        .iter()
        .enumerate()
        .map(|(index, cue)| {
            let label = cue.label.trim();
            let (num, name) = match slot_of[index] {
                Some(slot) => (
                    slot as i32,
                    if label.is_empty() {
                        format!("Jump {}", HOT_CUE_LETTERS[slot])
                    } else {
                        label.to_string()
                    },
                ),
                None => (
                    MEMORY_CUE_NUM,
                    if label.is_empty() {
                        format!("Jump {}", index + 1)
                    } else {
                        label.to_string()
                    },
                ),
            };
            PositionMark {
                name,
                start: cue.point,
                num,
                color: cue.color(),
            }
        })
        .collect();

    marks.extend(
        cue_points
            .iter()
            .filter(|point| point.time_sec.is_finite() && point.time_sec >= 0.0)
            .map(|point| PositionMark {
                name: if point.label.trim().is_empty() {
                    "Memory".to_string()
                } else {
                    point.label.trim().to_string()
                },
                start: point.time_sec,
                num: MEMORY_CUE_NUM,
                color: DEFAULT_MARK_COLOR,
            }),
    );

    // Hot cues before memory cues at the same instant, so a player that shows
    // only one marker there shows the one a DJ can jump to.
    marks.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(b.is_hot_cue().cmp(&a.is_hot_cue()))
            .then(a.num.cmp(&b.num))
    });
    marks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(id: usize, label: &str, point: f64) -> JumpCue {
        JumpCue::new(id, label, point, point + 8.0, point, id)
    }

    fn cues(labels: &[&str]) -> Vec<JumpCue> {
        labels
            .iter()
            .enumerate()
            .map(|(i, label)| cue(i, label, 10.0 * (i as f64 + 1.0)))
            .collect()
    }

    fn hot_nums(marks: &[PositionMark]) -> Vec<i32> {
        marks.iter().filter(|m| m.is_hot_cue()).map(|m| m.num).collect()
    }

    fn assert_hot_cues_are_unique(marks: &[PositionMark]) {
        let mut nums = hot_nums(marks);
        let count = nums.len();
        nums.sort_unstable();
        nums.dedup();
        assert_eq!(nums.len(), count, "two hot cues share a Num: {marks:#?}");
    }

    #[test]
    fn a_label_naming_a_button_gets_that_button() {
        assert_eq!(hot_cue_slot("A"), Some(0));
        assert_eq!(hot_cue_slot("h"), Some(7));
        assert_eq!(hot_cue_slot("Drop C"), Some(2));
        assert_eq!(hot_cue_slot("D (breakdown)"), Some(3));
        assert_eq!(hot_cue_slot("Ahead"), Some(0));
        assert_eq!(hot_cue_slot("I"), None);
        assert_eq!(hot_cue_slot("intro"), None);
        assert_eq!(hot_cue_slot(""), None);
    }

    #[test]
    fn three_cues_take_the_buttons_their_labels_name() {
        let marks = position_marks(&cues(&["A", "B", "C"]), &[]);
        assert_eq!(marks.len(), 3);
        assert_eq!(hot_nums(&marks), vec![0, 1, 2]);
        assert_hot_cues_are_unique(&marks);
    }

    #[test]
    fn eight_cues_fill_every_button_exactly_once() {
        let marks = position_marks(&cues(&["A", "B", "C", "D", "E", "F", "G", "H"]), &[]);
        assert_eq!(hot_nums(&marks), (0..8).collect::<Vec<i32>>());
        assert_hot_cues_are_unique(&marks);
    }

    /// The Python bug: cue nine falls back to `idx % 8` and takes button A back
    /// off cue one. Here the surplus becomes memory cues.
    #[test]
    fn more_than_eight_cues_never_share_a_hot_cue_button() {
        let labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"];
        let marks = position_marks(&cues(&labels), &[]);
        assert_eq!(marks.len(), 12);
        assert_hot_cues_are_unique(&marks);
        assert_eq!(hot_nums(&marks).len(), HOT_CUE_SLOTS);
        assert_eq!(marks.iter().filter(|m| !m.is_hot_cue()).count(), 4);
        // The first eight, in time order, keep the buttons.
        for mark in marks.iter().take(8) {
            assert!(mark.is_hot_cue(), "{} lost its button", mark.name);
        }
    }

    #[test]
    fn cues_that_name_no_button_fill_the_ones_left_over() {
        let marks = position_marks(&cues(&["C", "intro", "A", "outro"]), &[]);
        let by_name = |name: &str| marks.iter().find(|m| m.name == name).unwrap().num;
        assert_eq!(by_name("C"), 2);
        assert_eq!(by_name("A"), 0);
        assert_eq!(by_name("intro"), 1);
        assert_eq!(by_name("outro"), 3);
        assert_hot_cues_are_unique(&marks);
    }

    #[test]
    fn two_cues_wanting_the_same_button_do_not_collide() {
        let marks = position_marks(&cues(&["A", "A", "A"]), &[]);
        assert_eq!(hot_nums(&marks), vec![0, 1, 2]);
        assert_hot_cues_are_unique(&marks);
    }

    #[test]
    fn an_unlabelled_cue_is_named_after_the_button_it_lands_on() {
        let marks = position_marks(&cues(&["", "B"]), &[]);
        assert_eq!(marks[0].name, "Jump A");
        assert_eq!(marks[0].num, 0);
        assert_eq!(marks[1].name, "B");
    }

    /// Python never writes these, so the phrase analysis stops at the app.
    #[test]
    fn structural_cue_points_are_exported_as_memory_cues() {
        let points = vec![
            CuePoint::new(0, 32.0, "CHORUS_IN", "chorus 1"),
            CuePoint::new(1, 96.0, "OUTRO", ""),
        ];
        let marks = position_marks(&cues(&["A"]), &points);
        assert_eq!(marks.len(), 3);
        let memory: Vec<&PositionMark> = marks.iter().filter(|m| !m.is_hot_cue()).collect();
        assert_eq!(memory.len(), 2);
        assert_eq!(memory[0].name, "CHORUS_IN");
        assert_eq!(memory[0].num, MEMORY_CUE_NUM);
        assert_eq!(memory[1].name, "OUTRO");
    }

    #[test]
    fn marks_come_out_in_time_order_with_hot_cues_first_on_a_tie() {
        let jumps = vec![cue(0, "A", 30.0), cue(1, "B", 10.0)];
        let points = vec![CuePoint::new(0, 30.0, "OUTRO", "")];
        let marks = position_marks(&jumps, &points);
        let order: Vec<(&str, i32)> = marks.iter().map(|m| (m.name.as_str(), m.num)).collect();
        assert_eq!(order, vec![("B", 1), ("A", 0), ("OUTRO", -1)]);
    }

    #[test]
    fn marks_with_impossible_times_are_dropped() {
        let jumps = vec![cue(0, "A", f64::NAN), cue(1, "B", -5.0), cue(2, "C", 4.0)];
        let points = vec![
            CuePoint::new(0, f64::INFINITY, "BAD", ""),
            CuePoint::new(1, 8.0, "OK", ""),
        ];
        let marks = position_marks(&jumps, &points);
        assert_eq!(marks.len(), 2);
        assert_eq!(marks[0].name, "C");
        assert_eq!(marks[1].name, "OK");
    }

    #[test]
    fn no_cues_at_all_yields_no_marks() {
        assert!(position_marks(&[], &[]).is_empty());
    }

    #[test]
    fn the_element_carries_the_component_colour_and_a_millisecond_start() {
        let jumps = vec![cue(0, "A", 12.3456)];
        let element = position_marks(&jumps, &[])[0].to_element();
        assert_eq!(element.name(), "POSITION_MARK");
        assert_eq!(element.attribute("Start"), Some("12.346"));
        assert_eq!(element.attribute("Type"), Some("0"));
        assert_eq!(element.attribute("Num"), Some("0"));
        let (r, g, b) = jumps[0].color();
        assert_eq!(element.attribute("Red"), Some(r.to_string().as_str()));
        assert_eq!(element.attribute("Green"), Some(g.to_string().as_str()));
        assert_eq!(element.attribute("Blue"), Some(b.to_string().as_str()));
    }
}
