//! Point markers over the track: cue points, JumpCUE regions, and the playhead.

use egui::{Align2, FontId, Painter, Rect, Stroke};
use mixlyzer_core::{CuePoint, JumpCue};

use crate::theme::Theme;
use crate::viewport::Viewport;

/// Half-width of the triangle that marks a cue, in pixels.
const CUE_MARKER_HALF_WIDTH: f32 = 6.0;

/// Draw a downward triangle at each structural cue point.
pub fn draw_cue_points(painter: &Painter, view: &Viewport, cues: &[CuePoint], theme: &Theme) {
    let rect = view.rect();
    let (from, to) = view.padded_span(0.05);
    for cue in cues {
        if cue.time_sec < from || cue.time_sec > to {
            continue;
        }
        let x = view.x_of(cue.time_sec);
        let top = rect.top();
        painter.add(egui::Shape::convex_polygon(
            vec![
                egui::pos2(x - CUE_MARKER_HALF_WIDTH, top),
                egui::pos2(x + CUE_MARKER_HALF_WIDTH, top),
                egui::pos2(x, top + CUE_MARKER_HALF_WIDTH * 2.0),
            ],
            theme.cue,
            Stroke::NONE,
        ));
    }
}

/// Draw each JumpCUE's region as a translucent block with its letter.
///
/// The colour comes from the cue's component, so the two ends of a jump are
/// visibly the same pair.
pub fn draw_jump_cues(painter: &Painter, view: &Viewport, cues: &[JumpCue], theme: &Theme) {
    let rect = view.rect();
    for cue in cues {
        if !view.intersects(cue.start, cue.end) {
            continue;
        }
        let left = view.x_of(cue.start).max(rect.left());
        let right = view.x_of(cue.end).min(rect.right());
        if right <= left {
            continue;
        }
        let (r, g, b) = cue.color();
        let block = Rect::from_min_max(
            egui::pos2(left, rect.top()),
            egui::pos2(right, rect.bottom()),
        );
        painter.rect_filled(
            block,
            0.0,
            egui::Color32::from_rgba_unmultiplied(r, g, b, 110),
        );
        // The jump lands on the point, not on the start of the region.
        if view.intersects(cue.point, cue.point) {
            let x = view.x_of(cue.point);
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                Stroke::new(2.0, egui::Color32::from_rgb(r, g, b)),
            );
        }
        if block.width() > 16.0 {
            painter.text(
                egui::pos2(block.left() + 3.0, block.top() + 1.0),
                Align2::LEFT_TOP,
                &cue.label,
                FontId::monospace(11.0),
                theme.label,
            );
        }
    }
}

/// How far down the view the bar.beat readout sits, as a fraction of height.
///
/// Below the cue markers, which occupy the top of the view, so the two do not
/// overlap on a narrow window.
const PLAYHEAD_LABEL_TOP_FRACTION: f32 = 0.2;

/// Draw the playhead, and the bar.beat readout beside it when one is available.
pub fn draw_playhead(painter: &Painter, view: &Viewport, theme: &Theme, bar_beat: Option<&str>) {
    let rect = view.rect();
    let x = view.playhead_x();
    painter.line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        Stroke::new(theme.playhead_width, theme.playhead),
    );
    let Some(label) = bar_beat else {
        return;
    };
    let anchor = egui::pos2(x + 5.0, rect.top() + rect.height() * PLAYHEAD_LABEL_TOP_FRACTION);
    let galley = painter.layout_no_wrap(
        label.to_string(),
        FontId::monospace(12.0),
        theme.playhead,
    );
    // The readout sits over the waveform, so give it a backdrop; white text on
    // a bright transient is otherwise unreadable exactly when it matters.
    painter.rect_filled(
        Rect::from_min_size(anchor, galley.size()).expand(2.0),
        2.0,
        egui::Color32::from_black_alpha(180),
    );
    painter.galley(anchor, galley, theme.playhead);
}

/// Shade a selected time range.
pub fn draw_selection(painter: &Painter, view: &Viewport, range: Option<(f64, f64)>, theme: &Theme) {
    let Some((start, end)) = range else {
        return;
    };
    if end <= start || !view.intersects(start, end) {
        return;
    }
    let rect = view.rect();
    let left = view.x_of(start).max(rect.left());
    let right = view.x_of(end).min(rect.right());
    if right <= left {
        return;
    }
    painter.rect_filled(
        Rect::from_min_max(
            egui::pos2(left, rect.top()),
            egui::pos2(right, rect.bottom()),
        ),
        0.0,
        theme.selection,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::paint;
    use egui::vec2;

    fn viewport(rect: Rect, time: f64, window: f64) -> Viewport {
        Viewport::new(rect, time, window, 300.0)
    }

    fn cue(time: f64, label: &str) -> CuePoint {
        CuePoint::new(0, time, label, "")
    }

    #[test]
    fn a_visible_cue_becomes_a_triangle() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_cue_points(
                painter,
                &viewport(rect, 6.0, 12.0),
                &[cue(6.0, "CHORUS_IN")],
                &Theme::dark(),
            );
        });
        assert_eq!(painted.len(), 1);
    }

    #[test]
    fn cues_outside_the_window_are_skipped() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_cue_points(
                painter,
                &viewport(rect, 6.0, 12.0),
                &[cue(200.0, "OUTRO")],
                &Theme::dark(),
            );
        });
        assert!(painted.is_empty());
    }

    #[test]
    fn no_cues_paint_nothing() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_cue_points(painter, &viewport(rect, 6.0, 12.0), &[], &Theme::dark());
        });
        assert!(painted.is_empty());
    }

    #[test]
    fn a_jump_cue_draws_its_region_its_point_and_its_letter() {
        let cue = JumpCue::new(0, "A", 4.0, 10.0, 6.0, 0);
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_jump_cues(painter, &viewport(rect, 6.0, 12.0), &[cue], &Theme::dark());
        });
        assert_eq!(painted.rects().len(), 1, "the region block");
        assert_eq!(painted.vertical_line_xs().len(), 1, "the jump point");
        assert_eq!(painted.texts(), vec!["A".to_string()]);
    }

    #[test]
    fn the_two_ends_of_a_pair_share_a_colour() {
        let first = JumpCue::new(0, "A", 0.0, 4.0, 1.0, 0);
        let second = JumpCue::new(1, "B", 8.0, 12.0, 9.0, 0);
        assert_eq!(first.color(), second.color());
    }

    #[test]
    fn a_jump_region_running_off_screen_is_clipped() {
        let cue = JumpCue::new(0, "A", 0.0, 300.0, 150.0, 0);
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_jump_cues(painter, &viewport(rect, 6.0, 12.0), &[cue], &Theme::dark());
        });
        let block = painted.rects()[0].rect;
        assert_eq!(block.left(), 0.0);
        assert_eq!(block.right(), 400.0);
    }

    #[test]
    fn a_narrow_jump_region_drops_its_letter() {
        let cue = JumpCue::new(0, "A", 6.0, 6.05, 6.0, 0);
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_jump_cues(painter, &viewport(rect, 6.0, 12.0), &[cue], &Theme::dark());
        });
        assert!(painted.texts().is_empty());
    }

    #[test]
    fn the_playhead_is_a_line_at_its_fixed_position() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_playhead(painter, &viewport(rect, 6.0, 12.0), &Theme::dark(), None);
        });
        assert_eq!(painted.vertical_line_xs(), vec![200.0]);
        assert!(painted.texts().is_empty());
    }

    #[test]
    fn the_playhead_shows_the_bar_and_beat_when_given_one() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_playhead(
                painter,
                &viewport(rect, 6.0, 12.0),
                &Theme::dark(),
                Some("17.3"),
            );
        });
        assert_eq!(painted.texts(), vec!["17.3".to_string()]);
    }

    #[test]
    fn a_selection_shades_only_its_own_range() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw_selection(
                painter,
                &viewport(rect, 6.0, 12.0),
                Some((4.0, 8.0)),
                &Theme::dark(),
            );
        });
        let block = painted.rects()[0].rect;
        // 4s and 8s sit a third and two thirds across a window spanning 0..12.
        assert!((block.left() - 133.3).abs() < 2.0, "left at {}", block.left());
        assert!((block.right() - 266.7).abs() < 2.0, "right at {}", block.right());
    }

    #[test]
    fn an_absent_or_empty_selection_shades_nothing() {
        for range in [None, Some((5.0, 5.0)), Some((8.0, 4.0))] {
            let painted = paint(vec2(400.0, 100.0), |painter, rect| {
                draw_selection(painter, &viewport(rect, 6.0, 12.0), range, &Theme::dark());
            });
            assert!(painted.is_empty(), "range {range:?} should draw nothing");
        }
    }
}
