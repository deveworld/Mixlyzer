//! Beat lines, bar starts and the bar.beat readout.
//!
//! The grid is the thing a DJ actually aligns to, so it is drawn from the same
//! [`Beatgrid`] the analysis produced: bar lines land on real beats because
//! `downbeat_indices` chooses them, not because a separate walk in seconds
//! happened to arrive at the same place.

use egui::{Align2, FontId, Painter, Stroke};
use mixlyzer_core::Beatgrid;

use crate::theme::Theme;
use crate::viewport::Viewport;

/// How much of the window to draw beyond each edge, so lines do not pop in.
const OVERDRAW_FRACTION: f64 = 0.05;

/// Draw the beat grid over `view`'s rectangle.
///
/// Beats are drawn as thin lines and bar starts as thick ones. Bar starts also
/// carry a `bar.beat` label when there is room for it.
pub fn draw(painter: &Painter, view: &Viewport, grid: &Beatgrid, theme: &Theme, labels: bool) {
    if grid.is_empty() {
        return;
    }
    let rect = view.rect();
    let (from, to) = view.padded_span(OVERDRAW_FRACTION);
    let downbeats = grid.downbeat_indices();

    for (index, beat) in grid.beats().iter().enumerate() {
        if *beat < from {
            continue;
        }
        if *beat > to {
            break;
        }
        let x = view.x_of(*beat);
        // `downbeats` is sorted, so a binary search beats scanning it per beat
        // on a track with thousands of bars.
        let is_downbeat = downbeats.binary_search(&index).is_ok();
        let (color, width) = if is_downbeat {
            (theme.downbeat, theme.downbeat_line_width)
        } else {
            (theme.beat, theme.grid_line_width)
        };
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            Stroke::new(width, color),
        );
    }

    if !labels {
        return;
    }
    // One label per bar would be unreadable when zoomed out; drop them once the
    // bars are closer together than the text is wide.
    let bar_spacing_px = bar_spacing_pixels(view, grid, &downbeats);
    if bar_spacing_px < 48.0 {
        return;
    }
    for (bar_number, index) in downbeats.iter().enumerate() {
        let beat = grid.beats()[*index];
        if beat < from || beat > to {
            continue;
        }
        painter.text(
            egui::pos2(view.x_of(beat) + 3.0, rect.top() + 2.0),
            Align2::LEFT_TOP,
            format!("{}.1", bar_number + 1),
            FontId::monospace(10.0),
            theme.label,
        );
    }
}

/// Typical distance between bar lines on screen, in pixels.
fn bar_spacing_pixels(view: &Viewport, grid: &Beatgrid, downbeats: &[usize]) -> f32 {
    if downbeats.len() < 2 {
        return f32::INFINITY;
    }
    let first = grid.beats()[downbeats[0]];
    let second = grid.beats()[downbeats[1]];
    ((second - first) / view.seconds_per_pixel()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::paint;
    use egui::vec2;
    use mixlyzer_core::TempoSegment;

    fn grid(bpm: f64, beats: usize, ts: u8) -> Beatgrid {
        let period = 60.0 / bpm;
        let times: Vec<f64> = (0..beats).map(|i| i as f64 * period).collect();
        let end = times[beats - 1] + period;
        Beatgrid::new(times, vec![TempoSegment::new(0.0, end, bpm, 0.0, ts)])
    }

    fn viewport(rect: egui::Rect, time: f64, window: f64) -> Viewport {
        Viewport::new(rect, time, window, 300.0)
    }

    #[test]
    fn an_empty_grid_draws_nothing() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 0.0, 12.0),
                &Beatgrid::default(),
                &Theme::dark(),
                true,
            );
        });
        assert!(painted.is_empty());
    }

    #[test]
    fn one_line_is_drawn_per_visible_beat() {
        // 120 BPM is two beats a second; a 12 second window holds 24, plus the
        // small overdraw either side.
        let painted = paint(vec2(1200.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 30.0, 12.0),
                &grid(120.0, 200, 4),
                &Theme::dark(),
                false,
            );
        });
        let count = painted.vertical_line_xs().len();
        assert!(
            (24..=28).contains(&count),
            "expected about 24 beat lines, drew {count}"
        );
    }

    #[test]
    fn bar_starts_are_drawn_heavier_than_the_beats_between() {
        let theme = Theme::dark();
        let painted = paint(vec2(1200.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 30.0, 12.0),
                &grid(120.0, 200, 4),
                &theme,
                false,
            );
        });
        let segments = painted.line_segments();
        let heavy = segments
            .iter()
            .filter(|(_, stroke)| stroke.color == theme.downbeat)
            .count();
        let light = segments
            .iter()
            .filter(|(_, stroke)| stroke.color == theme.beat)
            .count();
        assert!(heavy > 0, "no bar lines");
        assert!(
            light > heavy * 2,
            "with four beats to a bar, beats should outnumber bars: {light} vs {heavy}"
        );
    }

    #[test]
    fn a_waltz_puts_a_bar_line_every_three_beats() {
        let theme = Theme::dark();
        let painted = paint(vec2(1200.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 30.0, 12.0),
                &grid(120.0, 200, 3),
                &theme,
                false,
            );
        });
        let segments = painted.line_segments();
        let heavy = segments
            .iter()
            .filter(|(_, stroke)| stroke.color == theme.downbeat)
            .count();
        let total = segments.len();
        assert!(
            (total as f64 / heavy as f64 - 3.0).abs() < 0.6,
            "{total} lines over {heavy} bars is not three to a bar"
        );
    }

    #[test]
    fn beat_lines_land_where_the_beats_are() {
        let grid = grid(120.0, 200, 4);
        let painted = paint(vec2(1200.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 30.0, 12.0),
                &grid,
                &Theme::dark(),
                false,
            );
        });
        let view = viewport(
            egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(1200.0, 100.0)),
            30.0,
            12.0,
        );
        for x in painted.vertical_line_xs() {
            let time = view.time_of(x);
            let nearest = grid
                .beats()
                .iter()
                .map(|b| (b - time).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 1e-6, "line at {time}s is not on a beat");
        }
    }

    #[test]
    fn lines_span_the_full_height_of_the_rectangle() {
        let painted = paint(vec2(400.0, 80.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 10.0, 12.0),
                &grid(120.0, 200, 4),
                &Theme::dark(),
                false,
            );
        });
        for (points, _) in painted.line_segments() {
            assert_eq!(points[0].y, 0.0);
            assert_eq!(points[1].y, 80.0);
        }
    }

    #[test]
    fn bar_numbers_are_shown_when_there_is_room() {
        let painted = paint(vec2(1200.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 6.0, 12.0),
                &grid(120.0, 200, 4),
                &Theme::dark(),
                true,
            );
        });
        let texts = painted.texts();
        assert!(!texts.is_empty(), "no bar labels drawn");
        assert!(texts.iter().all(|t| t.ends_with(".1")), "got {texts:?}");
    }

    #[test]
    fn bar_numbers_are_dropped_when_the_bars_get_too_close() {
        // A five minute window puts the bars a couple of pixels apart.
        let painted = paint(vec2(600.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 150.0, 300.0),
                &grid(120.0, 2000, 4),
                &Theme::dark(),
                true,
            );
        });
        assert!(
            painted.texts().is_empty(),
            "labels would be illegible at this zoom"
        );
    }

    #[test]
    fn labels_can_be_turned_off_entirely() {
        let painted = paint(vec2(1200.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 6.0, 12.0),
                &grid(120.0, 200, 4),
                &Theme::dark(),
                false,
            );
        });
        assert!(painted.texts().is_empty());
    }

    #[test]
    fn nothing_is_drawn_for_a_window_beyond_the_last_beat() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 500.0, 12.0),
                &grid(120.0, 200, 4),
                &Theme::dark(),
                true,
            );
        });
        assert!(painted.is_empty());
    }
}
