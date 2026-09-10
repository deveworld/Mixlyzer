//! The whole-track minimap.
//!
//! Where the main view shows a few seconds in detail, this shows the entire
//! track at once: the structure, the key changes, and a box marking what the
//! main view is looking at. Clicking it seeks.

use egui::{Painter, Rect, Stroke};
use mixlyzer_core::{CuePoint, KeySegment, Phrase};

use crate::strip;
use crate::theme::Theme;
use crate::waveform::WaveformData;

/// What the overview draws.
#[derive(Debug, Clone, Default)]
pub struct OverviewScene {
    pub waveform: WaveformData,
    pub key_segments: Vec<KeySegment>,
    pub phrases: Vec<Phrase>,
    pub cue_points: Vec<CuePoint>,
    pub duration_sec: f64,
}

/// Draw the overview into `rect`, marking `window` as the visible span.
pub fn draw(
    painter: &Painter,
    rect: Rect,
    scene: &OverviewScene,
    window: (f64, f64),
    playhead: f64,
    theme: &Theme,
) {
    painter.rect_filled(rect, 0.0, theme.background);
    let duration = scene.duration_sec;
    if duration <= 0.0 || rect.width() < 2.0 {
        return;
    }
    let x_of = |time: f64| {
        rect.left() + (time / duration).clamp(0.0, 1.0) as f32 * rect.width()
    };

    // Structure across the top, the waveform below it, key along the bottom.
    let phrase_band = band(rect, 0.0, 0.25);
    for span in strip::phrase_spans(&scene.phrases) {
        let block = Rect::from_min_max(
            egui::pos2(x_of(span.start), phrase_band.top()),
            egui::pos2(x_of(span.end), phrase_band.bottom()),
        );
        if block.width() < 0.5 {
            continue;
        }
        painter.rect_filled(block, 0.0, span.color);
        if block.width() > 18.0 {
            painter.text(
                block.center(),
                egui::Align2::CENTER_CENTER,
                span.label,
                egui::FontId::proportional(9.0),
                theme.label,
            );
        }
    }

    let wave_band = band(rect, 0.25, 0.8);
    draw_waveform_summary(painter, wave_band, &scene.waveform, duration, theme);

    let key_band = band(rect, 0.8, 1.0);
    for span in strip::key_spans(&scene.key_segments) {
        let block = Rect::from_min_max(
            egui::pos2(x_of(span.start), key_band.top()),
            egui::pos2(x_of(span.end), key_band.bottom()),
        );
        if block.width() >= 0.5 {
            painter.rect_filled(block, 0.0, span.color);
        }
    }

    for cue in &scene.cue_points {
        let x = x_of(cue.time_sec);
        painter.line_segment(
            [
                egui::pos2(x, rect.top()),
                egui::pos2(x, rect.top() + rect.height() * 0.25),
            ],
            Stroke::new(1.0, theme.cue),
        );
    }

    // The box showing what the detailed view is looking at.
    let (from, to) = window;
    let box_rect = Rect::from_min_max(
        egui::pos2(x_of(from), rect.top()),
        egui::pos2(x_of(to).max(x_of(from) + 2.0), rect.bottom()),
    );
    painter.rect_stroke(
        box_rect,
        0.0,
        Stroke::new(1.0, theme.label),
        egui::StrokeKind::Inside,
    );

    let playhead_x = x_of(playhead);
    painter.line_segment(
        [
            egui::pos2(playhead_x, rect.top()),
            egui::pos2(playhead_x, rect.bottom()),
        ],
        Stroke::new(theme.playhead_width, theme.playhead),
    );
}

/// One bar per pixel column, from the peak of all three bands.
fn draw_waveform_summary(
    painter: &Painter,
    rect: Rect,
    data: &WaveformData,
    duration: f64,
    theme: &Theme,
) {
    if data.is_empty() {
        return;
    }
    let columns = rect.width() as usize;
    let middle = rect.center().y;
    let half_height = rect.height() * 0.5;
    for column in 0..columns {
        let start = duration * column as f64 / columns as f64;
        let end = duration * (column + 1) as f64 / columns as f64;
        let first = ((start / data.frame_duration) as usize).min(data.len().saturating_sub(1));
        let last = ((end / data.frame_duration) as usize).min(data.len().saturating_sub(1));
        let peak = (first..=last)
            .map(|i| {
                data.low[i]
                    .abs()
                    .max(data.mid[i].abs())
                    .max(data.high[i].abs())
            })
            .fold(0.0f32, f32::max);
        if peak <= 0.0 {
            continue;
        }
        let extent = (peak.clamp(0.0, 1.0) * half_height).max(0.5);
        let x = rect.left() + column as f32;
        painter.rect_filled(
            Rect::from_min_max(
                egui::pos2(x, middle - extent),
                egui::pos2(x + 1.0, middle + extent),
            ),
            0.0,
            theme.wave_mid,
        );
    }
}

fn band(rect: Rect, top: f32, bottom: f32) -> Rect {
    Rect::from_min_max(
        egui::pos2(rect.left(), rect.top() + rect.height() * top),
        egui::pos2(rect.left() + rect.width(), rect.top() + rect.height() * bottom),
    )
}

/// The track time a click at `position` refers to.
pub fn time_at(rect: Rect, duration_sec: f64, position: egui::Pos2) -> Option<f64> {
    if !rect.contains(position) || duration_sec <= 0.0 {
        return None;
    }
    let fraction = f64::from((position.x - rect.left()) / rect.width().max(1.0));
    Some((fraction * duration_sec).clamp(0.0, duration_sec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::paint;
    use egui::{pos2, vec2};
    use mixlyzer_core::{Key, Mode};

    fn scene() -> OverviewScene {
        OverviewScene {
            waveform: WaveformData::new(vec![0.7; 30000], vec![0.5; 30000], vec![0.3; 30000], 0.01),
            key_segments: vec![
                KeySegment::new(0.0, 150.0, Key::new(9, Mode::Minor)),
                KeySegment::new(150.0, 300.0, Key::new(0, Mode::Major)),
            ],
            phrases: vec![
                Phrase::new(0.0, 30.0, "INTRO"),
                Phrase::new(30.0, 120.0, "VERSE"),
                Phrase::new(120.0, 300.0, "CHORUS"),
            ],
            cue_points: vec![CuePoint::new(0, 120.0, "CHORUS_IN", "")],
            duration_sec: 300.0,
        }
    }

    #[test]
    fn an_empty_overview_paints_only_its_background() {
        let painted = paint(vec2(600.0, 80.0), |painter, rect| {
            draw(
                painter,
                rect,
                &OverviewScene::default(),
                (0.0, 12.0),
                0.0,
                &Theme::dark(),
            );
        });
        assert_eq!(painted.rects().len(), 1);
    }

    #[test]
    fn the_whole_track_is_drawn_regardless_of_the_visible_window() {
        let theme = Theme::dark();
        let painted = paint(vec2(600.0, 80.0), |painter, rect| {
            draw(painter, rect, &scene(), (100.0, 112.0), 106.0, &theme);
        });
        let colors = painted.rect_colors();
        assert!(colors.contains(&theme.wave_mid), "no waveform summary");
        // Both keys should appear even though only one is in the window.
        assert!(colors.contains(&strip::key_color(Key::new(9, Mode::Minor))));
        assert!(colors.contains(&strip::key_color(Key::new(0, Mode::Major))));
    }

    #[test]
    fn the_window_box_tracks_what_the_main_view_shows() {
        let painted = paint(vec2(600.0, 80.0), |painter, rect| {
            draw(painter, rect, &scene(), (150.0, 162.0), 156.0, &Theme::dark());
        });
        // 150s of 300s is halfway across a 600 pixel strip.
        let boxes: Vec<_> = painted
            .rects()
            .iter()
            .filter(|r| r.fill == egui::Color32::TRANSPARENT)
            .map(|r| r.rect)
            .collect();
        assert_eq!(boxes.len(), 1, "expected exactly one outline");
        assert!((boxes[0].left() - 300.0).abs() < 2.0, "box at {:?}", boxes[0]);
    }

    #[test]
    fn the_playhead_is_drawn_at_its_position_in_the_track() {
        let painted = paint(vec2(600.0, 80.0), |painter, rect| {
            draw(painter, rect, &scene(), (0.0, 12.0), 75.0, &Theme::dark());
        });
        // A quarter of the way through 300s is a quarter across 600 pixels.
        assert!(painted.vertical_line_xs().contains(&150.0));
    }

    #[test]
    fn wide_phrases_are_labelled_and_narrow_ones_are_not() {
        let painted = paint(vec2(600.0, 80.0), |painter, rect| {
            draw(painter, rect, &scene(), (0.0, 12.0), 0.0, &Theme::dark());
        });
        let texts = painted.texts();
        assert!(texts.contains(&"C".to_string()), "got {texts:?}");
    }

    #[test]
    fn a_click_maps_to_a_position_in_the_track() {
        let rect = Rect::from_min_size(egui::Pos2::ZERO, vec2(600.0, 80.0));
        assert_eq!(time_at(rect, 300.0, pos2(0.0, 40.0)), Some(0.0));
        assert!((time_at(rect, 300.0, pos2(300.0, 40.0)).unwrap() - 150.0).abs() < 1.0);
        assert!((time_at(rect, 300.0, pos2(600.0, 40.0)).unwrap() - 300.0).abs() < 1.0);
    }

    #[test]
    fn a_click_outside_or_on_an_unloaded_track_maps_to_nothing() {
        let rect = Rect::from_min_size(egui::Pos2::ZERO, vec2(600.0, 80.0));
        assert!(time_at(rect, 300.0, pos2(700.0, 40.0)).is_none());
        assert!(time_at(rect, 0.0, pos2(300.0, 40.0)).is_none());
    }

    #[test]
    fn a_track_with_no_duration_draws_nothing_beyond_the_background() {
        let mut scene = scene();
        scene.duration_sec = 0.0;
        let painted = paint(vec2(600.0, 80.0), |painter, rect| {
            draw(painter, rect, &scene, (0.0, 12.0), 0.0, &Theme::dark());
        });
        assert_eq!(painted.rects().len(), 1);
    }
}
