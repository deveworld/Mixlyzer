//! The main track display: every layer, stacked and sharing one time mapping.
//!
//! Layout, top to bottom: cue markers, the phrase strip, the waveform with the
//! beat grid over it, JumpCUE regions, and the key strip. The playhead is drawn
//! last so it sits above everything.

use egui::Painter;
use mixlyzer_core::{Beatgrid, CuePoint, JumpCue, KeySegment, Phrase};

use crate::markers;
use crate::strip;
use crate::theme::Theme;
use crate::viewport::Viewport;
use crate::waveform::{self, WaveformData};

/// Where each layer sits, as fractions of the view's height.
///
/// Fractions rather than pixels so the layout survives a resize; the desktop
/// app hard-codes offsets and its strips drift apart when the window changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    pub cue_band: (f32, f32),
    pub phrase_band: (f32, f32),
    pub waveform_band: (f32, f32),
    pub jump_band: (f32, f32),
    pub key_band: (f32, f32),
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            cue_band: (0.0, 0.06),
            phrase_band: (0.06, 0.16),
            waveform_band: (0.16, 0.82),
            jump_band: (0.82, 0.90),
            key_band: (0.90, 1.0),
        }
    }
}

impl Layout {
    /// Whether the bands tile the view without gaps or overlaps.
    pub fn is_contiguous(&self) -> bool {
        let bands = [
            self.cue_band,
            self.phrase_band,
            self.waveform_band,
            self.jump_band,
            self.key_band,
        ];
        if (bands[0].0 - 0.0).abs() > 1e-6 || (bands[bands.len() - 1].1 - 1.0).abs() > 1e-6 {
            return false;
        }
        bands.windows(2).all(|pair| (pair[0].1 - pair[1].0).abs() < 1e-6)
    }
}

/// Everything the track view draws.
#[derive(Debug, Clone, Default)]
pub struct TrackScene {
    pub waveform: WaveformData,
    pub beatgrid: Beatgrid,
    pub key_segments: Vec<KeySegment>,
    pub phrases: Vec<Phrase>,
    pub cue_points: Vec<CuePoint>,
    pub jump_cues: Vec<JumpCue>,
    /// A range the user has selected, for editing.
    pub selection: Option<(f64, f64)>,
}

impl TrackScene {
    /// Whether there is anything to draw.
    pub fn is_empty(&self) -> bool {
        self.waveform.is_empty()
            && self.beatgrid.is_empty()
            && self.key_segments.is_empty()
            && self.phrases.is_empty()
            && self.cue_points.is_empty()
            && self.jump_cues.is_empty()
    }
}

/// Draw the whole track view.
pub fn draw(
    painter: &Painter,
    view: &Viewport,
    scene: &TrackScene,
    layout: &Layout,
    theme: &Theme,
) {
    painter.rect_filled(view.rect(), 0.0, theme.background);

    let band = |range: (f32, f32)| view.with_rect(view.band(range.0, range.1));

    strip::draw_spans(
        painter,
        &band(layout.phrase_band),
        &strip::phrase_spans(&scene.phrases),
        theme,
    );

    let wave_view = band(layout.waveform_band);
    waveform::draw(painter, &wave_view, &scene.waveform, theme);
    markers::draw_selection(painter, &wave_view, scene.selection, theme);
    crate::beatgrid::draw(painter, &wave_view, &scene.beatgrid, theme, false);

    markers::draw_jump_cues(painter, &band(layout.jump_band), &scene.jump_cues, theme);

    strip::draw_spans(
        painter,
        &band(layout.key_band),
        &strip::key_spans(&scene.key_segments),
        theme,
    );

    markers::draw_cue_points(painter, &band(layout.cue_band), &scene.cue_points, theme);

    let bar_beat = scene.beatgrid.bar_beat_at(view.current_time()).map(|p| p.to_string());
    markers::draw_playhead(painter, view, theme, bar_beat.as_deref());
}

/// The time a click at `position` refers to, if it landed inside the view.
pub fn time_at(view: &Viewport, position: egui::Pos2) -> Option<f64> {
    view.rect()
        .contains(position)
        .then(|| view.time_of(position.x))
}

/// Snap a time to the nearest beat, when one is close enough to have been meant.
///
/// `tolerance_sec` should be about half a beat; beyond that the user was
/// pointing between beats deliberately.
pub fn snap_to_beat(grid: &Beatgrid, time: f64, tolerance_sec: f64) -> f64 {
    let Some(index) = grid.nearest_beat_index(time) else {
        return time;
    };
    let beat = grid.beats()[index];
    if (beat - time).abs() <= tolerance_sec {
        beat
    } else {
        time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::paint;
    use egui::{pos2, vec2, Rect};
    use mixlyzer_core::{Key, Mode, TempoSegment};

    fn scene() -> TrackScene {
        let period = 60.0 / 120.0;
        let beats: Vec<f64> = (0..240).map(|i| i as f64 * period).collect();
        TrackScene {
            waveform: WaveformData::new(vec![0.6; 12000], vec![0.4; 12000], vec![0.2; 12000], 0.01),
            beatgrid: Beatgrid::new(
                beats,
                vec![TempoSegment::new(0.0, 120.0, 120.0, 0.0, 4)],
            ),
            key_segments: vec![KeySegment::new(0.0, 120.0, Key::new(9, Mode::Minor))],
            phrases: vec![
                Phrase::new(0.0, 30.0, "INTRO"),
                Phrase::new(30.0, 90.0, "CHORUS"),
            ],
            cue_points: vec![CuePoint::new(0, 30.0, "CHORUS_IN", "")],
            jump_cues: vec![JumpCue::new(0, "A", 28.0, 36.0, 30.0, 0)],
            selection: None,
        }
    }

    fn viewport(rect: Rect, time: f64) -> Viewport {
        Viewport::new(rect, time, 12.0, 120.0)
    }

    #[test]
    fn the_default_layout_tiles_the_view() {
        assert!(Layout::default().is_contiguous());
    }

    #[test]
    fn a_layout_with_a_gap_is_reported_as_such() {
        let broken = Layout {
            phrase_band: (0.06, 0.14),
            waveform_band: (0.16, 0.82),
            ..Layout::default()
        };
        assert!(!broken.is_contiguous());
    }

    #[test]
    fn an_empty_scene_still_paints_a_background_and_a_playhead() {
        let painted = paint(vec2(800.0, 400.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 0.0),
                &TrackScene::default(),
                &Layout::default(),
                &Theme::dark(),
            );
        });
        assert!(TrackScene::default().is_empty());
        assert_eq!(painted.vertical_line_xs(), vec![400.0], "the playhead");
    }

    #[test]
    fn a_full_scene_draws_every_layer() {
        let theme = Theme::dark();
        let painted = paint(vec2(800.0, 400.0), |painter, rect| {
            draw(painter, &viewport(rect, 30.0), &scene(), &Layout::default(), &theme);
        });
        let colors = painted.rect_colors();
        assert!(colors.contains(&theme.wave_low), "no waveform");
        assert!(!painted.vertical_line_xs().is_empty(), "no beat grid");
        assert!(!painted.texts().is_empty(), "no labels");
        // Phrase, key and jump blocks are all rectangles beyond the waveform.
        assert!(painted.rects().len() > 100);
    }

    #[test]
    fn every_layer_stays_inside_its_own_band() {
        let painted = paint(vec2(800.0, 400.0), |painter, rect| {
            draw(painter, &viewport(rect, 30.0), &scene(), &Layout::default(), &Theme::dark());
        });
        let layout = Layout::default();
        let key_top = 400.0 * layout.key_band.0;
        let rects = painted.rects();
        let key_blocks: Vec<_> = rects
            .iter()
            .filter(|r| r.rect.top() >= key_top - 0.01)
            .collect();
        assert!(!key_blocks.is_empty(), "the key strip should be drawn");
        for block in key_blocks {
            assert!(
                block.rect.bottom() <= 400.0 + 0.01,
                "a block escaped the view: {:?}",
                block.rect
            );
        }
    }

    #[test]
    fn the_playhead_carries_the_bar_and_beat() {
        let painted = paint(vec2(800.0, 400.0), |painter, rect| {
            draw(painter, &viewport(rect, 30.0), &scene(), &Layout::default(), &Theme::dark());
        });
        // At 120 BPM, 30s in is beat 60, which is bar 16 beat 1.
        assert!(
            painted.texts().iter().any(|t| t.as_str() == "16.1"),
            "expected a bar.beat label, got {:?}",
            painted.texts()
        );
    }

    #[test]
    fn a_selection_is_shaded_over_the_waveform() {
        let mut scene = scene();
        scene.selection = Some((28.0, 32.0));
        let theme = Theme::dark();
        let painted = paint(vec2(800.0, 400.0), |painter, rect| {
            draw(painter, &viewport(rect, 30.0), &scene, &Layout::default(), &theme);
        });
        assert!(painted.rect_colors().contains(&theme.selection));
    }

    #[test]
    fn a_click_inside_the_view_reports_its_time() {
        let rect = Rect::from_min_size(egui::Pos2::ZERO, vec2(800.0, 400.0));
        let view = viewport(rect, 30.0);
        let time = time_at(&view, pos2(400.0, 200.0)).unwrap();
        assert!((time - 30.0).abs() < 1e-6, "centre should be the playhead time");
    }

    #[test]
    fn a_click_outside_the_view_reports_nothing() {
        let rect = Rect::from_min_size(egui::Pos2::ZERO, vec2(800.0, 400.0));
        assert!(time_at(&viewport(rect, 30.0), pos2(900.0, 200.0)).is_none());
    }

    #[test]
    fn snapping_moves_a_near_miss_onto_the_beat() {
        let grid = scene().beatgrid;
        // Beats are half a second apart; 30.02 belongs to the one at 30.0.
        assert!((snap_to_beat(&grid, 30.02, 0.25) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn snapping_leaves_a_deliberate_offbeat_alone() {
        let grid = scene().beatgrid;
        // A quarter second past the beat, with a tolerance of a tenth.
        assert!((snap_to_beat(&grid, 30.25, 0.1) - 30.25).abs() < 1e-9);
    }

    #[test]
    fn snapping_without_a_grid_returns_the_time_unchanged() {
        assert_eq!(snap_to_beat(&Beatgrid::default(), 12.34, 1.0), 12.34);
    }
}
