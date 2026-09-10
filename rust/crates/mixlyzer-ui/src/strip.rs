//! The horizontal colour bands: musical key and song structure.
//!
//! Both are the same shape — a run of labelled spans drawn edge to edge — so
//! they share the drawing and differ only in what supplies the colour and text.

use egui::{Align2, FontId, Painter, Rect};
use mixlyzer_core::{phrase, KeySegment, Phrase};

use crate::theme::Theme;
use crate::viewport::Viewport;

/// One span of the strip: a time range, a colour and a label.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub start: f64,
    pub end: f64,
    pub color: egui::Color32,
    pub label: String,
}

/// Draw a run of spans across `view`'s rectangle.
///
/// A span narrower than the text it carries is drawn without a label rather
/// than with one spilling over its neighbours.
pub fn draw_spans(painter: &Painter, view: &Viewport, spans: &[Span], theme: &Theme) {
    let rect = view.rect();
    for span in spans {
        if !view.intersects(span.start, span.end) {
            continue;
        }
        // Clip to the rectangle: a span can run far past both edges, and egui
        // would otherwise build a shape thousands of pixels wide.
        let left = view.x_of(span.start).max(rect.left());
        let right = view.x_of(span.end).min(rect.right());
        if right <= left {
            continue;
        }
        let block = Rect::from_min_max(
            egui::pos2(left, rect.top()),
            egui::pos2(right, rect.bottom()),
        );
        painter.rect_filled(block, 0.0, span.color);

        let needed = span.label.len() as f32 * 7.0 + 6.0;
        if !span.label.is_empty() && block.width() >= needed {
            painter.text(
                block.center(),
                Align2::CENTER_CENTER,
                &span.label,
                FontId::proportional(11.0),
                theme.label,
            );
        }
    }
}

/// Spans for the key strip, one per key segment.
pub fn key_spans(segments: &[KeySegment]) -> Vec<Span> {
    segments
        .iter()
        .map(|segment| Span {
            start: segment.start,
            end: segment.end,
            color: key_color(segment.key),
            label: segment.key.camelot().to_string(),
        })
        .collect()
}

/// Colour for a key, placing the twelve Camelot positions around a hue wheel.
///
/// Neighbouring positions get neighbouring hues, so a harmonic move looks like
/// a small change and a clash looks like a large one. Minor keys are darker
/// than their relative major.
pub fn key_color(key: mixlyzer_core::Key) -> egui::Color32 {
    let hue = f32::from(key.camelot_number() - 1) / 12.0;
    let (saturation, value) = match key.mode() {
        mixlyzer_core::Mode::Major => (0.55f32, 0.85f32),
        mixlyzer_core::Mode::Minor => (0.65f32, 0.60f32),
    };
    let (r, g, b) = hsv_to_rgb(hue, saturation, value);
    egui::Color32::from_rgb(r, g, b)
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> (u8, u8, u8) {
    let sector = (hue.rem_euclid(1.0) * 6.0).floor();
    let offset = hue.rem_euclid(1.0) * 6.0 - sector;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * offset);
    let t = value * (1.0 - saturation * (1.0 - offset));
    let (r, g, b) = match sector as u32 % 6 {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

/// Spans for the phrase strip, with fills folded into their neighbours and
/// repeated sections numbered.
pub fn phrase_spans(phrases: &[Phrase]) -> Vec<Span> {
    let display = phrase::merge_fills_for_display(phrases);
    let labels = phrase::abbreviated_labels(&display);
    display
        .iter()
        .zip(labels)
        .map(|(segment, label)| Span {
            start: segment.start,
            end: segment.end,
            color: {
                let (r, g, b) = segment.color();
                egui::Color32::from_rgb(r, g, b)
            },
            label,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::paint;
    use egui::vec2;
    use mixlyzer_core::{Key, Mode};

    fn viewport(rect: Rect, time: f64, window: f64) -> Viewport {
        Viewport::new(rect, time, window, 300.0)
    }

    fn span(start: f64, end: f64, label: &str) -> Span {
        Span {
            start,
            end,
            color: egui::Color32::from_rgb(10, 20, 30),
            label: label.into(),
        }
    }

    #[test]
    fn no_spans_paint_nothing() {
        let painted = paint(vec2(400.0, 20.0), |painter, rect| {
            draw_spans(painter, &viewport(rect, 0.0, 12.0), &[], &Theme::dark());
        });
        assert!(painted.is_empty());
    }

    #[test]
    fn a_visible_span_becomes_one_block() {
        let painted = paint(vec2(400.0, 20.0), |painter, rect| {
            draw_spans(
                painter,
                &viewport(rect, 6.0, 12.0),
                &[span(4.0, 8.0, "")],
                &Theme::dark(),
            );
        });
        assert_eq!(painted.rects().len(), 1);
    }

    #[test]
    fn spans_off_screen_are_skipped_entirely() {
        let painted = paint(vec2(400.0, 20.0), |painter, rect| {
            draw_spans(
                painter,
                &viewport(rect, 100.0, 12.0),
                &[span(0.0, 10.0, "A"), span(200.0, 210.0, "B")],
                &Theme::dark(),
            );
        });
        assert!(painted.is_empty());
    }

    #[test]
    fn a_span_running_past_both_edges_is_clipped_to_the_rectangle() {
        let painted = paint(vec2(400.0, 20.0), |painter, rect| {
            draw_spans(
                painter,
                &viewport(rect, 150.0, 12.0),
                &[span(0.0, 300.0, "")],
                &Theme::dark(),
            );
        });
        let block = painted.rects()[0].rect;
        assert_eq!(block.left(), 0.0);
        assert_eq!(block.right(), 400.0);
    }

    #[test]
    fn a_wide_span_carries_its_label() {
        let painted = paint(vec2(400.0, 20.0), |painter, rect| {
            draw_spans(
                painter,
                &viewport(rect, 6.0, 12.0),
                &[span(0.0, 12.0, "CHORUS")],
                &Theme::dark(),
            );
        });
        assert_eq!(painted.texts(), vec!["CHORUS".to_string()]);
    }

    #[test]
    fn a_narrow_span_drops_its_label_rather_than_overflowing() {
        let painted = paint(vec2(400.0, 20.0), |painter, rect| {
            draw_spans(
                painter,
                &viewport(rect, 6.0, 12.0),
                &[span(6.0, 6.1, "BREAK_CHORUS")],
                &Theme::dark(),
            );
        });
        assert_eq!(painted.rects().len(), 1, "the block is still drawn");
        assert!(painted.texts().is_empty(), "but not its label");
    }

    #[test]
    fn key_spans_carry_camelot_labels() {
        let segments = vec![
            KeySegment::new(0.0, 30.0, Key::new(9, Mode::Minor)),
            KeySegment::new(30.0, 60.0, Key::new(0, Mode::Major)),
        ];
        let spans = key_spans(&segments);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].label, "8A");
        assert_eq!(spans[1].label, "8B");
    }

    #[test]
    fn harmonic_neighbours_get_neighbouring_hues() {
        // 8B and 9B sit next to each other on the wheel, 8B and 2B do not.
        let near = color_distance(
            key_color(Key::from_camelot(8, Mode::Major).unwrap()),
            key_color(Key::from_camelot(9, Mode::Major).unwrap()),
        );
        let far = color_distance(
            key_color(Key::from_camelot(8, Mode::Major).unwrap()),
            key_color(Key::from_camelot(2, Mode::Major).unwrap()),
        );
        assert!(near < far, "near {near} should be closer than far {far}");
    }

    fn color_distance(a: egui::Color32, b: egui::Color32) -> f32 {
        let d = |x: u8, y: u8| (f32::from(x) - f32::from(y)).powi(2);
        (d(a.r(), b.r()) + d(a.g(), b.g()) + d(a.b(), b.b())).sqrt()
    }

    #[test]
    fn a_minor_key_is_darker_than_its_relative_major() {
        let major = key_color(Key::from_camelot(8, Mode::Major).unwrap());
        let minor = key_color(Key::from_camelot(8, Mode::Minor).unwrap());
        let luma = |c: egui::Color32| {
            0.299 * f32::from(c.r()) + 0.587 * f32::from(c.g()) + 0.114 * f32::from(c.b())
        };
        assert!(luma(minor) < luma(major));
    }

    #[test]
    fn every_key_gets_a_distinct_colour() {
        let mut seen: Vec<[u8; 4]> = (0..24)
            .map(|i| key_color(Key::from_index(i)).to_array())
            .collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 24);
    }

    #[test]
    fn phrase_spans_number_repeats_and_absorb_fills() {
        let phrases = vec![
            Phrase::new(0.0, 10.0, "VERSE"),
            Phrase::new(10.0, 12.0, "FILL_IN"),
            Phrase::new(12.0, 30.0, "CHORUS"),
            Phrase::new(30.0, 40.0, "VERSE"),
        ];
        let spans = phrase_spans(&phrases);
        let labels: Vec<&str> = spans.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["V1", "C", "V2"], "the fill joins the chorus");
        assert_eq!(spans[1].start, 10.0);
    }

    #[test]
    fn phrase_colours_come_from_the_shared_palette() {
        let phrases = vec![Phrase::new(0.0, 10.0, "CHORUS")];
        let spans = phrase_spans(&phrases);
        let (r, g, b) = phrase::phrase_color("CHORUS");
        assert_eq!(spans[0].color, egui::Color32::from_rgb(r, g, b));
    }
}
