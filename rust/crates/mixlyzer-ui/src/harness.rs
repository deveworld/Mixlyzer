//! Running a view without a window, so its drawing can be asserted on.
//!
//! A view is a function that paints into a rectangle. Given an egui context and
//! a raw input, one pass produces a list of shapes; a test can then ask what was
//! actually drawn rather than comparing screenshots.

use egui::{Rect, Shape, Vec2};

/// The shapes one drawing pass produced.
#[derive(Debug, Clone, Default)]
pub struct Painted {
    shapes: Vec<Shape>,
}

impl Painted {
    /// Every shape, with composites flattened out.
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }

    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Every straight line drawn, as endpoint pairs.
    pub fn line_segments(&self) -> Vec<([egui::Pos2; 2], egui::Stroke)> {
        self.shapes
            .iter()
            .filter_map(|shape| match shape {
                Shape::LineSegment { points, stroke } => Some((*points, *stroke)),
                _ => None,
            })
            .collect()
    }

    /// Every rectangle drawn.
    pub fn rects(&self) -> Vec<&egui::epaint::RectShape> {
        self.shapes
            .iter()
            .filter_map(|shape| match shape {
                Shape::Rect(rect) => Some(rect),
                _ => None,
            })
            .collect()
    }

    /// The x positions of vertical lines, sorted.
    ///
    /// Beats, downbeats and the playhead are all vertical lines, so this is how
    /// a test checks where the grid landed.
    pub fn vertical_line_xs(&self) -> Vec<f32> {
        let mut xs: Vec<f32> = self
            .line_segments()
            .iter()
            .filter(|(points, _)| (points[0].x - points[1].x).abs() < 0.01)
            .map(|(points, _)| points[0].x)
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        xs
    }

    /// Every piece of text drawn.
    pub fn texts(&self) -> Vec<String> {
        self.shapes
            .iter()
            .filter_map(|shape| match shape {
                Shape::Text(text) => Some(text.galley.text().to_string()),
                _ => None,
            })
            .collect()
    }

    /// The distinct colours used by filled rectangles.
    pub fn rect_colors(&self) -> Vec<egui::Color32> {
        let mut colors: Vec<egui::Color32> =
            self.rects().iter().map(|rect| rect.fill).collect();
        colors.sort_by_key(|c| c.to_array());
        colors.dedup();
        colors
    }
}

/// Run one drawing pass over a rectangle of `size` and collect what it painted.
pub fn paint(size: Vec2, draw: impl FnOnce(&egui::Painter, Rect)) -> Painted {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, size)),
        ..Default::default()
    };
    let mut draw = Some(draw);
    let output = ctx.run_ui(input, |ui| {
        let rect = Rect::from_min_size(egui::Pos2::ZERO, size);
        let painter = ui.painter().clone();
        if let Some(draw) = draw.take() {
            draw(&painter, rect);
        }
    });
    Painted {
        shapes: flatten(output.shapes.into_iter().map(|clipped| clipped.shape)),
    }
}

/// Unwrap `Shape::Vec` so callers see a flat list.
fn flatten(shapes: impl IntoIterator<Item = Shape>) -> Vec<Shape> {
    let mut out = Vec::new();
    for shape in shapes {
        match shape {
            Shape::Vec(inner) => out.extend(flatten(inner)),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{vec2, Color32, Stroke};

    #[test]
    fn a_pass_that_draws_nothing_paints_nothing() {
        let painted = paint(vec2(100.0, 100.0), |_, _| {});
        assert!(painted.is_empty());
    }

    #[test]
    fn rectangles_and_their_colours_are_visible_to_a_test() {
        let painted = paint(vec2(100.0, 100.0), |painter, rect| {
            painter.rect_filled(rect, 0.0, Color32::RED);
        });
        assert_eq!(painted.rects().len(), 1);
        assert_eq!(painted.rect_colors(), vec![Color32::RED]);
    }

    #[test]
    fn vertical_lines_are_reported_by_x_in_order() {
        let painted = paint(vec2(100.0, 100.0), |painter, rect| {
            for x in [70.0f32, 10.0, 40.0] {
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    Stroke::new(1.0, Color32::WHITE),
                );
            }
        });
        assert_eq!(painted.vertical_line_xs(), vec![10.0, 40.0, 70.0]);
    }

    #[test]
    fn a_horizontal_line_is_not_counted_as_vertical() {
        let painted = paint(vec2(100.0, 100.0), |painter, _| {
            painter.line_segment(
                [egui::pos2(0.0, 50.0), egui::pos2(100.0, 50.0)],
                Stroke::new(1.0, Color32::WHITE),
            );
        });
        assert_eq!(painted.line_segments().len(), 1);
        assert!(painted.vertical_line_xs().is_empty());
    }

    #[test]
    fn text_is_reported_verbatim() {
        let painted = paint(vec2(200.0, 100.0), |painter, rect| {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "12.3",
                egui::FontId::monospace(12.0),
                Color32::WHITE,
            );
        });
        assert_eq!(painted.texts(), vec!["12.3".to_string()]);
    }
}
