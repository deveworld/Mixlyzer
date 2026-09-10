//! The one place that maps track time to screen position.
//!
//! Every view draws into the same scrolling window: the playhead sits at a
//! fixed point and the music moves past it. In the Python app each view worked
//! that mapping out for itself from `tl.center_t - tl.current_time`, so the
//! convention lived in eight places at once and the key strip, the waveform and
//! the overview each had their own idea of where the track ended. Here a view
//! is handed a [`Viewport`] and asks it where things go.

use egui::{Pos2, Rect};

/// Maps between track time and screen x for one drawing pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// The rectangle being drawn into.
    rect: Rect,
    /// Track time at the playhead, in seconds.
    current_time: f64,
    /// How much of the track is visible, in seconds.
    window_sec: f64,
    /// Where the playhead sits across the rectangle, `0.0` left to `1.0` right.
    playhead_fraction: f32,
    /// Total track length, for clamping and for the overview.
    duration_sec: f64,
}

impl Viewport {
    /// The playhead's default position: centred, so a DJ sees equal amounts of
    /// what has played and what is coming.
    pub const DEFAULT_PLAYHEAD_FRACTION: f32 = 0.5;

    /// The default visible span, matching the desktop app's twelve seconds.
    pub const DEFAULT_WINDOW_SEC: f64 = 12.0;

    pub fn new(rect: Rect, current_time: f64, window_sec: f64, duration_sec: f64) -> Self {
        Self {
            rect,
            current_time,
            window_sec: window_sec.max(0.1),
            playhead_fraction: Self::DEFAULT_PLAYHEAD_FRACTION,
            duration_sec: duration_sec.max(0.0),
        }
    }

    /// Move the playhead away from the centre, e.g. hard left for a deck view.
    pub fn with_playhead_fraction(mut self, fraction: f32) -> Self {
        self.playhead_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Draw into a different rectangle, keeping the time mapping.
    ///
    /// This is how the strips stack: each gets its own horizontal band but they
    /// all agree on where a given moment sits.
    pub fn with_rect(mut self, rect: Rect) -> Self {
        self.rect = rect;
        self
    }

    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    pub fn window_sec(&self) -> f64 {
        self.window_sec
    }

    pub fn duration_sec(&self) -> f64 {
        self.duration_sec
    }

    /// Screen x of the playhead.
    pub fn playhead_x(&self) -> f32 {
        self.rect.left() + self.rect.width() * self.playhead_fraction
    }

    /// Earliest track time on screen. May be negative before the track starts.
    pub fn start_time(&self) -> f64 {
        self.current_time - self.window_sec * f64::from(self.playhead_fraction)
    }

    /// Latest track time on screen.
    pub fn end_time(&self) -> f64 {
        self.start_time() + self.window_sec
    }

    /// Screen x for a track time. Times off screen give positions off the rect.
    pub fn x_of(&self, time: f64) -> f32 {
        let fraction = (time - self.start_time()) / self.window_sec;
        self.rect.left() + (fraction as f32) * self.rect.width()
    }

    /// Track time at a screen x. The inverse of [`Self::x_of`].
    pub fn time_of(&self, x: f32) -> f64 {
        let fraction = f64::from((x - self.rect.left()) / self.rect.width().max(1.0));
        self.start_time() + fraction * self.window_sec
    }

    /// Whether any of `[start, end]` is on screen.
    ///
    /// Views call this to skip work: a track has thousands of beats and only a
    /// few dozen are ever visible.
    pub fn intersects(&self, start: f64, end: f64) -> bool {
        end >= self.start_time() && start <= self.end_time()
    }

    /// The visible span, widened by a fraction of the window.
    ///
    /// Drawing slightly past the edges stops marks from popping in as they
    /// arrive.
    pub fn padded_span(&self, pad_fraction: f64) -> (f64, f64) {
        let pad = self.window_sec * pad_fraction;
        (self.start_time() - pad, self.end_time() + pad)
    }

    /// A horizontal band of this viewport, given as fractions of its height.
    ///
    /// `0.0` is the top. The strips are stacked with these rather than with
    /// absolute pixel offsets so the layout survives a resize.
    pub fn band(&self, top_fraction: f32, bottom_fraction: f32) -> Rect {
        let (top, bottom) = if top_fraction <= bottom_fraction {
            (top_fraction, bottom_fraction)
        } else {
            (bottom_fraction, top_fraction)
        };
        let height = self.rect.height();
        Rect::from_min_max(
            Pos2::new(self.rect.left(), self.rect.top() + height * top),
            Pos2::new(self.rect.right(), self.rect.top() + height * bottom),
        )
    }

    /// Seconds represented by one pixel, for deciding how much detail to draw.
    pub fn seconds_per_pixel(&self) -> f64 {
        self.window_sec / f64::from(self.rect.width().max(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};

    fn viewport() -> Viewport {
        Viewport::new(
            Rect::from_min_size(pos2(0.0, 0.0), vec2(1200.0, 400.0)),
            30.0,
            12.0,
            300.0,
        )
    }

    #[test]
    fn the_playhead_sits_at_the_centre_by_default() {
        let view = viewport();
        assert_eq!(view.playhead_x(), 600.0);
        assert!((view.x_of(30.0) - 600.0).abs() < 1e-3);
    }

    #[test]
    fn the_visible_span_is_centred_on_the_playhead() {
        let view = viewport();
        assert!((view.start_time() - 24.0).abs() < 1e-9);
        assert!((view.end_time() - 36.0).abs() < 1e-9);
    }

    #[test]
    fn time_and_position_are_inverses() {
        let view = viewport();
        for time in [24.0, 27.5, 30.0, 33.0, 36.0] {
            let back = view.time_of(view.x_of(time));
            assert!((back - time).abs() < 1e-6, "{time} round-tripped to {back}");
        }
    }

    #[test]
    fn the_window_edges_land_on_the_rectangle_edges() {
        let view = viewport();
        assert!((view.x_of(view.start_time()) - 0.0).abs() < 1e-3);
        assert!((view.x_of(view.end_time()) - 1200.0).abs() < 1e-3);
    }

    #[test]
    fn moving_the_playhead_moves_the_window_with_it() {
        let view = viewport().with_playhead_fraction(0.0);
        assert_eq!(view.playhead_x(), 0.0);
        assert!((view.start_time() - 30.0).abs() < 1e-9, "nothing before the playhead");
        assert!((view.end_time() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn a_playhead_fraction_outside_the_view_is_clamped() {
        assert_eq!(viewport().with_playhead_fraction(5.0).playhead_x(), 1200.0);
        assert_eq!(viewport().with_playhead_fraction(-1.0).playhead_x(), 0.0);
    }

    #[test]
    fn changing_the_rectangle_keeps_the_time_mapping() {
        let view = viewport();
        let strip = view.with_rect(Rect::from_min_size(pos2(0.0, 380.0), vec2(1200.0, 20.0)));
        assert_eq!(strip.x_of(30.0), view.x_of(30.0), "strips must line up");
        assert_eq!(strip.start_time(), view.start_time());
    }

    #[test]
    fn intersects_answers_for_spans_on_and_off_screen() {
        let view = viewport();
        assert!(view.intersects(20.0, 25.0), "overlaps the left edge");
        assert!(view.intersects(28.0, 32.0), "fully inside");
        assert!(view.intersects(35.0, 40.0), "overlaps the right edge");
        assert!(view.intersects(0.0, 100.0), "spans the whole window");
        assert!(!view.intersects(0.0, 20.0), "entirely before");
        assert!(!view.intersects(40.0, 50.0), "entirely after");
    }

    #[test]
    fn padding_widens_the_span_on_both_sides() {
        let (start, end) = viewport().padded_span(0.25);
        assert!((start - 21.0).abs() < 1e-9);
        assert!((end - 39.0).abs() < 1e-9);
    }

    #[test]
    fn bands_carve_the_rectangle_top_down() {
        let view = viewport();
        let top = view.band(0.0, 0.25);
        let bottom = view.band(0.75, 1.0);
        assert_eq!(top.top(), 0.0);
        assert_eq!(top.bottom(), 100.0);
        assert_eq!(bottom.top(), 300.0);
        assert_eq!(bottom.bottom(), 400.0);
        assert_eq!(top.width(), view.rect().width(), "bands span the full width");
    }

    #[test]
    fn a_band_given_upside_down_is_still_a_valid_rectangle() {
        let band = viewport().band(0.8, 0.2);
        assert!(band.top() < band.bottom());
        assert_eq!(band.top(), 80.0);
        assert_eq!(band.bottom(), 320.0);
    }

    #[test]
    fn the_window_is_never_zero_width() {
        let view = Viewport::new(
            Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 50.0)),
            0.0,
            0.0,
            10.0,
        );
        assert!(view.window_sec() > 0.0, "a zero window would divide by zero");
        assert!(view.seconds_per_pixel().is_finite());
    }

    #[test]
    fn early_in_a_track_the_window_reaches_before_zero() {
        let view = Viewport::new(
            Rect::from_min_size(pos2(0.0, 0.0), vec2(1200.0, 400.0)),
            1.0,
            12.0,
            300.0,
        );
        assert!(view.start_time() < 0.0, "the first seconds are drawn mid-window");
        assert!(view.x_of(0.0) > view.rect().left());
    }
}
