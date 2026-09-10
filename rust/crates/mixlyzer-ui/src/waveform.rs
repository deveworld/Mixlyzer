//! The waveform: three band envelopes stacked into one outline.
//!
//! Low, mid and high energy are drawn as separate colours in the same column so
//! a DJ can see at a glance where the bass drops out or the hats come in.

use egui::{Painter, Rect};

use crate::theme::Theme;
use crate::viewport::Viewport;

/// Band envelopes on a fixed frame grid, as the analysis produces them.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformData {
    pub low: Vec<f32>,
    pub mid: Vec<f32>,
    pub high: Vec<f32>,
    /// Seconds covered by one envelope frame.
    pub frame_duration: f64,
    /// Multiplier that brings the track's loudest moment to full height.
    ///
    /// The envelopes hold RMS, not peak, so a track mastered to full scale
    /// still measures around 0.2 and drawn as-is fills a fifth of the view.
    /// Scaling by the track's own loudest frame is what makes the shape
    /// readable, and it is what DJ software does.
    gain: f32,
}

impl Default for WaveformData {
    fn default() -> Self {
        Self {
            low: Vec::new(),
            mid: Vec::new(),
            high: Vec::new(),
            frame_duration: f64::MIN_POSITIVE,
            gain: 1.0,
        }
    }
}

impl WaveformData {
    pub fn new(low: Vec<f32>, mid: Vec<f32>, high: Vec<f32>, frame_duration: f64) -> Self {
        let peak = [&low, &mid, &high]
            .iter()
            .flat_map(|band| band.iter())
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        // A silent track has no peak to normalise against; leave it alone
        // rather than multiplying zeros by infinity.
        let gain = if peak > 1e-6 { 1.0 / peak } else { 1.0 };
        Self {
            low,
            mid,
            high,
            frame_duration: frame_duration.max(f64::MIN_POSITIVE),
            gain,
        }
    }

    /// The multiplier applied when drawing, so the loudest frame fills the view.
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Number of frames available in every band.
    pub fn len(&self) -> usize {
        self.low.len().min(self.mid.len()).min(self.high.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn duration(&self) -> f64 {
        self.len() as f64 * self.frame_duration
    }

    /// Frame index covering `time`, if the envelopes reach that far.
    fn frame_at(&self, time: f64) -> Option<usize> {
        if time < 0.0 {
            return None;
        }
        let index = (time / self.frame_duration) as usize;
        (index < self.len()).then_some(index)
    }

    /// Loudest value of each band across `[start, end)`.
    ///
    /// One screen pixel usually spans many frames; taking the peak rather than
    /// the mean is what keeps a single kick visible when a whole track is on
    /// screen.
    fn peak_between(&self, start: f64, end: f64) -> Option<(f32, f32, f32)> {
        // A column entirely outside the track draws nothing. Without this,
        // clamping the start to zero would fold every pre-roll column onto the
        // first frame and paint audio before the track began.
        if end <= 0.0 || start >= self.duration() {
            return None;
        }
        let first = self.frame_at(start.max(0.0))?;
        let last = self
            .frame_at(end.max(0.0))
            .unwrap_or(self.len().saturating_sub(1));
        let (first, last) = (first.min(last), last.max(first));
        let slice = |band: &[f32]| {
            band[first..=last.min(band.len() - 1)]
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()))
        };
        Some((slice(&self.low), slice(&self.mid), slice(&self.high)))
    }
}

/// Draw the waveform across `view`'s rectangle.
///
/// One vertical bar per pixel column, mirrored about the middle, with the three
/// bands drawn back to front so the quietest is still visible.
pub fn draw(painter: &Painter, view: &Viewport, data: &WaveformData, theme: &Theme) {
    let rect = view.rect();
    painter.rect_filled(rect, 0.0, theme.background);
    if data.is_empty() || rect.width() < 1.0 {
        return;
    }

    let middle = rect.center().y;
    let half_height = rect.height() * 0.5;
    let seconds_per_pixel = view.seconds_per_pixel();
    let columns = rect.width() as usize;

    for column in 0..columns {
        let x = rect.left() + column as f32;
        let start = view.time_of(x);
        let Some((low, mid, high)) = data.peak_between(start, start + seconds_per_pixel) else {
            continue;
        };
        // Back to front: the widest band is drawn first so the narrower ones
        // stay visible on top of it.
        for (amplitude, color) in [
            (low, theme.wave_low),
            (mid, theme.wave_mid),
            (high, theme.wave_high),
        ] {
            if amplitude <= 0.0 {
                continue;
            }
            let extent = ((amplitude * data.gain).clamp(0.0, 1.0) * half_height).max(0.5);
            painter.rect_filled(
                Rect::from_min_max(
                    egui::pos2(x, middle - extent),
                    egui::pos2(x + 1.0, middle + extent),
                ),
                0.0,
                color,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::paint;
    use egui::{pos2, vec2};

    fn data(len: usize) -> WaveformData {
        WaveformData::new(
            (0..len).map(|i| if i % 4 == 0 { 0.9 } else { 0.1 }).collect(),
            vec![0.5; len],
            vec![0.2; len],
            0.01,
        )
    }

    fn viewport(rect: Rect, time: f64) -> Viewport {
        Viewport::new(rect, time, 12.0, 60.0)
    }

    #[test]
    fn an_empty_waveform_still_paints_its_background() {
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw(
                painter,
                &viewport(rect, 0.0),
                &WaveformData::default(),
                &Theme::dark(),
            );
        });
        assert_eq!(painted.rects().len(), 1, "just the background");
        assert_eq!(painted.rect_colors(), vec![Theme::dark().background]);
    }

    #[test]
    fn each_band_is_drawn_in_its_own_colour() {
        let theme = Theme::dark();
        let painted = paint(vec2(200.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 1.0), &data(2000), &theme);
        });
        let colors = painted.rect_colors();
        for expected in [theme.wave_low, theme.wave_mid, theme.wave_high] {
            assert!(colors.contains(&expected), "missing band colour {expected:?}");
        }
    }

    #[test]
    fn the_waveform_is_mirrored_about_the_middle() {
        let painted = paint(vec2(200.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 1.0), &data(2000), &Theme::dark());
        });
        let middle = 50.0;
        for rect in painted.rects().iter().skip(1) {
            let above = middle - rect.rect.top();
            let below = rect.rect.bottom() - middle;
            assert!(
                (above - below).abs() < 0.01,
                "bar spans {:?}, not centred on {middle}",
                rect.rect
            );
        }
    }

    /// Within one track, a loud passage must stand above a quiet one.
    ///
    /// Comparing two separate tracks would prove nothing now that each is
    /// scaled to its own peak, which is the point of that scaling: every track
    /// fills the view, and the shape inside it carries the dynamics.
    #[test]
    fn a_louder_passage_is_drawn_taller_than_a_quiet_one() {
        // The first half is quiet, the second loud.
        let mut low = vec![0.1f32; 500];
        low.extend(vec![0.9f32; 500]);
        let data = WaveformData::new(low, vec![0.0; 1000], vec![0.0; 1000], 0.01);

        let height_over = |centre: f64| {
            let painted = paint(vec2(100.0, 100.0), |painter, rect| {
                // A one second window, so each pass sees only one half.
                let view = Viewport::new(rect, centre, 1.0, 10.0);
                draw(painter, &view, &data, &Theme::dark());
            });
            painted
                .rects()
                .iter()
                .skip(1)
                .map(|r| r.rect.height())
                .fold(0.0f32, f32::max)
        };
        let quiet = height_over(2.5);
        let loud = height_over(7.5);
        assert!(
            loud > quiet * 4.0,
            "loud passage {loud} should tower over quiet {quiet}"
        );
    }

    #[test]
    fn nothing_is_drawn_where_the_track_has_not_started() {
        // The playhead is at 0.5s with a 12s window, so most of the screen is
        // before the beginning of the track.
        let painted = paint(vec2(400.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 0.5), &data(100), &Theme::dark());
        });
        let left_edge_bars = painted
            .rects()
            .iter()
            .skip(1)
            .filter(|r| r.rect.left() < 100.0)
            .count();
        assert_eq!(left_edge_bars, 0, "drew audio before the track began");
    }

    #[test]
    fn peaks_survive_when_a_pixel_covers_many_frames() {
        // A lone spike in a quiet passage must still be visible zoomed out.
        let mut low = vec![0.05f32; 5000];
        low[2500] = 1.0;
        let spike = WaveformData::new(low, vec![0.0; 5000], vec![0.0; 5000], 0.01);
        let painted = paint(vec2(100.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 25.0), &spike, &Theme::dark());
        });
        let tallest = painted
            .rects()
            .iter()
            .skip(1)
            .map(|r| r.rect.height())
            .fold(0.0f32, f32::max);
        assert!(tallest > 50.0, "the spike was averaged away, tallest {tallest}");
    }

    #[test]
    fn the_loudest_frame_is_scaled_to_fill_the_view() {
        // RMS envelopes peak well below 1.0; drawn unscaled the waveform would
        // occupy a fraction of the height it has.
        let quiet = WaveformData::new(vec![0.2; 500], vec![0.1; 500], vec![0.05; 500], 0.01);
        assert!((quiet.gain() - 5.0).abs() < 1e-4, "gain was {}", quiet.gain());

        let painted = paint(vec2(100.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 1.0), &quiet, &Theme::dark());
        });
        let tallest = painted
            .rects()
            .iter()
            .skip(1)
            .map(|r| r.rect.height())
            .fold(0.0f32, f32::max);
        assert!(
            tallest > 90.0,
            "the loudest band should nearly fill 100 pixels, reached {tallest}"
        );
    }

    #[test]
    fn scaling_keeps_the_bands_in_proportion() {
        let data = WaveformData::new(vec![0.4; 500], vec![0.2; 500], vec![0.1; 500], 0.01);
        let painted = paint(vec2(60.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 1.0), &data, &Theme::dark());
        });
        let theme = Theme::dark();
        let height_of = |color| {
            painted
                .rects()
                .iter()
                .filter(|r| r.fill == color)
                .map(|r| r.rect.height())
                .fold(0.0f32, f32::max)
        };
        let (low, mid) = (height_of(theme.wave_low), height_of(theme.wave_mid));
        assert!((low / mid - 2.0).abs() < 0.1, "low {low} vs mid {mid}");
    }

    #[test]
    fn a_silent_track_is_not_amplified_into_noise() {
        let silence = WaveformData::new(vec![0.0; 500], vec![0.0; 500], vec![0.0; 500], 0.01);
        assert_eq!(silence.gain(), 1.0);
        let painted = paint(vec2(100.0, 100.0), |painter, rect| {
            draw(painter, &viewport(rect, 1.0), &silence, &Theme::dark());
        });
        assert_eq!(painted.rects().len(), 1, "only the background");
    }

    #[test]
    fn a_zero_frame_duration_cannot_divide_by_zero() {
        let data = WaveformData::new(vec![0.5; 10], vec![0.5; 10], vec![0.5; 10], 0.0);
        assert!(data.frame_duration > 0.0);
        assert!(data.duration().is_finite());
    }

    #[test]
    fn ragged_bands_report_the_shortest_length() {
        let data = WaveformData::new(vec![0.0; 10], vec![0.0; 4], vec![0.0; 7], 0.01);
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn a_rectangle_narrower_than_a_pixel_draws_only_the_background() {
        let painted = paint(vec2(100.0, 100.0), |painter, _| {
            let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(0.5, 100.0));
            draw(painter, &viewport(rect, 1.0), &data(500), &Theme::dark());
        });
        assert_eq!(painted.rects().len(), 1);
    }
}
