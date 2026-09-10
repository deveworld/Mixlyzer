//! Colours and sizes, in one place so the views agree.

use egui::Color32;

/// The palette and the few fixed sizes the views share.
#[derive(Debug, Clone, PartialEq)]
pub struct Theme {
    pub background: Color32,
    /// Waveform low, mid and high bands.
    pub wave_low: Color32,
    pub wave_mid: Color32,
    pub wave_high: Color32,
    /// Ordinary beat lines.
    pub beat: Color32,
    /// Bar starts, drawn heavier than the beats between them.
    pub downbeat: Color32,
    pub playhead: Color32,
    /// Structural cue markers.
    pub cue: Color32,
    /// Text drawn over the strips.
    pub label: Color32,
    /// A selected range.
    pub selection: Color32,
    pub grid_line_width: f32,
    pub downbeat_line_width: f32,
    pub playhead_width: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// The app's dark palette, matching the desktop build.
    pub fn dark() -> Self {
        Self {
            background: Color32::from_rgb(30, 30, 30),
            wave_low: Color32::from_rgb(80, 140, 220),
            wave_mid: Color32::from_rgb(90, 200, 140),
            wave_high: Color32::from_rgb(230, 200, 90),
            beat: Color32::from_rgb(110, 110, 110),
            downbeat: Color32::from_rgb(235, 70, 70),
            playhead: Color32::from_rgb(250, 250, 250),
            cue: Color32::from_rgb(255, 40, 40),
            label: Color32::from_rgb(225, 225, 225),
            selection: Color32::from_rgba_premultiplied(90, 140, 220, 60),
            grid_line_width: 1.0,
            downbeat_line_width: 2.0,
            playhead_width: 2.0,
        }
    }

    /// A light palette, for a bright room.
    pub fn light() -> Self {
        Self {
            background: Color32::from_rgb(245, 245, 245),
            wave_low: Color32::from_rgb(40, 90, 170),
            wave_mid: Color32::from_rgb(30, 140, 90),
            wave_high: Color32::from_rgb(180, 140, 20),
            beat: Color32::from_rgb(160, 160, 160),
            downbeat: Color32::from_rgb(200, 40, 40),
            playhead: Color32::from_rgb(20, 20, 20),
            cue: Color32::from_rgb(210, 30, 30),
            label: Color32::from_rgb(30, 30, 30),
            selection: Color32::from_rgba_premultiplied(90, 140, 220, 70),
            grid_line_width: 1.0,
            downbeat_line_width: 2.0,
            playhead_width: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_palettes_differ_in_background_and_text() {
        let dark = Theme::dark();
        let light = Theme::light();
        assert_ne!(dark.background, light.background);
        assert_ne!(dark.label, light.label);
    }

    #[test]
    fn downbeats_are_drawn_heavier_than_beats() {
        let theme = Theme::dark();
        assert!(theme.downbeat_line_width > theme.grid_line_width);
    }

    #[test]
    fn the_default_is_the_dark_palette() {
        assert_eq!(Theme::default(), Theme::dark());
    }
}
