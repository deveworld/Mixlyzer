//! The application's state and what user actions do to it.
//!
//! Deliberately free of egui: the state machine is what decides where the
//! playhead goes, what is selected and how far the view is zoomed, and keeping
//! it separate means those rules can be tested without drawing anything.

use mixlyzer_core::{Beatgrid, Track};

use crate::track_view::TrackScene;
use crate::viewport::Viewport;

/// Something the user did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    TogglePlay,
    Stop,
    /// Move the playhead to a time.
    Seek(f64),
    /// Move the playhead by a number of seconds.
    Nudge(f64),
    /// Jump to the previous or next bar start.
    SeekBar(Direction),
    ZoomIn,
    ZoomOut,
    /// Begin a selection at a time.
    SelectFrom(f64),
    /// Extend the open selection to a time.
    SelectTo(f64),
    ClearSelection,
}

/// Which way a bar-to-bar move goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Backward,
    Forward,
}

/// Whether the transport is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Playing,
    Paused,
}

/// The whole application state.
#[derive(Debug, Clone)]
pub struct AppState {
    /// The track on screen, if one is loaded.
    pub track: Option<Track>,
    pub scene: TrackScene,
    pub transport: Transport,
    /// Playhead position, in seconds.
    position: f64,
    /// Visible span, in seconds.
    window_sec: f64,
    /// Where a selection began, before its other end is known.
    selection_anchor: Option<f64>,
    /// The library listing shown beside the track.
    pub library: Vec<Track>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            track: None,
            scene: TrackScene::default(),
            transport: Transport::Paused,
            position: 0.0,
            window_sec: Viewport::DEFAULT_WINDOW_SEC,
            selection_anchor: None,
            library: Vec::new(),
        }
    }
}

/// Narrowest and widest the view can zoom, in seconds.
const MIN_WINDOW_SEC: f64 = 1.0;
const MAX_WINDOW_SEC: f64 = 600.0;

/// How much one zoom step changes the window.
const ZOOM_FACTOR: f64 = 1.5;

impl AppState {
    pub fn position(&self) -> f64 {
        self.position
    }

    pub fn window_sec(&self) -> f64 {
        self.window_sec
    }

    /// Track length, or zero when nothing is loaded.
    pub fn duration(&self) -> f64 {
        self.track
            .as_ref()
            .and_then(|track| track.duration)
            .unwrap_or(0.0)
    }

    pub fn is_playing(&self) -> bool {
        self.transport == Transport::Playing
    }

    /// The viewport for the main track display.
    pub fn viewport(&self, rect: egui::Rect) -> Viewport {
        Viewport::new(rect, self.position, self.window_sec, self.duration())
    }

    /// The visible span, for the overview's window box.
    pub fn visible_window(&self) -> (f64, f64) {
        let half = self.window_sec * 0.5;
        (self.position - half, self.position + half)
    }

    /// Load a track and its analysis, resetting the transport.
    pub fn load(&mut self, track: Track, scene: TrackScene) {
        self.track = Some(track);
        self.scene = scene;
        self.position = 0.0;
        self.transport = Transport::Paused;
        self.selection_anchor = None;
    }

    /// Advance the playhead by `delta` seconds of wall clock.
    ///
    /// Stops at the end of the track rather than running past it.
    pub fn tick(&mut self, delta: f64) {
        if !self.is_playing() {
            return;
        }
        let end = self.duration();
        self.position += delta;
        if end > 0.0 && self.position >= end {
            self.position = end;
            self.transport = Transport::Paused;
        }
    }

    /// Apply a user action.
    pub fn apply(&mut self, action: Action) {
        match action {
            Action::TogglePlay => {
                // Nothing loaded means nothing to play; silently starting the
                // transport would leave the UI claiming to play silence.
                if self.track.is_some() {
                    self.transport = match self.transport {
                        Transport::Playing => Transport::Paused,
                        Transport::Paused => Transport::Playing,
                    };
                }
            }
            Action::Stop => {
                self.transport = Transport::Paused;
                self.position = 0.0;
            }
            Action::Seek(time) => self.position = self.clamp_time(time),
            Action::Nudge(delta) => self.position = self.clamp_time(self.position + delta),
            Action::SeekBar(direction) => self.seek_bar(direction),
            Action::ZoomIn => {
                self.window_sec = (self.window_sec / ZOOM_FACTOR).max(MIN_WINDOW_SEC)
            }
            Action::ZoomOut => {
                self.window_sec = (self.window_sec * ZOOM_FACTOR).min(MAX_WINDOW_SEC)
            }
            Action::SelectFrom(time) => {
                let time = self.clamp_time(time);
                self.selection_anchor = Some(time);
                self.scene.selection = None;
            }
            Action::SelectTo(time) => {
                if let Some(anchor) = self.selection_anchor {
                    let time = self.clamp_time(time);
                    let (start, end) = if time >= anchor {
                        (anchor, time)
                    } else {
                        (time, anchor)
                    };
                    // A drag that has not moved is not yet a selection.
                    self.scene.selection = (end > start).then_some((start, end));
                }
            }
            Action::ClearSelection => {
                self.selection_anchor = None;
                self.scene.selection = None;
            }
        }
    }

    /// Move to the neighbouring bar start.
    fn seek_bar(&mut self, direction: Direction) {
        let grid: &Beatgrid = &self.scene.beatgrid;
        let downbeats: Vec<f64> = grid.downbeat_times();
        if downbeats.is_empty() {
            return;
        }
        // A small margin so "previous" from just after a bar goes to the one
        // before it rather than snapping back to where the playhead already is.
        const MARGIN: f64 = 0.05;
        let target = match direction {
            Direction::Forward => downbeats.iter().find(|t| **t > self.position + MARGIN),
            Direction::Backward => downbeats.iter().rev().find(|t| **t < self.position - MARGIN),
        };
        if let Some(time) = target {
            self.position = self.clamp_time(*time);
        }
    }

    /// Keep a time inside the track.
    fn clamp_time(&self, time: f64) -> f64 {
        let end = self.duration();
        if end > 0.0 {
            time.clamp(0.0, end)
        } else {
            time.max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mixlyzer_core::TempoSegment;

    fn track(duration: f64) -> Track {
        let mut track = Track::new("/music/song.flac");
        track.duration = Some(duration);
        track.title = "Song".into();
        track
    }

    fn scene_with_grid() -> TrackScene {
        let period = 60.0 / 120.0; // bars every 2 seconds
        let beats: Vec<f64> = (0..240).map(|i| i as f64 * period).collect();
        TrackScene {
            beatgrid: Beatgrid::new(beats, vec![TempoSegment::new(0.0, 120.0, 120.0, 0.0, 4)]),
            ..TrackScene::default()
        }
    }

    fn loaded() -> AppState {
        let mut state = AppState::default();
        state.load(track(120.0), scene_with_grid());
        state
    }

    #[test]
    fn a_fresh_state_has_nothing_loaded_and_is_paused() {
        let state = AppState::default();
        assert!(state.track.is_none());
        assert!(!state.is_playing());
        assert_eq!(state.position(), 0.0);
        assert_eq!(state.duration(), 0.0);
    }

    #[test]
    fn loading_a_track_rewinds_and_pauses() {
        let mut state = loaded();
        state.apply(Action::Seek(60.0));
        state.apply(Action::TogglePlay);
        state.load(track(90.0), TrackScene::default());
        assert_eq!(state.position(), 0.0);
        assert!(!state.is_playing());
        assert_eq!(state.duration(), 90.0);
    }

    #[test]
    fn play_cannot_be_started_without_a_track() {
        let mut state = AppState::default();
        state.apply(Action::TogglePlay);
        assert!(!state.is_playing(), "there is nothing to play");
    }

    #[test]
    fn play_toggles_and_stop_rewinds() {
        let mut state = loaded();
        state.apply(Action::TogglePlay);
        assert!(state.is_playing());
        state.apply(Action::TogglePlay);
        assert!(!state.is_playing());

        state.apply(Action::Seek(40.0));
        state.apply(Action::Stop);
        assert_eq!(state.position(), 0.0);
        assert!(!state.is_playing());
    }

    #[test]
    fn the_playhead_advances_only_while_playing() {
        let mut state = loaded();
        state.tick(1.0);
        assert_eq!(state.position(), 0.0, "paused should not advance");
        state.apply(Action::TogglePlay);
        state.tick(1.5);
        assert!((state.position() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn playback_stops_at_the_end_rather_than_running_past_it() {
        let mut state = loaded();
        state.apply(Action::Seek(119.0));
        state.apply(Action::TogglePlay);
        state.tick(5.0);
        assert_eq!(state.position(), 120.0);
        assert!(!state.is_playing());
    }

    #[test]
    fn seeking_is_clamped_to_the_track() {
        let mut state = loaded();
        state.apply(Action::Seek(-10.0));
        assert_eq!(state.position(), 0.0);
        state.apply(Action::Seek(500.0));
        assert_eq!(state.position(), 120.0);
    }

    #[test]
    fn nudging_moves_relative_to_where_the_playhead_is() {
        let mut state = loaded();
        state.apply(Action::Seek(10.0));
        state.apply(Action::Nudge(2.5));
        assert!((state.position() - 12.5).abs() < 1e-9);
        state.apply(Action::Nudge(-20.0));
        assert_eq!(state.position(), 0.0, "clamped at the start");
    }

    #[test]
    fn bar_navigation_lands_on_bar_starts() {
        let mut state = loaded();
        state.apply(Action::Seek(5.0));
        state.apply(Action::SeekBar(Direction::Forward));
        assert!((state.position() - 6.0).abs() < 1e-6, "at {}", state.position());
        state.apply(Action::SeekBar(Direction::Backward));
        assert!((state.position() - 4.0).abs() < 1e-6, "at {}", state.position());
    }

    #[test]
    fn stepping_back_from_just_after_a_bar_reaches_the_previous_one() {
        let mut state = loaded();
        state.apply(Action::Seek(6.01));
        state.apply(Action::SeekBar(Direction::Backward));
        assert!((state.position() - 4.0).abs() < 1e-6, "at {}", state.position());
    }

    #[test]
    fn bar_navigation_does_nothing_without_a_grid() {
        let mut state = AppState::default();
        state.load(track(120.0), TrackScene::default());
        state.apply(Action::Seek(30.0));
        state.apply(Action::SeekBar(Direction::Forward));
        assert_eq!(state.position(), 30.0);
    }

    #[test]
    fn zoom_narrows_and_widens_within_limits() {
        let mut state = loaded();
        let start = state.window_sec();
        state.apply(Action::ZoomIn);
        assert!(state.window_sec() < start);
        state.apply(Action::ZoomOut);
        assert!((state.window_sec() - start).abs() < 1e-9);

        for _ in 0..50 {
            state.apply(Action::ZoomIn);
        }
        assert!(state.window_sec() >= MIN_WINDOW_SEC);
        for _ in 0..100 {
            state.apply(Action::ZoomOut);
        }
        assert!(state.window_sec() <= MAX_WINDOW_SEC);
    }

    #[test]
    fn a_drag_makes_a_selection_in_either_direction() {
        let mut state = loaded();
        state.apply(Action::SelectFrom(10.0));
        state.apply(Action::SelectTo(20.0));
        assert_eq!(state.scene.selection, Some((10.0, 20.0)));

        state.apply(Action::SelectFrom(40.0));
        state.apply(Action::SelectTo(30.0));
        assert_eq!(state.scene.selection, Some((30.0, 40.0)), "dragged backwards");
    }

    #[test]
    fn a_drag_that_has_not_moved_is_not_yet_a_selection() {
        let mut state = loaded();
        state.apply(Action::SelectFrom(10.0));
        state.apply(Action::SelectTo(10.0));
        assert_eq!(state.scene.selection, None);
    }

    #[test]
    fn extending_without_an_anchor_selects_nothing() {
        let mut state = loaded();
        state.apply(Action::SelectTo(20.0));
        assert_eq!(state.scene.selection, None);
    }

    #[test]
    fn clearing_removes_the_selection_and_its_anchor() {
        let mut state = loaded();
        state.apply(Action::SelectFrom(10.0));
        state.apply(Action::SelectTo(20.0));
        state.apply(Action::ClearSelection);
        assert_eq!(state.scene.selection, None);
        state.apply(Action::SelectTo(30.0));
        assert_eq!(state.scene.selection, None, "the anchor is gone too");
    }

    #[test]
    fn the_visible_window_is_centred_on_the_playhead() {
        let mut state = loaded();
        state.apply(Action::Seek(30.0));
        let (from, to) = state.visible_window();
        assert!((from - 24.0).abs() < 1e-9);
        assert!((to - 36.0).abs() < 1e-9);
    }

    #[test]
    fn the_viewport_reflects_the_current_position_and_zoom() {
        let mut state = loaded();
        state.apply(Action::Seek(30.0));
        state.apply(Action::ZoomIn);
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 400.0));
        let view = state.viewport(rect);
        assert_eq!(view.current_time(), 30.0);
        assert!((view.window_sec() - state.window_sec()).abs() < 1e-9);
        assert_eq!(view.duration_sec(), 120.0);
    }
}
