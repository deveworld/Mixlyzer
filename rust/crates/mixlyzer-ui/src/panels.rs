//! The widget panels around the track display: transport, track info, library.
//!
//! These use egui's own widgets rather than raw painting, so they report what
//! the user did as [`Action`]s the caller applies to its state.

use egui::Ui;
use mixlyzer_core::Track;

use crate::app::{Action, AppState, Direction};

/// Transport controls: play, stop, bar stepping and zoom.
///
/// Returns the action the user asked for, if any.
pub fn transport(ui: &mut Ui, state: &AppState) -> Option<Action> {
    let mut action = None;
    ui.horizontal(|ui| {
        let loaded = state.track.is_some();
        ui.add_enabled_ui(loaded, |ui| {
            let label = if state.is_playing() { "Pause" } else { "Play" };
            if ui.button(label).clicked() {
                action = Some(Action::TogglePlay);
            }
            if ui.button("Stop").clicked() {
                action = Some(Action::Stop);
            }
            if ui.button("|<").on_hover_text("Previous bar").clicked() {
                action = Some(Action::SeekBar(Direction::Backward));
            }
            if ui.button(">|").on_hover_text("Next bar").clicked() {
                action = Some(Action::SeekBar(Direction::Forward));
            }
        });
        ui.separator();
        if ui.button("-").on_hover_text("Zoom out").clicked() {
            action = Some(Action::ZoomOut);
        }
        if ui.button("+").on_hover_text("Zoom in").clicked() {
            action = Some(Action::ZoomIn);
        }
        ui.separator();
        ui.monospace(format_position(state.position(), state.duration()));
    });
    action
}

/// `m:ss.t / m:ss`, the form a DJ reads a transport position in.
pub fn format_position(position: f64, duration: f64) -> String {
    format!("{} / {}", format_time(position, true), format_time(duration, false))
}

fn format_time(seconds: f64, tenths: bool) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--:--".into();
    }
    let whole = seconds.floor() as u64;
    let (minutes, secs) = (whole / 60, whole % 60);
    if tenths {
        let tenth = ((seconds - whole as f64) * 10.0).floor() as u64;
        format!("{minutes}:{secs:02}.{tenth}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

/// Title, artist, tempo and key for the loaded track.
pub fn track_info(ui: &mut Ui, track: Option<&Track>) {
    match track {
        Some(track) => {
            ui.horizontal(|ui| {
                ui.heading(if track.title.is_empty() {
                    track.path.as_str()
                } else {
                    track.title.as_str()
                });
                if !track.artist.is_empty() {
                    ui.label(&track.artist);
                }
            });
            ui.horizontal(|ui| {
                match track.bpm {
                    Some(bpm) => ui.monospace(format!("{bpm:.2} BPM")),
                    None => ui.monospace("-- BPM"),
                };
                ui.separator();
                match track.key {
                    Some(key) => ui.monospace(key.display()),
                    None => ui.monospace("unknown key"),
                };
            });
        }
        None => {
            ui.heading("No track loaded");
            ui.label("Open a file, or pick one from the library below.");
        }
    }
}

/// The library listing. Returns the path of a track the user chose to open.
pub fn library_table(ui: &mut Ui, tracks: &[Track]) -> Option<String> {
    let mut chosen = None;
    if tracks.is_empty() {
        ui.label("The library is empty. Analyse a track to add it.");
        return None;
    }
    egui::Grid::new("library")
        .striped(true)
        .num_columns(4)
        .show(ui, |ui| {
            ui.strong("Title");
            ui.strong("BPM");
            ui.strong("Key");
            ui.strong("Length");
            ui.end_row();

            for track in tracks {
                let title = if track.title.is_empty() {
                    track.path.as_str()
                } else {
                    track.title.as_str()
                };
                if ui.selectable_label(false, title).double_clicked() {
                    chosen = Some(track.path.clone());
                }
                match track.bpm {
                    Some(bpm) => ui.monospace(format!("{bpm:.2}")),
                    None => ui.monospace("--"),
                };
                ui.monospace(track.key_label());
                ui.monospace(format_time(track.duration.unwrap_or(0.0), false));
                ui.end_row();
            }
        });
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track_view::TrackScene;

    /// Run one pass over a panel and return whatever it reported.
    fn run<T>(build: impl FnOnce(&mut Ui) -> T) -> T {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 600.0),
            )),
            ..Default::default()
        };
        let mut result = None;
        let mut build = Some(build);
        let _ = ctx.run_ui(input, |ui| {
            if let Some(build) = build.take() {
                result = Some(build(ui));
            }
        });
        result.expect("the pass must run the panel")
    }

    fn track(title: &str) -> Track {
        let mut track = Track::new("/music/song.flac");
        track.title = title.into();
        track.artist = "Artist".into();
        track.bpm = Some(128.0);
        track.key = Some(mixlyzer_core::Key::from_index(21));
        track.duration = Some(215.0);
        track
    }

    fn loaded_state() -> AppState {
        let mut state = AppState::default();
        state.load(track("Song"), TrackScene::default());
        state
    }

    #[test]
    fn positions_read_as_minutes_seconds_and_tenths() {
        assert_eq!(format_position(0.0, 0.0), "0:00.0 / 0:00");
        assert_eq!(format_position(65.44, 215.0), "1:05.4 / 3:35");
    }

    #[test]
    fn an_unknown_time_is_marked_rather_than_shown_as_zero() {
        assert_eq!(format_time(f64::NAN, true), "--:--");
        assert_eq!(format_time(-1.0, false), "--:--");
    }

    #[test]
    fn the_transport_draws_and_reports_nothing_without_a_click() {
        let action = run(|ui| transport(ui, &loaded_state()));
        assert_eq!(action, None);
    }

    #[test]
    fn the_transport_renders_with_no_track_loaded() {
        let action = run(|ui| transport(ui, &AppState::default()));
        assert_eq!(action, None, "disabled buttons cannot fire");
    }

    #[test]
    fn track_info_renders_a_loaded_track_and_an_empty_slot() {
        run(|ui| track_info(ui, Some(&track("Song"))));
        run(|ui| track_info(ui, None));
    }

    #[test]
    fn track_info_handles_a_track_with_no_tempo_or_key() {
        let mut bare = track("Bare");
        bare.bpm = None;
        bare.key = None;
        bare.artist = String::new();
        bare.title = String::new();
        run(|ui| track_info(ui, Some(&bare)));
    }

    #[test]
    fn the_library_reports_nothing_until_a_row_is_opened() {
        let tracks = vec![track("One"), track("Two")];
        assert_eq!(run(|ui| library_table(ui, &tracks)), None);
    }

    #[test]
    fn an_empty_library_says_so_instead_of_drawing_a_bare_grid() {
        assert_eq!(run(|ui| library_table(ui, &[])), None);
    }
}
