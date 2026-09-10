//! The Mixlyzer desktop application.
//!
//! A thin shell: it owns the window, the file dialog and the analysis thread,
//! and hands everything else to `mixlyzer-ui`, which draws and decides. Keeping
//! the split there is what lets the whole interface be tested headlessly.

#![forbid(unsafe_code)]

mod analysis;

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

use mixlyzer_core::Config;
use mixlyzer_ui::app::{Action, AppState};
use mixlyzer_ui::{markers, overview, panels, track_view, Theme};

use analysis::{AnalysisRequest, AnalysisResult};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Mixlyzer"),
        ..Default::default()
    };
    eframe::run_native(
        "Mixlyzer",
        options,
        Box::new(|cc| Ok(Box::new(Mixlyzer::new(cc)))),
    )
}

/// The running application.
struct Mixlyzer {
    state: AppState,
    theme: Theme,
    config: Config,
    /// Set when something went wrong that the user should see.
    status: Option<String>,
    /// Work handed to the analysis thread.
    requests: Sender<AnalysisRequest>,
    /// Results coming back from it.
    results: Receiver<AnalysisResult>,
    /// Whether an analysis is in flight, so the UI can say so.
    analysing: Option<PathBuf>,
    /// Wall-clock reading from the previous frame, for advancing the playhead.
    last_frame: Option<std::time::Instant>,
}

impl Mixlyzer {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let config = Config::load("config.json").unwrap_or_default();
        let (requests, results) = analysis::spawn(cc.egui_ctx.clone());

        let mut app = Self {
            state: AppState::default(),
            theme: Theme::dark(),
            config,
            status: None,
            requests,
            results,
            analysing: None,
            last_frame: None,
        };
        app.reload_library();
        app
    }

    /// Read the library listing, reporting a failure rather than crashing.
    ///
    /// The Python app lets a damaged `library.db` escape from the widget's
    /// constructor, which kills startup before any window exists.
    fn reload_library(&mut self) {
        match analysis::load_library(&self.config) {
            Ok(tracks) => self.state.library = tracks,
            Err(err) => self.status = Some(format!("Could not open the library: {err}")),
        }
    }

    /// Hand a file to the analysis thread.
    fn analyze(&mut self, path: PathBuf) {
        self.status = None;
        self.analysing = Some(path.clone());
        let request = AnalysisRequest {
            path,
            config: self.config.clone(),
        };
        if self.requests.send(request).is_err() {
            self.status = Some("The analysis worker has stopped.".into());
            self.analysing = None;
        }
    }

    /// Take whatever the analysis thread has finished.
    fn drain_results(&mut self) {
        while let Ok(result) = self.results.try_recv() {
            self.analysing = None;
            match result {
                AnalysisResult::Ready(analyzed) => {
                    self.state.load(analyzed.track, analyzed.scene);
                    self.reload_library();
                }
                AnalysisResult::Failed { path, message } => {
                    self.status = Some(format!("{}: {message}", path.display()));
                }
            }
        }
    }

    /// Move the playhead on by however long the last frame took.
    fn advance_playhead(&mut self) {
        let now = std::time::Instant::now();
        if let Some(previous) = self.last_frame.replace(now) {
            self.state.tick(now.duration_since(previous).as_secs_f64());
        }
    }

    /// Keyboard shortcuts, so the transport is usable without the mouse.
    fn keyboard(&self, ctx: &egui::Context) -> Option<Action> {
        ctx.input(|input| {
            if input.key_pressed(egui::Key::Space) {
                return Some(Action::TogglePlay);
            }
            if input.key_pressed(egui::Key::ArrowLeft) {
                return Some(Action::SeekBar(mixlyzer_ui::Direction::Backward));
            }
            if input.key_pressed(egui::Key::ArrowRight) {
                return Some(Action::SeekBar(mixlyzer_ui::Direction::Forward));
            }
            if input.key_pressed(egui::Key::Minus) {
                return Some(Action::ZoomOut);
            }
            if input.key_pressed(egui::Key::Plus) || input.key_pressed(egui::Key::Equals) {
                return Some(Action::ZoomIn);
            }
            None
        })
    }
}

impl eframe::App for Mixlyzer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_results();
        self.advance_playhead();

        let ctx = ui.ctx().clone();
        let mut pending = self.keyboard(&ctx);

        egui::Panel::top("header").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() {
                    if let Some(path) = rfd_pick_file() {
                        self.analyze(path);
                    }
                }
                if let Some(path) = &self.analysing {
                    ui.spinner();
                    ui.label(format!("Analysing {}…", path.display()));
                }
            });
            panels::track_info(ui, self.state.track.as_ref());
            if let Some(action) = panels::transport(ui, &self.state) {
                pending = Some(action);
            }
            if let Some(status) = &self.status {
                ui.colored_label(egui::Color32::from_rgb(230, 120, 120), status);
            }
        });

        egui::Panel::bottom("library")
            .resizable(true)
            .exact_size(200.0)
            .show(ui, |ui| {
                ui.heading("Library");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(path) = panels::library_table(ui, &self.state.library) {
                        self.analyze(PathBuf::from(path));
                    }
                });
            });

        egui::Panel::top("overview")
            .exact_size(90.0)
            .show(ui, |ui| {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
                let scene = overview::OverviewScene {
                    waveform: self.state.scene.waveform.clone(),
                    key_segments: self.state.scene.key_segments.clone(),
                    phrases: self.state.scene.phrases.clone(),
                    cue_points: self.state.scene.cue_points.clone(),
                    duration_sec: self.state.duration(),
                };
                overview::draw(
                    &painter,
                    response.rect,
                    &scene,
                    self.state.visible_window(),
                    self.state.position(),
                    &self.theme,
                );
                if response.clicked() || response.dragged() {
                    if let Some(position) = response.interact_pointer_pos() {
                        if let Some(time) =
                            overview::time_at(response.rect, self.state.duration(), position)
                        {
                            pending = Some(Action::Seek(time));
                        }
                    }
                }
            });

        egui::CentralPanel::default().show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
            let view = self.state.viewport(response.rect);
            track_view::draw(
                &painter,
                &view,
                &self.state.scene,
                &track_view::Layout::default(),
                &self.theme,
            );
            markers::draw_selection(&painter, &view, self.state.scene.selection, &self.theme);

            // Dragging selects a range; a plain click seeks, snapped to a beat
            // so a DJ lands on the grid rather than a few milliseconds off it.
            if let Some(position) = response.interact_pointer_pos() {
                if let Some(time) = track_view::time_at(&view, position) {
                    if response.drag_started() {
                        pending = Some(Action::SelectFrom(time));
                    } else if response.dragged() {
                        pending = Some(Action::SelectTo(time));
                    } else if response.clicked() {
                        let tolerance = view.seconds_per_pixel() * 6.0;
                        let snapped =
                            track_view::snap_to_beat(&self.state.scene.beatgrid, time, tolerance);
                        pending = Some(Action::Seek(snapped));
                    }
                }
            }
        });

        if let Some(action) = pending {
            self.state.apply(action);
        }
        // Keep animating while the transport runs, so the playhead moves.
        if self.state.is_playing() {
            ctx.request_repaint();
        }
    }
}

/// Ask the desktop for a file.
///
/// Returns `None` when no picker is available, which is the case on a headless
/// machine; the library table still works there.
fn rfd_pick_file() -> Option<PathBuf> {
    None
}
