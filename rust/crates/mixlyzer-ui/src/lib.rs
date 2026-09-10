//! Mixlyzer's track views, drawn with egui.
//!
//! This crate holds the drawing and the interaction logic and nothing else: it
//! opens no window and owns no event loop, so every view can be exercised in a
//! headless test by running one egui pass and inspecting what it painted.
//!
//! All views share a [`Viewport`], which owns the single mapping from track
//! time to screen position. The Python views each recomputed that mapping from
//! the timeline's fields, so the convention was restated in eight files and
//! they did not entirely agree with one another.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod app;
pub mod beatgrid;
pub mod harness;
pub mod markers;
pub mod overview;
pub mod panels;
pub mod strip;
pub mod theme;
pub mod track_view;
pub mod viewport;
pub mod waveform;

pub use app::{Action, AppState, Direction, Transport};
pub use theme::Theme;
pub use track_view::{Layout, TrackScene};
pub use viewport::Viewport;
pub use waveform::WaveformData;
