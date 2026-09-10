//! Render the track view to a real image, so a person can look at it.
//!
//! The unit tests assert on individual shapes; this one rasterises a whole
//! frame the way the window would, which is the only way to catch a layout that
//! is technically correct and visually wrong.

use mixlyzer_core::{Beatgrid, CuePoint, JumpCue, Key, KeySegment, Mode, Phrase, TempoSegment};
use mixlyzer_ui::track_view::{self, Layout, TrackScene};
use mixlyzer_ui::waveform::WaveformData;
use mixlyzer_ui::{Theme, Viewport};

/// A track with tempo, key, structure and cues, so every layer has something.
fn scene(duration: f64) -> TrackScene {
    let bpm = 128.0;
    let period = 60.0 / bpm;
    let beat_count = (duration / period) as usize;
    let beats: Vec<f64> = (0..beat_count).map(|i| i as f64 * period).collect();

    let frames = (duration / 0.01) as usize;
    let envelope = |scale: f32, period_frames: usize| -> Vec<f32> {
        (0..frames)
            .map(|i| {
                let phase = (i % period_frames) as f32 / period_frames as f32;
                scale * (1.0 - phase).powi(2) * (0.6 + 0.4 * ((i / 400) % 3) as f32 / 2.0)
            })
            .collect()
    };
    let beat_frames = (period / 0.01) as usize;

    TrackScene {
        waveform: WaveformData::new(
            envelope(0.95, beat_frames),
            envelope(0.55, beat_frames / 2),
            envelope(0.3, beat_frames / 4),
            0.01,
        ),
        beatgrid: Beatgrid::new(
            beats,
            vec![TempoSegment::new(0.0, duration, bpm, 0.0, 4)],
        ),
        key_segments: vec![
            KeySegment::new(0.0, duration * 0.5, Key::new(9, Mode::Minor)),
            KeySegment::new(duration * 0.5, duration, Key::new(4, Mode::Minor)),
        ],
        phrases: vec![
            Phrase::new(0.0, 15.0, "INTRO"),
            Phrase::new(15.0, 45.0, "VERSE"),
            Phrase::new(45.0, 47.0, "FILL_IN"),
            Phrase::new(47.0, 75.0, "CHORUS"),
            Phrase::new(75.0, 105.0, "VERSE"),
            Phrase::new(105.0, duration, "CHORUS"),
        ],
        cue_points: vec![
            CuePoint::new(0, 47.0, "CHORUS_IN", "Chorus start"),
            CuePoint::new(1, 75.0, "CHORUS_OUT", "Chorus exit"),
        ],
        jump_cues: vec![
            JumpCue::new(0, "A", 47.0, 55.0, 47.0, 0),
            JumpCue::new(1, "B", 105.0, 113.0, 105.0, 0),
        ],
        selection: Some((50.0, 58.0)),
    }
}

/// Rasterise one frame of the track view and write it out as a PNG.
#[test]
fn the_track_view_renders_to_an_image() {
    let (width, height) = (1280u32, 520u32);
    let ctx = egui::Context::default();
    ctx.set_visuals(egui::Visuals::dark());

    // Two passes: the first lets egui load its font atlas, the second draws
    // with the glyphs actually available.
    let mut output = None;
    for _ in 0..2 {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };
        output = Some(ctx.run_ui(input, |ui| {
            let rect = egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            );
            let view = Viewport::new(rect, 52.0, 24.0, 135.0);
            track_view::draw(
                ui.painter(),
                &view,
                &scene(135.0),
                &Layout::default(),
                &Theme::dark(),
            );
        }));
    }
    let output = output.expect("two passes must run");

    let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
    let vertices: usize = primitives
        .iter()
        .map(|clipped| match &clipped.primitive {
            egui::epaint::Primitive::Mesh(mesh) => mesh.vertices.len(),
            egui::epaint::Primitive::Callback(_) => 0,
        })
        .sum();
    assert!(vertices > 1000, "the frame looks empty: {vertices} vertices");

    let pixels = rasterise(&primitives, width, height, &ctx);
    let painted = pixels.chunks_exact(4).filter(|p| p[3] > 0).count();
    assert!(
        painted > (width * height) as usize / 4,
        "only {painted} pixels were covered"
    );

    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("track_view.png");
    image::save_buffer(&out, &pixels, width, height, image::ColorType::Rgba8)
        .expect("the snapshot must be writable");
    println!("wrote {}", out.display());
}

/// Fill triangles into an RGBA buffer.
///
/// Just enough of a rasteriser to prove the frame is not blank: flat-shaded
/// triangles with no texturing, which is what every shape here reduces to
/// apart from text.
fn rasterise(
    primitives: &[egui::ClippedPrimitive],
    width: u32,
    height: u32,
    ctx: &egui::Context,
) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let font_image = ctx.fonts(|fonts| fonts.image());

    for clipped in primitives {
        let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive else {
            continue;
        };
        for triangle in mesh.indices.chunks_exact(3) {
            let corners: Vec<&egui::epaint::Vertex> = triangle
                .iter()
                .map(|index| &mesh.vertices[*index as usize])
                .collect();
            fill_triangle(&mut pixels, width, height, &corners, &font_image);
        }
    }
    pixels
}

fn fill_triangle(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    corners: &[&egui::epaint::Vertex],
    font_image: &egui::ColorImage,
) {
    let xs: Vec<f32> = corners.iter().map(|v| v.pos.x).collect();
    let ys: Vec<f32> = corners.iter().map(|v| v.pos.y).collect();
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
    let max_x = (xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as u32).min(width);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
    let max_y = (ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as u32).min(height);

    let area = edge(corners[0], corners[1], corners[2].pos);
    if area.abs() < 1e-6 {
        return;
    }

    for y in min_y..max_y {
        for x in min_x..max_x {
            let point = egui::pos2(x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(corners[1], corners[2], point) / area;
            let w1 = edge(corners[2], corners[0], point) / area;
            let w2 = edge(corners[0], corners[1], point) / area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let mut color = [
                blend(w0, w1, w2, corners, 0),
                blend(w0, w1, w2, corners, 1),
                blend(w0, w1, w2, corners, 2),
                blend(w0, w1, w2, corners, 3),
            ];
            // Text arrives as a mesh sampling the font atlas; take its coverage
            // from there so glyphs are not drawn as solid blocks.
            let u = w0 * corners[0].uv.x + w1 * corners[1].uv.x + w2 * corners[2].uv.x;
            let v = w0 * corners[0].uv.y + w1 * corners[1].uv.y + w2 * corners[2].uv.y;
            if let Some(coverage) = sample_font(font_image, u, v) {
                color[3] = (f32::from(color[3]) * coverage) as u8;
            }
            if color[3] == 0 {
                continue;
            }
            let offset = ((y * width + x) * 4) as usize;
            let alpha = f32::from(color[3]) / 255.0;
            for channel in 0..3 {
                let existing = f32::from(pixels[offset + channel]);
                pixels[offset + channel] =
                    (f32::from(color[channel]) * alpha + existing * (1.0 - alpha)) as u8;
            }
            pixels[offset + 3] = 255;
        }
    }
}

fn blend(w0: f32, w1: f32, w2: f32, corners: &[&egui::epaint::Vertex], channel: usize) -> u8 {
    let value = w0 * f32::from(corners[0].color.to_array()[channel])
        + w1 * f32::from(corners[1].color.to_array()[channel])
        + w2 * f32::from(corners[2].color.to_array()[channel]);
    value.clamp(0.0, 255.0) as u8
}

/// Coverage from the font atlas, or `None` for a shape that is not text.
///
/// egui points every solid shape at one fully-opaque texel in the atlas, so a
/// sample that comes back opaque means "not a glyph" and the vertex colour
/// stands on its own.
fn sample_font(font_image: &egui::ColorImage, u: f32, v: f32) -> Option<f32> {
    let [w, h] = font_image.size;
    let x = (u * w as f32) as usize;
    let y = (v * h as f32) as usize;
    if x >= w || y >= h {
        return None;
    }
    let alpha = f32::from(font_image.pixels[y * w + x].a()) / 255.0;
    (alpha < 1.0).then_some(alpha)
}

fn edge(a: &egui::epaint::Vertex, b: &egui::epaint::Vertex, point: egui::Pos2) -> f32 {
    (b.pos.x - a.pos.x) * (point.y - a.pos.y) - (b.pos.y - a.pos.y) * (point.x - a.pos.x)
}
