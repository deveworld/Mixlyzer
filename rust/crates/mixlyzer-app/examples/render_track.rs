//! Analyse a file and render the track view to a PNG.
//!
//! The desktop binary needs a display; this does not, so it is how the whole
//! stack — decode, analysis, layout, drawing — gets exercised on a headless
//! machine, and how a change to the views can be looked at rather than only
//! asserted on.
//!
//! ```text
//! cargo run -p mixlyzer-app --example render_track -- <audio file> [out.png] [seconds]
//! ```

use mixlyzer_core::AnalysisConfig;
use mixlyzer_dsp::pipeline;
use mixlyzer_ui::track_view::{self, Layout, TrackScene};
use mixlyzer_ui::waveform::WaveformData;
use mixlyzer_ui::{Theme, Viewport};

const WIDTH: u32 = 1440;
const HEIGHT: u32 = 560;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: render_track <audio file> [out.png] [seconds]");
        std::process::exit(2);
    };
    let output = args.next().unwrap_or_else(|| "track.png".into());
    let at: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30.0);

    let analysis = pipeline::analyze_file(&input, &AnalysisConfig::default())?;
    println!(
        "{input}: {:.2} BPM, {}, {} beats, {} jump cues",
        analysis.tempo_global,
        analysis
            .overall_key
            .map(|k| k.display())
            .unwrap_or_else(|| "unknown key".into()),
        analysis.beats().len(),
        analysis.jump_cues.cues().len()
    );

    let scene = TrackScene {
        waveform: WaveformData::new(
            analysis.envelopes.low.clone(),
            analysis.envelopes.mid.clone(),
            analysis.envelopes.high.clone(),
            analysis.envelopes.frame_duration(),
        ),
        beatgrid: analysis.beatgrid.clone(),
        key_segments: analysis.key_segments.clone(),
        phrases: Vec::new(),
        cue_points: Vec::new(),
        jump_cues: analysis.jump_cues.cues().to_vec(),
        selection: None,
    };

    let pixels = render(&scene, at, analysis.duration_sec);
    image::save_buffer(&output, &pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)?;
    println!("wrote {output}");
    Ok(())
}

/// Draw one frame and rasterise it.
fn render(scene: &TrackScene, at: f64, duration: f64) -> Vec<u8> {
    let ctx = egui::Context::default();
    ctx.set_visuals(egui::Visuals::dark());

    // Two passes: the first lets egui build its font atlas, the second draws
    // with the glyphs actually available.
    let mut output = None;
    for _ in 0..2 {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(WIDTH as f32, HEIGHT as f32),
            )),
            ..Default::default()
        };
        output = Some(ctx.run_ui(input, |ui| {
            let rect = egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(WIDTH as f32, HEIGHT as f32),
            );
            let view = Viewport::new(rect, at, 16.0, duration);
            track_view::draw(ui.painter(), &view, scene, &Layout::default(), &Theme::dark());
        }));
    }
    let output = output.expect("two passes must run");
    rasterise(&ctx.tessellate(output.shapes, output.pixels_per_point), &ctx)
}

/// Fill the tessellated triangles into an RGBA buffer.
fn rasterise(primitives: &[egui::ClippedPrimitive], ctx: &egui::Context) -> Vec<u8> {
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let atlas = ctx.fonts(|fonts| fonts.image());
    for clipped in primitives {
        let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive else {
            continue;
        };
        for triangle in mesh.indices.chunks_exact(3) {
            let corners: Vec<&egui::epaint::Vertex> = triangle
                .iter()
                .map(|index| &mesh.vertices[*index as usize])
                .collect();
            fill(&mut pixels, &corners, &atlas);
        }
    }
    pixels
}

fn fill(pixels: &mut [u8], corners: &[&egui::epaint::Vertex], atlas: &egui::ColorImage) {
    let xs: Vec<f32> = corners.iter().map(|v| v.pos.x).collect();
    let ys: Vec<f32> = corners.iter().map(|v| v.pos.y).collect();
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
    let max_x = (xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as u32).min(WIDTH);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
    let max_y = (ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil() as u32).min(HEIGHT);

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
            let weights = [w0, w1, w2];
            let mut color = [0u8; 4];
            for (channel, slot) in color.iter_mut().enumerate() {
                let value: f32 = (0..3)
                    .map(|i| weights[i] * f32::from(corners[i].color.to_array()[channel]))
                    .sum();
                *slot = value.clamp(0.0, 255.0) as u8;
            }
            let u: f32 = (0..3).map(|i| weights[i] * corners[i].uv.x).sum();
            let v: f32 = (0..3).map(|i| weights[i] * corners[i].uv.y).sum();
            if let Some(coverage) = glyph_coverage(atlas, u, v) {
                color[3] = (f32::from(color[3]) * coverage) as u8;
            }
            if color[3] == 0 {
                continue;
            }
            let offset = ((y * WIDTH + x) * 4) as usize;
            let alpha = f32::from(color[3]) / 255.0;
            for channel in 0..3 {
                let behind = f32::from(pixels[offset + channel]);
                pixels[offset + channel] =
                    (f32::from(color[channel]) * alpha + behind * (1.0 - alpha)) as u8;
            }
            pixels[offset + 3] = 255;
        }
    }
}

/// Coverage from the font atlas, or `None` when the shape is not text.
///
/// egui points every solid shape at one fully-opaque texel, so an opaque
/// sample means the vertex colour stands on its own.
fn glyph_coverage(atlas: &egui::ColorImage, u: f32, v: f32) -> Option<f32> {
    let [w, h] = atlas.size;
    let x = (u * w as f32) as usize;
    let y = (v * h as f32) as usize;
    if x >= w || y >= h {
        return None;
    }
    let alpha = f32::from(atlas.pixels[y * w + x].a()) / 255.0;
    (alpha < 1.0).then_some(alpha)
}

fn edge(a: &egui::epaint::Vertex, b: &egui::epaint::Vertex, point: egui::Pos2) -> f32 {
    (b.pos.x - a.pos.x) * (point.y - a.pos.y) - (b.pos.y - a.pos.y) * (point.x - a.pos.x)
}
