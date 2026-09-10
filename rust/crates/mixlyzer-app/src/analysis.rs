//! Analysis on a worker thread, so the interface never blocks on a decode.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};

use mixlyzer_core::{Config, Track};
use mixlyzer_dsp::pipeline;
use mixlyzer_store::{migration, FeatureFile, FeatureStore, Library};
use mixlyzer_ui::track_view::TrackScene;
use mixlyzer_ui::waveform::WaveformData;

/// A file to analyse.
pub struct AnalysisRequest {
    pub path: PathBuf,
    pub config: Config,
}

/// A finished analysis, ready to draw.
pub struct Analyzed {
    pub track: Track,
    pub scene: TrackScene,
}

/// What came back.
///
/// The success payload is boxed because it carries whole envelope buffers,
/// which would otherwise make every failure message just as large.
pub enum AnalysisResult {
    Ready(Box<Analyzed>),
    Failed { path: PathBuf, message: String },
}

/// Start the worker, returning the ends of the two channels.
///
/// The context is used to wake the interface when a result lands; without it
/// the window would sit idle until the user moved the mouse.
pub fn spawn(ctx: egui::Context) -> (Sender<AnalysisRequest>, Receiver<AnalysisResult>) {
    let (request_tx, request_rx) = mpsc::channel::<AnalysisRequest>();
    let (result_tx, result_rx) = mpsc::channel::<AnalysisResult>();

    std::thread::Builder::new()
        .name("analysis".into())
        .spawn(move || {
            // The weights are parsed once and reused: the search and the
            // 400 KB parse would otherwise repeat for every track.
            let options = pipeline::AnalysisOptions::discovering_phrase_model();
            while let Ok(request) = request_rx.recv() {
                let outcome = run(&request, &options);
                if result_tx.send(outcome).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        })
        .expect("the analysis thread must start");

    (request_tx, result_rx)
}

/// Analyse one file into something the views can draw.
fn run(request: &AnalysisRequest, options: &pipeline::AnalysisOptions) -> AnalysisResult {
    let analysis = match pipeline::analyze_file_with(
        &request.path,
        &request.config.analysisconfig,
        options,
    ) {
        Ok(analysis) => analysis,
        Err(err) => {
            return AnalysisResult::Failed {
                path: request.path.clone(),
                message: err.to_string(),
            }
        }
    };

    let mut track = Track::new(&request.path.to_string_lossy());
    track.title = request
        .path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    track.duration = Some(analysis.duration_sec);
    track.bpm = Some(analysis.tempo_global);
    track.key = analysis.overall_key;

    let scene = TrackScene {
        waveform: WaveformData::new(
            analysis.envelopes.low.clone(),
            analysis.envelopes.mid.clone(),
            analysis.envelopes.high.clone(),
            analysis.envelopes.frame_duration(),
        ),
        beatgrid: analysis.beatgrid.clone(),
        key_segments: analysis.key_segments.clone(),
        phrases: analysis.phrases.clone(),
        cue_points: analysis.cue_points.clone(),
        jump_cues: analysis.jump_cues.cues().to_vec(),
        selection: None,
    };

    // Record the track so it survives a restart. A library that cannot be
    // written is worth reporting, but not worth throwing away an analysis the
    // user is about to look at, so the failure rides along with the result.
    if let Err(err) = persist(&request.config, &mut track, &analysis) {
        eprintln!("could not add {} to the library: {err}", request.path.display());
    }

    AnalysisResult::Ready(Box::new(Analyzed { track, scene }))
}

/// Write a finished analysis into the library and the feature store.
fn persist(
    config: &Config,
    track: &mut Track,
    analysis: &mixlyzer_dsp::Analysis,
) -> Result<(), String> {
    let dir = config.ensure_library_dir().map_err(|err| err.to_string())?;
    let library =
        Library::open(migration::database_path(&dir)).map_err(|err| err.to_string())?;

    // Keep the uid of a row that already exists, so its stored features stay
    // linked to it rather than being orphaned by a fresh one.
    if let Ok(Some(existing)) = library.get(&track.path) {
        track.uid = existing.uid;
        track.added_ts = existing.added_ts;
    }
    if track.added_ts == 0 {
        track.added_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
    }
    library.upsert(track).map_err(|err| err.to_string())?;

    let uid = track.require_uid().map_err(|err| err.to_string())?.to_string();
    library
        .replace_bpm_segments(
            &uid,
            &mixlyzer_core::linear::build_bpm_segments(analysis.tempo_segments()),
        )
        .map_err(|err| err.to_string())?;
    library
        .replace_key_segments(
            &uid,
            &mixlyzer_core::linear::build_key_segments(&analysis.key_segments),
        )
        .map_err(|err| err.to_string())?;

    let mut features = FeatureFile::new();
    features.set_beats_time_sec(analysis.beats());
    features.set_tempo_segments(analysis.tempo_segments());
    features.set_key_segments(&analysis.key_segments);
    features.set_phrases(&analysis.phrases);
    features.set_cue_points(&analysis.cue_points);
    FeatureStore::new(&dir)
        .save(&uid, &features)
        .map_err(|err| err.to_string())?;
    Ok(())
}

/// Read the library listing.
///
/// Every failure is reported rather than raised, so a damaged database leaves
/// the app usable with an empty list and a message.
pub fn load_library(config: &Config) -> Result<Vec<Track>, String> {
    let dir = config
        .ensure_library_dir()
        .map_err(|err| err.to_string())?;
    let library =
        Library::open(migration::database_path(&dir)).map_err(|err| err.to_string())?;
    library.list_all().map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_comes_back_as_a_failure_naming_it() {
        let request = AnalysisRequest {
            path: PathBuf::from("/nonexistent/track.flac"),
            config: Config::default(),
        };
        match run(&request, &pipeline::AnalysisOptions::default()) {
            AnalysisResult::Failed { path, message } => {
                assert_eq!(path, PathBuf::from("/nonexistent/track.flac"));
                assert!(!message.is_empty());
            }
            AnalysisResult::Ready(_) => panic!("a missing file cannot analyse"),
        }
    }

    #[test]
    fn an_unreachable_library_path_is_reported_not_raised() {
        let mut config = Config::default();
        config.libconfig.libpath = "/proc/nonexistent/deep/lib".into();
        let err = load_library(&config).unwrap_err();
        assert!(err.contains("/proc/nonexistent"), "got {err}");
    }
}
