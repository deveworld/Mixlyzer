//! Command-line front end to the Mixlyzer analyzer and library.
//!
//! The desktop app is a Qt program; this is the same analysis and the same
//! library, driven from a terminal and scriptable. Everything it prints comes
//! from the library crates, so a result here and a result in the app are the
//! same computation.

#![forbid(unsafe_code)]

mod args;
mod render;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mixlyzer_core::{Config, Track};
use mixlyzer_dsp::pipeline;
use mixlyzer_store::{migration, FeatureFile, FeatureStore, Library};

use args::{Args, Command};

fn main() -> std::process::ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match args::parse(&argv) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("mixlyzer: {err}\n\n{}", args::usage());
            return std::process::ExitCode::from(2);
        }
    };

    match run(parsed) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("mixlyzer: {err:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Help => {
            print!("{}", args::usage());
            Ok(())
        }
        Command::Version => {
            println!("mixlyzer {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Analyze { ref path, json } => analyze(&args, path, json),
        Command::Add { ref path, force } => add(&args, path, force),
        Command::List { ref order_by } => list(&args, order_by),
        Command::Export { ref path, ref out } => export(&args, path, out.as_deref()),
        Command::Migrate { dry_run } => migrate(&args, dry_run),
        Command::Transitions {
            from_bpm,
            to_bpm,
            tolerance,
        } => transitions(&args, from_bpm, to_bpm, tolerance),
    }
}

/// Load the config, reporting which file was at fault when it cannot be read.
fn load_config(args: &Args) -> Result<Config> {
    Config::load(&args.config)
        .with_context(|| format!("reading {}", args.config.display()))
}

/// The phrase detector options for this invocation.
///
/// An explicitly named model that cannot be read is an error: the user asked
/// for phrases by naming it. Without `--phrase-model` the weights are merely
/// looked for, and a build that does not ship them still analyses tempo and
/// key.
fn analysis_options(args: &Args) -> Result<pipeline::AnalysisOptions> {
    match &args.phrase_model {
        Some(path) => {
            let model = mixlyzer_dsp::PhraseModel::load(path)
                .with_context(|| format!("loading the phrase model {}", path.display()))?;
            Ok(pipeline::AnalysisOptions::with_phrase_model(model))
        }
        None => Ok(pipeline::AnalysisOptions::discovering_phrase_model()),
    }
}

/// The library directory: the `--library` override, or the configured path.
///
/// Creating it is a step that can fail on its own, so a bad path produces a
/// message naming it rather than aborting somewhere deeper.
fn library_dir(args: &Args, config: &Config) -> Result<PathBuf> {
    match &args.library {
        Some(path) => {
            std::fs::create_dir_all(path)
                .with_context(|| format!("creating library directory {}", path.display()))?;
            Ok(path.clone())
        }
        None => config
            .ensure_library_dir()
            .context("preparing the configured library directory"),
    }
}

fn open_library(args: &Args, config: &Config) -> Result<(Library, FeatureStore, PathBuf)> {
    let dir = library_dir(args, config)?;
    let library = Library::open(migration::database_path(&dir))
        .with_context(|| format!("opening the library in {}", dir.display()))?;
    let features = FeatureStore::new(&dir);
    Ok((library, features, dir))
}

fn analyze(args: &Args, path: &Path, json: bool) -> Result<()> {
    let config = load_config(args)?;
    let analysis = pipeline::analyze_file_with(path, &config.analysisconfig, &analysis_options(args)?)
        .with_context(|| format!("analysing {}", path.display()))?;
    if json {
        println!("{}", render::analysis_json(path, &analysis));
    } else {
        print!("{}", render::analysis_text(path, &analysis));
    }
    Ok(())
}

fn add(args: &Args, path: &Path, force: bool) -> Result<()> {
    let config = load_config(args)?;
    let (library, features, _dir) = open_library(args, &config)?;

    let normalized = mixlyzer_core::track::normalize_track_path(&path.to_string_lossy());
    if !force {
        if let Some(existing) = library.get(&normalized)? {
            println!(
                "already in the library: {} ({})",
                render::display_title(&existing),
                existing.uid.as_deref().unwrap_or("no uid")
            );
            println!("pass --force to analyse it again");
            return Ok(());
        }
    }

    let analysis = pipeline::analyze_file_with(path, &config.analysisconfig, &analysis_options(args)?)
        .with_context(|| format!("analysing {}", path.display()))?;

    // Keep the uid of an existing row so its stored features stay linked.
    let mut track = match library.get(&normalized)? {
        Some(existing) => existing,
        None => Track::new(&path.to_string_lossy()),
    };
    track.duration = Some(analysis.duration_sec);
    track.bpm = Some(analysis.tempo_global);
    track.key = analysis.overall_key;
    if track.title.is_empty() {
        track.title = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
    }
    if track.added_ts == 0 {
        track.added_ts = now_epoch_secs();
    }
    if let Ok(meta) = std::fs::metadata(path) {
        track.file_size = meta.len() as i64;
    }

    library.upsert(&track)?;
    let uid = track
        .require_uid()
        .context("the track row has no uid to link its features to")?
        .to_string();

    library.replace_bpm_segments(
        &uid,
        &mixlyzer_core::linear::build_bpm_segments(analysis.tempo_segments()),
    )?;
    library.replace_key_segments(
        &uid,
        &mixlyzer_core::linear::build_key_segments(&analysis.key_segments),
    )?;

    let mut file = FeatureFile::new();
    file.set_beats_time_sec(analysis.beats());
    file.set_tempo_segments(analysis.tempo_segments());
    file.set_key_segments(&analysis.key_segments);
    file.set_phrases(&analysis.phrases);
    file.set_cue_points(&analysis.cue_points);
    features
        .save(&uid, &file)
        .context("writing the analysis features")?;

    println!("added {}", render::display_title(&track));
    print!("{}", render::analysis_text(path, &analysis));
    Ok(())
}

fn list(args: &Args, order_by: &str) -> Result<()> {
    let config = load_config(args)?;
    let (library, _features, dir) = open_library(args, &config)?;
    let tracks = library.list_ordered(order_by, None, 0)?;
    if tracks.is_empty() {
        println!("no tracks in {}", dir.display());
        return Ok(());
    }
    print!("{}", render::track_table(&tracks));
    Ok(())
}

fn export(args: &Args, path: &Path, out: Option<&Path>) -> Result<()> {
    let config = load_config(args)?;
    let (library, features, _dir) = open_library(args, &config)?;

    let normalized = mixlyzer_core::track::normalize_track_path(&path.to_string_lossy());
    let track = library
        .get(&normalized)?
        .with_context(|| format!("{} is not in the library; add it first", path.display()))?;
    let uid = track.require_uid()?.to_string();

    let stored = features
        .load_optional(&uid)?
        .with_context(|| format!("no stored analysis for {}", render::display_title(&track)))?;

    let options = mixlyzer_export::ExportOptions {
        audio_path: path.canonicalize().ok(),
        file_size: std::fs::metadata(path).ok().map(|m| m.len()),
        ..Default::default()
    };
    let document = mixlyzer_export::build_rekordbox_xml(
        &track,
        &stored.tempo_segments(),
        &[],
        &stored.cue_points(),
        &options,
    )?;

    let target = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(&document.filename));
    std::fs::write(&target, &document.document)
        .with_context(|| format!("writing {}", target.display()))?;
    println!("wrote {}", target.display());
    Ok(())
}

fn migrate(args: &Args, dry_run: bool) -> Result<()> {
    let config = load_config(args)?;
    let dir = library_dir(args, &config)?;
    let current = migration::effective_library_version(&dir)?;
    let target = mixlyzer_core::LIBRARY_VERSION;

    if current == target {
        println!("library in {} is already at {target}", dir.display());
        return Ok(());
    }

    let steps = migration::plan(&current, target)?;
    println!("{} -> {target}, {} step(s):", current, steps.len());
    for step in &steps {
        println!("  {} -> {}", step.from_version(), step.to_version());
    }
    if dry_run {
        println!("dry run; nothing was changed");
        return Ok(());
    }

    let report = migration::run(&dir)?;
    println!("converted {} track(s)", report.tracks_converted());
    // Per-track problems are reported, not treated as a reason to fail the
    // whole migration and lock the user out of their library.
    let skipped: Vec<_> = report.skipped().collect();
    if !skipped.is_empty() {
        println!("{} track(s) skipped:", skipped.len());
        for item in skipped {
            println!("  {item:?}");
        }
    }
    println!("library is now at {target}");
    Ok(())
}

fn transitions(args: &Args, from_bpm: f64, to_bpm: f64, tolerance: f64) -> Result<()> {
    let config = load_config(args)?;
    let (library, _features, _dir) = open_library(args, &config)?;
    let found = library.search_bpm_transitions(from_bpm, to_bpm, tolerance, 4.0, false)?;
    if found.is_empty() {
        println!("no track moves from {from_bpm} to {to_bpm} BPM (±{tolerance}%)");
        return Ok(());
    }
    print!("{}", render::transition_table(&found));
    Ok(())
}

/// Seconds since the epoch, or zero if the clock is before it.
fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
