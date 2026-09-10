//! JumpCUE detection: finding the stretches of a track that sound alike.
//!
//! A DJ jumping from one part of a track to another wants the mix to keep
//! making sense, so the two parts have to sound alike. This module looks for
//! those pairs and reports each one as a [`JumpCue`] the player can jump to.
//!
//! The chain is the usual self-similarity one:
//!
//! 1. a beat-synchronous mel feature matrix, four beats to a column, so that
//!    two positions are compared as musical phrases rather than as instants;
//! 2. a cosine self-similarity matrix over those columns, smoothed and with
//!    its main diagonal suppressed (every passage is trivially like itself);
//! 3. the mean of each diagonal, which turns the matrix into similarity
//!    against repeat distance — a curve whose peaks are candidate lags;
//! 4. for each candidate lag, the contiguous run over which the two positions
//!    stay alike, which is the pair of regions;
//! 5. the most alike beat inside that run, which is where the jump lands.
//!
//! # Differences from the Python original
//!
//! The Python `self_correlation` module decimates its feature matrix on long
//! tracks but then indexes the decimated matrix with full-resolution beat
//! numbers, so on a long DJ mix every cue lands at a fraction of its true
//! position. Here the matrix owns the mapping — [`BeatMatrix::first_beat`] and
//! the times derived from it — and nothing outside it ever guesses at the
//! relationship between a column and a beat.
//!
//! Python also treats degenerate input as an error — its early exits return a
//! bare report where the caller unpacks a tuple, so silence raises `TypeError`
//! and takes the whole track analysis down with it. A track with no repeats,
//! no beats, or no sound at all is an ordinary empty result here.

use mixlyzer_core::jumpcue::{label_from_index, merge_coincident, JumpCue, JumpCueGraph};

use crate::error::AnalysisError;
use crate::onset::{mel_filterbank, mel_power_spectrogram};

/// How JumpCUE detection is tuned.
#[derive(Debug, Clone, Copy)]
pub struct JumpCueOptions {
    /// Mel bands in the beat-synchronous feature matrix.
    pub n_mels: usize,
    /// Lowest mel band edge, in Hz.
    pub mel_fmin: f64,
    /// Highest mel band edge, in Hz. `None` means Nyquist.
    pub mel_fmax: Option<f64>,
    /// Beats stacked into one feature column. One beat alone is too short to
    /// tell a phrase apart from any other phrase over the same drum pattern.
    pub n_beats_seg: usize,
    /// Cap on feature columns. The similarity matrix is O(columns²), so a
    /// two-hour mix has to be decimated to stay within reach.
    pub max_cols: usize,
    /// Gaussian smoothing applied to the similarity matrix, in columns.
    pub smooth_sigma: f64,
    /// Columns either side of the main diagonal that are suppressed.
    pub diag_band: usize,
    /// Lags nearer than this to either end of the profile are ignored: a few
    /// beats apart is continuation, not a repeat.
    pub margin_beats: f64,
    /// How far a lag peak must stand above its surroundings, `0..=1`.
    pub peak_prominence: f64,
    /// Most candidate lags to follow up.
    pub max_peaks: usize,
    /// Floor on the cosine similarity a pair of regions must hold to count as
    /// alike. The threshold actually used also rises with how self-similar the
    /// track is overall; see [`baseline_similarity`].
    pub min_score: f64,
    /// Shortest region worth jumping to.
    pub min_duration_sec: f64,
    /// Most links to keep, best-scoring first.
    pub max_pairs: usize,
    /// Softmax temperature turning link scores into confidences.
    pub score_temperature: f64,
}

impl Default for JumpCueOptions {
    fn default() -> Self {
        Self {
            n_mels: 128,
            mel_fmin: 30.0,
            mel_fmax: None,
            n_beats_seg: 4,
            max_cols: 2000,
            smooth_sigma: 1.0,
            diag_band: 2,
            margin_beats: 16.0,
            peak_prominence: 0.09,
            max_peaks: 6,
            min_score: 0.3,
            min_duration_sec: 6.0,
            max_pairs: 4,
            score_temperature: 0.35,
        }
    }
}

/// One end of a link: a stretch of track, and the instant to jump to inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub start: f64,
    pub end: f64,
    /// The beat inside `[start, end]` where the two regions are most alike.
    pub point: f64,
}

impl Region {
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    /// Seconds these two regions have in common.
    fn overlap(&self, other: &Region) -> f64 {
        (self.end.min(other.end) - self.start.max(other.start)).max(0.0)
    }

    /// Whether these describe the same stretch of track.
    ///
    /// Two lag peaks — a repeat and its double, say — routinely land on the
    /// same passage. Python emitted each as its own labelled cue, splitting one
    /// reachable group into fragments that the DJ cannot jump between.
    fn is_same_as(&self, other: &Region) -> bool {
        let shortest = self.duration().min(other.duration());
        shortest > 0.0 && self.overlap(other) > 0.5 * shortest
    }
}

/// Two regions of a track that sound alike.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarLink {
    /// The earlier of the two regions.
    pub earlier: Region,
    /// The later of the two regions.
    pub later: Region,
    /// Repeat distance, in beats.
    pub lag_beats: f64,
    /// Repeat distance, in seconds.
    pub lag_sec: f64,
    /// Mean cosine similarity over the overlapping run, `-1..=1`.
    pub score: f64,
    /// This link's share of the emitted links' scores, `0..=1`.
    ///
    /// Computed over the links actually returned. Python computed it over
    /// every candidate and then truncated the list, so what it emitted summed
    /// to whatever was left over.
    pub confidence: f64,
}

/// Find the pairs of similar regions in a track.
///
/// `beats` are beat times in seconds, ascending. Degenerate input — no beats,
/// silence, a track with nothing that repeats — is an empty graph, not an
/// error; the `Result` is here because detection shares the pipeline's error
/// type and may grow failure modes.
pub fn detect(
    samples: &[f32],
    sample_rate: u32,
    beats: &[f64],
    options: JumpCueOptions,
) -> Result<JumpCueGraph, AnalysisError> {
    let links = detect_links(samples, sample_rate, beats, options)?;
    Ok(graph_from_links(&links))
}

/// The links behind [`detect`], with their scores and confidences.
pub fn detect_links(
    samples: &[f32],
    sample_rate: u32,
    beats: &[f64],
    options: JumpCueOptions,
) -> Result<Vec<SimilarLink>, AnalysisError> {
    let Some(matrix) = beat_matrix(samples, sample_rate, beats, &options) else {
        return Ok(Vec::new());
    };

    let ssm = self_similarity(&matrix, &options);
    let profile = lag_profile(&ssm);
    let min_overlap = matrix.columns_spanning(options.min_duration_sec);
    let peaks = pick_peaks(&profile, &matrix, min_overlap, &options);
    let baseline = baseline_similarity(&matrix);

    let mut links: Vec<SimilarLink> = Vec::new();
    for lag in peaks {
        if let Some(link) = link_at_lag(&matrix, lag, min_overlap, baseline, &options) {
            links.push(link);
        }
    }

    links.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // A repeat and its double describe the same pair of regions; keep the
    // better-scoring one so the two do not become two cues on one spot.
    let mut kept: Vec<SimilarLink> = Vec::new();
    for link in links {
        if kept.len() >= options.max_pairs {
            break;
        }
        let duplicate = kept.iter().any(|other| {
            link.later.is_same_as(&other.later) && link.earlier.is_same_as(&other.earlier)
        });
        if !duplicate {
            kept.push(link);
        }
    }

    let scores: Vec<f64> = kept.iter().map(|link| link.score).collect();
    for (link, confidence) in kept
        .iter_mut()
        .zip(softmax_confidence(&scores, options.score_temperature))
    {
        link.confidence = confidence;
    }
    Ok(kept)
}

/// Turn link scores into a distribution over the links that were emitted.
fn softmax_confidence(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let temperature = temperature.max(1e-6);
    let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores
        .iter()
        .map(|score| ((score.max(0.0) - top.max(0.0)) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / scores.len() as f64; scores.len()];
    }
    weights.into_iter().map(|w| w / total).collect()
}

/// The beat-synchronous feature matrix, and the mapping back to beats.
///
/// Each column is `span` consecutive beats of whitened log-mel energy, stacked
/// into one vector and normalised to unit length so that similarity is a plain
/// dot product. Columns are `step` beats apart: `step` is 1 for an ordinary
/// track and larger for one long enough that a column-per-beat matrix would
/// not fit, which is why every conversion between a column and a time goes
/// through this type rather than through an assumed beat number.
#[derive(Debug, Clone)]
struct BeatMatrix {
    columns: Vec<Vec<f64>>,
    /// Beats between one column and the next.
    step: usize,
    /// Beats stacked into each column.
    span: usize,
    beats: Vec<f64>,
    avg_beat_sec: f64,
}

impl BeatMatrix {
    fn len(&self) -> usize {
        self.columns.len()
    }

    /// The first beat that `column` covers.
    fn first_beat(&self, column: usize) -> usize {
        column * self.step
    }

    /// Time of a beat, extrapolating past the last one at the final spacing.
    fn beat_time(&self, beat: usize) -> f64 {
        match self.beats.len() {
            0 => 0.0,
            1 => self.beats[0],
            n if beat < n => self.beats[beat],
            n => {
                let gap = self.beats[n - 1] - self.beats[n - 2];
                self.beats[n - 1] + gap * (beat - (n - 1)) as f64
            }
        }
    }

    /// When `column` starts.
    fn start_time(&self, column: usize) -> f64 {
        self.beat_time(self.first_beat(column))
    }

    /// When `column` ends: the end of the last beat it stacks.
    fn end_time(&self, column: usize) -> f64 {
        self.beat_time(self.first_beat(column) + self.span)
    }

    /// Cosine similarity of two columns.
    fn similarity(&self, a: usize, b: usize) -> f64 {
        self.columns[a]
            .iter()
            .zip(&self.columns[b])
            .map(|(x, y)| x * y)
            .sum()
    }

    /// How many columns cover `seconds` of track.
    fn columns_spanning(&self, seconds: f64) -> usize {
        let per_column = self.avg_beat_sec * self.step as f64;
        if per_column <= 0.0 {
            return 1;
        }
        ((seconds / per_column).ceil() as usize).max(1)
    }
}

/// Build the beat-synchronous feature matrix, or `None` if there is nothing
/// to build one from: too few beats, no audio, or no sound in the audio.
fn beat_matrix(
    samples: &[f32],
    sample_rate: u32,
    beats: &[f64],
    options: &JumpCueOptions,
) -> Option<BeatMatrix> {
    if sample_rate == 0 || beats.len() < 2 || samples.is_empty() {
        return None;
    }
    let rate = f64::from(sample_rate);
    let beat_count = beats.len() - 1;

    // Window the beats themselves: a frame much longer than a beat would smear
    // neighbouring beats together and a much shorter one would resolve nothing.
    let mut lengths: Vec<f64> = beats.windows(2).map(|w| (w[1] - w[0]) * rate).collect();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_len = lengths[lengths.len() / 2].max(1.0);
    let n_fft = (median_len * 0.5).clamp(512.0, 4096.0) as usize;
    let n_fft = n_fft.max(32).next_power_of_two();
    let hop = (n_fft / 4).clamp(128, 1024);

    let n_mels = options.n_mels.max(8);
    let filters = mel_filterbank(
        n_mels,
        n_fft,
        rate,
        options.mel_fmin,
        options.mel_fmax.unwrap_or(rate * 0.5),
    );
    let power = mel_power_spectrogram(samples, n_fft, hop, &filters);
    if power.is_empty() {
        return None;
    }

    // A frame is centred half a window after it starts, so that is where its
    // energy belongs on the timeline.
    let frame_of = |time: f64| -> usize {
        let centred = (time * rate - n_fft as f64 * 0.5) / hop as f64;
        (centred.round().max(0.0) as usize).min(power.len() - 1)
    };

    // Median rather than mean over the beat: a stray transient inside the beat
    // should not decide what the beat sounds like.
    let mut beat_mel = vec![vec![0.0f64; beat_count]; n_mels];
    let mut scratch: Vec<f64> = Vec::new();
    for beat in 0..beat_count {
        let from = frame_of(beats[beat]);
        let to = frame_of(beats[beat + 1]).max(from + 1).min(power.len());
        for (band, row) in beat_mel.iter_mut().enumerate() {
            scratch.clear();
            scratch.extend(power[from..to].iter().map(|frame| frame[band]));
            scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row[beat] = scratch[scratch.len() / 2];
        }
    }
    if beat_mel
        .iter()
        .all(|row| row.iter().all(|value| *value < 1e-12))
    {
        // Digital silence. Python discovers this as a `TypeError` from an
        // early return of the wrong shape; there is simply nothing to link.
        return None;
    }

    // Log-compress, then whiten each band across time so that a band which is
    // always loud contributes no more to similarity than a quiet one.
    for row in beat_mel.iter_mut() {
        for value in row.iter_mut() {
            *value = value.max(0.0).ln_1p();
        }
        let mean = row.iter().sum::<f64>() / beat_count as f64;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / beat_count as f64;
        let scale = variance.sqrt() + 1e-12;
        for value in row.iter_mut() {
            *value = (*value - mean) / scale;
        }
    }

    let span = options.n_beats_seg.max(1).min(beat_count);
    let full_cols = beat_count - span + 1;
    // The similarity matrix is O(columns²); past the cap, sample the columns.
    let step = (full_cols / options.max_cols.max(1)).max(1);

    let mut columns = Vec::with_capacity(full_cols.div_ceil(step));
    let mut first = 0usize;
    while first < full_cols {
        let mut column = Vec::with_capacity(n_mels * span);
        for offset in 0..span {
            for row in &beat_mel {
                column.push(row[first + offset]);
            }
        }
        let norm = column.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-9 {
            for value in column.iter_mut() {
                *value /= norm;
            }
        }
        columns.push(column);
        first += step;
    }

    let avg_beat_sec = (beats[beats.len() - 1] - beats[0]) / beat_count as f64;
    Some(BeatMatrix {
        columns,
        step,
        span,
        beats: beats.to_vec(),
        avg_beat_sec,
    })
}

/// A self-similarity matrix: every column compared with every other.
#[derive(Debug, Clone)]
struct Ssm {
    values: Vec<f64>,
    size: usize,
}

impl Ssm {
    fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * self.size + column]
    }

    /// Mean of the `lag`-th diagonal: how alike positions that far apart are.
    ///
    /// Only the non-negative lags are needed. Cosine similarity is symmetric
    /// and so is everything done to it here, so the Python original's fold of
    /// `+k` onto `-k` averages each diagonal with itself.
    fn diagonal_mean(&self, lag: usize) -> f64 {
        if lag >= self.size {
            return 0.0;
        }
        let count = self.size - lag;
        let total: f64 = (0..count).map(|i| self.get(i + lag, i)).sum();
        total / count as f64
    }
}

/// Cosine similarity between every pair of columns, smoothed, with the main
/// diagonal band removed.
fn self_similarity(matrix: &BeatMatrix, options: &JumpCueOptions) -> Ssm {
    let size = matrix.len();
    let mut values = vec![0.0f64; size * size];
    for row in 0..size {
        for column in row..size {
            let similarity = matrix.similarity(row, column);
            values[row * size + column] = similarity;
            values[column * size + row] = similarity;
        }
    }
    let mut ssm = Ssm { values, size };

    if options.smooth_sigma > 0.0 {
        // Smoothing along both axes widens the diagonal stripes a repeat makes,
        // so a repeat that drifts by a beat still reads as one ridge.
        let kernel = gaussian_kernel(options.smooth_sigma);
        smooth_rows(&mut ssm, &kernel);
        transpose(&mut ssm);
        smooth_rows(&mut ssm, &kernel);
        transpose(&mut ssm);
    }

    // A passage is trivially similar to itself and to its immediate
    // neighbours; leaving that in would drown every real repeat.
    for row in 0..size {
        let lo = row.saturating_sub(options.diag_band);
        let hi = (row + options.diag_band).min(size.saturating_sub(1));
        for column in lo..=hi {
            ssm.values[row * size + column] = 0.0;
        }
    }
    ssm
}

/// Normalised Gaussian weights out to three standard deviations.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil().max(1.0) as usize;
    let mut kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-0.5 * x * x / (sigma * sigma)).exp()
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    for weight in kernel.iter_mut() {
        *weight /= total;
    }
    kernel
}

/// Convolve each row with `kernel`, holding the edge value beyond the ends.
fn smooth_rows(ssm: &mut Ssm, kernel: &[f64]) {
    let size = ssm.size;
    if size == 0 {
        return;
    }
    let radius = kernel.len() / 2;
    let mut row_buffer = vec![0.0f64; size];
    for row in 0..size {
        let base = row * size;
        row_buffer.copy_from_slice(&ssm.values[base..base + size]);
        for column in 0..size {
            let mut total = 0.0;
            for (offset, weight) in kernel.iter().enumerate() {
                let at = (column + offset).saturating_sub(radius).min(size - 1);
                total += row_buffer[at] * weight;
            }
            ssm.values[base + column] = total;
        }
    }
}

fn transpose(ssm: &mut Ssm) {
    let size = ssm.size;
    for row in 0..size {
        for column in row + 1..size {
            ssm.values.swap(row * size + column, column * size + row);
        }
    }
}

/// Similarity against repeat distance, with its slow trend removed and scaled
/// so the strongest lag reads 1.
///
/// The trend removal matters because nearby positions in a track are alike for
/// reasons that have nothing to do with repetition — the same instruments, the
/// same key — and that baseline falls off smoothly with distance. What is left
/// after subtracting it is the part that only a repeat explains.
fn lag_profile(ssm: &Ssm) -> Vec<f64> {
    let size = ssm.size;
    if size == 0 {
        return Vec::new();
    }
    let raw: Vec<f64> = (0..size).map(|lag| ssm.diagonal_mean(lag)).collect();
    let window = ((size / 50) * 2 + 1).max(5);
    let radius = window / 2;
    let mut residual = Vec::with_capacity(size);
    for lag in 0..size {
        let mut total = 0.0;
        for offset in 0..window {
            let at = (lag + offset).saturating_sub(radius).min(size - 1);
            total += raw[at];
        }
        residual.push((raw[lag] - total / window as f64).max(0.0));
    }
    let peak = residual.iter().cloned().fold(0.0f64, f64::max);
    if peak > 0.0 {
        for value in residual.iter_mut() {
            *value /= peak;
        }
    }
    residual
}

/// Height of `peak` above the higher of the two valleys that flank it.
fn prominence(profile: &[f64], peak: usize) -> f64 {
    let height = profile[peak];
    let mut left = height;
    for value in profile[..peak].iter().rev() {
        if *value > height {
            break;
        }
        left = left.min(*value);
    }
    let mut right = height;
    for value in &profile[peak + 1..] {
        if *value > height {
            break;
        }
        right = right.min(*value);
    }
    height - left.max(right)
}

/// The candidate repeat distances, in columns, strongest first.
fn pick_peaks(
    profile: &[f64],
    matrix: &BeatMatrix,
    min_overlap: usize,
    options: &JumpCueOptions,
) -> Vec<usize> {
    let size = profile.len();
    if size < 3 {
        return Vec::new();
    }
    let longest_lag_beats = matrix.first_beat(size - 1) as f64;
    // A diagonal shorter than the shortest link we would emit is averaged over
    // a handful of cells: noise, not evidence. Python bounded this only by
    // `margin_beats`, which on a decimated matrix is a few columns wide.
    let usable = min_overlap.max(2);

    let mut candidates: Vec<usize> = Vec::new();
    for lag in 1..size - 1 {
        let lag_beats = matrix.first_beat(lag) as f64;
        if lag_beats < options.margin_beats
            || lag_beats > longest_lag_beats - options.margin_beats
            || size - lag < usable
        {
            continue;
        }
        let is_local_max = profile[lag] > profile[lag - 1] && profile[lag] >= profile[lag + 1];
        if is_local_max && prominence(profile, lag) >= options.peak_prominence {
            candidates.push(lag);
        }
    }

    candidates.sort_by(|a, b| {
        profile[*b]
            .partial_cmp(&profile[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let separation = (size / 50).max(1);
    let mut chosen: Vec<usize> = Vec::new();
    for lag in candidates {
        if chosen.len() >= options.max_peaks {
            break;
        }
        if chosen
            .iter()
            .all(|other| lag.abs_diff(*other) >= separation)
        {
            chosen.push(lag);
        }
    }
    chosen
}

/// How alike two unrelated parts of *this* track are.
///
/// A track played on one drum kit from end to end is similar to itself
/// everywhere, and a fixed similarity threshold would either accept all of it
/// or none of it. The median over a coarse grid of well-separated column pairs
/// is a robust estimate of that floor: a repeat covers too little of the grid
/// to move it.
fn baseline_similarity(matrix: &BeatMatrix) -> f64 {
    let size = matrix.len();
    let stride = (size / 40).max(1);
    let separation = matrix.span.max(1);
    let mut sampled: Vec<f64> = Vec::new();
    for row in (0..size).step_by(stride) {
        for column in (0..size).step_by(stride) {
            if row.abs_diff(column) > separation {
                sampled.push(matrix.similarity(row, column));
            }
        }
    }
    if sampled.is_empty() {
        return 0.0;
    }
    sampled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sampled[sampled.len() / 2]
}

/// Where between the track's own baseline and a perfect match two regions
/// count as alike.
const ALIKE_ABOVE_BASELINE: f64 = 0.65;

/// The pair of regions a repeat distance describes, if it holds up.
///
/// The similarity is recomputed from the features rather than read off the
/// smoothed matrix: smoothing is what finds *where* a repeat is, but it also
/// pulls the values towards their neighbours, and the score has to say how
/// alike the two regions really are.
fn link_at_lag(
    matrix: &BeatMatrix,
    lag: usize,
    min_overlap: usize,
    baseline: f64,
    options: &JumpCueOptions,
) -> Option<SimilarLink> {
    let size = matrix.len();
    if lag == 0 || lag >= size {
        return None;
    }
    let mut curve: Vec<f64> = (lag..size).map(|i| matrix.similarity(i, i - lag)).collect();
    smooth_curve(&mut curve);
    let threshold = options
        .min_score
        .max(baseline + ALIKE_ABOVE_BASELINE * (1.0 - baseline));

    // The run over which the two positions stay alike is the overlap.
    let (mut best_start, mut best_len) = (0usize, 0usize);
    let mut run_start = None;
    for (index, value) in curve.iter().enumerate() {
        if *value >= threshold {
            run_start.get_or_insert(index);
        } else if let Some(start) = run_start.take() {
            if index - start > best_len {
                (best_start, best_len) = (start, index - start);
            }
        }
    }
    if let Some(start) = run_start {
        if curve.len() - start > best_len {
            (best_start, best_len) = (start, curve.len() - start);
        }
    }
    if best_len < min_overlap.max(1) {
        return None;
    }

    let run = &curve[best_start..best_start + best_len];
    let score = run.iter().sum::<f64>() / best_len as f64;
    // Where the two are most alike is where the jump should land.
    let peak = run
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| best_start + index)?;

    let (first, last, jump) = (
        best_start + lag,
        best_start + best_len - 1 + lag,
        peak + lag,
    );
    let later = Region {
        start: matrix.start_time(first),
        end: matrix.end_time(last),
        point: matrix.start_time(jump),
    };
    let earlier = Region {
        start: matrix.start_time(first - lag),
        end: matrix.end_time(last - lag),
        point: matrix.start_time(jump - lag),
    };
    if later.duration() < options.min_duration_sec || earlier.duration() < options.min_duration_sec
    {
        return None;
    }
    // A run longer than the lag itself means the two regions are largely the
    // same stretch of track — a loop, not somewhere to jump to.
    if later.is_same_as(&earlier) {
        return None;
    }

    Some(SimilarLink {
        lag_beats: matrix.first_beat(lag) as f64,
        lag_sec: later.point - earlier.point,
        score,
        confidence: 0.0,
        earlier,
        later,
    })
}

/// Three-point moving average, so that one dull beat does not split a run.
fn smooth_curve(curve: &mut [f64]) {
    if curve.len() < 3 {
        return;
    }
    let original = curve.to_vec();
    for (index, value) in curve.iter_mut().enumerate() {
        let lo = index.saturating_sub(1);
        let hi = (index + 1).min(original.len() - 1);
        *value = original[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64;
    }
}

/// Turn links into labelled cues grouped into reachable components.
///
/// Two links that name the same region — a repeat and the same repeat found
/// again at twice the distance, say — share one cue here. Python labelled each
/// end of each link separately, so one group of three mutually reachable
/// regions came out as two unrelated pairs and the DJ lost the jump between
/// them.
fn graph_from_links(links: &[SimilarLink]) -> JumpCueGraph {
    let mut nodes: Vec<Region> = Vec::new();
    let mut confidence: Vec<f64> = Vec::new();
    let mut parent: Vec<usize> = Vec::new();

    for link in links {
        let earlier = intern_region(
            &mut nodes,
            &mut confidence,
            &mut parent,
            link.earlier,
            link.confidence,
        );
        let later = intern_region(
            &mut nodes,
            &mut confidence,
            &mut parent,
            link.later,
            link.confidence,
        );
        union(&mut parent, earlier, later);
    }

    // One pass of the union-find, then the groups are just equal root numbers.
    let roots: Vec<usize> = (0..nodes.len())
        .map(|node| find(&mut parent, node))
        .collect();

    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by(|a, b| {
        nodes[*a]
            .point
            .partial_cmp(&nodes[*b].point)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Components are numbered in the order their first cue appears, so the
    // colours run down the track rather than in detection order.
    let mut components: Vec<(usize, usize)> = Vec::new();
    let mut labels: Vec<(usize, String)> = Vec::new();
    for (position, node) in order.iter().enumerate() {
        let Some(label) = label_from_index(position) else {
            break; // labels are single letters and key the storage format
        };
        let root = roots[*node];
        if !components.iter().any(|(seen, _)| *seen == root) {
            components.push((root, components.len()));
        }
        labels.push((*node, label));
    }

    let mut cues: Vec<JumpCue> = Vec::new();
    for (id, (node, label)) in labels.iter().enumerate() {
        let root = roots[*node];
        let component = components
            .iter()
            .find(|(seen, _)| *seen == root)
            .map_or(0, |(_, index)| *index);
        let peers: Vec<&str> = labels
            .iter()
            .filter(|(other, _)| *other != *node && roots[*other] == root)
            .map(|(_, other_label)| other_label.as_str())
            .collect();
        let comment = if peers.is_empty() {
            "jump cue".to_string()
        } else {
            format!(
                "jump to {} ({:.0}% confidence)",
                peers.join(", "),
                confidence[*node] * 100.0
            )
        };
        let region = nodes[*node];
        let mut cue = JumpCue::new(
            id,
            label.clone(),
            region.start,
            region.end,
            region.point,
            component,
        );
        cue.comment = comment;
        cues.push(cue);
    }

    JumpCueGraph::new(merge_coincident(&cues))
}

/// Find the node for `region`, adding one if no existing node is that region.
fn intern_region(
    nodes: &mut Vec<Region>,
    confidence: &mut Vec<f64>,
    parent: &mut Vec<usize>,
    region: Region,
    link_confidence: f64,
) -> usize {
    if let Some(index) = nodes.iter().position(|node| node.is_same_as(&region)) {
        confidence[index] = confidence[index].max(link_confidence);
        return index;
    }
    nodes.push(region);
    confidence.push(link_confidence);
    parent.push(nodes.len() - 1);
    nodes.len() - 1
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (root_a, root_b) = (find(parent, a), find(parent, b));
    if root_a != root_b {
        parent[root_b] = root_a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 8_000;
    const BEAT_SEC: f64 = 0.5;

    /// A pitch that depends only on `index`, so a pattern is reproducible and
    /// no two neighbouring beats sound alike.
    fn pitch(index: usize) -> f64 {
        // Hashed rather than cyclic: a sequence that repeated every so many
        // beats would put repeats in the material the test says has none.
        let mixed = (index as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9;
        110.0 * 2f64.powf((mixed >> 11) as f64 % 36.0 / 12.0)
    }

    /// A track that plays one pitched, plucked note per beat.
    ///
    /// Returns the samples and the beat times, so detection is exercised on
    /// its own rather than on whatever the tempo stage happened to find.
    fn track(pitches: &[f64], beat_sec: f64, rate: u32) -> (Vec<f32>, Vec<f64>) {
        let period = (beat_sec * f64::from(rate)) as usize;
        let mut samples = vec![0.0f32; pitches.len() * period];
        for (beat, hz) in pitches.iter().enumerate() {
            for offset in 0..period {
                let t = offset as f64 / f64::from(rate);
                let decay = (-6.0 * t / beat_sec).exp();
                let tone = (2.0 * std::f64::consts::PI * hz * t).sin() * decay;
                // A short broadband click gives the beat a drum-like transient.
                let click = if offset < 24 {
                    1.0 - offset as f64 / 24.0
                } else {
                    0.0
                };
                samples[beat * period + offset] = (0.7 * tone + 0.5 * click) as f32;
            }
        }
        let beats = (0..=pitches.len()).map(|i| i as f64 * beat_sec).collect();
        (samples, beats)
    }

    /// Eight bars of A, eight of B, then A again.
    fn a_b_a() -> (Vec<f32>, Vec<f64>) {
        let section_a: Vec<f64> = (0..32).map(pitch).collect();
        let section_b: Vec<f64> = (0..32).map(|i| pitch(i + 5) * 2.0).collect();
        let pitches: Vec<f64> = section_a
            .iter()
            .chain(&section_b)
            .chain(&section_a)
            .copied()
            .collect();
        track(&pitches, BEAT_SEC, RATE)
    }

    fn region(start: f64, end: f64) -> Region {
        Region {
            start,
            end,
            point: start,
        }
    }

    fn link(earlier: Region, later: Region, score: f64) -> SimilarLink {
        SimilarLink {
            earlier,
            later,
            lag_beats: 0.0,
            lag_sec: later.point - earlier.point,
            score,
            confidence: 0.0,
        }
    }

    #[test]
    fn a_section_that_returns_is_found_at_both_its_positions() {
        let (samples, beats) = a_b_a();
        let links = detect_links(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
        assert_eq!(links.len(), 1, "expected the one A-to-A repeat: {links:#?}");

        let found = &links[0];
        // A runs 0-16s and again 32-48s.
        assert!(
            (found.earlier.start - 0.0).abs() < 2.0 && (found.earlier.end - 16.0).abs() < 2.0,
            "earlier region {:?} is not the first A",
            found.earlier
        );
        assert!(
            (found.later.start - 32.0).abs() < 2.0 && (found.later.end - 48.0).abs() < 2.0,
            "later region {:?} is not the returning A",
            found.later
        );
        assert!((found.lag_sec - 32.0).abs() < 2.0, "lag {}s", found.lag_sec);
        assert!(
            (found.lag_beats - 64.0).abs() < 4.0,
            "lag {} beats",
            found.lag_beats
        );
        assert!(
            found.score > 0.8,
            "a literal repeat should score high, got {}",
            found.score
        );
    }

    #[test]
    fn the_jump_point_lies_inside_its_own_region() {
        let (samples, beats) = a_b_a();
        let graph = detect(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
        assert!(!graph.is_empty());
        for cue in graph.cues() {
            assert!(
                cue.point >= cue.start && cue.point <= cue.end,
                "jump point {} is outside [{}, {}]",
                cue.point,
                cue.start,
                cue.end
            );
        }
    }

    #[test]
    fn the_two_ends_of_a_repeat_share_one_component() {
        let (samples, beats) = a_b_a();
        let graph = detect(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
        assert_eq!(graph.cues().len(), 2);
        assert_eq!(graph.components().len(), 1, "both ends must be reachable");
        assert!(!graph.links().is_empty());
    }

    #[test]
    fn labels_are_unique_single_letters() {
        let (samples, beats) = a_b_a();
        let graph = detect(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
        assert!(graph.validate_labels().is_ok());
        let labels: Vec<&str> = graph.cues().iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, vec!["A", "B"], "labels run in track order");
    }

    #[test]
    fn a_track_with_no_repeats_yields_an_empty_graph() {
        // A rising sequence: every beat differs from every other.
        let pitches: Vec<f64> = (0..96)
            .map(|i| 110.0 * 2f64.powf(i as f64 / 32.0))
            .collect();
        let (samples, beats) = track(&pitches, BEAT_SEC, RATE);
        let graph = detect(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
        assert!(
            graph.is_empty(),
            "found cues in a track with no repeats: {graph:#?}"
        );
    }

    #[test]
    fn silence_yields_an_empty_graph_rather_than_an_error() {
        let samples = vec![0.0f32; RATE as usize * 48];
        let beats: Vec<f64> = (0..96).map(|i| i as f64 * BEAT_SEC).collect();
        let graph = detect(&samples, RATE, &beats, JumpCueOptions::default())
            .expect("digital silence is an empty result, not a failure");
        assert!(graph.is_empty());
    }

    #[test]
    fn a_track_with_no_beats_yields_an_empty_graph() {
        let (samples, _) = a_b_a();
        for beats in [vec![], vec![0.0]] {
            let graph = detect(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
            assert!(graph.is_empty());
        }
    }

    #[test]
    fn a_track_shorter_than_one_analysis_window_yields_an_empty_graph() {
        let (samples, beats) = track(&[220.0, 330.0], BEAT_SEC, RATE);
        let graph = detect(&samples[..100], RATE, &beats, JumpCueOptions::default()).unwrap();
        assert!(graph.is_empty());
    }

    #[test]
    fn cues_land_on_the_repeat_even_when_the_matrix_is_decimated() {
        // Over four thousand beats, so the feature matrix has to be decimated.
        // Python kept indexing the decimated matrix with full-resolution beat
        // numbers, which put every cue at a fraction of its true position.
        const LONG_RATE: u32 = 2_000;
        const LONG_BEAT: f64 = 0.2;
        let half: Vec<f64> = (0..2_100).map(pitch).collect();
        let pitches: Vec<f64> = half.iter().chain(&half).copied().collect();
        assert!(pitches.len() > 4_000, "the regression needs a long track");
        let (samples, beats) = track(&pitches, LONG_BEAT, LONG_RATE);

        let options = JumpCueOptions {
            // Smaller than the default only to keep the test quick; decimation
            // is what is under test, and a lower cap decimates harder.
            n_mels: 8,
            max_cols: 700,
            ..JumpCueOptions::default()
        };
        let links = detect_links(&samples, LONG_RATE, &beats, options).unwrap();
        assert_eq!(
            links.len(),
            1,
            "expected the one half-track repeat: {links:#?}"
        );

        let found = &links[0];
        let half_sec = 2_100.0 * LONG_BEAT;
        assert!(
            found.earlier.start < 5.0 && (found.earlier.end - half_sec).abs() < 5.0,
            "earlier region {:?} should be the first half (0-{half_sec}s)",
            found.earlier
        );
        assert!(
            (found.later.start - half_sec).abs() < 5.0
                && (found.later.end - 2.0 * half_sec).abs() < 5.0,
            "later region {:?} should be the second half",
            found.later
        );
        assert!(
            (found.lag_sec - half_sec).abs() < 2.0,
            "lag {}s should be half the track",
            found.lag_sec
        );
        assert!(
            (found.lag_beats - 2_100.0).abs() < 10.0,
            "lag {} should be 2100 beats",
            found.lag_beats
        );
    }

    #[test]
    fn a_decimated_column_reports_the_time_of_its_own_beat() {
        let beats: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
        let matrix = BeatMatrix {
            columns: vec![vec![1.0]; 8],
            step: 5,
            span: 4,
            beats: beats.clone(),
            avg_beat_sec: 0.5,
        };
        // Column 3 starts at beat 15, not at beat 3.
        assert_eq!(matrix.first_beat(3), 15);
        assert_eq!(matrix.start_time(3), beats[15]);
        assert_eq!(matrix.end_time(3), beats[19]);
        // Past the last beat, times extrapolate at the final spacing.
        assert!((matrix.beat_time(41) - 20.5).abs() < 1e-9);
    }

    #[test]
    fn a_column_spans_the_beats_it_stacks() {
        let (samples, beats) = a_b_a();
        let matrix = beat_matrix(&samples, RATE, &beats, &JumpCueOptions::default()).unwrap();
        assert_eq!(matrix.step, 1, "a short track needs no decimation");
        assert_eq!(matrix.span, 4);
        assert_eq!(matrix.len(), beats.len() - 1 - 4 + 1);
        assert!((matrix.end_time(0) - matrix.start_time(0) - 4.0 * BEAT_SEC).abs() < 1e-9);
    }

    #[test]
    fn features_are_normalised_so_similarity_is_a_dot_product() {
        let (samples, beats) = a_b_a();
        let matrix = beat_matrix(&samples, RATE, &beats, &JumpCueOptions::default()).unwrap();
        for column in 0..matrix.len() {
            assert!(
                (matrix.similarity(column, column) - 1.0).abs() < 1e-9,
                "column {column} is not unit length"
            );
        }
    }

    #[test]
    fn the_main_diagonal_band_is_suppressed() {
        let (samples, beats) = a_b_a();
        let options = JumpCueOptions::default();
        let matrix = beat_matrix(&samples, RATE, &beats, &options).unwrap();
        let ssm = self_similarity(&matrix, &options);
        for row in 0..ssm.size {
            for column in
                row.saturating_sub(options.diag_band)..=(row + options.diag_band).min(ssm.size - 1)
            {
                assert_eq!(ssm.get(row, column), 0.0, "({row}, {column}) survived");
            }
        }
    }

    #[test]
    fn the_lag_profile_peaks_at_the_repeat_distance() {
        let (samples, beats) = a_b_a();
        let options = JumpCueOptions::default();
        let matrix = beat_matrix(&samples, RATE, &beats, &options).unwrap();
        let ssm = self_similarity(&matrix, &options);
        let profile = lag_profile(&ssm);
        let peak = profile
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        assert!(
            peak.abs_diff(64) <= 2,
            "the profile peaks at lag {peak}, not at the 64-beat repeat"
        );
        assert!(profile.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn prominence_measures_height_above_the_flanking_valleys() {
        let profile = [0.0, 0.2, 1.0, 0.4, 0.9, 0.1];
        // The 1.0 peak drops to 0.0 on the left and, past the lower 0.9, to
        // 0.1 on the right; the shallower of the two descents is what counts.
        assert!((prominence(&profile, 2) - 0.9).abs() < 1e-12);
        // The 0.9 peak is flanked by 0.4 and 0.1.
        assert!((prominence(&profile, 4) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_smoothing_kernel_conserves_energy() {
        let kernel = gaussian_kernel(1.0);
        assert!((kernel.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(kernel.len() % 2, 1, "a symmetric kernel needs a centre");
    }

    #[test]
    fn confidence_is_computed_over_the_links_that_are_emitted() {
        // Five candidates, four kept: the confidences of the four must be a
        // distribution on their own. Python normalised over all five and then
        // truncated, so what it emitted summed to less than one.
        let scores = [0.9, 0.8, 0.7, 0.6];
        let confidences = softmax_confidence(&scores, 0.35);
        let total: f64 = confidences.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "emitted confidences sum to {total}"
        );
        assert!(
            confidences[0] > confidences[3],
            "a better score must carry more confidence"
        );
    }

    #[test]
    fn confidences_of_a_detected_track_sum_to_one() {
        let (samples, beats) = a_b_a();
        let links = detect_links(&samples, RATE, &beats, JumpCueOptions::default()).unwrap();
        let total: f64 = links.iter().map(|l| l.confidence).sum();
        assert!((total - 1.0).abs() < 1e-9, "confidences sum to {total}");
    }

    #[test]
    fn softmax_of_no_links_is_empty() {
        assert!(softmax_confidence(&[], 0.35).is_empty());
    }

    #[test]
    fn regions_that_mostly_overlap_are_the_same_region() {
        assert!(region(10.0, 20.0).is_same_as(&region(11.0, 21.0)));
        assert!(!region(10.0, 20.0).is_same_as(&region(18.0, 28.0)));
        assert!(!region(10.0, 20.0).is_same_as(&region(30.0, 40.0)));
    }

    #[test]
    fn links_that_share_a_region_become_one_cue_in_one_component() {
        // A repeats at 0s, 30s and 60s: two links, three regions, one group.
        // Python labelled each link's ends separately, so the same region at 0s
        // became two cues and the group fell apart into two unrelated pairs.
        let links = vec![
            link(region(0.0, 10.0), region(30.0, 40.0), 0.9),
            link(region(0.0, 10.0), region(60.0, 70.0), 0.8),
        ];
        let graph = graph_from_links(&links);
        assert_eq!(
            graph.cues().len(),
            3,
            "the shared region must not be duplicated"
        );
        assert_eq!(
            graph.components().len(),
            1,
            "all three are mutually reachable"
        );
        // Every pair is jumpable in both directions: 3 pairs, 6 links.
        assert_eq!(graph.links().len(), 6);
    }

    #[test]
    fn unrelated_links_stay_in_separate_components() {
        let links = vec![
            link(region(0.0, 10.0), region(30.0, 40.0), 0.9),
            link(region(80.0, 90.0), region(120.0, 130.0), 0.8),
        ];
        let graph = graph_from_links(&links);
        assert_eq!(graph.cues().len(), 4);
        assert_eq!(graph.components().len(), 2);
        assert!(graph.validate_labels().is_ok());
    }

    #[test]
    fn a_cue_says_which_cues_it_can_reach() {
        let links = vec![link(region(0.0, 10.0), region(30.0, 40.0), 0.9)];
        let graph = graph_from_links(&links);
        assert!(graph.by_label("A").unwrap().comment.contains('B'));
        assert!(graph.by_label("B").unwrap().comment.contains('A'));
    }

    #[test]
    fn no_links_make_an_empty_graph() {
        let graph = graph_from_links(&[]);
        assert!(graph.is_empty());
        assert!(graph.links().is_empty());
    }

    #[test]
    fn only_the_best_pairs_are_kept() {
        let (samples, beats) = a_b_a();
        let options = JumpCueOptions {
            max_pairs: 1,
            min_score: 0.0,
            peak_prominence: 0.0,
            ..JumpCueOptions::default()
        };
        let links = detect_links(&samples, RATE, &beats, options).unwrap();
        assert!(
            links.len() <= 1,
            "max_pairs was not honoured: {}",
            links.len()
        );
    }

    #[test]
    fn a_region_shorter_than_the_minimum_is_dropped() {
        let (samples, beats) = a_b_a();
        let options = JumpCueOptions {
            min_duration_sec: 60.0, // longer than the whole track
            ..JumpCueOptions::default()
        };
        assert!(detect_links(&samples, RATE, &beats, options)
            .unwrap()
            .is_empty());
    }
}
