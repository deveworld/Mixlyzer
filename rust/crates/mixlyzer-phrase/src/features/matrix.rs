//! A dense row-major matrix, sized once and never resized.
//!
//! Every feature in this crate is a `(channels, frames)` grid, so a tiny
//! purpose-built type beats pulling in a linear-algebra crate: it keeps the
//! numerics visible next to the librosa code they mirror.

/// `rows` × `cols` of `f64`, stored row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_rows(rows: Vec<Vec<f64>>) -> Self {
        let n_rows = rows.len();
        let n_cols = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for row in rows {
            debug_assert_eq!(row.len(), n_cols);
            data.extend_from_slice(&row);
        }
        Self {
            rows: n_rows,
            cols: n_cols,
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, value: f64) {
        self.data[r * self.cols + c] = value;
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// One column, copied out. Columns are strided, so this allocates.
    pub fn column(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Stack matrices that share a column count, one above the next.
    pub fn vstack(parts: &[&Mat]) -> Mat {
        let cols = parts.first().map_or(0, |m| m.cols);
        let rows = parts.iter().map(|m| m.rows).sum();
        let mut data = Vec::with_capacity(rows * cols);
        for part in parts {
            debug_assert_eq!(part.cols, cols);
            data.extend_from_slice(&part.data);
        }
        Mat { rows, cols, data }
    }

    /// Elementwise map, keeping the shape.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Mat {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().copied().map(f).collect(),
        }
    }

    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Median of a slice, matching NumPy: the mean of the two middle values when
/// the count is even.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    // Quickselect rather than a full sort: HPSS runs this once per spectrogram
    // cell over a 31-sample window, so the log factor is worth removing.
    let mut buffer: Vec<f64> = values.to_vec();
    let len = buffer.len();
    let mid = len / 2;
    let (lower, middle, _) = buffer.select_nth_unstable_by(mid, f64::total_cmp);
    let middle = *middle;
    if len % 2 != 0 {
        return middle;
    }
    let below = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    0.5 * (below + middle)
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation, as NumPy's `std` computes it (`ddof=0`).
pub fn std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_an_even_count_averages_the_middle_pair() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[3.0, 1.0]), 2.0);
    }

    #[test]
    fn vstack_concatenates_rows_and_keeps_columns() {
        let a = Mat::from_rows(vec![vec![1.0, 2.0]]);
        let b = Mat::from_rows(vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
        let stacked = Mat::vstack(&[&a, &b]);
        assert_eq!((stacked.rows(), stacked.cols()), (3, 2));
        assert_eq!(stacked.row(2), &[5.0, 6.0]);
    }
}
