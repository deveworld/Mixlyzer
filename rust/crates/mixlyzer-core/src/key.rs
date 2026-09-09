//! Musical key indices, Camelot / classical labels, and harmonic compatibility.
//!
//! Index convention, unchanged from the Python implementation so that existing
//! libraries keep their meaning: `0..=11` are major keys (C..B, Camelot "B"),
//! `12..=23` are minor keys (Cm..Bm, Camelot "A"), and generally
//! `key_value = pitch_class + 12 * mode_flag` with `mode_flag` 0 = major,
//! 1 = minor.

/// Camelot wheel labels indexed by key value.
pub const CAMELOT_LABELS: [&str; 24] = [
    "8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B", //
    "5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A",
];

/// Classical labels indexed by key value.
pub const CLASSICAL_LABELS: [&str; 24] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", //
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
];

/// Mode of a key value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Mode {
    Major,
    Minor,
}

impl Mode {
    /// The flag stored alongside a pitch class in `key_segments`.
    pub const fn flag(self) -> u8 {
        match self {
            Mode::Major => 0,
            Mode::Minor => 1,
        }
    }

    /// The Camelot letter for this mode ("B" major, "A" minor).
    pub const fn camelot_letter(self) -> char {
        match self {
            Mode::Major => 'B',
            Mode::Minor => 'A',
        }
    }

    pub const fn from_flag(flag: u8) -> Self {
        if flag == 0 {
            Mode::Major
        } else {
            Mode::Minor
        }
    }
}

/// A musical key: a pitch class plus a mode, stored as an index in `0..24`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Key(u8);

impl Key {
    /// Build a key from any integer, wrapping into `0..24` the way the Python
    /// code does with `int(anchor_key) % 24`.
    pub fn from_index(index: i64) -> Self {
        Key(index.rem_euclid(24) as u8)
    }

    /// Build a key from a pitch class (`0..12`) and a mode.
    pub fn new(pitch_class: u8, mode: Mode) -> Self {
        Key((pitch_class % 12) + 12 * mode.flag())
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn pitch_class(self) -> u8 {
        self.0 % 12
    }

    pub const fn mode(self) -> Mode {
        if self.0 < 12 {
            Mode::Major
        } else {
            Mode::Minor
        }
    }

    pub fn camelot(self) -> &'static str {
        CAMELOT_LABELS[self.0 as usize]
    }

    pub fn classical(self) -> &'static str {
        CLASSICAL_LABELS[self.0 as usize]
    }

    /// The label shown in the UI, e.g. `"Am (8A)"`.
    pub fn display(self) -> String {
        format!("{} ({})", self.classical(), self.camelot())
    }

    /// Position on the Camelot wheel, `1..=12`.
    pub fn camelot_number(self) -> u8 {
        let label = self.camelot();
        label[..label.len() - 1]
            .parse()
            .expect("camelot labels are well-formed")
    }

    /// The relative key: same pitch material, opposite mode.
    ///
    /// This is the key that shares a Camelot number, e.g. `C` (8B) and
    /// `Am` (8A).
    pub fn relative(self) -> Key {
        let target_mode = match self.mode() {
            Mode::Major => Mode::Minor,
            Mode::Minor => Mode::Major,
        };
        let number = self.camelot_number();
        Key::from_camelot(number, target_mode).expect("every camelot number has both modes")
    }

    /// Look a key up by Camelot number and mode.
    pub fn from_camelot(number: u8, mode: Mode) -> Option<Key> {
        let wanted = format!("{}{}", number, mode.camelot_letter());
        CAMELOT_LABELS
            .iter()
            .position(|label| *label == wanted)
            .map(|idx| Key(idx as u8))
    }

    /// Keys considered harmonically compatible for mixing: the key itself, its
    /// two neighbours on the Camelot wheel, and its relative key.
    ///
    /// Matches `core.linear_segments.harmonic_compatible_keys`, but returns a
    /// sorted `Vec` rather than a set so callers get a stable order.
    pub fn harmonic_neighbours(self) -> Vec<Key> {
        let mode = self.mode();
        let number = self.camelot_number();
        let prev = ((number + 10) % 12) + 1; // number - 2 mod 12, then +1
        let next = (number % 12) + 1;
        let mut out = vec![
            self,
            Key::from_camelot(prev, mode).expect("wheel is complete"),
            Key::from_camelot(next, mode).expect("wheel is complete"),
            self.relative(),
        ];
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.camelot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_conventions_match_python() {
        assert_eq!(Key::from_index(0).camelot(), "8B");
        assert_eq!(Key::from_index(0).classical(), "C");
        assert_eq!(Key::from_index(12).camelot(), "5A");
        assert_eq!(Key::from_index(12).classical(), "Cm");
        assert_eq!(Key::from_index(23).camelot(), "10A");
        assert_eq!(Key::from_index(23).classical(), "Bm");
        assert_eq!(Key::from_index(0).mode(), Mode::Major);
        assert_eq!(Key::from_index(12).mode(), Mode::Minor);
    }

    #[test]
    fn key_value_is_pitch_plus_twelve_times_mode() {
        for pitch in 0u8..12 {
            assert_eq!(Key::new(pitch, Mode::Major).index(), pitch);
            assert_eq!(Key::new(pitch, Mode::Minor).index(), pitch + 12);
        }
    }

    #[test]
    fn negative_and_large_indices_wrap() {
        assert_eq!(Key::from_index(-1), Key::from_index(23));
        assert_eq!(Key::from_index(24), Key::from_index(0));
        assert_eq!(Key::from_index(-24), Key::from_index(0));
    }

    #[test]
    fn relative_key_shares_camelot_number() {
        for idx in 0..24 {
            let key = Key::from_index(idx);
            let rel = key.relative();
            assert_eq!(key.camelot_number(), rel.camelot_number());
            assert_ne!(key.mode(), rel.mode());
            assert_eq!(rel.relative(), key, "relative is an involution");
        }
    }

    /// C major (8B) mixes with 7B, 9B and its relative A minor (8A).
    #[test]
    fn harmonic_neighbours_are_wheel_adjacent_plus_relative() {
        let c_major = Key::from_index(0);
        let labels: Vec<&str> = c_major.harmonic_neighbours().iter().map(|k| k.camelot()).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec!["7B", "8A", "8B", "9B"]);
    }

    #[test]
    fn harmonic_neighbours_wrap_around_the_wheel() {
        // 1B has neighbours 12B and 2B.
        let key = Key::from_camelot(1, Mode::Major).unwrap();
        let mut labels: Vec<&str> = key.harmonic_neighbours().iter().map(|k| k.camelot()).collect();
        labels.sort_unstable();
        assert_eq!(labels, vec!["12B", "1A", "1B", "2B"]);
    }

    #[test]
    fn every_key_has_exactly_four_neighbours_including_itself() {
        for idx in 0..24 {
            let key = Key::from_index(idx);
            let n = key.harmonic_neighbours();
            assert_eq!(n.len(), 4, "key {} ({})", idx, key.camelot());
            assert!(n.contains(&key));
        }
    }

    #[test]
    fn labels_are_unique() {
        let mut cam = CAMELOT_LABELS.to_vec();
        cam.sort_unstable();
        cam.dedup();
        assert_eq!(cam.len(), 24);
        let mut cls = CLASSICAL_LABELS.to_vec();
        cls.sort_unstable();
        cls.dedup();
        assert_eq!(cls.len(), 24);
    }
}
