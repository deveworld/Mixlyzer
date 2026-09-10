//! Memory value specs: how a run of bytes at the end of a chain becomes a
//! number, a flag or a path.
//!
//! This is `memoryvalueconfig` from the Python config — an offset chain, a type
//! (`float`/`int`/`bool`/`str`), a byte length, a text encoding, a bit position
//! and a multiplier.
//!
//! Deliberate differences from `ExternalSyncController._read_memory_value`:
//!
//! * **UTF-16 works.** Python calls `pymem.read_string(..., encoding=...)`,
//!   which stops at the first NUL *byte*. Every ASCII-range UTF-16 character
//!   has a NUL as its second byte, so a UTF-16 path read this way is always one
//!   character long. Decoding here is driven by the encoding: UTF-16 stops at
//!   the first NUL *unit*. Pinned by `utf16_paths_survive_their_nul_high_bytes`.
//! * **`length` is honoured for numbers.** Python's `read_int` is always four
//!   bytes and `read_float` always four, whatever the config says. A 64-bit
//!   playhead therefore reads as its low half. Here `length` selects 1/2/4/8
//!   for `int` and 4/8 for `float`; 0 keeps the Python default of 4.
//! * **An out-of-range bit position is an error.** Python falls back to "the
//!   whole byte is truthy", so asking for bit 9 silently answers a different
//!   question.
//! * **The multiplier is part of the value**, not of one field. Python applies
//!   `multiplier` only to `time`, so the same setting on `sample_index` is
//!   silently ignored.

use crate::error::SyncError;

/// The four value types the Python config allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValueType {
    /// IEEE binary32 by default, binary64 when `length` is 8.
    Float,
    /// Signed little-endian integer.
    Int,
    /// One bit of one byte, chosen by `bit_pos`.
    Bool,
    /// Text, `length` bytes long, in `encoding`.
    Str,
}

impl ValueType {
    /// Name used in error messages and in `config.json`.
    pub fn as_str(self) -> &'static str {
        match self {
            ValueType::Float => "float",
            ValueType::Int => "int",
            ValueType::Bool => "bool",
            ValueType::Str => "str",
        }
    }
}

/// Text encoding of a `str` value in the target's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringEncoding {
    /// UTF-8 (also covers ASCII and Latin-1-in-ASCII-range configs).
    Utf8,
    /// UTF-16, little endian — what every Windows program stores.
    Utf16Le,
    /// UTF-16, big endian.
    Utf16Be,
}

impl StringEncoding {
    /// Map a config string onto a supported encoding.
    ///
    /// An empty setting means UTF-8, matching Python's `spec.encoding or
    /// "utf-8"`. An unsupported name is an error rather than a decode failure
    /// deep in the poll loop.
    pub fn parse(name: &str) -> Result<Self, SyncError> {
        let key: String = name
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' '))
            .collect();
        match key.as_str() {
            "" | "utf8" | "ascii" | "latin1" | "cp1252" => Ok(StringEncoding::Utf8),
            "utf16" | "utf16le" | "unicode" | "widechar" => Ok(StringEncoding::Utf16Le),
            "utf16be" => Ok(StringEncoding::Utf16Be),
            _ => Err(SyncError::ValueSpec(format!(
                "unsupported string encoding {name:?}"
            ))),
        }
    }
}

/// A decoded value read out of the target.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryValue {
    /// A number, with the multiplier already applied.
    Float(f64),
    /// A whole number, read with no multiplier.
    Int(i64),
    /// A flag.
    Bool(bool),
    /// Text, cut at its terminator.
    Str(String),
}

impl MemoryValue {
    /// The value as a number, for the time and sample-index fields.
    ///
    /// A `str` has no numeric reading and yields `None`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MemoryValue::Float(v) => Some(*v),
            MemoryValue::Int(v) => Some(*v as f64),
            MemoryValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            MemoryValue::Str(_) => None,
        }
    }

    /// The value as text, for the path field.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MemoryValue::Str(v) => Some(v),
            _ => None,
        }
    }

    /// Python's `bool(raw)`: what `loaded` and `active` are tested with.
    ///
    /// Kept identical on purpose — a deck flag is sometimes configured as an
    /// `int` or even a `str` in the wild.
    pub fn truthy(&self) -> bool {
        match self {
            MemoryValue::Float(v) => *v != 0.0,
            MemoryValue::Int(v) => *v != 0,
            MemoryValue::Bool(v) => *v,
            MemoryValue::Str(v) => !v.is_empty(),
        }
    }
}

/// One configured value: where it lives and how to read it.
///
/// Field names match the Python `memoryvalueconfig` dataclass so an existing
/// `config.json` deserialises unchanged.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ValueSpec {
    /// The offset chain, e.g. `"0x1A2B, 0x10"`. Parsed by
    /// [`crate::address::AddressChain`].
    pub offsets: String,
    /// How the bytes at the end of the chain are to be read.
    pub value_type: ValueType,
    /// Byte length. Meaningful for `str`, and selects the width of `int` and
    /// `float`; `0` means "the default width for the type".
    #[serde(default)]
    pub length: usize,
    /// Text encoding for a `str` value; see [`StringEncoding`].
    #[serde(default = "default_encoding")]
    pub encoding: String,
    /// Which bit of the byte a `bool` lives in, `0..=7`.
    #[serde(default)]
    pub bit_pos: u32,
    /// Scale applied to numeric values, e.g. milliseconds to seconds.
    #[serde(default = "default_multiplier")]
    pub multiplier: f64,
}

fn default_encoding() -> String {
    "utf-8".to_string()
}

fn default_multiplier() -> f64 {
    1.0
}

impl ValueSpec {
    /// A numeric spec with the Python defaults.
    pub fn new(offsets: impl Into<String>, value_type: ValueType) -> Self {
        Self {
            offsets: offsets.into(),
            value_type,
            length: 0,
            encoding: default_encoding(),
            bit_pos: 0,
            multiplier: 1.0,
        }
    }

    /// How many bytes must be read for this value.
    pub fn read_len(&self) -> Result<usize, SyncError> {
        match self.value_type {
            ValueType::Float => match self.length {
                0 | 4 => Ok(4),
                8 => Ok(8),
                other => Err(SyncError::ValueSpec(format!(
                    "a float must be 4 or 8 bytes, not {other}"
                ))),
            },
            ValueType::Int => match self.length {
                0 | 4 => Ok(4),
                1 => Ok(1),
                2 => Ok(2),
                8 => Ok(8),
                other => Err(SyncError::ValueSpec(format!(
                    "an int must be 1, 2, 4 or 8 bytes, not {other}"
                ))),
            },
            ValueType::Bool => Ok(1),
            ValueType::Str => Ok(self.length.max(1)),
        }
    }

    /// Turn the bytes read at the end of the chain into a value.
    ///
    /// The multiplier is applied to numeric results here, so it means the same
    /// thing wherever the spec is used.
    pub fn decode(&self, bytes: &[u8]) -> Result<MemoryValue, SyncError> {
        let need = self.read_len()?;
        if bytes.len() < need {
            return Err(SyncError::Decode {
                value_type: self.value_type.as_str(),
                detail: format!("needed {need} bytes, got {}", bytes.len()),
            });
        }
        let bytes = &bytes[..need];
        match self.value_type {
            ValueType::Float => {
                let raw = if need == 4 {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                } else {
                    f64::from_le_bytes(bytes.try_into().expect("checked length"))
                };
                Ok(MemoryValue::Float(self.scale(raw)))
            }
            ValueType::Int => {
                let raw = match need {
                    1 => i64::from(bytes[0] as i8),
                    2 => i64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
                    4 => i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
                    _ => i64::from_le_bytes(bytes.try_into().expect("checked length")),
                };
                if self.multiplier == 1.0 {
                    Ok(MemoryValue::Int(raw))
                } else {
                    // A scaled int is no longer an integer; say so rather than
                    // truncating silently.
                    Ok(MemoryValue::Float(self.scale(raw as f64)))
                }
            }
            ValueType::Bool => {
                let byte = bytes[0];
                if self.bit_pos > 7 {
                    return Err(SyncError::ValueSpec(format!(
                        "bit_pos must be 0..=7 for a bool, not {}",
                        self.bit_pos
                    )));
                }
                Ok(MemoryValue::Bool((byte >> self.bit_pos) & 1 == 1))
            }
            ValueType::Str => {
                let text = match StringEncoding::parse(&self.encoding)? {
                    StringEncoding::Utf8 => decode_utf8(bytes)?,
                    StringEncoding::Utf16Le => decode_utf16(bytes, u16::from_le_bytes)?,
                    StringEncoding::Utf16Be => decode_utf16(bytes, u16::from_be_bytes)?,
                };
                Ok(MemoryValue::Str(text))
            }
        }
    }

    /// Apply the configured multiplier, guarding against a nonsense setting.
    pub fn scale(&self, value: f64) -> f64 {
        if self.multiplier.is_finite() {
            value * self.multiplier
        } else {
            value
        }
    }
}

/// Stop at the first NUL byte, then decode.
fn decode_utf8(bytes: &[u8]) -> Result<String, SyncError> {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end])
        .map(str::to_string)
        .map_err(|err| SyncError::Decode {
            value_type: "str",
            detail: format!("not valid UTF-8: {err}"),
        })
}

/// Stop at the first NUL *unit*, then decode.
///
/// A trailing odd byte is ignored: the buffer length is a byte count and the
/// user is free to configure an odd one.
fn decode_utf16(bytes: &[u8], to_unit: fn([u8; 2]) -> u16) -> Result<String, SyncError> {
    let mut units = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let unit = to_unit([pair[0], pair[1]]);
        if unit == 0 {
            break;
        }
        units.push(unit);
    }
    String::from_utf16(&units).map_err(|err| SyncError::Decode {
        value_type: "str",
        detail: format!("not valid UTF-16: {err}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(value_type: ValueType) -> ValueSpec {
        ValueSpec::new("0x10", value_type)
    }

    #[test]
    fn a_float_is_four_little_endian_bytes() {
        let decoded = spec(ValueType::Float)
            .decode(&1.5f32.to_le_bytes())
            .unwrap();
        assert_eq!(decoded, MemoryValue::Float(1.5));
    }

    /// Python's `read_float` is always 4 bytes, so a double playhead reads as
    /// noise.
    #[test]
    fn a_float_can_be_eight_bytes_when_configured() {
        let mut spec = spec(ValueType::Float);
        spec.length = 8;
        assert_eq!(
            spec.decode(&1.25f64.to_le_bytes()).unwrap(),
            MemoryValue::Float(1.25)
        );
    }

    #[test]
    fn an_int_is_signed_and_four_bytes_by_default() {
        assert_eq!(spec(ValueType::Int).read_len().unwrap(), 4);
        assert_eq!(
            spec(ValueType::Int).decode(&(-7i32).to_le_bytes()).unwrap(),
            MemoryValue::Int(-7)
        );
    }

    #[test]
    fn an_int_honours_a_configured_width() {
        for (length, bytes, expected) in [
            (1usize, vec![0xFFu8], -1i64),
            (2, vec![0x00, 0x80], -32768),
            (8, 0x1_0000_0000i64.to_le_bytes().to_vec(), 0x1_0000_0000),
        ] {
            let mut spec = spec(ValueType::Int);
            spec.length = length;
            assert_eq!(spec.decode(&bytes).unwrap(), MemoryValue::Int(expected));
        }
    }

    #[test]
    fn an_unusable_width_is_rejected_before_any_read() {
        let mut odd_int = spec(ValueType::Int);
        odd_int.length = 3;
        assert!(matches!(odd_int.read_len(), Err(SyncError::ValueSpec(_))));

        let mut short_float = spec(ValueType::Float);
        short_float.length = 2;
        assert!(matches!(
            short_float.read_len(),
            Err(SyncError::ValueSpec(_))
        ));
    }

    #[test]
    fn a_bool_reads_the_configured_bit() {
        let mut spec = spec(ValueType::Bool);
        spec.bit_pos = 2;
        assert_eq!(
            spec.decode(&[0b0000_0100]).unwrap(),
            MemoryValue::Bool(true)
        );
        assert_eq!(
            spec.decode(&[0b1111_1011]).unwrap(),
            MemoryValue::Bool(false)
        );
    }

    #[test]
    fn bit_zero_is_the_default() {
        assert_eq!(
            spec(ValueType::Bool).decode(&[1]).unwrap(),
            MemoryValue::Bool(true)
        );
        assert_eq!(
            spec(ValueType::Bool).decode(&[2]).unwrap(),
            MemoryValue::Bool(false),
            "bit 0 of 0b10 is clear"
        );
    }

    /// Python answers "is the byte non-zero" for an out-of-range bit, which is
    /// a different question from the one that was asked.
    #[test]
    fn an_out_of_range_bit_position_is_rejected() {
        let mut spec = spec(ValueType::Bool);
        spec.bit_pos = 9;
        assert!(matches!(spec.decode(&[0xFF]), Err(SyncError::ValueSpec(_))));
    }

    #[test]
    fn a_utf8_string_stops_at_its_nul() {
        let mut spec = spec(ValueType::Str);
        spec.length = 16;
        let mut bytes = b"D:\\a.flac\0".to_vec();
        bytes.resize(16, 0xFF);
        assert_eq!(
            spec.decode(&bytes).unwrap(),
            MemoryValue::Str("D:\\a.flac".into())
        );
    }

    #[test]
    fn a_utf8_string_that_fills_the_buffer_needs_no_nul() {
        let mut spec = spec(ValueType::Str);
        spec.length = 4;
        assert_eq!(
            spec.decode(b"abcd").unwrap(),
            MemoryValue::Str("abcd".into())
        );
    }

    /// The Python UTF-16 read cannot work at all: `pymem.read_string` stops at
    /// the first NUL byte, which is the high byte of every ASCII-range UTF-16
    /// character, so `"D:\..."` always came back as `"D"`.
    #[test]
    fn utf16_paths_survive_their_nul_high_bytes() {
        let mut spec = spec(ValueType::Str);
        spec.encoding = "utf-16".into();
        spec.length = 64;
        let mut bytes: Vec<u8> = "D:\\Music\\track.flac"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        bytes.extend_from_slice(&[0, 0]);
        bytes.resize(64, 0x41);
        assert_eq!(
            spec.decode(&bytes).unwrap(),
            MemoryValue::Str("D:\\Music\\track.flac".into())
        );
    }

    #[test]
    fn utf16_handles_non_ascii_and_big_endian() {
        let mut spec = spec(ValueType::Str);
        spec.encoding = "utf-16le".into();
        spec.length = 32;
        let mut le: Vec<u8> = "한글 ok"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        le.extend_from_slice(&[0, 0]);
        le.resize(32, 0);
        assert_eq!(
            spec.decode(&le).unwrap(),
            MemoryValue::Str("한글 ok".into())
        );

        let mut spec_be = spec.clone();
        spec_be.encoding = "utf-16be".into();
        let mut be: Vec<u8> = "한글 ok"
            .encode_utf16()
            .flat_map(|u| u.to_be_bytes())
            .collect();
        be.extend_from_slice(&[0, 0]);
        be.resize(32, 0);
        assert_eq!(
            spec_be.decode(&be).unwrap(),
            MemoryValue::Str("한글 ok".into())
        );
    }

    #[test]
    fn a_trailing_odd_byte_in_a_utf16_buffer_is_ignored() {
        let mut spec = spec(ValueType::Str);
        spec.encoding = "utf-16".into();
        spec.length = 5; // two units plus a stray byte
        assert_eq!(
            spec.decode(&[b'h', 0, b'i', 0, 0x41]).unwrap(),
            MemoryValue::Str("hi".into())
        );
    }

    #[test]
    fn invalid_text_is_an_error_not_a_mojibake_path() {
        let mut spec = spec(ValueType::Str);
        spec.length = 4;
        assert!(matches!(
            spec.decode(&[0xFF, 0xFE, 0xFD, 0x00]),
            Err(SyncError::Decode { .. })
        ));

        let mut spec16 = spec.clone();
        spec16.encoding = "utf-16le".into();
        spec16.length = 4;
        // A lone high surrogate.
        assert!(matches!(
            spec16.decode(&[0x00, 0xD8, 0x41, 0x00]),
            Err(SyncError::Decode { .. })
        ));
    }

    #[test]
    fn an_unknown_encoding_is_rejected() {
        let mut spec = spec(ValueType::Str);
        spec.encoding = "shift-jis".into();
        assert!(matches!(spec.decode(b"abc"), Err(SyncError::ValueSpec(_))));
    }

    #[test]
    fn encoding_names_are_matched_loosely() {
        for name in ["", "UTF-8", "utf8", "ascii"] {
            assert_eq!(StringEncoding::parse(name).unwrap(), StringEncoding::Utf8);
        }
        for name in ["UTF-16", "utf_16le", "utf 16 le", "unicode"] {
            assert_eq!(
                StringEncoding::parse(name).unwrap(),
                StringEncoding::Utf16Le
            );
        }
    }

    #[test]
    fn a_short_buffer_is_an_error_rather_than_a_partial_value() {
        assert!(matches!(
            spec(ValueType::Float).decode(&[1, 2]),
            Err(SyncError::Decode { .. })
        ));
    }

    /// Python applies `multiplier` only to the `time` field, so the same
    /// setting on `sample_index` does nothing at all.
    #[test]
    fn the_multiplier_belongs_to_the_value_not_to_one_field() {
        let mut spec = spec(ValueType::Float);
        spec.multiplier = 0.001;
        assert_eq!(
            spec.decode(&1500.0f32.to_le_bytes()).unwrap(),
            MemoryValue::Float(1.5)
        );

        let mut ints = ValueSpec::new("0x10", ValueType::Int);
        ints.multiplier = 2.0;
        assert_eq!(
            ints.decode(&21i32.to_le_bytes()).unwrap(),
            MemoryValue::Float(42.0),
            "a scaled int is no longer an integer"
        );
    }

    #[test]
    fn a_nonsense_multiplier_leaves_the_value_alone() {
        let mut spec = spec(ValueType::Float);
        spec.multiplier = f64::NAN;
        assert_eq!(
            spec.decode(&2.0f32.to_le_bytes()).unwrap(),
            MemoryValue::Float(2.0)
        );
    }

    #[test]
    fn truthiness_matches_pythons_bool() {
        assert!(MemoryValue::Int(3).truthy());
        assert!(!MemoryValue::Int(0).truthy());
        assert!(MemoryValue::Float(0.5).truthy());
        assert!(!MemoryValue::Float(0.0).truthy());
        assert!(MemoryValue::Str("x".into()).truthy());
        assert!(!MemoryValue::Str(String::new()).truthy());
        assert!(MemoryValue::Bool(true).truthy());
    }

    #[test]
    fn numeric_and_text_views_only_answer_for_their_own_type() {
        assert_eq!(MemoryValue::Int(4).as_f64(), Some(4.0));
        assert_eq!(MemoryValue::Str("x".into()).as_f64(), None);
        assert_eq!(MemoryValue::Str("x".into()).as_str(), Some("x"));
        assert_eq!(MemoryValue::Bool(true).as_str(), None);
        assert_eq!(MemoryValue::Bool(true).as_f64(), Some(1.0));
    }

    #[test]
    fn a_spec_round_trips_through_the_python_field_names() {
        let json = r#"{"offsets":"0x10, 0x8","value_type":"str","length":2048,"encoding":"utf-16","bit_pos":0,"multiplier":1.0}"#;
        let spec: ValueSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.value_type, ValueType::Str);
        assert_eq!(spec.length, 2048);
        let back = serde_json::to_string(&spec).unwrap();
        assert!(back.contains("\"value_type\":\"str\""), "{back}");
    }

    #[test]
    fn missing_optional_fields_take_the_python_defaults() {
        let spec: ValueSpec =
            serde_json::from_str(r#"{"offsets":"0x10","value_type":"float"}"#).unwrap();
        assert_eq!(spec.length, 0);
        assert_eq!(spec.encoding, "utf-8");
        assert_eq!(spec.bit_pos, 0);
        assert_eq!(spec.multiplier, 1.0);
    }
}
