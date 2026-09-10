//! A minimal reader for NumPy `.npz` archives.
//!
//! An `.npz` is a zip file whose members are `.npy` arrays, and
//! `np.savez_compressed` writes them with plain DEFLATE. Reading it here rather
//! than shelling out to Python keeps the detector a self-contained library: the
//! shipped weights load the same way in a desktop build, a test and a CLI.
//!
//! Only what the phrase model actually stores is supported — little-endian
//! `f8`/`i4`/`i8`, `u1`, `b1` and UTF-32 `U` strings, C order, no pickles.

use std::collections::HashMap;
use std::io::Read;

use crate::error::PhraseError;

/// One array out of the archive, already decoded to a usable Rust type.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyArray {
    F64(Vec<f64>),
    I64(Vec<i64>),
    /// `u1` and `b1` both land here; NumPy stores booleans as single bytes.
    U8(Vec<u8>),
    /// A `U`-dtype array, one `String` per element.
    Str(Vec<String>),
}

impl NpyArray {
    /// Values as `f64`, whatever the stored type, for numeric arrays.
    pub fn to_f64(&self, name: &str) -> Result<Vec<f64>, PhraseError> {
        match self {
            NpyArray::F64(v) => Ok(v.clone()),
            NpyArray::I64(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            NpyArray::U8(v) => Ok(v.iter().map(|x| f64::from(*x)).collect()),
            NpyArray::Str(_) => Err(PhraseError::ModelField {
                field: name.to_string(),
                reason: "expected a numeric array, found strings".into(),
            }),
        }
    }

    /// Values as `usize`, for the index and offset arrays.
    pub fn to_usize(&self, name: &str) -> Result<Vec<usize>, PhraseError> {
        match self {
            NpyArray::I64(v) => v
                .iter()
                .map(|x| {
                    usize::try_from(*x).map_err(|_| PhraseError::ModelField {
                        field: name.to_string(),
                        reason: format!("negative index {x}"),
                    })
                })
                .collect(),
            NpyArray::U8(v) => Ok(v.iter().map(|x| usize::from(*x)).collect()),
            NpyArray::F64(v) => Ok(v.iter().map(|x| *x as usize).collect()),
            NpyArray::Str(_) => Err(PhraseError::ModelField {
                field: name.to_string(),
                reason: "expected an index array, found strings".into(),
            }),
        }
    }

    /// Booleans, for the `is_leaf` and `missing_go_to_left` flags.
    pub fn to_bool(&self, name: &str) -> Result<Vec<bool>, PhraseError> {
        match self {
            NpyArray::U8(v) => Ok(v.iter().map(|x| *x != 0).collect()),
            NpyArray::I64(v) => Ok(v.iter().map(|x| *x != 0).collect()),
            NpyArray::F64(v) => Ok(v.iter().map(|x| *x != 0.0).collect()),
            NpyArray::Str(_) => Err(PhraseError::ModelField {
                field: name.to_string(),
                reason: "expected flags, found strings".into(),
            }),
        }
    }

    /// The single element of a zero-dimensional string array.
    pub fn to_scalar_string(&self, name: &str) -> Result<String, PhraseError> {
        match self {
            NpyArray::Str(v) if v.len() == 1 => Ok(v[0].clone()),
            NpyArray::U8(v) if v.len() == 1 => Ok(if v[0] != 0 { "True" } else { "False" }.into()),
            _ => Err(PhraseError::ModelField {
                field: name.to_string(),
                reason: "expected a single string".into(),
            }),
        }
    }

    pub fn to_strings(&self, name: &str) -> Result<Vec<String>, PhraseError> {
        match self {
            NpyArray::Str(v) => Ok(v.clone()),
            _ => Err(PhraseError::ModelField {
                field: name.to_string(),
                reason: "expected a string array".into(),
            }),
        }
    }
}

/// Every array in the archive, keyed by member name without the `.npy`.
pub type Npz = HashMap<String, NpyArray>;

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, PhraseError> {
    bytes
        .get(offset..offset + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or_else(|| PhraseError::MalformedModel("truncated zip structure".into()))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, PhraseError> {
    bytes
        .get(offset..offset + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| PhraseError::MalformedModel("truncated zip structure".into()))
}

/// Parse an `.npz` from bytes already in memory.
pub fn parse(bytes: &[u8]) -> Result<Npz, PhraseError> {
    const EOCD_SIG: u32 = 0x0605_4b50;
    const CENTRAL_SIG: u32 = 0x0201_4b50;
    const LOCAL_SIG: u32 = 0x0403_4b50;

    // The end-of-central-directory record sits at the tail, after a comment of
    // unknown length, so it has to be found by scanning backwards.
    let eocd = (0..bytes.len().saturating_sub(21))
        .map(|back| bytes.len() - 22 - back)
        .find(|offset| u32_at(bytes, *offset).ok() == Some(EOCD_SIG))
        .ok_or_else(|| PhraseError::MalformedModel("not a zip archive".into()))?;

    let entries = u16_at(bytes, eocd + 10)? as usize;
    let mut cursor = u32_at(bytes, eocd + 16)? as usize;

    let mut out = Npz::new();
    for _ in 0..entries {
        if u32_at(bytes, cursor)? != CENTRAL_SIG {
            return Err(PhraseError::MalformedModel(
                "corrupt zip central directory".into(),
            ));
        }
        let method = u16_at(bytes, cursor + 10)?;
        let compressed_size = u32_at(bytes, cursor + 20)? as usize;
        let uncompressed_size = u32_at(bytes, cursor + 24)? as usize;
        let name_len = u16_at(bytes, cursor + 28)? as usize;
        let extra_len = u16_at(bytes, cursor + 30)? as usize;
        let comment_len = u16_at(bytes, cursor + 32)? as usize;
        let local_offset = u32_at(bytes, cursor + 42)? as usize;
        let name = String::from_utf8_lossy(
            bytes
                .get(cursor + 46..cursor + 46 + name_len)
                .ok_or_else(|| PhraseError::MalformedModel("truncated member name".into()))?,
        )
        .to_string();
        cursor += 46 + name_len + extra_len + comment_len;

        if u32_at(bytes, local_offset)? != LOCAL_SIG {
            return Err(PhraseError::MalformedModel("corrupt zip member".into()));
        }
        let local_name_len = u16_at(bytes, local_offset + 26)? as usize;
        let local_extra_len = u16_at(bytes, local_offset + 28)? as usize;
        let data_start = local_offset + 30 + local_name_len + local_extra_len;
        let raw = bytes
            .get(data_start..data_start + compressed_size)
            .ok_or_else(|| PhraseError::MalformedModel("truncated zip member data".into()))?;

        let payload = match method {
            0 => raw.to_vec(),
            8 => {
                let mut decoded = Vec::with_capacity(uncompressed_size);
                flate2::read::DeflateDecoder::new(raw)
                    .read_to_end(&mut decoded)
                    .map_err(|e| PhraseError::MalformedModel(format!("inflate failed: {e}")))?;
                decoded
            }
            other => {
                return Err(PhraseError::MalformedModel(format!(
                    "unsupported zip compression method {other}"
                )))
            }
        };

        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(key, parse_npy(&payload)?);
    }
    Ok(out)
}

/// Parse one `.npy` member.
fn parse_npy(bytes: &[u8]) -> Result<NpyArray, PhraseError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(PhraseError::MalformedModel("not a .npy array".into()));
    }
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        (u16_at(bytes, 8)? as usize, 10)
    } else {
        (u32_at(bytes, 8)? as usize, 12)
    };
    let header = String::from_utf8_lossy(
        bytes
            .get(header_start..header_start + header_len)
            .ok_or_else(|| PhraseError::MalformedModel("truncated .npy header".into()))?,
    )
    .to_string();
    let data = &bytes[header_start + header_len..];

    if header.contains("'fortran_order': True") {
        return Err(PhraseError::MalformedModel(
            "Fortran-ordered arrays are not supported".into(),
        ));
    }
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|rest| rest.split('\'').nth(1))
        .ok_or_else(|| PhraseError::MalformedModel("missing dtype in .npy header".into()))?
        .to_string();

    // `descr` is like "<f8", "|u1", "<U25"; the first character is byte order.
    let kind = descr
        .chars()
        .nth(1)
        .ok_or_else(|| PhraseError::MalformedModel(format!("bad dtype {descr}")))?;
    let width: usize = descr[2..].parse().unwrap_or(1);

    match kind {
        'f' if width == 8 => Ok(NpyArray::F64(
            data.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                .collect(),
        )),
        'f' if width == 4 => Ok(NpyArray::F64(
            data.chunks_exact(4)
                .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap_or([0; 4]))))
                .collect(),
        )),
        'i' if width == 8 => Ok(NpyArray::I64(
            data.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                .collect(),
        )),
        'i' if width == 4 => Ok(NpyArray::I64(
            data.chunks_exact(4)
                .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap_or([0; 4]))))
                .collect(),
        )),
        'u' | 'b' if width == 1 => Ok(NpyArray::U8(data.to_vec())),
        'U' => {
            // NumPy stores `U` as fixed-width UTF-32LE, null-padded.
            let stride = width * 4;
            if stride == 0 {
                return Ok(NpyArray::Str(Vec::new()));
            }
            Ok(NpyArray::Str(
                data.chunks_exact(stride)
                    .map(|chunk| {
                        chunk
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                            .take_while(|c| *c != 0)
                            .filter_map(char::from_u32)
                            .collect::<String>()
                    })
                    .collect(),
            ))
        }
        _ => Err(PhraseError::MalformedModel(format!(
            "unsupported .npy dtype {descr}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_zip_file_is_rejected_rather_than_misread() {
        let err = parse(b"this is not a model at all").unwrap_err();
        assert!(matches!(err, PhraseError::MalformedModel(_)));
    }

    #[test]
    fn an_empty_buffer_is_rejected() {
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn npy_headers_without_a_dtype_are_rejected() {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let header = b"{'shape': (0,), }".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header);
        assert!(parse_npy(&bytes).is_err());
    }
}
