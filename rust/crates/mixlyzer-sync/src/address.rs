//! Offset chains: turning `"0x1A2B, 0x10, 0x8"` into an address in the target.
//!
//! A configured chain is a comma-separated list of hexadecimal offsets. The
//! first entry is the base; the rest are dereference steps. A leading `g` on
//! the first entry marks the base as an absolute address rather than one
//! relative to the module base (`"g0x7FF6ABCD1234, 0x20"`).
//!
//! Two things differ from the Python implementation on purpose.
//!
//! * Parsing is checked. Python's `_hex_to_int` does
//!   `token.lower().replace("0x", "")` and then `int(token, 16)`, so `"1p0"`
//!   raises deep inside a poll (which disables the feature), `"g"` alone
//!   becomes an exception, an empty entry between commas is silently dropped,
//!   and `"0x10x20"` silently becomes `0x1020`. Here every malformed chain is
//!   an [`AddressError`] reported once, when the engine is built.
//! * The pointer read width comes from the *target process*, not from the
//!   address being dereferenced. `pymem`'s `RemotePointer` picks a 4- or 8-byte
//!   read depending on whether the address itself exceeds 2^31, so a 64-bit
//!   target whose pointers happen to live low in the address space is read four
//!   bytes at a time and every chain after the first step is garbage. See
//!   [`PointerWidth`] and `chain_width_comes_from_the_target_not_the_address`.

use crate::error::SyncError;
use crate::reader::MemoryReader;

/// Pointer size in the *target* process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerWidth {
    /// A 32-bit (WOW64) process: pointers are 4 bytes.
    Bits32,
    /// A 64-bit process: pointers are 8 bytes.
    Bits64,
}

impl PointerWidth {
    /// Number of bytes one pointer occupies.
    pub fn bytes(self) -> usize {
        match self {
            PointerWidth::Bits32 => 4,
            PointerWidth::Bits64 => 8,
        }
    }

    /// Decode a little-endian pointer of this width.
    pub fn decode(self, bytes: &[u8]) -> Result<u64, SyncError> {
        if bytes.len() < self.bytes() {
            return Err(SyncError::Decode {
                value_type: "pointer",
                detail: format!("needed {} bytes, got {}", self.bytes(), bytes.len()),
            });
        }
        Ok(match self {
            PointerWidth::Bits32 => {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).into()
            }
            PointerWidth::Bits64 => u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
        })
    }
}

/// Why an offset string could not be turned into a chain.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AddressError {
    /// No offsets at all.
    #[error("the offset chain is empty")]
    Empty,

    /// An entry between two commas is blank, e.g. `"0x10,,0x8"`.
    ///
    /// Python skips blank entries, quietly shortening the chain by one
    /// dereference and reading from an unrelated address.
    #[error("offset {index} is blank")]
    BlankOffset {
        /// Position in the chain, base first.
        index: usize,
    },

    /// An entry is not a hexadecimal number.
    #[error("offset {index} ({token:?}) is not a hexadecimal number")]
    NotHex {
        /// Position in the chain, base first.
        index: usize,
        /// The entry as it was written.
        token: String,
    },

    /// An entry does not fit in a 64-bit address.
    #[error("offset {index} ({token:?}) does not fit in 64 bits")]
    TooLarge {
        /// Position in the chain, base first.
        index: usize,
        /// The entry as it was written.
        token: String,
    },

    /// `g` appeared on something other than the base.
    ///
    /// Python strips a leading `g` from *every* entry but only honours it on
    /// the first, so `"0x10, g0x20"` silently means `"0x10, 0x20"`.
    #[error("the absolute-address marker 'g' is only valid on the first offset, found at offset {index}")]
    MisplacedAbsoluteMarker {
        /// Position in the chain, base first.
        index: usize,
    },
}

/// A parsed offset chain: a base plus zero or more dereference steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressChain {
    absolute: bool,
    offsets: Vec<u64>,
}

impl AddressChain {
    /// Parse a configured offset string.
    ///
    /// ```
    /// use mixlyzer_sync::address::AddressChain;
    /// let chain = AddressChain::parse("0x1A2B, 0x10, 0x8").unwrap();
    /// assert!(!chain.is_absolute());
    /// assert_eq!(chain.offsets(), &[0x1A2B, 0x10, 0x8]);
    /// ```
    pub fn parse(text: &str) -> Result<Self, AddressError> {
        let raw = text.trim();
        if raw.is_empty() {
            return Err(AddressError::Empty);
        }
        let mut absolute = false;
        let mut offsets = Vec::new();
        for (index, token) in raw.split(',').enumerate() {
            let token = token.trim();
            if token.is_empty() {
                return Err(AddressError::BlankOffset { index });
            }
            let body = match token.strip_prefix(['g', 'G']) {
                Some(rest) => {
                    if index != 0 {
                        return Err(AddressError::MisplacedAbsoluteMarker { index });
                    }
                    absolute = true;
                    rest.trim()
                }
                None => token,
            };
            offsets.push(parse_hex(index, token, body)?);
        }
        Ok(Self { absolute, offsets })
    }

    /// Whether the base is an absolute address (the `g` marker).
    pub fn is_absolute(&self) -> bool {
        self.absolute
    }

    /// The parsed offsets, base first.
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Number of dereferences this chain performs.
    pub fn dereference_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Walk the chain in the target process and return the final address.
    ///
    /// Mirrors the shape of `pymem`'s `RemotePointer` walk — `read(addr) +
    /// offset`, repeated — but reads pointers at the target's own width and
    /// reports a null pointer as [`SyncError::NullPointer`] instead of reading
    /// address 0.
    pub fn resolve<R: MemoryReader + ?Sized>(&self, reader: &R) -> Result<u64, SyncError> {
        let base_offset = self.offsets[0];
        let mut address = if self.absolute {
            base_offset
        } else {
            reader.module_base()?.wrapping_add(base_offset)
        };
        let width = reader.pointer_width();
        for (step, offset) in self.offsets.iter().copied().enumerate().skip(1) {
            if address == 0 {
                return Err(SyncError::NullPointer { step: step - 1 });
            }
            let bytes = reader.read(address, width.bytes())?;
            let pointer = width.decode(&bytes)?;
            if pointer == 0 {
                return Err(SyncError::NullPointer { step });
            }
            address = pointer.wrapping_add(offset);
        }
        if address == 0 {
            // Python raises a bare `ValueError("Invalid address")` here, which
            // is caught by the poll loop and disables the feature.
            return Err(SyncError::NullPointer {
                step: self.offsets.len() - 1,
            });
        }
        Ok(address)
    }
}

fn parse_hex(index: usize, token: &str, body: &str) -> Result<u64, AddressError> {
    let digits = body
        .strip_prefix("0x")
        .or_else(|| body.strip_prefix("0X"))
        .unwrap_or(body);
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(AddressError::NotHex {
            index,
            token: token.to_string(),
        });
    }
    u64::from_str_radix(digits, 16).map_err(|_| AddressError::TooLarge {
        index,
        token: token.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::fake::FakeReader;

    #[test]
    fn a_single_offset_is_relative_to_the_module_base() {
        let chain = AddressChain::parse("0x120").unwrap();
        assert_eq!(chain.dereference_count(), 0);
        let reader = FakeReader::new().with_module_base(0x1000);
        assert_eq!(chain.resolve(&reader).unwrap(), 0x1120);
    }

    #[test]
    fn a_leading_g_marks_an_absolute_address() {
        let chain = AddressChain::parse("g0x7FF6ABCD1234, 0x20").unwrap();
        assert!(chain.is_absolute());
        assert_eq!(chain.offsets(), &[0x7FF6_ABCD_1234, 0x20]);

        let mut reader = FakeReader::new().with_module_base(0x1000);
        reader.write_u64(0x7FF6_ABCD_1234, 0x4000);
        // The module base must play no part in an absolute chain.
        assert_eq!(chain.resolve(&reader).unwrap(), 0x4020);
    }

    #[test]
    fn an_uppercase_g_and_uppercase_hex_both_parse() {
        let chain = AddressChain::parse("G0XABCDEF").unwrap();
        assert!(chain.is_absolute());
        assert_eq!(chain.offsets(), &[0xABCDEF]);
    }

    #[test]
    fn a_chain_dereferences_each_step_in_turn() {
        let chain = AddressChain::parse("0x10, 0x8, 0x4").unwrap();
        let mut reader = FakeReader::new().with_module_base(0x1000);
        // base 0x1010 -> 0x2000; 0x2008 -> 0x3000; final = 0x3004
        reader.write_u64(0x1010, 0x2000);
        reader.write_u64(0x2008, 0x3000);
        assert_eq!(chain.resolve(&reader).unwrap(), 0x3004);
    }

    /// `pymem`'s `RemotePointer.value` chooses a 4- or 8-byte read from the
    /// *address* it is reading (`> 0x7FFFFFFF`), not from the bitness of the
    /// target. A 64-bit program whose pointer table sits low in memory is read
    /// four bytes at a time, and every step after the first lands on garbage.
    #[test]
    fn chain_width_comes_from_the_target_not_the_address() {
        let chain = AddressChain::parse("0x0, 0x0").unwrap();
        let mut reader = FakeReader::new().with_module_base(0x1000);
        // A 64-bit pointer stored at a low address. Reading it as 4 bytes
        // yields 0x10 (what Python does); reading 8 yields 0x1_0000_0010.
        reader.write_bytes(0x1000, &[0x10, 0, 0, 0, 0x01, 0, 0, 0]);
        assert_eq!(chain.resolve(&reader).unwrap(), 0x1_0000_0010);

        let narrow = reader.with_pointer_width(PointerWidth::Bits32);
        assert_eq!(chain.resolve(&narrow).unwrap(), 0x10);
    }

    #[test]
    fn a_null_pointer_in_the_middle_is_reported_not_followed() {
        let chain = AddressChain::parse("0x10, 0x8, 0x4").unwrap();
        let mut reader = FakeReader::new().with_module_base(0x1000);
        reader.write_u64(0x1010, 0);
        assert_eq!(
            chain.resolve(&reader).unwrap_err(),
            SyncError::NullPointer { step: 1 }
        );
    }

    #[test]
    fn a_chain_that_resolves_to_zero_is_reported() {
        let chain = AddressChain::parse("0x0").unwrap();
        let reader = FakeReader::new().with_module_base(0);
        assert_eq!(
            chain.resolve(&reader).unwrap_err(),
            SyncError::NullPointer { step: 0 }
        );
    }

    #[test]
    fn a_failing_module_base_lookup_propagates() {
        let chain = AddressChain::parse("0x10").unwrap();
        let reader = FakeReader::new().without_module_base();
        assert!(matches!(
            chain.resolve(&reader).unwrap_err(),
            SyncError::ModuleBase(_)
        ));
    }

    #[test]
    fn an_unmapped_read_is_an_error_not_a_zero() {
        let chain = AddressChain::parse("0x10, 0x8").unwrap();
        let reader = FakeReader::new().with_module_base(0x1000);
        assert!(matches!(
            chain.resolve(&reader).unwrap_err(),
            SyncError::Read { .. }
        ));
    }

    #[test]
    fn an_empty_chain_is_rejected() {
        assert_eq!(AddressChain::parse("   ").unwrap_err(), AddressError::Empty);
    }

    /// Python drops the blank entry and reads one dereference short.
    #[test]
    fn a_blank_offset_between_commas_is_rejected() {
        assert_eq!(
            AddressChain::parse("0x10,,0x8").unwrap_err(),
            AddressError::BlankOffset { index: 1 }
        );
        assert_eq!(
            AddressChain::parse("0x10,").unwrap_err(),
            AddressError::BlankOffset { index: 1 }
        );
    }

    /// Python's `replace("0x", "")` turns this into `0x1020` without a word.
    #[test]
    fn a_doubled_prefix_is_rejected_rather_than_stitched_together() {
        assert!(matches!(
            AddressChain::parse("0x10x20").unwrap_err(),
            AddressError::NotHex { index: 0, .. }
        ));
    }

    #[test]
    fn non_hex_text_is_rejected_with_its_position() {
        assert_eq!(
            AddressChain::parse("0x10, zz").unwrap_err(),
            AddressError::NotHex {
                index: 1,
                token: "zz".into()
            }
        );
        // A bare marker has no digits behind it.
        assert!(matches!(
            AddressChain::parse("g").unwrap_err(),
            AddressError::NotHex { index: 0, .. }
        ));
    }

    #[test]
    fn an_offset_wider_than_an_address_is_rejected() {
        assert!(matches!(
            AddressChain::parse("0x1FFFFFFFFFFFFFFFF").unwrap_err(),
            AddressError::TooLarge { index: 0, .. }
        ));
    }

    /// Python honours `g` only on the first token but strips it from all of
    /// them, so this chain silently loses the marker.
    #[test]
    fn an_absolute_marker_on_a_later_offset_is_rejected() {
        assert_eq!(
            AddressChain::parse("0x10, g0x20").unwrap_err(),
            AddressError::MisplacedAbsoluteMarker { index: 1 }
        );
    }

    #[test]
    fn whitespace_around_offsets_is_ignored() {
        let chain = AddressChain::parse("  g 0x10 ,  0x20  ").unwrap();
        assert!(chain.is_absolute());
        assert_eq!(chain.offsets(), &[0x10, 0x20]);
    }

    #[test]
    fn a_short_pointer_read_is_a_decode_error() {
        assert!(matches!(
            PointerWidth::Bits64.decode(&[0, 1, 2]).unwrap_err(),
            SyncError::Decode { .. }
        ));
        assert_eq!(PointerWidth::Bits32.bytes(), 4);
        assert_eq!(PointerWidth::Bits64.bytes(), 8);
    }
}
