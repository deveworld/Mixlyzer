//! The one thing the portable core needs from the operating system: the
//! ability to read another process's memory.
//!
//! Everything above this trait — chains, value specs, deck selection, the
//! failure policy — is exercised on any platform against
//! [`fake::FakeReader`], a `HashMap`-backed target.

use crate::address::PointerWidth;
use crate::error::SyncError;

/// What is known about the process being followed.
///
/// Used for denylist checks. `company` comes from the executable's version
/// information; the Windows backend does not query it (see
/// [`crate::denylist`]), so it is empty unless a caller fills it in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessIdentity {
    /// Process id.
    pub pid: u32,
    /// Image name as the OS reports it, e.g. `rekordbox.exe`.
    pub name: String,
    /// Full path to the executable.
    pub image_path: String,
    /// Company name from the executable's version resource, when known.
    pub company: String,
}

/// Read access to a target process.
///
/// Implementors are expected to be cheap to call: [`SyncEngine::poll`] runs at
/// the UI frame rate.
///
/// [`SyncEngine::poll`]: crate::engine::SyncEngine::poll
pub trait MemoryReader {
    /// Read exactly `len` bytes at `address`.
    ///
    /// A short read is a failure, not a truncated buffer: a half-read pointer
    /// is worse than no pointer.
    fn read(&self, address: u64, len: usize) -> Result<Vec<u8>, SyncError>;

    /// Base address of the module the configured offsets are relative to.
    fn module_base(&self) -> Result<u64, SyncError>;

    /// Pointer size in the target process.
    ///
    /// This is the fix for `pymem`'s `RemotePointer`, which picks the read
    /// width from the value of the address being dereferenced. Defaults to
    /// 64-bit, which is what every current DJ program ships.
    fn pointer_width(&self) -> PointerWidth {
        PointerWidth::Bits64
    }

    /// Who the target is, for denylist checks. Empty by default.
    fn identity(&self) -> ProcessIdentity {
        ProcessIdentity::default()
    }
}

impl<T: MemoryReader + ?Sized> MemoryReader for &T {
    fn read(&self, address: u64, len: usize) -> Result<Vec<u8>, SyncError> {
        (**self).read(address, len)
    }

    fn module_base(&self) -> Result<u64, SyncError> {
        (**self).module_base()
    }

    fn pointer_width(&self) -> PointerWidth {
        (**self).pointer_width()
    }

    fn identity(&self) -> ProcessIdentity {
        (**self).identity()
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! An in-memory stand-in for a live process, so every layer above
    //! [`super::MemoryReader`] is testable on Linux.

    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// A target process made of `HashMap<u64, Vec<u8>>` regions.
    #[derive(Debug)]
    pub struct FakeReader {
        regions: HashMap<u64, Vec<u8>>,
        module_base: Option<u64>,
        pointer_width: PointerWidth,
        identity: ProcessIdentity,
        /// Set to make the next reads fail; used to drive the failure policy.
        next_error: RefCell<Option<SyncError>>,
    }

    impl FakeReader {
        pub fn new() -> Self {
            Self {
                regions: HashMap::new(),
                module_base: Some(0),
                pointer_width: PointerWidth::Bits64,
                identity: ProcessIdentity::default(),
                next_error: RefCell::new(None),
            }
        }

        pub fn with_module_base(mut self, base: u64) -> Self {
            self.module_base = Some(base);
            self
        }

        pub fn without_module_base(mut self) -> Self {
            self.module_base = None;
            self
        }

        pub fn with_pointer_width(self, width: PointerWidth) -> Self {
            Self {
                pointer_width: width,
                ..self
            }
        }

        pub fn with_identity(mut self, identity: ProcessIdentity) -> Self {
            self.identity = identity;
            self
        }

        pub fn write_bytes(&mut self, address: u64, bytes: &[u8]) {
            self.regions.insert(address, bytes.to_vec());
        }

        pub fn write_u64(&mut self, address: u64, value: u64) {
            self.write_bytes(address, &value.to_le_bytes());
        }

        pub fn write_i32(&mut self, address: u64, value: i32) {
            self.write_bytes(address, &value.to_le_bytes());
        }

        pub fn write_f32(&mut self, address: u64, value: f32) {
            self.write_bytes(address, &value.to_le_bytes());
        }

        pub fn write_u8(&mut self, address: u64, value: u8) {
            self.write_bytes(address, &[value]);
        }

        /// Write a NUL-terminated UTF-8 string padded out to `len` bytes, as a
        /// fixed-size buffer in the target would be.
        pub fn write_utf8_buffer(&mut self, address: u64, text: &str, len: usize) {
            let mut bytes = text.as_bytes().to_vec();
            bytes.push(0);
            bytes.resize(len.max(bytes.len()), 0);
            self.write_bytes(address, &bytes);
        }

        /// Write a NUL-terminated UTF-16 string padded out to `len` bytes.
        pub fn write_utf16_buffer(&mut self, address: u64, text: &str, len: usize) {
            let mut bytes = Vec::new();
            for unit in text.encode_utf16().chain(std::iter::once(0)) {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            bytes.resize(len.max(bytes.len()), 0);
            self.write_bytes(address, &bytes);
        }

        /// Make every read fail until [`FakeReader::clear_error`].
        pub fn fail_with(&self, err: SyncError) {
            *self.next_error.borrow_mut() = Some(err);
        }

        pub fn clear_error(&self) {
            *self.next_error.borrow_mut() = None;
        }
    }

    impl MemoryReader for FakeReader {
        fn read(&self, address: u64, len: usize) -> Result<Vec<u8>, SyncError> {
            if let Some(err) = self.next_error.borrow().clone() {
                return Err(err);
            }
            for (start, bytes) in &self.regions {
                let end = start + bytes.len() as u64;
                if address >= *start && address + len as u64 <= end {
                    let from = (address - start) as usize;
                    return Ok(bytes[from..from + len].to_vec());
                }
            }
            Err(SyncError::Read {
                address,
                len,
                detail: "address is not mapped in the fake target".into(),
            })
        }

        fn module_base(&self) -> Result<u64, SyncError> {
            self.module_base
                .ok_or_else(|| SyncError::ModuleBase("no module base in the fake target".into()))
        }

        fn pointer_width(&self) -> PointerWidth {
            self.pointer_width
        }

        fn identity(&self) -> ProcessIdentity {
            self.identity.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fake::FakeReader;

    #[test]
    fn the_fake_reads_from_within_a_region() {
        let mut reader = FakeReader::new();
        reader.write_bytes(0x100, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(reader.read(0x102, 2).unwrap(), vec![3, 4]);
    }

    #[test]
    fn the_fake_refuses_a_read_that_runs_off_the_end() {
        let mut reader = FakeReader::new();
        reader.write_bytes(0x100, &[1, 2]);
        assert!(reader.read(0x101, 4).is_err());
    }

    #[test]
    fn a_reference_forwards_every_method() {
        let reader = FakeReader::new().with_module_base(0x40);
        let by_ref: &dyn MemoryReader = &reader;
        assert_eq!(by_ref.module_base().unwrap(), 0x40);
        assert_eq!(by_ref.pointer_width(), PointerWidth::Bits64);
        assert_eq!(by_ref.identity(), ProcessIdentity::default());
    }
}
