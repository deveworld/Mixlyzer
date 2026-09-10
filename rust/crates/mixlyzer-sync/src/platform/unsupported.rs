//! Stub backend for everything that is not Windows.
//!
//! Reading another process's memory has no portable equivalent, and Mixlyzer
//! only ever follows Windows DJ software. Every entry point here returns
//! [`SyncError::Unsupported`], which the failure policy treats as permanent, so
//! a host that asks for external sync on Linux or macOS gets one clear message
//! instead of a silently dead feature.

use crate::address::PointerWidth;
use crate::error::SyncError;
use crate::reader::{MemoryReader, ProcessIdentity};

/// The Windows process-memory reader — not available on this platform.
///
/// This type cannot be constructed here; it exists so that code calling into
/// the backend compiles on every platform.
#[derive(Debug)]
pub struct ProcessMemoryReader {
    /// Uninhabited in practice: no constructor returns a value.
    _private: (),
}

fn unsupported() -> SyncError {
    SyncError::Unsupported(std::env::consts::OS)
}

impl ProcessMemoryReader {
    /// Attach to a process by pid. Always fails off Windows.
    pub fn open_by_pid(_pid: u32, _module: Option<&str>) -> Result<Self, SyncError> {
        Err(unsupported())
    }

    /// Attach to the first process with this image name. Always fails off
    /// Windows.
    pub fn open_by_name(_name: &str, _module: Option<&str>) -> Result<Self, SyncError> {
        Err(unsupported())
    }

    /// Find a running process by image name. Always fails off Windows.
    pub fn find_pid_by_name(_name: &str) -> Result<u32, SyncError> {
        Err(unsupported())
    }
}

impl MemoryReader for ProcessMemoryReader {
    fn read(&self, _address: u64, _len: usize) -> Result<Vec<u8>, SyncError> {
        Err(unsupported())
    }

    fn module_base(&self) -> Result<u64, SyncError> {
        Err(unsupported())
    }

    fn pointer_width(&self) -> PointerWidth {
        PointerWidth::Bits64
    }

    fn identity(&self) -> ProcessIdentity {
        ProcessIdentity::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_point_reports_the_platform() {
        let err = ProcessMemoryReader::open_by_name("rekordbox.exe", None).unwrap_err();
        assert!(matches!(err, SyncError::Unsupported(_)));
        assert!(err.to_string().contains(std::env::consts::OS), "{err}");
        assert!(err.is_permanent(), "there is no point retrying this");

        assert!(ProcessMemoryReader::open_by_pid(1234, None).is_err());
        assert!(ProcessMemoryReader::find_pid_by_name("x").is_err());
    }
}
