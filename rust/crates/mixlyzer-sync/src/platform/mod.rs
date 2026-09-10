//! The platform backend: opening a real process and reading its memory.
//!
//! Reading another process's memory is a Windows API, and every DJ program
//! Mixlyzer follows is a Windows program. The backend is therefore
//! `#[cfg(target_os = "windows")]` and everything else gets a stub whose calls
//! return [`SyncError::Unsupported`] — a typed error the host can show, rather
//! than a missing symbol or a panic.
//!
//! Both versions expose the same type, [`ProcessMemoryReader`], with the same
//! constructors, so calling code compiles unchanged on either platform. All the
//! logic worth testing lives above [`crate::reader::MemoryReader`] and is
//! exercised on Linux against a fake target.
//!
//! [`SyncError::Unsupported`]: crate::SyncError::Unsupported

#[cfg(target_os = "windows")]
mod windows_backend;
#[cfg(target_os = "windows")]
pub use windows_backend::ProcessMemoryReader;

#[cfg(not(target_os = "windows"))]
mod unsupported;
#[cfg(not(target_os = "windows"))]
pub use unsupported::ProcessMemoryReader;
