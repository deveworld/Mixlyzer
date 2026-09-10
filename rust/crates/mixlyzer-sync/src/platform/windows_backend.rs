//! Windows process-memory backend.
//!
//! This is the only code in the crate that talks to the operating system, and
//! the only code that uses `unsafe`. It does four things and nothing else:
//! find a process by name, open it for reading, resolve a module's base
//! address, and read bytes. Every decision about *what* to read lives in the
//! portable modules, which are tested on Linux.
//!
//! The `unsafe` blocks are all Win32 calls. Each one is annotated with the
//! contract it has to keep — chiefly that a `HANDLE` is still open (guaranteed
//! by [`OwnedHandle`], which owns it for its lifetime and closes it on drop)
//! and that any pointer passed into a call points at a buffer of at least the
//! length also passed in.
//!
//! The process is opened with `PROCESS_QUERY_LIMITED_INFORMATION |
//! PROCESS_VM_READ`: enough to read memory and ask for the image path, and no
//! more. Nothing here writes to the target.

// The Win32 API cannot be called without `unsafe`; the rest of the crate
// forbids it.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::mem::size_of;

use windows::core::{BOOL, PWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Module32FirstW, Module32NextW, Process32FirstW, Process32NextW,
    CREATE_TOOLHELP_SNAPSHOT_FLAGS, MODULEENTRY32W, PROCESSENTRY32W, TH32CS_SNAPMODULE,
    TH32CS_SNAPMODULE32, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{
    GetExitCodeProcess, IsWow64Process, OpenProcess, QueryFullProcessImageNameW,
    PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
};

use crate::address::PointerWidth;
use crate::denylist::normalize_process_name;
use crate::error::SyncError;
use crate::reader::{MemoryReader, ProcessIdentity};

/// `GetExitCodeProcess` reports this while the process is still running.
const STILL_RUNNING: u32 = 259;

/// A `HANDLE` that is closed when it goes out of scope.
#[derive(Debug)]
struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from a Win32 call that returned a valid handle
        // and has not been closed before — this type is the only owner.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// Reads memory from a running Windows process.
#[derive(Debug)]
pub struct ProcessMemoryReader {
    process: OwnedHandle,
    identity: ProcessIdentity,
    module_base: u64,
    pointer_width: PointerWidth,
}

impl ProcessMemoryReader {
    /// Attach to a process by pid.
    ///
    /// `module` names the module the configured offsets are relative to;
    /// `None` means the process's own executable, which is what the Python
    /// implementation used (`pymem`'s `base_address`).
    pub fn open_by_pid(pid: u32, module: Option<&str>) -> Result<Self, SyncError> {
        // SAFETY: a plain Win32 call; the returned handle is checked by
        // windows-rs and taken over by `OwnedHandle` immediately.
        let handle = unsafe {
            OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                false,
                pid,
            )
        }
        .map_err(|err| SyncError::ProcessOpen {
            pid,
            detail: err.message(),
        })?;
        let process = OwnedHandle(handle);

        let image_path = query_image_path(&process);
        let name = image_path
            .rsplit(['\\', '/'])
            .next()
            .unwrap_or_default()
            .to_string();
        let (module_base, _) = resolve_module_base(pid, module)?;

        Ok(Self {
            identity: ProcessIdentity {
                pid,
                name,
                image_path,
                // The version resource is not read; see `crate::denylist`.
                company: String::new(),
            },
            pointer_width: pointer_width_of(&process),
            process,
            module_base,
        })
    }

    /// Attach to the first running process with this image name.
    pub fn open_by_name(name: &str, module: Option<&str>) -> Result<Self, SyncError> {
        Self::open_by_pid(Self::find_pid_by_name(name)?, module)
    }

    /// Find a running process by image name, ignoring case and `.exe`.
    ///
    /// Replaces the Python version's `tasklist /FO CSV` subprocess, which
    /// spawns a console process on every poll that misses the one-second cache.
    pub fn find_pid_by_name(name: &str) -> Result<u32, SyncError> {
        let target = normalize_process_name(name);
        if target.is_empty() {
            return Err(SyncError::ProcessNotFound(name.to_string()));
        }
        let snapshot = take_snapshot(TH32CS_SNAPPROCESS, 0)?;
        let mut entry = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        // SAFETY: `snapshot` is open for the whole loop and `entry` is a
        // properly sized, initialised PROCESSENTRY32W.
        if unsafe { Process32FirstW(snapshot.0, &mut entry) }.is_err() {
            return Err(SyncError::ProcessNotFound(name.to_string()));
        }
        loop {
            if normalize_process_name(&wide_to_string(&entry.szExeFile)) == target {
                return Ok(entry.th32ProcessID);
            }
            // SAFETY: as above; the call only writes into `entry`.
            if unsafe { Process32NextW(snapshot.0, &mut entry) }.is_err() {
                return Err(SyncError::ProcessNotFound(name.to_string()));
            }
        }
    }

    /// Whether the process is still running.
    fn is_alive(&self) -> bool {
        let mut code = 0u32;
        // SAFETY: the handle is open and `code` is a valid u32 to write into.
        match unsafe { GetExitCodeProcess(self.process.0, &mut code) } {
            Ok(()) => code == STILL_RUNNING,
            // If the question cannot be asked, assume it is alive and let the
            // read error speak for itself.
            Err(_) => true,
        }
    }
}

impl MemoryReader for ProcessMemoryReader {
    fn read(&self, address: u64, len: usize) -> Result<Vec<u8>, SyncError> {
        let mut buffer = vec![0u8; len];
        let mut read = 0usize;
        // SAFETY: the handle is open; `buffer` holds exactly `len` bytes and is
        // the only thing written to; `read` is a valid usize to write into.
        let result = unsafe {
            ReadProcessMemory(
                self.process.0,
                address as *const c_void,
                buffer.as_mut_ptr().cast::<c_void>(),
                len,
                Some(&mut read),
            )
        };
        if let Err(err) = result {
            // A process that has exited is permanent; anything else is worth
            // retrying, because the target reallocates while it plays.
            if !self.is_alive() {
                return Err(SyncError::ProcessGone);
            }
            return Err(SyncError::Read {
                address,
                len,
                detail: err.message(),
            });
        }
        if read != len {
            return Err(SyncError::Read {
                address,
                len,
                detail: format!("short read: {read} of {len} bytes"),
            });
        }
        Ok(buffer)
    }

    fn module_base(&self) -> Result<u64, SyncError> {
        Ok(self.module_base)
    }

    fn pointer_width(&self) -> PointerWidth {
        self.pointer_width
    }

    fn identity(&self) -> ProcessIdentity {
        self.identity.clone()
    }
}

fn take_snapshot(
    flags: CREATE_TOOLHELP_SNAPSHOT_FLAGS,
    pid: u32,
) -> Result<OwnedHandle, SyncError> {
    // SAFETY: a plain Win32 call; windows-rs turns INVALID_HANDLE_VALUE into
    // an error, and the handle is owned from here on.
    let handle = unsafe { CreateToolhelp32Snapshot(flags, pid) }.map_err(|err| {
        SyncError::ModuleBase(format!(
            "could not snapshot process {pid}: {}",
            err.message()
        ))
    })?;
    Ok(OwnedHandle(handle))
}

/// Base address (and name) of the module the offsets are relative to.
///
/// The first module in a process snapshot is the executable itself, which is
/// what an omitted `module` means.
fn resolve_module_base(pid: u32, module: Option<&str>) -> Result<(u64, String), SyncError> {
    let snapshot = take_snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)?;
    let mut entry = MODULEENTRY32W {
        dwSize: size_of::<MODULEENTRY32W>() as u32,
        ..Default::default()
    };
    // SAFETY: `snapshot` is open for the whole loop and `entry` is a properly
    // sized, initialised MODULEENTRY32W.
    if unsafe { Module32FirstW(snapshot.0, &mut entry) }.is_err() {
        return Err(SyncError::ModuleBase(format!(
            "process {pid} has no readable module list"
        )));
    }
    let wanted = module.map(|m| m.trim().to_lowercase());
    loop {
        let name = wide_to_string(&entry.szModule);
        let matches = match &wanted {
            None => true, // the first entry is the executable
            Some(wanted) => name.to_lowercase() == *wanted,
        };
        if matches {
            return Ok((entry.modBaseAddr as u64, name));
        }
        // SAFETY: as above; the call only writes into `entry`.
        if unsafe { Module32NextW(snapshot.0, &mut entry) }.is_err() {
            return Err(SyncError::ModuleBase(format!(
                "process {pid} has no module named {:?}",
                module.unwrap_or_default()
            )));
        }
    }
}

/// Full path of the executable, or an empty string if it cannot be read.
///
/// An empty path only weakens the denylist's path check; the name check still
/// applies, so this is not worth failing the whole attach over.
fn query_image_path(process: &OwnedHandle) -> String {
    // The documented maximum for an extended-length path.
    let mut buffer = vec![0u16; 32_768];
    let mut size = buffer.len() as u32;
    // SAFETY: the handle is open; `buffer` holds `size` u16s, which is what the
    // call is told, and `size` is updated to the length actually written.
    let ok = unsafe {
        QueryFullProcessImageNameW(
            process.0,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut size,
        )
    };
    match ok {
        Ok(()) => String::from_utf16_lossy(&buffer[..size as usize]),
        Err(_) => String::new(),
    }
}

/// Pointer width of the target.
///
/// `IsWow64Process` answers "is this a 32-bit process on 64-bit Windows",
/// which is the only case that matters: Mixlyzer itself is a 64-bit build, so
/// a target that is not under WOW64 is 64-bit. This replaces `pymem`'s
/// `RemotePointer`, which guesses the width from the address being read.
fn pointer_width_of(process: &OwnedHandle) -> PointerWidth {
    let mut wow64 = BOOL(0);
    // SAFETY: the handle is open and `wow64` is a valid BOOL to write into.
    match unsafe { IsWow64Process(process.0, &mut wow64) } {
        Ok(()) if wow64.as_bool() => PointerWidth::Bits32,
        _ => PointerWidth::Bits64,
    }
}

/// Decode a NUL-terminated UTF-16 field from a Win32 struct.
fn wide_to_string(buffer: &[u16]) -> String {
    let end = buffer
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(buffer.len());
    String::from_utf16_lossy(&buffer[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_field_stops_at_its_nul() {
        let mut field = [0u16; 8];
        for (slot, unit) in field.iter_mut().zip("ab".encode_utf16()) {
            *slot = unit;
        }
        assert_eq!(wide_to_string(&field), "ab");
        assert_eq!(wide_to_string(&[]), "");
    }

    #[test]
    fn an_empty_process_name_is_not_searched_for() {
        assert!(matches!(
            ProcessMemoryReader::find_pid_by_name("   "),
            Err(SyncError::ProcessNotFound(_))
        ));
    }

    /// The current process is always there, so this exercises the real
    /// snapshot, open, module-base and read paths end to end.
    #[test]
    fn the_test_process_can_be_read_from_its_own_module_base() {
        let pid = std::process::id();
        let reader = ProcessMemoryReader::open_by_pid(pid, None).expect("open self");
        assert_eq!(MemoryReader::identity(&reader).pid, pid);
        assert!(!MemoryReader::identity(&reader).image_path.is_empty());
        assert!(reader.module_base().unwrap() > 0);
        // The first two bytes of a PE image are "MZ".
        let base = reader.module_base().unwrap();
        assert_eq!(reader.read(base, 2).unwrap(), b"MZ".to_vec());
        assert!(reader.is_alive());
    }

    #[test]
    fn an_unmapped_address_is_a_read_error_not_a_panic() {
        let reader = ProcessMemoryReader::open_by_pid(std::process::id(), None).expect("open self");
        assert!(matches!(reader.read(0x10, 8), Err(SyncError::Read { .. })));
    }
}
