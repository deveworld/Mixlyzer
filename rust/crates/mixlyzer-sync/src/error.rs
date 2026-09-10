//! Typed failures for external deck sync, and how permanent each one is.
//!
//! The Python implementation has a single failure path: `_disable_due_to_failure`
//! turns the whole feature off on the *first* problem, writes `enabled=false`
//! back to `config.json`, and the user has to find the setting again by hand. A
//! null pointer in an offset chain is an ordinary transient condition — it
//! happens every time the DJ program is between tracks — so the feature
//! disables itself during normal use.
//!
//! Every failure here therefore carries a [`Severity`]. Only genuinely
//! permanent conditions (the process exited, the process is on the denylist,
//! the platform has no backend, the configuration cannot be parsed) end the
//! session; everything else is retried and backed off by
//! [`crate::failure::FailureTracker`].

use crate::address::AddressError;

/// How the failure policy should treat an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Expected to clear on its own: retry, then back off.
    Transient,
    /// Will not clear without the user doing something: stop polling.
    Permanent,
}

/// Anything that can go wrong while following an external deck.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SyncError {
    /// The configured offset string is not a usable pointer chain.
    #[error("invalid offset chain: {0}")]
    Address(#[from] AddressError),

    /// A pointer in the chain was null, so the chain cannot be followed.
    ///
    /// Transient by design: the DJ program nulls its deck pointers while a
    /// track is being loaded.
    #[error("offset chain hit a null pointer at step {step}")]
    NullPointer {
        /// Index into the offset list, counting the base as step 0.
        step: usize,
    },

    /// The target's memory could not be read at this address.
    #[error("could not read {len} byte(s) at {address:#x}: {detail}")]
    Read {
        /// Address the read was attempted at.
        address: u64,
        /// Number of bytes wanted.
        len: usize,
        /// What the platform said.
        detail: String,
    },

    /// The module the offsets are relative to could not be located.
    #[error("could not resolve the module base address: {0}")]
    ModuleBase(String),

    /// Bytes were read but do not form the configured value.
    #[error("could not decode a {value_type} value: {detail}")]
    Decode {
        /// The configured type that could not be produced.
        value_type: &'static str,
        /// What went wrong.
        detail: String,
    },

    /// The value spec itself is unusable (bad length, encoding, bit position).
    #[error("invalid memory value spec: {0}")]
    ValueSpec(String),

    /// The `externalsyncconfig` section could not be understood.
    #[error("invalid external sync configuration: {0}")]
    Config(String),

    /// The target process is on the denylist and must not be attached to.
    #[error("process {name:?} must not be attached to: {reason}")]
    ProcessDenied {
        /// The process that was blocked.
        name: String,
        /// Which denylist rule matched.
        reason: String,
    },

    /// The denylist could not be read, so every process is blocked.
    ///
    /// Transient: the file may be briefly locked by an editor or a sync
    /// client, and a later poll can succeed. The failure is never cached.
    #[error("process denylist unavailable ({0}); blocking every process until it can be read")]
    DenylistUnavailable(String),

    /// No running process matches the configured name.
    ///
    /// Transient: the DJ program is usually started *after* Mixlyzer.
    #[error("no running process named {0:?}")]
    ProcessNotFound(String),

    /// A process that was attached to has exited.
    #[error("the target process has exited")]
    ProcessGone,

    /// Opening the process failed (usually a privilege problem).
    #[error("could not open process {pid}: {detail}")]
    ProcessOpen {
        /// The process that could not be opened.
        pid: u32,
        /// What the platform said.
        detail: String,
    },

    /// There is no memory backend for this platform.
    #[error("external sync is not supported on this platform ({0})")]
    Unsupported(&'static str),
}

impl SyncError {
    /// Whether the failure policy should keep trying.
    ///
    /// Deliberately different from Python, which disables the feature for every
    /// variant here. In particular [`SyncError::ProcessNotFound`] is transient:
    /// the DJ program not being open yet is the normal state of affairs at
    /// startup, not a reason to switch the feature off.
    pub fn severity(&self) -> Severity {
        match self {
            SyncError::NullPointer { .. }
            | SyncError::Read { .. }
            | SyncError::ModuleBase(_)
            | SyncError::Decode { .. }
            | SyncError::DenylistUnavailable(_)
            | SyncError::ProcessNotFound(_)
            | SyncError::ProcessOpen { .. } => Severity::Transient,

            SyncError::Address(_)
            | SyncError::ValueSpec(_)
            | SyncError::Config(_)
            | SyncError::ProcessDenied { .. }
            | SyncError::ProcessGone
            | SyncError::Unsupported(_) => Severity::Permanent,
        }
    }

    /// Convenience for [`Severity::Permanent`].
    pub fn is_permanent(&self) -> bool {
        self.severity() == Severity::Permanent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the rewrite: a null pointer must not kill the feature.
    #[test]
    fn a_null_pointer_is_transient_but_a_dead_process_is_not() {
        assert_eq!(
            SyncError::NullPointer { step: 1 }.severity(),
            Severity::Transient
        );
        assert_eq!(SyncError::ProcessGone.severity(), Severity::Permanent);
    }

    /// Python disables Memory Sync when the DJ program simply is not running
    /// yet, which is the common case at startup.
    #[test]
    fn a_process_that_has_not_started_yet_is_transient() {
        assert!(!SyncError::ProcessNotFound("dj.exe".into()).is_permanent());
    }

    #[test]
    fn a_denied_process_is_permanent() {
        assert!(SyncError::ProcessDenied {
            name: "lsass".into(),
            reason: "process name is on the denylist".into(),
        }
        .is_permanent());
    }

    /// Fail-closed, but recoverable: a locked file must not be a life sentence.
    #[test]
    fn an_unreadable_denylist_is_transient() {
        assert!(!SyncError::DenylistUnavailable("locked".into()).is_permanent());
    }

    #[test]
    fn messages_name_the_thing_that_failed() {
        let err = SyncError::Read {
            address: 0x1000,
            len: 4,
            detail: "out of range".into(),
        };
        assert!(err.to_string().contains("0x1000"), "{err}");
    }
}
