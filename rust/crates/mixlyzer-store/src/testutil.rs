//! A throwaway directory per test, outside the repository.
//!
//! Tests must never write into the source tree and must not collide with each
//! other, so each one gets a directory named from the process id and a counter
//! under the system temp directory, removed when the guard drops.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A temporary directory that deletes itself.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// Create a directory whose name mentions `label`, to make a leaked one
    /// traceable back to the test that made it.
    pub fn new(label: &str) -> Self {
        let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "mixlyzer-store-{label}-{}-{seq}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside the directory.
    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A failure here would mask the test's own assertion failure, so it is
        // deliberately ignored; the worst case is a stale directory in /tmp.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
