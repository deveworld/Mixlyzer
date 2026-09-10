//! Following an external DJ program's decks by reading its process memory.
//!
//! Mixlyzer can shadow another program: which deck is playing, which file it
//! has loaded, and where the playhead is. The host polls [`SyncEngine::poll`],
//! loads the track it names and follows the time it reports.
//!
//! The crate is in two halves.
//!
//! * A **portable core** — [`address`], [`value`], [`denylist`], [`path`],
//!   [`timing`], [`failure`], [`engine`] — which is where all the behaviour
//!   lives and which is fully testable on any platform. Its one dependency on
//!   the operating system is the [`MemoryReader`] trait, so the whole poll loop
//!   runs against an in-memory fake target in tests.
//! * A **Windows backend** in [`platform`], compiled only for
//!   `target_os = "windows"`, that opens a process by name or pid, reads its
//!   memory, resolves a module base and queries its image path. Everywhere else
//!   the same type exists as a stub returning [`SyncError::Unsupported`].
//!
//! It reimplements `core/external_sync.py`. The module documentation says
//! where the two deliberately differ; the headline differences are:
//!
//! * A read failure no longer disables the feature and rewrites `config.json`.
//!   See [`failure`].
//! * Offset chains are parsed and reported instead of silently producing a
//!   wrong address. See [`address`].
//! * Pointers are read at the target's width, not at a width guessed from the
//!   address. See [`address::PointerWidth`].
//! * UTF-16 paths can be read at all. See [`value`].
//! * Path validation is not cached, so a file that appears or vanishes is
//!   noticed. See [`path`].
//! * `blocked_company_keywords`, which nothing in the Python reads, is
//!   implemented. See [`denylist`].
//!
//! ```no_run
//! use mixlyzer_sync::{DenylistGuard, ProcessMemoryReader, SyncConfig, SyncEngine};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = SyncConfig::default();
//! let reader = ProcessMemoryReader::open_by_name(&config.memory_process_name, None)?;
//! let mut engine = SyncEngine::new(
//!     config,
//!     reader,
//!     DenylistGuard::from_file("process_denylist.json"),
//! )?;
//! if let Some(deck) = engine.poll()? {
//!     println!("deck {} at {:.2}s: {:?}", deck.deck, deck.time_sec, deck.path);
//! }
//! # Ok(())
//! # }
//! ```

// `unsafe` exists in exactly one module, the Windows backend, which opts back
// in with its own `allow` and documents every block.
#![deny(unsafe_code)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

pub mod address;
pub mod config;
pub mod denylist;
pub mod engine;
pub mod error;
pub mod failure;
pub mod path;
pub mod platform;
pub mod reader;
pub mod timing;
pub mod value;

pub use address::{AddressChain, AddressError, PointerWidth};
pub use config::{DeckConfig, SyncConfig, SyncMode};
pub use denylist::{DenyReason, Denylist, DenylistGuard};
pub use engine::{DeckState, SyncEngine};
pub use error::{Severity, SyncError};
pub use failure::{DisableReason, FailurePolicy, FailureTracker, SyncStatus};
pub use path::{validate_track_path, PathRejection, ValidatedPath};
pub use platform::ProcessMemoryReader;
pub use reader::{MemoryReader, ProcessIdentity};
pub use timing::{sample_index_to_time, total_samples_for, TotalSampleSource, TrackInfo};
pub use value::{MemoryValue, StringEncoding, ValueSpec, ValueType};
