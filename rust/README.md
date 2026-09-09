# Mixlyzer in Rust

A reimplementation of Mixlyzer's analysis and library layer as a Rust workspace,
plus a command-line front end. The Qt desktop application is not part of this;
what is here is everything underneath it — decoding, the analysis pipeline,
persistence, and Rekordbox export.

```
mixlyzer-core     domain types and pure logic: beatgrid, key, phrases,
                  cue points, JumpCUEs, configuration
mixlyzer-dsp      decoding and analysis: envelopes, onsets, tempo, key
mixlyzer-store    SQLite library, feature files, schema migrations
mixlyzer-export   Rekordbox XML
mixlyzer-cli      the `mixlyzer` binary
```

## Building and running

```sh
cd rust
cargo test --workspace       # 380 tests
cargo build --release
./target/release/mixlyzer analyze /path/to/track.flac
```

No external binary is required. Decoding is in-process through Symphonia, so
unlike the Python app there is no FFmpeg to install, find, or ship.

```
mixlyzer analyze <file> [--json]        analyse and print, storing nothing
mixlyzer add <file> [--force]           analyse and record in the library
mixlyzer list [--order-by <column>]     list the library
mixlyzer export <file> [--out <path>]   write a Rekordbox XML
mixlyzer migrate [--dry-run]            bring the library schema up to date
mixlyzer transitions --from <bpm> --to <bpm> [--tolerance <percent>]
```

`--config` points at a `config.json` (the same file the desktop app uses) and
`--library` overrides the library directory.

## What it does on a real file

A synthetic 40-second test track at 128 BPM over an A minor bed:

```
tempo      128.00 BPM
key        Am (8A)
beats      85 (22 bars)
```

The same file through the Python pipeline, timed stage by stage:

| stage | Python | Rust |
|---|---:|---:|
| decode | 1.73s | |
| HPSS | 2.52s | |
| onset detection | 0.19s | |
| tempo and grid | 0.30s | |
| **total** | **4.74s** | **1.68s** |

The Python column stops after the beatgrid; the Rust total also includes the
chromagram, the key decode and the key segments. Both agree on 128.00 BPM.

## Agreement with the Python implementation

`parity/compare.py` builds the same fixtures through both implementations and
diffs every field:

```sh
.venv/bin/python rust/parity/compare.py
```

The 24 key labels and their harmonic neighbours, the database segment rows,
phrase numbering, fill merging, phrase colours, cue points, downbeat indices and
`bar.beat` labels all match exactly. The only differences it reports are two
deliberate ones, listed below.

## Where this deliberately differs

Each of these is a bug in the original, documented at the code that changes it
and pinned by a test whose name states the new behaviour.

**Migration cannot mistake a count for an error.** The Python 0.2.0 → 0.3.0 step
returns the number of files it converted, and the runner treats any non-zero
return as failure — so every existing library with at least one track fails
migration on every launch, never gets its version stamped, and the app exits. A
step here returns `Result<MigrationOutcome, _>`, with the count inside the
success value where it cannot be read as a status.

**One definition of a downbeat.** Python computes bar starts two different ways:
by accumulating seconds (for the beatgrid view) and by counting beat indices
(for the playhead and metronome). On a detected grid with ordinary jitter the
two drift apart by 145 ms on average and 360 ms at worst over an hour. Here
`downbeat_times` is derived from `downbeat_indices`, so a bar line always lands
on a real beat.

**The autocorrelation is indexed correctly.** Python slices the autocorrelation
to the searched lag range and then indexes that slice with unsliced lags minus
the range start, subtracting the offset twice. At the default range the negative
indices wrap and merely rank candidates by unrelated values; narrow the range to
120–140 BPM in the settings and it raises `IndexError` and the track fails.

**Both modes are tracked.** Python scores only the twelve major templates and
then forces one mode across the whole track. All 24 keys are states in the same
decode here, so A minor is reported as A minor rather than as its relative
major, and a modal change is just another transition.

**Startup failures are recoverable.** An unreachable library path, a corrupt
`library.db` and a bad log path each kill the Python app before a window exists,
with no way to reach the setting that caused it. Every one of them is a typed
error here, naming the path.

**Configuration is type-checked.** Python coerces with `bool(value)`, so the
JSON string `"false"` becomes `true`, and a malformed file is overwritten with
defaults — silently discarding the library path along with everything else. A
malformed file is an error here and is left alone. Sections the Rust tools do
not use are round-tripped untouched, so sharing `config.json` with the desktop
app is safe.

**Hot cues do not collide.** Rekordbox has eight hot-cue buttons; Python assigns
the ninth cue onward with `idx % 8`, overwriting cues that already hold those
buttons. Allocation here gives each cue the button its label names, fills the
free ones, and emits the surplus as memory cues.

**Phrase abbreviations stay distinct.** Python abbreviates by first letter, so
INTRO and INTERLUDE both render as "I" and BRIDGE and BREAK_CHORUS both as "B".
An explicit table gives IL and BC.

**Times are `f64`.** Cue points are stored as float32 in Python, whose
resolution two hours in is 0.49 ms — inside the 1 ms deduplication window, so
distinct cues in a long mix silently merge.

**Custom phrase colours are stable.** Python picks them with the randomised
builtin `hash()`, so a user's own label changes colour on every restart.

## Interoperating with an existing library

The SQLite schema is byte-identical to the Python one, including the
self-healing `time_signature` column, so `list`, `transitions` and the track
table work directly against a library the desktop app wrote.

Feature files are not shared. Python stores per-track arrays as NumPy `.npz`;
this workspace uses its own `.mxf` format, which is self-describing, versioned,
CRC-checked and written atomically. A library carrying `.npz` features will
migrate and list correctly, but its stored analysis has to be recomputed with
`mixlyzer add --force` before `export` can use it.

## Not reimplemented

- The Qt user interface, its views and the waveform renderer.
- External deck sync, which reads another process's memory through Windows APIs.
- JumpCUE detection and phrase detection. The domain types, storage, editing
  operations and export for both are here; the detectors themselves — a
  self-similarity search and a pair of gradient-boosted models — are not.
