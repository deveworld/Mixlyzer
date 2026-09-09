//! Argument parsing.
//!
//! Hand-rolled rather than pulled from a crate: the surface is small, and the
//! error messages can then name the flag the user actually typed.

use std::path::PathBuf;

/// What the user asked for.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Analyse a file and print the result without storing anything.
    Analyze { path: PathBuf, json: bool },
    /// Analyse a file and record it in the library.
    Add { path: PathBuf, force: bool },
    /// List the tracks in the library.
    List { order_by: String },
    /// Write a Rekordbox XML for one track.
    Export { path: PathBuf, out: Option<PathBuf> },
    /// Bring the library schema up to date.
    Migrate { dry_run: bool },
    /// Find tracks that move between two tempos.
    Transitions { from_bpm: f64, to_bpm: f64, tolerance: f64 },
    Help,
    Version,
}

/// Everything the CLI needs to run one command.
#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub command: Command,
    /// Config file to read. Defaults to `config.json` in the working directory.
    pub config: PathBuf,
    /// Overrides the library path from the config file.
    pub library: Option<PathBuf>,
}

/// Why the command line could not be understood.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ArgError {
    #[error("unknown command {0:?}; run `mixlyzer help` for the list")]
    UnknownCommand(String),

    #[error("unknown option {0:?} for `{1}`")]
    UnknownOption(String, &'static str),

    #[error("`{0}` needs {1}")]
    MissingValue(&'static str, &'static str),

    #[error("{0:?} is not a number")]
    NotANumber(String),
}

const USAGE: &str = "\
mixlyzer - analyse DJ tracks: beatgrid, tempo, key

USAGE:
    mixlyzer <command> [options]

COMMANDS:
    analyze <file> [--json]        Analyse a file and print the result
    add <file> [--force]           Analyse a file and store it in the library
    list [--order-by <column>]     List the tracks in the library
    export <file> [--out <path>]   Write a Rekordbox XML for one track
    migrate [--dry-run]            Bring the library schema up to date
    transitions --from <bpm> --to <bpm> [--tolerance <percent>]
                                   Find tracks that move between two tempos
    help, version

GLOBAL OPTIONS:
    --config <path>                Config file (default: ./config.json)
    --library <path>               Library directory, overriding the config
";

/// The usage text, for `help` and for errors.
pub fn usage() -> &'static str {
    USAGE
}

/// Parse the arguments following the program name.
pub fn parse(argv: &[String]) -> Result<Args, ArgError> {
    let mut config = PathBuf::from("config.json");
    let mut library: Option<PathBuf> = None;
    let mut rest: Vec<String> = Vec::new();

    // Global options can appear anywhere, so pull them out first.
    let mut index = 0;
    while index < argv.len() {
        match argv[index].as_str() {
            "--config" => {
                index += 1;
                config = PathBuf::from(
                    argv.get(index)
                        .ok_or(ArgError::MissingValue("--config", "a path"))?,
                );
            }
            "--library" => {
                index += 1;
                library = Some(PathBuf::from(
                    argv.get(index)
                        .ok_or(ArgError::MissingValue("--library", "a path"))?,
                ));
            }
            other => rest.push(other.to_string()),
        }
        index += 1;
    }

    let Some((name, options)) = rest.split_first() else {
        return Ok(Args {
            command: Command::Help,
            config,
            library,
        });
    };

    let command = match name.as_str() {
        "help" | "--help" | "-h" => Command::Help,
        "version" | "--version" | "-V" => Command::Version,
        "analyze" | "analyse" => {
            let (path, flags) = take_path(options, "analyze")?;
            let mut json = false;
            for flag in flags {
                match flag.as_str() {
                    "--json" => json = true,
                    other => return Err(ArgError::UnknownOption(other.into(), "analyze")),
                }
            }
            Command::Analyze { path, json }
        }
        "add" => {
            let (path, flags) = take_path(options, "add")?;
            let mut force = false;
            for flag in flags {
                match flag.as_str() {
                    "--force" => force = true,
                    other => return Err(ArgError::UnknownOption(other.into(), "add")),
                }
            }
            Command::Add { path, force }
        }
        "list" => {
            let mut order_by = "added_ts DESC".to_string();
            let mut it = options.iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--order-by" => {
                        order_by = it
                            .next()
                            .ok_or(ArgError::MissingValue("--order-by", "a column"))?
                            .clone()
                    }
                    other => return Err(ArgError::UnknownOption(other.into(), "list")),
                }
            }
            Command::List { order_by }
        }
        "export" => {
            let (path, flags) = take_path(options, "export")?;
            let mut out = None;
            let mut it = flags.iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--out" => {
                        out = Some(PathBuf::from(
                            it.next().ok_or(ArgError::MissingValue("--out", "a path"))?,
                        ))
                    }
                    other => return Err(ArgError::UnknownOption(other.into(), "export")),
                }
            }
            Command::Export { path, out }
        }
        "migrate" => {
            let mut dry_run = false;
            for flag in options {
                match flag.as_str() {
                    "--dry-run" => dry_run = true,
                    other => return Err(ArgError::UnknownOption(other.into(), "migrate")),
                }
            }
            Command::Migrate { dry_run }
        }
        "transitions" => {
            let (mut from_bpm, mut to_bpm, mut tolerance) = (None, None, 3.0);
            let mut it = options.iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--from" => from_bpm = Some(number(&mut it, "--from")?),
                    "--to" => to_bpm = Some(number(&mut it, "--to")?),
                    "--tolerance" => tolerance = number(&mut it, "--tolerance")?,
                    other => return Err(ArgError::UnknownOption(other.into(), "transitions")),
                }
            }
            Command::Transitions {
                from_bpm: from_bpm.ok_or(ArgError::MissingValue("transitions", "--from <bpm>"))?,
                to_bpm: to_bpm.ok_or(ArgError::MissingValue("transitions", "--to <bpm>"))?,
                tolerance,
            }
        }
        other => return Err(ArgError::UnknownCommand(other.to_string())),
    };

    Ok(Args {
        command,
        config,
        library,
    })
}

/// Take the first non-flag argument as a path, returning the remaining flags.
fn take_path(options: &[String], command: &'static str) -> Result<(PathBuf, Vec<String>), ArgError> {
    let position = options
        .iter()
        .position(|arg| !arg.starts_with("--"))
        .ok_or(ArgError::MissingValue(command, "a file path"))?;
    let path = PathBuf::from(&options[position]);
    let mut flags = options.to_vec();
    flags.remove(position);
    Ok((path, flags))
}

fn number<'a>(
    it: &mut impl Iterator<Item = &'a String>,
    flag: &'static str,
) -> Result<f64, ArgError> {
    let raw = it.next().ok_or(ArgError::MissingValue(flag, "a number"))?;
    raw.parse()
        .map_err(|_| ArgError::NotANumber(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_arguments_shows_help() {
        assert_eq!(parse(&[]).unwrap().command, Command::Help);
    }

    #[test]
    fn help_and_version_are_recognised_in_every_spelling() {
        for spelling in ["help", "--help", "-h"] {
            assert_eq!(parse(&argv(&[spelling])).unwrap().command, Command::Help);
        }
        for spelling in ["version", "--version", "-V"] {
            assert_eq!(parse(&argv(&[spelling])).unwrap().command, Command::Version);
        }
    }

    #[test]
    fn analyze_takes_a_path_and_an_optional_json_flag() {
        let args = parse(&argv(&["analyze", "song.flac"])).unwrap();
        assert_eq!(
            args.command,
            Command::Analyze {
                path: PathBuf::from("song.flac"),
                json: false
            }
        );
        let args = parse(&argv(&["analyze", "--json", "song.flac"])).unwrap();
        assert!(matches!(args.command, Command::Analyze { json: true, .. }));
    }

    #[test]
    fn british_spelling_works_too() {
        assert!(matches!(
            parse(&argv(&["analyse", "song.flac"])).unwrap().command,
            Command::Analyze { .. }
        ));
    }

    #[test]
    fn a_missing_path_is_reported_against_the_command() {
        assert_eq!(
            parse(&argv(&["analyze"])),
            Err(ArgError::MissingValue("analyze", "a file path"))
        );
    }

    #[test]
    fn an_unknown_option_names_the_command_it_was_given_to() {
        assert_eq!(
            parse(&argv(&["add", "x.flac", "--wat"])),
            Err(ArgError::UnknownOption("--wat".into(), "add"))
        );
    }

    #[test]
    fn an_unknown_command_is_reported_verbatim() {
        assert_eq!(
            parse(&argv(&["frobnicate"])),
            Err(ArgError::UnknownCommand("frobnicate".into()))
        );
    }

    #[test]
    fn global_options_are_accepted_before_or_after_the_command() {
        let before = parse(&argv(&["--library", "lib", "list"])).unwrap();
        let after = parse(&argv(&["list", "--library", "lib"])).unwrap();
        assert_eq!(before.library, Some(PathBuf::from("lib")));
        assert_eq!(after.library, Some(PathBuf::from("lib")));
        assert_eq!(before.command, after.command);
    }

    #[test]
    fn the_config_path_defaults_and_can_be_overridden() {
        assert_eq!(parse(&argv(&["list"])).unwrap().config, PathBuf::from("config.json"));
        assert_eq!(
            parse(&argv(&["--config", "other.json", "list"])).unwrap().config,
            PathBuf::from("other.json")
        );
    }

    #[test]
    fn a_global_option_without_its_value_is_reported() {
        assert_eq!(
            parse(&argv(&["--config"])),
            Err(ArgError::MissingValue("--config", "a path"))
        );
    }

    #[test]
    fn transitions_requires_both_tempos() {
        assert_eq!(
            parse(&argv(&["transitions", "--from", "128"])),
            Err(ArgError::MissingValue("transitions", "--to <bpm>"))
        );
        let args = parse(&argv(&["transitions", "--from", "128", "--to", "140"])).unwrap();
        assert_eq!(
            args.command,
            Command::Transitions {
                from_bpm: 128.0,
                to_bpm: 140.0,
                tolerance: 3.0
            }
        );
    }

    #[test]
    fn a_non_numeric_tempo_is_rejected_rather_than_silently_zero() {
        assert_eq!(
            parse(&argv(&["transitions", "--from", "fast", "--to", "140"])),
            Err(ArgError::NotANumber("fast".into()))
        );
    }

    #[test]
    fn export_takes_an_optional_output_path() {
        let args = parse(&argv(&["export", "song.flac", "--out", "out.xml"])).unwrap();
        assert_eq!(
            args.command,
            Command::Export {
                path: PathBuf::from("song.flac"),
                out: Some(PathBuf::from("out.xml"))
            }
        );
    }

    #[test]
    fn list_defaults_to_newest_first() {
        assert_eq!(
            parse(&argv(&["list"])).unwrap().command,
            Command::List {
                order_by: "added_ts DESC".into()
            }
        );
    }

    #[test]
    fn migrate_accepts_a_dry_run() {
        assert_eq!(
            parse(&argv(&["migrate", "--dry-run"])).unwrap().command,
            Command::Migrate { dry_run: true }
        );
    }
}
