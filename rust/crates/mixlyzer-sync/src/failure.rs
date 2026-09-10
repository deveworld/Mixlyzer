//! What to do when a poll fails.
//!
//! This replaces Python's `_disable_due_to_failure`, which reacts to the
//! *first* failure of any kind by setting `enabled = False`, writing that back
//! to `config.json` and emitting `sig_external_sync_enabled(False)`. The user
//! then has to find the setting and turn it on again. Since a deck pointer is
//! null every time the DJ program loads a track, and a read straddling a
//! reallocation fails now and then, the feature switches itself off during
//! ordinary use — permanently, because the config file has been rewritten.
//!
//! The policy here has three steps:
//!
//! 1. Tolerate a few consecutive failures; nothing changes.
//! 2. Past that, back off — skip polls, doubling the gap up to a cap — so a
//!    process that is wedged is not hammered 60 times a second.
//! 3. Disable only on a permanent condition ([`Severity::Permanent`]): the
//!    process exited, it is on the denylist, the configuration cannot be
//!    parsed, or there is no backend for this platform.
//!
//! Nothing here writes to `config.json`. Disabling is a fact about this
//! session, reported through [`SyncStatus`], and [`FailureTracker::reset`]
//! clears it.

use crate::error::{Severity, SyncError};

/// How tolerant the engine is of failing polls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailurePolicy {
    /// Consecutive failures allowed before backing off.
    pub tolerated_failures: u32,
    /// Polls skipped at the first backoff step; doubles each further failure.
    pub backoff_polls: u32,
    /// Ceiling for the backoff, in skipped polls.
    pub max_backoff_polls: u32,
    /// Consecutive failures after which even transient trouble gives up.
    ///
    /// `None` — the default — means transient failures never disable the
    /// feature; they only slow it down. Set it if a host wants a hard stop.
    pub disable_after: Option<u32>,
}

impl Default for FailurePolicy {
    fn default() -> Self {
        Self {
            // At 60 Hz these are fractions of a second: a track load may null a
            // pointer for several frames.
            tolerated_failures: 5,
            backoff_polls: 8,
            max_backoff_polls: 240, // ~4 s at 60 Hz
            disable_after: None,
        }
    }
}

/// Why the engine stopped polling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisableReason {
    /// The target is on the denylist.
    Denied(String),
    /// The target process exited.
    ProcessGone,
    /// The configuration cannot be used as written.
    InvalidConfiguration(String),
    /// This platform has no memory backend.
    Unsupported(String),
    /// `disable_after` was reached.
    TooManyFailures {
        /// How many consecutive failures were seen.
        consecutive: u32,
        /// The last failure's message.
        last: String,
    },
    /// The host asked the engine to stop.
    Stopped,
}

impl std::fmt::Display for DisableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DisableReason::Denied(detail) => write!(f, "{detail}"),
            DisableReason::ProcessGone => write!(f, "the target process has exited"),
            DisableReason::InvalidConfiguration(detail) => {
                write!(f, "external sync is misconfigured: {detail}")
            }
            DisableReason::Unsupported(detail) => write!(f, "{detail}"),
            DisableReason::TooManyFailures { consecutive, last } => write!(
                f,
                "gave up after {consecutive} consecutive failures, last: {last}"
            ),
            DisableReason::Stopped => write!(f, "stopped by the application"),
        }
    }
}

impl DisableReason {
    fn from_permanent(err: &SyncError) -> Self {
        match err {
            SyncError::ProcessDenied { .. } => DisableReason::Denied(err.to_string()),
            SyncError::ProcessGone => DisableReason::ProcessGone,
            SyncError::Unsupported(_) => DisableReason::Unsupported(err.to_string()),
            other => DisableReason::InvalidConfiguration(other.to_string()),
        }
    }
}

/// What the engine is doing right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncStatus {
    /// Polling normally.
    Active,
    /// Failing, but still polling within the tolerance.
    Struggling {
        /// Failures seen in a row so far.
        consecutive_failures: u32,
    },
    /// Skipping polls after too many consecutive failures.
    BackingOff {
        /// Polls still to be skipped before trying again.
        polls_remaining: u32,
    },
    /// Stopped for good, this session.
    Disabled(DisableReason),
}

/// Applies a [`FailurePolicy`] to a stream of poll results.
#[derive(Debug, Clone)]
pub struct FailureTracker {
    policy: FailurePolicy,
    consecutive: u32,
    skip_remaining: u32,
    disabled: Option<DisableReason>,
}

impl FailureTracker {
    /// A tracker with the given policy.
    pub fn new(policy: FailurePolicy) -> Self {
        Self {
            policy,
            consecutive: 0,
            skip_remaining: 0,
            disabled: None,
        }
    }

    /// The policy in force.
    pub fn policy(&self) -> &FailurePolicy {
        &self.policy
    }

    /// Whether this poll should be skipped because of backoff.
    ///
    /// Consumes one poll of the backoff budget.
    pub fn should_skip(&mut self) -> bool {
        if self.skip_remaining == 0 {
            return false;
        }
        self.skip_remaining -= 1;
        true
    }

    /// Record a poll that worked. Clears the failure streak and any backoff.
    pub fn record_success(&mut self) {
        self.consecutive = 0;
        self.skip_remaining = 0;
    }

    /// Record a failed poll and decide what happens next.
    pub fn record_failure(&mut self, err: &SyncError) {
        if err.severity() == Severity::Permanent {
            self.disabled = Some(DisableReason::from_permanent(err));
            self.skip_remaining = 0;
            return;
        }
        self.consecutive = self.consecutive.saturating_add(1);
        if let Some(limit) = self.policy.disable_after {
            if self.consecutive >= limit {
                self.disabled = Some(DisableReason::TooManyFailures {
                    consecutive: self.consecutive,
                    last: err.to_string(),
                });
                return;
            }
        }
        if self.consecutive > self.policy.tolerated_failures {
            let steps = self.consecutive - self.policy.tolerated_failures - 1;
            let scaled = self
                .policy
                .backoff_polls
                .saturating_mul(1u32.checked_shl(steps.min(31)).unwrap_or(u32::MAX));
            self.skip_remaining = scaled.min(self.policy.max_backoff_polls);
        }
    }

    /// Stop polling for a reason of the host's own.
    pub fn disable(&mut self, reason: DisableReason) {
        self.disabled = Some(reason);
    }

    /// Why polling stopped, if it did.
    pub fn disable_reason(&self) -> Option<&DisableReason> {
        self.disabled.as_ref()
    }

    /// Whether polling has stopped for good.
    pub fn is_disabled(&self) -> bool {
        self.disabled.is_some()
    }

    /// Current state, for the status line.
    pub fn status(&self) -> SyncStatus {
        match &self.disabled {
            Some(reason) => SyncStatus::Disabled(reason.clone()),
            None if self.skip_remaining > 0 => SyncStatus::BackingOff {
                polls_remaining: self.skip_remaining,
            },
            None if self.consecutive > 0 => SyncStatus::Struggling {
                consecutive_failures: self.consecutive,
            },
            None => SyncStatus::Active,
        }
    }

    /// Consecutive failures so far.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive
    }

    /// Clear everything, including a disable. This is what "the user turned it
    /// back on" does — and unlike Python it is the only way the setting
    /// changes, because nothing here edits `config.json`.
    pub fn reset(&mut self) {
        self.consecutive = 0;
        self.skip_remaining = 0;
        self.disabled = None;
    }
}

impl Default for FailureTracker {
    fn default() -> Self {
        Self::new(FailurePolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transient() -> SyncError {
        SyncError::NullPointer { step: 1 }
    }

    /// The behaviour Python gets wrong: one null pointer must not end the
    /// session.
    #[test]
    fn a_single_transient_failure_is_tolerated_and_then_forgotten() {
        let mut tracker = FailureTracker::default();
        tracker.record_failure(&transient());
        assert!(!tracker.is_disabled());
        assert_eq!(
            tracker.status(),
            SyncStatus::Struggling {
                consecutive_failures: 1
            }
        );
        assert!(!tracker.should_skip(), "still polling every frame");

        tracker.record_success();
        assert_eq!(tracker.status(), SyncStatus::Active);
        assert_eq!(tracker.consecutive_failures(), 0);
    }

    #[test]
    fn repeated_transient_failures_back_off_but_never_disable() {
        let mut tracker = FailureTracker::default();
        for _ in 0..5 {
            tracker.record_failure(&transient());
        }
        assert!(!tracker.should_skip(), "5 is within tolerance");

        tracker.record_failure(&transient());
        assert_eq!(
            tracker.status(),
            SyncStatus::BackingOff { polls_remaining: 8 }
        );

        for _ in 0..1_000 {
            tracker.record_failure(&transient());
        }
        assert!(!tracker.is_disabled(), "transient trouble never disables");
        assert_eq!(
            tracker.status(),
            SyncStatus::BackingOff {
                polls_remaining: 240
            },
            "backoff is capped"
        );
    }

    #[test]
    fn the_backoff_doubles() {
        let policy = FailurePolicy {
            tolerated_failures: 0,
            backoff_polls: 2,
            max_backoff_polls: 16,
            disable_after: None,
        };
        let mut tracker = FailureTracker::new(policy);
        for expected in [2u32, 4, 8, 16, 16] {
            tracker.record_failure(&transient());
            assert_eq!(
                tracker.status(),
                SyncStatus::BackingOff {
                    polls_remaining: expected
                }
            );
            while tracker.should_skip() {}
        }
    }

    #[test]
    fn skipping_consumes_the_backoff_one_poll_at_a_time() {
        let policy = FailurePolicy {
            tolerated_failures: 0,
            backoff_polls: 3,
            max_backoff_polls: 3,
            disable_after: None,
        };
        let mut tracker = FailureTracker::new(policy);
        tracker.record_failure(&transient());
        assert!(tracker.should_skip());
        assert!(tracker.should_skip());
        assert!(tracker.should_skip());
        assert!(!tracker.should_skip(), "budget spent, poll again");
    }

    #[test]
    fn a_success_cancels_an_outstanding_backoff() {
        let policy = FailurePolicy {
            tolerated_failures: 0,
            ..FailurePolicy::default()
        };
        let mut tracker = FailureTracker::new(policy);
        tracker.record_failure(&transient());
        assert!(matches!(tracker.status(), SyncStatus::BackingOff { .. }));
        tracker.record_success();
        assert_eq!(tracker.status(), SyncStatus::Active);
        assert!(!tracker.should_skip());
    }

    #[test]
    fn a_permanent_failure_disables_immediately() {
        let mut tracker = FailureTracker::default();
        tracker.record_failure(&SyncError::ProcessGone);
        assert_eq!(
            tracker.status(),
            SyncStatus::Disabled(DisableReason::ProcessGone)
        );
    }

    #[test]
    fn a_denied_process_disables_with_the_rule_in_the_message() {
        let mut tracker = FailureTracker::default();
        tracker.record_failure(&SyncError::ProcessDenied {
            name: "lsass.exe".into(),
            reason: "process name \"lsass\" is on the denylist".into(),
        });
        let SyncStatus::Disabled(reason) = tracker.status() else {
            panic!("expected a disable");
        };
        assert!(reason.to_string().contains("lsass"), "{reason}");
    }

    #[test]
    fn a_hard_limit_can_be_opted_into() {
        let mut tracker = FailureTracker::new(FailurePolicy {
            disable_after: Some(3),
            ..FailurePolicy::default()
        });
        tracker.record_failure(&transient());
        tracker.record_failure(&transient());
        assert!(!tracker.is_disabled());
        tracker.record_failure(&transient());
        assert!(matches!(
            tracker.status(),
            SyncStatus::Disabled(DisableReason::TooManyFailures { consecutive: 3, .. })
        ));
    }

    #[test]
    fn resetting_clears_a_disable() {
        let mut tracker = FailureTracker::default();
        tracker.record_failure(&SyncError::ProcessGone);
        assert!(tracker.is_disabled());
        tracker.reset();
        assert!(!tracker.is_disabled());
        assert_eq!(tracker.status(), SyncStatus::Active);
    }

    #[test]
    fn the_host_can_stop_the_engine_itself() {
        let mut tracker = FailureTracker::default();
        tracker.disable(DisableReason::Stopped);
        assert_eq!(
            tracker.disable_reason(),
            Some(&DisableReason::Stopped),
            "the host's own stop is reported like any other"
        );
    }
}
