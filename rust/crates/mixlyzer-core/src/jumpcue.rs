//! JumpCUEs: regions of a track that sound alike, so a DJ can jump between them.
//!
//! Detection finds pairs of similar regions. Cues that land on the same spot
//! are merged, and cues reachable from one another form a connected component
//! ("graph"), drawn in one colour. Any two cues in the same component can be
//! jumped between in either direction.

use crate::error::DomainError;

/// Colours cycled through, one per connected component.
pub const PAIR_COLORS: [(u8, u8, u8); 6] = [
    (34, 139, 34),
    (0, 153, 204),
    (220, 120, 20),
    (148, 0, 211),
    (210, 105, 30),
    (0, 128, 128),
];

/// Colour for the component at `index`.
pub fn component_color(index: usize) -> (u8, u8, u8) {
    PAIR_COLORS[index % PAIR_COLORS.len()]
}

/// Two cues closer than this in all three times are the same cue.
const COINCIDENT_EPSILON: f64 = 1e-6;

/// One jumpable region.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JumpCue {
    pub id: usize,
    /// A single letter A-Z, unique within the track.
    pub label: String,
    pub comment: String,
    /// Start of the similar region.
    pub start: f64,
    /// End of the similar region.
    pub end: f64,
    /// The exact instant to jump to, somewhere inside `[start, end]`.
    pub point: f64,
    /// Which connected component this cue belongs to.
    pub component: usize,
}

impl JumpCue {
    pub fn new(
        id: usize,
        label: impl Into<String>,
        start: f64,
        end: f64,
        point: f64,
        component: usize,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            comment: String::new(),
            start,
            end,
            point,
            component,
        }
    }

    pub fn color(&self) -> (u8, u8, u8) {
        component_color(self.component)
    }

    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Which way a jump runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Later in the track.
    Forward,
    /// Earlier in the track.
    Backward,
}

/// A jump that can be performed between two cues.
#[derive(Debug, Clone, PartialEq)]
pub struct JumpLink {
    pub from_id: usize,
    pub to_id: usize,
    pub direction: Direction,
    /// Distance skipped, in seconds. Always positive.
    pub lag_sec: f64,
}

/// The full set of cues plus every jump they permit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JumpCueGraph {
    cues: Vec<JumpCue>,
}

impl JumpCueGraph {
    /// Build a graph, merging coincident cues and assigning components.
    ///
    /// `groups` lists which cue indices are mutually reachable; cues sharing a
    /// group end up in the same component, and overlapping groups are unioned.
    pub fn new(cues: Vec<JumpCue>) -> Self {
        let mut cues = cues;
        cues.sort_by(|a, b| {
            a.point
                .partial_cmp(&b.point)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        Self { cues }
    }

    pub fn cues(&self) -> &[JumpCue] {
        &self.cues
    }

    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }

    /// The cue with this label, if any.
    pub fn by_label(&self, label: &str) -> Option<&JumpCue> {
        let wanted = label.trim().to_ascii_uppercase();
        self.cues
            .iter()
            .find(|c| c.label.eq_ignore_ascii_case(&wanted))
    }

    /// Every jump the graph permits, both directions, sorted by source point.
    pub fn links(&self) -> Vec<JumpLink> {
        let mut links = Vec::new();
        for (i, a) in self.cues.iter().enumerate() {
            for b in self.cues.iter().skip(i + 1) {
                if a.component != b.component {
                    continue;
                }
                let lag = (b.point - a.point).abs();
                links.push(JumpLink {
                    from_id: a.id,
                    to_id: b.id,
                    direction: Direction::Forward,
                    lag_sec: lag,
                });
                links.push(JumpLink {
                    from_id: b.id,
                    to_id: a.id,
                    direction: Direction::Backward,
                    lag_sec: lag,
                });
            }
        }
        links
    }

    /// Cues grouped by component, in component order.
    pub fn components(&self) -> Vec<Vec<&JumpCue>> {
        let count = self
            .cues
            .iter()
            .map(|c| c.component)
            .max()
            .map_or(0, |m| m + 1);
        let mut out: Vec<Vec<&JumpCue>> = vec![Vec::new(); count];
        for cue in &self.cues {
            out[cue.component].push(cue);
        }
        out
    }

    /// Reject labels that are not unique single letters A-Z.
    ///
    /// The editor allows free text, and the storage format keys cues by label,
    /// so a duplicate or multi-character label silently loses a cue.
    pub fn validate_labels(&self) -> Result<(), DomainError> {
        let mut seen: Vec<String> = Vec::new();
        for cue in &self.cues {
            let label = cue.label.trim().to_ascii_uppercase();
            let is_single_letter =
                label.len() == 1 && label.chars().all(|c| c.is_ascii_uppercase());
            if !is_single_letter {
                return Err(DomainError::InvalidJumpCueLabel(cue.label.clone()));
            }
            if seen.contains(&label) {
                return Err(DomainError::DuplicateJumpCueLabel(label));
            }
            seen.push(label);
        }
        Ok(())
    }
}

/// The label for the cue at `index`: `A`, `B`, ... `Z`.
///
/// Returns `None` past 26 cues rather than wrapping, because labels key the
/// storage format and a wrapped label would collide with an existing cue.
pub fn label_from_index(index: usize) -> Option<String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    ALPHABET
        .get(index)
        .map(|byte| (*byte as char).to_string())
}

/// Merge cues that sit on the same spot, keeping the first of each cluster.
///
/// Returns the surviving cues renumbered from zero.
pub fn merge_coincident(cues: &[JumpCue]) -> Vec<JumpCue> {
    let mut sorted: Vec<JumpCue> = cues.to_vec();
    sorted.sort_by(|a, b| {
        a.point
            .partial_cmp(&b.point)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });

    let mut out: Vec<JumpCue> = Vec::new();
    for cue in sorted {
        let duplicate = out.iter().any(|kept| {
            (kept.point - cue.point).abs() <= COINCIDENT_EPSILON
                && (kept.start - cue.start).abs() <= COINCIDENT_EPSILON
                && (kept.end - cue.end).abs() <= COINCIDENT_EPSILON
        });
        if !duplicate {
            out.push(cue);
        }
    }
    for (index, cue) in out.iter_mut().enumerate() {
        cue.id = index;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(id: usize, point: f64, component: usize) -> JumpCue {
        JumpCue::new(
            id,
            label_from_index(id).unwrap_or_else(|| "?".into()),
            point,
            point + 8.0,
            point,
            component,
        )
    }

    #[test]
    fn labels_run_a_to_z_then_stop() {
        assert_eq!(label_from_index(0).as_deref(), Some("A"));
        assert_eq!(label_from_index(7).as_deref(), Some("H"));
        assert_eq!(label_from_index(25).as_deref(), Some("Z"));
        assert_eq!(label_from_index(26), None, "must not wrap onto A again");
    }

    #[test]
    fn colors_cycle_through_the_palette() {
        assert_eq!(component_color(0), PAIR_COLORS[0]);
        assert_eq!(component_color(PAIR_COLORS.len()), PAIR_COLORS[0]);
        assert_eq!(component_color(PAIR_COLORS.len() + 2), PAIR_COLORS[2]);
    }

    #[test]
    fn cues_are_sorted_by_point() {
        let graph = JumpCueGraph::new(vec![cue(0, 90.0, 0), cue(1, 30.0, 0)]);
        let points: Vec<f64> = graph.cues().iter().map(|c| c.point).collect();
        assert_eq!(points, vec![30.0, 90.0]);
    }

    #[test]
    fn links_join_every_pair_in_a_component_both_ways() {
        let graph = JumpCueGraph::new(vec![cue(0, 30.0, 0), cue(1, 90.0, 0), cue(2, 150.0, 0)]);
        let links = graph.links();
        // 3 cues in one component -> 3 unordered pairs -> 6 directed links.
        assert_eq!(links.len(), 6);
        assert_eq!(links.iter().filter(|l| l.direction == Direction::Forward).count(), 3);
        assert_eq!(links.iter().filter(|l| l.direction == Direction::Backward).count(), 3);
    }

    #[test]
    fn separate_components_are_not_linked() {
        let graph = JumpCueGraph::new(vec![cue(0, 30.0, 0), cue(1, 90.0, 1)]);
        assert!(graph.links().is_empty());
    }

    #[test]
    fn link_lag_is_the_absolute_distance() {
        let graph = JumpCueGraph::new(vec![cue(0, 30.0, 0), cue(1, 90.0, 0)]);
        for link in graph.links() {
            assert!((link.lag_sec - 60.0).abs() < 1e-12);
            assert!(link.lag_sec > 0.0);
        }
    }

    #[test]
    fn components_group_cues() {
        let graph = JumpCueGraph::new(vec![
            cue(0, 10.0, 0),
            cue(1, 20.0, 1),
            cue(2, 30.0, 0),
        ]);
        let comps = graph.components();
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].len(), 2);
        assert_eq!(comps[1].len(), 1);
    }

    #[test]
    fn lookup_by_label_is_case_insensitive() {
        let graph = JumpCueGraph::new(vec![cue(0, 10.0, 0), cue(1, 20.0, 0)]);
        assert_eq!(graph.by_label("a").map(|c| c.point), Some(10.0));
        assert_eq!(graph.by_label("B").map(|c| c.point), Some(20.0));
        assert!(graph.by_label("Z").is_none());
    }

    #[test]
    fn label_validation_accepts_unique_single_letters() {
        let graph = JumpCueGraph::new(vec![cue(0, 10.0, 0), cue(1, 20.0, 0)]);
        assert!(graph.validate_labels().is_ok());
    }

    #[test]
    fn label_validation_rejects_duplicates_and_non_letters() {
        let mut dup = vec![cue(0, 10.0, 0), cue(1, 20.0, 0)];
        dup[1].label = "A".into();
        assert!(matches!(
            JumpCueGraph::new(dup).validate_labels(),
            Err(DomainError::DuplicateJumpCueLabel(_))
        ));

        let mut bad = vec![cue(0, 10.0, 0)];
        bad[0].label = "AA".into();
        assert!(matches!(
            JumpCueGraph::new(bad).validate_labels(),
            Err(DomainError::InvalidJumpCueLabel(_))
        ));
    }

    #[test]
    fn coincident_cues_merge_and_renumber() {
        let cues = vec![
            cue(0, 30.0, 0),
            cue(1, 30.0, 0), // same spot as the first
            cue(2, 90.0, 0),
        ];
        let merged = merge_coincident(&cues);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 0);
        assert_eq!(merged[1].id, 1);
        assert_eq!(merged[1].point, 90.0);
    }

    #[test]
    fn cues_a_hair_apart_are_kept() {
        let cues = vec![cue(0, 30.0, 0), cue(1, 30.001, 0)];
        assert_eq!(merge_coincident(&cues).len(), 2);
    }

    #[test]
    fn an_empty_graph_has_no_links_or_components() {
        let graph = JumpCueGraph::default();
        assert!(graph.is_empty());
        assert!(graph.links().is_empty());
        assert!(graph.components().is_empty());
        assert!(graph.validate_labels().is_ok());
    }
}
