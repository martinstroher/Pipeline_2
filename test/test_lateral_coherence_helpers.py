import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.validate.critic import (
    _apply_frame_completion,
    _apply_taxonomy_edits,
    _merge_worthiness_decision,
    _materialize_bearer_carries,
    _mutual_drop_guard,
)


class LateralCoherenceHelperTests(unittest.TestCase):
    def test_dropped_parent_collapses_to_survivor(self):
        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "silica", "Parent_Term": "Earth Material", "Category": "Earth Material"},
            {"_critic_id": 1, "Term": "Silica Phase", "Parent_Term": "silica", "Category": "Earth Material"},
            {"_critic_id": 2, "Term": "chalcedony", "Parent_Term": "Silica Phase", "Category": "Earth Material"},
        ])
        edits = {
            0: {"action": "KEEP"},
            1: {"action": "DROP_AS_REDUNDANT", "survivor": "silica"},
            2: {"action": "KEEP"},
        }

        cleaned, *_ = _apply_taxonomy_edits(taxonomy, edits, {"Earth Material"})

        self.assertEqual(cleaned.set_index("Term").loc["chalcedony", "Parent_Term"], "silica")

    def test_worthiness_demote_is_authoritative(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {
                "fate": "DEMOTE_TO_PROPERTY",
                "demoted_as": {
                    "base_class": "Mineral",
                    "property": "has_quality",
                    "filler": "CrystalHabit",
                },
                "confidence": 0.9,
            },
            True,
        )

        self.assertEqual(merged["action"], "DEMOTE_TO_PROPERTY")
        self.assertEqual(merged["class_fate"], "demote")

    def test_low_confidence_drop_is_kept_for_review(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {"fate": "DROP_CLASS", "confidence": 0.4, "reason": "uncertain"},
            True,
            conservative_drop=True,
            min_confidence_apply=0.7,
            needs_review_below=0.85,
        )

        self.assertEqual(merged["action"], "KEEP")
        self.assertEqual(merged["proposed_fate"], "drop")
        self.assertEqual(merged["class_fate"], "primitive")
        self.assertTrue(merged["needs_review"])

    def test_mutual_drop_restores_pre_dedup_action(self):
        edits = {
            1: {
                "action": "DROP_AS_REDUNDANT",
                "survivor": "other",
                "_pre_dedup_edit": {"action": "KEEP_AS_DEFINED", "defined_by": {"base_class": "Base"}},
            },
            2: {"action": "DROP_AS_OVER_SPECIFIC"},
        }
        _mutual_drop_guard(edits, {1: "term", 2: "other"}, {1: "parent", 2: "parent"})

        self.assertEqual(edits[1]["action"], "KEEP_AS_DEFINED")

    def test_bearer_companion_has_generic_scope(self):
        _, relations = _materialize_bearer_carries(
            [{
                "bearer": "reservoir rock", "bearer_category": "Rock",
                "property": "has_role", "filler": "ReservoirRole",
                "filler_parent": "role", "filler_nld": "A role borne by rock.",
            }],
            None,
            [
                "Term", "Category", "Property", "Property_IRI", "Filler",
                "Filler_Source", "Confidence", "Evidence", "Validation_Status",
                "Validation_Reason", "Relation_Scope", "Scope_Reason",
                "Scope_Confidence", "Scope_Needs_Review",
            ],
        )

        self.assertEqual(relations.iloc[0]["Relation_Scope"], "generic")
        self.assertEqual(relations.iloc[0]["Scope_Confidence"], 1.0)

    def test_frame_completion_adds_only_accepted_candidate(self):
        taxonomy = pd.DataFrame([
            {
                "_critic_id": 0,
                "Term": "process stage",
                "Parent_Term": "process",
                "Relationship_Type": "rdfs:subClassOf",
                "Category": "process",
                "Is_Intermediate": False,
                "NLD": "A process stage is a process that occupies a phase.",
                "FALLBACK": False,
            }
        ])
        audit_rows = [{
            "Candidate_ID": 1,
            "Candidate": "later stage",
            "Parent_Term": "process stage",
            "Category": "process",
            "Status": "ATTESTED",
            "NLD": "",
            "Decision_Reason": "",
        }]
        decisions = [{
            "candidate_id": 1,
            "accept": True,
            "nld": "A later stage is a process stage that occurs later.",
            "reason": "attested conventional kind",
        }]

        completed, audit = _apply_frame_completion(taxonomy, audit_rows, decisions, {"process"})

        self.assertIn("later stage", completed["Term"].tolist())
        self.assertEqual(audit.iloc[0]["Status"], "ADDED")


if __name__ == "__main__":
    unittest.main()
