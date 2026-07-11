import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.validate.critic import (
    _apply_frame_completion,
    _apply_cross_category_reconciliations,
    _apply_taxonomy_edits,
    _build_cross_category_candidates,
    _build_entity_context_lookup,
    _build_relation_payload,
    _generate_completion_nlds,
    _guard_reparent_cycles,
    _merge_worthiness_decision,
    _materialize_bearer_carries,
    _mutual_drop_guard,
)
from src.modules.construct.relation_extractor import extract_relations_for_terms


class LateralCoherenceHelperTests(unittest.TestCase):
    def test_cross_category_candidates_use_nld_similarity_only_as_shortlist(self):
        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]

        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "carbonate", "Parent_Term": "material", "Category": "Material", "NLD": "carbonate material"},
            {"_critic_id": 1, "Term": "carbonate rock", "Parent_Term": "rock", "Category": "Rock", "NLD": "rock made of carbonate"},
            {"_critic_id": 2, "Term": "fault", "Parent_Term": "structure", "Category": "Structure", "NLD": "displaced fracture"},
        ])
        with patch("src.modules.validate.critic.get_embedding_model", return_value=FakeEmbeddings()):
            candidates = _build_cross_category_candidates(taxonomy, {}, 2, 0.8, 0.85)

        pairs = {
            frozenset((candidate["term_a"]["id"], candidate["term_b"]["id"]))
            for candidate in candidates
        }
        self.assertIn(frozenset((0, 1)), pairs)

    def test_cross_category_same_kind_creates_alias(self):
        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "carbonate", "Parent_Term": "material", "Category": "Material"},
            {"_critic_id": 1, "Term": "carbonate rock", "Parent_Term": "rock", "Category": "Rock"},
        ])
        candidates = [{
            "pair_id": 0,
            "term_a": {"id": 0, "label": "carbonate", "category": "Material"},
            "term_b": {"id": 1, "label": "carbonate rock", "category": "Rock"},
        }]
        decisions = [{
            "pair_id": 0, "decision": "SAME_KIND", "survivor_id": 1,
            "duplicate_id": 0, "confidence": 0.95, "reason": "same extension",
        }]
        edits = {}

        aliases, audit = _apply_cross_category_reconciliations(
            candidates, decisions, taxonomy, edits, 0.7,
        )

        self.assertEqual(aliases["carbonate"], "carbonate rock")
        self.assertEqual(edits[0]["action"], "DROP_AS_REDUNDANT")
        self.assertTrue(audit[0]["Applied"])

    def test_reparent_cycle_is_reverted(self):
        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "A", "Parent_Term": "Root", "Category": "Root"},
            {"_critic_id": 1, "Term": "B", "Parent_Term": "A", "Category": "Root"},
        ])
        edits = {0: {"action": "REPARENT", "new_parent": "B", "reason": "bad cycle"}}

        reverted = _guard_reparent_cycles(taxonomy, edits)

        self.assertEqual(reverted, ["A"])
        self.assertEqual(edits[0]["action"], "KEEP")

    def test_relation_payload_contains_subject_and_filler_context(self):
        taxonomy = pd.DataFrame([
            {"Term": "Process A", "Parent_Term": "process", "Category": "process", "NLD": "A process."},
            {"Term": "Material B", "Parent_Term": "object", "Category": "object", "NLD": "A material."},
        ])
        relations = pd.DataFrame([
            {"_critic_id": 1, "Term": "Process A", "Property": "has_participant", "Filler": "Material B", "Evidence": "B participates"},
        ])
        context = _build_entity_context_lookup(taxonomy)

        payload = _build_relation_payload(relations, pd.DataFrame(), context)

        self.assertEqual(payload[0]["subject_context"]["nld"], "A process.")
        self.assertEqual(payload[0]["filler_context"]["nld"], "A material.")

    def test_completion_uses_standard_nld_generator(self):
        evidence = [{
            "candidate_id": 1, "label": "New Kind",
            "evidence": [{"source": "doc", "excerpt": "New Kind is a Process Stage."}],
        }]
        decisions = [{"candidate_id": 1, "accept": True, "reason": "attested"}]
        with patch(
            "src.modules.define.nld_generator.generate_nld",
            return_value=('{"Definition": "New Kind is a Process Stage that is attested."}', "prompt"),
        ):
            enriched = _generate_completion_nlds(evidence, decisions)

        self.assertTrue(enriched[0]["accept"])
        self.assertIn("Process Stage", enriched[0]["nld"])

    def test_targeted_new_term_relation_extraction_uses_standard_validator(self):
        known = pd.DataFrame([
            {"Term": "dolomitization", "Category": "Geological Process"},
            {"Term": "calcite", "Category": "Earth Material"},
        ])
        generated = [{
            "relations": [{
                "property": "has_participant", "filler": "calcite",
                "confidence": 1.0, "evidence": "calcite participates in dolomitization",
            }]
        }]
        with patch("src.modules.construct.relation_extractor._extract_batch", return_value=generated):
            rows = extract_relations_for_terms(
                [{
                    "term": "dolomitization",
                    "nld": "Dolomitization is a geological process that replaces calcite.",
                    "category": "Geological Process",
                }],
                known,
            )

        self.assertEqual(rows.iloc[0]["Validation_Status"], "ACCEPTED")
        self.assertEqual(rows.iloc[0]["Property"], "has_participant")

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
