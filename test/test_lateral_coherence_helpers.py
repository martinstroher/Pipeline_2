import unittest
import sys
import threading
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
    _build_worthiness_payload,
    _build_worthiness_sibling_context,
    _collect_global_reconciliation_decisions,
    _finalize_reconciliation_audit,
    _generate_completion_nlds,
    _guard_reparent_cycles,
    _merge_worthiness_decision,
    _materialize_bearer_carries,
    _mutual_drop_guard,
)
from src.modules.construct.relation_extractor import extract_relations_for_terms
from src.utils.ontology_config import get_config


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
            candidates = _build_cross_category_candidates(taxonomy, {}, 2)

        pairs = {
            frozenset((candidate["term_a"]["id"], candidate["term_b"]["id"]))
            for candidate in candidates
        }
        self.assertIn(frozenset((0, 1)), pairs)

    def test_top_k_includes_low_similarity_neighbor_for_llm_review(self):
        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0], [0.6, 0.8]]

        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "lacustrine system", "Parent_Term": "system", "Category": "Environment", "NLD": "a lake depositional system"},
            {"_critic_id": 1, "Term": "petroleum system", "Parent_Term": "system", "Category": "Petroleum", "NLD": "a hydrocarbon generation and trapping system"},
        ])
        with patch("src.modules.validate.critic.get_embedding_model", return_value=FakeEmbeddings()):
            candidates = _build_cross_category_candidates(taxonomy, {}, 3)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["weak_similarity_signals"]["nld_cosine"], 0.6)

    def test_top_k_is_selected_before_reverse_pair_deduplication(self):
        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0], [0.9848, 0.1736], [0.866, 0.5]]

        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "alpha", "Category": "A", "NLD": "alpha kind"},
            {"_critic_id": 1, "Term": "beta", "Category": "B", "NLD": "beta kind"},
            {"_critic_id": 2, "Term": "gamma", "Category": "C", "NLD": "gamma kind"},
        ])
        with patch("src.modules.validate.critic.get_embedding_model", return_value=FakeEmbeddings()):
            candidates = _build_cross_category_candidates(taxonomy, {}, 1)

        pairs = {
            frozenset((candidate["term_a"]["id"], candidate["term_b"]["id"]))
            for candidate in candidates
        }
        self.assertEqual(pairs, {frozenset((0, 1)), frozenset((1, 2))})

    def test_global_reconciliation_batches_run_concurrently_in_pair_order(self):
        second_batch_finished = threading.Event()

        def invoke_batch(batch):
            pair_id = batch[0]["pair_id"]
            if pair_id == 0:
                self.assertTrue(second_batch_finished.wait(timeout=2))
            elif pair_id == 1:
                second_batch_finished.set()
            if pair_id == 2:
                return []
            return [{"pair_id": pair_id, "decision": "DISTINCT"}]

        candidates = [{"pair_id": pair_id} for pair_id in range(3)]
        decisions = _collect_global_reconciliation_decisions(
            candidates, batch_size=1, max_workers=2, invoke_batch=invoke_batch,
        )

        self.assertEqual([decision["pair_id"] for decision in decisions], [0, 1, 2])
        self.assertEqual(
            [decision["decision"] for decision in decisions],
            ["DISTINCT", "DISTINCT", "NEEDS_REVIEW"],
        )

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

    def test_reconciliation_does_not_override_core_exclusion(self):
        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "detail", "Parent_Term": "material", "Category": "A"},
            {"_critic_id": 1, "Term": "kind", "Parent_Term": "material", "Category": "B"},
        ])
        candidates = [{
            "pair_id": 0,
            "term_a": {"id": 0, "label": "detail", "category": "A"},
            "term_b": {"id": 1, "label": "kind", "category": "B"},
        }]
        decisions = [{
            "pair_id": 0, "decision": "SAME_KIND", "survivor_id": 1,
            "duplicate_id": 0, "confidence": 0.95,
        }]
        edits = {0: {"action": "DROP_AS_OVER_SPECIFIC"}}

        aliases, audit = _apply_cross_category_reconciliations(
            candidates, decisions, taxonomy, edits, 0.7,
        )

        self.assertEqual(edits[0]["action"], "DROP_AS_OVER_SPECIFIC")
        self.assertEqual(aliases, {})
        self.assertFalse(audit[0]["Applied"])

    def test_conflicting_reconciliation_parents_are_not_applied(self):
        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "child", "Parent_Term": "root", "Category": "A"},
            {"_critic_id": 1, "Term": "parent one", "Parent_Term": "root", "Category": "B"},
            {"_critic_id": 2, "Term": "parent two", "Parent_Term": "root", "Category": "C"},
        ])
        candidates = [
            {"pair_id": 0, "term_a": {"id": 0, "label": "child", "category": "A"},
             "term_b": {"id": 1, "label": "parent one", "category": "B"}},
            {"pair_id": 1, "term_a": {"id": 0, "label": "child", "category": "A"},
             "term_b": {"id": 2, "label": "parent two", "category": "C"}},
        ]
        decisions = [
            {"pair_id": 0, "decision": "A_SUBCLASS_OF_B", "confidence": 0.95},
            {"pair_id": 1, "decision": "A_SUBCLASS_OF_B", "confidence": 0.91},
        ]
        edits = {}

        _, audit = _apply_cross_category_reconciliations(
            candidates, decisions, taxonomy, edits, 0.7,
        )

        self.assertNotIn(0, edits)
        self.assertTrue(all(not row["Applied"] and row["Needs_Review"] for row in audit))

    def test_reconciliation_audit_matches_materialized_taxonomy(self):
        audit = [{
            "Applied": True, "Needs_Review": False, "Reason": "candidate",
            "Child": "child", "New_Parent": "proposed", "Duplicate": "",
        }]
        final_taxonomy = pd.DataFrame([
            {"Term": "child", "Parent_Term": "final parent"},
            {"Term": "proposed", "Parent_Term": "root"},
        ])

        _finalize_reconciliation_audit(audit, final_taxonomy, {})

        self.assertFalse(audit[0]["Applied"])
        self.assertTrue(audit[0]["Needs_Review"])
        self.assertIn("not reflected", audit[0]["Reason"])

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

    def test_dropped_parent_does_not_alias_child_to_itself(self):
        taxonomy = pd.DataFrame([
            {"_critic_id": 0, "Term": "diagenesis", "Parent_Term": "Geological Process", "Category": "Geological Process"},
            {"_critic_id": 1, "Term": "early diagenesis", "Parent_Term": "diagenesis", "Category": "Geological Process"},
            {"_critic_id": 2, "Term": "eodiagenesis", "Parent_Term": "early diagenesis", "Category": "Geological Process"},
        ])
        edits = {
            0: {"action": "KEEP"},
            1: {"action": "DROP_AS_REDUNDANT", "survivor": "eodiagenesis"},
            2: {"action": "KEEP"},
        }

        cleaned, *_ = _apply_taxonomy_edits(taxonomy, edits, {"Geological Process"})

        self.assertEqual(cleaned.set_index("Term").loc["eodiagenesis", "Parent_Term"], "diagenesis")

    def test_upper_class_identity_suppresses_self_parent(self):
        taxonomy = pd.DataFrame([{
            "_critic_id": 0,
            "Term": "sedimentary facies",
            "Parent_Term": "Sedimentary Facies",
            "Category": "Sedimentary Facies",
        }])

        cleaned, *_ = _apply_taxonomy_edits(
            taxonomy, {0: {"action": "KEEP"}}, {"Sedimentary Facies"},
        )

        self.assertEqual(cleaned.iloc[0]["Parent_Term"], "")

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

    def test_incomplete_demotion_keeps_primitive_audit_fate(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {"fate": "DEMOTE_TO_PROPERTY", "demoted_as": {}, "confidence": 0.9},
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "KEEP")
        self.assertEqual(merged["class_fate"], "primitive")
        self.assertTrue(merged["needs_review"])

    def test_incomplete_definition_keeps_primitive_audit_fate(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {"fate": "KEEP_DEFINED", "defined_by": {}, "confidence": 0.9},
            True,
        )

        self.assertEqual(merged["action"], "KEEP")
        self.assertEqual(merged["class_fate"], "primitive")
        self.assertTrue(merged["needs_review"])

    def test_valid_core_exclusion_applies_even_when_needs_review(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {
                "fate": "DROP_CLASS", "drop_basis": "NARROW_EXTENSION_DETAIL",
                "confidence": 0.55, "needs_review": True,
                "reason": "valid detail but adds no marginal core value",
            },
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "DROP_AS_OVER_SPECIFIC")
        self.assertEqual(merged["drop_basis"], "NARROW_EXTENSION_DETAIL")
        self.assertTrue(merged["needs_review"])

    def test_drop_without_core_exclusion_basis_is_kept(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {
                "fate": "DROP_CLASS", "confidence": 0.95,
                "reason": "duplicate of a sibling",
            },
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "KEEP")
        self.assertTrue(merged["needs_review"])

    def test_invalid_drop_preserves_existing_bearer_audit_fate(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP_AS_BEARER"},
            {"fate": "DROP_CLASS", "confidence": 0.95, "reason": "duplicate"},
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "KEEP_AS_BEARER")
        self.assertEqual(merged["class_fate"], "defined")

    def test_realizable_bearer_rejects_marginality_drop(self):
        merged = _merge_worthiness_decision(
            {
                "action": "KEEP_AS_BEARER",
                "carried_by": {"property": "has_disposition", "filler": "StorageDisposition"},
            },
            {
                "fate": "DROP_CLASS", "drop_basis": "NO_MARGINAL_VALUE",
                "confidence": 0.9, "reason": "parent covers its CQ",
            },
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "KEEP_AS_BEARER")
        self.assertEqual(merged["class_fate"], "defined")
        self.assertTrue(merged["needs_review"])

    def test_realizable_bearer_rejects_property_demotion(self):
        merged = _merge_worthiness_decision(
            {
                "action": "KEEP_AS_BEARER",
                "carried_by": {"property": "has_role", "filler": "CarrierRole"},
            },
            {
                "fate": "DEMOTE_TO_PROPERTY",
                "demoted_as": {"base_class": "Material", "property": "has_quality", "filler": "Useful"},
                "confidence": 0.95,
            },
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "KEEP_AS_BEARER")
        self.assertEqual(merged["class_fate"], "defined")

    def test_quality_bearer_can_still_be_demoted(self):
        merged = _merge_worthiness_decision(
            {
                "action": "KEEP_AS_BEARER",
                "carried_by": {"property": "has_quality", "filler": "FlowSimilarity"},
            },
            {
                "fate": "DEMOTE_TO_PROPERTY",
                "demoted_as": {"base_class": "Material", "property": "has_quality", "filler": "FlowSimilarity"},
                "confidence": 0.9,
            },
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "DEMOTE_TO_PROPERTY")
        self.assertEqual(merged["class_fate"], "demote")

    def test_atomic_coherent_frame_member_rejects_narrow_detail_drop(self):
        merged = _merge_worthiness_decision(
            {"action": "KEEP"},
            {
                "fate": "DROP_CLASS", "drop_basis": "NARROW_EXTENSION_DETAIL",
                "coherent_frame": True, "frame_axis": "composition",
                "frame_siblings": ["Species A", "Species B"],
                "confidence": 0.9,
            },
            True,
            conservative_drop=False,
        )

        self.assertEqual(merged["action"], "KEEP")
        self.assertEqual(merged["core_basis"], "COHERENT_FRAME_MEMBER")
        self.assertTrue(merged["needs_review"])

    def test_worthiness_sibling_context_contains_comparative_evidence(self):
        rows = [
            pd.Series({"_critic_id": 0, "Term": "parent", "Parent_Term": "root", "NLD": "A parent."}),
            pd.Series({"_critic_id": 1, "Term": "child", "Parent_Term": "parent", "NLD": "A child."}),
        ]
        evidence = {
            "parent": {"frequency": 8, "cq_count": 2, "matched_cqs": ["CQ1", "CQ2"]},
            "child": {"frequency": 5, "cq_count": 1, "matched_cqs": ["CQ1"]},
        }

        context = _build_worthiness_sibling_context(rows, {}, evidence)

        by_term = {row["term"]: row for row in context}
        self.assertEqual(by_term["parent"]["child_count"], 1)
        self.assertEqual(by_term["parent"]["matched_cqs"], ["CQ1", "CQ2"])

    def test_worthiness_payload_contains_definition_and_relation_mentions(self):
        rows = [pd.Series({
            "_critic_id": 7, "Term": "carrier", "Parent_Term": "material",
            "Category": "material", "NLD": "A carrier material.",
        })]
        edits = {7: {
            "action": "KEEP_AS_BEARER",
            "carried_by": {"property": "has_role", "filler": "CarrierRole"},
        }}

        payload = _build_worthiness_payload(
            rows, edits, {"carrier": {"frequency": 6}}, {"carrier": 4},
        )

        self.assertEqual(payload[0]["taxonomy_definition"]["property"], "has_role")
        self.assertEqual(payload[0]["relation_mentions"], 4)

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

    def test_frame_completion_auto_add_is_disabled_by_default(self):
        self.assertTrue(get_config().lateral_coherence().frame_completion_enabled)
        self.assertFalse(get_config().lateral_coherence().frame_completion_auto_add)


if __name__ == "__main__":
    unittest.main()
