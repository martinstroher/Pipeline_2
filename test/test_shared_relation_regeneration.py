"""Reproduce the approved GeoPreSalt graph with committed inputs and no network.

Run: python test/test_shared_relation_regeneration.py -v
"""

import hashlib
import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from rdflib import Graph
from rdflib.compare import graph_diff, isomorphic, to_isomorphic
from rdflib.namespace import OWL, RDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FrozenOntologyRegenerationTests(unittest.TestCase):
    def test_approved_graph_is_reproduced_without_network(self):
        inputs = ROOT / "evaluation_study" / "inputs"
        manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
        frozen = ROOT / manifest["approved_ontology_path"]
        digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["approved_ontology_sha256"])
        self.assertEqual(digest, "5b1c950609b5a8b2cd2e2ade78877f769f950c891eb26b84bba166956bd195bb")
        for name, expected in manifest["files"].items():
            with self.subTest(input=name):
                self.assertEqual(hashlib.sha256((inputs / name).read_bytes()).hexdigest(), expected)

        scratch = ROOT / "output" / f"test_frozen_regeneration_{uuid.uuid4().hex}"
        scratch.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, scratch)
        generated = scratch / "regenerated.ttl"
        approved = inputs / "approved_run"

        from src.utils.ontology_config import _build_config
        import yaml

        config_path = ROOT / "domains/presalt/ontology_config.yaml"
        with patch.dict(os.environ, {
            "ONTOLOGY_CONFIG_PATH": str(config_path), "RELATION_PROVENANCE_TIERS": "",
        }), patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            from src.modules.emit import owl_exporter, verifier
            cfg = _build_config(yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path)
            with patch.object(owl_exporter, "_CFG", cfg):
                owl_exporter.run_owl_export(
                    str(approved / "validate_taxonomy.csv"),
                    output_path=str(generated),
                    relations_csv=str(approved / "validate_relations.csv"),
                    instances_csv=str(approved / "validate_instances.csv"),
                    defined_csv=str(approved / "validate_defined_classes.csv"),
                    minted_csv="",
                    disjointness_csv="",
                )
            verifier.run_ontology_verification(
                str(generated), skip_oops=True, skip_reasoner=True
            )

        expected = Graph().parse(frozen, format="turtle")
        actual = Graph().parse(generated, format="turtle")
        equal = isomorphic(expected, actual)
        common, removed, added = graph_diff(to_isomorphic(expected), to_isomorphic(actual))
        self.assertTrue(equal)
        self.assertEqual((len(expected), len(actual), len(common)), (1819, 1819, 1819))
        self.assertEqual((len(removed), len(added)), (0, 0))
        self.assertEqual(len(set(actual.subjects(RDF.type, OWL.Class))), 249)
        self.assertEqual(len(set(actual.subjects(RDF.type, OWL.NamedIndividual))), 59)
        self.assertEqual(len(set(actual.subjects(RDF.type, OWL.Restriction))), 141)
        self.assertEqual(len(list(actual.triples((None, OWL.equivalentClass, None)))), 13)
        report = json.loads((scratch / "emit_verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), digest)
        print(f"Frozen input hashes: {len(manifest['files'])}/{len(manifest['files'])} match")
        print(f"Frozen ontology SHA-256 unchanged: {digest}")
        print(f"Triples: frozen={len(expected)}, regenerated={len(actual)}, common={len(common)}")
        print(f"Graph-isomorphic: {equal}; removed={len(removed)}, added={len(added)}")
        print("Counts: 249 classes, 59 individuals, 141 restrictions, 13 equivalent-class axioms")
        print("Network disabled; syntax/structure PASS; OOPS!/HermiT skipped")


if __name__ == "__main__":
    unittest.main()
