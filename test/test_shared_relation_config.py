"""Shared relation defaults, compatibility, and new-domain scaffold tests.

Run: python -m unittest discover -s test -p 'test_shared_relation*.py' -v
"""

import copy
import hashlib
import io
import json
import os
import shutil
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import new_domain
from src.utils.ontology_config import (
    _build_config,
    _with_relation_defaults,
    get_config,
    reload_config,
)

PRESALT = ROOT / "domains/presalt/ontology_config.yaml"
TEMPLATE = ROOT / "domains/_template/ontology_config.yaml"
SHARED = ROOT / "domains/_shared/bfo_ro_relations.yaml"


def _raw(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize(value):
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


class SharedRelationConfigTests(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "output" / f"test_shared_relations_{uuid.uuid4().hex}"
        self.scratch.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.scratch)
        self.environment = patch.dict(os.environ, {"RELATION_PROVENANCE_TIERS": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.raw = _raw(TEMPLATE)
        self.raw["relation_defaults"] = str(SHARED)
        self.source = self.scratch / "ontology_config.yaml"

    def test_frozen_presalt_relation_contract_including_order(self):
        cfg = _build_config(_raw(PRESALT), PRESALT)
        payload = {
            "groups": _normalize(cfg.metatype_groups),
            "relations": [_normalize(pc) for pc in cfg.all_relations().values()],
            "specializations": _normalize(cfg.property_specializations()),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        # Captured before refactoring main bd2108a; includes every constraint
        # field, notes, critic flags, aliases, group expansion, and rule order.
        self.assertEqual(
            digest,
            "6ce38dda72b7e15e0893ba76b2c1dd22942b570d757d0fdd15b5d6a24544a4b1",
        )

    def test_template_inherits_all_generic_relations_and_rules(self):
        presalt = _build_config(_raw(PRESALT), PRESALT)
        template = _build_config(_raw(TEMPLATE), TEMPLATE)
        generic = {
            name: pc for name, pc in presalt.all_relations().items()
            if pc.iri.startswith("http://purl.obolibrary.org/obo/")
        }
        self.assertEqual(len(generic), 61)
        self.assertEqual(sum("BFO_" in pc.iri for pc in generic.values()), 44)
        self.assertEqual(sum("RO_" in pc.iri for pc in generic.values()), 17)
        self.assertEqual(template.all_relations(), generic)
        self.assertEqual(list(template.relations), list(generic))
        self.assertEqual(template.metatype_groups, presalt.metatype_groups)
        self.assertEqual(template.property_specializations(), presalt.property_specializations())
        self.assertEqual(len(template.property_constraints()), 61)
        self.assertEqual(template.waterfall_ontologies(), ["bfo"])
        self.assertEqual(len(_raw(PRESALT)["relations"]), 10)
        self.assertTrue(all(
            "GEOCORE_" in entry["iri"] or "GEORES_" in entry["iri"]
            for entry in _raw(PRESALT)["relations"].values()
        ))
        self.assertFalse(any("GEO" in pc.iri for pc in template.relations.values()))
        self.assertEqual(len(_raw(SHARED)["relations"]), 61)

    def test_legacy_standalone_config_needs_no_shared_file(self):
        self.raw.pop("relation_defaults")
        self.raw["metatype_groups"] = {"CUSTOM": ["MaterialEntity"]}
        self.raw["property_specializations"] = []
        self.assertIs(_with_relation_defaults(self.raw, self.source), self.raw)
        cfg = _build_config(self.raw, self.source)
        self.assertEqual(cfg.all_relations(), {})
        self.assertEqual(cfg.metatype_groups, {"CUSTOM": frozenset({"MaterialEntity"})})
        self.assertEqual(cfg.property_specializations(), ())

    def test_local_extensions_and_whole_entry_overrides(self):
        self.raw["metatype_groups"] = {
            "CUSTOM": ["MATERIAL", "CustomMeta"],
            "QUALITY": ["CustomQuality"],
        }
        self.raw["relations"] = {
            "custom_relation": {
                "iri": "https://example.org/custom",
                "domain": ["CUSTOM"],
                "range": ["QUALITY"],
                "provenance": "owl_axiom",
            },
            "has_part": {
                "iri": "https://example.org/local-part",
                "domain": ["MATERIAL"],
                "range": ["MATERIAL"],
                "provenance": "owl_axiom",
            },
        }
        original = copy.deepcopy(self.raw)
        cfg = _build_config(self.raw, self.source)
        self.assertEqual(self.raw, original)
        self.assertEqual(len(cfg.relations), 62)
        self.assertEqual(list(cfg.relations)[-1], "custom_relation")
        self.assertEqual(list(cfg.relations)[1], "has_part")
        custom = cfg.relations["custom_relation"]
        self.assertEqual(custom.domain, frozenset({"MaterialEntity", "Object", "CustomMeta"}))
        self.assertEqual(custom.range, frozenset({"CustomQuality"}))
        override = cfg.relations["has_part"]
        self.assertEqual(override.iri, "https://example.org/local-part")
        self.assertIsNone(override.inverse)
        self.assertFalse(override.critic_menu)
        self.assertEqual(override.notes, "")

    def test_specializations_inherit_or_replace_including_empty_list(self):
        inherited = _build_config(self.raw, self.source).property_specializations()
        self.assertEqual(len(inherited), 2)
        self.raw["property_specializations"] = []
        self.assertEqual(_build_config(self.raw, self.source).property_specializations(), ())
        self.raw["property_specializations"] = [
            {"generic": "has_part", "rules": [
                {"when": {"subject_in": "MATERIAL"}, "specialize_to": "has_member_part"}
            ]}
        ]
        specs = _build_config(self.raw, self.source).property_specializations()
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].rules[0].specialize_to, "has_member_part")

    def test_paths_are_relative_to_config_not_working_directory(self):
        local = self.scratch / "domain"
        local.mkdir()
        fragment = self.scratch / "relations.yaml"
        fragment.write_text(SHARED.read_text(encoding="utf-8"), encoding="utf-8")
        self.raw["relation_defaults"] = "../relations.yaml"
        cfg = _build_config(self.raw, local / "ontology_config.yaml")
        self.assertEqual(len(cfg.relations), 61)
        self.assertEqual(cfg._source_path.parent, local)
        self.assertEqual(cfg.ontologies["bfo"].owl_path, local / "resources/bfo-core.owl")

    def test_environment_filter_and_cached_reload_cover_defaults(self):
        self.source.write_text(yaml.safe_dump(self.raw), encoding="utf-8")
        self.addCleanup(get_config.cache_clear)
        with patch.dict(os.environ, {
            "ONTOLOGY_CONFIG_PATH": str(self.source),
            "RELATION_PROVENANCE_TIERS": "ro_release",
        }):
            cfg = reload_config()
            self.assertIs(cfg, get_config())
            self.assertEqual(len(cfg.relations), 61)
            self.assertEqual(len(cfg.property_constraints()), 21)
            with patch.dict(os.environ, {"RELATION_PROVENANCE_TIERS": "owl_axiom"}):
                self.assertIs(cfg, get_config())
                reloaded = reload_config()
                self.assertEqual(len(reloaded.property_constraints()), 40)

    def test_invalid_default_references_fail_explicitly(self):
        for reference in ("", "  ", None, [], 42):
            with self.subTest(reference=reference):
                self.raw["relation_defaults"] = reference
                with self.assertRaisesRegex(RuntimeError, "non-empty file path"):
                    _build_config(self.raw, self.source)
        self.raw["relation_defaults"] = "missing.yaml"
        with self.assertRaisesRegex(RuntimeError, "Cannot load relation_defaults.*missing.yaml"):
            _build_config(self.raw, self.source)

    def test_invalid_fragments_and_nested_imports_are_rejected(self):
        fragment = self.scratch / "bad.yaml"
        self.raw["relation_defaults"] = str(fragment)
        for text in (
            "", "[]", "relations: []", "metatype_groups: null",
            "property_specializations: {}", "relation_defaults: nested.yaml",
            "project: {}", "relations: [", "!include nested.yaml",
            "!!python/object:builtins.object {}",
        ):
            with self.subTest(text=text):
                fragment.write_text(text, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    _build_config(self.raw, self.source)

    def test_invalid_local_sections_are_rejected(self):
        for key, value in (
            ("relations", None), ("metatype_groups", []), ("property_specializations", {})
        ):
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                raw[key] = value
                with self.assertRaisesRegex(RuntimeError, key):
                    _build_config(raw, self.source)

    def test_existing_provenance_and_rule_guards_apply_after_merge(self):
        self.raw["relations"] = {"has_part": {"iri": "https://example.org/incomplete"}}
        with self.assertRaisesRegex(RuntimeError, "missing required 'provenance'"):
            _build_config(self.raw, self.source)
        self.raw["relations"] = {}
        self.raw["property_specializations"] = [
            {"generic": "has_part", "rules": [{"specialize_to": "missing"}]}
        ]
        with self.assertRaisesRegex(RuntimeError, "specialize_to 'missing'"):
            _build_config(self.raw, self.source)

    def test_scaffold_loads_shared_registry_and_copies_reference_resources(self):
        name = f"test_relations_{uuid.uuid4().hex[:10]}"
        target = ROOT / "domains" / name
        self.addCleanup(shutil.rmtree, target, True)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(new_domain.main([name]), 0)
        config_path = target / "ontology_config.yaml"
        cfg = _build_config(_raw(config_path), config_path)
        self.assertEqual(len(cfg.relations), 61)
        self.assertEqual(len(cfg.property_specializations()), 2)
        self.assertEqual(
            {path.name for path in cfg.owl_file_paths()}, {"bfo-core.owl", "ro-core.owl"}
        )
        self.assertEqual(len(list(target.glob("*relations*.yaml"))), 0)
        self.assertEqual(_raw(config_path)["relation_defaults"], "../_shared/bfo_ro_relations.yaml")


if __name__ == "__main__":
    unittest.main()
