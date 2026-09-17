"""Offline checks for complete, isolated domain scaffolding."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from scripts import new_domain
from src.utils.domain_validation import PROMPT_FIELDS, validate_domain
from src.utils.ontology_config import get_config, load_config
from src.utils.prompt_loader import load_prompt, load_prompt_blocks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for relative in ("domains/_template", "domains/_shared", "domains/presalt/prompts", "domains/presalt/resources"):
        shutil.copytree(ROOT / relative, root / relative)
    monkeypatch.setattr(new_domain, "_REPO_ROOT", root)
    monkeypatch.setattr(new_domain, "_TEMPLATE_DIR", root / "domains/_template")
    monkeypatch.setattr(new_domain, "_PRESALT_DIR", root / "domains/presalt")
    monkeypatch.delenv("RELATION_PROVENANCE_TIERS", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args, **kwargs: pytest.fail("Network forbidden"))
    return root


def _edit_yaml(path, edit):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_generated_domain_has_all_current_contracts(sandbox):
    active = get_config()
    before = asdict(active)
    assert new_domain.main(["medical"]) == 0
    domain = sandbox / "domains/medical"
    assert validate_domain(domain) == {
        "prompts": 14, "prompt_blocks": 28, "relations": 61, "questions": 3,
    }
    assert set(path.name for path in (domain / "prompts").iterdir()) == set(PROMPT_FIELDS)
    assert not (domain / "domain_filters.yaml").exists()
    assert not (domain / "competency_questions.txt").exists()
    assert not list((sandbox / "domains").glob(".new-domain-*"))
    assert get_config() is active
    assert asdict(get_config()) == before
    cfg = load_config(domain / "ontology_config.yaml")
    assert cfg.waterfall == ("bfo",)
    assert set(cfg.ontologies) == {"bfo", "ro"}
    assert len(cfg.property_specializations()) == 2
    blocks = load_prompt_blocks(domain)
    assert "personas_scope_auditor" in blocks
    assert "personas_cq_auditor" not in blocks
    assert "carbonate" not in "\n".join(blocks.values()).lower()


@pytest.mark.parametrize("path", [
    "domains/_template/README.md",
    "domains/_template/ontology_config.yaml",
    "domains/_template/prompt_blocks.yaml",
    "domains/presalt/prompts/cq_scoring.txt",
    "domains/presalt/resources/bfo-core.owl",
    "domains/presalt/resources/ro-core.owl",
    "domains/_shared/bfo_ro_relations.yaml",
])
def test_missing_source_files_never_publish_a_partial_domain(sandbox, capsys, path):
    (sandbox / path).unlink()
    with pytest.raises(SystemExit):
        new_domain.main(["medical"])
    captured = capsys.readouterr()
    assert "Created:" not in captured.out
    assert "error:" in captured.err
    assert not (sandbox / "domains/medical").exists()
    assert not list((sandbox / "domains").glob(".new-domain-*"))


@pytest.mark.parametrize("mutation", ["missing_block", "old_json", "question_ids", "unescaped_json", "malformed_yaml"])
def test_invalid_starter_text_never_publishes(sandbox, mutation, capsys):
    path = sandbox / "domains/_template/prompt_blocks.yaml"
    if mutation == "malformed_yaml":
        path.write_text("personas: [\n", encoding="utf-8")
    else:
        def edit(data):
            if mutation == "missing_block":
                del data["personas"]["scope_auditor"]
            elif mutation == "old_json":
                data["examples"]["cq_scoring_output"] = '{{"per_cq": [], "max_score": 2}}'
            elif mutation == "question_ids":
                data["examples"]["cq_identifiers"] = "CQ1, CQ9"
            else:
                data["examples"]["cq_scoring_output"] = '[{"term": "x", "relevant_cqs": []}]'
        _edit_yaml(path, edit)
    with pytest.raises(SystemExit):
        new_domain.main(["medical"])
    assert "Created:" not in capsys.readouterr().out
    assert not (sandbox / "domains/medical").exists()
    assert not list((sandbox / "domains").glob(".new-domain-*"))


def test_unknown_runtime_field_is_rejected(sandbox, capsys):
    path = sandbox / "domains/presalt/prompts/term_extraction.txt"
    path.write_text(path.read_text() + "\n{unknown_batch}\n")
    with pytest.raises(SystemExit):
        new_domain.main(["medical"])
    assert "runtime fields" in capsys.readouterr().err
    assert not (sandbox / "domains/medical").exists()


def test_existing_domain_is_not_overwritten(sandbox):
    target = sandbox / "domains/medical"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("Existing user content.")
    with pytest.raises(SystemExit):
        new_domain.main(["medical"])
    assert sentinel.read_text() == "Existing user content."


def test_runtime_uses_question_block_and_configured_property_menu(sandbox):
    new_domain.main(["medical"])
    domain = sandbox / "domains/medical"
    path = domain / "prompt_blocks.yaml"
    _edit_yaml(path, lambda data: data["examples"].__setitem__(
        "cq_questions",
        "CQ1 - Which specimens are studied?\nCQ2 - Which processes occur?\nCQ3 - Which qualities matter?",
    ))
    _, body = load_prompt("cq_scoring.txt", domain_dir=domain)
    assert "Which specimens are studied?" in body
    assert validate_domain(domain)["questions"] == 3
    cfg = load_config(domain / "ontology_config.yaml")
    original = yaml.safe_load((sandbox / "domains/_shared/bfo_ro_relations.yaml").read_text())
    local = dict(original["relations"]["has_part"], critic_menu=False)
    _edit_yaml(domain / "ontology_config.yaml", lambda data: data["relations"].__setitem__("has_part", local))
    table = load_prompt_blocks(domain)["examples_relation_property_table"]
    assert "| has_part |" not in table
    assert len(load_config(domain / "ontology_config.yaml").all_relations()) == len(cfg.all_relations())


@pytest.mark.parametrize("invalid", [
    {"relation_defaults": "recursive.yaml"},
    {"relations": []},
    {"metatype_groups": None},
    {"property_specializations": None},
])
def test_invalid_or_recursive_shared_fragments_fail(sandbox, invalid):
    shared = sandbox / "domains/_shared/bfo_ro_relations.yaml"
    shared.write_text(yaml.safe_dump(invalid))
    with pytest.raises(RuntimeError):
        load_config(sandbox / "domains/_template/ontology_config.yaml")


def test_shared_defaults_override_whole_entries_and_specializations(sandbox):
    path = sandbox / "domains/_template/ontology_config.yaml"
    shared = yaml.safe_load((sandbox / "domains/_shared/bfo_ro_relations.yaml").read_text())
    original = load_config(path)
    local = dict(shared["relations"]["has_part"], notes="Local override.")
    def edit(data):
        data["relations"] = {"has_part": local}
        data["property_specializations"] = []
    _edit_yaml(path, edit)
    changed = load_config(path)
    assert len(changed.all_relations()) == 61
    assert changed.all_relations()["has_part"].notes == "Local override."
    assert changed.property_specializations() == ()
    assert original.property_specializations()
    _edit_yaml(path, lambda data: data["relations"].__setitem__("has_part", {"notes": "incomplete"}))
    with pytest.raises((RuntimeError, KeyError)):
        load_config(path)


def test_shared_entries_preserve_the_existing_presalt_definitions():
    starter = load_config(ROOT / "domains/_template/ontology_config.yaml")
    presalt = load_config(ROOT / "domains/presalt/ontology_config.yaml")
    assert len(starter.all_relations()) == 61
    assert len(presalt.all_relations()) == 71
    for name, definition in starter.all_relations().items():
        assert definition == presalt.all_relations()[name], name
    assert starter.property_specializations() == presalt.property_specializations()


def test_new_domain_can_validate_relations_export_and_verify_without_geology(sandbox):
    new_domain.main(["medical"])
    domain = sandbox / "domains/medical"
    env = os.environ.copy()
    env.update({
        "ONTOLOGY_CONFIG_PATH": str(domain / "ontology_config.yaml"),
        "PYTHONPATH": str(ROOT),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    script = """
import json, socket, sys
from pathlib import Path
def deny(*args, **kwargs):
    raise AssertionError("Network forbidden")
socket.socket.connect = deny
socket.socket.connect_ex = deny
import pandas as pd
from rdflib import Graph, OWL
from src.utils.csv_io import write_csv
from src.utils.relation_validator import specialize_property, validate_relation_full
from src.modules.emit.owl_exporter import run_owl_export
from src.modules.emit.verifier import run_ontology_verification
root = Path(sys.argv[1])
prop = specialize_property("has_part", "material entity", "material entity")
assert prop == "has_continuant_part", prop
result = validate_relation_full("Container", "material entity", prop, "Lid", "material entity", 1.0, "has a lid as a part")
assert result.is_valid, result
taxonomy = root / "taxonomy.csv"
write_csv(pd.DataFrame([{"Term": "Specimen", "Parent_Term": "material entity", "Category": "material entity", "Relationship_Type": "rdfs:subClassOf", "Is_Intermediate": False, "FALLBACK": False, "NLD": "A specimen is a material entity selected for examination."}]), taxonomy)
ttl = root / "ontology.ttl"
run_owl_export(str(taxonomy), output_path=str(ttl))
report = run_ontology_verification(str(ttl), skip_oops=True, skip_reasoner=True)
assert report["overall_status"] == "PASS", report
graph = Graph().parse(ttl)
assert not any("inf.ufrgs.br" in str(node) or "presalt-onto" in str(node) for triple in graph for node in triple)
assert len(graph) > 0
print("New-domain relation validation, export and offline verification passed.")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(domain)],
        cwd=sandbox, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_frozen_artifact_hashes_remain_intact():
    inputs = ROOT / "evaluation_study/inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    assert sha256((ROOT / manifest["approved_ontology_path"]).read_bytes()).hexdigest() == manifest["approved_ontology_sha256"]
    for name, expected in manifest["files"].items():
        assert sha256((inputs / name).read_bytes()).hexdigest() == expected
