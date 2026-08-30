from pathlib import Path
import shutil

import pandas as pd
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDFS

from src.modules.emit.correction_manifest import load_correction_manifest
from src.modules.emit.cq_answerability import evaluate_competency_questions
from src.modules.emit.owl_exporter import _mint_presalt_iri, run_owl_export
from src.modules.emit.verifier import _verify_hermit
from src.utils.ontology_config import get_config


ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / "evaluation_study" / "inputs" / "approved_run"


def test_unicode_iri_minting_transliterates_diacritics():
    assert str(_mint_presalt_iri("Búzios Field")).endswith("#BuziosField")
    assert str(_mint_presalt_iri("Piçarras Formation")).endswith("#PicarrasFormation")
    assert str(_mint_presalt_iri("São Paulo Plateau")).endswith("#SaoPauloPlateau")


def test_manifest_contains_complete_demotion_mapping():
    manifest = load_correction_manifest(
        ROOT / "domains" / "presalt" / "corrections" / "geopresalt_0_2.yaml"
    )
    demotions = [
        decision
        for decision in manifest["decisions"]
        if decision["action"] == "demotion_trace"
    ]
    assert len(demotions) == 34
    assert sum("F046" in decision["defect_id"] for decision in demotions) == 6
    assert sum(
        decision["resolution"] == "reexpressed as an emitted restriction"
        for decision in demotions
    ) == 1


def test_approved_replay_emits_coherent_versioned_release(tmp_path, monkeypatch):
    homebrew_java = Path("/opt/homebrew/opt/openjdk@21/bin/java")
    java = str(homebrew_java) if homebrew_java.exists() else shutil.which("java")
    if not java:
        pytest.skip("Java is required for the HermiT release proof")
    monkeypatch.setenv("JAVA_EXE", java)
    cfg = get_config()
    ontology = tmp_path / "release.ttl"
    provenance = tmp_path / "provenance.csv"
    corrections = tmp_path / "corrections.json"
    run_owl_export(
        str(APPROVED / "validate_taxonomy.csv"),
        output_path=str(ontology),
        relations_csv=str(APPROVED / "validate_relations.csv"),
        instances_csv=str(APPROVED / "validate_instances.csv"),
        defined_csv=str(APPROVED / "validate_defined_classes.csv"),
        correction_manifest=str(cfg.correction_manifest_path()),
        correction_log_output=str(corrections),
        provenance_output=str(provenance),
    )

    graph = Graph().parse(str(ontology), format="turtle")
    ontology_iri = URIRef(cfg.project_ontology_iri())
    assert (ontology_iri, OWL.versionInfo, Literal("0.2")) in graph
    assert (
        ontology_iri,
        OWL.versionIRI,
        URIRef("https://w3id.org/presalt-onto/0.2"),
    ) in graph
    assert len(list(graph.objects(ontology_iri, OWL.imports))) == 3
    assert not any(
        "[critic" in str(comment)
        for comment in graph.objects(None, RDFS.comment)
    )
    provenance_rows = pd.read_csv(provenance, encoding="utf-8-sig")
    assert len(provenance_rows) == len(graph)
    source_text = "\n".join(provenance_rows["sources"])
    assert "manifest:COH-011" in source_text
    assert "input:" in source_text
    assert "config:" in source_text

    reasoner = _verify_hermit(str(ontology), cfg.owl_class_paths())
    assert reasoner["consistent"] is True
    assert reasoner["coherent"] is True
    assert reasoner["unsatisfiable_classes"] == []

    cq = evaluate_competency_questions(
        ontology,
        cfg.cq_answerability_path(),
        cfg.owl_class_paths(),
    )
    assert cq["summary"] == {
        "FULL": 2,
        "PARTIAL": 3,
        "NOT ANSWERABLE": 5,
    }
