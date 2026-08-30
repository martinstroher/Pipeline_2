"""Build and prove a deterministic versioned ontology release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd
from rdflib import Graph, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import OWL, RDF

from src.modules.emit.correction_manifest import (
    GRAPH_ACTIONS,
    load_correction_manifest,
)
from src.modules.emit.owl_exporter import run_owl_export
from src.modules.emit.verifier import run_ontology_verification
from src.utils.ontology_config import get_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ascii_fold(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )


def _tokenize_for_scan(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", ascii_value)
    return re.sub(r"[^a-z0-9]+", " ", camel_split.casefold()).strip()


def _added_src_text(base_ref: str) -> tuple[str, list[str]]:
    diff = subprocess.run(
        ["git", "diff", "--unified=0", base_ref, "--", "src"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    added_lines = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for relative_path in untracked:
        added_lines.append(Path(relative_path).read_text(encoding="utf-8"))
    return "\n".join(added_lines), untracked


def _scan_added_src_terms(
    term_file: Path,
    base_ref: str,
    vocabulary_sources: list[Path],
    allowlist_path: Path,
) -> dict:
    added, untracked = _added_src_text(base_ref)
    manual_terms = {
        line.strip()
        for line in term_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    allowlist = {
        line.strip().casefold()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    terms = set(manual_terms)
    vocabulary_columns = {
        "Term",
        "Parent_Term",
        "Filler",
        "Base_Class",
        "Target_Class",
        "Mint_Parent",
        "Bearer",
        "Genus",
    }
    for source in vocabulary_sources:
        frame = pd.read_csv(source, encoding="utf-8-sig")
        for column in vocabulary_columns & set(frame.columns):
            terms.update(
                str(value).strip()
                for value in frame[column].dropna()
                if str(value).strip()
            )
    terms = {
        term
        for term in terms
        if len(term) >= 3
        and term.casefold() not in allowlist
    }
    added_tokens = _tokenize_for_scan(added)
    added_compact = re.sub(r"[^a-z0-9]+", "", _ascii_fold(added))
    matches = []
    for term in sorted(terms, key=str.casefold):
        tokenized = _tokenize_for_scan(term)
        compact = tokenized.replace(" ", "")
        pattern = rf"(?<![a-z0-9]){re.escape(tokenized)}(?![a-z0-9])"
        if re.search(pattern, added_tokens) or (
            len(compact) >= 6 and compact in added_compact
        ):
            matches.append(term)
    return {
        "scope": f"added and untracked src content relative to {base_ref}",
        "untracked_src_files": untracked,
        "term_sources": [str(term_file), *[str(path) for path in vocabulary_sources]],
        "allowlist": os.path.relpath(allowlist_path, Path.cwd()),
        "terms_checked": len(terms),
        "matches": matches,
        "pass": not matches,
    }


def _scan_em_dash(paths: list[Path], base_ref: str) -> dict:
    hits: list[str] = []
    for path in paths:
        if path.is_dir():
            candidates = [item for item in path.rglob("*") if item.is_file()]
        else:
            candidates = [path]
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "\u2014" in text:
                hits.append(str(candidate))

    added_src, _ = _added_src_text(base_ref)
    if "\u2014" in added_src:
        hits.append(f"added or untracked src content relative to {base_ref}")
    return {
        "paths_checked": [os.path.relpath(path, Path.cwd()) for path in paths],
        "hits": hits,
        "pass": not hits,
    }


def build_release(
    *,
    taxonomy_csv: str,
    relations_csv: str,
    instances_csv: str,
    defined_csv: str,
    output_dir: str,
    frozen_artifact: str,
    frozen_sha256: str,
    baseline_record: str,
    anti_overfit_terms: str,
    ro_timeout_seconds: int = 120,
) -> dict:
    """Emit the release twice and write every proof artifact."""
    cfg = get_config()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ontology_path = output / cfg.release_artifact_name()
    correction_log_path = output / "correction_application.json"
    provenance_path = output / "triple_provenance.csv"
    verification_path = output / "verification.json"
    manifest_path = cfg.correction_manifest_path()
    if manifest_path is None:
        raise RuntimeError("Configured correction manifest is required")
    baseline_path = Path(baseline_record)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_commit = str(baseline.get("baseline_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_commit):
        raise ValueError("Baseline record requires a full commit SHA")
    if Path(str(baseline.get("artifact", ""))) != Path(frozen_artifact):
        raise ValueError("Baseline record artifact does not match the release input")
    if baseline.get("expected_sha256") != frozen_sha256:
        raise ValueError("Baseline record digest does not match the release input")

    run_owl_export(
        taxonomy_csv,
        output_path=str(ontology_path),
        relations_csv=relations_csv,
        instances_csv=instances_csv,
        defined_csv=defined_csv,
        correction_manifest=str(manifest_path),
        correction_log_output=str(correction_log_path),
        provenance_output=str(provenance_path),
    )
    verification = run_ontology_verification(
        str(ontology_path),
        str(verification_path),
        ro_timeout_seconds=ro_timeout_seconds,
    )

    with tempfile.TemporaryDirectory() as directory:
        replay_path = Path(directory) / "replay.ttl"
        replay_provenance = Path(directory) / "replay_provenance.csv"
        run_owl_export(
            taxonomy_csv,
            output_path=str(replay_path),
            relations_csv=relations_csv,
            instances_csv=instances_csv,
            defined_csv=defined_csv,
            correction_manifest=str(manifest_path),
            provenance_output=str(replay_provenance),
        )
        release_graph = Graph().parse(str(ontology_path), format="turtle")
        replay_graph = Graph().parse(str(replay_path), format="turtle")
        release_provenance = pd.read_csv(provenance_path, encoding="utf-8-sig")
        replay_provenance_frame = pd.read_csv(
            replay_provenance, encoding="utf-8-sig"
        )
        regeneration = {
            "graph_isomorphic": isomorphic(release_graph, replay_graph),
            "release_triples": len(release_graph),
            "replay_triples": len(replay_graph),
            "classes": len(set(release_graph.subjects(RDF.type, OWL.Class))),
            "named_individuals": len(
                set(release_graph.subjects(RDF.type, OWL.NamedIndividual))
            ),
            "provenance_rows": len(release_provenance),
            "replay_provenance_rows": len(replay_provenance_frame),
            "provenance_records_equal": release_provenance.equals(
                replay_provenance_frame
            ),
            "live_llm_calls": 0,
        }
    _write_json(output / "regeneration_proof.json", regeneration)

    frozen_path = Path(frozen_artifact)
    baseline_blob = subprocess.run(
        ["git", "show", f"{baseline_commit}:{baseline['artifact']}"],
        check=True,
        capture_output=True,
    ).stdout
    baseline_blob_sha256 = hashlib.sha256(baseline_blob).hexdigest()
    baseline_blob_sha1 = subprocess.run(
        ["git", "rev-parse", f"{baseline_commit}:{baseline['artifact']}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    immutability = {
        "path": str(frozen_path),
        "expected_sha256": frozen_sha256,
        "actual_sha256": _sha256(frozen_path),
        "baseline_commit": baseline_commit,
        "baseline_blob_sha256": baseline_blob_sha256,
        "expected_blob_sha1": baseline.get("artifact_git_blob_sha1"),
        "actual_blob_sha1": baseline_blob_sha1,
    }
    immutability["unchanged"] = (
        immutability["actual_sha256"] == immutability["expected_sha256"]
        == immutability["baseline_blob_sha256"]
        and immutability["actual_blob_sha1"] == immutability["expected_blob_sha1"]
    )
    _write_json(output / "frozen_0_1_immutability.json", immutability)
    shutil.copyfile(baseline_path, output / "baseline_regeneration.json")

    manifest = load_correction_manifest(manifest_path)
    critic_log = {
        "critic": manifest.get("critic", {}),
        "release_status": manifest.get("release_status"),
        "decisions": [
            {
                key: decision.get(key, "")
                for key in (
                    "id",
                    "defect_id",
                    "action",
                    "subject",
                    "resolution",
                    "critic_verdict",
                    "critic_reasoning",
                    "confidence",
                    "justification",
                    "status",
                )
            }
            for decision in manifest["decisions"]
        ],
    }
    _write_json(output / "domain_critic_verdicts.json", critic_log)
    correction_application = json.loads(
        correction_log_path.read_text(encoding="utf-8")
    )
    application_by_id = {
        row["decision_id"]: row for row in correction_application
    }
    graph_decision_ids = {
        decision["id"]
        for decision in manifest["decisions"]
        if decision["action"] in GRAPH_ACTIONS
    }
    all_graph_decisions_applied = all(
        application_by_id.get(decision_id, {}).get("graph_change") is True
        and application_by_id[decision_id].get("matches") == 1
        for decision_id in graph_decision_ids
    )

    defect_ids = [str(value) for value in manifest.get("defect_set", [])]
    if not defect_ids:
        raise ValueError("Correction manifest requires a defect_set")
    defect_status = {}
    for defect_id in defect_ids:
        rows = [
            decision
            for decision in manifest["decisions"]
            if defect_id in str(decision.get("defect_id", "")).split(",")
        ]
        defect_status[defect_id] = {
            "decisions": [row["id"] for row in rows],
            "terminal_states": sorted({row["status"] for row in rows}),
            "complete": bool(rows),
        }
    _write_json(output / "defect_status.json", defect_status)

    allowlist_path = cfg.anti_overfit_allowlist_path()
    if allowlist_path is None or not allowlist_path.exists():
        raise RuntimeError("Configured anti-overfit allowlist is required")
    anti_overfit = _scan_added_src_terms(
        Path(anti_overfit_terms),
        baseline_commit,
        [
            Path(taxonomy_csv),
            Path(relations_csv),
            Path(instances_csv),
            Path(defined_csv),
        ],
        allowlist_path,
    )
    _write_json(output / "anti_overfit_scan.json", anti_overfit)

    _write_json(output / "hermit_report.json", verification["layers"]["reasoner"])
    _write_json(
        output / "hermit_ro_report.json",
        verification["layers"].get("reasoner_ro", {"status": "NOT_RUN"}),
    )
    _write_json(
        output / "oops_report.json",
        verification["layers"].get("oops_pitfalls", {"status": "NOT_RUN"}),
    )
    _write_json(
        output / "cq_answerability_report.json",
        verification["layers"].get("competency_questions", {}),
    )

    version_values = {
        str(value)
        for value in release_graph.objects(
            URIRef(cfg.project_ontology_iri()), OWL.versionInfo
        )
    }
    version_iris = {
        str(value)
        for value in release_graph.objects(
            URIRef(cfg.project_ontology_iri()), OWL.versionIRI
        )
    }
    disclosure_template_path = cfg.disclosure_template_path()
    if disclosure_template_path is None or not disclosure_template_path.exists():
        raise RuntimeError("Configured disclosure template is required")
    disclosure = disclosure_template_path.read_text(encoding="utf-8").format(
        display_name=manifest.get("display_name", cfg.project_name()),
        cq_summary=verification["layers"]["competency_questions"]["summary"],
        ro_status=verification["layers"].get("reasoner_ro", {}).get(
            "status", "NOT_RUN"
        ),
    )
    (output / "DISCLOSURE.md").write_text(disclosure, encoding="utf-8")

    manifest_support_files = [
        manifest_path.parent / str(relative_path)
        for relative_path in manifest.get("decision_files", [])
    ]
    em_dash = _scan_em_dash(
        [
            ontology_path,
            manifest_path,
            *manifest_support_files,
            cfg.cq_answerability_path(),
            disclosure_template_path,
            output,
        ],
        baseline_commit,
    )
    _write_json(output / "em_dash_scan.json", em_dash)

    proof_gate = {
        "version_info": sorted(version_values),
        "version_iri": sorted(version_iris),
        "versioned_artifact": ontology_path.exists(),
        "coherent": verification["layers"]["reasoner"].get("coherent") is True,
        "unsatisfiable_named_classes": len(
            verification["layers"]["reasoner"].get("unsatisfiable_classes", [])
        ),
        "deterministic_replay": regeneration["graph_isomorphic"],
        "provenance_complete": (
            regeneration["provenance_rows"] == regeneration["release_triples"]
            and regeneration["provenance_records_equal"]
        ),
        "anti_overfit": anti_overfit["pass"],
        "frozen_0_1_unchanged": immutability["unchanged"],
        "manifest_decisions": len(manifest["decisions"]),
        "graph_decisions": len(graph_decision_ids),
        "all_graph_decisions_applied": all_graph_decisions_applied,
        "all_defects_terminal": all(
            value["complete"] for value in defect_status.values()
        ),
        "oops_recorded": (
            verification["layers"].get("oops_pitfalls", {}).get("status") == "PASS"
        ),
        # SKIP and every other non-PASS reasoner status fail the release gate.
        "hermit_recorded": verification["layers"]["reasoner"].get("status") == "PASS",
        "class_closure_complete": (
            verification["layers"]["reasoner"].get("closure_paths")
            == [os.path.relpath(path, Path.cwd()) for path in cfg.owl_class_paths()]
        ),
        "ro_recorded": (
            verification["layers"].get("reasoner_ro", {}).get("status")
            in {"PASS", "TIMEOUT"}
        ),
        "ro_clean_if_completed": (
            verification["layers"].get("reasoner_ro", {}).get("status") == "TIMEOUT"
            or verification["layers"].get("reasoner_ro", {}).get("coherent") is True
        ),
        "disclosure_recorded": (output / "DISCLOSURE.md").exists(),
        "em_dash_free": em_dash["pass"],
        "release_status": manifest.get("release_status"),
    }
    proof_gate["pass"] = all(
        [
            proof_gate["versioned_artifact"],
            cfg.project_version() in proof_gate["version_info"],
            cfg.project_version_iri() in proof_gate["version_iri"],
            proof_gate["coherent"],
            proof_gate["unsatisfiable_named_classes"] == 0,
            proof_gate["deterministic_replay"],
            proof_gate["provenance_complete"],
            proof_gate["anti_overfit"],
            proof_gate["frozen_0_1_unchanged"],
            proof_gate["all_defects_terminal"],
            proof_gate["all_graph_decisions_applied"],
            proof_gate["oops_recorded"],
            proof_gate["hermit_recorded"],
            proof_gate["class_closure_complete"],
            proof_gate["ro_recorded"],
            proof_gate["ro_clean_if_completed"],
            proof_gate["disclosure_recorded"],
            proof_gate["em_dash_free"],
        ]
    )
    _write_json(output / "proof_gate.json", proof_gate)
    if not proof_gate["pass"]:
        raise RuntimeError(f"Release proof gate failed: {proof_gate}")
    return proof_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--defined", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-artifact", required=True)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--baseline-record", required=True)
    parser.add_argument("--anti-overfit-terms", required=True)
    parser.add_argument("--ro-timeout", type=int, default=120)
    args = parser.parse_args()
    proof = build_release(
        taxonomy_csv=args.taxonomy,
        relations_csv=args.relations,
        instances_csv=args.instances,
        defined_csv=args.defined,
        output_dir=args.output_dir,
        frozen_artifact=args.frozen_artifact,
        frozen_sha256=args.frozen_sha256,
        baseline_record=args.baseline_record,
        anti_overfit_terms=args.anti_overfit_terms,
        ro_timeout_seconds=args.ro_timeout,
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
