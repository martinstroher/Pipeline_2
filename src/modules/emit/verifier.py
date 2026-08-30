"""
Ontology verifier for pipeline step 7b.

Post-export verification of the OWL ontology artifact.

Layers:
  1. Syntax Verification (RDFLib): parse Turtle, detect malformed triples.
  2. Structural Analysis (RDFLib): orphan classes, missing labels/comments,
     self-referential subClassOf, upper-ontology anchoring.
  3. OOPS! Pitfall Detection (optional): calls OOPS! REST API if OOPS_URL
     env var is set. Works with the remote API, a local Docker instance,
     or any compatible endpoint.
  4. HermiT Reasoner (optional): runs HermiT via owlready2 to check
     ontology consistency and detect unsatisfiable classes.

No LLM repair. Verification results are logged and saved to JSON
for thesis reporting.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import requests
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from src.utils import log
from src.utils.ontology_config import get_config
from src.modules.emit.cq_answerability import evaluate_competency_questions

# Known upper-ontology IRI prefixes are sourced from ontology_config.yaml.
_CFG = get_config()
_VERIFIER_PREFIXES = _CFG.verifier_prefixes
_BFO_PREFIX = _VERIFIER_PREFIXES["bfo"]
_GEO_PREFIX = _VERIFIER_PREFIXES["geo"]
_ONTO_PREFIX = _VERIFIER_PREFIXES["presalt"]

# OOPS! configuration (set OOPS_URL to enable, e.g. https://oops.linkeddata.es/rest
# or http://localhost:8080/OOPS/rest for Docker: docker run -p 8080:8080 mpovedavillalon/oops:v1)
_OOPS_URL = os.environ.get("OOPS_URL", "")
_OOPS_TIMEOUT = int(os.environ.get("OOPS_API_TIMEOUT_SECONDS", "120"))


def _display_path(path: str | os.PathLike) -> str:
    return os.path.relpath(os.fspath(path), os.getcwd())


# ── Layer 1: Syntax Verification ────────────────────────────────────────

def _verify_syntax(ttl_path: str) -> dict:
    """Parse the Turtle file with RDFLib and report syntax errors."""
    result = {"layer": "syntax", "status": "PASS", "errors": [], "triple_count": 0}
    try:
        g = Graph()
        g.parse(ttl_path, format="turtle")
        result["triple_count"] = len(g)
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(str(e))
    return result


# ── Layer 2: Structural Analysis ────────────────────────────────────────

def _verify_structure(ttl_path: str) -> dict:
    """Analyse the ontology graph for structural issues."""
    g = Graph()
    g.parse(ttl_path, format="turtle")

    issues = []

    # Collect classes and individuals
    classes = set(g.subjects(RDF.type, OWL.Class))
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))
    all_entities = classes | individuals

    # --- Self-referential rdfs:subClassOf ---
    self_refs = [
        str(s) for s, _, o in g.triples((None, RDFS.subClassOf, None))
        if s == o
    ]
    for iri in self_refs:
        issues.append({
            "type": "SELF_REFERENTIAL_SUBCLASSOF",
            "severity": "CRITICAL",
            "entity": iri,
            "detail": "Class is declared as subClassOf itself.",
        })

    # --- Orphan classes (no parent, not an upper-level IRI, not the Ontology) ---
    presalt_classes = [c for c in classes if str(c).startswith(_ONTO_PREFIX)]
    for cls in presalt_classes:
        parents = list(g.objects(cls, RDFS.subClassOf))
        if not parents:
            issues.append({
                "type": "ORPHAN_CLASS",
                "severity": "IMPORTANT",
                "entity": str(cls),
                "detail": "Class has no rdfs:subClassOf parent (root float).",
            })

    # --- Missing rdfs:label ---
    entities_without_label = [
        str(e) for e in all_entities
        if str(e).startswith(_ONTO_PREFIX)
        and not list(g.objects(e, RDFS.label))
    ]
    for iri in entities_without_label:
        issues.append({
            "type": "MISSING_LABEL",
            "severity": "MINOR",
            "entity": iri,
            "detail": "Entity has no rdfs:label.",
        })

    # --- Missing rdfs:comment (NLD) ---
    entities_without_comment = [
        str(e) for e in all_entities
        if str(e).startswith(_ONTO_PREFIX)
        and not list(g.objects(e, RDFS.comment))
    ]
    for iri in entities_without_comment:
        issues.append({
            "type": "MISSING_COMMENT",
            "severity": "MINOR",
            "entity": iri,
            "detail": "Entity has no rdfs:comment (NLD not propagated).",
        })

    # --- Upper-ontology anchoring ---
    upper_iris_used = set()
    for _, _, o in g.triples((None, RDFS.subClassOf, None)):
        o_str = str(o)
        if o_str.startswith(_BFO_PREFIX) or o_str.startswith(_GEO_PREFIX):
            upper_iris_used.add(o_str)
    for _, _, o in g.triples((None, RDF.type, None)):
        o_str = str(o)
        if o_str.startswith(_BFO_PREFIX) or o_str.startswith(_GEO_PREFIX):
            upper_iris_used.add(o_str)

    # Summary counts
    severity_counts = Counter(i["severity"] for i in issues)

    return {
        "layer": "structure",
        "status": "PASS" if not any(i["severity"] == "CRITICAL" for i in issues) else "FAIL",
        "classes": len(classes),
        "individuals": len(individuals),
        "triples": len(g),
        "upper_iris_referenced": len(upper_iris_used),
        "upper_iris_list": sorted(upper_iris_used),
        "issues": issues,
        "issue_summary": dict(severity_counts),
    }


# ── Layer 3: OOPS! Pitfall Detection ───────────────────────────────────

def _verify_oops(ttl_path: str) -> dict:
    """Run OOPS! pitfall scanner via the configured OOPS_URL endpoint."""
    result = {
        "layer": "oops_pitfalls",
        "status": "SKIP",
        "pitfalls": [],
        "pitfall_summary": {},
        "errors": [],
        "oops_url": _OOPS_URL,
        "request_omitted_imports": [],
    }

    if not _OOPS_URL:
        result["errors"].append(
            "OOPS_URL not configured. Set OOPS_URL env var to enable "
            "(e.g. https://oops.linkeddata.es/rest or http://localhost:8080/OOPS/rest)"
        )
        return result

    try:
        graph = Graph()
        graph.parse(ttl_path, format="turtle")
        omitted_imports = [str(value) for value in graph.objects(None, OWL.imports)]
        graph.remove((None, OWL.imports, None))
        result["request_omitted_imports"] = omitted_imports
        rdfxml_content = graph.serialize(format="xml")
    except Exception as e:
        result["errors"].append(f"Failed to read TTL: {e}")
        return result

    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<OOPSRequest>\n"
        "  <OntologyURI></OntologyURI>\n"
        f"  <OntologyContent><![CDATA[{rdfxml_content}]]></OntologyContent>\n"
        "  <Pitfalls></Pitfalls>\n"
        "  <OutputFormat>RDF/XML</OutputFormat>\n"
        "</OOPSRequest>"
    )

    try:
        resp = requests.post(
            _OOPS_URL,
            data=xml_body.encode("utf-8"),
            headers={"Content-Type": "application/xml"},
            timeout=_OOPS_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        result["errors"].append(f"OOPS! timeout after {_OOPS_TIMEOUT}s")
        return result
    except requests.exceptions.ConnectionError:
        result["errors"].append("OOPS! unreachable (connection error)")
        return result
    except requests.exceptions.HTTPError as e:
        result["errors"].append(f"OOPS! HTTP error: {e}")
        return result

    # Parse the XML response for pitfall elements
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)

        rdf_namespace = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        oops_namespace = "http://oops.linkeddata.es/def#"
        rdf_resource = f"{{{rdf_namespace}}}resource"
        root_is_rdf = root.tag == f"{{{rdf_namespace}}}RDF"
        has_response_marker = any(
            element.tag == f"{{{rdf_namespace}}}type"
            and element.attrib.get(rdf_resource) == f"{oops_namespace}response"
            for element in root.iter()
        )
        pitfall_node_count = sum(
            element.tag == f"{{{rdf_namespace}}}type"
            and element.attrib.get(rdf_resource) == f"{oops_namespace}pitfall"
            for element in root.iter()
        )
        by_code: dict[str, dict] = {}
        for description_node in root.iter():
            if description_node.tag != f"{{{rdf_namespace}}}Description":
                continue
            properties: dict[str, str] = {}
            affected_values: list[str] = []
            for child in description_node:
                if not child.tag.startswith(f"{{{oops_namespace}}}"):
                    continue
                tag = child.tag.removeprefix(f"{{{oops_namespace}}}")
                text = (child.text or "").strip()
                if tag in {"hasAffectedElement", "hasEquivalentClass"}:
                    if text:
                        affected_values.append(text)
                else:
                    properties[tag] = text
            code = properties.get("hasCode", "")
            if not code:
                continue
            slot = by_code.setdefault(
                code,
                {
                    "code": code,
                    "name": "",
                    "importance": "",
                    "description": "",
                    "affected_values": [],
                },
            )
            slot["name"] = slot["name"] or properties.get("hasName", "")
            slot["importance"] = (
                slot["importance"] or properties.get("hasImportanceLevel", "")
            )
            slot["description"] = (
                slot["description"] or properties.get("hasDescription", "")
            )
            slot["affected_values"].extend(affected_values)

        pitfalls = []
        valid_importance = {"Critical", "Important", "Minor"}
        invalid_pitfall_records = []
        for slot in by_code.values():
            if (
                not slot["code"]
                or not slot["name"]
                or slot["importance"] not in valid_importance
            ):
                invalid_pitfall_records.append(slot["code"] or "<missing code>")
            pitfalls.append(
                {
                    "code": slot["code"],
                    "name": slot["name"],
                    "importance": slot["importance"],
                    "description": slot["description"][:200],
                    "affected_elements": len(slot["affected_values"]),
                    "affected_values": slot["affected_values"][:40],
                }
            )
        pitfalls.sort(key=lambda pitfall: pitfall["code"])

        result["pitfalls"] = pitfalls
        if (
            not root_is_rdf
            or not has_response_marker
            or pitfall_node_count != len(pitfalls)
            or invalid_pitfall_records
        ):
            result["status"] = "ERROR"
            result["errors"].append(
                "OOPS returned an incomplete or unrecognized response schema"
            )
        elif not pitfalls and "unexpected_error" in resp.text:
            result["status"] = "ERROR"
            result["errors"].append("OOPS returned unexpected_error")
        elif not pitfalls and "wrong_execution" in resp.text:
            result["status"] = "ERROR"
            result["errors"].append("OOPS returned wrong_execution")
        else:
            result["status"] = "PASS" if not any(
                p["importance"].lower() == "critical" for p in pitfalls
            ) else "FAIL"

        severity_counts = Counter(p["importance"] for p in pitfalls)
        result["pitfall_summary"] = dict(severity_counts)

    except ET.ParseError as e:
        result["errors"].append(f"Failed to parse OOPS! response XML: {e}")

    return result

# ── Layer 4: HermiT Reasoner Consistency Check ─────────────────────────────────

def _verify_hermit(
    ttl_path: str,
    closure_paths: list[Path] | None = None,
    mode: str = "configured class closure",
) -> dict:
    """Run HermiT over the artifact and an explicit configured closure."""
    result = {
        "layer": "reasoner",
        "status": "SKIP",
        "mode": mode,
        "consistent": None,
        "coherent": None,
        "unsatisfiable_classes": [],
        "closure_paths": [_display_path(path) for path in closure_paths or []],
        "errors": [],
    }

    try:
        import owlready2
    except ImportError:
        result["errors"].append("owlready2 not installed, skipping reasoner check")
        return result

    # Set Java path (JAVA_EXE env var or system default)
    java_exe = os.environ.get("JAVA_EXE", "")
    if java_exe:
        owlready2.JAVA_EXE = java_exe

    from rdflib import Graph as RDFGraph

    try:
        rdf_g = RDFGraph()
        rdf_g.parse(ttl_path, format="turtle")
        for closure_path in closure_paths or []:
            rdf_g.parse(str(closure_path))
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Failed to parse reasoner closure: {e}")
        return result

    tmp_file = tempfile.NamedTemporaryFile(suffix=".nt", delete=False, mode="wb")
    try:
        rdf_g.serialize(tmp_file, format="ntriples", encoding="utf-8")
        tmp_file.close()

        # Use a fresh world to avoid cross-contamination between runs
        world = owlready2.World()
        onto = world.get_ontology(_CFG.project_ontology_iri()).load(
            fileobj=open(tmp_file.name, "rb"), format="ntriples"
        )

        with onto:
            owlready2.sync_reasoner_hermit(world, infer_property_values=False)

        # Check for unsatisfiable classes (reclassified under Nothing)
        unsatisfiable = [
            c for c in world.inconsistent_classes()
            if str(getattr(c, "iri", "")) != "http://www.w3.org/2002/07/owl#Nothing"
        ]

        if unsatisfiable:
            result["status"] = "FAIL"
            result["consistent"] = True
            result["coherent"] = False
            result["unsatisfiable_classes"] = sorted(
                str(getattr(c, "iri", c)) for c in unsatisfiable
            )
        else:
            result["status"] = "PASS"
            result["consistent"] = True
            result["coherent"] = True

    except Exception as e:
        err_str = str(e)
        if "InconsistentOntology" in type(e).__name__ or "Inconsistent" in err_str:
            result["status"] = "FAIL"
            result["consistent"] = False
            result["coherent"] = False
            result["errors"].append("Ontology is globally inconsistent (HermiT)")
        else:
            result["status"] = "FAIL"
            result["errors"].append(f"Reasoner error: {err_str}")
    finally:
        os.unlink(tmp_file.name)

    return result


def _verify_hermit_bounded(
    ttl_path: str,
    closure_paths: list[Path],
    timeout_seconds: int,
) -> dict:
    """Run a non-gating closure in a process group with a hard timeout."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = handle.name
    command = [
        sys.executable,
        "-m",
        "src.modules.emit.verifier",
        "--reasoner-worker",
        ttl_path,
        output_path,
        *[str(path) for path in closure_paths],
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        process.wait(timeout=timeout_seconds)
        with open(output_path, encoding="utf-8") as handle:
            result = json.load(handle)
        result["gating"] = False
        return result
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return {
            "layer": "reasoner_ro",
            "status": "TIMEOUT",
            "mode": "configured class and property closure",
            "consistent": None,
            "coherent": None,
            "unsatisfiable_classes": [],
            "closure_paths": [_display_path(path) for path in closure_paths],
            "errors": [f"HermiT timed out after {timeout_seconds} seconds"],
            "gating": False,
        }
    finally:
        Path(output_path).unlink(missing_ok=True)

# ── Main Entry Point ────────────────────────────────────────────────────

def run_ontology_verification(
    ttl_path: str,
    output_path: str | None = None,
    skip_oops: bool = False,
    skip_reasoner: bool = False,
    cq_spec_path: str | None = None,
    ro_timeout_seconds: int | None = None,
) -> dict:
    """Run verification layers on the OWL ontology.

    Args:
        ttl_path: Path to .ttl file (output of owl_exporter).
        output_path: Path to save the verification report JSON.
                     Default: same dir as ttl_path, named emit_verification.json.
        skip_oops: If True, skip the OOPS! API call (for offline or test runs).
        skip_reasoner: If True, skip the HermiT reasoner check.

    Returns:
        dict with full verification report.
    """
    if output_path is None:
        base_dir = os.path.dirname(ttl_path) or "."
        output_path = os.path.join(base_dir, "emit_verification.json")

    log.banner("7b", "Ontology Verification")
    log.info(f"Verifying: {ttl_path}")

    report = {"input": ttl_path, "layers": {}}

    # ── Layer 1: Syntax ──
    log.info("Layer 1: Syntax verification (RDFLib)...")
    syntax = _verify_syntax(ttl_path)
    report["layers"]["syntax"] = syntax
    if syntax["status"] == "PASS":
        log.success(f"  Syntax: PASS, {syntax['triple_count']} triples parsed")
    else:
        log.error(f"  Syntax: FAIL, {syntax['errors']}")
        # If syntax fails, skip structure analysis (can't parse the graph)
        report["overall_status"] = "FAIL"
        _save_report(report, output_path)
        return report

    # ── Layer 2: Structure ──
    log.info("Layer 2: Structural analysis...")
    structure = _verify_structure(ttl_path)
    report["layers"]["structure"] = structure

    n_critical = structure["issue_summary"].get("CRITICAL", 0)
    n_important = structure["issue_summary"].get("IMPORTANT", 0)
    n_minor = structure["issue_summary"].get("MINOR", 0)

    log.info(f"  Classes: {structure['classes']}, Individuals: {structure['individuals']}, Triples: {structure['triples']}")
    log.info(f"  Upper-ontology IRIs referenced: {structure['upper_iris_referenced']}")

    if n_critical:
        log.error(f"  Structural issues: {n_critical} CRITICAL, {n_important} IMPORTANT, {n_minor} MINOR")
    elif n_important:
        log.warn(f"  Structural issues: {n_important} IMPORTANT, {n_minor} MINOR")
    elif n_minor:
        log.detail(f"  Structural issues: {n_minor} MINOR")
    else:
        log.success("  Structural issues: none")

    # ── Layer 3: OOPS! ──
    if skip_oops:
        log.info("Layer 3: OOPS! pitfall scan skipped (--skip-oops)")
        report["layers"]["oops_pitfalls"] = {"layer": "oops_pitfalls", "status": "SKIP"}
    else:
        log.info("Layer 3: OOPS! pitfall scan...")
        oops = _verify_oops(ttl_path)
        report["layers"]["oops_pitfalls"] = oops

        if oops["errors"]:
            log.warn(f"  OOPS!: {oops['errors'][0]}")
        elif oops["pitfalls"]:
            n_pit_critical = oops["pitfall_summary"].get("Critical", 0)
            n_pit_important = oops["pitfall_summary"].get("Important", 0)
            n_pit_minor = oops["pitfall_summary"].get("Minor", 0)
            log.info(f"  OOPS! pitfalls: {n_pit_critical} Critical, {n_pit_important} Important, {n_pit_minor} Minor")
            for p in oops["pitfalls"]:
                level = p["importance"]
                if level.lower() == "critical":
                    log.error(f"    [{level}] {p['code']}: {p['name']}")
                elif level.lower() == "important":
                    log.warn(f"    [{level}] {p['code']}: {p['name']}")
                else:
                    log.detail(f"    [{level}] {p['code']}: {p['name']}")
        else:
            log.success("  OOPS!: no pitfalls detected")

    # ── Layer 4: HermiT Reasoner ──
    if skip_reasoner:
        log.info("Layer 4: HermiT reasoner skipped (--skip-reasoner)")
        report["layers"]["reasoner"] = {"layer": "reasoner", "status": "SKIP"}
    else:
        log.info("Layer 4: HermiT reasoner closure check...")
        class_closure = _CFG.owl_class_paths()
        hermit = _verify_hermit(ttl_path, class_closure)
        hermit["gating"] = True
        report["layers"]["reasoner"] = hermit

        if hermit["errors"]:
            log.error(f"  Reasoner: {hermit['errors'][0]}")
        elif hermit["coherent"] is True:
            log.success("  Reasoner: PASS, ontology is consistent and coherent")
        elif hermit["coherent"] is False:
            n_unsat = len(hermit["unsatisfiable_classes"])
            log.error(f"  Reasoner: FAIL, {n_unsat} unsatisfiable classes")
            for cls_iri in hermit["unsatisfiable_classes"][:10]:
                log.error(f"    {cls_iri}")
            if n_unsat > 10:
                log.error(f"    ... and {n_unsat - 10} more")

        property_closure = _CFG.owl_property_paths()
        if property_closure:
            timeout = ro_timeout_seconds or int(
                os.environ.get("RO_REASONER_TIMEOUT_SECONDS", "120")
            )
            log.info("Layer 4b: bounded property-ontology closure check...")
            report["layers"]["reasoner_ro"] = _verify_hermit_bounded(
                ttl_path,
                class_closure + property_closure,
                timeout,
            )

    resolved_cq_spec = cq_spec_path
    if resolved_cq_spec is None and _CFG.cq_answerability_path():
        resolved_cq_spec = str(_CFG.cq_answerability_path())
    if resolved_cq_spec:
        log.info("Layer 5: competency-question answerability...")
        report["layers"]["competency_questions"] = evaluate_competency_questions(
            ttl_path,
            resolved_cq_spec,
            _CFG.owl_class_paths(),
        )

    # ── Overall status ──
    layer_statuses = [
        value.get("status", "SKIP")
        for value in report["layers"].values()
        if value.get("gating", True)
    ]
    if "FAIL" in layer_statuses:
        report["overall_status"] = "FAIL"
    elif all(s == "SKIP" for s in layer_statuses):
        report["overall_status"] = "SKIP"
    else:
        report["overall_status"] = "PASS"

    _save_report(report, output_path)

    if report["overall_status"] == "PASS":
        log.success(f"Verification: PASS, report saved to {output_path}")
    elif report["overall_status"] == "FAIL":
        log.error(f"Verification: FAIL, see report at {output_path}")
    else:
        log.info(f"Verification: PARTIAL, see report at {output_path}")

    return report


def _save_report(report: dict, output_path: str) -> None:
    """Save the verification report as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify OWL ontology (Step 7b)")
    parser.add_argument("ttl_path", nargs="?", help="Path to .ttl file")
    parser.add_argument("--output", default=None, help="Path for verification report JSON")
    parser.add_argument("--skip-oops", action="store_true", help="Skip OOPS! API call")
    parser.add_argument("--skip-reasoner", action="store_true", help="Skip HermiT reasoner check")
    parser.add_argument("--cq-spec", default=None, help="CQ answerability YAML")
    parser.add_argument("--ro-timeout", type=int, default=None)
    parser.add_argument("--reasoner-worker", nargs="+", default=None)
    args = parser.parse_args()
    if args.reasoner_worker:
        worker_ttl, worker_output, *worker_closure = args.reasoner_worker
        worker_result = _verify_hermit(
            worker_ttl,
            [Path(path) for path in worker_closure],
            mode="configured class and property closure",
        )
        worker_result["layer"] = "reasoner_ro"
        with open(worker_output, "w", encoding="utf-8") as handle:
            json.dump(worker_result, handle, indent=2, ensure_ascii=False)
    else:
        if not args.ttl_path:
            parser.error("ttl_path is required")
        run_ontology_verification(
            args.ttl_path,
            args.output,
            args.skip_oops,
            args.skip_reasoner,
            args.cq_spec,
            args.ro_timeout,
        )
