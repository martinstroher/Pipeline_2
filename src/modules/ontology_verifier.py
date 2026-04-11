"""
Ontology Verifier — Step 7b of the PreSaltOntoLearn pipeline.

Post-export verification of the OWL ontology artifact.

Layers:
  1. Syntax Verification (RDFLib): parse Turtle, detect malformed triples.
  2. Structural Analysis (RDFLib): orphan classes, missing labels/comments,
     self-referential subClassOf, upper-ontology anchoring.
  3. OOPS! Pitfall Detection (optional): calls OOPS! REST API if OOPS_URL
     env var is set. Works with the remote API, a local Docker instance,
     or any compatible endpoint.

No LLM repair — verification-only. Results are logged + saved to JSON
for thesis reporting.
"""

import json
import os
from collections import Counter

import requests
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from src.utils import log

# Known upper-ontology IRI prefixes
_BFO_PREFIX = "http://purl.obolibrary.org/obo/"
_GEO_PREFIX = "https://www.inf.ufrgs.br/bdi/ontologies/"
_ONTO_PREFIX = "https://w3id.org/presalt-onto#"

# OOPS! configuration (set OOPS_URL to enable, e.g. https://oops.linkeddata.es/rest
# or http://localhost:8080/OOPS/rest for Docker: docker run -p 8080:8080 mpovedavillalon/oops:v1)
_OOPS_URL = os.environ.get("OOPS_URL", "")
_OOPS_TIMEOUT = int(os.environ.get("OOPS_API_TIMEOUT_SECONDS", "120"))


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
    }

    if not _OOPS_URL:
        result["errors"].append(
            "OOPS_URL not configured. Set OOPS_URL env var to enable "
            "(e.g. https://oops.linkeddata.es/rest or http://localhost:8080/OOPS/rest)"
        )
        return result

    try:
        with open(ttl_path, "r", encoding="utf-8") as f:
            ttl_content = f.read()
    except Exception as e:
        result["errors"].append(f"Failed to read TTL: {e}")
        return result

    # Build OOPS! XML request
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<OOPSRequest>\n"
        f"  <OntologyContent><![CDATA[{ttl_content}]]></OntologyContent>\n"
        "  <Pitfalls></Pitfalls>\n"
        "  <OutputFormat>XML</OutputFormat>\n"
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

        # OOPS! uses RDF/XML with oops namespace
        ns = {"oops": "http://www.oeg-upm.net/oops#"}

        pitfalls = []
        for pit in root.iter():
            # Look for elements that contain pitfall data
            if "Pitfall" in pit.tag:
                code = ""
                name = ""
                description = ""
                importance = ""

                for child in pit:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    text = (child.text or "").strip()
                    if tag == "hasCode":
                        code = text
                    elif tag == "hasName":
                        name = text
                    elif tag == "hasDescription":
                        description = text
                    elif tag == "hasImportanceLevel":
                        importance = text

                if code:
                    pitfalls.append({
                        "code": code,
                        "name": name,
                        "importance": importance,
                        "description": description[:200],
                    })

        result["pitfalls"] = pitfalls
        result["status"] = "PASS" if not any(
            p["importance"].lower() == "critical" for p in pitfalls
        ) else "FAIL"

        severity_counts = Counter(p["importance"] for p in pitfalls)
        result["pitfall_summary"] = dict(severity_counts)

    except ET.ParseError as e:
        result["errors"].append(f"Failed to parse OOPS! response XML: {e}")

    return result



# ── Main Entry Point ────────────────────────────────────────────────────

def run_ontology_verification(
    ttl_path: str,
    output_path: str | None = None,
    skip_oops: bool = False,
) -> dict:
    """Run verification layers on the OWL ontology.

    Args:
        ttl_path: Path to .ttl file (output of owl_exporter).
        output_path: Path to save the verification report JSON.
                     Default: same dir as ttl_path, named 7b_verification_report.json.
        skip_oops: If True, skip the OOPS! API call (for offline or test runs).

    Returns:
        dict with full verification report.
    """
    if output_path is None:
        base_dir = os.path.dirname(ttl_path) or "."
        output_path = os.path.join(base_dir, "7b_verification_report.json")

    log.banner("7b", "Ontology Verification")
    log.info(f"Verifying: {ttl_path}")

    report = {"input": ttl_path, "layers": {}}

    # ── Layer 1: Syntax ──
    log.info("Layer 1: Syntax verification (RDFLib)...")
    syntax = _verify_syntax(ttl_path)
    report["layers"]["syntax"] = syntax
    if syntax["status"] == "PASS":
        log.success(f"  Syntax: PASS — {syntax['triple_count']} triples parsed")
    else:
        log.error(f"  Syntax: FAIL — {syntax['errors']}")
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
        log.info("Layer 3: OOPS! pitfall scan — SKIPPED (--skip-oops)")
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

    # ── Overall status ──
    layer_statuses = [v.get("status", "SKIP") for v in report["layers"].values()]
    if "FAIL" in layer_statuses:
        report["overall_status"] = "FAIL"
    elif all(s == "SKIP" for s in layer_statuses):
        report["overall_status"] = "SKIP"
    else:
        report["overall_status"] = "PASS"

    _save_report(report, output_path)

    if report["overall_status"] == "PASS":
        log.success(f"Verification: PASS — report saved to {output_path}")
    elif report["overall_status"] == "FAIL":
        log.error(f"Verification: FAIL — see report at {output_path}")
    else:
        log.info(f"Verification: PARTIAL — see report at {output_path}")

    return report


def _save_report(report: dict, output_path: str) -> None:
    """Save the verification report as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify OWL ontology (Step 7b)")
    parser.add_argument("ttl_path", help="Path to .ttl file")
    parser.add_argument("--output", default=None, help="Path for verification report JSON")
    parser.add_argument("--skip-oops", action="store_true", help="Skip OOPS! API call")
    args = parser.parse_args()
    run_ontology_verification(args.ttl_path, args.output, args.skip_oops)
