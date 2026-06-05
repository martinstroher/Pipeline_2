"""
Parity test — verifies ontology_config.yaml produces dicts/sets that are
1:1 identical to the legacy hardcoded literals BEFORE migration.

This is the safety net for Phase 2 migration. It MUST pass after Phase 1
and after every Phase 2 step. If it fails, the YAML drifted from the
literals — fix the YAML, do not adjust the test.

Run:
    python test/test_ontology_config_parity.py
"""

import sys
from pathlib import Path

# Make repo root importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.ontology_config import get_config


_FAILED: list[str] = []


def _assert_eq(label: str, expected, actual):
    if expected != actual:
        diff_extra = (set(actual) - set(expected)) if hasattr(expected, "__iter__") else None
        diff_missing = (set(expected) - set(actual)) if hasattr(expected, "__iter__") else None
        msg = f"[FAIL] {label}\n   expected: {expected!r}\n   actual:   {actual!r}"
        if diff_extra:
            msg += f"\n   extra in config: {sorted(diff_extra)}"
        if diff_missing:
            msg += f"\n   missing from config: {sorted(diff_missing)}"
        _FAILED.append(msg)
        print(msg)
    else:
        print(f"[OK]   {label}")


def main() -> int:
    cfg = get_config()

    # ── UPPER_IRIS parity ──
    from src.modules.taxonomy_builder import UPPER_IRIS as LEGACY_UPPER_IRIS
    _assert_eq("UPPER_IRIS dict equality", LEGACY_UPPER_IRIS, cfg.upper_iris())

    # ── _CATEGORY_TO_METATYPES parity ──
    from src.utils.relation_validator import _CATEGORY_TO_METATYPES as LEGACY_CAT_META
    _assert_eq("_CATEGORY_TO_METATYPES dict equality", LEGACY_CAT_META, cfg.category_to_metatypes())

    # ── GEORESERVOIR_CATEGORIES / GEOCORE_CATEGORIES parity ──
    from src.evaluation.expert_eval_generator import (
        GEORESERVOIR_CATEGORIES as LEGACY_GERES,
        GEOCORE_CATEGORIES as LEGACY_GCORE,
    )
    _assert_eq("GEORESERVOIR_CATEGORIES set equality", LEGACY_GERES, cfg.categories_for("georeservoir"))
    _assert_eq("GEOCORE_CATEGORIES set equality", LEGACY_GCORE, cfg.categories_for("geocore"))

    # ── OWL file list parity ──
    # Class-source OWL files (excludes property-only ontologies like ro-core.owl).
    from src.modules.owl_exporter import ONTO_NS as LEGACY_ONTO_NS
    expected_owl = {"bfo-core.owl", "geocore-full.owl", "geores-full.owl"}
    _assert_eq("OWL class-source file list", expected_owl, {p.name for p in cfg.owl_class_paths()})
    # Total OWL file list (includes ro-core.owl as property supplier)
    expected_all = {"bfo-core.owl", "geocore-full.owl", "geores-full.owl", "ro-core.owl"}
    _assert_eq("OWL total file list", expected_all, {p.name for p in cfg.owl_file_paths()})

    # ── Project namespace parity ──
    _assert_eq("project namespace", str(LEGACY_ONTO_NS), cfg.project_namespace())

    # ── BFO disjoint pairs parity ──
    from src.modules.owl_exporter import _BFO_DISJOINT as LEGACY_DISJOINT
    _assert_eq(
        "BFO disjoint pairs (as sorted tuples)",
        sorted(tuple(sorted(p)) for p in LEGACY_DISJOINT),
        sorted(tuple(sorted(p)) for p in cfg.bfo_disjoint_pairs()),
    )

    # ── Verifier prefixes parity ──
    from src.modules.ontology_verifier import _BFO_PREFIX, _GEO_PREFIX, _ONTO_PREFIX
    vp = cfg.verifier_prefixes
    _assert_eq("verifier BFO prefix", _BFO_PREFIX, vp.get("bfo"))
    _assert_eq("verifier Geo prefix", _GEO_PREFIX, vp.get("geo"))
    _assert_eq("verifier project prefix", _ONTO_PREFIX, vp.get("presalt"))

    # ── LLM definitions block sanity check ──
    # The .txt files were the legacy source; they are now deleted (YAML is canonical).
    # Verify the YAML produces non-empty definition blocks for every ontology key
    # and that every category has its own definition line.
    for ontology_key in ("georeservoir", "geocore", "bfo"):
        block = cfg.llm_definitions_block(ontology_key)
        config_labels = {ln.split(":", 1)[0].strip() for ln in block.split("\n") if ln.strip()}
        expected_labels = set(cfg.categories_for(ontology_key))
        # BFO has 3 additional ancestor entries (entity/continuant/occurrent) that
        # appear in definitions but not in categories_for(). Allow superset.
        missing = expected_labels - config_labels
        if missing:
            msg = f"[FAIL] {ontology_key} definitions missing labels: {sorted(missing)}"
            _FAILED.append(msg)
            print(msg)
        else:
            print(f"[OK]   {ontology_key} definitions complete ({len(config_labels)} lines, {len(expected_labels)} categories)")

    # ── Phase 3: relations / property-constraints sanity ──
    from src.utils.relation_validator import PROPERTY_CONSTRAINTS
    all_rels = cfg.all_relations()
    active_rels = cfg.property_constraints()
    if len(PROPERTY_CONSTRAINTS) != len(active_rels):
        msg = f"[FAIL] PROPERTY_CONSTRAINTS count {len(PROPERTY_CONSTRAINTS)} ≠ active relations {len(active_rels)}"
        _FAILED.append(msg)
        print(msg)
    else:
        print(f"[OK]   PROPERTY_CONSTRAINTS == active relations ({len(active_rels)} entries, {len(all_rels)} total)")

    # Provenance distribution (informational)
    from collections import Counter
    prov_dist = Counter(pc.provenance for pc in all_rels.values())
    print(f"[INFO] Provenance distribution: {dict(prov_dist)}")

    # Metatype-group expansion sanity
    cont_set = cfg.metatype_groups.get("CONTINUANT", frozenset())
    occ_set = cfg.metatype_groups.get("OCCURRENT", frozenset())
    if not cont_set or not occ_set:
        msg = "[FAIL] metatype_groups missing CONTINUANT or OCCURRENT"
        _FAILED.append(msg)
        print(msg)
    else:
        print(f"[OK]   metatype_groups expanded (CONTINUANT={len(cont_set)}, OCCURRENT={len(occ_set)})")

    # ── Phase 4: Step 6d mode + guardrail sanity ──
    import os
    from src.utils.ontology_config import reload_config

    # Default mode = refinement, min_evidence ≥ 2, strict_subclass = True
    _assert_eq("step6d default mode", "refinement", cfg.step6d_mode())
    if cfg.step6d_min_evidence() < 2:
        msg = f"[FAIL] step6d_min_evidence < 2 (got {cfg.step6d_min_evidence()})"
        _FAILED.append(msg)
        print(msg)
    else:
        print(f"[OK]   step6d_min_evidence >= 2 (got {cfg.step6d_min_evidence()})")
    _assert_eq("step6d strict_subclass", True, cfg.step6d_strict_subclass())

    # _is_strict_subclass logic
    from src.modules.relation_reclassifier import _is_strict_subclass
    parent = frozenset({"IndependentContinuant", "Continuant"})
    child = frozenset({"MaterialEntity", "IndependentContinuant", "Continuant"})
    sibling = frozenset({"ImmaterialEntity", "IndependentContinuant", "Continuant"})
    _assert_eq("strict_subclass child⊃parent", True, _is_strict_subclass(child, parent))
    _assert_eq("strict_subclass parent⊃child", False, _is_strict_subclass(parent, child))
    _assert_eq("strict_subclass sibling⊃sibling", False, _is_strict_subclass(sibling, child))
    _assert_eq("strict_subclass equal", False, _is_strict_subclass(parent, parent))

    # Env-override toggle: STEP6D_MODE=contradiction must round-trip
    _prev = os.environ.get("STEP6D_MODE")
    os.environ["STEP6D_MODE"] = "contradiction"
    try:
        cfg2 = reload_config()
        _assert_eq("step6d mode env override", "contradiction", cfg2.step6d_mode())
    finally:
        if _prev is None:
            os.environ.pop("STEP6D_MODE", None)
        else:
            os.environ["STEP6D_MODE"] = _prev
        reload_config()  # restore default

    # ── Phase 6.3: waterfall + categories_block accessors ──
    _assert_eq(
        "waterfall_ontologies() order",
        ["georeservoir", "geocore", "bfo"],
        cfg.waterfall_ontologies(),
    )
    block = cfg.categorization_block()
    expected_headers = [
        "### GeoReservoir Categories:",
        "### GeoCore Categories:",
        "### BFO Categories:",
    ]
    if not all(h in block for h in expected_headers):
        msg = f"[FAIL] categorization_block missing expected headers: {expected_headers}"
        _FAILED.append(msg)
        print(msg)
    else:
        # Verify ordering: each header must appear before the next.
        positions = [block.find(h) for h in expected_headers]
        if positions != sorted(positions):
            msg = f"[FAIL] categorization_block header order wrong: positions={positions}"
            _FAILED.append(msg)
            print(msg)
        else:
            print(f"[OK]   categorization_block: 3 ordered headers, {len(block.splitlines())} lines")

    # ── Final summary ──
    print()
    if _FAILED:
        print(f"=== PARITY FAILED — {len(_FAILED)} mismatch(es) ===")
        return 1
    print("=== PARITY PASSED — config produces identical literals ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
