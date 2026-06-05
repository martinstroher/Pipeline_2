"""
Ontology Configuration Loader — single source of truth for upper-ontology
scope, namespaces, and per-class metadata.

Reads `ontology_config.yaml` (or path from env var `ONTOLOGY_CONFIG_PATH`)
and exposes typed accessors to all pipeline consumers. Apply env-var
overrides at load time so a single .env tweak changes pipeline behaviour.

Key accessors (consumer → call):
  - taxonomy_builder.UPPER_IRIS              → get_config().upper_iris()
  - relation_validator._CATEGORY_TO_METATYPES → get_config().category_to_metatypes()
  - expert_eval_generator.{GEORESERVOIR,GEOCORE}_CATEGORIES → get_config().categories_for("georeservoir"/"geocore")
  - term_categorizer / ablation_study definitions → get_config().llm_definitions_block("georeservoir"/...)
  - owl_exporter namespaces / disjoint pairs / OWL file list → get_config().{project_namespace,bfo_disjoint_pairs,owl_file_paths}()
  - ontology_verifier prefixes → get_config().verifier_prefixes()

Env vars:
  ONTOLOGY_CONFIG_PATH       — config file path (default: ontology_config.yaml)
  RELATION_PROVENANCE_TIERS  — comma-separated tier names (overrides YAML)
  STEP6D_MODE                — "refinement" | "contradiction" (overrides YAML)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassDef:
    """A single upper-ontology class entry."""
    label: str
    iri_fragment: str
    namespace: str                       # parent ontology namespace
    ontology_key: str                    # e.g. "bfo", "geocore", "georeservoir"
    metatypes: frozenset[str]            # empty ⇒ ancestor-only, not a Category
    llm_definition: str | None           # None ⇒ auto-extract from OWL at runtime

    @property
    def iri(self) -> str:
        """Full IRI (namespace + iri_fragment)."""
        return f"{self.namespace}{self.iri_fragment}"


@dataclass(frozen=True)
class OntologyDef:
    """A single upper ontology in scope."""
    key: str
    owl_path: Path
    namespace: str
    prefix: str
    eval_tier: str
    import_iri: str | None
    classes: tuple[ClassDef, ...]
    disjoint_pairs: tuple[tuple[str, str], ...]   # (iri_fragment, iri_fragment)


@dataclass(frozen=True)
class Step6dConfig:
    mode: str                                  # "refinement" | "contradiction"
    refinement_min_evidence: int
    refinement_only_to_strict_subclass: bool


@dataclass(frozen=True)
class PropertyConstraint:
    """Domain/range constraint for a single object property."""
    name: str
    iri: str
    domain: frozenset[str]
    range: frozenset[str]
    inverse: str | None
    provenance: str            # owl_axiom | bfo_shape_axiom | ro_release | spec_curation
    notes: str = ""


@dataclass(frozen=True)
class ProjectMeta:
    name: str
    description: str
    namespace: str
    prefix: str
    version: str
    long_description: str


@dataclass(frozen=True)
class OntologyConfig:
    """Top-level configuration object, loaded once per process."""
    project: ProjectMeta
    ontologies: dict[str, OntologyDef]
    step6d: Step6dConfig
    provenance_tiers_active: frozenset[str]
    verifier_prefixes: dict[str, str]
    metatype_groups: dict[str, frozenset[str]]
    relations: dict[str, PropertyConstraint]
    _source_path: Path = field(repr=False)

    # ─── Convenience accessors ────────────────────────────────────────

    def project_name(self) -> str:
        return self.project.name

    def project_namespace(self) -> str:
        return self.project.namespace

    def project_prefix(self) -> str:
        return self.project.prefix

    def project_version(self) -> str:
        return self.project.version

    def project_description(self) -> str:
        return self.project.description

    def project_long_description(self) -> str:
        return self.project.long_description

    def upper_iris(self) -> dict[str, str]:
        """All classes across all ontologies, keyed by label → full IRI."""
        out: dict[str, str] = {}
        for onto in self.ontologies.values():
            for cls in onto.classes:
                if cls.label in out:
                    raise RuntimeError(
                        f"Duplicate class label across ontologies: '{cls.label}' "
                        f"in {cls.ontology_key} conflicts with prior definition."
                    )
                out[cls.label] = cls.iri
        return out

    def category_to_metatypes(self) -> dict[str, frozenset[str]]:
        """Categories used for relation validation. Excludes classes with no metatypes."""
        out: dict[str, frozenset[str]] = {}
        for onto in self.ontologies.values():
            for cls in onto.classes:
                if cls.metatypes:
                    out[cls.label] = cls.metatypes
        return out

    def categories_for(self, ontology_key: str) -> set[str]:
        """Set of valid category labels for a given ontology tier (used by expert eval)."""
        onto = self.ontologies.get(ontology_key)
        if onto is None:
            raise KeyError(f"Unknown ontology key: {ontology_key}")
        return {cls.label for cls in onto.classes if cls.metatypes}

    def llm_definitions_block(self, ontology_key: str) -> str:
        """Formatted 'Label: definition' block for LLM prompt injection.

        Includes every class that contributes prompt context — i.e. those with
        either a metatype set (a valid category) OR an explicit `llm_definition`
        (top-level ancestors like 'entity'/'continuant'/'occurrent' that frame
        the hierarchy for the LLM but are not used as categories themselves).

        For classes with no `llm_definition` set, falls back to skos:definition
        in the OWL file, then rdfs:comment, then the label itself.
        """
        onto = self.ontologies.get(ontology_key)
        if onto is None:
            raise KeyError(f"Unknown ontology key: {ontology_key}")
        lines: list[str] = []
        for cls in onto.classes:
            if not cls.metatypes and not cls.llm_definition:
                continue  # ancestor with no definition → skip
            definition = cls.llm_definition or _extract_owl_definition(onto.owl_path, cls.iri) or cls.label
            lines.append(f"{cls.label}: {definition}")
        return "\n".join(lines)

    def owl_file_paths(self) -> list[Path]:
        """Ordered list of all OWL files in scope (classes + property-only)."""
        return [onto.owl_path for onto in self.ontologies.values() if onto.owl_path.exists()]

    def owl_class_paths(self) -> list[Path]:
        """OWL files that contribute upper-level CLASSES (excludes property-only ontologies like RO)."""
        return [
            onto.owl_path
            for onto in self.ontologies.values()
            if onto.classes and onto.owl_path.exists()
        ]

    def bfo_disjoint_pairs(self) -> list[tuple[str, str]]:
        """Full-IRI tuples for disjointness axioms, sourced from the BFO ontology entry."""
        bfo = self.ontologies.get("bfo")
        if bfo is None:
            return []
        return [
            (f"{bfo.namespace}{a}", f"{bfo.namespace}{b}")
            for a, b in bfo.disjoint_pairs
        ]

    def namespace_for(self, ontology_key: str) -> str:
        return self.ontologies[ontology_key].namespace

    def prefix_for(self, ontology_key: str) -> str:
        return self.ontologies[ontology_key].prefix

    def import_iri_for(self, ontology_key: str) -> str | None:
        return self.ontologies[ontology_key].import_iri

    def step6d_mode(self) -> str:
        return self.step6d.mode

    def step6d_min_evidence(self) -> int:
        return self.step6d.refinement_min_evidence

    def step6d_strict_subclass(self) -> bool:
        return self.step6d.refinement_only_to_strict_subclass

    def active_provenance_tiers(self) -> frozenset[str]:
        return self.provenance_tiers_active

    # ─── Relations / property constraints ─────────────────────────────

    def property_constraints(self) -> dict[str, PropertyConstraint]:
        """All property constraints whose provenance tier is active.

        Filtered against `active_provenance_tiers()`. Returns a new dict each
        call (cheap — ≤ ~100 entries).
        """
        active = self.provenance_tiers_active
        return {
            name: pc for name, pc in self.relations.items()
            if pc.provenance in active
        }

    def all_relations(self) -> dict[str, PropertyConstraint]:
        """All relations regardless of active tiers — used by the audit script."""
        return dict(self.relations)


# ─────────────────────────────────────────────────────────────────────────
# OWL definition extractor (used when YAML llm_definition is absent)
# ─────────────────────────────────────────────────────────────────────────

_SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"
_RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"
_OWL_GRAPH_CACHE: dict[Path, object] = {}


def _extract_owl_definition(owl_path: Path, iri: str) -> str | None:
    """Read skos:definition (fallback rdfs:comment) for the given class IRI from an OWL file."""
    if not owl_path.exists():
        return None
    try:
        from rdflib import Graph, URIRef
    except ImportError:
        return None

    if owl_path not in _OWL_GRAPH_CACHE:
        g = Graph()
        try:
            g.parse(str(owl_path))
        except Exception:
            return None
        _OWL_GRAPH_CACHE[owl_path] = g
    g = _OWL_GRAPH_CACHE[owl_path]

    subj = URIRef(iri)
    for pred in (_SKOS_DEFINITION, _RDFS_COMMENT):
        for obj in g.objects(subj, URIRef(pred)):
            text = str(obj).strip()
            if text:
                return text
    return None


# ─────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path("domains") / "presalt" / "ontology_config.yaml"


def _resolve_config_path() -> Path:
    """Resolve config file path: env var → CWD → repo root."""
    env_path = os.environ.get("ONTOLOGY_CONFIG_PATH")
    if env_path:
        return Path(env_path).resolve()
    cwd = Path.cwd() / _DEFAULT_CONFIG_PATH
    if cwd.exists():
        return cwd.resolve()
    # Walk up from this file to find repo root containing the config
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _DEFAULT_CONFIG_PATH
        if candidate.exists():
            return candidate
    return cwd.resolve()  # fall through; loader will raise on missing file


def _parse_classes(
    raw_classes: list[dict],
    namespace: str,
    ontology_key: str,
) -> tuple[ClassDef, ...]:
    seen_iris: dict[str, str] = {}
    out: list[ClassDef] = []
    for entry in raw_classes:
        label = entry["label"]
        iri_frag = entry["iri_fragment"]
        full_iri = f"{namespace}{iri_frag}"
        if full_iri in seen_iris:
            raise RuntimeError(
                f"Duplicate IRI {full_iri} in ontology '{ontology_key}': "
                f"labels '{seen_iris[full_iri]}' and '{label}'."
            )
        seen_iris[full_iri] = label
        metatypes = frozenset(entry.get("metatypes") or [])
        llm_def = entry.get("llm_definition")
        if llm_def is not None:
            llm_def = str(llm_def).strip()
        out.append(ClassDef(
            label=label,
            iri_fragment=iri_frag,
            namespace=namespace,
            ontology_key=ontology_key,
            metatypes=metatypes,
            llm_definition=llm_def,
        ))
    return tuple(out)


def _expand_metatype_groups(raw_groups: dict[str, list[str]]) -> dict[str, frozenset[str]]:
    """Resolve metatype-group references to flat frozensets of metatype strings.

    Group definitions may reference other groups by name. Expansion is
    recursive with cycle detection. Unknown names are treated as literal
    BFO metatype strings.
    """
    resolved: dict[str, frozenset[str]] = {}
    in_progress: set[str] = set()

    def _expand(name: str) -> frozenset[str]:
        if name in resolved:
            return resolved[name]
        if name in in_progress:
            raise RuntimeError(f"Cycle in metatype_groups while expanding '{name}'")
        if name not in raw_groups:
            return frozenset({name})
        in_progress.add(name)
        out: set[str] = set()
        for item in raw_groups[name]:
            out |= _expand(item)
        in_progress.discard(name)
        resolved[name] = frozenset(out)
        return resolved[name]

    for group_name in raw_groups:
        _expand(group_name)
    return resolved


def _parse_relations(
    raw_relations: dict,
    metatype_groups: dict[str, frozenset[str]],
) -> dict[str, PropertyConstraint]:
    """Build PropertyConstraint entries from YAML, resolving group references."""
    valid_provenance = {"owl_axiom", "bfo_shape_axiom", "ro_release", "spec_curation"}
    out: dict[str, PropertyConstraint] = {}
    for name, entry in (raw_relations or {}).items():
        domain_set: set[str] = set()
        for token in entry.get("domain", []):
            domain_set |= metatype_groups.get(token, frozenset({token}))
        range_set: set[str] = set()
        for token in entry.get("range", []):
            range_set |= metatype_groups.get(token, frozenset({token}))
        provenance = entry.get("provenance", "spec_curation")
        if provenance not in valid_provenance:
            raise RuntimeError(
                f"Invalid provenance '{provenance}' for relation '{name}' "
                f"(allowed: {sorted(valid_provenance)})."
            )
        out[name] = PropertyConstraint(
            name=name,
            iri=entry["iri"],
            domain=frozenset(domain_set),
            range=frozenset(range_set),
            inverse=entry.get("inverse"),
            provenance=provenance,
            notes=str(entry.get("notes", "")).strip(),
        )
    return out


def _build_config(raw: dict, source_path: Path) -> OntologyConfig:
    repo_root = source_path.parent

    # project
    p = raw["project"]
    project = ProjectMeta(
        name=p["name"],
        description=p.get("description", ""),
        namespace=p["namespace"],
        prefix=p["prefix"],
        version=p.get("version", "0.1.0"),
        long_description=p.get("long_description", "").strip(),
    )

    # ontologies
    ontologies: dict[str, OntologyDef] = {}
    for key, entry in raw["ontologies"].items():
        owl_rel = entry["owl"]
        owl_path = (repo_root / owl_rel).resolve()
        classes = _parse_classes(
            entry.get("classes", []),
            namespace=entry["namespace"],
            ontology_key=key,
        )
        disjoint_raw = entry.get("disjoint_pairs", []) or []
        disjoint = tuple((a, b) for a, b in disjoint_raw)
        ontologies[key] = OntologyDef(
            key=key,
            owl_path=owl_path,
            namespace=entry["namespace"],
            prefix=entry["prefix"],
            eval_tier=entry.get("eval_tier", key),
            import_iri=entry.get("import_iri"),
            classes=classes,
            disjoint_pairs=disjoint,
        )

    # step6d (env override has priority)
    s = raw.get("step6d", {})
    mode = os.environ.get("STEP6D_MODE", s.get("mode", "refinement")).strip().lower()
    if mode not in {"refinement", "contradiction"}:
        raise RuntimeError(f"Invalid STEP6D_MODE / step6d.mode: '{mode}' (expected 'refinement' or 'contradiction')")
    step6d = Step6dConfig(
        mode=mode,
        refinement_min_evidence=int(s.get("refinement_min_evidence", 2)),
        refinement_only_to_strict_subclass=bool(s.get("refinement_only_to_strict_subclass", True)),
    )

    # provenance tiers (env override has priority)
    default_tiers = raw.get("provenance_tiers_active", []) or []
    env_tiers = os.environ.get("RELATION_PROVENANCE_TIERS")
    if env_tiers:
        tier_list = [t.strip() for t in env_tiers.split(",") if t.strip()]
    else:
        tier_list = list(default_tiers)
    valid_tiers = {"owl_axiom", "bfo_shape_axiom", "ro_release", "spec_curation"}
    for t in tier_list:
        if t not in valid_tiers:
            raise RuntimeError(f"Invalid provenance tier: '{t}' (allowed: {sorted(valid_tiers)})")
    provenance_tiers = frozenset(tier_list)

    verifier_prefixes = dict(raw.get("verifier_prefixes", {}))

    # metatype groups + relations
    metatype_groups = _expand_metatype_groups(raw.get("metatype_groups", {}) or {})
    relations = _parse_relations(raw.get("relations", {}) or {}, metatype_groups)

    cfg = OntologyConfig(
        project=project,
        ontologies=ontologies,
        step6d=step6d,
        provenance_tiers_active=provenance_tiers,
        verifier_prefixes=verifier_prefixes,
        metatype_groups=metatype_groups,
        relations=relations,
        _source_path=source_path,
    )

    _validate_uniqueness(cfg)
    return cfg


def _validate_uniqueness(cfg: OntologyConfig) -> None:
    """Enforce: every (label, IRI) is unique across all ontologies."""
    seen_labels: dict[str, str] = {}
    seen_iris: dict[str, str] = {}
    for onto in cfg.ontologies.values():
        for cls in onto.classes:
            if cls.label in seen_labels:
                raise RuntimeError(
                    f"Duplicate class label '{cls.label}' in ontology '{onto.key}' "
                    f"(prior: '{seen_labels[cls.label]}'). Labels must be globally unique."
                )
            seen_labels[cls.label] = onto.key
            if cls.iri in seen_iris:
                raise RuntimeError(
                    f"Duplicate class IRI {cls.iri} (labels: '{cls.label}' vs "
                    f"earlier definition). Use distinct IRIs per published class."
                )
            seen_iris[cls.iri] = cls.label


@lru_cache(maxsize=1)
def get_config() -> OntologyConfig:
    """Load and return the singleton config. Subsequent calls return the cached instance."""
    path = _resolve_config_path()
    if not path.exists():
        raise RuntimeError(
            f"ontology_config.yaml not found at {path}. "
            f"Set ONTOLOGY_CONFIG_PATH env var or place the file at repo root."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build_config(raw, path)


def reload_config() -> OntologyConfig:
    """Clear the cache and reload from disk. Used by tests; not for normal pipeline use."""
    get_config.cache_clear()
    _OWL_GRAPH_CACHE.clear()
    return get_config()
