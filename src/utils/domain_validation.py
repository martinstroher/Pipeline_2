"""Offline structural checks for a domain using the current production prompts."""

import argparse
import json
import math
from pathlib import Path
import re
from string import Formatter

from src.utils import log
from src.utils.ontology_config import OntologyConfig, load_config
from src.utils.prompt_loader import load_prompt, load_prompt_blocks, prompt_files

# Call-time fields are the interface between prompt text and production callers.
PROMPT_FIELDS = {
    "term_extraction.txt": {"chunk_text"},
    "nld_generation.txt": {"term", "context"},
    "term_categorization.txt": {"categories_block", "json_batch"},
    "taxonomy_building.txt": {"category", "upper_vocab", "terms_json"},
    "relation_extraction.txt": {"known_terms", "json_batch", "batch_size"},
    "cq_scoring.txt": {"batch_size", "terms_json"},
    "cq_synonym_triage.txt": {"clusters_json"},
    "critic_taxonomy.txt": {
        "category", "parent_context_json", "relations_context_json",
        "target_classes_json", "terms_json", "weak_observations_json",
    },
    "critic_taxonomy_dedup.txt": {"cross_candidates_json", "survivors_json"},
    "critic_class_worthiness.txt": {
        "candidates_json", "category", "existing_classes_json", "properties_json",
        "relations_context_json", "sibling_context_json", "weak_observations_json",
    },
    "critic_facet_frames.txt": {
        "category", "existing_classes_json", "max_candidates", "survivors_json",
        "target_context_json",
    },
    "critic_frame_completion.txt": {"candidates_json"},
    "critic_relation_scope.txt": {
        "category", "relations_json", "taxonomy_context_json", "taxonomy_decisions_json",
    },
    "critic_relations.txt": {
        "category", "previously_minted_json", "relations_json", "relations_menu_json",
        "taxonomy_context_json", "taxonomy_decisions_json",
    },
}
_BLOCK = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)>>")
_QUESTION = re.compile(r"^(CQ(?:[1-9]|10))\s*(?:-|—|:)\s*(\S.*)$")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text_fields(item: dict, fields: tuple[str, ...], label: str) -> None:
    _check(
        all(isinstance(item.get(key), str) and item[key].strip() for key in fields),
        f"{label}: requires non-empty text fields {', '.join(fields)}.",
    )


def _outputs(text: str, label: str, *, system: bool = False, standalone: bool = False) -> list:
    rendered = text if system else text.format()
    if standalone:
        candidates = [rendered]
    else:
        starts = list(re.finditer(r"(?m)^\s*Output:\s*", rendered))
        _check(bool(starts), f"{label}: expected an Output: JSON example.")
        candidates = [rendered[match.end():] for match in starts]
    outputs = []
    for candidate in candidates:
        candidate = candidate.lstrip()
        if candidate.startswith("```"):
            candidate = candidate.partition("\n")[2].lstrip()
        # Existing domains also use escaped JSON in system-side examples.
        if system and re.match(r"^(?:\[\s*)?\{\{", candidate):
            candidate = candidate.format()
        try:
            output, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}: invalid JSON example: {exc}") from exc
        if standalone:
            _check(candidate[end:].strip() in ("", "```"), f"{label}: unexpected text after JSON.")
        outputs.append(output)
    return outputs


def _validate_questions(blocks: dict[str, str]) -> set[str]:
    lines = [line.strip() for line in blocks["examples_cq_questions"].splitlines() if line.strip()]
    matches = [_QUESTION.fullmatch(line) for line in lines]
    _check(bool(matches) and all(matches), "cq_questions: use CQ1 through CQ10 followed by a question.")
    ids = [match.group(1) for match in matches if match is not None]
    _check(len(ids) == len(set(ids)), "cq_questions: duplicate question identifiers.")
    declared = re.findall(r"\bCQ\d+\b", blocks["examples_cq_identifiers"])
    _check(
        len(declared) == len(set(declared)) and set(declared) == set(ids),
        "cq_identifiers must list exactly the identifiers in cq_questions.",
    )
    return set(ids)


def _validate_examples(
    blocks: dict[str, str], cfg: OntologyConfig, question_ids: set[str]
) -> None:
    categories = {label for key in cfg.waterfall for label in cfg.categories_for(key)}
    relations = cfg.property_constraints()
    outputs = _outputs(
        blocks["examples_term_extraction_output"], "term_extraction_output", standalone=True
    )
    _check(
        isinstance(outputs[0], list) and bool(outputs[0])
        and all(isinstance(term, str) and term.strip() for term in outputs[0]),
        "term_extraction_output: expected a non-empty JSON array of term strings.",
    )
    for item in _outputs(blocks["examples_nld_generation"], "nld_generation"):
        _check(isinstance(item, dict), "nld_generation: expected a JSON object.")
        _text_fields(item, ("Definition",), "nld_generation")
        _check(type(item.get("Context_Used")) is bool, "nld_generation: Context_Used must be boolean.")
    specs = [
        ("examples_term_categorization", False, False),
        ("examples_relation_extraction", False, False),
        ("examples_cq_scoring_output", False, True),
    ]
    worked_cqs = blocks["examples_cq_scoring"]
    if re.search(r"(?m)^\s*Output:", worked_cqs):
        specs.append(("examples_cq_scoring", True, False))
    else:
        mentioned = set(re.findall(r"\bCQ\d+\b", worked_cqs))
        _check(
            bool(mentioned) and mentioned <= question_ids,
            "examples_cq_scoring: prose examples must reference configured questions.",
        )
    for name, system, standalone in specs:
        for batch in _outputs(blocks[name], name, system=system, standalone=standalone):
            _check(isinstance(batch, list) and bool(batch), f"{name}: expected a non-empty JSON array.")
            for item in batch:
                _check(isinstance(item, dict), f"{name}: array entries must be objects.")
                _text_fields(item, ("term",), name)
                if name == "examples_term_categorization":
                    _text_fields(item, ("category", "reasoning"), name)
                    _check(
                        item["category"] in categories | {"NOT_CLASSIFIED"},
                        f"{name}: category {item['category']!r} is not configured.",
                    )
                elif name == "examples_relation_extraction":
                    _check(isinstance(item.get("relations"), list), f"{name}: relations must be an array.")
                    for relation in item["relations"]:
                        _check(isinstance(relation, dict), f"{name}: relation entries must be objects.")
                        _text_fields(relation, ("property", "filler", "evidence"), name)
                        _check(relation["property"] in relations, f"{name}: unknown/inactive property {relation['property']!r}.")
                        confidence = relation.get("confidence")
                        _check(
                            type(confidence) in (int, float) and math.isfinite(confidence)
                            and 0 <= confidence <= 1,
                            f"{name}: confidence must be a finite number between 0 and 1.",
                        )
                else:
                    _text_fields(item, ("reasoning",), name)
                    ids = item.get("relevant_cqs")
                    _check(
                        isinstance(ids, list) and all(isinstance(q, str) and q in question_ids for q in ids),
                        f"{name}: relevant_cqs must be an array of configured question identifiers.",
                    )


def validate_domain(domain_dir: str | Path) -> dict[str, int]:
    """Check files, placeholders, current example schemas, and relation configuration."""
    domain_dir = Path(domain_dir).resolve()
    for name in ("ontology_config.yaml", "prompt_blocks.yaml"):
        _check((domain_dir / name).is_file(), f"Required domain file missing: {domain_dir / name}")
    cfg = load_config(domain_dir / "ontology_config.yaml")
    for ontology in cfg.ontologies.values():
        _check(ontology.owl_path.is_file(), f"Required ontology resource missing: {ontology.owl_path}")
    _check(bool(cfg.property_constraints()), "The domain has no active relation constraints.")
    files = dict(prompt_files(domain_dir=domain_dir))
    _check(
        set(files) == set(PROMPT_FIELDS),
        f"Prompt files do not match production: missing={sorted(set(PROMPT_FIELDS) - set(files))}, "
        f"unexpected={sorted(set(files) - set(PROMPT_FIELDS))}.",
    )
    blocks = load_prompt_blocks(domain_dir)
    required = {
        name for path in files.values()
        for name in _BLOCK.findall(path.read_text(encoding="utf-8"))
    }
    _check(not required - blocks.keys(), f"Missing prompt blocks: {sorted(required - blocks.keys())}")
    _check(all(blocks[name].strip() for name in required), "Required prompt blocks must not be blank.")
    for name, expected in PROMPT_FIELDS.items():
        system, body = load_prompt(name, domain_dir=domain_dir)
        _check(bool(system) and bool(body), f"{name}: system instruction and template must not be empty.")
        fields = {field for _, field, _, _ in Formatter().parse(body) if field is not None}
        _check(fields == expected, f"{name}: runtime fields {sorted(fields)} differ from {sorted(expected)}.")
        values = {field: 1 if field in {"batch_size", "max_candidates"} else "[]" for field in fields}
        body.format(**values)
    question_ids = _validate_questions(blocks)
    _validate_examples(blocks, cfg, question_ids)
    return {"prompts": len(files), "prompt_blocks": len(required), "relations": len(cfg.property_constraints()), "questions": len(question_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_dir", type=Path)
    args = parser.parse_args()
    try:
        report = validate_domain(args.domain_dir)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    log.success(f"Domain structure checked: {report}")
    log.info("This does not validate domain meaning or make any LLM calls.")


if __name__ == "__main__":
    main()
