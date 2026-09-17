"""Scaffold a new domain from `domains/_template/`.

Usage:
    python scripts/new_domain.py <domain_name>

Creates `domains/<domain_name>/` populated with:
  * The ontology configuration, prompt blocks, and README from `domains/_template/`.
  * The production prompts copied from `domains/presalt/prompts/`
    (these are domain-agnostic — `<<block>>` markers resolve from
    `prompt_blocks.yaml` at load time).
  * The saved BFO and RO resources, with generic relation constraints inherited
    from `domains/_shared/bfo_ro_relations.yaml`.

The staged folder must pass offline structural validation before publication.
After running, edit the two configuration files, set
`ONTOLOGY_CONFIG_PATH=domains/<domain_name>/ontology_config.yaml` in your
`.env`, and follow SETUP.md.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.domain_validation import validate_domain

_TEMPLATE_DIR = _REPO_ROOT / "domains" / "_template"
_PRESALT_DIR = _REPO_ROOT / "domains" / "presalt"

# Files the user edits — copied from _template/.
_USER_EDIT_FILES = (
    "ontology_config.yaml",
    "prompt_blocks.yaml",
    "README.md",
)

# Generic assets — copied from presalt/ (not user-edit; tuned later if needed).
_GENERIC_ASSETS = (
    ("prompts", "prompts"),
    ("resources/bfo-core.owl", "resources/bfo-core.owl"),
    ("resources/ro-core.owl", "resources/ro-core.owl"),
)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


def _validate_name(name: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise SystemExit(
            f"Invalid domain name '{name}'. Use lowercase letters, digits, "
            "and underscores (2-32 chars, starting with a letter)."
        )
    if name == "_template":
        raise SystemExit("'_template' is reserved.")


def _check_preconditions(target: Path) -> None:
    if not _TEMPLATE_DIR.is_dir():
        raise SystemExit(
            f"Template directory missing: {_TEMPLATE_DIR}. "
            "Run from a clean checkout."
        )
    if not _PRESALT_DIR.is_dir():
        raise SystemExit(
            f"Reference domain missing: {_PRESALT_DIR}. "
            "The scaffold script copies generic prompts and BFO from here."
        )
    if target.exists():
        raise SystemExit(
            f"Target already exists: {target}. "
            "Pick another name or delete the directory first."
        )


def _copy_user_files(target: Path) -> list[str]:
    copied: list[str] = []
    for fname in _USER_EDIT_FILES:
        src = _TEMPLATE_DIR / fname
        if not src.is_file():
            raise FileNotFoundError(f"Required starter file missing: {src}")
        shutil.copy2(src, target / fname)
        copied.append(fname)
    return copied


def _copy_generic_assets(target: Path) -> list[str]:
    copied: list[str] = []
    for src_rel, dst_rel in _GENERIC_ASSETS:
        src = _PRESALT_DIR / src_rel
        dst = target / dst_rel
        if not src.exists():
            raise FileNotFoundError(f"Required starter asset missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(dst_rel)
    return copied


def _print_next_steps(name: str, target: Path) -> None:
    rel = target.relative_to(_REPO_ROOT)
    print()
    print(f"  Created: {rel}")
    print()
    print("  Next steps:")
    print(f"    1. Edit {rel}/prompt_blocks.yaml (identity, questions, and examples).")
    print(f"    2. Edit {rel}/ontology_config.yaml (project.name, namespace, prefix).")
    print(f"    3. Check: python -m src.utils.domain_validation {rel.as_posix()}")
    print(f"    4. Activate the domain by adding this line to your .env:")
    print(f"         ONTOLOGY_CONFIG_PATH={rel.as_posix()}/ontology_config.yaml")
    print("    5. Follow SETUP.md for isolated input/output paths and a live smoke test.")
    print("       Customize the neutral examples before making billable model calls.")
    print()
    print("  See SETUP.md at the repo root for the full walk-through.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a new pipeline domain from domains/_template/."
    )
    parser.add_argument(
        "name",
        help="Domain name (lowercase letters, digits, underscores; 2-32 chars).",
    )
    args = parser.parse_args(argv)

    _validate_name(args.name)
    target = _REPO_ROOT / "domains" / args.name
    try:
        _check_preconditions(target)
        with tempfile.TemporaryDirectory(prefix=".new-domain-", dir=target.parent) as directory:
            staging = Path(directory)
            user_files = _copy_user_files(staging)
            generic_files = _copy_generic_assets(staging)
            report = validate_domain(staging)
            if target.exists():
                raise FileExistsError(f"Target already exists: {target}")
            staging.rename(target)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"  Structural checks passed: {report}")
    print(f"  Copied {len(user_files)} user-edit file(s) from _template/")
    print(f"  Copied {len(generic_files)} generic asset(s) from presalt/")

    _print_next_steps(args.name, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
