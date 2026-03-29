"""
Minimal console output helpers for PreSaltOntoLearn.

Uses ANSI escape codes (Windows 10+ terminals support them natively).
"""

import sys

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_DIM = "\033[2m"


def banner(step_number: int | str, title: str) -> None:
    """Print a prominent step banner."""
    line = f"\n{'═' * 3} Step {step_number}: {title} {'═' * 3}"
    print(f"{_BOLD}{_CYAN}{line}{_RESET}")


def info(msg: str) -> None:
    print(msg)


def success(msg: str) -> None:
    print(f"{_GREEN}{msg}{_RESET}")


def warn(msg: str) -> None:
    print(f"{_YELLOW}[WARN] {msg}{_RESET}")


def error(msg: str) -> None:
    print(f"{_RED}[ERROR] {msg}{_RESET}", file=sys.stderr)


def detail(msg: str) -> None:
    print(f"{_DIM}  {msg}{_RESET}")
