#!/usr/bin/env python
"""The repo's CI-ish check task: lint, format-check, then tests.

Runs, in order, against the repo root:

1. ``ruff check``          — lint
2. ``ruff format --check`` — formatting drift
3. ``pytest``              — the test suite

Every step runs even if an earlier one fails, so one invocation reports every
problem. The process exits non-zero if any step failed.

pytest exits ``5`` when it collects zero tests. During early phases the suite is
nearly empty, so that code is reported as a *warning* and does not fail the task
(see ``PYTEST_NO_TESTS_COLLECTED``).

Tools are invoked as ``sys.executable -m <tool>``, so the script always uses the
interpreter it was launched with -- no ``make``, no PATH lookups, no shell
quoting differences between PowerShell and POSIX shells.

Usage::

    .venv/Scripts/python.exe scripts/check.py        # Windows
    .venv/bin/python scripts/check.py                # Unix
    uv run scripts/check.py                          # either, via uv
    ... scripts/check.py --fix                       # autofix lint + format first
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: pytest's exit code for "no tests were collected". Treated as a warning, not a
#: failure: the suite is legitimately empty in early phases of the build plan.
PYTEST_NO_TESTS_COLLECTED = 5

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

REQUIRED_MODULES = ("ruff", "pytest")


@dataclass(frozen=True)
class Step:
    """One command in the check pipeline."""

    name: str
    argv: Sequence[str]
    #: Exit codes that are tolerated with a warning, mapped to the reason shown.
    warn_codes: dict[int, str] | None = None


@dataclass(frozen=True)
class Result:
    """The outcome of running a :class:`Step`."""

    step: Step
    status: str
    returncode: int
    seconds: float
    note: str = ""


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


_COLORS = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}


def _paint(text: str, status: str) -> str:
    if not _supports_color():
        return text
    return f"{_COLORS[status]}{text}\033[0m"


def _steps(*, fix: bool) -> list[Step]:
    py = sys.executable
    steps: list[Step] = []
    if fix:
        steps += [
            Step("ruff check --fix", [py, "-m", "ruff", "check", "--fix", "."]),
            Step("ruff format", [py, "-m", "ruff", "format", "."]),
        ]
    else:
        steps += [
            Step("ruff check", [py, "-m", "ruff", "check", "."]),
            Step("ruff format --check", [py, "-m", "ruff", "format", "--check", "."]),
        ]
    steps.append(
        Step(
            "pytest",
            [py, "-m", "pytest"],
            warn_codes={PYTEST_NO_TESTS_COLLECTED: "no tests collected"},
        )
    )
    return steps


def _run(step: Step) -> Result:
    print(f"\n==> {step.name}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(step.argv, cwd=REPO_ROOT, check=False)
    elapsed = time.perf_counter() - started

    code = completed.returncode
    if code == 0:
        return Result(step, PASS, code, elapsed)
    warn_reason = (step.warn_codes or {}).get(code)
    if warn_reason is not None:
        print(
            _paint(
                f"warning: {step.name}: {warn_reason} (exit {code}) -- not a failure",
                WARN,
            )
        )
        return Result(step, WARN, code, elapsed, warn_reason)
    return Result(step, FAIL, code, elapsed, f"exit {code}")


def _missing_tools() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _report(results: Sequence[Result]) -> None:
    width = max(len(r.step.name) for r in results)
    print("\n" + "-" * 60)
    print(f"check summary  ({sys.executable})")
    for r in results:
        note = f"  {r.note}" if r.note else ""
        print(f"  {_paint(r.status, r.status)}  {r.step.name:<{width}}  {r.seconds:6.2f}s{note}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check",
        description="Run ruff check, ruff format --check, and pytest.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="autofix mode: run 'ruff check --fix' and 'ruff format' instead of the "
        "read-only checks, then run pytest.",
    )
    args = parser.parse_args(argv)

    missing = _missing_tools()
    if missing:
        print(
            f"error: {', '.join(missing)} not importable by {sys.executable}.\n"
            "Install the dev dependency group first: "
            ".venv-tools/Scripts/uv.exe sync --group dev",
            file=sys.stderr,
        )
        return 2

    results = [_run(step) for step in _steps(fix=args.fix)]
    _report(results)

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    if failures:
        names = ", ".join(r.step.name for r in failures)
        print(_paint(f"\nFAILED: {names}", FAIL))
        return 1
    suffix = f" ({len(warnings)} warning{'s' if len(warnings) != 1 else ''})" if warnings else ""
    print(_paint(f"\nAll checks passed{suffix}.", WARN if warnings else PASS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
