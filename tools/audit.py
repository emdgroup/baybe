"""Wrapper around ``uv audit`` that only fails on direct dependency vulnerabilities.

This script runs ``uv audit`` with any extra arguments forwarded, parses its output
to determine which packages are flagged, and cross-references them against the
project's direct dependencies (from ``pyproject.toml``). The output is reorganized
into clearly separated sections for direct vs. transitive vulnerabilities, and the
exit code only reflects whether *direct* dependencies are affected.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# PEP 503 normalization: lowercase, runs of [-_.] become a single dash
_NORMALIZE_RE = re.compile(r"[-_.]+")


def _normalize(name: str) -> str:
    return _NORMALIZE_RE.sub("-", name).lower()


def _extract_package_name(spec: str) -> str:
    """Return the bare package name from a PEP 508 dependency string."""
    return re.split(r"[\[><=!~;\s]", spec, maxsplit=1)[0].strip()


def _get_direct_dependencies(pyproject_path: Path) -> set[str]:
    """Collect all direct dependency names from ``pyproject.toml``.

    This includes core ``dependencies`` as well as every entry listed under
    ``[project.optional-dependencies]``.  Self-references (e.g. ``baybe[chem]``)
    are excluded.
    """
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    project_name = _normalize(project.get("name", ""))

    deps: set[str] = set()

    # Core dependencies
    for spec in project.get("dependencies", []):
        name = _normalize(_extract_package_name(spec))
        if name != project_name:
            deps.add(name)

    # Optional dependencies (all extras)
    for group_deps in project.get("optional-dependencies", {}).values():
        for spec in group_deps:
            name = _normalize(_extract_package_name(spec))
            if name != project_name:
                deps.add(name)

    return deps


# Regex matching the header line for each vulnerable package in ``uv audit``
# output, e.g. ``requests 2.32.5 has 1 known vulnerability:``
_VULN_HEADER_RE = re.compile(
    r"^(\S+)\s+\S+\s+has\s+\d+\s+known\s+vulnerabilit", re.MULTILINE
)


def _parse_vulnerability_blocks(output: str) -> list[tuple[str, str]]:
    """Parse ``uv audit`` output into (package_name, block_text) pairs.

    Each block starts with a header like ``requests 2.32.5 has 1 known
    vulnerability:`` and includes all subsequent lines until the next header
    or end of output.
    """
    # Find all header positions
    headers = list(_VULN_HEADER_RE.finditer(output))
    if not headers:
        return []

    blocks: list[tuple[str, str]] = []
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(output)
        pkg_name = _normalize(match.group(1))
        block_text = output[start:end].rstrip()
        blocks.append((pkg_name, block_text))

    return blocks


def _count_vulns(blocks: list[tuple[str, str]]) -> int:
    """Count the total number of individual vulnerabilities across blocks."""
    total = 0
    for _, text in blocks:
        # Each vulnerability entry starts with "- GHSA-" or "- CVE-" etc.
        total += len(re.findall(r"^- \S+", text, re.MULTILINE))
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SEPARATOR = "=" * 72


def main() -> int:
    """Run ``uv audit`` and only fail on direct dependency vulnerabilities."""
    # Locate pyproject.toml relative to this script (repo root)
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"ERROR: {pyproject_path} not found", file=sys.stderr)
        return 1

    direct_deps = _get_direct_dependencies(pyproject_path)

    # Run uv audit, forwarding any extra CLI arguments (e.g. --ignore-until-fixed)
    result = subprocess.run(
        ["uv", "audit", *sys.argv[1:]],
        capture_output=True,
        text=True,
        check=False,
    )

    # Forward stderr (warnings, progress, etc.)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    # If uv audit passed, we pass too
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        return 0

    # Parse vulnerability blocks and classify them
    blocks = _parse_vulnerability_blocks(result.stdout)
    direct_blocks = [(name, text) for name, text in blocks if name in direct_deps]
    transitive_blocks = [
        (name, text) for name, text in blocks if name not in direct_deps
    ]

    n_direct = _count_vulns(direct_blocks)
    n_transitive = _count_vulns(transitive_blocks)

    # Print transitive vulnerabilities (informational)
    if transitive_blocks:
        print(_SEPARATOR)
        names = ", ".join(sorted({n for n, _ in transitive_blocks}))
        print(
            f"TRANSITIVE dependency vulnerabilities ({n_transitive} total, "
            f"not causing failure): {names}"
        )
        print(_SEPARATOR)
        for _, text in transitive_blocks:
            print()
            print(text)
        print()

    # Print direct vulnerabilities (these cause failure)
    if direct_blocks:
        print(_SEPARATOR)
        names = ", ".join(sorted({n for n, _ in direct_blocks}))
        print(
            f"DIRECT dependency vulnerabilities ({n_direct} total, "
            f"CAUSING FAILURE): {names}"
        )
        print(_SEPARATOR)
        for _, text in direct_blocks:
            print()
            print(text)
        print()
        return 1

    print(_SEPARATOR)
    print(
        f"All {n_transitive} vulnerabilities are in transitive dependencies -- passing."
    )
    print(_SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
