"""Fail closed when a release wheel contains files outside the public package."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

PACKAGE_PREFIX = "mm_gtgnnwr"
REQUIRED_MEMBER = f"{PACKAGE_PREFIX}/__init__.py"
FORBIDDEN_MEMBER_TOKENS = ("license_selection", "licence_selection", "licence_decision_required")


def resolve_wheel(path: Path) -> Path:
    """Resolve one wheel file from a file or single-wheel directory."""
    if path.is_file():
        if path.suffix != ".whl":
            raise ValueError(f"Expected a .whl file: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Wheel path does not exist: {path}")
    wheels = sorted(path.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel in {path}, found {len(wheels)}")
    return wheels[0]


def inspect_wheel(path: Path) -> list[str]:
    """Return release-boundary violations found inside one wheel archive."""
    findings: list[str] = []
    try:
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
    except BadZipFile as error:
        return [f"invalid wheel archive: {error}"]

    if REQUIRED_MEMBER not in members:
        findings.append(f"missing required package member: {REQUIRED_MEMBER}")

    dist_info_roots: set[str] = set()
    for member in members:
        pure = PurePosixPath(member)
        lowered = member.lower()
        if pure.is_absolute() or ".." in pure.parts:
            findings.append(f"unsafe archive path: {member}")
            continue
        root = pure.parts[0]
        if root.endswith(".dist-info"):
            dist_info_roots.add(root)
        elif root != PACKAGE_PREFIX:
            findings.append(f"unexpected top-level wheel member: {member}")
        if any(token in lowered for token in FORBIDDEN_MEMBER_TOKENS):
            findings.append(f"licence decision memo packaged as licence: {member}")

    if len(dist_info_roots) != 1:
        findings.append(f"expected one .dist-info directory, found {len(dist_info_roots)}")
    return findings


def main() -> int:
    """Validate a wheel path and return a process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="wheel file or directory containing one wheel")
    args = parser.parse_args()
    wheel = resolve_wheel(args.path.resolve())
    findings = inspect_wheel(wheel)
    if findings:
        for finding in findings:
            print(f"INVALID_WHEEL_CONTENT: {finding}")
        return 1
    print(f"Wheel content boundary passed: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
