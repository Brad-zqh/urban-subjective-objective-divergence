"""Fail-closed checks for files that must not enter the public release."""

from __future__ import annotations

import argparse
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".osm.pbf",
    ".tif",
    ".tiff",
    ".pem",
    ".key",
}
FORBIDDEN_PARTS = {"raw", "restricted", "credentials", "resident_level"}
FORBIDDEN_NAMES = {".env", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
GENERATED_DIRS = {"draft", "outputs", "reports_v4", ".pytest_cache", ".ruff_cache"}
LOCAL_ONLY_PATHS = {"configs/paths.yaml", "configs/experiment_v1.yaml"}


def find_forbidden_paths(root: Path) -> list[Path]:
    """Return tracked-release candidates that violate the data boundary."""
    findings: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        relative_posix = relative.as_posix()
        if relative_posix in LOCAL_ONLY_PATHS:
            continue
        if any(part.lower() in GENERATED_DIRS for part in relative.parts):
            continue
        lowered = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if (
            lowered in FORBIDDEN_NAMES
            or any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
            or parts.intersection(FORBIDDEN_PARTS)
        ):
            findings.append(path)
    return sorted(findings)


def main() -> int:
    """Run the release-boundary check and return a process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = find_forbidden_paths(args.root.resolve())
    if findings:
        for path in findings:
            print(f"FORBIDDEN_RELEASE_PATH: {path}")
        return 1
    print(f"Release boundary passed: {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
