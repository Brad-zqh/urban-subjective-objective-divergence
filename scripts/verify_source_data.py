"""Verify the released V211 figure and table source-data files.

Use ``--write-manifest`` only while preparing a reviewed release. The default
mode is read-only and fails when a file is missing, changed, or unregistered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "source_data"
MANIFEST = DATA_ROOT / "manifest_v211.json"
EXPECTED_COUNTS = {"figure": 54, "table_input": 10, "table_output": 14}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def classify(relative: Path) -> tuple[str, int | None]:
    parts = relative.parts
    if len(parts) == 2 and parts[0].startswith("fig") and parts[0][3:].isdigit():
        return "figure", int(parts[0][3:])
    if len(parts) == 3 and parts[:2] == ("tables", "inputs"):
        return "table_input", None
    if len(parts) == 3 and parts[:2] == ("tables", "output"):
        return "table_output", None
    raise ValueError(f"Unrecognised source-data path: {relative.as_posix()}")


def record(path: Path) -> dict[str, object]:
    relative = path.relative_to(DATA_ROOT)
    role, figure = classify(relative)
    item: dict[str, object] = {
        "path": relative.as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
    if figure is not None:
        item["figure"] = figure
        item["evidence_version"] = "V154 sealed simulation" if figure == 6 else "V211 R3 proxy"
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            item["columns"] = next(reader)
            item["rows"] = sum(1 for _ in reader)
    return item


def data_files() -> list[Path]:
    files = [path for path in DATA_ROOT.rglob("*") if path.is_file()]
    return sorted(
        (path for path in files if path.name not in {MANIFEST.name, "README.md", "LICENSE.md"}),
        key=lambda path: path.relative_to(DATA_ROOT).as_posix(),
    )


def make_manifest() -> dict[str, object]:
    files = [record(path) for path in data_files()]
    counts = {role: sum(item["role"] == role for item in files) for role in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected source-data counts: {counts} != {EXPECTED_COUNTS}")
    return {
        "schema_version": 1,
        "empirical_analysis_version": "V211 R3 compact proxy",
        "synthetic_validation_version": "V154 sealed simulation (Fig. 6 only)",
        "proxy_analysis": True,
        "formal": False,
        "scope": "Figure 2–13 plotted-data exports, added Fig. 2/6 inputs, and Table 1/S1–S13 reproduction inputs/outputs",
        "counts": counts,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    current = make_manifest()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {MANIFEST.relative_to(ROOT)}: {len(current['files'])} files")
        return 0
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError("Released source data differ from the reviewed manifest")
    print(f"Verified {len(current['files'])} V211 source-data files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
