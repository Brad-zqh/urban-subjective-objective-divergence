"""Build or validate the metadata-only registry of external V211 figure tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

AUDIT = Path("figures/V211_CURRENT_ONLY_FIGURE_QA_20260910.csv")
EXPECTED_FIGURES = 18
EXPECTED_TABLES = 46
STATUS = "pending_disclosure_and_licence_review"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one source table."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_registry(workspace_root: Path) -> dict[str, object]:
    """Discover the current 18-figure source-table inventory without copying data."""
    audit_path = workspace_root / AUDIT
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing current figure QA table: {audit_path}")
    figures: list[dict[str, object]] = []
    tables: list[dict[str, object]] = []
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_FIGURES:
        raise RuntimeError(f"Expected {EXPECTED_FIGURES} current figures, found {len(rows)}")

    for row in rows:
        folder = row["folder"]
        expected = int(row["source_table_count"])
        source_root = workspace_root / "figures" / folder
        source_files = sorted(
            path
            for path in source_root.glob("source_*")
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
        )
        if len(source_files) != expected:
            raise RuntimeError(
                f"Source-table count drift for {folder}: expected {expected}, found {len(source_files)}"
            )
        figures.append(
            {
                "figure_folder": folder,
                "label": row["label"],
                "panels": int(row["panels"]),
                "declared_source_table_count": expected,
            }
        )
        for path in source_files:
            workspace_path = path.relative_to(workspace_root).as_posix()
            tables.append(
                {
                    "figure_folder": folder,
                    "workspace_relative_path": workspace_path,
                    "intended_release_path": f"source_data/{folder}/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "included_in_release": False,
                    "redistribution_status": STATUS,
                }
            )
    if len(tables) != EXPECTED_TABLES:
        raise RuntimeError(f"Expected {EXPECTED_TABLES} source tables, found {len(tables)}")
    return {
        "registry_schema": "v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analysis_version": "V211",
        "proxy_analysis": True,
        "formal": False,
        "figures": figures,
        "tables": tables,
        "figures_registered": len(figures),
        "tables_registered": len(tables),
        "source_tables_external": True,
        "restricted_media_embedded": False,
        "notice": (
            "This metadata registry does not grant redistribution rights and does not include table content."
        ),
    }


def validate_registry(payload: dict[str, object]) -> list[str]:
    """Return structural violations in an external source-table registry."""
    findings: list[str] = []
    figures = payload.get("figures", [])
    tables = payload.get("tables", [])
    if not isinstance(figures, list) or len(figures) != EXPECTED_FIGURES:
        findings.append("figure count is not 18")
        figures = []
    if not isinstance(tables, list) or len(tables) != EXPECTED_TABLES:
        findings.append("table count is not 46")
        tables = []
    if payload.get("proxy_analysis") is not True or payload.get("formal") is not False:
        findings.append("scientific boundary is missing")
    if payload.get("source_tables_external") is not True:
        findings.append("external source-table boundary is missing")

    counts: Counter[str] = Counter()
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    for table in tables:
        if not isinstance(table, dict):
            findings.append("table entry is not an object")
            continue
        folder = str(table.get("figure_folder", ""))
        counts[folder] += 1
        for key in ("workspace_relative_path", "intended_release_path"):
            value = str(table.get(key, ""))
            pure = PurePosixPath(value)
            if not value or pure.is_absolute() or ".." in pure.parts or ":" in value:
                findings.append(f"unsafe {key}: {value}")
        if not sha_pattern.fullmatch(str(table.get("sha256", ""))):
            findings.append(f"invalid SHA-256 for {table.get('workspace_relative_path')}")
        if table.get("included_in_release") is not False:
            findings.append(f"external table marked included: {table.get('workspace_relative_path')}")
        if table.get("redistribution_status") != STATUS:
            findings.append(f"invalid disclosure status: {table.get('workspace_relative_path')}")

    for figure in figures:
        if not isinstance(figure, dict):
            findings.append("figure entry is not an object")
            continue
        folder = str(figure.get("figure_folder", ""))
        expected = int(figure.get("declared_source_table_count", -1))
        if counts[folder] != expected:
            findings.append(f"registered source-table count mismatch for {folder}")
    return findings


def main() -> int:
    """Build the registry from the workspace or validate an existing copy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data_manifest"
        / "v211_external_figure_source_tables.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"External source registry does not exist: {args.output}")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        findings = validate_registry(payload)
        if findings:
            raise RuntimeError("; ".join(findings))
        print(f"External registry contract passed: {EXPECTED_FIGURES} figures, {EXPECTED_TABLES} tables")
        return 0

    payload = build_registry(args.workspace_root.resolve())
    findings = validate_registry(payload)
    if findings:
        raise RuntimeError("; ".join(findings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote external registry: {EXPECTED_FIGURES} figures, {EXPECTED_TABLES} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
