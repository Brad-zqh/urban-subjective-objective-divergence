"""Clean-clone checks for the released figure and table source data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts/verify_source_data.py"
REBUILD = ROOT / "scripts/rebuild_study_tables.py"
FIGURE06 = ROOT / "scripts/rebuild_figure06.py"
TABLES = ROOT / "source_data/tables/output"


def test_source_data_manifest_matches_all_released_files() -> None:
    subprocess.run(
        [sys.executable, str(VERIFY)], cwd=ROOT, check=True, capture_output=True, text=True
    )


def test_table_csvs_rebuild_byte_identically(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REBUILD), "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    published = sorted(TABLES.glob("*.csv"))
    rebuilt = sorted(tmp_path.glob("*.csv"))
    assert len(published) == len(rebuilt) == 14
    for original in published:
        assert original.read_bytes() == (tmp_path / original.name).read_bytes(), original.name


def test_figure06_rebuilds_from_separate_v154_synthetic_data() -> None:
    subprocess.run(
        [sys.executable, str(FIGURE06)], cwd=ROOT, check=True, capture_output=True, text=True
    )
