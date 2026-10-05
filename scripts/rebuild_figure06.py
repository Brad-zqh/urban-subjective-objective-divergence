"""Rebuild the paper's V154 known-truth Figure 6 from its released synthetic data.

The numbered visual revisions are retained for provenance. This script runs
the missing revisions in order and verifies the final panel/source-data gate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figure_snapshot/figures"
OUTPUT = FIGURES / "v154_known_truth_maintext_v9"
REFERENCE = ROOT / "source_data/fig06/source_model_metric_intervals.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    for version in range(2, 10):
        output = FIGURES / f"v154_known_truth_maintext_v{version}"
        if output.exists():
            if not (output / "validation.json").is_file():
                raise RuntimeError(f"Incomplete existing figure revision: {output}")
            continue
        script = FIGURES / f"build_v154_known_truth_maintext_v{version}.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)

    validation = json.loads((OUTPUT / "validation.json").read_text(encoding="utf-8"))
    if validation.get("panels") != 20 or not all(validation["checks"].values()):
        raise RuntimeError("Figure 6 panel or layout validation failed")
    actual = OUTPUT / "source_model_metric_intervals.csv"
    if sha256(actual) != sha256(REFERENCE):
        raise RuntimeError("Figure 6 numeric source data changed during rendering")
    for extension in ("jpg", "svg"):
        if not (OUTPUT / f"Fig_v154_known_truth_maintext_v9.{extension}").is_file():
            raise FileNotFoundError(extension)
    print("Verified V154 Figure 6: 20 panels and byte-identical interval source data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
