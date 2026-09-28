"""Keep all heatmap values and identify the four MM-GTGNNWR graph variants."""
from __future__ import annotations

import json

import build_v211_matched_ablation_stability_20panel_v5 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v211_matched_ablation_stability_20panel_v6_labeled"
OLD = ROOT / "figures/v211_matched_ablation_stability_20panel_v5_complete"
STEM = "Fig_v211_matched_ablation_stability_20panel_v6_labeled"
ROW_LABELS = [
    "Ridge", "GTWR", "GNNWR", "GTNNWR", "GAM",
    "MM·mob", "MM·4r", "MM·f.mob", "MM·f.4r",
]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD, prior.STEM = OUT, OLD, STEM
    prior.v3.prior.baseline.HEAT_ROW_LABEL = ROW_LABELS
    prior.main()
    for name in prior.v3.prior.SOURCE_FILES:
        if sha256(OLD / name) != sha256(OUT / name):
            raise RuntimeError(f"Fig. 6 source changed: {name}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["row_label_key"] = "MM = MM-GTGNNWR; rows 6–9 are graph variants"
    validation["source_tables_byte_identical_to_v5"] = True
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
