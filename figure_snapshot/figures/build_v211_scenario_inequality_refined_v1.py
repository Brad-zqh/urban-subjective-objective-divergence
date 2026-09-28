"""Non-destructive label refinement of the complete 12-panel V211 plate.

The registered source tables and original drawing script are left untouched.
Only annotation placement, repeated labels, and the JPG/SVG export contract
are changed here. Panels a–l and every source row are retained.
"""

from __future__ import annotations

import importlib
import json
import string
from pathlib import Path

import matplotlib.pyplot as plt

from v211_refinement_common import save_jpg_svg, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_scenario_inequality_refined_v1"
BASE = importlib.import_module("build_v211_scenario_inequality_20panel")
SOURCE = ROOT / "figures/build_v211_scenario_inequality_20panel.py"
SOURCE_HASH = sha256(SOURCE)

SHORT_STRATEGIES = ["Green", "Upkeep", "Heat", "Air", "Quiet", "Traffic", "Package"]


def find_panel(fig, title: str):
    matches = [ax for ax in fig.axes if ax.get_title(loc="left") == title]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one panel titled {title!r}; got {len(matches)}")
    return matches[0]


def refine_and_save(fig, _out_dir, _stem, *, dpi=600):
    if len(fig.axes) != 12:
        raise RuntimeError(f"Expected 12 main axes, got {len(fig.axes)}")

    h = find_panel(fig, "Nested response intervals")
    h.set_yticks(range(7), [SHORT_STRATEGIES[i] for i in [5, 1, 2, 3, 0, 6, 4]])
    h.tick_params(axis="y", labelsize=6.0, pad=2.2)

    i = find_panel(fig, "Standardized SES prediction gaps")
    i.set_yticks(range(13), [
        "Age ≥65", "Age <18", "Higher ed.", "Black pop.", "Transit",
        "Walk", "Rent burden", "Hisp./Lat.", "Log income", "Male",
        "Population", "Poverty", "Unemployed",
    ])
    i.tick_params(axis="y", labelsize=6.0, pad=2.2)
    i.set_xticks(range(8), ["AR10", "AR20", "PC10", "PC20", "R45·10", "R45·20", "EQ10", "EQ20"])
    i.tick_params(axis="x", labelrotation=90, labelsize=5.7, pad=2)
    for label in i.get_xticklabels():
        label.set_ha("center")
        label.set_va("top")

    j = find_panel(fig, "Directional agreement")
    j.set_yticks(range(13), [""] * 13)
    j.tick_params(axis="y", length=0)

    k = find_panel(fig, "Magnitude–heterogeneity")
    # The original seven full labels overlap points and each other. A compact
    # numerical key links directly to panels a–g and is explained in the caption.
    for label in list(k.texts):
        if label.get_text() in BASE.SHORT.values():
            label.remove()
    offsets = [(3, 4), (-5, 4), (3, 4), (3, 4), (3, 4), (-5, 4), (3, 4)]
    scenario_medians = BASE.pd.read_csv(OUT / "source_scenario_nested_intervals.csv") if (OUT / "source_scenario_nested_intervals.csv").exists() else None
    # Point order is the source SCENARIOS order, not sorted legend order.
    if scenario_medians is None:
        scenario = BASE.pd.read_parquet(BASE.RUNS / "v211_compact_proxy_scenarios_r3_indexed_seed_filtered/outer_test_scenario_predictions.parquet")
        stats = []
        for key in BASE.SCENARIOS:
            values = scenario.loc[scenario.scenario.eq(key), "prediction_difference"].to_numpy(float)
            q05, q25, median, q75, q95 = BASE.qstats(values)
            stats.append((median, q75 - q25))
    else:
        records = scenario_medians.set_index("scenario")
        stats = [(float(records.loc[key, "median"]), float(records.loc[key, "q75"] - records.loc[key, "q25"])) for key in BASE.SCENARIOS]
    for index, ((x, y), offset) in enumerate(zip(stats, offsets), start=1):
        k.annotate(str(index), (x, y), xytext=offset, textcoords="offset points",
                   ha="left" if offset[0] > 0 else "right", va="bottom",
                   fontsize=5.9, color="black", fontweight="normal")
    k.set_xlabel("Median change (pp)")
    k.set_xticks([-0.05, 0.00, 0.05])

    l = find_panel(fig, "Response direction")
    l.set_yticks(range(7), SHORT_STRATEGIES)
    l.tick_params(axis="y", labelsize=6.0, pad=2.2)

    # Tight but still disjoint panel geometry. The left matrix is the sole
    # carrier of SES row labels; all paired rows align by registered order.
    fig.canvas.draw()
    delivery = save_jpg_svg(fig, OUT, dpi=dpi)
    return delivery


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    BASE.OUT = OUT
    BASE.STEM = OUT.name
    BASE.save_delivery = refine_and_save
    BASE.main()
    if sha256(SOURCE) != SOURCE_HASH:
        raise RuntimeError("Original drawing script changed during refinement")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["refinement"] = {
        "source_script": str(SOURCE),
        "source_script_sha256": SOURCE_HASH,
        "changes": "labels and annotation geometry only; all 12 panels and source rows retained",
        "panel_k_key": {string.ascii_lowercase[index]: index + 1 for index in range(7)},
        "proxy_analysis": True,
        "formal": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["delivery"] = manifest["delivery"]
    validation["checks"]["visible_text_bbox_collision_free"] = manifest["delivery"]["text_overlap_count"] == 0
    validation["checks"]["only_jpg_svg_visual_exports"] = sorted(p.suffix.lower() for p in OUT.iterdir() if p.suffix.lower() in {".jpg", ".svg", ".pdf", ".png", ".tif", ".tiff"}) == [".jpg", ".svg"]
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(OUT)
    print(json.dumps({"overlaps": manifest["delivery"]["text_overlap_count"], "all_checks_passed": validation["all_checks_passed"]}))


if __name__ == "__main__":
    main()
