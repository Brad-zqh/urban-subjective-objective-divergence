"""Non-destructive V154 Figure 7 typography refinement.

The two top plots share the same model order.  The second plot's duplicate
row labels collided with the first panel; suppress only those duplicates.
All 20 evidence panels and the sealed V154 data stay unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import build_v154_known_truth_maintext_v2 as base
from nature_viz_common import ROOT, sha256

OUT = ROOT / "figures/v154_known_truth_maintext_v3"
ORIGINAL_INTERVALS = base.intervals


def intervals_without_duplicate_right_labels(ax, metrics, metric, index, title, xlabel):
    result = ORIGINAL_INTERVALS(ax, metrics, metric, index, title, xlabel)
    if index == 1:
        ax.tick_params(axis="y", left=False, labelleft=False)
    return result


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Version exists; do not overwrite: {OUT}")
    base.OUT = OUT
    base.STEM = "Fig_v154_known_truth_maintext_v3"
    base.intervals = intervals_without_duplicate_right_labels
    base.main()
    prior = ROOT / "figures/v154_known_truth_maintext_v2/source_model_metric_intervals.csv"
    current = OUT / "source_model_metric_intervals.csv"
    if sha256(prior) != sha256(current):
        raise RuntimeError("V154 interval source changed")
    validation_path = OUT / "validation.json"
    data = json.loads(validation_path.read_text(encoding="utf-8"))
    data["v3_visual_change"] = "Hide duplicate model y labels in top-right panel only"
    data["source_interval_sha256_equal_v2"] = True
    data["scientific_boundary"] = "Sealed V154 simulation, not V211 empirical evidence"
    validation_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "QA_CN.md").write_text(
        "# V154 Fig. 7 v3 版式核验\n\n"
        "- 20 个既有子图完整保留；只隐藏顶部右图与左图重复的模型行名，消除跨面板碰撞。\n"
        "- 左图仍完整标出全部六个模型；色标仍与矩阵等高，蓝红色域和数值未变。\n"
        "- 来源区间 CSV 与 v2 SHA-256 完全相同；V154 sealed simulation 不混入 V211 实证。\n"
        "- 只交付 JPG 与可编辑 SVG；旧版本不覆盖。\n", encoding="utf-8")
    print("V154 Fig. 7 v3: 20 panels; source intervals unchanged")


if __name__ == "__main__":
    main()
