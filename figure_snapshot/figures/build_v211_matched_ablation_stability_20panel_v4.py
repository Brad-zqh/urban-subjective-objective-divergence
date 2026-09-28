"""Row-extreme value labels for Fig. 6 heatmaps, retaining the wide layout."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np

import build_v211_matched_ablation_stability_20panel_v3 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_matched_ablation_stability_20panel_v4"
OLD = ROOT / "figures/v211_matched_ablation_stability_20panel_v3"
STEM = "Fig_v211_matched_ablation_stability_20panel_v4"
_save = prior.prior.save_pair


def annotated_save(fig, out_dir, stem, *, dpi=600):
    if len(fig.axes) != 20:
        raise RuntimeError("Fig. 6 panel count changed")
    for index in (12, 13, 14, 15, 19):
        ax = fig.axes[index]
        if len(ax.images) != 1:
            raise RuntimeError(f"Fig. 6 panel {index} has no single heatmap")
        image = ax.images[0]
        matrix = np.asarray(image.get_array(), float)
        if matrix.ndim != 2 or matrix.shape[0] != 9:
            raise RuntimeError(f"Unexpected Fig. 6 heatmap shape {matrix.shape}")
        for row, values in enumerate(matrix):
            column = int(np.argmax(np.abs(values)))
            value = float(values[column])
            rgba = image.cmap(image.norm(value))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=4.8, color="white" if luminance < .55 else "#172B48",
                    zorder=5)
    return _save(fig, out_dir, stem, dpi=dpi)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD, prior.STEM = OUT, OLD, STEM
    prior.prior.save_pair = annotated_save
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"]["five_heatmaps_label_row_extreme_values"] = True
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 6 annotation QA failed")


if __name__ == "__main__":
    main()
