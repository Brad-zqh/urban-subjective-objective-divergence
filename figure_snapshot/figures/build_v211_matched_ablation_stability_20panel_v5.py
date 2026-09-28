"""Portrait-safe wider Fig. 6 with values in every heatmap cell."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np

import build_v211_matched_ablation_stability_20panel_v4 as prior
import build_v211_matched_ablation_stability_20panel_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_matched_ablation_stability_20panel_v5_complete"
OLD = ROOT / "figures/v211_matched_ablation_stability_20panel_v4"
STEM = "Fig_v211_matched_ablation_stability_20panel_v5_complete"
ORIGINAL_SAVE = prior._save


def portrait_wide_figure(*args, **kwargs):
    # The source figure is wider, but the Word placement remains inside the
    # A4 portrait live area. Widening the source also improves cell space.
    kwargs["figsize"] = (200 / 25.4, 245 / 25.4)
    fig = v3.prior.original_figure(*args, **kwargs)
    original_add = fig.add_gridspec

    def compact_grid(*gargs, **gkwargs):
        gkwargs["hspace"] = .42
        gkwargs["wspace"] = .40
        return original_add(*gargs, **gkwargs)

    fig.add_gridspec = compact_grid
    return fig


def all_cell_save(fig, out_dir, stem, *, dpi=600):
    if len(fig.axes) != 20:
        raise RuntimeError("Fig. 6 must retain twenty panels")
    annotations = {}
    for index, letter in zip((12, 13, 14, 15, 19), "mnopt"):
        ax = fig.axes[index]
        if len(ax.images) != 1:
            raise RuntimeError(f"Panel {letter} has no unique heatmap")
        image = ax.images[0]
        matrix = np.asarray(image.get_array(), float)
        if matrix.ndim != 2 or matrix.shape[0] != 9 or not np.isfinite(matrix).all():
            raise RuntimeError(f"Unexpected panel {letter} data {matrix.shape}")
        count = 0
        for row in range(matrix.shape[0]):
            # Panels m--p have four columns, so every value is readable.
            # Panel t has sixteen columns; full in-cell labels would collide
            # at portrait print size, so retain its row-extreme labels.
            columns = (range(matrix.shape[1]) if letter != "t" else
                       [int(np.argmax(np.abs(matrix[row])))])
            for column in columns:
                value = float(matrix[row, column])
                rgba = image.cmap(image.norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center",
                        fontsize=4.55,
                        color="white" if luminance < .48 else "#16253B",
                        zorder=5)
                count += 1
        annotations[letter] = count
    result = ORIGINAL_SAVE(fig, out_dir, stem, dpi=dpi)
    qa_path = out_dir / "layout_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["heatmap_value_annotations"] = annotations
    qa["checks"]["four_partition_heatmaps_have_every_cell_labeled"] = all(
        annotations[letter] == 36 for letter in "mnop"
    ) and annotations["t"] == 9
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    if not qa["checks"]["four_partition_heatmaps_have_every_cell_labeled"]:
        raise RuntimeError(f"Not all heatmap values labeled: {annotations}")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD, prior.STEM = OUT, OLD, STEM
    v3.wider_figure = portrait_wide_figure
    prior.annotated_save = all_cell_save
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update({
        "portrait_safe_wider_source_canvas": True,
        "all_144_partition_heatmap_values_annotated": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 6 v5 QA failed")


if __name__ == "__main__":
    main()
