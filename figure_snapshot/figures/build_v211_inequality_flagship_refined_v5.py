"""Add legible row-extreme values inside the existing SES heatmap."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np

import build_v211_inequality_flagship_refined_v4 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_inequality_flagship_refined_v5"
OLD = ROOT / "figures/v211_inequality_flagship_refined_v4"
_save = prior._save


def annotated_save(fig, out_dir):
    ax = fig.axes[0]
    if len(ax.images) != 1:
        raise RuntimeError("Expected the original SES heatmap")
    image = ax.images[0]
    values = np.asarray(image.get_array(), float)
    if values.shape != (13, 8):
        raise RuntimeError(f"Expected 13 variables by 8 schemes, got {values.shape}")
    count = 0
    for row, vector in enumerate(values):
        for column in sorted({int(np.argmin(vector)), int(np.argmax(vector))}):
            value = float(values[row, column])
            rgba = image.cmap(image.norm(value))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center",
                    fontsize=5.7, color="white" if luminance < .55 else "#182B48",
                    zorder=5)
            count += 1
    if count < 13 or count > 26:
        raise RuntimeError(f"Unexpected annotation count: {count}")
    return _save(fig, out_dir)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior._save = annotated_save
    prior.main()
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_revision"] = "v5_heatmap_row_minimum_and_maximum_labels"
    manifest["heatmap_annotations"] = "minimum and maximum scheme value per SES variable row"
    manifest["source_tables_byte_identical_to_v4"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
