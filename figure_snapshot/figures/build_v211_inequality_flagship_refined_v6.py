"""Complete Fig. 11 labels and ordered within-facet signed hue ramps."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.colors import to_rgb

import build_v211_inequality_flagship_refined_v5 as prior
import build_v211_inequality_flagship_refined_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_inequality_flagship_refined_v6"
OLD = ROOT / "figures/v211_inequality_flagship_refined_v5"
ORIGINAL_SAVE = prior._save


def scheme_ordered_colours(values: np.ndarray):
    """Use a consistent top-to-bottom luminance ramp; hue still denotes sign."""
    if len(values) != 8:
        raise RuntimeError("Every SES facet must retain eight schemes")
    colours = []
    for row, value in enumerate(values):
        endpoint = np.asarray(to_rgb("#B40426" if value > 0 else "#2443A8"))
        amount = .47 + .50 * row / 7
        colours.append(tuple(1 - amount * (1 - endpoint)))
    return colours


def fully_annotated_save(fig, out_dir):
    ax = fig.axes[0]
    if len(ax.images) != 1:
        raise RuntimeError("Expected the original SES heatmap")
    image = ax.images[0]
    values = np.asarray(image.get_array(), float)
    if values.shape != (13, 8) or not np.isfinite(values).all():
        raise RuntimeError(f"Unexpected SES heatmap values: {values.shape}")
    count = 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            rgba = image.cmap(image.norm(value))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center",
                    fontsize=5.15, color="white" if luminance < .49 else "#182B48",
                    zorder=5)
            count += 1
    if count != 104:
        raise RuntimeError("All 104 SES values must be annotated")
    return ORIGINAL_SAVE(fig, out_dir)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    v4.local_signed_colours = scheme_ordered_colours
    prior.annotated_save = fully_annotated_save
    prior.main()
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "visual_revision": "v6_all_104_heatmap_values_and_ordered_signed_facet_ramps",
        "heatmap_annotations": "all 13 variables by all 8 registered schemes",
        "gradient_scale": "scheme display order within each facet; hue remains sign",
        "source_tables_byte_identical_to_v5": True,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
