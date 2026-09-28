"""Within-variable signed saturation gradients for all thirteen SES facets."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgb

import build_v211_inequality_flagship_refined_v3 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_inequality_flagship_refined_v4"
OLD = ROOT / "figures/v211_inequality_flagship_refined_v3"
_save = prior.original_save


def local_signed_colours(values: np.ndarray):
    magnitudes = np.abs(values)
    low, high = float(np.min(magnitudes)), float(np.max(magnitudes))
    rank = (magnitudes - low) / max(high - low, 1e-12)
    return [tuple(1 - (.43 + .54 * strength) *
                  (1 - np.asarray(to_rgb("#B40426" if value > 0 else "#2443A8"))))
            for value, strength in zip(values, rank)]


def gradient_save(fig, out_dir):
    if len(fig.axes) != 15:
        raise RuntimeError("SES atlas must retain fifteen panels")
    for ax in fig.axes[2:]:
        if len(ax.collections) != 2:
            raise RuntimeError("SES facet does not contain its original stems and points")
        stems, points = ax.collections
        if not isinstance(stems, LineCollection) or not isinstance(points, PathCollection):
            raise RuntimeError("Unexpected SES facet artists")
        values = np.asarray(points.get_offsets(), float)[:, 0]
        if len(values) != 8:
            raise RuntimeError("The eight registered schemes must remain visible")
        colours = local_signed_colours(values)
        stems.set_colors(colours)
        points.set_facecolors(colours)
        points.set_edgecolors("#203A68")
    return _save(fig, out_dir)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.original_save = gradient_save
    prior.main()
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_revision"] = "v4_within_variable_signed_saturation_gradient"
    manifest["gradient_scale"] = "within SES facet only; not cross-variable comparable"
    manifest["source_tables_byte_identical_to_v3"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
