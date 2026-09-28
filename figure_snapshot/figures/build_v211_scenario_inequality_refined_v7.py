"""Within-sign light-to-dark gradients for Fig. 13 directional bars."""
from __future__ import annotations

import json

import matplotlib as mpl

mpl.use("Agg")
import numpy as np

import build_v211_scenario_inequality_refined_v5 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v211_scenario_inequality_refined_v7"
OLD = ROOT / "figures/v211_scenario_inequality_refined_v6"
ORIGINAL_SAVE = prior._save


def gradient_bars(ax, *, expected: int) -> int:
    bars = [patch for patch in ax.patches
            if abs(float(patch.get_width())) > 1e-10 and float(patch.get_height()) > 0]
    if len(bars) != expected:
        raise RuntimeError(f"Expected {expected} directional bars, got {len(bars)}")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for patch in bars:
        x0 = float(patch.get_x())
        x1 = x0 + float(patch.get_width())
        left, right = sorted((x0, x1))
        y0 = float(patch.get_y())
        y1 = y0 + float(patch.get_height())
        base = np.asarray(patch.get_facecolor()[:3], dtype=float)
        # Blue fades toward the zero/boundary side; red deepens toward right.
        strength = (np.linspace(.95, .43, 256) if base[2] > base[0]
                    else np.linspace(.43, .95, 256))
        rgb = 1 - strength[:, None] * (1 - base[None, :])
        raster = np.ones((2, 256, 4), dtype=float)
        raster[:, :, :3] = rgb[None, :, :]
        im = ax.imshow(raster, extent=[left, right, y0, y1], aspect="auto",
                       interpolation="bicubic", zorder=patch.get_zorder() + .1)
        im.set_clip_path(patch)
        patch.set_facecolor("none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return len(bars)


def graduated_bar_save(fig, output, *, dpi=600):
    if len(fig.axes) != 12:
        raise RuntimeError("Expected all twelve scenario panels")
    counts = {
        "j_directional_agreement": gradient_bars(fig.axes[9], expected=16),
        "l_response_direction": gradient_bars(fig.axes[11], expected=14),
    }
    result = ORIGINAL_SAVE(fig, output, dpi=dpi)
    validation = OUT / "bar_gradient_qa.json"
    validation.write_text(json.dumps({"counts": counts,
                                      "same_hue_only": True,
                                      "bar_geometry_unchanged": True},
                                     indent=2), encoding="utf-8")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior._save = graduated_bar_save
    prior.main()
    for path in OLD.glob("source_*.csv"):
        if sha256(path) != sha256(OUT / path.name):
            raise RuntimeError(f"Fig. 13 source changed: {path.name}")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_revision"] = "v7_same_hue_internal_gradients_in_j_l_bars"
    manifest["source_tables_byte_identical_to_v6"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
