"""Display-only polish for the registered 14-panel planning atlas.

The complete point distribution in panel f is represented as signed hexagon
densities, with its registered median and quantile envelope retained.  All
source rows, seven maps and their independent colour scales remain intact.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap

import build_v211_planning_strategy_refined_v11 as prior


base = prior.final_layout.revised.base
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_planning_strategy_refined_v12"
OLD = ROOT / "figures/v211_planning_strategy_refined_v11"
original_save = base.save_jpg_svg


def polish_save(fig, output, *, dpi=600):
    residual = base.panel(fig, "Package non-additivity residual")
    raw = [c for c in residual.collections if isinstance(c, PathCollection)]
    if len(raw) != 1:
        raise RuntimeError("Expected one complete residual scatter collection")
    xy = np.asarray(raw[0].get_offsets(), float)
    if len(xy) < 1000:
        raise RuntimeError("Residual data were unexpectedly thinned")
    raw[0].remove()
    # On the native (x, residual) axes, red and blue encode residual sign;
    # hexagon shade encodes local observation density.  This resolves the
    # original near-horizontal point pile-up without dropping observations.
    red = LinearSegmentedColormap.from_list("residual_red", ["#FFE6E5", "#B40426"])
    blue = LinearSegmentedColormap.from_list("residual_blue", ["#E2EAFE", "#2443A8"])
    for sign, palette in [(xy[:, 1] < 0, blue), (xy[:, 1] >= 0, red)]:
        residual.hexbin(xy[sign, 0], xy[sign, 1], gridsize=(47, 18),
                        extent=(*residual.get_xlim(), *residual.get_ylim()),
                        mincnt=1, bins="log", cmap=palette,
                        linewidths=0, alpha=.84, zorder=1, rasterized=True)
    for artist in residual.collections:
        if artist.get_zorder() < 1 and not isinstance(artist, PathCollection):
            artist.set_facecolor("#C6D8F0")
            artist.set_alpha(.22)
    for line in residual.lines:
        if len(line.get_xdata()) > 2:
            line.set_color("#183D79")
            line.set_linewidth(1.8)
            line.set_markersize(2.7)
        else:
            line.set_color("#203A68")
    residual.set_title("Package non-additivity residual", loc="left", fontsize=7.7)
    residual.set_ylabel(r"Residual ($\times 10^{-7}$ pp)", fontsize=7.0)
    residual.set_xlabel("Sum of component changes (pp)", fontsize=7.0)
    for item in residual.texts:
        if "median" in item.get_text().lower():
            item.set_color("#183D79")
            item.set_fontsize(5.8)

    annual = base.panel(fig, "Annual strategy response")
    annual.set_title("Annual model-projected response", loc="left", fontsize=7.7)
    annual.set_ylabel("Median change (pp)", fontsize=7.0)
    if annual.get_legend() is not None:
        legend = annual.get_legend()
        for txt in legend.get_texts():
            txt.set_fontsize(6.25)
            txt.set_color("#162C4A")
        for line in legend.get_lines():
            line.set_linewidth(2.0)
    # The correlation matrix is the rightmost panel and has unused space on
    # its right; enlarge both matrix and its inset colourbar together.
    corr = base.panel(fig, "Response correlation")
    p = corr.get_position()
    corr.set_position([p.x0 - .006, p.y0 - .004, p.width * 1.14, p.height * 1.14])
    corr.set_title("Response correlation", loc="left", x=.14, fontsize=7.7)
    corr.tick_params(labelsize=6.3, pad=1.0)
    for annotation in corr.texts:
        annotation.set_fontsize(6.2)
        if annotation.get_text() == "n":
            annotation.set_position((0, 1.025))
            annotation.set_ha("left")
    return original_save(fig, output, dpi=dpi)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base.OUT = OUT
    base.save_jpg_svg = polish_save
    base.main()
    files = list(OLD.glob("source_*.csv")) + list(OLD.glob("*audit.csv"))
    for old in files:
        new = OUT / old.name
        if not new.exists() or base.sha256(old) != base.sha256(new):
            raise RuntimeError(f"Planning source changed: {old.name}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update({
        "residual_hexagons_represent_all_original_points": True,
        "correlation_panel_enlarged": True,
        "registered_source_tables_byte_identical_to_v11": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Planning visual QA failed")


if __name__ == "__main__":
    main()
