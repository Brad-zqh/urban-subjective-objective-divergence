"""Twelve-panel scenario plate with signed blue–red ECDF-area gradients.

Each area gradient is keyed to the exact projected-change value on the x axis:
negative is blue, zero is white and positive is red. No observations, schemes,
intervals, or interpretations are changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

import build_v211_scenario_inequality_refined_v1 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_scenario_inequality_refined_v4"
OLD = ROOT / "figures/v211_scenario_inequality_refined_v3"
original_save = prior.save_jpg_svg


def signed_fill(x: float, limit: float):
    base = np.asarray(to_rgb("#B40426" if x > 0 else "#2443A8"))
    strength = .16 + .79 * np.clip(abs(x) / max(limit, 1e-9), 0, 1)
    return 1 - strength * (1 - base)


def gradient_save(fig, output, *, dpi=600):
    if len(fig.axes) != 12:
        raise RuntimeError("Expected the registered twelve panels")
    for ax in fig.axes[:7]:
        if len(ax.lines) < 3:
            raise RuntimeError("ECDF curve and reference axes missing")
        xs = np.asarray(ax.lines[0].get_xdata(), float)
        ys = np.asarray(ax.lines[0].get_ydata(), float)
        if len(xs) < 100 or len(xs) != len(ys):
            raise RuntimeError("Registered ECDF points changed")
        areas = [c for c in ax.collections if isinstance(c, PolyCollection)]
        if len(areas) != 1:
            raise RuntimeError("Expected one original ECDF area")
        areas[0].remove()
        low, high = ax.get_xlim()
        edges = np.linspace(max(low, xs[0]), min(high, xs[-1]), 145)
        heights = np.interp(edges, xs, ys)
        limit = max(abs(low), abs(high))
        for left, right, y1, y2 in zip(edges[:-1], edges[1:], heights[:-1], heights[1:]):
            ax.fill_between([left, right], [0, 0], [y1, y2],
                            color=signed_fill((left + right) / 2, limit),
                            alpha=.63, linewidth=0, zorder=0, rasterized=True)
        for artist in ax.texts:
            if "median" in artist.get_text().lower():
                artist.set_fontsize(5.2)
                artist.set_color("#182B48")
        ax.tick_params(axis="x", labelsize=6.3)
        ax.tick_params(axis="y", labelsize=6.3)
    return original_save(fig, output, dpi=dpi)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.save_jpg_svg = gradient_save
    prior.main()
    for old in OLD.glob("source_*.csv"):
        new = OUT / old.name
        if not new.exists() or prior.sha256(old) != prior.sha256(new):
            raise RuntimeError(f"Scenario source table changed: {old.name}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update({
        "seven_ecdf_areas_use_signed_x_value_gradient": True,
        "twelve_original_panels_retained": True,
        "registered_source_tables_byte_identical_to_v3": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Scenario gradient visual QA failed")


if __name__ == "__main__":
    main()
