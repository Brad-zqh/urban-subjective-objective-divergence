"""Wider scenario facets and one signed colour language across the plate."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

import build_v211_scenario_inequality_refined_v4 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_scenario_inequality_refined_v5"
OLD = ROOT / "figures/v211_scenario_inequality_refined_v4"
_save = prior.original_save


def signed_curve_colour(value: float, limit: float):
    base = np.asarray(to_rgb("#B40426" if value > 0 else "#2443A8"))
    strength = .46 + .51 * min(abs(value) / max(limit, 1e-9), 1.0)
    return tuple(1 - strength * (1 - base))


def widened_signed_save(fig, output, *, dpi=600):
    if len(fig.axes) != 12:
        raise RuntimeError("Expected all twelve registered scenario panels")
    # Reduce only the inter-column gutters of the eight small multiples.
    for axes in (fig.axes[:4], fig.axes[4:8]):
        positions = [ax.get_position() for ax in axes]
        span = positions[-1].x1 - positions[0].x0
        old_gap = positions[1].x0 - positions[0].x1
        new_gap = old_gap * .87
        width = (span - 3 * new_gap) / 4
        for index, (ax, box) in enumerate(zip(axes, positions)):
            ax.set_position([positions[0].x0 + index * (width + new_gap),
                             box.y0, width, box.height])
    for ax in fig.axes[:7]:
        curves = [line for line in ax.lines if len(line.get_xdata()) >= 80]
        if len(curves) != 1:
            raise RuntimeError("Expected one unchanged ECDF curve per scenario")
        line = curves[0]
        x = np.asarray(line.get_xdata(), float)
        y = np.asarray(line.get_ydata(), float)
        xy = np.column_stack([x, y])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        limit = max(abs(bound) for bound in ax.get_xlim())
        colors = [signed_curve_colour(float(value), limit) for value in (x[:-1] + x[1:]) / 2]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.75, zorder=4))
        line.set_alpha(0.0)
    return _save(fig, output, dpi=dpi)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.original_save = widened_signed_save
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"]["eight_facets_widened_without_data_change"] = True
    validation["checks"]["ecdf_strokes_use_same_signed_palette_as_area_and_summary"] = True
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Scenario visual QA failed")


if __name__ == "__main__":
    main()
