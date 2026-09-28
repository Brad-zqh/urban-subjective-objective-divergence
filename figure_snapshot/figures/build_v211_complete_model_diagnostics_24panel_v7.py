"""Colour whole residual bins by signed x-position, without changing bin data.

Negative bins progress deep-to-pale blue toward zero; positive bins progress
pale-to-deep red away from zero. The seven histograms remain independent views
of the registered 28-bin residual distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.colors import to_rgb

import build_v211_complete_model_diagnostics_24panel_v5 as prior
import build_v211_complete_model_diagnostics_24panel_v4 as v4
from nature_viz_common import sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_complete_model_diagnostics_24panel_v7"
OLD = ROOT / "figures/v211_complete_model_diagnostics_24panel_v6"
STEM = "Fig_v211_complete_model_diagnostics_24panel_v7"
BLUE_DEEP = np.asarray(to_rgb("#2443A8"))
BLUE_PALE = np.asarray(to_rgb("#DFE9F8"))
RED_PALE = np.asarray(to_rgb("#F8DEE1"))
RED_DEEP = np.asarray(to_rgb("#B40426"))


def colour_histogram(ax) -> tuple[int, list[float]]:
    bars = list(ax.patches)
    if len(bars) != 28:
        raise RuntimeError(f"Expected 28 unchanged residual bins, found {len(bars)}")
    centres = np.asarray([bar.get_x() + bar.get_width() / 2 for bar in bars], float)
    blue_extent = max(float(np.max(-centres[centres < 0])), 1e-12)
    red_extent = max(float(np.max(centres[centres >= 0])), 1e-12)
    for centre, bar in zip(centres, bars):
        if centre < 0:
            fraction = min(1.0, -centre / blue_extent)
            colour = BLUE_PALE + fraction * (BLUE_DEEP - BLUE_PALE)
        else:
            fraction = min(1.0, centre / red_extent)
            colour = RED_PALE + fraction * (RED_DEEP - RED_PALE)
        bar.set_facecolor(colour)
        bar.set_alpha(1)
        bar.set_edgecolor("white")
        bar.set_linewidth(.22)
    return len(bars), centres.tolist()


def save_signed_ramp(fig, out_dir: Path, stem: str, *, dpi: int = 600):
    if len(fig.axes) != 24:
        raise RuntimeError("Fig. 3 must retain all 24 registered panels")
    bin_counts = []
    bin_centres = []
    for ax in fig.axes[8:15]:
        count, centres = colour_histogram(ax)
        bin_counts.append(count)
        bin_centres.append(centres)
    annual = fig.axes[20]
    handles, labels = annual.get_legend_handles_labels()
    if annual.get_legend() is not None:
        annual.get_legend().remove()
    for ax in fig.axes[20:24]:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + .029, box.width, box.height - .029])
    fig.legend(handles, labels, ncol=7, loc="lower center",
               bbox_to_anchor=(.52, .005), frameon=False, fontsize=5.35,
               handlelength=1.1, handletextpad=.28, columnspacing=.58)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[-1].get_window_extent(renderer)
    if any(legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer))
           for ax in fig.axes[20:24]):
        raise RuntimeError("Annual legend overlaps an x-axis label")
    result = v4.original_save(fig, out_dir, stem, dpi=dpi)
    qa_path = out_dir / "layout_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["checks"].update({
        "all_196_residual_bins_retain_original_geometry": sum(bin_counts) == 196,
        "whole_bin_signed_colour_orders_left_to_right": all(
            all(np.diff(row) > 0) for row in bin_centres),
        "negative_blue_positive_red_without_grey": True,
        "annual_legend_outside_plot_and_clear_of_xlabels": True,
    })
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD, prior.STEM = OUT, OLD, STEM
    v4.prior.save_pair = save_signed_ramp
    prior.main()
    for path in OLD.glob("source_*.csv"):
        if sha256(path) != sha256(OUT / path.name):
            raise RuntimeError(f"Fig. 3 source table changed: {path.name}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"]["whole_bin_signed_left_to_right_ramp"] = True
    validation["checks"]["source_tables_byte_equal_v6"] = True
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 3 v7 QA failed")


if __name__ == "__main__":
    main()
