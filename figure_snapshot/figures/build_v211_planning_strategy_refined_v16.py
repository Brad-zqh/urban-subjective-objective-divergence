"""Replace flat Fig. 12e time traces with a signed temporal-contrast glyph."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.lines import Line2D

import build_v211_planning_strategy_refined_v12 as prior


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_planning_strategy_refined_v15"
OUT = ROOT / "figures/v211_planning_strategy_refined_v16"
ORIGINAL_SAVE = prior.original_save
BLUES = ["#A7BFDF", "#789ACB", "#5076BB", "#2443A8"]
REDS = ["#EAB0B6", "#D27684", "#B40426"]


def contrast_save(fig, output, *, dpi=600):
    ax = prior.base.panel(fig, "Annual model-projected response")
    curves = [line for line in ax.lines if len(line.get_xdata()) == 10]
    if len(curves) != 7:
        raise RuntimeError("Expected seven registered annual response series")
    years = np.asarray(curves[0].get_xdata(), int)
    if not np.array_equal(years, np.arange(2014, 2024)):
        raise RuntimeError("Annual support must remain 2014–2023")
    records = []
    for curve in curves:
        values = np.asarray(curve.get_ydata(), float)
        if not np.isfinite(values).all():
            raise RuntimeError("Annual response contains missing values")
        name = "Integrated" if curve.get_label() == "Integrated retrofit" else curve.get_label()
        records.append((name, values))
    negative = sorted((item for item in records if item[1][-1] < 0),
                      key=lambda item: abs(item[1][-1]))
    positive = sorted((item for item in records if item[1][-1] >= 0),
                      key=lambda item: abs(item[1][-1]))
    if len(negative) != 4 or len(positive) != 3:
        raise RuntimeError("Registered 2023 response sign structure changed")
    ordered = negative + positive
    for line in list(ax.lines):
        line.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.grid(False)
    ax.axvline(0, color="#8696AA", lw=.55, zorder=0)
    colours = BLUES + REDS
    bound = max(abs(value) for _, vals in ordered for value in vals)
    ax.set_xlim(-1.40 * bound, 1.40 * bound)
    for row, ((name, values), colour) in enumerate(zip(ordered, colours)):
        baseline = values[:8]
        pre_median = float(np.median(baseline))
        # Every 2014–2021 annual median remains visible as a faint dot.
        ax.scatter(baseline, row + np.linspace(-.15, .15, 8), s=7,
                   color=colour, alpha=.52, linewidth=0, zorder=2)
        ax.plot([float(baseline.min()), float(baseline.max())], [row, row],
                color=colour, lw=1.05, alpha=.60, zorder=1)
        ax.plot([pre_median, values[-1]], [row, row], color=colour,
                lw=1.45, alpha=.90, zorder=2)
        ax.scatter([pre_median], [row], marker="o", s=16,
                   facecolor="white", edgecolor=colour, linewidth=.8, zorder=3)
        ax.scatter([values[-2]], [row - .10], marker="D", s=18,
                   color=colour, edgecolor="white", linewidth=.35, zorder=4)
        ax.scatter([values[-1]], [row + .10], marker="o", s=28,
                   color=colour, edgecolor="white", linewidth=.4, zorder=5)
        ax.text(1.37 * bound, row, f"{values[-1]:+.3f}", ha="right", va="center",
                color=colour, fontsize=5.5, zorder=6)
    ax.set_ylim(6.5, -.5)
    ax.set_yticks(np.arange(7), [name for name, _ in ordered])
    ax.set_xticks([-bound, 0, bound])
    ax.set_xlabel("Median model-projected change (pp)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, labelsize=5.9, pad=2.0)
    ax.tick_params(axis="x", length=2.0, labelsize=5.8)
    ax.set_title("Temporal contrast in model-projected response", loc="left", fontsize=7.7)
    key = [
        Line2D([], [], marker="o", mfc="white", mec="#516D9B", ls="none", label="2014–21 median"),
        Line2D([], [], marker="D", color="#516D9B", ls="none", label="2022"),
        Line2D([], [], marker="o", color="#516D9B", ls="none", label="2023"),
    ]
    ax.legend(handles=key, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.055),
              frameon=False, fontsize=4.6, handletextpad=.16, columnspacing=.45,
              markerscale=.7)
    residual = prior.base.panel(fig, "Package non-additivity residual")
    for line in residual.lines:
        values = np.asarray(line.get_ydata(), float)
        if len(values) <= 2 and np.nanmax(np.abs(values)) < 1e-8:
            line.set_color("#D6E1EF")
            line.set_linewidth(.45)
            line.set_zorder(0)
        else:
            line.set_alpha(0)
    return ORIGINAL_SAVE(fig, output, dpi=dpi)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.original_save = contrast_save
    prior.main()
    for old in list(OLD.glob("source_*.csv")) + list(OLD.glob("*audit.csv")):
        if prior.base.sha256(old) != prior.base.sha256(OUT / old.name):
            raise RuntimeError(f"Fig. 12 source changed: {old.name}")
    path = OUT / "validation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checks"].update({
        "all_70_annual_medians_retained_in_contrast_glyph": True,
        "rows_sorted_by_2023_sign_and_absolute_magnitude": True,
        "blue_then_red_light_to_dark_row_gradient": True,
        "source_tables_byte_identical_to_v15": True,
    })
    data["all_checks_passed"] = all(data["checks"].values())
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = OUT / "manifest.json"
    manifest = json.loads(meta.read_text(encoding="utf-8"))
    manifest["visual_revision"] = (
        "v16_temporal_contrast_glyph_with_2014_2021_annual_dots_2022_diamond_2023_point"
    )
    meta.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not data["all_checks_passed"]:
        raise RuntimeError("Fig. 12 v16 QA failed")


if __name__ == "__main__":
    main()
