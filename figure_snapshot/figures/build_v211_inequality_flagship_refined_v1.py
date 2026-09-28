"""Readable 15-panel V211 SES prediction-gap atlas; legacy art is untouched."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from nature_viz_common import BLUE, INK, RED, setup_style
from v211_refinement_common import save_jpg_svg, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_multidimensional_inequality_r3_indexed_seed_filtered/ses_prediction_disparity_by_scheme.csv"
OUT = ROOT / "figures/v211_inequality_flagship_refined_v1"

SES = [
    "ses__age_65plus_pct", "ses__age_under18_pct",
    "ses__bachelors_or_higher_pct_25plus", "ses__black_alone_pct",
    "ses__commute_transit_pct", "ses__commute_walk_pct",
    "ses__gross_rent_30plus_pct", "ses__hispanic_latino_pct",
    "ses__log_income_nominal", "ses__male_pct", "ses__population_log1p",
    "ses__poverty_pct", "ses__unemployment_pct",
]
LABELS = [
    "Age ≥65", "Age <18", "Higher education", "Black population",
    "Transit commute", "Walk commute", "Rent burden", "Hispanic/Latino",
    "Log income", "Male population", "Population size", "Poverty", "Unemployment",
]


def scheme_code(value: str) -> str:
    names = {
        "axis_recursive": "AR", "polar_north_clockwise_equal_count": "PC",
        "rotated45_recursive": "R45", "y_equal_count_stripes": "EQ",
    }
    stem, vintage = value.rsplit("__v", 1)
    return f"{names[stem]}·{vintage[-2:]}"


def panel(ax, index: int, title: str, x: float = -0.10) -> None:
    ax.set_title(title, loc="left", pad=3.0, fontsize=7.7, color="black")
    ax.text(x, 1.02, chr(97 + index), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.3, fontweight="bold", color="black")


def main() -> None:
    before = sha256(SOURCE)
    setup_style(7.0)
    mpl.rcParams.update({"svg.fonttype": "none", "font.weight": "normal"})
    source = pd.read_csv(SOURCE)
    schemes = list(source.scheme.drop_duplicates())
    if len(source) != 104 or len(schemes) != 8 or set(source.ses_variable) != set(SES):
        raise RuntimeError("Unexpected V211 inequality source structure")
    matrix = source.pivot(index="ses_variable", columns="scheme",
                          values="standardized_prediction_gap").reindex(index=SES, columns=schemes)
    values = matrix.to_numpy(float)
    limit = max(float(np.nanquantile(np.abs(values), .98)), 1e-9)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list(
        "ses_red_blue", [BLUE, "#8DB0D5", "#FFFFFF", "#E6A6A1", RED]
    )
    codes = [scheme_code(value) for value in schemes]

    # One aligned hero row: the heatmap supplies the sole SES row labels, so
    # the robustness summary cannot collide with its independent colour bar.
    fig = plt.figure(figsize=(183 / 25.4, 204 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        4, 1, left=.115, right=.965, bottom=.055, top=.965,
        height_ratios=[1.40, 1, 1, 1], hspace=.36,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.43, 1], wspace=.31)
    ax_heat = fig.add_subplot(top[0, 0])
    ax_robust = fig.add_subplot(top[0, 1], sharey=ax_heat)

    image = ax_heat.imshow(values, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
    ax_heat.set_xticks(range(8), codes, rotation=43, ha="right")
    ax_heat.set_yticks(range(13), LABELS)
    ax_heat.tick_params(length=0, labelsize=6.3, pad=2.1)
    panel(ax_heat, 0, "Prediction gaps by scheme", x=-.055)
    cax = ax_heat.inset_axes([1.035, 0, .026, 1.0])
    colourbar = fig.colorbar(image, cax=cax)
    colourbar.set_ticks([-limit, 0, limit])
    colourbar.set_ticklabels([f"−{limit:.1f}", "0", f"+{limit:.1f}"])
    colourbar.ax.tick_params(labelsize=6.0, length=1.5, pad=1.0)
    colourbar.outline.set_linewidth(.4)

    summary = []
    for yi, variable in enumerate(SES):
        row = matrix.loc[variable].to_numpy(float)
        low, median, high = float(np.min(row)), float(np.median(row)), float(np.max(row))
        colour = cmap(norm(median))
        ax_robust.hlines(yi, low, high, color=colour, lw=1.6, alpha=.75)
        ax_robust.scatter(row, np.full(8, yi), s=7, color=colour, alpha=.35,
                          edgecolor="none")
        ax_robust.scatter(median, yi, s=23, color=colour, edgecolor="white",
                          linewidth=.4, zorder=3)
        summary.append({"ses_variable": variable, "median_across_schemes": median,
                        "minimum": low, "maximum": high})
    ax_robust.axvline(0, color="#707987", lw=.65)
    ax_robust.set_xlim(-limit * 1.08, limit * 1.08)
    ax_robust.tick_params(axis="y", left=False, labelleft=False)
    ax_robust.grid(axis="x", color="#E4E9F0", linewidth=.4)
    ax_robust.set_axisbelow(True)
    ax_robust.set_xlabel("Standardized gap")
    panel(ax_robust, 1, "Across-scheme range", x=-.05)

    # Keep all 13 dimension panels while using repeated geometry and one
    # visible scheme-key column per row instead of 13 redundant key columns.
    rows = [SES[:4], SES[4:9], SES[9:]]
    letter = 2
    for row_index, variables in enumerate(rows, 1):
        grid = outer[row_index].subgridspec(1, len(variables), wspace=.32)
        for col_index, variable in enumerate(variables):
            ax = fig.add_subplot(grid[0, col_index])
            row = source.loc[source.ses_variable.eq(variable)].set_index("scheme").reindex(schemes)
            dots = row.standardized_prediction_gap.to_numpy(float)
            y = np.arange(8)
            ax.axvline(0, color="#707987", lw=.6)
            ax.hlines(y, 0, dots, color="#CAD4E0", lw=1.0)
            ax.scatter(dots, y, s=20, c=[cmap(norm(v)) for v in dots],
                       edgecolor="#6C7480", linewidth=.2, zorder=3)
            ax.set_xlim(-limit * 1.08, limit * 1.08)
            ax.set_ylim(7.4, -.4)
            ax.set_yticks(y, codes if col_index == 0 else [""] * 8)
            ax.tick_params(axis="y", length=0, labelsize=5.9, pad=1)
            ax.tick_params(axis="x", labelsize=6.0)
            ax.grid(axis="x", color="#E4E9F0", linewidth=.4)
            ax.set_axisbelow(True)
            panel(ax, letter, LABELS[SES.index(variable)], x=-.12)
            letter += 1
            if row_index == 3:
                ax.set_xlabel("Standardized gap")

    delivery = save_jpg_svg(fig, OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    source.to_csv(OUT / "source_all_scheme_ses_gaps.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "source_cross_scheme_summary.csv", index=False)
    after = sha256(SOURCE)
    manifest = {
        "figure": OUT.name, "panels": 15, "rows": len(source),
        "ses_dimensions": 13, "schemes": 8,
        "source_sha256_before": before, "source_sha256_after": after,
        "proxy_analysis": True, "formal": False,
        "effect": "standardized Q3-minus-Q1 area-level prediction gap",
        "interpretation": "descriptive model prediction, not individual disparity or causal effect",
        "delivery": delivery,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after or not delivery["svg_text_editable"] or min(delivery["jpg_dpi_metadata"]) < 599:
        raise RuntimeError("Refined inequality delivery validation failed")
    print(json.dumps({"output": str(OUT), "panels": 15, "source_unchanged": before == after}))


if __name__ == "__main__":
    main()
