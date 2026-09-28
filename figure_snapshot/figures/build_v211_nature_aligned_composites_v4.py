"""Build new, non-destructive aligned V211 Nature-family composites.

This renderer intentionally writes to ``v211_nature_aligned_v4`` and never
overwrites the v3 source script or its exports.  The scientific values are
read from the registered V211 source tables without refitting any model.
"""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_main_editorial_v2 as editorial  # noqa: E402
from nature_viz_common import (  # noqa: E402
    ROOT,
    DIV,
    MISSING,
    INK,
    GRID,
    MUTED,
    DIMENSION_LABELS,
    setup_style,
    clean_axes,
    save_delivery,
    write_validation,
    signed_magnitude_quantile_scale,
    symmetric_colourbar_ticks,
    sha256,
)

OUT_ROOT = ROOT / "figures" / "v211_nature_aligned_v4"
BLUE = "#3B4CC0"
BLUE_MID = "#7699C6"
BLUE_SOFT = "#B9CDE0"
RED = "#B40426"
RED_MID = "#D46F6B"
RED_SOFT = "#EDB5AE"
NEUTRAL = "#727A84"


def _panel_letter(ax: plt.Axes, index: int, x: float = -0.05, y: float = 1.015) -> None:
    ax.text(
        x,
        y,
        string.ascii_lowercase[index],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def _map_cell(
    fig: plt.Figure,
    spec,
    frame: gpd.GeoDataFrame,
    value: str,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    extent: tuple[float, float, float, float],
    letter: int,
    title: str,
    ticks: list[float],
    decimals: int,
    show_n: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    """Draw one map and one explicitly allocated equal-height colour bar."""
    cell = spec.subgridspec(1, 2, width_ratios=[1.0, 0.060], wspace=0.025)
    ax = fig.add_subplot(cell[0, 0])
    cax = fig.add_subplot(cell[0, 1])
    frame.plot(
        column=value,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.028,
        missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": 0.024},
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="center", pad=4.0, fontsize=6.8, fontweight="bold" if title == "MM-GTGNNWR" else "normal")
    _panel_letter(ax, letter)
    if show_n:
        n = int(frame[value].notna().sum())
        ax.text(0.985, 0.010, f"n={n:,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.9, color=MUTED)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{tick:.{decimals}f}" for tick in ticks])
    cb.ax.tick_params(labelsize=4.8, length=1.0, pad=0.7)
    cb.outline.set_linewidth(0.35)
    labels = cb.ax.get_yticklabels()
    if labels:
        labels[0].set_va("bottom")
        labels[-1].set_va("top")
    return ax, cax


def _alignment_metrics(map_axes: list[plt.Axes], cbar_axes: list[plt.Axes], nrows: int, ncols: int) -> dict:
    """Numerically verify equal map/cbar geometry and row/column registration."""
    map_boxes = [ax.get_position() for ax in map_axes]
    cbar_boxes = [ax.get_position() for ax in cbar_axes]
    widths = np.array([box.width for box in map_boxes])
    heights = np.array([box.height for box in map_boxes])
    cbar_heights = np.array([box.height for box in cbar_boxes])
    row_centres = np.array([[map_boxes[r * ncols + c].y0 + map_boxes[r * ncols + c].height / 2 for c in range(ncols)] for r in range(nrows)])
    col_centres = np.array([[map_boxes[r * ncols + c].x0 + map_boxes[r * ncols + c].width / 2 for r in range(nrows)] for c in range(ncols)])
    return {
        "map_width_range": float(widths.max() - widths.min()),
        "map_height_range": float(heights.max() - heights.min()),
        "colourbar_height_range": float(cbar_heights.max() - cbar_heights.min()),
        "max_within_row_centre_deviation": float(np.max(np.ptp(row_centres, axis=1))),
        "max_within_column_centre_deviation": float(np.max(np.ptp(col_centres, axis=1))),
        "equal_map_sizes": bool(np.ptp(widths) < 1e-6 and np.ptp(heights) < 1e-6),
        "equal_colourbar_heights": bool(np.ptp(cbar_heights) < 1e-6),
        "rows_aligned": bool(np.max(np.ptp(row_centres, axis=1)) < 1e-6),
        "columns_aligned": bool(np.max(np.ptp(col_centres, axis=1)) < 1e-6),
    }


def _finish(
    fig: plt.Figure,
    folder: str,
    stem: str,
    panels: int,
    sources: list[Path],
    conclusion: str,
    evidence_chain: str,
    alignment: dict,
) -> Path:
    out = OUT_ROOT / folder
    delivery = save_delivery(fig, out, stem, dpi=600)
    plt.close(fig)
    payload = {
        "figure": stem,
        "panels": panels,
        "analysis_version": "V211",
        "layout_version": "aligned_v4_non_destructive",
        "proxy_analysis": True,
        "formal": False,
        "core_conclusion": conclusion,
        "evidence_chain": evidence_chain,
        "exclusions": ["GTCNNWR"],
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "alignment": alignment,
        "delivery": delivery,
        "checks": {
            "source_files_exist": all(path.exists() for path in sources),
            "five_delivery_formats": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
            "svg_text_editable": delivery["svg_text_editable"],
            "png_or_jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
            "equal_map_sizes": alignment["equal_map_sizes"],
            "equal_colourbar_heights": alignment["equal_colourbar_heights"],
            "rows_aligned": alignment["rows_aligned"],
            "columns_aligned": alignment["columns_aligned"],
            "proxy_only": True,
            "not_formal": True,
        },
    }
    write_validation(out, payload)
    (out / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "figure_contract.md").write_text(
        "# Figure contract\n\n"
        f"- Core conclusion: {conclusion}\n"
        f"- Evidence chain: {evidence_chain}\n"
        "- Layout: all map panels use explicit map and colour-bar grid cells; no automatic colour-bar placement.\n"
        "- Preservation: this is a new v4 export; all earlier scripts and exports remain untouched.\n"
        "- Scientific status: V211 compact proxy (`proxy_analysis=true`, `formal=false`).\n",
        encoding="utf-8",
    )
    return out / f"{stem}.png"


def build_fig2() -> Path:
    """Build 15 equal-size maps in a 3-year × 5-model plate plus 3 summaries."""
    setup_style(6.5)
    source = ROOT / "figures/current_spatial_model_matrix_25panel/source_spatial_model_predictions.csv"
    data = pd.read_csv(source, dtype={"GEOID": str})
    data["GEOID"] = data.GEOID.str.zfill(11)
    data = data.loc[data.model.ne("GTCNNWR")].copy()
    years = [2014, 2019, 2023]
    models = ["MM-GTGNNWR", "OLS/Ridge", "GTWR", "GTNNWR", "GTGNNWR"]
    all_years = sorted(data.health_year.dropna().astype(int).unique())
    geos = {year: editorial._geo(year) for year in years}
    frames: dict[tuple[str, int], gpd.GeoDataFrame] = {}
    for model in models:
        for year in years:
            sub = data.loc[data.model.eq(model) & data.health_year.eq(year), ["GEOID", "prediction"]]
            frames[(model, year)] = geos[year].merge(sub, on="GEOID", how="left", validate="one_to_one")

    values = np.concatenate([frame.prediction.dropna().to_numpy(float) for frame in frames.values()])
    lo, hi = np.quantile(values, [0.005, 0.995])
    centre = float(np.median(values))
    norm = TwoSlopeNorm(vmin=float(lo), vcenter=centre, vmax=float(hi))
    ticks = [float(lo), centre, float(hi)]
    extent = editorial._extent(list(geos.values()))

    fig = plt.figure(figsize=(183 / 25.4, 158 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.060,
        right=0.952,
        bottom=0.095,
        top=0.935,
        height_ratios=[3.75, 1.0],
        hspace=0.18,
    )
    maps_grid = outer[0].subgridspec(3, 5, wspace=0.105, hspace=0.105)
    support = outer[1].subgridspec(1, 3, wspace=0.33)
    map_axes: list[plt.Axes] = []
    cbar_axes: list[plt.Axes] = []
    letter = 0
    for row, year in enumerate(years):
        for col, model in enumerate(models):
            ax, cax = _map_cell(
                fig,
                maps_grid[row, col],
                frames[(model, year)],
                "prediction",
                norm,
                DIV,
                extent,
                letter,
                model if row == 0 else "",
                ticks,
                decimals=1,
            )
            map_axes.append(ax)
            cbar_axes.append(cax)
            letter += 1

    fig.canvas.draw()
    for row, year in enumerate(years):
        boxes = [map_axes[row * len(models) + col].get_position() for col in range(len(models))]
        y = float(np.mean([box.y0 + box.height / 2 for box in boxes]))
        fig.text(0.022, y, str(year), rotation=90, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK)

    grouped = data.loc[data.model.isin(models) & data.health_year.notna()].groupby(["model", "health_year"]).prediction
    summary = grouped.agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
    summary["iqr"] = summary.q75 - summary.q25
    colours = {
        "MM-GTGNNWR": RED,
        "OLS/Ridge": NEUTRAL,
        "GTWR": BLUE_MID,
        "GTNNWR": BLUE_SOFT,
        "GTGNNWR": RED_SOFT,
    }

    ax = fig.add_subplot(support[0, 0])
    for model in models:
        sub = summary.loc[summary.model.eq(model)].sort_values("health_year")
        ax.plot(sub.health_year, sub["median"], color=colours[model], lw=1.3, marker="o", ms=2.7)
    ax.set_xticks([2014, 2019, 2023])
    ax.set_xlabel("Health-reference year")
    ax.set_ylabel("Median prediction (%)")
    clean_axes(ax, "both")
    ax.set_title("Temporal level", loc="left", pad=2.5)
    _panel_letter(ax, letter, x=-0.12, y=1.02)
    letter += 1

    ax = fig.add_subplot(support[0, 1])
    for model in models:
        sub = summary.loc[summary.model.eq(model)].sort_values("health_year")
        ax.plot(sub.health_year, sub.iqr, color=colours[model], lw=1.3, marker="o", ms=2.7)
    ax.set_xticks([2014, 2019, 2023])
    ax.set_xlabel("Health-reference year")
    ax.set_ylabel("Spatial IQR (pp)")
    clean_axes(ax, "both")
    ax.set_title("Spatial dispersion", loc="left", pad=2.5)
    _panel_letter(ax, letter, x=-0.12, y=1.02)
    letter += 1

    ax = fig.add_subplot(support[0, 2])
    positions = np.arange(len(models))
    arrays = [data.loc[data.model.eq(model) & data.health_year.eq(2023), "prediction"].dropna().to_numpy(float) for model in models]
    bp = ax.boxplot(
        arrays,
        orientation="horizontal",
        positions=positions,
        widths=0.58,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 0.8},
        whiskerprops={"color": NEUTRAL, "linewidth": 0.7},
        capprops={"color": NEUTRAL, "linewidth": 0.7},
    )
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(colours[model])
        patch.set_alpha(0.80)
        patch.set_edgecolor("white")
    ax.set_yticks(positions, models)
    ax.invert_yaxis()
    ax.set_xlabel("2023 prediction (%)")
    clean_axes(ax, "x")
    ax.set_title("Final-year distributions", loc="left", pad=2.5)
    _panel_letter(ax, letter, x=-0.12, y=1.02)

    fig.legend(
        [Line2D([0], [0], color=colours[model], lw=2.2) for model in models],
        models,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.017),
        ncol=5,
        frameon=False,
        fontsize=5.5,
        handlelength=1.35,
        columnspacing=0.95,
    )
    fig.canvas.draw()
    alignment = _alignment_metrics(map_axes, cbar_axes, 3, 5)
    out = _finish(
        fig,
        "fig2_spatial_prediction",
        "Fig2_spatial_prediction_aligned_v4",
        18,
        [source],
        "MM-GTGNNWR and four registered comparators are compared on the same aligned temporal-spatial grid.",
        "Fifteen equal-size maps (3 years × 5 models) → temporal level → spatial dispersion → final-year distributions.",
        alignment,
    )
    summary.to_csv(out.parent / "source_temporal_spatial_summaries.csv", index=False)
    return out


def build_fig8() -> Path:
    """Build five coefficient maps on one baseline plus four aligned diagnostics."""
    setup_style(6.5)
    base = ROOT / "figures/v211_coefficient_heterogeneity_flagship_25panel"
    spread_path = base / "source_pooled_spatial_spread.csv"
    means_path = base / "source_scheme_means_2023.csv"
    stability_path = base / "source_2022_2023_stability.csv"
    maps_path = base / "source_data_2023_scheme_specific_s_minus_o.csv"
    spread = pd.read_csv(spread_path)
    means = pd.read_csv(means_path)
    stability = pd.read_csv(stability_path)
    maps = pd.read_csv(maps_path, dtype={"GEOID": str})
    maps["GEOID"] = maps.GEOID.str.zfill(11)
    dimensions = [DIMENSION_LABELS[index] for index in (1, 3, 5, 8, 10)]
    dim_colours = [BLUE, BLUE_MID, BLUE_SOFT, RED_SOFT, RED]
    geo = editorial._geo(2023)
    pooled = maps.groupby(["GEOID", "dimension"], as_index=False).local_coefficient.mean()
    frames = {
        dimension: geo.merge(
            pooled.loc[pooled.dimension.eq(dimension), ["GEOID", "local_coefficient"]],
            on="GEOID",
            how="left",
            validate="one_to_one",
        )
        for dimension in dimensions
    }
    extent = editorial._extent(list(frames.values()))

    fig = plt.figure(figsize=(183 / 25.4, 112 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.092,
        right=0.950,
        bottom=0.110,
        top=0.940,
        height_ratios=[1.86, 1.0],
        hspace=0.24,
    )
    maps_grid = outer[0].subgridspec(1, 5, wspace=0.105)
    summary_grid = outer[1].subgridspec(1, 4, wspace=0.42, width_ratios=[1.08, 1.00, 1.00, 1.18])
    map_axes: list[plt.Axes] = []
    cbar_axes: list[plt.Axes] = []
    for index, dimension in enumerate(dimensions):
        frame = frames[dimension]
        cmap, norm, boundaries = signed_magnitude_quantile_scale(frame.local_coefficient.to_numpy(float))
        ax, cax = _map_cell(
            fig,
            maps_grid[0, index],
            frame,
            "local_coefficient",
            norm,
            cmap,
            extent,
            index,
            dimension,
            symmetric_colourbar_ticks(boundaries[0], boundaries[-1]),
            decimals=4,
        )
        map_axes.append(ax)
        cbar_axes.append(cax)

    schemes = list(means.scheme.drop_duplicates())
    markers = ["o", "s", "^", "D"]
    offsets = np.linspace(-0.17, 0.17, len(schemes))
    matrix = means.pivot(index="dimension", columns="scheme", values="mean").reindex(dimensions)
    yy = np.arange(len(dimensions))

    ax = fig.add_subplot(summary_grid[0, 0])
    for scheme, marker, offset in zip(schemes, markers, offsets):
        ax.scatter(
            matrix[scheme],
            yy + offset,
            marker=marker,
            s=16,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.7,
        )
    ax.axvline(0, color=NEUTRAL, lw=0.65)
    compact_dimensions = ["Green / blue", "Access", "Clean", "Air pollution", "Traffic"]
    ax.set_yticks(yy, compact_dimensions)
    ax.invert_yaxis()
    ax.set_xlabel("Mean coefficient")
    clean_axes(ax, "x")
    ax.set_title("Partition means", loc="left", pad=2.5)
    _panel_letter(ax, 5, x=-0.20, y=1.02)

    ax = fig.add_subplot(summary_grid[0, 1])
    vals = stability.set_index("dimension").reindex(dimensions).corr_22_23.to_numpy(float)
    ax.hlines(yy, 0.985, vals, color=dim_colours, lw=1.7)
    ax.scatter(vals, yy, s=27, color=dim_colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy, [])
    ax.invert_yaxis()
    ax.set_xlim(0.985, 1.0005)
    ax.set_xlabel("Tract correlation")
    clean_axes(ax, "x")
    ax.set_title("2022–2023 stability", loc="left", pad=2.5)
    _panel_letter(ax, 6, x=-0.12, y=1.02)

    ax = fig.add_subplot(summary_grid[0, 2])
    spread_by_scheme = matrix.max(axis=1) - matrix.min(axis=1)
    ax.hlines(yy, 0, spread_by_scheme.to_numpy(float), color=dim_colours, lw=1.8)
    ax.scatter(spread_by_scheme.to_numpy(float), yy, s=29, color=dim_colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy, [])
    ax.invert_yaxis()
    ax.set_xlabel("Range of scheme means")
    clean_axes(ax, "x")
    ax.set_title("Cross-scheme spread", loc="left", pad=2.5)
    _panel_letter(ax, 7, x=-0.12, y=1.02)

    ax = fig.add_subplot(summary_grid[0, 3])
    for dimension, colour in zip(dimensions, dim_colours):
        sub = spread.loc[spread.dimension.eq(dimension)]
        ax.plot(sub.year, sub.sd, marker="o", ms=2.4, lw=1.15, color=colour)
    ax.set_xlabel("Health-reference year")
    ax.set_ylabel("SD")
    ax.set_xticks([2014, 2017, 2020, 2023])
    clean_axes(ax, "both")
    ax.set_title("Spatial spread", loc="left", pad=2.5)
    _panel_letter(ax, 8, x=-0.16, y=1.02)

    fig.text(
        0.52,
        0.977,
        "Signed subjective−objective coefficient   ·   blue = lower platform-expression coefficient   ·   red = higher",
        ha="center",
        va="top",
        fontsize=5.5,
        color=MUTED,
    )
    fig.canvas.draw()
    alignment = _alignment_metrics(map_axes, cbar_axes, 1, 5)
    out = _finish(
        fig,
        "fig8_coefficient_heterogeneity",
        "Fig8_coefficient_heterogeneity_aligned_v4",
        9,
        [spread_path, means_path, stability_path, maps_path],
        "Five signed coefficient surfaces remain spatially structured and stable across time and registered partitions.",
        "Five equal-size coefficient maps → partition means → temporal stability → cross-scheme spread → temporal spatial spread.",
        alignment,
    )
    pooled.to_csv(out.parent / "source_pooled_2023_maps.csv", index=False)
    spread_by_scheme.rename("range_of_scheme_means").reset_index().to_csv(
        out.parent / "source_cross_scheme_spread.csv", index=False
    )
    return out


def main() -> None:
    outputs = [build_fig2(), build_fig8()]
    manifest = {
        "layout_version": "aligned_v4_non_destructive",
        "preserves_existing_versions": True,
        "outputs": [str(path) for path in outputs],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "aligned_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
