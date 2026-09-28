"""Non-destructive v5 alignment and typography refinement for V211 figures.

This renderer keeps every earlier export untouched.  It emits JPG and SVG
only, uses one shared continuous scale for the spatial prediction matrix, and
shortens the map colour bars without changing any source value.
"""
from __future__ import annotations

import json
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
from PIL import Image

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_nature_aligned_composites_v4 as v4  # noqa: E402
import build_v211_main_editorial_v2 as editorial  # noqa: E402
from nature_viz_common import (  # noqa: E402
    ROOT,
    DIV,
    MISSING,
    INK,
    MUTED,
    DIMENSION_LABELS,
    setup_style,
    clean_axes,
    signed_magnitude_quantile_scale,
    symmetric_colourbar_ticks,
    sha256,
)

OUT_ROOT = ROOT / "figures" / "v211_nature_aligned_v5"
BLUE = "#3B4CC0"
BLUE_MID = "#7699C6"
BLUE_SOFT = "#B9CDE0"
RED = "#B40426"
RED_SOFT = "#EDB5AE"
NEUTRAL = "#727A84"
_V4_ALIGNMENT_METRICS = v4._alignment_metrics


def save_jpg_svg(fig: plt.Figure, out: Path, stem: str) -> dict:
    """Save only the two formats requested for the refined figure set."""
    out.mkdir(parents=True, exist_ok=True)
    jpg = out / f"{stem}.jpg"
    svg = out / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(jpg, dpi=600, bbox_inches="tight", pil_kwargs={"quality": 95})
    with Image.open(jpg) as image:
        dpi = image.info.get("dpi", (0, 0))
        size = image.size
    svg_text = svg.read_text(encoding="utf-8", errors="ignore")
    return {
        "jpg": str(jpg),
        "svg": str(svg),
        "delivery_formats": ["jpg", "svg"],
        "jpg_dpi_metadata": [float(dpi[0]), float(dpi[1])],
        "jpg_pixel_size": [int(size[0]), int(size[1])],
        "svg_text_editable": "<text" in svg_text,
    }


def map_cell(
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
    letter_inside: bool = False,
) -> tuple[plt.Axes, plt.Axes]:
    """Draw an aligned map with a centred 70%-height colour bar."""
    cell = spec.subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.080)
    ax = fig.add_subplot(cell[0, 0])
    slot = fig.add_subplot(cell[0, 1])
    slot.set_axis_off()
    cax = slot.inset_axes([0.0, 0.15, 1.0, 0.70])
    frame.plot(
        column=value,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="#F4F5F6",
        linewidth=0.010,
        missing_kwds={"color": MISSING, "edgecolor": "#F4F5F6", "linewidth": 0.010},
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="center", pad=3.2, fontsize=6.8, fontweight="normal")
    if letter_inside:
        ax.text(.018, .985, chr(ord("a") + letter), transform=ax.transAxes, ha="left", va="top",
                fontsize=8.2, fontweight="bold", color=INK, clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .72, "pad": .25})
    else:
        v4._panel_letter(ax, letter)
    if show_n:
        n = int(frame[value].notna().sum())
        ax.text(0.985, 0.010, f"n={n:,}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.9, color=MUTED, fontweight="normal")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{tick:.{decimals}f}" for tick in ticks])
    cb.ax.tick_params(labelsize=4.8, length=1.0, pad=0.7)
    cb.outline.set_linewidth(0.30)
    return ax, cax


def finish(
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
    delivery = save_jpg_svg(fig, out, stem)
    plt.close(fig)
    cbar_fraction = alignment.get("mean_colourbar_to_map_height", np.nan)
    payload = {
        "figure": stem,
        "panels": panels,
        "analysis_version": "V211",
        "layout_version": "aligned_v5_non_destructive",
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
            "jpg_svg_only": delivery["delivery_formats"] == ["jpg", "svg"],
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
            "equal_map_sizes": alignment["equal_map_sizes"],
            "equal_colourbar_heights": alignment["equal_colourbar_heights"],
            "rows_aligned": alignment["rows_aligned"],
            "columns_aligned": alignment["columns_aligned"],
            "short_centred_colourbars": bool(0.66 <= cbar_fraction <= 0.74),
            "only_panel_letters_bold_by_construction": True,
            "proxy_only": True,
            "not_formal": True,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "figure_contract.md").write_text(
        "# Figure contract\n\n"
        f"- Core conclusion: {conclusion}\n"
        f"- Evidence chain: {evidence_chain}\n"
        "- Typography: only subplot letters are bold.\n"
        "- Map rendering: softened 0.010-pt tract outlines and centred 70%-height colour bars.\n"
        "- Export: JPG (600 dpi) and editable-text SVG only.\n"
        "- Preservation: this is a new v5 export; every earlier version remains untouched.\n",
        encoding="utf-8",
    )
    return out / f"{stem}.jpg"


def alignment_metrics(map_axes: list[plt.Axes], cbar_axes: list[plt.Axes], nrows: int, ncols: int) -> dict:
    result = _V4_ALIGNMENT_METRICS(map_axes, cbar_axes, nrows, ncols)
    map_boxes = [ax.get_position() for ax in map_axes]
    cbar_boxes = [ax.get_position() for ax in cbar_axes]
    ratios = np.array([c.height / m.height for m, c in zip(map_boxes, cbar_boxes)])
    result["mean_colourbar_to_map_height"] = float(ratios.mean())
    result["colourbar_to_map_height_range"] = float(np.ptp(ratios))
    return result


def centre_colourbars_on_maps(
    map_axes: list[plt.Axes],
    cbar_axes: list[plt.Axes],
    fraction: float = .70,
    final_inset: float = 0.0,
    terminal_every: int | None = None,
) -> None:
    """Re-anchor bars after equal-aspect map axes have reached final size."""
    for index, (map_ax, cbar_ax) in enumerate(zip(map_axes, cbar_axes)):
        map_box = map_ax.get_position()
        cbar_box = cbar_ax.get_position()
        height = map_box.height * fraction
        y0 = map_box.y0 + (map_box.height - height) / 2
        cbar_ax.set_axes_locator(None)
        is_terminal = (
            (terminal_every is not None and (index + 1) % terminal_every == 0)
            or (terminal_every is None and index == len(cbar_axes) - 1)
        )
        x0 = cbar_box.x0 - (final_inset if is_terminal else 0.0)
        cbar_ax.set_position([x0, y0, cbar_box.width, height])


def build_fig2() -> Path:
    """Fifteen maps with an accurate OLS-anchor label and one shared scale."""
    setup_style(6.5)
    source = ROOT / "figures/current_spatial_model_matrix_25panel/source_spatial_model_predictions.csv"
    data = pd.read_csv(source, dtype={"GEOID": str})
    data["GEOID"] = data.GEOID.str.zfill(11)
    data["model"] = data.model.replace({"OLS/Ridge": "OLS anchor"})
    data = data.loc[data.model.ne("GTCNNWR")].copy()
    years = [2014, 2019, 2023]
    models = ["MM-GTGNNWR", "OLS anchor", "GTWR", "GTNNWR", "GTGNNWR"]
    geos = {year: editorial._geo(year) for year in years}
    frames: dict[tuple[str, int], gpd.GeoDataFrame] = {}
    for model in models:
        for year in years:
            sub = data.loc[data.model.eq(model) & data.health_year.eq(year), ["GEOID", "prediction"]]
            frames[(model, year)] = geos[year].merge(sub, on="GEOID", how="left", validate="one_to_one")

    values = np.concatenate([frame.prediction.dropna().to_numpy(float) for frame in frames.values()])
    lo, hi = np.quantile(values, [.005, .995])
    centre = float(np.median(values))
    norm = TwoSlopeNorm(vmin=float(lo), vcenter=centre, vmax=float(hi))
    ticks = [float(lo), centre, float(hi)]
    extent = editorial._extent(list(geos.values()))

    fig = plt.figure(figsize=(183 / 25.4, 160 / 25.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=.060, right=.952, bottom=.095, top=.935,
                             height_ratios=[3.80, 1.0], hspace=.18)
    maps_grid = outer[0].subgridspec(3, 5, wspace=.105, hspace=.105)
    support = outer[1].subgridspec(1, 3, wspace=.33)
    map_axes: list[plt.Axes] = []
    cbar_axes: list[plt.Axes] = []
    letter = 0
    for row, year in enumerate(years):
        for col, model in enumerate(models):
            ax, cax = map_cell(fig, maps_grid[row, col], frames[(model, year)], "prediction",
                                norm, DIV, extent, letter, model if row == 0 else "", ticks, 1)
            map_axes.append(ax); cbar_axes.append(cax); letter += 1

    fig.canvas.draw()
    for row, year in enumerate(years):
        boxes = [map_axes[row * len(models) + col].get_position() for col in range(len(models))]
        y = float(np.mean([box.y0 + box.height / 2 for box in boxes]))
        fig.text(.022, y, str(year), rotation=90, ha="center", va="center", fontsize=6.8,
                 fontweight="normal", color=INK)

    grouped = data.loc[data.model.isin(models)].groupby(["model", "health_year"]).prediction
    summary = grouped.agg(median="median", q25=lambda x: x.quantile(.25),
                          q75=lambda x: x.quantile(.75)).reset_index()
    summary["iqr"] = summary.q75 - summary.q25
    colours = {"MM-GTGNNWR": RED, "OLS anchor": NEUTRAL, "GTWR": BLUE_MID,
               "GTNNWR": BLUE_SOFT, "GTGNNWR": RED_SOFT}

    ax = fig.add_subplot(support[0, 0])
    for model in models:
        sub = summary.loc[summary.model.eq(model)].sort_values("health_year")
        ax.plot(sub.health_year, sub["median"], color=colours[model], lw=1.3, marker="o", ms=2.7)
    ax.set_xticks([2014, 2019, 2023]); ax.set_xlabel("Health-reference year")
    ax.set_ylabel("Median prediction (%)"); clean_axes(ax, "both")
    ax.set_title("Temporal level", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, letter, x=-.12, y=1.02); letter += 1

    ax = fig.add_subplot(support[0, 1])
    for model in models:
        sub = summary.loc[summary.model.eq(model)].sort_values("health_year")
        ax.plot(sub.health_year, sub.iqr, color=colours[model], lw=1.3, marker="o", ms=2.7)
    ax.set_xticks([2014, 2019, 2023]); ax.set_xlabel("Health-reference year")
    ax.set_ylabel("Spatial IQR (pp)"); clean_axes(ax, "both")
    ax.set_title("Spatial dispersion", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, letter, x=-.12, y=1.02); letter += 1

    ax = fig.add_subplot(support[0, 2])
    arrays = [data.loc[data.model.eq(model) & data.health_year.eq(2023), "prediction"].dropna().to_numpy(float)
              for model in models]
    positions = np.arange(len(models))
    bp = ax.boxplot(arrays, orientation="horizontal", positions=positions, widths=.58,
                    showfliers=False, patch_artist=True,
                    medianprops={"color": INK, "linewidth": .8},
                    whiskerprops={"color": NEUTRAL, "linewidth": .7},
                    capprops={"color": NEUTRAL, "linewidth": .7})
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(colours[model]); patch.set_alpha(.80); patch.set_edgecolor("white")
    ax.set_yticks(positions, models); ax.invert_yaxis(); ax.set_xlabel("2023 prediction (%)")
    clean_axes(ax, "x"); ax.set_title("Final-year distributions", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, letter, x=-.12, y=1.02)

    fig.legend([Line2D([0], [0], color=colours[model], lw=2.2) for model in models], models,
               loc="lower center", bbox_to_anchor=(.53, .017), ncol=5, frameon=False,
               fontsize=5.5, handlelength=1.35, columnspacing=.95)
    fig.canvas.draw()
    # The fifth-column tick labels extend beyond the colourbar axes; inset the
    # terminal bar so the labels remain inside the exported canvas.
    centre_colourbars_on_maps(map_axes, cbar_axes, final_inset=0.022, terminal_every=5)
    fig.canvas.draw()
    alignment = alignment_metrics(map_axes, cbar_axes, 3, 5)
    out = finish(
        fig, "fig2_spatial_prediction", "Fig2_spatial_prediction_aligned_v5", 18, [source],
        "MM-GTGNNWR and four registered comparators are compared on the same aligned temporal-spatial grid.",
        "Fifteen equal-size maps (3 years × 5 models) → temporal level → spatial dispersion → final-year distributions.",
        alignment,
    )
    summary.to_csv(out.parent / "source_temporal_spatial_summaries.csv", index=False)
    return out


def build_fig8() -> Path:
    """Five aligned coefficient maps plus four coloured diagnostics."""
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
            on="GEOID", how="left", validate="one_to_one",
        )
        for dimension in dimensions
    }
    extent = editorial._extent(list(frames.values()))

    fig = plt.figure(figsize=(183 / 25.4, 115 / 25.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=0.092, right=0.925, bottom=0.105, top=0.930,
                             height_ratios=[1.92, 1.0], hspace=0.25)
    maps_grid = outer[0].subgridspec(1, 5, wspace=0.105)
    summary_grid = outer[1].subgridspec(1, 4, wspace=0.42, width_ratios=[1.12, 1.0, 1.0, 1.18])
    map_axes: list[plt.Axes] = []
    cbar_axes: list[plt.Axes] = []
    for index, dimension in enumerate(dimensions):
        frame = frames[dimension]
        cmap, norm, boundaries = signed_magnitude_quantile_scale(frame.local_coefficient.to_numpy(float))
        ax, cax = map_cell(
            fig, maps_grid[0, index], frame, "local_coefficient", norm, cmap, extent, index,
            dimension, symmetric_colourbar_ticks(boundaries[0], boundaries[-1]), decimals=4,
            letter_inside=True,
        )
        map_axes.append(ax); cbar_axes.append(cax)

    schemes = list(means.scheme.drop_duplicates())
    markers = ["o", "s", "^", "D"]
    offsets = np.linspace(-0.17, 0.17, len(schemes))
    matrix = means.pivot(index="dimension", columns="scheme", values="mean").reindex(dimensions)
    yy = np.arange(len(dimensions))

    ax = fig.add_subplot(summary_grid[0, 0])
    for y, (dimension, colour) in enumerate(zip(dimensions, dim_colours)):
        row = matrix.loc[dimension, schemes].to_numpy(float)
        ax.hlines(y, float(np.min(row)), float(np.max(row)), color=colour, lw=1.35, alpha=0.72)
        for scheme, marker, offset, value in zip(schemes, markers, offsets, row):
            ax.scatter(value, y + offset, marker=marker, s=18, facecolor=colour,
                       edgecolor="white", linewidth=0.45, alpha=0.96)
    ax.axvline(0, color=NEUTRAL, lw=0.65)
    ax.set_yticks(yy, ["Green / blue", "Access", "Clean", "Air pollution", "Traffic"])
    ax.invert_yaxis(); ax.set_xlabel("Mean coefficient")
    clean_axes(ax, "x"); ax.set_title("Partition means", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, 5, x=-0.20, y=1.02)

    ax = fig.add_subplot(summary_grid[0, 1])
    vals = stability.set_index("dimension").reindex(dimensions).corr_22_23.to_numpy(float)
    ax.hlines(yy, 0.985, vals, color=dim_colours, lw=1.7)
    ax.scatter(vals, yy, s=27, color=dim_colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy, []); ax.invert_yaxis(); ax.set_xlim(0.985, 1.0005)
    ax.set_xlabel("Tract correlation"); clean_axes(ax, "x")
    ax.set_title("2022–2023 stability", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, 6, x=-0.12, y=1.02)

    ax = fig.add_subplot(summary_grid[0, 2])
    spread_by_scheme = matrix.max(axis=1) - matrix.min(axis=1)
    ax.hlines(yy, 0, spread_by_scheme.to_numpy(float), color=dim_colours, lw=1.8)
    ax.scatter(spread_by_scheme.to_numpy(float), yy, s=29, color=dim_colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy, []); ax.invert_yaxis(); ax.set_xlabel("Range of scheme means")
    clean_axes(ax, "x"); ax.set_title("Cross-scheme spread", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, 7, x=-0.12, y=1.02)

    ax = fig.add_subplot(summary_grid[0, 3])
    for dimension, colour in zip(dimensions, dim_colours):
        sub = spread.loc[spread.dimension.eq(dimension)]
        ax.plot(sub.year, sub.sd, marker="o", ms=2.4, lw=1.15, color=colour)
    ax.set_xlabel("Health-reference year"); ax.set_ylabel("SD")
    ax.set_xticks([2014, 2017, 2020, 2023]); clean_axes(ax, "both")
    ax.set_title("Spatial spread", loc="left", pad=2.5, fontweight="normal")
    v4._panel_letter(ax, 8, x=-0.16, y=1.02)

    fig.text(0.52, 0.975,
             "Signed subjective−objective coefficient   ·   blue = lower platform-expression coefficient   ·   red = higher",
             ha="center", va="top", fontsize=5.5, color=MUTED, fontweight="normal")
    fig.canvas.draw()
    # Pull the terminal colourbar inward so its right-hand tick labels remain
    # inside the journal-width canvas without shrinking the map itself.
    centre_colourbars_on_maps(map_axes, cbar_axes, final_inset=0.022, terminal_every=5)
    fig.canvas.draw()
    alignment = alignment_metrics(map_axes, cbar_axes, 1, 5)
    out = finish(
        fig, "fig8_coefficient_heterogeneity", "Fig8_coefficient_heterogeneity_aligned_v5", 9,
        [spread_path, means_path, stability_path, maps_path],
        "Five signed coefficient surfaces remain spatially structured and stable across time and registered partitions.",
        "Five equal-size coefficient maps → coloured partition means → temporal stability → cross-scheme spread → temporal spatial spread.",
        alignment,
    )
    pooled.to_csv(out.parent / "source_pooled_2023_maps.csv", index=False)
    spread_by_scheme.rename("range_of_scheme_means").reset_index().to_csv(
        out.parent / "source_cross_scheme_spread.csv", index=False)
    return out


def main() -> None:
    outputs = [build_fig2(), build_fig8()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layout_version": "aligned_v5_non_destructive",
        "preserves_existing_versions": True,
        "delivery_formats": ["jpg", "svg"],
        "outputs": [str(path) for path in outputs],
    }
    (OUT_ROOT / "aligned_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
