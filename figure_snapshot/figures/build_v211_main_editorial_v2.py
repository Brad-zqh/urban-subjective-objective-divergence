"""Build reader-facing V211 main-text figures at final double-column size.

The original full technical plates remain untouched.  These editorial figures
use only their exported source-data tables (plus the registered annual tract
geometry for maps), reduce redundant panels, and give the primary evidence a
larger visual hierarchy.  No model is re-fit and no statistical value is
changed.
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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

from nature_viz_common import (  # noqa: E402
    ROOT,
    DIV,
    MISSING,
    INK,
    GRID,
    MUTED,
    DIMENSION_LABELS,
    setup_style,
    panel,
    clean_axes,
    save_delivery,
    write_validation,
    signed_magnitude_quantile_scale,
    symmetric_colourbar_ticks,
    add_equal_height_colorbar,
    sha256,
)
import build_v211_signed_delta_sensitivity_20panel as signed_full  # noqa: E402


OUT_ROOT = ROOT / "figures" / "v211_main_editorial_v2"
GEO_ROOT = ROOT / "Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg"
GEO_2022 = GEO_ROOT / "2022/chicago_tiger_2022.gpkg"

BLUE = "#3B4CC0"
BLUE_MID = "#7297C6"
BLUE_SOFT = "#B8CDE0"
RED = "#B40426"
RED_MID = "#D36A67"
RED_SOFT = "#EDB7B0"
NEUTRAL = "#6F7882"


def _geo(year: int) -> gpd.GeoDataFrame:
    geometry_year = {2014: 2014, 2019: 2019, 2023: 2022}.get(year, 2022)
    path = GEO_ROOT / str(geometry_year) / f"chicago_tiger_{geometry_year}.gpkg"
    frame = gpd.read_file(
        path,
        layer="tracts_intersect_chicago",
        columns=["GEOID", "ALAND", "AWATER", "geometry"],
    ).to_crs(26916)
    frame["GEOID"] = frame.GEOID.astype(str).str.zfill(11)
    frame["water_share"] = frame.AWATER / (frame.ALAND + frame.AWATER)
    return frame.loc[frame.ALAND.gt(0) & frame.water_share.le(.5)].copy()


def _extent(frames: list[gpd.GeoDataFrame]) -> tuple[float, float, float, float]:
    bounds = np.vstack([frame.total_bounds for frame in frames])
    xmin, ymin, xmax, ymax = (
        bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()
    )
    dx, dy = xmax - xmin, ymax - ymin
    return xmin - .012 * dx, xmax + .012 * dx, ymin - .012 * dy, ymax + .012 * dy


def _finish(
    fig: plt.Figure,
    out_name: str,
    stem: str,
    panels: int,
    source_paths: list[Path],
    contract: str,
) -> Path:
    out = OUT_ROOT / out_name
    delivery = save_delivery(fig, out, stem, dpi=600)
    plt.close(fig)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    payload = {
        "figure": stem,
        "panels": panels,
        "analysis_version": "V211",
        "proxy_analysis": True,
        "formal": False,
        "editorial_contract": contract,
        "source_hashes": source_hashes,
        "delivery": delivery,
        "checks": {
            "source_files_exist": all(path.exists() for path in source_paths),
            "panel_count_reduced_for_main_text": panels <= 15,
            "five_delivery_formats": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
        },
    }
    write_validation(out, payload)
    (out / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / f"{stem}.png"


def build_fig2() -> Path:
    """All five models at three representative years; full 5-year atlas stays technical."""
    setup_style(6.8)
    source = ROOT / "figures/current_spatial_model_matrix_25panel/source_spatial_model_predictions.csv"
    data = pd.read_csv(source, dtype={"GEOID": str})
    data["GEOID"] = data.GEOID.str.zfill(11)
    years = [2014, 2019, 2023]
    models = ["OLS/Ridge", "GTWR", "GTNNWR", "GTGNNWR", "MM-GTGNNWR"]
    geos = {year: _geo(year) for year in years}
    frames: dict[tuple[str, int], gpd.GeoDataFrame] = {}
    for model in models:
        for year in years:
            sub = data.loc[(data.model.eq(model)) & (data.health_year.eq(year)), ["GEOID", "prediction"]]
            frames[(model, year)] = geos[year].merge(sub, on="GEOID", how="left", validate="one_to_one")
    values = np.concatenate([f.prediction.dropna().to_numpy(float) for f in frames.values()])
    lo, hi = np.quantile(values, [.005, .995])
    centre = float(np.median(values))
    norm = TwoSlopeNorm(vmin=float(lo), vcenter=centre, vmax=float(hi))
    extent = _extent(list(geos.values()))

    fig = plt.figure(figsize=(183 / 25.4, 177 / 25.4), facecolor="white")
    gs = fig.add_gridspec(5, 3, left=.115, right=.955, bottom=.035, top=.970, wspace=.34, hspace=.10)
    index = 0
    for rr, model in enumerate(models):
        for cc, year in enumerate(years):
            ax = fig.add_subplot(gs[rr, cc])
            frame = frames[(model, year)]
            frame.plot(
                column="prediction", ax=ax, cmap=DIV, norm=norm,
                edgecolor="white", linewidth=.032,
                missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .025},
            )
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
            cb, _ = add_equal_height_colorbar(fig, ax, ScalarMappable(norm=norm, cmap=DIV), width=.030, gap=.018)
            ticks = [norm.vmin, norm.vcenter, norm.vmax]
            cb.set_ticks(ticks); cb.set_ticklabels([f"{v:.1f}" for v in ticks])
            cb.ax.tick_params(labelsize=5.7, length=1.1, pad=.5)
            panel(ax, index, "", x=-.025, y=1.015)
            if rr == 0:
                ax.set_title(str(year), fontsize=7.2, fontweight="normal", pad=1.8)
            if cc == 0:
                ax.text(-.12, .50, model, transform=ax.transAxes, ha="right", va="center", fontsize=6.7)
            index += 1
    return _finish(
        fig, "fig2_spatial_prediction", "Fig2_spatial_prediction_editorial_v2", 15, [source],
        "Five retained models by three representative years; every map has its own full-height colour bar on a shared scale.",
    )


def build_fig4() -> Path:
    """Eight panels centred on the paired-fold estimand."""
    setup_style(7.0)
    fold_path = ROOT / "figures/v211_signed_delta_sensitivity_20panel/source_fold_metrics.csv"
    pred_path = ROOT / "figures/v211_signed_delta_sensitivity_20panel/source_outer_test_predictions.csv"
    fold = pd.read_csv(fold_path)
    predictions = pd.read_csv(pred_path)
    fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 102 / 25.4), facecolor="white")
    fig.subplots_adjust(left=.075, right=.965, bottom=.110, top=.940, wspace=.49, hspace=.50)
    metrics = signed_full.METRICS
    for cc, (metric, title, _) in enumerate(metrics):
        signed_full.paired_scatter(axes[0, cc], fold, metric, title, cc)
    # Keep the coverage panel visually subordinate: scheme dots show the
    # composition, while the pooled diamond carries the reader-facing result.
    coverage_rows = []
    ax = axes[0, 3]
    for metric_index, (metric, title, _direction) in enumerate(metrics):
        indicator = f"full_better_{metric}"
        pooled = 100.0 * float(fold[indicator].mean())
        coverage_rows.append({"metric": metric, "scheme": "pooled", "improved_percent": pooled})
        ax.hlines(metric_index, 0, pooled, color="#B8BEC7", linewidth=2.0, zorder=1)
        for offset, scheme in zip(np.linspace(-.15, .15, len(signed_full.SCHEME_ORDER)), signed_full.SCHEME_ORDER):
            share = 100.0 * float(fold.loc[fold.scheme.eq(scheme), indicator].mean())
            coverage_rows.append({"metric": metric, "scheme": scheme, "improved_percent": share})
            ax.scatter(share, metric_index + offset, s=17, color=BLUE_MID,
                       edgecolor="white", linewidth=.35, zorder=2)
        ax.scatter(pooled, metric_index, s=29, marker="D", color=INK,
                   edgecolor="white", linewidth=.45, zorder=3)
        ax.text(min(pooled + 3, 93), metric_index, f"{pooled:.0f}%",
                ha="left", va="center", fontsize=5.0, color=MUTED)
    ax.axvline(50, color=GRID, linewidth=.65, linestyle="--", zorder=0)
    ax.set_yticks(range(len(metrics)), [title.replace("Outer-test ", "") for _, title, _ in metrics])
    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("Folds improved (%)")
    clean_axes(ax, "x"); panel(ax, 3, "Directional coverage")
    coverage = pd.DataFrame(coverage_rows)
    for cc, (metric, _title, direction) in enumerate(metrics):
        signed_full.fold_delta(axes[1, cc], fold, metric, f"Fold Δ{metric.upper()}", direction, 4 + cc)
    signed_full.row_error_shift(axes[1, 3], predictions, 7)
    axes[1, 3].set_title("Annual |error| shift", loc="left")
    out = _finish(
        fig, "fig4_signed_increment", "Fig4_signed_increment_editorial_v2", 8,
        [fold_path, pred_path],
        "Paired fold comparisons are the hero evidence; coverage and annual row-level error shift are validation evidence.",
    )
    coverage.to_csv(out.parent / "source_directional_coverage.csv", index=False)
    return out


def build_fig8() -> Path:
    """Three summary panels plus five enlarged pooled coefficient maps."""
    setup_style(6.9)
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
    dimensions = [DIMENSION_LABELS[d] for d in (1, 3, 5, 8, 10)]
    dim_colors = [BLUE, BLUE_MID, BLUE_SOFT, RED_SOFT, RED]
    geo = _geo(2023)
    pooled = maps.groupby(["GEOID", "dimension"], as_index=False).local_coefficient.mean()
    map_frames = {
        dim: geo.merge(pooled.loc[pooled.dimension.eq(dim), ["GEOID", "local_coefficient"]], on="GEOID", how="left", validate="one_to_one")
        for dim in dimensions
    }
    extent = _extent(list(map_frames.values()))

    fig = plt.figure(figsize=(183 / 25.4, 176 / 25.4), facecolor="white")
    outer = fig.add_gridspec(3, 1, left=.075, right=.955, bottom=.035, top=.955,
                             height_ratios=[1.00, 1.55, 1.55], hspace=.29)
    top = outer[0].subgridspec(1, 3, wspace=.47)
    maps_top = outer[1].subgridspec(1, 3, wspace=.34)
    maps_bottom = outer[2].subgridspec(1, 3, wspace=.34)

    ax = fig.add_subplot(top[0, 0])
    for dim, color in zip(dimensions, dim_colors):
        sub = spread.loc[spread.dimension.eq(dim)]
        ax.plot(sub.year, sub.sd, marker="o", ms=3.1, lw=1.25, color=color, label=dim)
    ax.set_ylabel("SD of local coefficient"); ax.set_xlabel("Health-reference year")
    clean_axes(ax, "y"); panel(ax, 0, "Spatial spread over time")

    ax = fig.add_subplot(top[0, 1])
    schemes = list(means.scheme.drop_duplicates())
    markers = ["o", "s", "^", "D"]
    offsets = np.linspace(-.18, .18, len(schemes))
    matrix = means.pivot(index="dimension", columns="scheme", values="mean").reindex(dimensions)
    for scheme, marker, offset in zip(schemes, markers, offsets):
        ax.scatter(matrix[scheme], np.arange(len(dimensions)) + offset, marker=marker, s=22,
                   facecolor="white", edgecolor=INK, linewidth=.75, label=scheme)
    ax.axvline(0, color=NEUTRAL, lw=.7)
    ax.set_yticks(range(len(dimensions)), dimensions); ax.invert_yaxis(); ax.set_xlabel("Mean local coefficient")
    clean_axes(ax, "x"); panel(ax, 1, "Partition-specific means")

    ax = fig.add_subplot(top[0, 2])
    vals = stability.set_index("dimension").reindex(dimensions).corr_22_23.to_numpy(float)
    yy = np.arange(len(dimensions))
    ax.hlines(yy, .985, vals, color=dim_colors, lw=1.8)
    ax.scatter(vals, yy, s=35, color=dim_colors, edgecolor="white", linewidth=.55, zorder=3)
    ax.set_yticks(yy, dimensions); ax.invert_yaxis(); ax.set_xlim(.985, 1.0005)
    ax.set_xlabel("Tract correlation")
    for y, value in enumerate(vals):
        ax.text(value - .00045, y, f"{value:.3f}", ha="right", va="center", fontsize=5.5)
    clean_axes(ax, "x"); panel(ax, 2, "2022–2023 stability")

    map_slots = [(maps_top, 0, 0), (maps_top, 0, 1), (maps_top, 0, 2),
                 (maps_bottom, 0, 0), (maps_bottom, 0, 1)]
    for idx, (dim, slot) in enumerate(zip(dimensions, map_slots), start=3):
        grid, rr, cc = slot
        ax = fig.add_subplot(grid[rr, cc])
        frame = map_frames[dim]
        cmap, norm, boundaries = signed_magnitude_quantile_scale(frame.local_coefficient.to_numpy(float))
        frame.plot(column="local_coefficient", ax=ax, cmap=cmap, norm=norm,
                   edgecolor="white", linewidth=.035,
                   missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .03})
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal"); ax.set_axis_off()
        cb, _ = add_equal_height_colorbar(fig, ax, ScalarMappable(norm=norm, cmap=cmap), width=.030, gap=.018)
        cb.set_ticks(symmetric_colourbar_ticks(boundaries[0], boundaries[-1])); cb.ax.tick_params(labelsize=5.6)
        panel(ax, idx, dim, x=-.01, y=1.01)
    # Reserve the final map cell for both keys.  Moving the partition legend out
    # of panel b prevents it from colliding with the panel title and data.
    key_ax = fig.add_subplot(maps_bottom[0, 2]); key_ax.set_axis_off()
    partition_handles = [Line2D([0], [0], marker=m, color="none", markerfacecolor="white",
                                markeredgecolor=INK, markeredgewidth=.75, markersize=4.5,
                                label=s) for s, m in zip(schemes, markers)]
    leg1 = key_ax.legend(handles=partition_handles, title="Registered partition",
                         loc="upper left", bbox_to_anchor=(.05, .86), frameon=False,
                         fontsize=5.8, title_fontsize=6.2, handletextpad=.5, labelspacing=.65)
    key_ax.add_artist(leg1)
    key_ax.legend([Line2D([0], [0], color=c, marker="o", lw=1.4) for c in dim_colors], dimensions,
                  title="Mapped dimension", loc="lower left", bbox_to_anchor=(.05, .04),
                  frameon=False, fontsize=5.8, title_fontsize=6.2,
                  handlelength=1.4, handletextpad=.5, labelspacing=.65)
    out = _finish(
        fig, "fig8_coefficient_heterogeneity", "Fig8_coefficient_heterogeneity_editorial_v2", 8,
        [spread_path, means_path, stability_path, maps_path],
        "Five pooled 2023 coefficient maps are the hero evidence; spread, partition means and temporal stability provide validation.",
    )
    pooled.to_csv(out.parent / "source_pooled_2023_maps.csv", index=False)
    return out


def build_fig9() -> Path:
    """Eight preselected response curves with explicit interval hierarchy."""
    setup_style(7.0)
    base = ROOT / "figures/v211_model_perturbation_moderation_20panel"
    curves_path = base / "source_cubic_response_curves.csv"
    stats_path = base / "source_p10_p90_contrasts.csv"
    curves = pd.read_csv(curves_path)
    stats = pd.read_csv(stats_path)
    specs = [
        ("joint_six_dimensions", "mod__median_household_income", "Package × income", "Household income (US$1,000)"),
        ("joint_six_dimensions", "mod__poverty_pct", "Package × poverty", "Below poverty (%)"),
        ("joint_six_dimensions", "mod__bachelors_or_higher_pct", "Package × education", "Bachelor's or higher (%)"),
        ("joint_six_dimensions", "ses__unemployment_pct", "Package × unemployment", "Unemployment (%)"),
        ("D1_green_water_plus_1sd", "mod__median_household_income", "Greening × income", "Household income (US$1,000)"),
        ("D5_clean_maintenance_minus_1sd", "mod__median_household_income", "Upkeep × income", "Household income (US$1,000)"),
        ("D8_air_pollution_minus_1sd", "mod__median_household_income", "Clean-air × income", "Household income (US$1,000)"),
        ("D10_traffic_pressure_minus_1sd", "mod__median_household_income", "Traffic calming × income", "Household income (US$1,000)"),
    ]
    selected = []
    fig, axes = plt.subplots(2, 4, figsize=(183 / 25.4, 104 / 25.4), facecolor="white")
    fig.subplots_adjust(left=.075, right=.975, bottom=.115, top=.945, wspace=.47, hspace=.54)
    for idx, (scenario, moderator, title, xlabel) in enumerate(specs):
        ax = axes.flat[idx]
        sub = curves.loc[curves.scenario.eq(scenario) & curves.moderator.eq(moderator)].sort_values("x")
        stat = stats.loc[stats.scenario.eq(scenario) & stats.moderator.eq(moderator)].iloc[0]
        color = BLUE if idx < 4 else RED
        ax.fill_between(sub.x, sub.lo90, sub.hi90, color=color, alpha=.10, linewidth=0)
        ax.fill_between(sub.x, sub.lo50, sub.hi50, color=color, alpha=.20, linewidth=0)
        ax.plot(sub.x, sub["mean"], color=color, lw=1.65)
        ax.axhline(0, color=NEUTRAL, lw=.6)
        ax.scatter([stat.p10, stat.p90], [stat.response_p10, stat.response_p90], s=18,
                   color=color, edgecolor="white", linewidth=.4, zorder=3)
        ax.set_xlabel(xlabel)
        if idx % 4 == 0:
            ax.set_ylabel("Model response (pp)")
        clean_axes(ax, "both")
        panel(ax, idx, title)
        ax.text(.98, .93, f"p90−p10 {stat.p90_minus_p10:+.3f} pp", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.6, color=MUTED)
        selected.append(sub)
    out = _finish(
        fig, "fig9_moderation", "Fig9_moderation_editorial_v2", 8, [curves_path, stats_path],
        "Four integrated-package social gradients and four registered strategy-by-income curves; 50% and 90% mean-response intervals are nested.",
    )
    pd.concat(selected, ignore_index=True).to_csv(out.parent / "source_selected_curves.csv", index=False)
    return out


def build_fig11() -> Path:
    """Four summaries and four large representative response maps."""
    setup_style(6.9)
    base = ROOT / "figures/v211_planning_strategy_20panel"
    intervals_path = base / "source_tract_year_response_intervals.csv"
    coverage_path = base / "source_directional_coverage.csv"
    annual_path = base / "source_annual_median_response.csv"
    additivity_path = base / "source_package_additivity_residual.csv"
    map_path = base / "source_2023_scheme_averaged_spatial_response.csv"
    labels_path = base / "strategy_label_dictionary.csv"
    intervals = pd.read_csv(intervals_path)
    coverage = pd.read_csv(coverage_path)
    annual = pd.read_csv(annual_path)
    additivity = pd.read_csv(additivity_path)
    maps = pd.read_csv(map_path, dtype={"tract_geoid_native": str})
    maps["tract_geoid_native"] = maps.tract_geoid_native.str.zfill(11)
    labels = pd.read_csv(labels_path).set_index("scenario").short_label.to_dict()
    scenarios = list(intervals.scenario)
    colors = {key: (RED if float(intervals.set_index("scenario").loc[key, "median"]) >= 0 else BLUE) for key in scenarios}

    fig = plt.figure(figsize=(183 / 25.4, 136 / 25.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=.125, right=.940, bottom=.040, top=.955,
                             height_ratios=[1.00, 1.72], hspace=.25)
    top = outer[0].subgridspec(1, 4, wspace=.48)
    bottom = outer[1].subgridspec(1, 4, wspace=.34)

    order = intervals.sort_values("median").scenario.tolist()
    ax = fig.add_subplot(top[0, 0])
    for y, key in enumerate(order):
        row = intervals.set_index("scenario").loc[key]
        ax.hlines(y, row.q05, row.q95, color=colors[key], lw=1.1, alpha=.8)
        ax.hlines(y, row.q25, row.q75, color=colors[key], lw=4.6)
        ax.scatter(row["median"], y, s=28, color=colors[key], edgecolor="white", linewidth=.5, zorder=3)
    ax.axvline(0, color=INK, lw=.65); ax.set_yticks(range(len(order)), [labels[k] for k in order]); ax.invert_yaxis()
    ax.set_xlabel("Model-projected change (pp)"); clean_axes(ax, "x"); panel(ax, 0, "Response intervals")

    ax = fig.add_subplot(top[0, 1])
    cov = coverage.set_index("scenario").reindex(order)
    for y, key in enumerate(order):
        value = 100 * float(cov.loc[key, "fraction_improved"])
        ax.hlines(y, 50, value, color=colors[key], lw=2.0)
        ax.scatter(value, y, s=34, color=colors[key], edgecolor="white", linewidth=.5)
    ax.axvline(50, color=INK, lw=.65, ls="--"); ax.set_xlim(0, 100)
    ax.set_yticks(range(len(order)), [labels[k] for k in order]); ax.invert_yaxis(); ax.set_xlabel("Tracts improved (%)")
    clean_axes(ax, "x"); panel(ax, 1, "Directional coverage")

    ax = fig.add_subplot(top[0, 2])
    years = [int(c) for c in annual.columns if c != "strategy"]
    for _, row in annual.iterrows():
        key = row.strategy
        ax.plot(years, row[[str(y) for y in years]].astype(float), color=colors[key], lw=1.25,
                marker="o", ms=2.6, alpha=.78)
    ax.axhline(0, color=INK, lw=.6); ax.set_xlabel("Health-reference year"); ax.set_ylabel("Median change (pp)")
    ax.set_xticks([2014, 2017, 2020, 2023])
    clean_axes(ax, "y"); panel(ax, 2, "Annual response")

    ax = fig.add_subplot(top[0, 3])
    x = additivity.sum_component_change.to_numpy(float)
    y = additivity.non_additivity_residual.to_numpy(float) * 1e7
    ax.scatter(x, y, s=5.5, color=np.where(y >= 0, RED_MID, BLUE_MID), alpha=.25, linewidth=0, rasterized=True)
    ax.axhline(0, color=INK, lw=.65); ax.set_xlabel("Sum of component changes (pp)")
    ax.set_ylabel("Non-additivity residual (×10^-7 pp)")
    clean_axes(ax, "both"); panel(ax, 3, "Additivity residual")

    selected = ["D1_green_water_plus_1sd", "D5_clean_maintenance_minus_1sd",
                "D10_traffic_pressure_minus_1sd", "joint_six_dimensions"]
    geo = _geo(2023); extent = _extent([geo])
    for ii, key in enumerate(selected):
        ax = fig.add_subplot(bottom[0, ii])
        sub = maps.loc[maps.scenario.eq(key), ["tract_geoid_native", "prediction_difference"]]
        frame = geo.merge(sub, left_on="GEOID", right_on="tract_geoid_native", how="left", validate="one_to_one")
        cmap, norm, boundaries = signed_magnitude_quantile_scale(frame.prediction_difference.to_numpy(float))
        frame.plot(column="prediction_difference", ax=ax, cmap=cmap, norm=norm,
                   edgecolor="white", linewidth=.035,
                   missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .03})
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal"); ax.set_axis_off()
        cb, _ = add_equal_height_colorbar(fig, ax, ScalarMappable(norm=norm, cmap=cmap), width=.030, gap=.018)
        cb.set_ticks(symmetric_colourbar_ticks(boundaries[0], boundaries[-1])); cb.ax.tick_params(labelsize=5.6)
        panel(ax, 4 + ii, labels[key], x=-.01, y=1.01)
    return _finish(
        fig, "fig11_planning", "Fig11_planning_editorial_v2", 8,
        [intervals_path, coverage_path, annual_path, additivity_path, map_path, labels_path],
        "All-strategy summaries lead; four large representative maps span provision, upkeep, mobility and the integrated package.",
    )


def build_fig12() -> Path:
    """Five non-redundant scenario and SES panels."""
    setup_style(6.9)
    base = ROOT / "figures/v211_scenario_inequality_20panel"
    interval_path = base / "source_scenario_nested_intervals.csv"
    disparity_path = base / "source_ses_prediction_disparity.csv"
    agreement_path = base / "source_ses_directional_agreement.csv"
    magnitude_path = base / "source_scenario_magnitude_heterogeneity.csv"
    direction_path = base / "source_scenario_directional_coverage.csv"
    intervals = pd.read_csv(interval_path)
    disparity = pd.read_csv(disparity_path)
    agreement = pd.read_csv(agreement_path)
    magnitude = pd.read_csv(magnitude_path)
    direction = pd.read_csv(direction_path)
    label_path = ROOT / "figures/v211_planning_strategy_20panel/strategy_label_dictionary.csv"
    labels = pd.read_csv(label_path).set_index("scenario").short_label.to_dict()
    ses_labels = {
        "ses__age_65plus_pct": "Age 65+", "ses__age_under18_pct": "Under 18",
        "ses__bachelors_or_higher_pct_25plus": "Higher education", "ses__black_alone_pct": "Black population",
        "ses__commute_transit_pct": "Transit commute", "ses__commute_walk_pct": "Walk commute",
        "ses__gross_rent_30plus_pct": "Rent burden", "ses__hispanic_latino_pct": "Hispanic / Latino",
        "ses__log_income_nominal": "Household income", "ses__male_pct": "Male population",
        "ses__population_log1p": "Population size", "ses__poverty_pct": "Poverty",
        "ses__unemployment_pct": "Unemployment",
    }
    scheme_order = list(disparity.scheme.drop_duplicates())
    ses_order = list(disparity.ses_variable.drop_duplicates())
    matrix = disparity.pivot(index="ses_variable", columns="scheme", values="standardized_prediction_gap").reindex(index=ses_order, columns=scheme_order)
    lim = max(float(np.abs(matrix.to_numpy()).max()), 1e-9)

    fig = plt.figure(figsize=(183 / 25.4, 147 / 25.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=.135, right=.900, bottom=.080, top=.955,
                             height_ratios=[1.25, 1.0], hspace=.38)
    top = outer[0].subgridspec(1, 2, width_ratios=[.92, 1.38], wspace=.46)
    bottom = outer[1].subgridspec(1, 3, wspace=.50)

    order = intervals.sort_values("median").scenario.tolist()
    ax = fig.add_subplot(top[0, 0])
    for y, key in enumerate(order):
        row = intervals.set_index("scenario").loc[key]
        color = RED if row["median"] >= 0 else BLUE
        ax.hlines(y, row.q05, row.q95, color=color, lw=1.15, alpha=.75)
        ax.hlines(y, row.q25, row.q75, color=color, lw=4.8)
        ax.scatter(row["median"], y, s=31, color=color, edgecolor="white", linewidth=.5)
    ax.axvline(0, color=INK, lw=.65); ax.set_yticks(range(len(order)), [labels[k] for k in order]); ax.invert_yaxis()
    ax.set_xlabel("Model-projected change (pp)"); clean_axes(ax, "x"); panel(ax, 0, "Scenario response intervals")

    ax = fig.add_subplot(top[0, 1])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    image = ax.imshow(matrix.to_numpy(), cmap=DIV, norm=norm, aspect="auto", interpolation="nearest")
    short_schemes = []
    for scheme in scheme_order:
        prefix = "AR" if scheme.startswith("axis") else "PC" if scheme.startswith("polar") else "R45" if scheme.startswith("rotated") else "EQ"
        short_schemes.append(f"{prefix}·{'10' if scheme.endswith('2010') else '20'}")
    ax.set_xticks(range(len(scheme_order)), short_schemes)
    ax.set_yticks(range(len(ses_order)), [ses_labels[x] for x in ses_order])
    ax.tick_params(length=0, pad=2)
    for rr in range(matrix.shape[0]):
        for cc in range(matrix.shape[1]):
            value = float(matrix.iloc[rr, cc])
            ax.text(cc, rr, f"{value:+.1f}", ha="center", va="center", fontsize=5.4,
                    color="white" if abs(value) > .62 * lim else INK)
    cb, _ = add_equal_height_colorbar(fig, ax, image, width=.025, gap=.018)
    cb.set_ticks([-lim, 0, lim]); cb.ax.tick_params(labelsize=5.7); cb.set_label("Standardized Q3 − Q1 gap", fontsize=5.8)
    panel(ax, 1, "Socioeconomic prediction gaps", x=-.055, y=1.02)

    ax = fig.add_subplot(bottom[0, 0])
    agree = agreement.set_index("ses_variable").reindex(ses_order)
    yy = np.arange(len(ses_order))
    ax.barh(yy, -agree.negative_share_pct, color=BLUE_MID, height=.62)
    ax.barh(yy, agree.positive_share_pct, color=RED_MID, height=.62)
    ax.axvline(0, color=INK, lw=.65); ax.set_xlim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100], ["100%", "50%", "0", "50%", "100%"])
    ax.set_yticks(yy, [ses_labels[x] for x in ses_order]); ax.invert_yaxis(); ax.set_xlabel("Share of eight registered schemes")
    clean_axes(ax, "x"); panel(ax, 2, "Directional agreement")

    ax = fig.add_subplot(bottom[0, 1])
    mag = magnitude.set_index("scenario").reindex(order)
    yy = np.arange(len(order))
    iqr_values = mag.iqr.to_numpy(float)
    point_colors = [RED if value >= 0 else BLUE for value in mag["median"].to_numpy(float)]
    ax.hlines(yy, 0, iqr_values, color=point_colors, lw=1.8)
    ax.scatter(iqr_values, yy, s=36, color=point_colors, edgecolor="white", linewidth=.5, zorder=3)
    ax.set_yticks(yy, [labels[key] for key in order]); ax.invert_yaxis()
    ax.set_xlabel("Tract–fold IQR (pp)")
    clean_axes(ax, "x"); panel(ax, 3, "Response heterogeneity")

    ax = fig.add_subplot(bottom[0, 2])
    direct = direction.set_index("scenario").reindex(order)
    y = np.arange(len(order))
    ax.barh(y, direct.negative_share_pct, color=BLUE_MID, height=.62, label="negative")
    ax.barh(y, direct.positive_share_pct, left=direct.negative_share_pct, color=RED_MID, height=.62, label="positive")
    ax.axvline(50, color="white", lw=.7, ls="--"); ax.set_xlim(0, 100)
    ax.set_yticks(y, [labels[k] for k in order]); ax.invert_yaxis(); ax.set_xlabel("Tract–fold response share")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=5.6)
    clean_axes(ax, "x"); panel(ax, 4, "Response direction")
    return _finish(
        fig, "fig12_scenario_ses", "Fig12_scenario_ses_editorial_v2", 5,
        [interval_path, disparity_path, agreement_path, magnitude_path, direction_path, label_path],
        "Scenario intervals and the complete SES matrix are the hero evidence; three compact summaries replace repeated ECDFs.",
    )


def _interval(ax, y: float, values: np.ndarray, color: str) -> float:
    q05, q25, med, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95])
    ax.hlines(y, q05, q95, color=color, lw=1.1, alpha=.72)
    ax.hlines(y, q25, q75, color=color, lw=4.8)
    ax.scatter(med, y, s=31, color=color, edgecolor="white", linewidth=.5, zorder=3)
    return float(med)


def build_fig13() -> Path:
    """Six route/nonlinear panels with one clear diagnostic per cell."""
    setup_style(7.0)
    base = ROOT / "figures/v211_routing_nonlinear_flagship_11panel"
    paired_path = base / "source_fold_paired_comparison.csv"
    faith_path = base / "source_faithfulness_summary.csv"
    curvature_path = base / "source_curvature_sign_audit.csv"
    dose_path = base / "source_pooled_dose_response_surface.csv"
    paired = pd.read_csv(paired_path)
    faith = pd.read_csv(faith_path)
    curvature = pd.read_csv(curvature_path)
    dose = pd.read_csv(dose_path, index_col=0)

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 109 / 25.4), facecolor="white")
    fig.subplots_adjust(left=.075, right=.920, bottom=.105, top=.945, wspace=.52, hspace=.55)
    a, b, c, d, e, f = axes.flat

    lo = min(paired.baseline_rmse.min(), paired.nonlinear_rmse.min())
    hi = max(paired.baseline_rmse.max(), paired.nonlinear_rmse.max())
    delta = paired.delta_rmse.to_numpy(float)
    norm = TwoSlopeNorm(vmin=-max(abs(delta)), vcenter=0, vmax=max(abs(delta)))
    a.plot([lo, hi], [lo, hi], color=INK, lw=.75, ls="--")
    a.scatter(paired.baseline_rmse, paired.nonlinear_rmse, s=28, c=DIV(norm(delta)), edgecolor="white", linewidth=.55)
    a.set(xlabel="Transparent-head RMSE (pp)", ylabel="Feature-dependent RMSE (pp)", xlim=(lo, hi), ylim=(lo, hi))
    a.text(.97, .05, f"improved: {(delta < 0).sum()}/32", transform=a.transAxes, ha="right", fontsize=5.6)
    clean_axes(a, "both"); panel(a, 0, "Fold-paired error")

    _interval(b, 1, paired.delta_rmse.to_numpy(float), BLUE)
    _interval(b, 0, paired.delta_mae.to_numpy(float), RED)
    b.axvline(0, color=INK, lw=.65); b.set_yticks([1, 0], ["ΔRMSE", "ΔMAE"])
    b.set_xlabel("Feature-dependent − transparent (pp)")
    clean_axes(b, "x"); panel(b, 1, "Paired error change")

    faith_arrays = [faith.sufficiency_abs_error.to_numpy(float), faith.random_sufficiency_abs_error.to_numpy(float), faith.comprehensiveness_abs_change.to_numpy(float)]
    for y, vals, color in zip([2, 1, 0], faith_arrays, [BLUE, BLUE_MID, RED]):
        _interval(c, y, vals, color)
    c.set_yticks([2, 1, 0], ["Selected error", "Random error", "Removal change"])
    c.set_xlabel("Absolute prediction quantity (pp)")
    clean_axes(c, "x"); panel(c, 2, "Explainer faithfulness")

    relations = [("Queen", "relation_mass_spatial_queen"), ("Road", "relation_mass_road_connectivity"),
                 ("Mobility", "relation_mass_mobility_flow"), ("Temporal", "relation_mass_temporal_forward")]
    for y, (label, col) in enumerate(relations[::-1]):
        vals = faith[col].to_numpy(float)
        _interval(d, y, vals, RED if label == "Mobility" else BLUE_MID)
    d.set_yticks(range(4), [x[0] for x in relations[::-1]]); d.set_xlabel("Relation-mask mass")
    clean_axes(d, "x"); panel(d, 3, "Admitted graph relations")

    bins = np.linspace(.48, .86, 13)
    counts, edges = np.histogram(curvature.same_sign_fraction, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    e.bar(centres, counts, width=np.diff(edges) * .92, color=DIV(Normalize(.50, .85)(centres)), edgecolor="white", linewidth=.35)
    e.axvline(.75, color=RED, ls="--", lw=.9); e.axvline(.90, color=INK, ls=":", lw=.9)
    e.set_xlabel("Same-sign fraction across folds"); e.set_ylabel("Feature × SES cells")
    clean_axes(e, "y"); panel(e, 4, "Curvature-sign stability")

    dose_values = dose.to_numpy(float)
    dose_lim = max(float(np.nanquantile(np.abs(dose_values), .98)), .01)
    im = f.imshow(
        dose_values,
        aspect="auto",
        cmap=DIV,
        norm=TwoSlopeNorm(vmin=-dose_lim, vcenter=0, vmax=dose_lim),
    )
    dims = [DIMENSION_LABELS[i] for i in range(1, 11)]
    f.set_yticks(range(10), dims); f.set_xticks(range(dose.shape[1]), [f"{float(x):+g}" for x in dose.columns])
    f.set_xlabel("S−O perturbation (fold-training SD)")
    cb, _ = add_equal_height_colorbar(fig, f, im, width=.025, gap=.018)
    cb.set_ticks([-dose_lim, 0, dose_lim]); cb.set_ticklabels([f"{-dose_lim:.3f}", "0", f"{dose_lim:.3f}"])
    cb.ax.tick_params(labelsize=5.7); cb.set_label("Median projected change (pp)", fontsize=5.8)
    panel(f, 5, "Pooled fitted response surface", x=-.07, y=1.02)
    return _finish(
        fig, "fig13_routing_nonlinear", "Fig13_routing_nonlinear_editorial_v2", 6,
        [paired_path, faith_path, curvature_path, dose_path],
        "Fold-paired performance and dose response lead; faithfulness, admitted relations and sign stability delimit interpretation.",
    )


def main() -> None:
    outputs = {
        "fig2": str(build_fig2()),
        "fig4": str(build_fig4()),
        "fig8": str(build_fig8()),
        "fig9": str(build_fig9()),
        "fig11": str(build_fig11()),
        "fig12": str(build_fig12()),
        "fig13": str(build_fig13()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "editorial_manifest.json").write_text(
        json.dumps({"analysis_version": "V211", "proxy_analysis": True, "formal": False, "outputs": outputs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
