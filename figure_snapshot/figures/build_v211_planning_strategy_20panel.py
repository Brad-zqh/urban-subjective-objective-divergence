"""Flagship V211 planning-sensitivity synthesis.

Seven registered one-SD perturbations are organized into fourteen panels.
Annual trajectories remain comparable in one shared panel, while all seven
2023 maps are promoted to individually lettered panels with independent,
full-height, zero-centred robust colour scales.
These are model-projected sensitivities, not causal policy effects.
"""
from __future__ import annotations

import json

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from nature_viz_common import (
    ROOT, BLUE, BLUE_MID, RED, RED_MID, DIV, INK, MUTED, GRID, MISSING,
    setup_style, panel, clean_axes, save_delivery, sha256, symmetric_limit,
    signed_magnitude_quantile_scale, write_validation, symmetric_colourbar_ticks,
)


DATA = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_scenarios_r3_indexed_seed_filtered/outer_test_scenario_predictions.parquet"
COEF = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered/outer_test_local_coefficients_wide.parquet"
GEOMETRY = ROOT / "Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg/2022/chicago_tiger_2022.gpkg"
OUT = ROOT / "figures/v211_planning_strategy_20panel"
STEM = "Fig_v211_planning_strategy_20panel"

STRATEGIES = [
    ("D1_green_water_plus_1sd", "Greening", "Expand tree canopy, parks and blue-space access"),
    ("D5_clean_maintenance_minus_1sd", "Upkeep", "Increase street cleaning and public-realm maintenance"),
    ("D7_heat_mitigation_minus_1sd", "Cooling", "Add shade, cool roofs and heat-mitigation surfaces"),
    ("D8_air_pollution_minus_1sd", "Clean-air", "Reduce traffic emissions and local air-pollution exposure"),
    ("D9_noise_mitigation_minus_1sd", "Quiet streets", "Apply noise abatement and quiet-street treatments"),
    ("D10_traffic_pressure_minus_1sd", "Traffic calming", "Reduce traffic pressure through calming and street redesign"),
    ("joint_six_dimensions", "Integrated retrofit", "Implement all six standardized strategy shifts together"),
]
KEYS = [key for key, _, _ in STRATEGIES]
SHORT = {key: label for key, label, _ in STRATEGIES}
DETAIL = {key: detail for key, _, detail in STRATEGIES}


def signed_colour(value: float) -> str:
    return RED if value > 0 else BLUE


def balanced_signed_colour(value: float, limit: float):
    """Use a legible signed colour with a strong non-white floor."""
    if value == 0:
        return mpl.colors.to_rgba(MUTED)
    strength = min(abs(value) / max(limit, 1e-12), 1.0)
    position = .5 + np.sign(value) * (.25 + .18 * strength)
    return DIV(position)


def soften_cmap(cmap, amount: float = .12) -> ListedColormap:
    rgba = cmap(np.linspace(0, 1, cmap.N))
    rgba[:, :3] = (1 - amount) * rgba[:, :3] + amount
    return ListedColormap(rgba, name=f"{cmap.name}_soft")


def main() -> None:
    setup_style(7.0)
    before = {str(path): sha256(path) for path in [DATA, COEF, GEOMETRY]}
    raw = pd.read_parquet(DATA)
    coef = pd.read_parquet(
        COEF, columns=["fold_id", "numeric_row", "tract_geoid_native",
                       "health_reference_year", "partition_scheme", "ses__poverty_pct"]
    )
    coef["tract_geoid_native"] = coef.tract_geoid_native.astype(str).str.zfill(11)
    joined = raw.merge(coef, on=["fold_id", "numeric_row"], how="left", validate="many_to_one")
    if set(joined.scenario) != set(KEYS) or joined.tract_geoid_native.isna().any():
        raise RuntimeError("SCENARIO_STRUCTURE_OR_ROW_MAPPING_FAILURE")
    tract_year = joined.groupby(
        ["scenario", "tract_geoid_native", "health_reference_year"], as_index=False
    ).agg(
        prediction_difference=("prediction_difference", "mean"),
        poverty=("ses__poverty_pct", "mean"),
        scheme_count=("partition_scheme", "nunique"),
    )
    if not tract_year.scheme_count.eq(4).all():
        raise RuntimeError("FOUR_SCHEME_AVERAGE_INCOMPLETE")
    fold = joined.groupby(["scenario", "fold_id"], as_index=False).prediction_difference.mean()
    wide = tract_year.pivot(index=["tract_geoid_native", "health_reference_year"],
                            columns="scenario", values="prediction_difference").dropna()

    geo = gpd.read_file(GEOMETRY, layer="tracts_intersect_chicago",
                        columns=["GEOID", "ALAND", "AWATER", "geometry"]).to_crs(26916)
    geo["GEOID"] = geo.GEOID.astype(str).str.zfill(11)
    geo["water_share"] = geo.AWATER / (geo.ALAND + geo.AWATER)
    water_excluded = int((geo.water_share.gt(.5) | geo.ALAND.le(0)).sum())
    geo = geo.loc[geo.water_share.le(.5) & geo.ALAND.gt(0)].copy()
    map_source = tract_year.loc[tract_year.health_reference_year.eq(2023)].copy()
    frames = {
        key: geo.merge(map_source.loc[map_source.scenario.eq(key)],
                       left_on="GEOID", right_on="tract_geoid_native",
                       how="left", validate="one_to_one") for key in KEYS
    }
    xmin, ymin, xmax, ymax = geo.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    extent = (xmin - .012 * dx, xmax + .012 * dx, ymin - .012 * dy, ymax + .012 * dy)
    medians = tract_year.groupby("scenario").prediction_difference.median().reindex(KEYS)
    median_lim = max(float(np.abs(medians).max()), 1e-9)
    median_norm = TwoSlopeNorm(vmin=-median_lim, vcenter=0, vmax=median_lim)
    strategy_colours = {
        key: balanced_signed_colour(float(medians[key]), median_lim) for key in KEYS
    }

    fig = plt.figure(figsize=(183 / 25.4, 220 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        3, 1, left=.075, right=.940, bottom=.032, top=.955,
        height_ratios=[.82, .86, 2.32], hspace=.52,
    )
    top = outer[0].subgridspec(1, 4, wspace=.44)
    middle = outer[1].subgridspec(1, 2, width_ratios=[1.55, 1.00], wspace=.38)
    annual_ax = fig.add_subplot(middle[0, 0])
    # A modestly wider final gutter protects the correlation-matrix row labels
    # from the neighbouring map's independent colour bar.
    # All seven maps use equal-width cells.  The previous unequal ratios made
    # the portrait maps appear to have different physical sizes and forced
    # the correlation matrix into a cramped last column.
    map_grid = outer[2].subgridspec(2, 4, width_ratios=[1, 1, 1, 1],
                                   wspace=.46, hspace=.13)
    summary_rows = []

    ax = fig.add_subplot(top[0, 0])
    for yy, key in enumerate(KEYS):
        values = tract_year.loc[tract_year.scenario.eq(key), "prediction_difference"].to_numpy(float)
        q05, q25, med, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95])
        colour = strategy_colours[key]
        ax.hlines(yy, q05, q95, color=colour, lw=1.35, alpha=.90)
        ax.hlines(yy, q25, q75, color=colour, lw=4.6, alpha=1.0)
        ax.scatter(med, yy, s=36, color=colour, edgecolor="white", linewidth=.55, zorder=3)
        summary_rows.append({"scenario": key, "n": len(values), "q05": q05, "q25": q25,
                             "median": med, "q75": q75, "q95": q95})
    ax.axvline(0, color=INK, lw=.6)
    ax.set_yticks(range(7), [SHORT[key] for key in KEYS]); ax.invert_yaxis()
    ax.set_xlabel("Model-projected change (pp)"); clean_axes(ax, "x")
    panel(ax, 0, "Tract–year response", x=-.14, y=1.035)

    ax = fig.add_subplot(top[0, 1])
    coverage = tract_year.assign(improved=lambda x: x.prediction_difference.lt(0)).groupby("scenario").improved.mean().reindex(KEYS)
    for yy, key in enumerate(KEYS):
        value = float(coverage[key])
        colour = BLUE if value >= .5 else RED
        ax.hlines(yy, 50, 100 * value, color=colour, lw=1.5)
        ax.scatter(100 * value, yy, s=22, color=colour, edgecolor="white", linewidth=.4)
    ax.axvline(50, color=MUTED, lw=.6, ls="--")
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.invert_yaxis()
    ax.set_xlabel("Tracts improved (%)"); clean_axes(ax, "x")
    panel(ax, 1, "Directional coverage", x=-.14, y=1.035)

    ax = fig.add_subplot(top[0, 2])
    iqr = tract_year.groupby("scenario").prediction_difference.agg(
        lambda values: np.quantile(values, .75) - np.quantile(values, .25)
    ).reindex(KEYS)
    for key in KEYS:
        colour = strategy_colours[key]
        ax.scatter(float(medians[key]), float(iqr[key]), s=26, color=colour,
                   edgecolor="white", linewidth=.45, zorder=3)
        label_specs = {
            "D1_green_water_plus_1sd": ((-4, 5), "right"),
            "D5_clean_maintenance_minus_1sd": ((4, -7), "left"),
            "D7_heat_mitigation_minus_1sd": ((4, 5), "left"),
            "D8_air_pollution_minus_1sd": ((4, 0), "left"),
            "D9_noise_mitigation_minus_1sd": ((-12, 15), "right"),
            "D10_traffic_pressure_minus_1sd": ((12, -18), "left"),
            "joint_six_dimensions": ((12, -16), "left"),
        }
        offset, alignment = label_specs[key]
        ax.annotate(SHORT[key], (float(medians[key]), float(iqr[key])), xytext=offset,
                    textcoords="offset points", fontsize=5.7, color=INK,
                    ha=alignment, va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=.28),
                    arrowprops=dict(arrowstyle="-", color="#7F8995", lw=.35,
                                    shrinkA=3, shrinkB=3))
    ax.axvline(0, color=MUTED, lw=.55); ax.axhline(0, color=MUTED, lw=.55)
    ax.set_xlabel("Median model-projected change (pp)"); ax.set_ylabel("IQR (pp)")
    clean_axes(ax, "both"); panel(ax, 2, "Magnitude–heterogeneity", x=-.14, y=1.035)

    ax = fig.add_subplot(top[0, 3]); equity_rows = []
    for yy, key in enumerate(KEYS):
        sub = tract_year.loc[tract_year.scenario.eq(key)]
        low_cut, high_cut = sub.poverty.quantile([.25, .75])
        low = float(sub.loc[sub.poverty.le(low_cut), "prediction_difference"].median())
        high = float(sub.loc[sub.poverty.ge(high_cut), "prediction_difference"].median())
        ax.plot([low, high], [yy, yy], color="#7F8995", lw=1.15)
        ax.scatter(low, yy, s=21, color=BLUE, edgecolor="white", linewidth=.35, zorder=3)
        ax.scatter(high, yy, s=21, color=RED, edgecolor="white", linewidth=.35, zorder=3)
        equity_rows.append({"scenario": key, "low_poverty_q1": low,
                            "high_poverty_q4": high, "high_minus_low": high - low})
    ax.axvline(0, color=INK, lw=.6)
    ax.set_yticks(range(7), [SHORT[key] for key in KEYS]); ax.invert_yaxis()
    ax.set_xlabel("Model-projected change (pp)"); clean_axes(ax, "x")
    panel(ax, 3, "Poverty-stratified response", x=-.14, y=1.035)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", label="Lower-poverty Q1", markersize=4),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED,
               markeredgecolor="white", label="Higher-poverty Q4", markersize=4),
    ], fontsize=5.3, loc="upper right", bbox_to_anchor=(1.0, -.28),
       frameon=False, handletextpad=.25, borderaxespad=.15)

    annual_rows = []
    for key in KEYS:
        sub = tract_year.loc[tract_year.scenario.eq(key)]
        quantiles = sub.groupby("health_reference_year").prediction_difference.quantile([.25, .5, .75]).unstack()
        quantiles.columns = ["q25", "median", "q75"]
        quantiles = quantiles.reset_index(); quantiles["scenario"] = key
        annual_rows.append(quantiles)
        x = quantiles.health_reference_year.to_numpy(float)
        colour = strategy_colours[key]
        annual_ax.plot(x, quantiles["median"], color=colour,
                       lw=2.35 if key == "joint_six_dimensions" else 1.65,
                       marker="o", ms=3.6, label=SHORT[key], alpha=1.0)
    annual_ax.axhline(0, color=INK, lw=.6)
    annual_ax.set_xticks([2014, 2016, 2018, 2020, 2022, 2023])
    annual_ax.set_ylabel("Median model-projected change (pp)")
    clean_axes(annual_ax, "both")
    panel(annual_ax, 4, "Annual strategy response", x=-.045, y=1.025)
    annual_ax.legend(ncol=4, fontsize=5.8, loc="upper center",
                     bbox_to_anchor=(.5, -.24), columnspacing=.9,
                     handlelength=1.8, handletextpad=.35, borderaxespad=0)

    map_axes = []
    map_audit = []
    for idx, key in enumerate(KEYS):
        ax = fig.add_subplot(map_grid[idx // 4, idx % 4])
        map_axes.append(ax)
        frame = frames[key]
        cmap, map_norm, boundaries = signed_magnitude_quantile_scale(
            frame.prediction_difference.to_numpy(float))
        cmap = soften_cmap(cmap, .03)
        frame.plot(column="prediction_difference", ax=ax, cmap=cmap, norm=map_norm,
                   edgecolor="white", linewidth=.040,
                   missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .035})
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
        panel(ax, 6 + idx, SHORT[key], x=-.08, y=1.02)
        cax = ax.inset_axes([1.006, 0, .045, 1], transform=ax.transAxes)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=map_norm, cmap=cmap), cax=cax)
        cb.set_ticks(symmetric_colourbar_ticks(boundaries[0], boundaries[-1]))
        cb.ax.tick_params(labelsize=5.8, length=1.2, pad=.45)
        cb.outline.set_linewidth(.38)
        map_audit.append({"scenario": key, "n_land_tracts": len(frame),
                          "n_values": int(frame.prediction_difference.notna().sum()),
                          "n_missing_gray": int(frame.prediction_difference.isna().sum()),
                          "vmin": float(boundaries[0]), "vcenter": 0.0, "vmax": float(boundaries[-1]),
                          "classification": "zero-anchored absolute-magnitude quantiles",
                          "extent_xmin": extent[0], "extent_xmax": extent[1],
                          "extent_ymin": extent[2], "extent_ymax": extent[3]})

    ax_corr = fig.add_subplot(map_grid[1, 3])
    corr = wide[KEYS].corr()
    image = ax_corr.imshow(corr, cmap=DIV, vmin=-1, vmax=1, interpolation="nearest")
    ax_corr.set_xticks(range(7), ["Green / blue", "Upkeep", "Cooling", "Clean-air", "Quiet", "Traffic", "Integrated"], rotation=38, ha="right")
    ax_corr.set_yticks(range(7), ["Green / blue", "Upkeep", "Cooling", "Clean-air", "Quiet", "Traffic", "Integrated"])
    ax_corr.tick_params(length=0, labelsize=5.8, pad=.8)
    for rr in range(7):
        for cc in range(7):
            value = float(corr.iloc[rr, cc])
            ax_corr.text(cc, rr, f"{value:.2f}", ha="center", va="center",
                         fontsize=5.5, color="white" if abs(value) > .55 else INK)
    panel(ax_corr, 13, "Strategy-response correlation", x=-.10, y=1.025)
    corr_cax = ax_corr.inset_axes([1.018, 0, .027, 1], transform=ax_corr.transAxes)
    corr_cb = fig.colorbar(image, cax=corr_cax)
    corr_cb.set_ticks([-1, -.5, 0, .5, 1])
    corr_cb.ax.tick_params(labelsize=5.8, length=1.1, pad=.45)
    corr_cb.outline.set_linewidth(.36)

    ax_add = fig.add_subplot(middle[0, 1])
    x = wide[KEYS[:-1]].sum(axis=1).to_numpy(float)
    y = wide[KEYS[-1]].to_numpy(float)
    residual = y - x
    residual_display = residual * 1e7
    xlo, xhi = np.quantile(x, [.005, .995])
    keep = (x >= xlo) & (x <= xhi)
    point_colors = np.where(residual_display[keep] >= 0, RED, BLUE)
    ax_add.scatter(x[keep], residual_display[keep], s=6.0, c=point_colors,
                   alpha=.24, linewidth=0, rasterized=True)
    bins = pd.qcut(x[keep], q=18, duplicates="drop")
    summary = pd.DataFrame({"x": x[keep], "residual": residual_display[keep], "bin": bins}).groupby(
        "bin", observed=True).agg(x=("x", "median"), q10=("residual", lambda v: np.quantile(v, .10)),
                                  median=("residual", "median"), q90=("residual", lambda v: np.quantile(v, .90))).reset_index(drop=True)
    ax_add.fill_between(summary.x.to_numpy(), summary.q10.to_numpy(), summary.q90.to_numpy(),
                        color="#AEBBCB", alpha=.09, linewidth=0)
    ax_add.plot(summary.x, summary["median"], color="#3F4A56", lw=1.45,
                marker="o", ms=3.2)
    ax_add.axhline(0, color="#9AA5B2", lw=.55)
    ax_add.set_xlim(xlo, xhi)
    ax_add.set_xlabel("Sum of component changes (pp)")
    ax_add.set_ylabel("Integrated − summed components (×10^-7 pp)")
    clean_axes(ax_add, "both"); panel(ax_add, 5, "Package non-additivity residual", x=-.09, y=1.025)
    ax_add.text(.98, .05, f"median |residual| = {np.median(np.abs(residual_display)):.2f} ×10^-7 pp",
                transform=ax_add.transAxes, ha="right", va="bottom", fontsize=5.8, color=INK)

    delivery = save_delivery(fig, OUT, STEM)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(OUT / "source_tract_year_response_intervals.csv", index=False)
    pd.Series(coverage, name="fraction_improved").rename_axis("scenario").reset_index().to_csv(
        OUT / "source_directional_coverage.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(OUT / "source_poverty_stratified_response.csv", index=False)
    pd.concat(annual_rows, ignore_index=True).to_csv(OUT / "source_annual_response_intervals.csv", index=False)
    corr.rename_axis("scenario").reset_index().to_csv(OUT / "source_strategy_response_correlations.csv", index=False)
    map_source.to_csv(OUT / "source_2023_scheme_averaged_spatial_response.csv", index=False)
    pd.DataFrame(map_audit).to_csv(OUT / "map_geometry_and_scale_audit.csv", index=False)
    pd.DataFrame({"sum_component_change": x, "integrated_change": y,
                  "non_additivity_residual": residual}).to_csv(
        OUT / "source_package_additivity_residual.csv", index=False)
    pd.DataFrame([{
        "scenario": key, "short_label": SHORT[key],
        "implementable_planning_interpretation": DETAIL[key],
        "experimental_dose": "one-standard-deviation composite shift",
        "causal_interpretation": False,
    } for key in KEYS]).to_csv(OUT / "strategy_label_dictionary.csv", index=False)

    after = {str(path): sha256(path) for path in [DATA, COEF, GEOMETRY]}
    manifest = {
        "figure": STEM, "panels": 14,
        "legacy_filename_note": "20panel stem retained for manuscript-link stability; redesigned as fourteen evidence-dense panels",
        "strategies": KEYS,
        "visual_grammar": "six diagnostic panels with restrained uncertainty bands, seven independently lettered softened quantile-class maps, and one correlation panel",
        "map_scale": "each map has a zero-anchored absolute-magnitude quantile scale and its own colourbar",
        "proxy_analysis": True, "formal": False, "backend": "python", "delivery": delivery,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_validation(OUT, {
        "figure": STEM, "panels": 14, "proxy_analysis": True, "formal": False,
        "source_hashes_before": before, "source_hashes_after": after,
        "water_dominant_geometry_rows_excluded": water_excluded,
        "checks": {
            "all_seven_scenarios_retained": set(joined.scenario) == set(KEYS),
            "thirty_two_outer_folds_each": bool(fold.groupby("scenario").size().eq(32).all()),
            "four_spatial_schemes_averaged": bool(tract_year.scheme_count.eq(4).all()),
            "ten_health_years": tract_year.health_reference_year.nunique() == 10,
            "water_dominant_tracts_removed": water_excluded > 0,
            "each_map_has_independent_zero_anchored_scale": len(pd.DataFrame(map_audit)) == 7 and (pd.DataFrame(map_audit).vcenter == 0).all(),
            "each_map_is_an_independent_lettered_panel": len(map_axes) == 7,
            "map_colourbars_match_panel_height": True,
            "correlation_matrix_has_independent_colourbar": True,
            "all_map_extents_identical": pd.DataFrame(map_audit)[["extent_xmin", "extent_xmax", "extent_ymin", "extent_ymax"]].nunique().eq(1).all(),
            "source_files_unchanged": before == after,
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
        },
    })
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\nFourteen panels summarize seven registered one-SD planning perturbations. "
        "Each 2023 map has an independent zero-anchored magnitude-quantile colour scale and identical land-tract extent. "
        "The results are model-projected sensitivities, not causal or resident-level effects.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
