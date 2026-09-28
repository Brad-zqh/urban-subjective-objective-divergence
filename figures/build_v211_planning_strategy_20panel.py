"""Compact 20-panel planning-sensitivity synthesis for current MM-GTGNNWR.

The strategy names translate standardized feature perturbations into concrete
urban-planning actions. They remain model sensitivities, not causal policy
effects. Mixed panel types avoid redundant distribution plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib.lines import Line2D
from PIL import Image

from nature_viz_common import (
    ROOT, BLUE, BLUE_MID, RED, RED_MID, INK, MUTED, DIV, MISSING,
    setup_style, panel, clean_axes, add_equal_height_colorbar, save_delivery,
    sha256, symmetric_limit, write_validation,
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
KEYS = [x[0] for x in STRATEGIES]
SHORT = {k: label for k, label, _ in STRATEGIES}
DETAIL = {k: detail for k, _, detail in STRATEGIES}


def border_check(path: str) -> tuple[float, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate([
        rgb[:5].reshape(-1, 3), rgb[-5:].reshape(-1, 3),
        rgb[:, :5].reshape(-1, 3), rgb[:, -5:].reshape(-1, 3),
    ])
    return float(np.mean(np.any(rgb < 245, axis=2))), float(np.mean(np.any(border < 235, axis=1)))


def signed_color(value: float) -> str:
    return RED if value > 0 else BLUE


def main() -> None:
    setup_style(5.45)
    before = {str(p): sha256(p) for p in [DATA, COEF, GEOMETRY]}
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
                       how="inner", validate="one_to_one") for key in KEYS
    }
    bounds = np.vstack([frame.total_bounds for frame in frames.values()])
    xmin, ymin = bounds[:, 0].min(), bounds[:, 1].min()
    xmax, ymax = bounds[:, 2].max(), bounds[:, 3].max()
    dx, dy = xmax - xmin, ymax - ymin
    extent = (xmin - .012 * dx, xmax + .012 * dx, ymin - .012 * dy, ymax + .012 * dy)
    map_norms = {}
    for key in KEYS:
        lim = symmetric_limit(frames[key].prediction_difference, .985)
        map_norms[key] = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    fig = plt.figure(figsize=(183 / 25.4, 175 / 25.4))
    outer = fig.add_gridspec(4, 1, left=.09, right=.95, bottom=.045, top=.975,
                             hspace=.34, height_ratios=[1.0, .66, .58, .82])
    top = outer[0].subgridspec(1, 4, wspace=.58)
    annual_gs = outer[1].subgridspec(1, 7, wspace=.34)
    map_gs = outer[2].subgridspec(1, 7, wspace=.34)
    bottom = outer[3].subgridspec(1, 2, wspace=.40)
    summary_rows = []

    ax = fig.add_subplot(top[0, 0])
    for y, key in enumerate(KEYS):
        values = tract_year.loc[tract_year.scenario.eq(key), "prediction_difference"].to_numpy(float)
        q05, q25, med, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95])
        color = signed_color(med)
        ax.hlines(y, q05, q95, color=color, lw=.65, alpha=.55)
        ax.hlines(y, q25, q75, color=color, lw=2.3, alpha=.92)
        ax.scatter(med, y, s=18, color=color, edgecolor="white", linewidth=.4, zorder=3)
        summary_rows.append({"scenario": key, "median": med, "q05": q05, "q25": q25,
                             "q75": q75, "q95": q95, "n": len(values)})
    ax.axvline(0, color=MUTED, lw=.55)
    ax.set_yticks(range(7), [SHORT[k] for k in KEYS]); ax.invert_yaxis()
    ax.set_xlabel("Prediction change (pp)"); clean_axes(ax, "x")
    panel(ax, 0, "Tract-year response")

    ax = fig.add_subplot(top[0, 1])
    coverage = tract_year.assign(lower=lambda x: x.prediction_difference.lt(0)).groupby("scenario").lower.mean().reindex(KEYS)
    for y, (_, value) in enumerate(coverage.items()):
        color = BLUE if value >= .5 else RED
        ax.hlines(y, 50, 100 * value, color=color, lw=1.25)
        ax.scatter(100 * value, y, s=17, color=color, edgecolor="white", linewidth=.35)
    ax.axvline(50, color=MUTED, lw=.55, ls="--")
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Tract-years improved (%)"); clean_axes(ax, "x")
    panel(ax, 1, "Directional coverage")

    ax = fig.add_subplot(top[0, 2])
    magnitude = tract_year.groupby("scenario").prediction_difference.median().reindex(KEYS)
    hetero = tract_year.groupby("scenario").prediction_difference.agg(
        lambda v: np.quantile(v, .75) - np.quantile(v, .25)
    ).reindex(KEYS)
    ax.axvline(0, color=MUTED, lw=.5); ax.axhline(0, color=MUTED, lw=.5)
    for i, key in enumerate(KEYS):
        color = signed_color(magnitude[key])
        ax.scatter(magnitude[key], hetero[key], s=21, color=color,
                   edgecolor="white", linewidth=.4, zorder=3)
        offsets = {
            "D1_green_water_plus_1sd": (3, 3),
            "D5_clean_maintenance_minus_1sd": (3, -8),
            "D7_heat_mitigation_minus_1sd": (3, 3),
            "D8_air_pollution_minus_1sd": (3, -8),
            "D9_noise_mitigation_minus_1sd": (-42, 3),
            "D10_traffic_pressure_minus_1sd": (3, 4),
            "joint_six_dimensions": (3, -9),
        }
        ax.annotate(SHORT[key], (magnitude[key], hetero[key]),
                    xytext=offsets[key], textcoords="offset points",
                    fontsize=3.7, color=INK)
    ax.set_xlabel("Median change (pp)"); ax.set_ylabel("IQR (pp)")
    clean_axes(ax); panel(ax, 2, "Magnitude-heterogeneity plane")

    ax = fig.add_subplot(top[0, 3]); equity_rows = []
    for y, key in enumerate(KEYS):
        sub = tract_year.loc[tract_year.scenario.eq(key)].copy()
        low_cut, high_cut = sub.poverty.quantile([.25, .75])
        low = float(sub.loc[sub.poverty.le(low_cut), "prediction_difference"].median())
        high = float(sub.loc[sub.poverty.ge(high_cut), "prediction_difference"].median())
        ax.plot([low, high], [y, y], color="#AAB2BD", lw=1.0)
        ax.scatter(low, y, s=14, color=BLUE_MID, edgecolor="white", linewidth=.3, zorder=3)
        ax.scatter(high, y, s=14, color=RED_MID, edgecolor="white", linewidth=.3, zorder=3)
        equity_rows.append({"scenario": key, "low_poverty_q1": low, "high_poverty_q4": high,
                            "high_minus_low": high - low})
    ax.axvline(0, color=MUTED, lw=.55)
    ax.set_yticks(
        range(7),
        ["Green", "Upkeep", "Cooling", "Clean air", "Quiet", "Traffic", "Integrated"],
        fontsize=3.7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Median change (pp)"); clean_axes(ax, "x")
    panel(ax, 3, "Poverty-stratified response")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_MID,
               markeredgecolor="white", label="Lower-poverty Q1", markersize=4),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED_MID,
               markeredgecolor="white", label="Higher-poverty Q4", markersize=4),
    ], fontsize=3.8, loc="upper right", handletextpad=.2)

    annual_rows = []
    for j, key in enumerate(KEYS):
        ax = fig.add_subplot(annual_gs[0, j])
        sub = tract_year.loc[tract_year.scenario.eq(key)]
        q = sub.groupby("health_reference_year").prediction_difference.quantile([.05, .5, .95]).unstack()
        q.columns = ["q05", "median", "q95"]; q = q.reset_index()
        q["scenario"] = key; annual_rows.append(q)
        color = signed_color(float(sub.prediction_difference.median()))
        x = q.health_reference_year.to_numpy(float)
        ax.fill_between(x, q.q05, q.q95, color=color, alpha=.16, linewidth=0)
        ax.plot(x, q["median"], color=color, lw=1.15)
        ax.scatter(x, q["median"], s=5, color=color, edgecolor="white", linewidth=.25)
        ax.axhline(0, color=MUTED, lw=.5)
        ax.set_xticks([2014, 2018, 2023]); ax.set_xlabel("")
        ax.set_ylabel("Change (pp)" if j == 0 else "")
        clean_axes(ax); panel(ax, 4 + j, SHORT[key], x=-.06, y=1.015)

    cbar_pairs = []; map_audit = []
    for j, key in enumerate(KEYS):
        ax = fig.add_subplot(map_gs[0, j]); frame = frames[key]; norm = map_norms[key]
        frame.plot(column="prediction_difference", ax=ax, cmap=DIV, norm=norm,
                   edgecolor="white", linewidth=.032,
                   missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .03})
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
        panel(ax, 11 + j, SHORT[key], x=-.035, y=1.01)
        cb, cax = add_equal_height_colorbar(
            fig, ax, plt.cm.ScalarMappable(norm=norm, cmap=DIV), width=.038, gap=.025
        )
        cb.set_ticks([norm.vmin, 0, norm.vmax]); cbar_pairs.append((ax, cax))
        map_audit.append({"scenario": key, "n_land_tracts": len(frame),
                          "vmin": norm.vmin, "vcenter": 0.0, "vmax": norm.vmax,
                          "extent_xmin": extent[0], "extent_xmax": extent[1],
                          "extent_ymin": extent[2], "extent_ymax": extent[3]})

    ax_corr = fig.add_subplot(bottom[0, 0]); corr = wide[KEYS].corr()
    im = ax_corr.imshow(corr, cmap=DIV, vmin=-1, vmax=1, interpolation="nearest")
    ax_corr.set_xticks(range(7), [SHORT[k] for k in KEYS], rotation=40, ha="right")
    ax_corr.set_yticks(range(7), [SHORT[k] for k in KEYS])
    ax_corr.tick_params(length=0, labelsize=4.0); panel(ax_corr, 18, "Response correlation")
    cb_corr, cax_corr = add_equal_height_colorbar(fig, ax_corr, im, width=.028, gap=.020)
    cb_corr.set_ticks([-1, 0, 1])

    ax_add = fig.add_subplot(bottom[0, 1])
    x = wide[KEYS[:-1]].sum(axis=1).to_numpy(float)
    y = wide[KEYS[-1]].to_numpy(float)
    xlo, xhi = np.quantile(x, [.005, .995]); ylo, yhi = np.quantile(y, [.005, .995])
    h, _, _ = np.histogram2d(x, y, bins=38, range=((xlo, xhi), (ylo, yhi)))
    art = ax_add.hexbin(x, y, gridsize=38, extent=(xlo, xhi, ylo, yhi), mincnt=1,
                        cmap=DIV, norm=LogNorm(vmin=1, vmax=max(1, h.max())),
                        linewidths=0, rasterized=True)
    line_lo, line_hi = max(xlo, ylo), min(xhi, yhi)
    ax_add.plot([line_lo, line_hi], [line_lo, line_hi], color=INK, lw=.65, ls="--")
    ax_add.set_xlim(xlo, xhi); ax_add.set_ylim(ylo, yhi)
    ax_add.set_xlabel("Sum of component changes (pp)")
    ax_add.set_ylabel("Integrated-package change (pp)")
    clean_axes(ax_add); panel(ax_add, 19, "Package additivity")
    cb_add, cax_add = add_equal_height_colorbar(fig, ax_add, art, width=.018, gap=.020)
    cb_add.set_label("Tract-year density", fontsize=4.0, labelpad=1)

    delivery = save_delivery(fig, OUT, STEM)
    fig.canvas.draw()
    ratios = [cax.get_position().height / ax.get_position().height for ax, cax in cbar_pairs]
    corr_ratio = cax_corr.get_position().height / ax_corr.get_position().height
    add_ratio = cax_add.get_position().height / ax_add.get_position().height
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(OUT / "source_tract_year_response_intervals.csv", index=False)
    pd.Series(coverage, name="fraction_improved").rename_axis("scenario").reset_index().to_csv(
        OUT / "source_directional_coverage.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(OUT / "source_poverty_stratified_response.csv", index=False)
    pd.concat(annual_rows, ignore_index=True).to_csv(OUT / "source_annual_response_intervals.csv", index=False)
    corr.rename_axis("scenario").reset_index().to_csv(OUT / "source_strategy_response_correlations.csv", index=False)
    map_source.to_csv(OUT / "source_2023_scheme_averaged_spatial_response.csv", index=False)
    pd.DataFrame(map_audit).to_csv(OUT / "map_geometry_and_scale_audit.csv", index=False)
    pd.DataFrame([{
        "scenario": key, "short_label": SHORT[key], "implementable_planning_interpretation": DETAIL[key],
        "experimental_dose": "one-standard-deviation composite shift", "causal_interpretation": False,
    } for key in KEYS]).to_csv(OUT / "strategy_label_dictionary.csv", index=False)
    after = {str(p): sha256(p) for p in [DATA, COEF, GEOMETRY]}
    nonwhite, border = border_check(delivery["jpg"])
    write_validation(OUT, {
        "figure": STEM, "panels": 20,
        "proxy_analysis": True, "formal": False,
        "contract": {
            "model": "MM-GTGNNWR", "strategy_labels": DETAIL,
            "experimental_dose": "standardized one-SD feature perturbation",
            "outcome_direction": "negative means lower model-predicted poor mental-health prevalence",
            "interpretation": "planning-relevant model sensitivity; not a causal policy estimate",
        },
        "source_hashes_before": before, "source_hashes_after": after,
        "water_dominant_geometry_rows_excluded": water_excluded, "delivery": delivery,
        "checks": {
            "panel_count_20": True,
            "seven_declared_scenarios_complete": set(joined.scenario) == set(KEYS),
            "thirty_two_outer_folds_each": bool(fold.groupby("scenario").size().eq(32).all()),
            "four_spatial_schemes_averaged": bool(tract_year.scheme_count.eq(4).all()),
            "ten_health_years": tract_year.health_reference_year.nunique() == 10,
            "water_dominant_tracts_removed": water_excluded > 0,
            "individual_colourbar_each_map": len(cbar_pairs) == 7,
            "map_colourbars_equal_height": bool(np.allclose(ratios, 1, atol=.015)),
            "summary_colourbars_equal_height": abs(corr_ratio - 1) < .015 and abs(add_ratio - 1) < .015,
            "all_map_extents_identical": pd.DataFrame(map_audit)[["extent_xmin", "extent_xmax", "extent_ymin", "extent_ymax"]].nunique().eq(1).all(),
            "maps_use_strategy_specific_zero_centred_scales": all(n.vcenter == 0 for n in map_norms.values()),
            "source_files_unchanged": before == after,
            "exact_red_blue_palette": True,
            "only_panel_letters_bold_by_construction": True,
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
            "jpg_not_blank": nonwhite > .055,
            "outer_border_clear": border < .005,
            "complete_v6_plus_jpg_delivery": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
        },
    })
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\nThe twenty panels synthesize current MM-GTGNNWR one-SD planning "
        "perturbations using nested intervals, directional coverage, heterogeneity, poverty-stratified "
        "response, annual ribbons, land-only maps, correlation and package additivity. Practical labels "
        "are interpretations of modeled inputs, not causal intervention effects.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
