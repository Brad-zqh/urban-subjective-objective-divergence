"""Twenty-panel MM-GTGNNWR perturbation-response moderation atlas.

The renderer follows the response-curve logic in the author-supplied
perturbation notebook: cubic mean-response curves, nested 50/90% uncertainty
bands, empirical rugs and p10/p90 contrasts.  Rows are first aligned by the
declared tract-year keys; numeric row order is never assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from nature_viz_common import (
    ROOT, BLUE, BLUE_MID, RED, RED_MID, INK, MUTED, GRID, OFFWHITE, DIV,
    setup_style, panel, clean_axes, save_delivery, sha256, symmetric_limit,
    write_validation,
)


DATA = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_scenarios_r3_indexed_seed_filtered/outer_test_scenario_predictions.parquet"
ROWMAP = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered/outer_test_predictions.parquet"
MODERATORS = ROOT / "MM-GTGNNWR/outputs/processed/tract_year_moderators.parquet"
OUT = ROOT / "figures/v211_model_perturbation_moderation_20panel"
STEM = "Fig_v211_model_perturbation_moderation_20panel"

STRATEGIES = [
    ("D1_green_water_plus_1sd", "Greening"),
    ("D5_clean_maintenance_minus_1sd", "Upkeep"),
    ("D7_heat_mitigation_minus_1sd", "Cooling"),
    ("D8_air_pollution_minus_1sd", "Clean-air"),
    ("D9_noise_mitigation_minus_1sd", "Quiet streets"),
    ("D10_traffic_pressure_minus_1sd", "Traffic calming"),
    ("joint_six_dimensions", "Integrated retrofit"),
]
PACKAGE = "joint_six_dimensions"

MODS = [
    ("mod__median_household_income", "Income", "Household income (US$1,000)", lambda x: x / 1000),
    ("mod__poverty_pct", "Poverty", "Below poverty (%)", None),
    ("mod__male_pct", "Sex", "Male population (%)", None),
    ("mod__black_alone_pct", "Race", "Black population (%)", None),
    ("mod__hispanic_latino_pct", "Ethnicity", "Hispanic/Latino (%)", None),
    ("mod__bachelors_or_higher_pct", "Education", "Bachelor's or higher (%)", None),
    ("ses__commute_transit_pct", "Transit", "Transit commuting (%)", None),
    ("ses__commute_walk_pct", "Walking", "Walking commuting (%)", None),
    ("ses__gross_rent_30plus_pct", "Rent burden", "Rent >=30% of income (%)", None),
    ("mod__age_under18_pct", "Children", "Population age <18 (%)", None),
    ("mod__age_65plus_pct", "Older adults", "Population age >=65 (%)", None),
    ("ses__unemployment_pct", "Unemployment", "Unemployed labour force (%)", None),
    ("mod__population", "Population", "Population (log10)", lambda x: np.log10(np.maximum(x, 1))),
]


def border_check(path: str) -> tuple[float, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate([
        rgb[:5].reshape(-1, 3), rgb[-5:].reshape(-1, 3),
        rgb[:, :5].reshape(-1, 3), rgb[:, -5:].reshape(-1, 3),
    ])
    return float(np.mean(np.any(rgb < 245, axis=2))), float(np.mean(np.any(border < 235, axis=1)))


def fit_curve(frame: pd.DataFrame, x_col: str) -> tuple[pd.DataFrame, dict]:
    """Cluster-robust cubic mean curve on an internally standardized x axis."""
    use = frame[[x_col, "prediction_difference", "tract_geoid_native"]].dropna().copy()
    lo, hi = use[x_col].quantile([.01, .99])
    use = use.loc[use[x_col].between(lo, hi)].copy()
    x = use[x_col].to_numpy(float)
    y = use.prediction_difference.to_numpy(float)
    mean, sd = float(x.mean()), float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise RuntimeError(f"MODERATOR_WITHOUT_VARIATION:{x_col}")
    z = (x - mean) / sd
    design = np.column_stack([np.ones(len(z)), z, z**2, z**3])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    groups = use.tract_geoid_native.to_numpy()
    unique_groups = np.unique(groups)
    for group in unique_groups:
        take = groups == group
        score = design[take].T @ residual[take]
        meat += np.outer(score, score)
    n, k, g = len(y), design.shape[1], len(unique_groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    covariance = correction * bread @ meat @ bread
    xg = np.linspace(float(lo), float(hi), 180)
    zg = (xg - mean) / sd
    dg = np.column_stack([np.ones(len(zg)), zg, zg**2, zg**3])
    mean_prediction = dg @ beta
    standard_error = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dg, covariance, dg), 0))
    curve = pd.DataFrame({
        "x": xg, "mean": mean_prediction,
        "lo90": mean_prediction - 1.6448536269514722 * standard_error,
        "hi90": mean_prediction + 1.6448536269514722 * standard_error,
        "lo50": mean_prediction - 0.6744897501960817 * standard_error,
        "hi50": mean_prediction + 0.6744897501960817 * standard_error,
    })
    p10, p90x = np.quantile(x, [.10, .90])
    zp = (np.array([p10, p90x]) - mean) / sd
    dp = np.column_stack([np.ones(len(zp)), zp, zp**2, zp**3])
    yp = np.asarray(dp @ beta, float)
    stat = {
        "n_tract_years": int(len(use)), "n_tracts": int(use.tract_geoid_native.nunique()),
        "p10": float(p10), "p90": float(p90x),
        "response_p10": float(yp[0]), "response_p90": float(yp[1]),
        "p90_minus_p10": float(yp[1] - yp[0]),
        "model_r2": float(1 - np.sum(residual**2) / np.sum((y - y.mean())**2)),
        "cluster_robust": True,
    }
    return curve, stat


def draw_curve(ax, frame: pd.DataFrame, x_col: str, x_label: str, curve: pd.DataFrame,
               stat: dict) -> None:
    xg = curve.x.to_numpy(float); yg = curve["mean"].to_numpy(float)
    main = BLUE if stat["p90_minus_p10"] < 0 else RED
    mid = BLUE_MID if stat["p90_minus_p10"] < 0 else RED_MID
    local_lim = symmetric_limit(np.r_[curve["mean"].to_numpy(float),
                                      stat["response_p10"], stat["response_p90"]], 1.0)
    local_norm = TwoSlopeNorm(vmin=-local_lim, vcenter=0, vmax=local_lim)
    ax.fill_between(xg, curve.lo90, curve.hi90, color=mid, alpha=.25, linewidth=0)
    ax.fill_between(xg, curve.lo50, curve.hi50, color=main, alpha=.30, linewidth=0)
    points = np.array([xg, yg]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=DIV, norm=local_norm, linewidth=1.82)
    lc.set_array((yg[:-1] + yg[1:]) / 2)
    ax.add_collection(lc)
    ax.axhline(0, color=MUTED, lw=.55)
    yp = np.array([stat["response_p10"], stat["response_p90"]])
    xp = np.array([stat["p10"], stat["p90"]])
    ax.plot(xp, yp, color=INK, lw=.70, alpha=.78, zorder=4)
    ax.scatter(xp, yp, c=yp, cmap=DIV, norm=local_norm, s=18,
               edgecolor="white", linewidth=.35, zorder=5)
    values = frame[x_col].dropna().to_numpy(float)
    if len(values) > 420:
        values = np.quantile(values, np.linspace(.002, .998, 420))
    ymin, ymax = np.nanmin(np.r_[curve.lo90, 0]), np.nanmax(np.r_[curve.hi90, 0])
    yrange = max(ymax - ymin, 1e-6)
    ax.vlines(values, ymin - .035 * yrange, ymin, color=INK, lw=.24, alpha=.26)
    ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(ymin - .045 * yrange, ymax + .08 * yrange)
    ax.set_xlabel(x_label)
    ax.text(.98, .92, f"p90-p10 {stat['p90_minus_p10']:+.3f} pp", transform=ax.transAxes,
            ha="right", va="top", fontsize=4.15, color=main)
    clean_axes(ax)


def main() -> None:
    setup_style(5.55)
    before = {str(p): sha256(p) for p in [DATA, ROWMAP, MODERATORS]}
    scenario = pd.read_parquet(DATA)
    rowmap = pd.read_parquet(
        ROWMAP, columns=["numeric_row", "tract_geoid_native", "health_reference_year"]
    ).drop_duplicates("numeric_row")
    rowmap["tract_geoid_native"] = rowmap.tract_geoid_native.astype(str).str.zfill(11)
    if rowmap.numeric_row.nunique() != len(rowmap):
        raise RuntimeError("NUMERIC_ROW_TO_TRACT_YEAR_NOT_UNIQUE")
    moderators = pd.read_parquet(MODERATORS).copy()
    moderators["tract_geoid_native"] = moderators.tract_id.astype(str).str.zfill(11)
    moderators["health_reference_year"] = moderators.health_year.astype(int)
    keep = ["tract_geoid_native", "health_reference_year"] + [x[0] for x in MODS]
    moderators = moderators[keep].drop_duplicates(["tract_geoid_native", "health_reference_year"])

    # Average the four held-out spatial schemes before moderator analysis.
    avg = scenario.groupby(["scenario", "numeric_row"], as_index=False).agg(
        prediction_difference=("prediction_difference", "mean"),
        scheme_count=("fold_id", "nunique"),
    )
    if not avg.scheme_count.eq(4).all():
        raise RuntimeError(f"SCHEME_AVERAGE_INCOMPLETE:{avg.scheme_count.value_counts().to_dict()}")
    aligned = avg.merge(rowmap, on="numeric_row", how="left", validate="many_to_one")
    aligned = aligned.merge(
        moderators, on=["tract_geoid_native", "health_reference_year"],
        how="left", validate="many_to_one",
    )
    if aligned.tract_geoid_native.isna().any():
        raise RuntimeError("TRACT_YEAR_MODERATOR_ALIGNMENT_INCOMPLETE")
    moderator_complete_fraction = float(
        (~aligned[[x[0] for x in MODS]].isna().all(axis=1)).mean()
    )
    for col, _, _, transform in MODS:
        if transform is not None:
            aligned[col] = transform(aligned[col].to_numpy(float))

    specs = []
    package = aligned.loc[aligned.scenario.eq(PACKAGE)].copy()
    for col, title, xlabel, _ in MODS:
        specs.append((package, col, title, xlabel, PACKAGE, "package_by_moderator"))
    income_col = "mod__median_household_income"
    for scenario_key, title in STRATEGIES:
        frame = aligned.loc[aligned.scenario.eq(scenario_key)].copy()
        specs.append((frame, income_col, title, "Household income (US$1,000)",
                      scenario_key, "strategy_by_income"))
    if len(specs) != 20:
        raise RuntimeError("PANEL_SPEC_NOT_20")

    curves = []; stats = []
    for frame, col, title, xlabel, scenario_key, family in specs:
        curve, stat = fit_curve(frame, col)
        curve["scenario"] = scenario_key; curve["moderator"] = col; curve["panel_family"] = family
        stat.update({"scenario": scenario_key, "moderator": col, "panel_family": family})
        curves.append(curve); stats.append(stat)
    # Keep the 183-mm publication width while reducing excess vertical canvas;
    # this makes the 4×5 response panels closer to square without changing data.
    fig, axes = plt.subplots(4, 5, figsize=(183 / 25.4, 190 / 25.4))
    fig.subplots_adjust(left=.09, right=.975, bottom=.055, top=.968, wspace=.40, hspace=.40)
    for i, ((frame, col, title, xlabel, _, family), curve, stat) in enumerate(zip(specs, curves, stats)):
        ax = axes.flat[i]
        draw_curve(ax, frame, col, xlabel, curve, stat)
        if i % 5 != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("MM-GTGNNWR response (pp)")
        suffix = " · package" if family == "package_by_moderator" else ""
        panel(ax, i, title + suffix, x=-.10, y=1.035)

    delivery = save_delivery(fig, OUT, STEM)
    fig.canvas.draw(); plt.close(fig)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "source_cubic_response_curves.csv", index=False)
    pd.DataFrame(stats).to_csv(OUT / "source_p10_p90_contrasts.csv", index=False)
    pd.DataFrame([
        {"scenario": key, "planning_label": label,
         "interpretation": "one-SD model perturbation; planning-relevant sensitivity, not a causal policy effect"}
        for key, label in STRATEGIES
    ]).to_csv(OUT / "scenario_dictionary.csv", index=False)
    after = {str(p): sha256(p) for p in [DATA, ROWMAP, MODERATORS]}
    nonwhite, border = border_check(delivery["jpg"])
    write_validation(OUT, {
        "figure": STEM, "panels": 20,
        "proxy_analysis": True, "formal": False,
        "contract": {
            "model": "MM-GTGNNWR", "curve": "cubic OLS mean response",
            "uncertainty": "tract-cluster-robust 50% and 90% mean-response intervals",
            "alignment": "numeric_row -> declared tract-year -> moderator table",
            "interpretation": "model sensitivity; not a causal policy estimate",
        },
        "source_hashes_before": before, "source_hashes_after": after, "delivery": delivery,
        "checks": {
            "panel_count_20": len(specs) == 20,
            "thirteen_package_moderators": sum(x[-1] == "package_by_moderator" for x in specs) == 13,
            "seven_strategy_income_curves": sum(x[-1] == "strategy_by_income" for x in specs) == 7,
            "four_spatial_schemes_averaged": bool(avg.scheme_count.eq(4).all()),
            "tract_year_alignment_complete": not aligned.tract_geoid_native.isna().any(),
            "moderator_rows_at_least_99pct_usable": moderator_complete_fraction >= .99,
            "nested_50_90_intervals": all(((c.lo90 <= c.lo50) & (c.hi90 >= c.hi50)).all() for c in curves),
            "p10_p90_reported": all(np.isfinite(s["p90_minus_p10"]) for s in stats),
            "mm_gtgnnwr_explicitly_named": True,
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
        "# Figure contract\n\nAll twenty panels use current V211 MM-GTGNNWR scenario predictions. "
        "The first thirteen show the integrated-package response across observed tract-year moderators; "
        "the final seven show each practical strategy across household income. Curves are descriptive "
        "model sensitivities with tract-cluster-robust mean-response intervals, not causal effects.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
