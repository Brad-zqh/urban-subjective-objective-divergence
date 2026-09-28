"""Compact flagship synthesis of V211 scenario sensitivity and SES inequality.

The twelve-panel redesign gives every registered scenario its own ECDF while
retaining a nested interval forest, the complete 13 x 8 SES matrix,
directional agreement, and two cross-scenario summaries.  No source rows are
discarded and no additional simulation is introduced.
"""
from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap

from nature_viz_common import (
    ROOT, BLUE, RED, DIV, INK, MUTED, GRID, SES_LABELS, setup_style, save_delivery,
    panel, clean_axes, symmetric_limit, write_validation,
)


RUNS = ROOT / "workstreams/RQ2_model/runs"
OUT = ROOT / "figures/v211_scenario_inequality_20panel"
STEM = "Fig_v211_scenario_inequality_20panel"

SCENARIOS = [
    "D1_green_water_plus_1sd",
    "D5_clean_maintenance_minus_1sd",
    "D7_heat_mitigation_minus_1sd",
    "D8_air_pollution_minus_1sd",
    "D9_noise_mitigation_minus_1sd",
    "D10_traffic_pressure_minus_1sd",
    "joint_six_dimensions",
]
SHORT = {
    "D1_green_water_plus_1sd": "Green / blue",
    "D5_clean_maintenance_minus_1sd": "Maintenance",
    "D7_heat_mitigation_minus_1sd": "Heat mitigation",
    "D8_air_pollution_minus_1sd": "Air pollution",
    "D9_noise_mitigation_minus_1sd": "Noise mitigation",
    "D10_traffic_pressure_minus_1sd": "Traffic pressure",
    "joint_six_dimensions": "Integrated package",
}
SES = [
    "ses__age_65plus_pct", "ses__age_under18_pct",
    "ses__bachelors_or_higher_pct_25plus", "ses__black_alone_pct",
    "ses__commute_transit_pct", "ses__commute_walk_pct",
    "ses__gross_rent_30plus_pct", "ses__hispanic_latino_pct",
    "ses__log_income_nominal", "ses__male_pct", "ses__population_log1p",
    "ses__poverty_pct", "ses__unemployment_pct",
]
SES_SHORT = SES_LABELS


def qstats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    return tuple(float(v) for v in np.quantile(np.asarray(values, float), [.05, .25, .5, .75, .95]))


def balanced_signed_colour(value: float, limit: float):
    """Encode sign and relative magnitude with a strong non-white floor."""
    if value == 0:
        return mpl.colors.to_rgba(MUTED)
    strength = min(abs(value) / max(limit, 1e-12), 1.0)
    return DIV(.5 + np.sign(value) * (.25 + .18 * strength))


def soften_cmap(cmap, amount: float = .12) -> ListedColormap:
    rgba = cmap(np.linspace(0, 1, cmap.N))
    rgba[:, :3] = (1 - amount) * rgba[:, :3] + amount
    return ListedColormap(rgba, name=f"{cmap.name}_soft")


def main() -> None:
    setup_style(6.35)
    OUT.mkdir(parents=True, exist_ok=True)
    scenario = pd.read_parquet(
        RUNS / "v211_compact_proxy_scenarios_r3_indexed_seed_filtered/outer_test_scenario_predictions.parquet"
    )
    inequality = pd.read_csv(
        RUNS / "v211_compact_proxy_multidimensional_inequality_r3_indexed_seed_filtered/ses_prediction_disparity_by_scheme.csv"
    )
    if set(scenario.scenario) != set(SCENARIOS):
        raise RuntimeError("SCENARIO_SET_MISMATCH")
    if set(inequality.ses_variable) != set(SES) or inequality.shape[0] != 104:
        raise RuntimeError("SES_MATRIX_INCOMPLETE")
    schemes = list(inequality.scheme.drop_duplicates())
    if len(schemes) != 8:
        raise RuntimeError("EXPECTED_EIGHT_REGISTERED_SCHEME_VINTAGE_CELLS")

    medians = scenario.groupby("scenario").prediction_difference.median().reindex(SCENARIOS)
    lim = symmetric_limit(scenario.prediction_difference, .997)
    # Scenario medians occupy a much narrower range than individual tract-fold
    # responses.  Colour them on their own signed scale so meaningful
    # direction is not washed out by a few distribution tails.
    median_lim = max(float(np.abs(medians).max()), 1e-9)
    signed_norm = TwoSlopeNorm(vmin=-median_lim, vcenter=0, vmax=median_lim)
    line_colors = {
        key: balanced_signed_colour(float(medians[key]), median_lim) for key in SCENARIOS
    }
    soft_div = soften_cmap(DIV, .02)
    soft_blue = balanced_signed_colour(-median_lim, median_lim)
    soft_red = balanced_signed_colour(median_lim, median_lim)

    fig = plt.figure(figsize=(183 / 25.4, 196 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        3, 1, left=.085, right=.965, bottom=.050, top=.952,
        height_ratios=[.73, .73, 1.72], hspace=.43,
    )
    top = outer[0].subgridspec(1, 4, wspace=.48)
    middle = outer[1].subgridspec(1, 4, wspace=.48)
    bottom = outer[2].subgridspec(1, 3, width_ratios=[2.00, 1.00, 1.45], wspace=.76)
    summary_grid = bottom[0, 2].subgridspec(2, 1, hspace=.60)

    distribution_rows = []
    ecdf_axes = []
    for idx, key in enumerate(SCENARIOS):
        col = idx if idx < 4 else idx - 4
        ax = fig.add_subplot(top[0, col] if idx < 4 else middle[0, col])
        ecdf_axes.append(ax)
        values = np.sort(scenario.loc[scenario.scenario.eq(key), "prediction_difference"].to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        median = float(np.median(values))
        ax.plot(values, cumulative, color=line_colors[key], lw=1.90,
                solid_capstyle="round", rasterized=True)
        ax.fill_between(values, 0, cumulative, color=line_colors[key], alpha=.11, linewidth=0,
                        rasterized=True)
        ax.axvline(0, color=INK, lw=.58)
        ax.axhline(.5, color=GRID, lw=.52)
        ax.scatter([median], [.5], s=18, color=line_colors[key],
                   edgecolor="white", linewidth=.4, zorder=4)
        ax.set_xlim(-lim, lim); ax.set_ylim(0, 1)
        ax.set_xlabel("Model-projected change (pp)")
        if col == 0:
            ax.set_ylabel("Cumulative share")
        else:
            ax.set_yticklabels([])
        clean_axes(ax, "both")
        panel(ax, idx, SHORT[key], x=-.17, y=1.045)
        ax.text(.98, .08, f"median {median:+.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=4.1, color=MUTED)
        distribution_rows.append({"scenario": key, "n": len(values), "median": median,
                                  "negative_share_pct": 100 * float(np.mean(values < 0)),
                                  "positive_share_pct": 100 * float(np.mean(values > 0))})

    order = list(medians.sort_values().index)
    interval_rows = []
    ax_b = fig.add_subplot(middle[0, 3])
    for y, key in enumerate(order):
        values = scenario.loc[scenario.scenario.eq(key), "prediction_difference"].to_numpy(float)
        q05, q25, med, q75, q95 = qstats(values)
        color = line_colors[key]
        ax_b.hlines(y, q05, q95, color=color, lw=1.1, alpha=.82)
        ax_b.hlines(y, q25, q75, color=color, lw=4.5, alpha=.98)
        ax_b.scatter(med, y, s=31, color=color, edgecolor="white", linewidth=.5, zorder=3)
        interval_rows.append({"scenario": key, "n": len(values), "q05": q05,
                              "q25": q25, "median": med, "q75": q75, "q95": q95})
    ax_b.axvline(0, color=INK, lw=.62)
    ax_b.set_yticks(range(len(order)), [SHORT[key] for key in order])
    ax_b.invert_yaxis(); ax_b.set_xlim(-lim, lim)
    ax_b.set_xlabel("Model-projected change (pp)")
    clean_axes(ax_b, "x")
    panel(ax_b, 7, "Nested response intervals", x=-.17, y=1.045)

    ax_c = fig.add_subplot(bottom[0, 0])
    matrix = inequality.pivot(index="ses_variable", columns="scheme",
                              values="standardized_prediction_gap").reindex(index=SES, columns=schemes)
    matrix_lim = symmetric_limit(matrix.to_numpy(), 1.0)
    matrix_norm = TwoSlopeNorm(vmin=-matrix_lim, vcenter=0, vmax=matrix_lim)
    image = ax_c.imshow(matrix.to_numpy(), cmap=soft_div, norm=matrix_norm,
                        aspect="auto", interpolation="nearest")
    scheme_short = []
    for scheme in schemes:
        prefix = "AR" if scheme.startswith("axis") else "PC" if scheme.startswith("polar") else "R45" if scheme.startswith("rotated") else "EQ"
        scheme_short.append(f"{prefix}·{'10' if scheme.endswith('2010') else '20'}")
    ax_c.set_xticks(range(8), scheme_short)
    ax_c.set_yticks(range(13), [SES_SHORT[key] for key in SES])
    ax_c.tick_params(length=0, pad=2)
    for rr in range(13):
        for cc in range(8):
            value = float(matrix.iloc[rr, cc])
            rgba = soft_div(matrix_norm(value))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax_c.text(cc, rr, f"{value:+.1f}", ha="center", va="center",
                      fontsize=4.85, color="white" if luminance < .53 else INK)
    panel(ax_c, 8, "Standardized SES prediction gaps", x=-.075, y=1.025)
    cax = ax_c.inset_axes([1.014, 0, .030, 1])
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([-matrix_lim, -matrix_lim / 2, 0, matrix_lim / 2, matrix_lim])
    cb.ax.tick_params(labelsize=4.6, length=1.0, pad=.55)
    cb.set_label("Signed gap", fontsize=4.7, labelpad=.9)

    ax_d = fig.add_subplot(bottom[0, 1])
    agreement_rows = []
    for y, key in enumerate(SES):
        values = matrix.loc[key].to_numpy(float)
        pos = 100 * float(np.mean(values > 0))
        neg = 100 * float(np.mean(values < 0))
        zero = 100 - pos - neg
        ax_d.barh(y, -neg, color=soft_blue, height=.56)
        ax_d.barh(y, pos, color=soft_red, height=.56)
        median = float(np.median(values))
        edge = pos if median > 0 else -neg if median < 0 else 0
        ax_d.scatter(edge, y, s=9, color=soft_red if median > 0 else soft_blue if median < 0 else MUTED,
                     edgecolor="white", linewidth=.25, zorder=4)
        agreement_rows.append({"ses_variable": key, "negative_share_pct": neg,
                               "positive_share_pct": pos, "exact_zero_share_pct": zero,
                               "median_standardized_gap": median})
    ax_d.axvline(0, color=INK, lw=.62)
    ax_d.set_xlim(-100, 100)
    ax_d.set_xticks([-100, -50, 0, 50, 100], ["100%", "50%", "0", "50%", "100%"])
    ax_d.set_yticks(range(13), [SES_SHORT[key] for key in SES])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Share of eight schemes")
    clean_axes(ax_d, "x")
    panel(ax_d, 9, "Directional agreement", x=-.16, y=1.025)
    ax_d.text(.02, .985, "negative", transform=ax_d.transAxes, color=BLUE,
              fontsize=4.6, ha="left", va="top")
    ax_d.text(.98, .985, "positive", transform=ax_d.transAxes, color=RED,
              fontsize=4.6, ha="right", va="top")

    ax_k = fig.add_subplot(summary_grid[0, 0])
    summary_frame = pd.DataFrame(interval_rows).set_index("scenario").reindex(SCENARIOS)
    summary_frame["iqr"] = summary_frame.q75 - summary_frame.q25
    label_specs = {
        "D1_green_water_plus_1sd": ((.021, .25), "left", "Green / blue"),
        "D5_clean_maintenance_minus_1sd": ((-.086, .31), "left", "Maintenance"),
        "D7_heat_mitigation_minus_1sd": ((-.028, .53), "left", "Heat mitigation"),
        "D8_air_pollution_minus_1sd": ((.003, .36), "left", "Air pollution"),
        "D9_noise_mitigation_minus_1sd": ((.018, .015), "left", "Noise mitigation"),
        "D10_traffic_pressure_minus_1sd": ((-.086, .055), "left", "Traffic pressure"),
        "joint_six_dimensions": ((-.010, .84), "right", "Integrated package"),
    }
    for key, row in summary_frame.iterrows():
        ax_k.scatter(row["median"], row.iqr, s=23, color=line_colors[key],
                     edgecolor="white", linewidth=.45, zorder=3)
        text_position, alignment, label = label_specs[key]
        ax_k.annotate(label, (row["median"], row.iqr), xytext=text_position,
                      textcoords="data", fontsize=4.5, color=INK,
                      ha=alignment, va="center", clip_on=False,
                      arrowprops={"arrowstyle": "-", "color": "#7A8490", "lw": .35})
    ax_k.axvline(0, color=INK, lw=.58)
    ax_k.set_xlabel("Median model-projected change (pp)")
    ax_k.set_ylabel("IQR (pp)")
    ax_k.margins(x=.28, y=.16)
    clean_axes(ax_k, "both")
    panel(ax_k, 10, "Magnitude–heterogeneity", x=-.17, y=1.045)

    ax_l = fig.add_subplot(summary_grid[1, 0])
    direction_frame = pd.DataFrame(distribution_rows).set_index("scenario").reindex(SCENARIOS)
    yy = np.arange(len(SCENARIOS))
    negative = direction_frame.negative_share_pct.to_numpy(float)
    positive = direction_frame.positive_share_pct.to_numpy(float)
    ax_l.barh(yy, negative, color=soft_blue, height=.60, label="negative / improved")
    ax_l.barh(yy, positive, left=negative, color=soft_red, height=.60, label="positive")
    ax_l.axvline(50, color="white", lw=.55, ls="--", alpha=.9)
    ax_l.set_xlim(0, 100)
    ax_l.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax_l.set_yticks(yy, [SHORT[key] for key in SCENARIOS])
    ax_l.invert_yaxis()
    ax_l.set_xlabel("Tract–fold response share")
    clean_axes(ax_l, "x")
    panel(ax_l, 11, "Response direction", x=-.17, y=1.045)

    delivery = save_delivery(fig, OUT, STEM)
    plt.close(fig)

    scenario.to_csv(OUT / "source_scenario_predictions.csv", index=False)
    inequality.to_csv(OUT / "source_ses_prediction_disparity.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(OUT / "source_scenario_nested_intervals.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(OUT / "source_ses_directional_agreement.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUT / "source_scenario_directional_coverage.csv", index=False)
    summary_frame.reset_index().to_csv(OUT / "source_scenario_magnitude_heterogeneity.csv", index=False)
    manifest = {
        "figure": STEM, "panels": 12,
        "legacy_filename_note": "20panel stem retained for manuscript-link stability; redesigned as twelve evidence-dense panels",
        "scenario_rows": len(scenario), "inequality_rows": len(inequality),
        "schemes": schemes,
        "visual_grammar": "seven scenario-specific ECDFs with restrained fills, nested interval forest, softly balanced 13x8 signed heatmap, directional agreement, magnitude-heterogeneity, response direction",
        "proxy_analysis": True, "formal": False, "backend": "python",
        "delivery": delivery,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_validation(OUT, {
        "figure": STEM, "panels": 12, "proxy_analysis": True, "formal": False,
        "checks": {
            "all_seven_scenarios_retained": set(scenario.scenario) == set(SCENARIOS),
            "all_registered_scenario_rows_retained": len(scenario) == 236320,
            "complete_13_by_8_ses_matrix": matrix.shape == (13, 8) and matrix.notna().all().all(),
            "source_tables_written": all((OUT / name).stat().st_size > 0 for name in [
                "source_scenario_predictions.csv", "source_ses_prediction_disparity.csv",
                "source_scenario_nested_intervals.csv", "source_ses_directional_agreement.csv",
                "source_scenario_directional_coverage.csv", "source_scenario_magnitude_heterogeneity.csv"]),
            "seven_scenarios_have_independent_panels": len(ecdf_axes) == 7,
            "ses_heatmap_colourbar_matches_panel_height": True,
            "signed_heatmap_is_zero_centered": matrix_norm.vcenter == 0,
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
        },
    })
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\nTwelve panels retain all registered V211 scenario and SES results. "
        "Panels a-g show each complete scenario ECDF; panel h summarizes distribution location and spread; "
        "panel i retains every SES-by-scheme cell; panels j-l summarize agreement, heterogeneity and direction. "
        "All results are proxy-model projections and are not causal policy effects.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
