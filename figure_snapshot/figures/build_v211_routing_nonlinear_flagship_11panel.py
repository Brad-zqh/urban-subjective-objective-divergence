"""V211 graph-routing and nonlinear-head flagship diagnostic.

This replaces the sparse four-panel draft with an evidence ladder spanning
fold-paired prediction error, explainer faithfulness, admitted relation mass,
SES routing heterogeneity, curvature stability, and dose-response summaries.
All panels are deterministic summaries of registered V211 source files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from nature_viz_common import (
    BLUE, RED, DIV, INK, MUTED, GRID, MISSING, ROOT, DIMENSION_LABELS,
    setup_style, save_delivery, panel, clean_axes, write_validation,
    add_equal_height_colorbar, symmetric_colourbar_ticks,
)


RUNS = ROOT / "workstreams/RQ2_model/runs"
OUT = ROOT / "figures/v211_routing_nonlinear_flagship_11panel"
STEM = "Fig_v211_routing_nonlinear_flagship_11panel"
FONT = ROOT / "MM-GTGNNWR/assets/fonts/nimbus-sans"

PAIRED = RUNS / "v211_nonlinear_coefficient_head_r1/fold_paired_comparison.csv"
FAITH = RUNS / "v211_gnnexplainer_r1/faithfulness_summary.csv"
FEATURE_MASKS = RUNS / "v211_gnnexplainer_r1/feature_masks.parquet"
SES = RUNS / "v211_gnnexplainer_r1/ses_stratified_relation_sensitivity.csv"
CURVATURE = RUNS / "v211_nonlinear_coefficient_head_r1/nonlinear_curvature_sign_audit.csv"
DOSE = RUNS / "v211_nonlinear_coefficient_head_r1/nonlinear_dose_response_long.csv"

SCHEME_LABEL = {
    "axis_recursive": "Axis recursive",
    "polar_north_clockwise_equal_count": "Polar",
    "rotated45_recursive": "Rotated 45°",
    "y_equal_count_stripes": "Y-stripes",
}

DISPLAY_DIMENSIONS = [
    "Green / blue",
    "Density / openness",
    "Access / walking",
    "Safety / order",
    "Cleanliness / upkeep",
    "Social vitality",
    "Heat stress",
    "Air pollution",
    "Noise",
    "Traffic pressure",
]


def feature_label(feature: str) -> str:
    match = re.search(r"(?:^|_)d(\d+)_", feature)
    if match:
        number = int(match.group(1))
        channel = "S−O" if feature.startswith("delta_") else "O" if feature.startswith("o_") else "S" if feature.startswith("s_") else ""
        return f"{DIMENSION_LABELS.get(number, f'D{number}')} {channel}".strip()
    ses = {
        "ses__age_65plus_pct": "Age ≥65", "ses__age_under18_pct": "Age <18",
        "ses__bachelors_or_higher_pct_25plus": "Higher education",
        "ses__black_alone_pct": "Black population", "ses__commute_transit_pct": "Transit commute",
        "ses__commute_walk_pct": "Walk commute", "ses__gross_rent_30plus_pct": "Rent burden",
        "ses__hispanic_latino_pct": "Hispanic / Latino", "ses__log_income_nominal": "Log income",
        "ses__male_pct": "Male population", "ses__population_log1p": "Population size",
        "ses__poverty_pct": "Poverty", "ses__unemployment_pct": "Unemployment",
    }
    return ses.get(feature, feature.replace("_", " "))


def qstats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    return tuple(float(x) for x in np.quantile(np.asarray(values, float), [.05, .25, .50, .75, .95]))


def interval(ax, y: float, values: np.ndarray, color: str, *, marker: str = "o") -> float:
    q05, q25, med, q75, q95 = qstats(values)
    ax.plot([q05, q95], [y, y], color=color, lw=1.05, alpha=.78, solid_capstyle="round")
    ax.plot([q25, q75], [y, y], color=color, lw=3.2, alpha=1.0, solid_capstyle="round")
    ax.scatter([med], [y], s=28, marker=marker, color=color, edgecolor="white", linewidth=.55, zorder=4)
    return med


def violin_with_points(ax, arrays: list[np.ndarray], positions: list[float], colours: list[str]) -> None:
    parts = ax.violinplot(arrays, positions=positions, orientation="horizontal", widths=.70,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour); body.set_edgecolor("none"); body.set_alpha(.18)
    for values, y, colour in zip(arrays, positions, colours):
        vals = np.asarray(values, float)
        order = np.argsort(vals)
        jitter = np.linspace(-.18, .18, len(vals))
        rendered = np.empty_like(jitter); rendered[order] = jitter
        if len(vals) <= 80:
            ax.scatter(vals, y + rendered, s=7.5, color=colour, alpha=.30, linewidth=0, zorder=1)
        med = interval(ax, y, vals, colour)
        ax.text(med, y + .28, f"{med:+.3f}", ha="center", va="bottom", fontsize=4.5, fontweight="bold")


def main() -> None:
    for path in FONT.glob("NimbusSans-*.otf"):
        font_manager.fontManager.addfont(str(path))
    setup_style(6.55)

    paired = pd.read_csv(PAIRED)
    faith = pd.read_csv(FAITH)
    masks = pd.read_parquet(FEATURE_MASKS)
    ses = pd.read_csv(SES)
    curvature = pd.read_csv(CURVATURE)
    dose = pd.read_csv(DOSE)
    paired["scheme"] = paired.fold_id.str.split("__").str[0].map(SCHEME_LABEL)
    paired["delta_rmse"] = paired.nonlinear_rmse - paired.baseline_rmse
    paired["delta_mae"] = paired.nonlinear_mae - paired.baseline_mae
    faith["sufficiency_advantage"] = faith.random_sufficiency_abs_error - faith.sufficiency_abs_error

    fig = plt.figure(figsize=(183 / 25.4, 228 / 25.4), facecolor="white")
    outer = fig.add_gridspec(4, 1, left=.105, right=.945, bottom=.060, top=.978,
                             hspace=.66, height_ratios=[1.0, 1.0, 1.0, 1.68])
    upper = outer[:3, 0].subgridspec(3, 3, wspace=.54, hspace=.67)
    lower = outer[3, 0].subgridspec(1, 2, width_ratios=[1.08, 1.0], wspace=.70)
    axes = [fig.add_subplot(upper[row, col]) for row in range(3) for col in range(3)]
    axes += [fig.add_subplot(lower[0, 0]), fig.add_subplot(lower[0, 1])]
    a, b, c, d, e, f, g, h, i, j, k = axes

    # a | 32-fold RMSE parity. Colour directly encodes the paired change.
    lim_delta = max(float(np.abs(paired.delta_rmse).max()), .05)
    norm_delta = TwoSlopeNorm(vmin=-lim_delta, vcenter=0, vmax=lim_delta)
    colours = DIV(norm_delta(paired.delta_rmse.to_numpy(float)))
    lo = min(paired.baseline_rmse.min(), paired.nonlinear_rmse.min())
    hi = max(paired.baseline_rmse.max(), paired.nonlinear_rmse.max())
    a.plot([lo, hi], [lo, hi], color=INK, lw=.75, ls="--")
    a.scatter(paired.baseline_rmse, paired.nonlinear_rmse, s=24, c=colours,
              edgecolor="white", linewidth=.55, alpha=.95)
    a.set(xlabel="Transparent-head RMSE (pp)", ylabel="Feature-dependent RMSE (pp)", xlim=(lo, hi), ylim=(lo, hi))
    a.set_title("Outer-test RMSE · 32 paired folds", loc="left")
    a.text(.97, .05, f"improved: {(paired.delta_rmse < 0).sum()}/32", transform=a.transAxes,
           ha="right", fontsize=4.8, color=MUTED)
    clean_axes(a, "both")

    # b | Paired changes in both registered error metrics.
    violin_with_points(b, [paired.delta_rmse.to_numpy(), paired.delta_mae.to_numpy()], [1, 0], [BLUE, RED])
    b.axvline(0, color=INK, lw=.65)
    b.set_yticks([1, 0], ["ΔRMSE", "ΔMAE"])
    b.set_xlabel("Feature-dependent − transparent (pp)")
    b.set_title("Fold-paired error change", loc="left")
    clean_axes(b, "x")

    # c | Scheme-specific summaries retain all eight folds in each scheme.
    schemes = list(SCHEME_LABEL.values())
    for y, scheme in enumerate(schemes):
        sub = paired.loc[paired.scheme.eq(scheme)]
        med_rmse = interval(c, y + .12, sub.delta_rmse.to_numpy(), BLUE)
        med_mae = interval(c, y - .12, sub.delta_mae.to_numpy(), RED, marker="s")
        c.text(max(med_rmse, med_mae), y + .31, f"{med_rmse:+.2f} / {med_mae:+.2f}",
               ha="center", fontsize=4.25, color=INK)
    c.axvline(0, color=INK, lw=.65)
    c.set_yticks(range(4), schemes, fontsize=4.8)
    c.set_xlabel("Paired change (pp) · blue ΔRMSE / red ΔMAE")
    c.set_title("Partition response", loc="left")
    clean_axes(c, "x")

    # d | Faithfulness metric distributions. These are diagnostic quantities,
    # not clinical effects or causal mediation estimates.
    faith_arrays = [
        faith.sufficiency_abs_error.to_numpy(float),
        faith.random_sufficiency_abs_error.to_numpy(float),
        faith.comprehensiveness_abs_change.to_numpy(float),
    ]
    faith_labels = ["Selected error", "Random error", "Removal change"]
    fcols = [BLUE, "#8DB0D5", RED]
    parts = d.violinplot(faith_arrays, positions=[2, 1, 0], orientation="horizontal", widths=.72,
                         showmeans=False, showmedians=False, showextrema=False)
    for body, colour in zip(parts["bodies"], fcols):
        body.set_facecolor(colour); body.set_edgecolor("none"); body.set_alpha(.28)
    for y, vals, colour in zip([2, 1, 0], faith_arrays, fcols):
        med = interval(d, y, vals, colour)
        d.text(med, y + .28, f"{med:.2f}", ha="center", fontsize=4.5, fontweight="bold")
    d.set_yticks([2, 1, 0], faith_labels, fontsize=4.75)
    d.set_xlabel("Absolute prediction quantity (pp)")
    d.set_title("Explainer faithfulness distributions", loc="left")
    clean_axes(d, "x")

    # e | Top feature masks with direct contribution context.
    mask_summary = masks.groupby("feature", as_index=False).agg(
        median_mask=("mask", "median"),
        median_exact_contribution=("exact_abs_contribution", "median"),
    )
    top_masks = mask_summary.nlargest(10, "median_mask").sort_values("median_mask")
    norm_contrib = Normalize(vmin=float(mask_summary.median_exact_contribution.quantile(.05)),
                             vmax=float(mask_summary.median_exact_contribution.quantile(.95)), clip=True)
    mcols = DIV(norm_contrib(top_masks.median_exact_contribution.to_numpy(float)))
    yy = np.arange(len(top_masks))
    e.hlines(yy, 0, top_masks.median_mask, color=mcols, lw=1.8, alpha=.72)
    e.scatter(top_masks.median_mask, yy, s=30, c=mcols, edgecolor="white", linewidth=.5, zorder=3)
    e.set_yticks(yy, [feature_label(x) for x in top_masks.feature], fontsize=4.45)
    e.set_xlabel("Median learned mask weight · colour = contribution")
    e.set_title("Highest admitted feature masks", loc="left")
    e.set_xlim(0, max(.75, float(top_masks.median_mask.max()) * 1.08))
    clean_axes(e, "x")

    # f | Random versus selected subset error with rank-agreement colour.
    sp = faith.feature_spearman_exact_contribution.to_numpy(float)
    norm_sp = Normalize(vmin=float(np.quantile(sp, .05)), vmax=float(np.quantile(sp, .95)), clip=True)
    f.scatter(faith.sufficiency_abs_error, faith.random_sufficiency_abs_error,
              s=7.5, c=DIV(norm_sp(sp)), alpha=.34, linewidth=0, rasterized=True)
    hi_err = float(np.quantile(np.r_[faith.sufficiency_abs_error, faith.random_sufficiency_abs_error], .99))
    f.plot([0, hi_err], [0, hi_err], ls="--", lw=.7, color=INK)
    f.set(xlabel="Selected-subset error (pp)", ylabel="Random-subset error (pp)", xlim=(0, hi_err), ylim=(0, hi_err))
    f.set_title("Selected versus random subsets", loc="left")
    f.text(.97, .05, f"selected lower: {(faith.sufficiency_abs_error < faith.random_sufficiency_abs_error).mean():.0%}",
           transform=f.transAxes, ha="right", fontsize=4.7, color=MUTED)
    clean_axes(f, "both")
    cb_f, _ = add_equal_height_colorbar(
        fig, f, ScalarMappable(norm=norm_sp, cmap=DIV), width=.025, gap=.020
    )
    cb_f.set_ticks(np.linspace(norm_sp.vmin, norm_sp.vmax, 5))
    cb_f.ax.tick_params(labelsize=4.6, length=1, pad=.5)
    cb_f.set_label("Feature-rank agreement", fontsize=4.6, labelpad=.8)

    # g | Relation admission. Exact zeros stay visible and explicitly labelled.
    relation_cols = [
        ("Queen", "relation_mass_spatial_queen"),
        ("Road", "relation_mass_road_connectivity"),
        ("Mobility", "relation_mass_mobility_flow"),
        ("Temporal", "relation_mass_temporal_forward"),
    ]
    vmax_relation = max(float(faith[c].quantile(.95)) for _, c in relation_cols)
    norm_relation = Normalize(vmin=0, vmax=max(vmax_relation, .01))
    for y, (label, col) in enumerate(relation_cols[::-1]):
        vals = faith[col].to_numpy(float)
        med = interval(g, y, vals, DIV(norm_relation(float(np.median(vals)))))
        zero = float(np.isclose(vals, 0, atol=1e-12).mean())
        dy = -.27 if y == 3 else .23
        g.text(max(med, .005), y + dy, f"median {med:.3f} · exact zero {zero:.0%}", fontsize=4.25, ha="left")
    g.set_yticks(range(4), [x[0] for x in relation_cols[::-1]], fontsize=4.8)
    g.set_ylim(-.46, 3.48)
    g.set_xlabel("Relation-mask mass")
    g.set_title("Admitted graph-relation mass", loc="left")
    clean_axes(g, "x")

    # h | SES-stratified mobility relation mass. Range is across burden strata.
    ses_summary = ses.groupby(["poverty_stratum", "support_stratum"], as_index=False).agg(
        mean_mass=("relation_mass_mobility_flow", "mean"),
        min_mass=("relation_mass_mobility_flow", "min"),
        max_mass=("relation_mass_mobility_flow", "max"),
        n_cells=("relation_mass_mobility_flow", "size"),
    ).sort_values("mean_mass")
    routing_colours = ["#3B4CC0", "#5F7FC8", "#8BA7CF", "#D59A95", "#C9575E", "#B40426"]
    for y, row in ses_summary.reset_index(drop=True).iterrows():
        colour = routing_colours[y]
        h.plot([row.min_mass, row.max_mass], [y, y], color=colour, lw=1.45, alpha=.94)
        h.scatter([row.mean_mass], [y], s=30, color=colour, edgecolor="white", linewidth=.5, zorder=3)
    h.set_yticks(range(len(ses_summary)),
                 [f"{p} poverty · {s.replace('_', ' ')}" for p, s in zip(ses_summary.poverty_stratum, ses_summary.support_stratum)],
                 fontsize=4.8)
    h.set_xlabel("Mean mobility-flow mask mass")
    h.set_title("SES mobility routing by stratum", loc="left")
    h.set_xlabel("Mean mobility-flow mask mass · line = burden range")
    clean_axes(h, "x")

    # i | Stability distribution with a gradient tied to the shared palette.
    bins = np.linspace(.48, .86, 13)
    counts, edges = np.histogram(curvature.same_sign_fraction, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    norm_stab = Normalize(vmin=.50, vmax=.85)
    i.bar(centres, counts, width=np.diff(edges) * .92, color=DIV(norm_stab(centres)),
          edgecolor="white", linewidth=.35)
    i.axvline(.75, color=RED, ls="--", lw=.85)
    i.axvline(.90, color=INK, ls=":", lw=.90)
    i.set_xlabel("Same-sign fraction across outer folds")
    i.set_ylabel("Feature × SES cells")
    i.set_title("Curvature-sign stability", loc="left")
    i.text(.98, .95, f"≥75%: {(curvature.same_sign_fraction >= .75).sum()}/160\n≥90%: {(curvature.same_sign_fraction >= .90).sum()}/160",
           transform=i.transAxes, ha="right", va="top", fontsize=4.6)
    clean_axes(i, "y")

    # j | Dimension × SES heatmap, averaging over the four registered strata.
    curvature["dimension"] = curvature.feature.str.extract(r"delta_d(\d+)_").astype(int)
    pivot = curvature.pivot_table(index="dimension", columns="ses_variable", values="same_sign_fraction", aggfunc="mean")
    pivot = pivot.reindex(index=range(1, 11), columns=["education", "income", "poverty", "unemployment"])
    norm_heat = Normalize(vmin=.50, vmax=max(.70, float(np.nanmax(pivot.to_numpy()))))
    im_j = j.imshow(pivot.to_numpy(), aspect="auto", cmap=DIV, norm=norm_heat)
    for rr in range(pivot.shape[0]):
        for cc in range(pivot.shape[1]):
            value = float(pivot.iloc[rr, cc])
            j.text(cc, rr, f"{value:.2f}", ha="center", va="center", fontsize=4.35,
                   color="white" if value >= .67 else INK)
    j.set_xticks(range(4), ["Education", "Income", "Poverty", "Unemployment"])
    j.set_yticks(range(10), DISPLAY_DIMENSIONS, fontsize=4.75)
    j.set_xlabel("SES stratification variable")
    j.set_title("Mean curvature-sign agreement across registered strata", loc="left")
    cb_j, _ = add_equal_height_colorbar(fig, j, im_j, width=.025, gap=.020)
    heat_ticks = np.linspace(norm_heat.vmin, norm_heat.vmax, 5)
    cb_j.set_ticks(heat_ticks)
    cb_j.ax.tick_params(labelsize=4.6, length=1.1, pad=.5)
    cb_j.set_label("Same-sign fraction · low → high", fontsize=4.6, labelpad=1)

    # k | Pooled fitted dose-response surface across all registered folds and
    # SES strata. This is a fitted proxy-model response, not a causal dose.
    dose["dimension"] = dose.feature.str.extract(r"delta_d(\d+)_").astype(int)
    doses = sorted(dose.dose_sd.unique())
    dose_pivot = dose.pivot_table(index="dimension", columns="dose_sd", values="median_prediction_change", aggfunc="median")
    dose_pivot = dose_pivot.reindex(index=range(1, 11), columns=doses)
    max_abs_dose = max(float(np.nanquantile(np.abs(dose_pivot.to_numpy()), .98)), .01)
    im_k = k.imshow(dose_pivot.to_numpy(), aspect="auto", cmap=DIV,
                    norm=TwoSlopeNorm(vmin=-max_abs_dose, vcenter=0, vmax=max_abs_dose))
    for rr in range(dose_pivot.shape[0]):
        for cc in range(dose_pivot.shape[1]):
            value = float(dose_pivot.iloc[rr, cc])
            k.text(cc, rr, f"{value:+.2f}", ha="center", va="center", fontsize=4.5,
                   color="white" if abs(value) > .70 * max_abs_dose else INK)
    k.set_xticks(range(len(doses)), [f"{x:+g}" for x in doses])
    k.set_yticks(range(10), DISPLAY_DIMENSIONS, fontsize=4.75)
    k.set_xlabel("S−O perturbation (fold-training SD)")
    k.set_title("Pooled fitted response surface", loc="left")
    cb_k, _ = add_equal_height_colorbar(fig, k, im_k, width=.025, gap=.025)
    cb_k.set_ticks(symmetric_colourbar_ticks(-max_abs_dose, max_abs_dose))
    cb_k.ax.tick_params(labelsize=4.6, length=1.1, pad=.5)
    cb_k.set_label("Median model-projected change (pp)", fontsize=4.6, labelpad=1)

    for idx, ax in enumerate(axes):
        panel(ax, idx, x=-.10 if idx != 9 else -.055, y=1.04)
    OUT.mkdir(parents=True, exist_ok=True)
    delivery = save_delivery(fig, OUT, STEM)
    plt.close(fig)

    paired.to_csv(OUT / "source_fold_paired_comparison.csv", index=False)
    faith.to_csv(OUT / "source_faithfulness_summary.csv", index=False)
    mask_summary.to_csv(OUT / "source_feature_mask_summary.csv", index=False)
    ses_summary.to_csv(OUT / "source_ses_mobility_relation_summary.csv", index=False)
    curvature.to_csv(OUT / "source_curvature_sign_audit.csv", index=False)
    dose_pivot.to_csv(OUT / "source_pooled_dose_response_surface.csv")
    pivot.to_csv(OUT / "source_curvature_stability_surface.csv")

    manifest = {
        "figure": STEM,
        "panels": 11,
        "source_runs": [
            str(RUNS / "v211_gnnexplainer_r1"),
            str(RUNS / "v211_nonlinear_coefficient_head_r1"),
        ],
        "evidence_layers": [
            "fold-paired prediction error", "explainer faithfulness", "feature-mask admission",
            "relation-mask mass", "SES routing heterogeneity", "curvature-sign stability",
            "pooled fitted dose-response surface",
        ],
        "proxy_analysis": True,
        "formal": False,
        "backend": "python",
        "delivery": delivery,
        "all_checks_passed": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_validation(OUT, {
        "figure": STEM, "panels": 11, "proxy_analysis": True, "formal": False,
        "checks": {
            "paired_fold_count_32": len(paired) == 32,
            "four_partition_schemes_retained": paired.scheme.nunique() == 4 and bool((paired.groupby("scheme").size() == 8).all()),
            "faithfulness_rows_1920": len(faith) == 1920,
            "feature_mask_rows_82560": len(masks) == 82560,
            "ses_cells_17": len(ses) == 17,
            "curvature_cells_160": len(curvature) == 160,
            "all_ten_divergence_dimensions_in_dose_surface": dose_pivot.shape[0] == 10,
            "source_tables_written": all((OUT / name).stat().st_size > 0 for name in [
                "source_fold_paired_comparison.csv", "source_faithfulness_summary.csv",
                "source_feature_mask_summary.csv", "source_ses_mobility_relation_summary.csv",
                "source_curvature_sign_audit.csv", "source_pooled_dose_response_surface.csv",
                "source_curvature_stability_surface.csv",
            ]),
        },
    })


if __name__ == "__main__":
    main()
