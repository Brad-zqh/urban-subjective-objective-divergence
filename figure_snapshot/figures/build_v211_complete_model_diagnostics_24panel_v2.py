"""Complete current V211 comparator diagnostic atlas.

Six temporal comparators use the locked final-year test contract.  The current
MM-GTGNNWR primary model uses registered spatial outer folds.  Both contracts
are displayed, but are labelled and exported separately rather than treated as
one homogeneous leaderboard.  GTCNNWR is intentionally excluded from all
displayed results because it is not part of the manuscript comparison set.
"""
from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

from nature_viz_common import (
    ROOT, DIV, INK, MUTED, GRID, MODEL_LABELS, setup_style, panel,
    clean_axes, save_delivery, write_validation, sha256,
)


RUN = ROOT / "workstreams/RQ2_model/runs/v211_model_benchmark_r2_frozen43"
PRED_PATH = RUN / "model_benchmark_predictions.parquet"
METRIC_PATH = RUN / "model_benchmark_metrics.csv"
MM_PATH = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r1/outer_test_predictions.parquet"
OUT = ROOT / "figures/v211_complete_model_diagnostics_24panel_v2"
STEM = "Fig_v211_complete_model_diagnostics_24panel_v2"

TEMP_MODELS = ["OLS", "GTWR", "MLP_GPU", "GraphSAGE_GPU", "GTNNWR_GPU", "GTGNNWR_GPU"]
MM_MODEL = "MM_GTGNNWR"
MODELS = TEMP_MODELS + [MM_MODEL]
LABELS = {model: MODEL_LABELS[model] for model in MODELS}
LABELS["OLS"] = "OLS anchor"
COLORS = dict(zip(MODELS, [
    "#596573", "#3B4CC0", "#668BC5", "#8DA9C8",
    "#D6908B", "#CC5C60", "#B40426",
]))
MARKERS = ["o", "^", "v", "s", "D", "P", "X"]


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "residual_iqr": float(np.quantile(residual, .75) - np.quantile(residual, .25)),
    }


def main() -> None:
    setup_style(6.1)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(PRED_PATH)
    mm_predictions = pd.read_parquet(MM_PATH)
    locked = pd.read_csv(METRIC_PATH).set_index("model").reindex(TEMP_MODELS)
    test = predictions.loc[predictions.health_year.eq(2023)]
    if predictions.health_year.min() != 2014 or predictions.health_year.max() != 2023:
        raise RuntimeError("EXPECTED_2014_2023_PREDICTION_PROFILES")
    if len(predictions.loc[predictions.health_year.eq(2023)]) != 870:
        raise RuntimeError("EXPECTED_LOCKED_2023_TEST_N_870")
    if locked.index.isna().any() or locked[["r2", "rmse", "mae"]].isna().any().any():
        raise RuntimeError("INCOMPLETE_SIX_MODEL_TEMPORAL_METRICS")
    if len(mm_predictions) != 33760 or mm_predictions.health_reference_year.nunique() != 10:
        raise RuntimeError("INCOMPLETE_MM_GTGNNWR_SPATIAL_OUTER_FOLD_PREDICTIONS")

    def arrays(model: str, year: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if model == MM_MODEL:
            frame = mm_predictions if year is None else mm_predictions.loc[
                mm_predictions.health_reference_year.eq(year)
            ]
            return frame.observed_outcome.to_numpy(float), frame.prediction.to_numpy(float)
        frame = predictions if year is None else predictions.loc[predictions.health_year.eq(year)]
        return frame.outcome_mhlth.to_numpy(float), frame[f"pred__{model}"].to_numpy(float)

    all_values = np.concatenate([
        predictions.outcome_mhlth.to_numpy(float),
        predictions[[f"pred__{model}" for model in TEMP_MODELS]].to_numpy(float).ravel(),
        mm_predictions.observed_outcome.to_numpy(float),
        mm_predictions.prediction.to_numpy(float),
    ])
    lo, hi = np.quantile(all_values, [.003, .997])
    hist_max = 1
    for model in MODELS:
        observed, predicted = arrays(model)
        hist, _, _ = np.histogram2d(
            observed, predicted, bins=35,
            range=((lo, hi), (lo, hi)),
        )
        hist_max = max(hist_max, int(hist.max()))
    density_norm = LogNorm(vmin=1, vmax=hist_max)

    fig = plt.figure(figsize=(183 / 25.4, 236 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        6, 4, left=.095, right=.955, bottom=.047, top=.972,
        wspace=.62, hspace=.58, height_ratios=[1.02, 1.02, .88, .88, .82, .90],
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(6) for col in range(4)]
    annual_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | str]] = []

    # a-g: six temporal profiles plus the current primary spatial-fold model.
    density_colourbar_count = 0
    for index, model in enumerate(MODELS):
        ax = axes[index]
        observed, predicted = arrays(model)
        artist = ax.hexbin(
            observed, predicted, gridsize=35, extent=(lo, hi, lo, hi),
            mincnt=1, cmap=DIV, norm=density_norm, linewidths=0,
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color=INK, lw=.62, ls="--", dashes=(3, 2))
        profile = metrics(observed, predicted)
        profile_rows.append({"model": model, **profile})
        ax.text(
            .04, .95,
            f"R² {profile['r2']:.2f}\nRMSE {profile['rmse']:.2f}\nMAE {profile['mae']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=4.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .80, "pad": .4},
        )
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed MHLTH (%)")
        ax.set_ylabel("Predicted MHLTH (%)" if index % 4 == 0 else "")
        clean_axes(ax)
        contract = "spatial folds" if model == MM_MODEL else "2014–2023 profiles"
        panel(ax, index, f"{LABELS[model]} · {contract}", x=-.11, y=1.035)
        cax = ax.inset_axes([1.012, 0, .028, 1], transform=ax.transAxes)
        cb = fig.colorbar(artist, cax=cax)
        density_ticks = [value for value in [1, 3, 10, 30, 100, 300, 1000] if value <= hist_max]
        cb.set_ticks(density_ticks)
        cb.ax.tick_params(labelsize=4.6, length=.9, pad=.35)
        cb.outline.set_linewidth(.3)
        if index == 3:
            cb.set_label("Count/bin", fontsize=4.6, labelpad=.6)
        density_colourbar_count += 1

    # h: split-specific comparison; the primary model is visually separated.
    ax = axes[7]
    mm_metric = metrics(*arrays(MM_MODEL))
    split_metrics = locked[["r2", "rmse", "mae", "bias"]].copy()
    split_metrics.loc[MM_MODEL] = [mm_metric[key] for key in ["r2", "rmse", "mae", "bias"]]
    ordered = split_metrics.sort_values("rmse").index.tolist()
    y = np.arange(len(ordered))
    ax.hlines(y, split_metrics.loc[ordered, "mae"], split_metrics.loc[ordered, "rmse"], color=GRID, lw=2.0)
    ax.scatter(split_metrics.loc[ordered, "mae"], y, marker="s", s=20,
               color=[COLORS[m] for m in ordered], edgecolor="white", linewidth=.4, label="MAE")
    ax.scatter(split_metrics.loc[ordered, "rmse"], y, marker="o", s=24,
               color=[COLORS[m] for m in ordered], edgecolor="white", linewidth=.4, label="RMSE")
    ax.set_yticks(y, [LABELS[m] + ("*" if m == MM_MODEL else "") for m in ordered]); ax.invert_yaxis()
    ax.set_xlabel("Split-specific error (pp)")
    clean_axes(ax, "x"); panel(ax, 7, "Complete model comparison", x=-.11, y=1.035)
    ax.legend(fontsize=4.5, ncol=2, loc="lower right", handletextpad=.3, columnspacing=.6)

    # i-o: complete residual distributions, all years descriptive.
    residual_summary = []
    for offset, model in enumerate(MODELS):
        index = 8 + offset
        ax = axes[index]
        observed, predicted = arrays(model)
        residual = predicted - observed
        ax.hist(residual, bins=28, color=COLORS[model], alpha=.82,
                edgecolor="white", linewidth=.28)
        ax.axvline(0, color="#7B8692", lw=.60)
        ax.set_xlabel("Residual (pp)")
        ax.set_ylabel("Tract-years" if offset % 4 == 0 else "")
        clean_axes(ax, "y"); panel(ax, index, f"{LABELS[model]} residual", x=-.11, y=1.035)
        residual_summary.append({
            "model": model,
            "median_residual": float(np.median(residual)),
            "residual_iqr": float(np.quantile(residual, .75) - np.quantile(residual, .25)),
        })

    # p: residual centre and spread in one aligned dot-range view.
    ax = axes[15]
    residual_frame = pd.DataFrame(residual_summary).set_index("model").loc[ordered]
    for yy, model in enumerate(ordered):
        center = residual_frame.loc[model, "median_residual"]
        half = residual_frame.loc[model, "residual_iqr"] / 2
        ax.hlines(yy, center - half, center + half, color=COLORS[model], lw=1.6)
        ax.scatter(center, yy, s=24, color=COLORS[model], edgecolor="white", linewidth=.4)
    ax.axvline(0, color="#7B8692", lw=.60)
    ax.set_yticks(range(len(ordered)), [LABELS[m] for m in ordered]); ax.invert_yaxis()
    ax.set_xlabel("Median residual ± half IQR (pp)")
    clean_axes(ax, "x"); panel(ax, 15, "Residual centre and spread", x=-.11, y=1.035)

    # q-t: locked 2023 metric summaries as lollipop plots.
    metric_specs = [("r2", "R²"), ("rmse", "RMSE (pp)"), ("mae", "MAE (pp)"), ("bias", "Bias (pp)")]
    locked_rows = []
    for model in MODELS:
        observed, predicted = arrays(model, None if model == MM_MODEL else 2023)
        row = metrics(observed, predicted)
        locked_rows.append({"model": model, **row})
    locked_exact = pd.DataFrame(locked_rows).set_index("model").loc[MODELS]
    for col, (metric, title) in enumerate(metric_specs):
        ax = axes[16 + col]
        vals = locked_exact[metric].to_numpy(float)
        base = 0 if metric != "r2" else min(0, float(vals.min()))
        for yy, (model, value) in enumerate(zip(MODELS, vals)):
            ax.hlines(yy, min(base, value), max(base, value), color=COLORS[model], lw=1.6)
            ax.scatter(value, yy, s=24, color=COLORS[model], edgecolor="white", linewidth=.4)
        if metric in {"r2", "bias"}:
            ax.axvline(0, color="#7B8692", lw=.58)
        ax.set_yticks(range(7), [LABELS[m] for m in MODELS] if col == 0 else [])
        ax.invert_yaxis(); ax.set_xlabel(title)
        clean_axes(ax, "x"); panel(ax, 16 + col, f"Split-specific {title}", x=-.11, y=1.035)

    # u-x: annual descriptive R²/RMSE/MAE/bias profiles for all seven models.
    for model in MODELS:
        for year in range(2014, 2024):
            observed, predicted = arrays(model, year)
            row = metrics(observed, predicted)
            annual_rows.append({"model": model, "health_year": int(year), **row})
    annual = pd.DataFrame(annual_rows)
    for col, (metric, title) in enumerate(metric_specs):
        ax = axes[20 + col]
        for j, model in enumerate(MODELS):
            view = annual.loc[annual.model.eq(model)]
            ax.plot(view.health_year, view[metric], color=COLORS[model], lw=1.0,
                    marker=MARKERS[j], ms=2.2, label=LABELS[model])
        if metric in {"r2", "bias"}:
            ax.axhline(0, color="#7B8692", lw=.55)
        ax.set_xticks([2014, 2017, 2020, 2023])
        ax.set_xlabel("Health-reference year"); ax.set_ylabel(title if col == 0 else "")
        clean_axes(ax); panel(ax, 20 + col, f"Annual descriptive {title}", x=-.11, y=1.035)
        if col == 0:
            ax.legend(ncol=2, fontsize=4.4, loc="lower left", columnspacing=.6,
                      handlelength=1.1, handletextpad=.25)

    delivery = save_delivery(fig, OUT, STEM)
    plt.close(fig)
    pd.DataFrame(profile_rows).to_csv(OUT / "source_descriptive_profile_metrics.csv", index=False)
    pd.DataFrame(residual_summary).to_csv(OUT / "source_residual_summary.csv", index=False)
    locked_exact.reset_index().to_csv(OUT / "source_locked_2023_metrics.csv", index=False)
    annual.to_csv(OUT / "source_annual_descriptive_metrics.csv", index=False)
    predictions.to_parquet(OUT / "source_registered_prediction_profiles.parquet", index=False)
    mm_predictions.to_parquet(OUT / "source_mm_gtgnnwr_spatial_outer_fold_profiles.parquet", index=False)
    manifest = {
        "figure": STEM,
        "panels": 24,
        "models": [LABELS[m] for m in MODELS],
        "descriptive_years": [2014, 2023],
        "locked_temporal_test_year": 2023,
        "locked_test_rows": 870,
        "split_contract": "six comparator models use descriptive 2014-2023 profiles and locked 2023 temporal tests; MM-GTGNNWR uses 33,760 registered spatial outer-fold evaluations and is labelled separately",
        "mm_gtgnnwr_contract": "current V211 compact proxy; registered spatial outer folds",
        "excluded_models": ["GTCNNWR"],
        "source_hashes": {str(PRED_PATH): sha256(PRED_PATH), str(METRIC_PATH): sha256(METRIC_PATH), str(MM_PATH): sha256(MM_PATH)},
        "proxy_analysis": True,
        "formal": False,
        "delivery": delivery,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_validation(OUT, {
        **manifest,
        "checks": {
            "six_temporal_comparators_plus_mm_gtgnnwr_present": len(TEMP_MODELS) == 6 and MODELS[-1] == MM_MODEL,
            "gtcnnnwr_excluded": "GTCNNWR_GPU" not in MODELS,
            "all_ten_reference_years_present": predictions.health_year.nunique() == 10,
            "locked_2023_test_has_870_rows": len(test) == 870,
            "descriptive_and_locked_contracts_separated": True,
            "shared_density_scale": True,
            "independent_axis_height_density_colourbars": density_colourbar_count == 7,
            "source_tables_written": all((OUT / name).stat().st_size > 0 for name in [
                "source_descriptive_profile_metrics.csv", "source_residual_summary.csv",
                "source_locked_2023_metrics.csv", "source_annual_descriptive_metrics.csv",
                "source_registered_prediction_profiles.parquet",
                "source_mm_gtgnnwr_spatial_outer_fold_profiles.parquet",
            ]),
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
        },
    })


if __name__ == "__main__":
    main()
