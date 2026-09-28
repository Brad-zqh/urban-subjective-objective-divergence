"""Render the matched signed S-O representation sensitivity for current V211."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nature_viz_common import BLUE, DIV, GRID, INK, MUTED, RED, clean_axes, panel, save_delivery, setup_style, sha256, write_validation


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "workstreams/RQ2_model/runs/v211_signed_delta_matched_sensitivity_r2_indexed_seed_filtered"
FOLD_SOURCE = RUN / "v211_signed_delta_fold_metrics.csv"
PARTITION_SOURCE = RUN / "v211_signed_delta_partition_metrics.csv"
PREDICTION_SOURCE = RUN / "v211_signed_delta_outer_predictions.parquet"
SUMMARY_SOURCE = RUN / "v211_signed_delta_summary.json"
CHAIN_VALIDATION = RUN / "independent_chain_validation.json"
OUT = ROOT / "figures/v211_signed_delta_sensitivity_12panel"
STEM = "Fig_v211_signed_delta_sensitivity_12panel"

SCHEME_ORDER = [
    "axis_recursive",
    "rotated45_recursive",
    "y_equal_count_stripes",
    "polar_north_clockwise_equal_count",
]
SCHEME_LABELS = {
    "axis_recursive": "Axis",
    "rotated45_recursive": "Rotated",
    "y_equal_count_stripes": "Stripes",
    "polar_north_clockwise_equal_count": "Polar",
}
SCHEME_COLORS = {
    "axis_recursive": BLUE,
    "rotated45_recursive": "#7C9FC7",
    "y_equal_count_stripes": "#D6938F",
    "polar_north_clockwise_equal_count": RED,
}
METRICS = [
    ("r2", "Outer-test R²", "positive favours explicit contrast"),
    ("rmse", "Outer-test RMSE", "negative favours explicit contrast"),
    ("mae", "Outer-test MAE", "negative favours explicit contrast"),
]


def metric_row(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual * residual) / sst),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
    }


def build_year_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, group in predictions.groupby("health_reference_year", sort=True):
        y = group["observed"].to_numpy(float)
        full = metric_row(y, group["prediction_full"].to_numpy(float))
        component = metric_row(y, group["prediction_os_only"].to_numpy(float))
        rows.append({
            "health_reference_year": int(year),
            "outer_evaluations": int(len(group)),
            **{f"full_{key}": value for key, value in full.items()},
            **{f"os_only_{key}": value for key, value in component.items()},
            "delta_r2_full_minus_os": full["r2"] - component["r2"],
            "delta_rmse_full_minus_os": full["rmse"] - component["rmse"],
            "delta_mae_full_minus_os": full["mae"] - component["mae"],
        })
    return pd.DataFrame(rows)


def paired_scatter(ax, frame: pd.DataFrame, metric: str, title: str, index: int) -> None:
    x = frame[f"os_only_{metric}"].to_numpy(float)
    y = frame[f"full_{metric}"].to_numpy(float)
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    pad = max((hi - lo) * 0.08, 1e-4)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#AEB5BF", linewidth=0.75, linestyle="--", zorder=1)
    for scheme in SCHEME_ORDER:
        mask = frame["scheme"].eq(scheme)
        ax.scatter(x[mask], y[mask], s=15, color=SCHEME_COLORS[scheme], alpha=0.82, edgecolor="white", linewidth=0.35, label=SCHEME_LABELS[scheme], zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("O + S + SES")
    ax.set_ylabel("+ explicit S−O")
    clean_axes(ax)
    panel(ax, index, title)
    if index == 0:
        ax.legend(loc="upper left", fontsize=4.6, ncol=2, handletextpad=0.25, columnspacing=0.6)


def fold_delta(ax, frame: pd.DataFrame, metric: str, title: str, direction: str, index: int) -> None:
    column = f"delta_{metric}_full_minus_os"
    ordered = frame.copy()
    ordered["scheme"] = pd.Categorical(ordered["scheme"], categories=SCHEME_ORDER, ordered=True)
    ordered = ordered.sort_values(["scheme", "boundary_vintage", "block"]).reset_index(drop=True)
    for scheme in SCHEME_ORDER:
        mask = ordered["scheme"].eq(scheme).to_numpy()
        ax.scatter(np.flatnonzero(mask), ordered.loc[mask, column], s=14, color=SCHEME_COLORS[scheme], alpha=0.86, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.7, zorder=2)
    for boundary in (7.5, 15.5, 23.5):
        ax.axvline(boundary, color=GRID, linewidth=0.55, zorder=1)
    median = float(ordered[column].median())
    q25, q75 = ordered[column].quantile([0.25, 0.75])
    ax.text(0.02, 0.96, f"median {median:+.3f}\nIQR [{q25:+.3f}, {q75:+.3f}]", transform=ax.transAxes, ha="left", va="top", fontsize=4.7, color=MUTED)
    ax.text(0.98, 0.04, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=4.2, color=MUTED)
    ax.set_xlim(-1, 32)
    ax.set_xticks([3.5, 11.5, 19.5, 27.5], [SCHEME_LABELS[value] for value in SCHEME_ORDER], rotation=25, ha="right")
    ax.set_ylabel(f"Δ{metric.upper()} (full − O+S)")
    clean_axes(ax, "y")
    panel(ax, index, title)


def partition_heatmap(fig, ax, frame: pd.DataFrame, metric: str, title: str, index: int) -> None:
    column = f"delta_{metric}_full_minus_os"
    matrix = frame.assign(boundary_vintage=frame["boundary_vintage"].astype(str).str.removeprefix("v")).pivot(index="scheme", columns="boundary_vintage", values=column).reindex(SCHEME_ORDER).reindex(columns=["2010", "2020"])
    values = matrix.to_numpy(float)
    limit = max(float(np.max(np.abs(values))), 1e-6)
    image = ax.imshow(values, cmap=DIV, norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for row in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(column_index, row, f"{values[row, column_index]:+.3f}", ha="center", va="center", fontsize=4.8, color=INK)
    ax.set_xticks([0, 1], ["2010 vintage", "2020 vintage"])
    ax.set_yticks(range(4), [SCHEME_LABELS[value] for value in SCHEME_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    color_ax = ax.inset_axes([1.025, 0.0, 0.035, 1.0])
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.ax.tick_params(labelsize=3.7, length=1.2, width=0.35, pad=0.6)
    colorbar.outline.set_linewidth(0.35)
    panel(ax, index, title)


def parity(ax, predictions: pd.DataFrame, column: str, title: str, index: int, cmap: str) -> None:
    observed = predictions["observed"].to_numpy(float)
    predicted = predictions[column].to_numpy(float)
    lo = min(float(observed.min()), float(predicted.min()))
    hi = max(float(observed.max()), float(predicted.max()))
    ax.hexbin(observed, predicted, gridsize=34, mincnt=1, bins="log", cmap=cmap, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.75, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed area-level MHLTH (%)")
    ax.set_ylabel("Prediction (%)")
    clean_axes(ax)
    panel(ax, index, title)


def main() -> None:
    setup_style(6.0)
    fold = pd.read_csv(FOLD_SOURCE)
    partition = pd.read_csv(PARTITION_SOURCE)
    predictions = pd.read_parquet(PREDICTION_SOURCE)
    summary = json.loads(SUMMARY_SOURCE.read_text(encoding="utf-8"))
    chain = json.loads(CHAIN_VALIDATION.read_text(encoding="utf-8"))
    if len(fold) != 32 or len(predictions) != 33760 or chain.get("passed") is not True:
        raise RuntimeError("V211 signed-delta source contract failed")
    year = build_year_metrics(predictions)

    fig, axes = plt.subplots(4, 3, figsize=(183 / 25.4, 178 / 25.4))
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.075, top=0.94, wspace=0.38, hspace=0.55)

    for column_index, (metric, title, _) in enumerate(METRICS):
        paired_scatter(axes[0, column_index], fold, metric, title, column_index)
    for column_index, (metric, title, direction) in enumerate(METRICS):
        fold_delta(axes[1, column_index], fold, metric, f"Paired fold Δ{metric.upper()}", direction, 3 + column_index)
    for column_index, (metric, _, _) in enumerate(METRICS):
        partition_heatmap(fig, axes[2, column_index], partition, metric, f"Partition-pooled Δ{metric.upper()}", 6 + column_index)

    parity(axes[3, 0], predictions, "prediction_full", "Explicit S−O representation", 9, "Reds")
    parity(axes[3, 1], predictions, "prediction_os_only", "O + S + SES representation", 10, "Blues")
    ax = axes[3, 2]
    ax.plot(year["health_reference_year"], year["delta_rmse_full_minus_os"], color=RED, marker="o", markersize=3.2, linewidth=1.15)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.fill_between(year["health_reference_year"], 0, year["delta_rmse_full_minus_os"], color=RED, alpha=0.10)
    ax.set_xticks(year["health_reference_year"].iloc[::2])
    ax.set_xlabel("Health-reference year")
    ax.set_ylabel("ΔRMSE (full − O+S)")
    ax.text(0.98, 0.96, "positive = explicit contrast worse", transform=ax.transAxes, ha="right", va="top", fontsize=4.3, color=MUTED)
    clean_axes(ax)
    panel(ax, 11, "Annual representation sensitivity")

    fig.text(0.09, 0.975, "Post-protocol matched representation sensitivity · 32 outcome-blind spatial folds", ha="left", va="top", fontsize=5.2, color=MUTED)
    fig.text(0.965, 0.025, "Proxy analysis; formal = false. S−O is derived from O and S; differences reflect representation and regularization, not new observations.", ha="right", va="bottom", fontsize=4.35, color=MUTED)

    delivery = save_delivery(fig, OUT, STEM, dpi=600)
    plt.close(fig)

    fold.to_csv(OUT / "source_fold_metrics.csv", index=False)
    partition.to_csv(OUT / "source_partition_metrics.csv", index=False)
    predictions.to_csv(OUT / "source_outer_test_predictions.csv", index=False)
    year.to_csv(OUT / "source_annual_metrics.csv", index=False)
    pooled = summary["pooled_outer"] | summary["pooled_outer_differences"] | summary["full_vs_os_only_fold_summary"]
    pd.DataFrame([pooled]).to_csv(OUT / "source_panel_statistics.csv", index=False)

    validation = {
        "figure": STEM,
        "panels": 12,
        "proxy_analysis": True,
        "formal": False,
        "post_protocol_sensitivity": True,
        "preregistered_gate": False,
        "checks": {
            "chain_validation_passed": chain.get("passed") is True,
            "thirty_two_folds": len(fold) == 32,
            "all_outer_rows_retained": len(predictions) == 33760,
            "four_appearances_per_row": predictions.groupby("numeric_row").size().eq(4).all(),
            "all_metrics_finite": np.isfinite(fold.select_dtypes(include=[np.number]).to_numpy(float)).all(),
            "five_delivery_formats": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
        },
        "source_sha256": {
            "fold_metrics": sha256(FOLD_SOURCE),
            "partition_metrics": sha256(PARTITION_SOURCE),
            "outer_predictions": sha256(PREDICTION_SOURCE),
            "summary": sha256(SUMMARY_SOURCE),
            "chain_validation": sha256(CHAIN_VALIDATION),
        },
        "delivery": delivery,
    }
    write_validation(OUT, validation)
    (OUT / "manifest.json").write_text(json.dumps({
        "figure": STEM,
        "panels": 12,
        "delivery_formats": ["jpg", "pdf", "png", "svg", "tiff"],
        "proxy_analysis": True,
        "formal": False,
        "post_protocol_sensitivity": True,
        "preregistered_gate": False,
        "estimand": summary["estimand"],
        "uncertainty": summary["uncertainty"],
        "interpretation_guard": summary["interpretation_guard"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
