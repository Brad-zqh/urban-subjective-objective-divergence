"""Render a 20-panel matched signed S-O representation sensitivity atlas for current V211."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from nature_viz_common import BLUE, DIV, GRID, INK, MUTED, RED, clean_axes, panel, save_delivery, setup_style, sha256, write_validation


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "workstreams/RQ2_model/runs/v211_signed_delta_matched_sensitivity_r2_indexed_seed_filtered"
FOLD_SOURCE = RUN / "v211_signed_delta_fold_metrics.csv"
PARTITION_SOURCE = RUN / "v211_signed_delta_partition_metrics.csv"
PREDICTION_SOURCE = RUN / "v211_signed_delta_outer_predictions.parquet"
SUMMARY_SOURCE = RUN / "v211_signed_delta_summary.json"
CHAIN_VALIDATION = RUN / "independent_chain_validation.json"
OUT = ROOT / "figures/v211_signed_delta_sensitivity_20panel"
STEM = "Fig_v211_signed_delta_sensitivity_20panel"

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

# Panels m/n encode observation density, not the sign or direction of a model
# contrast.  Use the same red–white–blue density ramp for both models: blue
# marks sparse support, white marks intermediate density and red marks the
# densest part of the observed–predicted cloud.  The palette is descriptive;
# it is not a positive/negative effect scale.
PARITY_DENSITY = mpl.colors.LinearSegmentedColormap.from_list(
    "v211_parity_density",
    ["#355C9B", "#AFC7E0", "#F7F7F7", "#E7AAA5", "#B40426"],
)


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
        points = frame.loc[mask]
        # Fold sizes are empirical and differ across vintages. Encoding them
        # in point area adds information without inventing another statistic.
        sizes = 9.0 + 13.0 * np.sqrt(
            points["outer_test_rows"].to_numpy(float) / float(frame["outer_test_rows"].max())
        )
        ax.scatter(x[mask], y[mask], s=sizes, color=SCHEME_COLORS[scheme], alpha=0.55,
                   edgecolor="white", linewidth=0.30, label=SCHEME_LABELS[scheme], zorder=2)
        med_x, med_y = float(np.median(x[mask])), float(np.median(y[mask]))
        qx = np.quantile(x[mask], [0.25, 0.75]); qy = np.quantile(y[mask], [0.25, 0.75])
        ax.errorbar(med_x, med_y,
                    xerr=[[med_x - qx[0]], [qx[1] - med_x]],
                    yerr=[[med_y - qy[0]], [qy[1] - med_y]],
                    fmt="D", ms=3.8, color=SCHEME_COLORS[scheme],
                    ecolor=SCHEME_COLORS[scheme], elinewidth=.75, capsize=1.7,
                    mec="white", mew=.35, zorder=4)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("O + S + SES")
    ax.set_ylabel("+ explicit S−O")
    clean_axes(ax)
    panel(ax, index, title)
    if index == 0:
        ax.text(.98, .04, "points = folds · diamonds = median ± IQR",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=3.5, color=MUTED)


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
    soft_div = mpl.colors.LinearSegmentedColormap.from_list(
        "gmh_soft_delta", ["#8DB0D5", "#F7F7F7", "#E6A6A1"])
    image = ax.imshow(values, cmap=soft_div,
                      norm=mpl.colors.TwoSlopeNorm(vmin=-1.18 * limit, vcenter=0, vmax=1.18 * limit),
                      aspect="auto")
    for row in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(column_index, row, f"{values[row, column_index]:+.3f}", ha="center", va="center", fontsize=4.8, color=INK)
    ax.set_xticks([0, 1], ["2010\nboundary", "2020\nboundary"])
    ax.set_yticks(range(4), [SCHEME_LABELS[value] for value in SCHEME_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    color_ax = ax.inset_axes([1.020, 0.0, 0.026, 1.0])
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.ax.tick_params(labelsize=3.7, length=1.2, width=0.35, pad=0.6)
    colorbar.outline.set_linewidth(0.35)
    panel(ax, index, title)


def parity(ax, predictions: pd.DataFrame, column: str, title: str, index: int) -> None:
    observed = predictions["observed"].to_numpy(float)
    predicted = predictions[column].to_numpy(float)
    lo = min(float(observed.min()), float(predicted.min()))
    hi = max(float(observed.max()), float(predicted.max()))
    artist = ax.hexbin(observed, predicted, gridsize=34, mincnt=1, bins="log",
                       cmap=PARITY_DENSITY, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.75, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed area-level MHLTH (%)")
    ax.set_ylabel("Predicted MHLTH (%)")
    clean_axes(ax)
    panel(ax, index, title)
    cax = ax.inset_axes([1.020, 0.0, .024, 1.0], transform=ax.transAxes)
    cb = ax.figure.colorbar(artist, cax=cax)
    cb.set_label("Count/bin", fontsize=3.6, labelpad=.45)
    cb.ax.tick_params(labelsize=3.2, length=.75, pad=.3)
    cb.outline.set_linewidth(.28)


def fold_outcomes(ax, frame: pd.DataFrame, index: int) -> pd.DataFrame:
    """Show directional coverage by metric, scheme and pooled folds.

    The previous stacked bars collapsed the 32 matched folds into three nearly
    identical percentages.  This dot-and-interval display retains the pooled
    estimate while exposing the scheme-level composition that drives it.
    """
    labels = [title.replace("Outer-test ", "") for _, title, _ in METRICS]
    y = np.arange(len(labels), dtype=float)
    rows: list[dict[str, str | float]] = []
    for metric_index, (metric, _, _) in enumerate(METRICS):
        indicator = f"full_better_{metric}"
        if indicator not in frame:
            delta = frame[f"delta_{metric}_full_minus_os"].to_numpy(float)
            gain = delta if metric == "r2" else -delta
            frame = frame.copy()
            frame[indicator] = gain > 0
        pooled = 100.0 * float(frame[indicator].mean())
        rows.append({"metric": metric, "scheme": "pooled", "improved_percent": pooled})
        ax.hlines(y[metric_index], 0, pooled, color="#B8BEC7", linewidth=2.0, zorder=1)
        for offset, scheme in zip(np.linspace(-.18, .18, len(SCHEME_ORDER)), SCHEME_ORDER):
            share = 100.0 * float(frame.loc[frame.scheme.eq(scheme), indicator].mean())
            rows.append({"metric": metric, "scheme": scheme, "improved_percent": share})
            ax.scatter(share, y[metric_index] + offset, s=25, color=SCHEME_COLORS[scheme],
                       edgecolor="white", linewidth=.35, zorder=3)
        ax.scatter(pooled, y[metric_index], s=31, marker="D", color=INK,
                   edgecolor="white", linewidth=.45, zorder=4)
        ax.text(min(pooled + 3.5, 94), y[metric_index], f"pooled {pooled:.0f}%",
                ha="left", va="center", fontsize=3.5, color=MUTED)
    ax.axvline(50, color=GRID, linewidth=.65, linestyle="--", zorder=0)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Folds improved (%)")
    ax.invert_yaxis()
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=SCHEME_COLORS[s],
                       markeredgecolor="white", markeredgewidth=.35, markersize=4,
                       label=SCHEME_LABELS[s]) for s in SCHEME_ORDER]
    handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=INK,
                           markeredgecolor="white", markersize=4, label="Pooled"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(.99, .86),
              fontsize=3.1, ncol=1, handletextpad=.2, columnspacing=.45,
              borderpad=.15, frameon=False)
    clean_axes(ax, "x")
    panel(ax, index, "Directional coverage by fold")
    return pd.DataFrame(rows)


def row_error_shift(ax, predictions: pd.DataFrame, index: int) -> None:
    """Summarize row-level absolute-error changes by year with empirical intervals."""
    frame = predictions.copy()
    frame["abs_error_shift"] = (
        np.abs(frame["prediction_full"] - frame["observed"])
        - np.abs(frame["prediction_os_only"] - frame["observed"])
    )
    years = sorted(frame["health_reference_year"].unique())
    for yi, year_value in enumerate(years):
        values = frame.loc[frame.health_reference_year.eq(year_value), "abs_error_shift"].to_numpy(float)
        q05, q25, q50, q75, q95 = np.quantile(values, [.05, .25, .50, .75, .95])
        color = RED if q50 > 0 else BLUE
        ax.plot([q05, q95], [yi, yi], color=color, lw=.65, alpha=.50)
        ax.plot([q25, q75], [yi, yi], color=color, lw=3.0, solid_capstyle="round")
        ax.plot(q50, yi, marker="D", ms=3.2, color=color, mec="white", mew=.35)
    ax.axvline(0, color=INK, lw=.7)
    ax.set_yticks(np.arange(len(years)), [str(value) for value in years])
    ax.set_xlabel("|error| change: full − O+S (pp)")
    ax.invert_yaxis()
    clean_axes(ax, "x")
    panel(ax, index, "Row-level error shift by year")


def scheme_gain_matrix(fig, ax, frame: pd.DataFrame, index: int) -> None:
    """Show scheme-wise median changes with a directionally aligned colour scale."""
    raw = np.zeros((len(SCHEME_ORDER), len(METRICS)), dtype=float)
    scaled = np.zeros_like(raw)
    for column_index, (metric, _, _) in enumerate(METRICS):
        for row_index, scheme in enumerate(SCHEME_ORDER):
            delta = frame.loc[frame.scheme.eq(scheme), f"delta_{metric}_full_minus_os"].median()
            raw[row_index, column_index] = delta
        gain = raw[:, column_index] if metric == "r2" else -raw[:, column_index]
        scale = max(float(np.max(np.abs(gain))), 1e-12)
        scaled[:, column_index] = gain / scale
    soft_div = mpl.colors.LinearSegmentedColormap.from_list(
        "gmh_soft_gain", ["#8DB0D5", "#F7F7F7", "#E6A6A1"])
    image = ax.imshow(scaled, cmap=soft_div,
                      norm=mpl.colors.TwoSlopeNorm(vmin=-1.18, vcenter=0, vmax=1.18), aspect="auto")
    for row_index in range(raw.shape[0]):
        for column_index in range(raw.shape[1]):
            ax.text(column_index, row_index, f"{raw[row_index, column_index]:+.3f}",
                    ha="center", va="center", fontsize=4.4, color=INK)
    ax.set_xticks(np.arange(3), ["ΔR²", "ΔRMSE", "ΔMAE"])
    ax.set_yticks(np.arange(4), [SCHEME_LABELS[value] for value in SCHEME_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([1.020, 0.0, .026, 1.0])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(["−1", "0", "+1"])
    colorbar.set_label("relative gain", fontsize=3.8, labelpad=1.2)
    colorbar.ax.tick_params(labelsize=3.8, length=.8, pad=.5)
    colorbar.outline.set_linewidth(.3)
    panel(ax, index, "Scheme median change")


def annual_delta(ax, year: pd.DataFrame, metric: str, title: str, index: int) -> None:
    column = f"delta_{metric}_full_minus_os"
    values = year[column].to_numpy(float)
    colors = np.where((values > 0) if metric == "r2" else (values < 0), RED, BLUE)
    ax.plot(year["health_reference_year"], values, color="#9BA4AF", lw=.8, zorder=1)
    ax.scatter(year["health_reference_year"], values, c=colors, s=15, edgecolor="white", linewidth=.35, zorder=2)
    ax.axhline(0, color=INK, linewidth=.7)
    ax.set_xticks(year["health_reference_year"].iloc[::2])
    ax.set_xlabel("Health-reference year")
    ax.set_ylabel(f"Δ{metric.upper()} (full − O+S)")
    clean_axes(ax)
    panel(ax, index, title)


def scheme_interval(ax, frame: pd.DataFrame, metric: str, title: str, index: int) -> None:
    column = f"delta_{metric}_full_minus_os"
    for yi, scheme in enumerate(SCHEME_ORDER):
        values = frame.loc[frame.scheme.eq(scheme), column].to_numpy(float)
        q05, q25, q50, q75, q95 = np.quantile(values, [.05, .25, .50, .75, .95])
        color = SCHEME_COLORS[scheme]
        jitter = np.linspace(-.12, .12, len(values))
        ax.scatter(values, yi + jitter, s=8, color=color, alpha=.42, edgecolor="none")
        ax.plot([q05, q95], [yi, yi], color=color, lw=.65, alpha=.55)
        ax.plot([q25, q75], [yi, yi], color=color, lw=3.0, solid_capstyle="round")
        ax.plot(q50, yi, marker="D", ms=3.2, color=color, mec="white", mew=.35)
    ax.axvline(0, color=INK, lw=.7)
    ax.set_yticks(np.arange(4), [SCHEME_LABELS[value] for value in SCHEME_ORDER])
    ax.set_xlabel(f"Δ{metric.upper()} (full − O+S)")
    ax.invert_yaxis()
    clean_axes(ax, "x")
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

    fig, axes = plt.subplots(5, 4, figsize=(183 / 25.4, 242 / 25.4))
    fig.subplots_adjust(left=0.090, right=0.950, bottom=0.078, top=0.975, wspace=0.68, hspace=0.66)

    for column_index, (metric, title, _) in enumerate(METRICS):
        paired_scatter(axes[0, column_index], fold, metric, title, column_index)
    coverage = fold_outcomes(axes[0, 3], fold, 3)
    for column_index, (metric, title, direction) in enumerate(METRICS):
        fold_delta(axes[1, column_index], fold, metric, f"Paired fold Δ{metric.upper()}", direction, 4 + column_index)
    row_error_shift(axes[1, 3], predictions, 7)
    for column_index, (metric, _, _) in enumerate(METRICS):
        partition_heatmap(fig, axes[2, column_index], partition, metric, f"Partition-pooled Δ{metric.upper()}", 8 + column_index)
    scheme_gain_matrix(fig, axes[2, 3], fold, 11)

    parity(axes[3, 0], predictions, "prediction_full", "Explicit S−O representation", 12)
    parity(axes[3, 1], predictions, "prediction_os_only", "O + S + SES representation", 13)
    annual_delta(axes[3, 2], year, "r2", "Annual ΔR²", 14)
    annual_delta(axes[3, 3], year, "rmse", "Annual ΔRMSE", 15)

    annual_delta(axes[4, 0], year, "mae", "Annual ΔMAE", 16)
    scheme_interval(axes[4, 1], fold, "r2", "Scheme-level ΔR²", 17)
    scheme_interval(axes[4, 2], fold, "rmse", "Scheme-level ΔRMSE", 18)
    scheme_interval(axes[4, 3], fold, "mae", "Scheme-level ΔMAE", 19)

    delivery = save_delivery(fig, OUT, STEM, dpi=600)
    plt.close(fig)

    fold.to_csv(OUT / "source_fold_metrics.csv", index=False)
    partition.to_csv(OUT / "source_partition_metrics.csv", index=False)
    predictions.to_csv(OUT / "source_outer_test_predictions.csv", index=False)
    year.to_csv(OUT / "source_annual_metrics.csv", index=False)
    coverage.to_csv(OUT / "source_fold_direction_coverage.csv", index=False)
    pooled = summary["pooled_outer"] | summary["pooled_outer_differences"] | summary["full_vs_os_only_fold_summary"]
    pd.DataFrame([pooled]).to_csv(OUT / "source_panel_statistics.csv", index=False)

    validation = {
        "figure": STEM,
        "panels": 20,
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
            "direction_coverage_source_written": (OUT / "source_fold_direction_coverage.csv").exists(),
            "each_parity_density_panel_has_independent_colourbar": True,
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
        "panels": 20,
        "delivery_formats": ["jpg", "pdf", "png", "svg", "tiff"],
        "proxy_analysis": True,
        "formal": False,
        "post_protocol_sensitivity": True,
        "preregistered_gate": False,
        "display_grammar": "size-encoded paired parity with scheme median/IQR, directional coverage dots, empirical interval-dot, heat map and annual trajectories; no filled KDE",
        "parity_density_palette": "blue-to-white-to-red density ramp with an independent count colourbar on each parity panel; colours encode bin density only, not effect direction",
        "estimand": summary["estimand"],
        "uncertainty": summary["uncertainty"],
        "interpretation_guard": summary["interpretation_guard"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
