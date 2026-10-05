"""Large-matrix, main-text-ready known-truth validation atlas (non-destructive).

The sealed V154 simulation is retained as a bounded method-validation result;
it is not merged with current V211 empirical coefficients.  This v2 renderer
enlarges every coefficient field, moves each colour bar farther outside the
matrix, and exports JPG/SVG only.
"""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

from nature_viz_common import ROOT, DIV, INK, GRID, setup_style, clean_axes, sha256  # noqa: E402

RUN = ROOT.parent / "source_data/fig06"
METRICS_PATH = RUN / "v154_combined_model_metrics.csv"
SURFACES_PATH = RUN / "v154_combined_recovery_surfaces.csv"
OUT = ROOT / "figures/v154_known_truth_maintext_v2"
STEM = "Fig_v154_known_truth_maintext_v2"
MODELS = ["GWR", "GTWR", "GNNWR", "GTNNWR", "GTGNNWR", "MM-GTGNNWR"]
ROWS = ["Truth", "GTWR", "GNNWR", "GTNNWR", "GTGNNWR", "MM-GTGNNWR"]
BETAS = ["beta0", "beta1", "beta2"]
MODEL_COLOURS = {model: DIV(value) for model, value in zip(MODELS, np.linspace(.04, .96, len(MODELS)))}


def panel_letter(ax: plt.Axes, index: int, x: float = -0.08, y: float = 1.025) -> None:
    ax.text(x, y, string.ascii_lowercase[index], transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.4, fontweight="bold", color=INK, clip_on=False)


def save_jpg_svg(fig: plt.Figure) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    jpg = OUT / f"{STEM}.jpg"
    svg = OUT / f"{STEM}.svg"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(jpg, dpi=600, bbox_inches="tight", pil_kwargs={"quality": 95})
    with Image.open(jpg) as image:
        dpi = image.info.get("dpi", (0, 0))
        pixels = image.size
    svg_text = svg.read_text(encoding="utf-8", errors="ignore")
    return {
        "jpg": str(jpg), "svg": str(svg), "delivery_formats": ["jpg", "svg"],
        "jpg_dpi_metadata": [float(dpi[0]), float(dpi[1])],
        "jpg_pixel_size": [int(pixels[0]), int(pixels[1])],
        "svg_text_editable": "<text" in svg_text,
    }


def intervals(ax: plt.Axes, metrics: pd.DataFrame, metric: str, index: int,
              title: str, xlabel: str) -> list[dict]:
    source = []
    for y, model in enumerate(MODELS):
        values = metrics.loc[metrics.model.eq(model), metric].to_numpy(float)
        lo, med, hi = np.quantile(values, [.025, .5, .975])
        rng = np.random.default_rng(15500 + index * 100 + y)
        ax.scatter(values, y + rng.uniform(-.10, .10, len(values)), s=7.5,
                   color=MODEL_COLOURS[model], alpha=.35, linewidth=0)
        ax.hlines(y, lo, hi, color=MODEL_COLOURS[model], lw=1.55)
        ax.scatter(med, y, s=21, color=MODEL_COLOURS[model], edgecolor="white", linewidth=.45)
        source.append({"model": model, "metric": metric, "median": float(med),
                       "q025": float(lo), "q975": float(hi), "n_fold_units": int(len(values))})
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.invert_yaxis(); ax.set_xlabel(xlabel)
    clean_axes(ax, "x")
    ax.set_title(title, loc="left", pad=2.8, fontweight="normal")
    panel_letter(ax, index, x=-.075, y=1.03)
    return source


def matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    return frame.pivot(index="v", columns="u", values=value).sort_index().to_numpy(float)


def main() -> None:
    setup_style(6.25)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_all = pd.read_csv(METRICS_PATH)
    surfaces_all = pd.read_csv(SURFACES_PATH)
    metrics = metrics_all.loc[metrics_all.model.isin(MODELS)].copy()
    surfaces = surfaces_all.loc[surfaces_all.model.isin(MODELS)].copy()
    if set(metrics.model) != set(MODELS) or set(surfaces.model) != set(MODELS):
        raise RuntimeError("V154_CURRENT_MODEL_SET_MISMATCH")
    if not metrics.groupby("model").size().eq(15).all():
        raise RuntimeError("V154_EXPECTED_15_FOLD_UNITS_PER_MODEL")

    truth = surfaces.loc[surfaces.model.eq("GWR")].drop_duplicates(["u", "v"])
    norms: dict[str, TwoSlopeNorm] = {}
    for beta in BETAS:
        values = np.r_[truth[beta].to_numpy(float), surfaces[f"{beta}_hat"].to_numpy(float)]
        lo, hi = np.quantile(values, [.01, .99])
        centre = float(np.median(truth[beta]))
        norms[beta] = TwoSlopeNorm(vmin=min(float(lo), centre - 1e-6), vcenter=centre,
                                   vmax=max(float(hi), centre + 1e-6))

    fig = plt.figure(figsize=(183 / 25.4, 276 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        7, 6, left=.108, right=.940, bottom=.032, top=.978,
        wspace=.62, hspace=.43, height_ratios=[.68, 1, 1, 1, 1, 1, 1],
    )
    summary_rows = []
    summary_rows += intervals(fig.add_subplot(gs[0, 0:3]), metrics, "r2", 0,
                              "Spatial-extrapolation prediction", "Test R²")
    summary_rows += intervals(fig.add_subplot(gs[0, 3:6]), metrics, "beta_rmse_mean", 1,
                              "Local-coefficient recovery", "Mean coefficient RMSE")

    map_axes: list[plt.Axes] = []
    cbar_axes: list[plt.Axes] = []
    index = 2
    for row, model in enumerate(ROWS, start=1):
        frame = truth if model == "Truth" else surfaces.loc[surfaces.model.eq(model)]
        for col, beta in enumerate(BETAS):
            ax = fig.add_subplot(gs[row, col * 2:(col + 1) * 2])
            value = beta if model == "Truth" else f"{beta}_hat"
            im = ax.imshow(matrix(frame, value), origin="lower", cmap=DIV, norm=norms[beta],
                           interpolation="nearest", rasterized=True)
            ax.set_xticks([0, 31, 63], [0, 31, 63])
            ax.set_yticks([0, 31, 63], [0, 31, 63])
            ax.tick_params(labelsize=5.0, length=1.35, width=.35, pad=.65)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color("#4C545D"); spine.set_linewidth(.38)
            if row == 1:
                ax.set_title(rf"$\beta_{col}$", pad=2.0, fontsize=7.4, fontweight="normal")
            if col == 0:
                ax.text(-.34, .50, model, transform=ax.transAxes, ha="right", va="center",
                        fontsize=6.4, fontweight="normal")
            note = "Ground truth" if model == "Truth" else (
                f"RMSE {metrics.loc[metrics.model.eq(model), beta + '_rmse'].mean():.3f}"
            )
            ax.text(.975, .035, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8,
                    fontweight="normal",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .76, "pad": .35})
            panel_letter(ax, index, x=-.06, y=1.015)
            # Larger visible gap than the legacy plate; the bar remains equal height.
            cax = ax.inset_axes([1.085, 0.0, .026, 1.0], transform=ax.transAxes)
            cb = fig.colorbar(im, cax=cax)
            cb.set_ticks([norms[beta].vmin, norms[beta].vcenter, norms[beta].vmax])
            cb.ax.tick_params(labelsize=4.8, length=1.0, width=.32, pad=.40)
            cb.outline.set_linewidth(.32)
            map_axes.append(ax); cbar_axes.append(cax)
            index += 1

    fig.canvas.draw()
    map_boxes = [ax.get_position() for ax in map_axes]
    cbar_boxes = [ax.get_position() for ax in cbar_axes]
    widths = np.array([box.width for box in map_boxes])
    heights = np.array([box.height for box in map_boxes])
    gap_ratios = np.array([(c.x0 - (m.x0 + m.width)) / m.width for m, c in zip(map_boxes, cbar_boxes)])
    cbar_height_ratios = np.array([c.height / m.height for m, c in zip(map_boxes, cbar_boxes)])

    delivery = save_jpg_svg(fig)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(OUT / "source_model_metric_intervals.csv", index=False)
    validation = {
        "figure": STEM,
        "panels": 20,
        "analysis_source": "sealed V154 known-truth simulation",
        "manuscript_role": "bounded method validation; not current V211 empirical coefficient evidence",
        "rows": ROWS,
        "models_in_summary": MODELS,
        "excluded_models": ["GTCNNWR"],
        "display_limits": "shared within each beta column; 1st-99th percentile display clipping only",
        "source_hashes": {str(METRICS_PATH.relative_to(ROOT.parent)): sha256(METRICS_PATH),
                          str(SURFACES_PATH.relative_to(ROOT.parent)): sha256(SURFACES_PATH)},
        "geometry": {
            "map_width_range": float(np.ptp(widths)),
            "map_height_range": float(np.ptp(heights)),
            "mean_colourbar_gap_in_map_widths": float(gap_ratios.mean()),
            "colourbar_gap_range": float(np.ptp(gap_ratios)),
            "mean_colourbar_to_map_height": float(cbar_height_ratios.mean()),
        },
        "delivery": delivery,
        "checks": {
            "panel_count_20": index == 20,
            "six_current_models_in_summary": metrics.model.nunique() == 6,
            "fifteen_fold_units_each": bool(metrics.groupby("model").size().eq(15).all()),
            "eighteen_complete_64_by_64_fields": len(map_axes) == 18 and bool(surfaces.groupby("model").size().eq(4096).all()),
            "equal_matrix_sizes": bool(np.ptp(widths) < 1e-6 and np.ptp(heights) < 1e-6),
            "colourbars_equal_matrix_height": bool(np.allclose(cbar_height_ratios, 1, atol=.01)),
            "colourbar_gap_at_least_7pct_matrix_width": bool(gap_ratios.min() >= .07),
            "only_panel_letters_bold_by_construction": True,
            "jpg_svg_only": delivery["delivery_formats"] == ["jpg", "svg"],
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
        },
    }
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\n"
        "- Claim: predictive performance and local-coefficient recovery are separate validation targets.\n"
        "- Evidence: 15 sealed fold units and 18 complete 64×64 coefficient fields.\n"
        "- Boundary: this is a V154 known-truth simulation, not current V211 Chicago empirical evidence.\n"
        "- Layout: matrices are enlarged; each colour bar is moved 8.5% of a matrix width outward.\n"
        "- Preservation: all previous scripts and exports remain untouched.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT / f"{STEM}.jpg"), "delivery": delivery}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
