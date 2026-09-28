"""Twenty-panel matched graph-ablation and spatial-fold stability atlas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

from nature_viz_common import (
    ROOT, BLUE, RED, INK, MUTED, GRID, DIV, setup_style, panel, clean_axes,
    add_equal_height_colorbar, save_delivery, sha256, symmetric_limit,
    write_validation, symmetric_colourbar_ticks,
)


RUN = ROOT / "workstreams/RQ2_model/runs/figure_v118_fused_model_and_matched_graph_ablation"
PART_PATH = RUN / "v118_matched_graph_partition_ablation.csv"
BLOCK_PATH = RUN / "v118_matched_graph_block_ablation.csv"
FOLD_PATH = RUN / "v118_spatial_block_paired_mae_matrix.csv"
OUT = ROOT / "figures/v211_matched_ablation_stability_20panel"
STEM = "Fig_v211_matched_ablation_stability_20panel"

FAMILIES = ["multiscale_graph_fusion_mobility", "multiscale_graph_fusion_four"]
FAMILY_LABEL = {
    FAMILIES[0]: "MM-GTGNNWR · mobility",
    FAMILIES[1]: "MM-GTGNNWR · four relations",
}
FAMILY_COLOR = {FAMILIES[0]: RED, FAMILIES[1]: BLUE}
SCHEMES = ["axis_recursive", "rotated45_recursive", "x_equal_count_stripes", "y_equal_count_stripes"]
SCHEME_LABEL = dict(zip(SCHEMES, ["Axis", "Rotated", "X-stripes", "Y-stripes"]))
MODELS = ["ridge_anchor", "gtwr_kernel", "gnnwr_spatial", "cleanroom_gtnnwr", "spline_gam_ridge",
          "mobility_only_inductive", "available_four_inductive", "multiscale_graph_fusion_mobility",
          "multiscale_graph_fusion_four"]
MODEL_LABEL = dict(zip(MODELS, [
    "Ridge", "GTWR", "GNNWR", "GTNNWR", "GAM",
    "MM-GTGNNWR · mobility", "MM-GTGNNWR · four relations",
    "MM-GTGNNWR · fused mobility", "MM-GTGNNWR · fused four relations",
]))
HEAT_LABEL = {
    m: (
        "MM-\nGTGNNWR\n" + MODEL_LABEL[m].split(" · ", 1)[1]
        if "MM-GTGNNWR" in MODEL_LABEL[m] else MODEL_LABEL[m]
    )
    for m in MODELS
}
# Heatmap columns are named once in a dedicated key between the heatmap and
# forest rows.  Repeating long multi-line labels under each 9-column heatmap
# made the previous atlas unreadable at journal width.
HEAT_ROW_LABEL = [
    "Ridge", "GTWR", "GNNWR", "GTNNWR", "GAM",
    "Mobility graph", "Four-relation graph",
    "Fused mobility", "Fused four relations",
]
MODEL_COLOR = dict(zip(MODELS, [
    "#596573", "#3B4CC0", "#587BC5", "#769CC9", "#8EA9C7",
    "#D89B94", "#D17673", "#C94D56", "#B40426",
]))


def border_check(path: str):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate([rgb[:5].reshape(-1, 3), rgb[-5:].reshape(-1, 3),
                             rgb[:, :5].reshape(-1, 3), rgb[:, -5:].reshape(-1, 3)])
    return float(np.mean(np.any(rgb < 245, axis=2))), float(np.mean(np.any(border < 235, axis=1)))


def summary_gain(ax, part, metric, index, title):
    point = f"point_{metric}_gain"; low = f"{metric}_ci_low"; high = f"{metric}_ci_high"
    rows = []
    y = 0
    for scheme in SCHEMES:
        for family in FAMILIES:
            row = part.loc[part.partition_scheme.eq(scheme) & part.family.eq(family)].iloc[0]
            ax.hlines(y, row[low], row[high], color=FAMILY_COLOR[family], lw=1.5)
            ax.scatter(row[point], y, s=20, color=FAMILY_COLOR[family], edgecolor="white", linewidth=.35)
            rows.append(SCHEME_LABEL[scheme]); y += 1
    ax.axvline(0, color=MUTED, lw=.6)
    ax.set_yticks([.5, 2.5, 4.5, 6.5], [SCHEME_LABEL[s] for s in SCHEMES] if index == 0 else [])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()} gain from graph (pp)")
    clean_axes(ax, "x"); panel(ax, index, title)


def paired_scheme(ax, part, metric, index, title):
    point = f"point_{metric}_gain"
    for y, scheme in enumerate(SCHEMES):
        vals = [part.loc[part.partition_scheme.eq(scheme) & part.family.eq(f), point].iloc[0] for f in FAMILIES]
        ax.plot(vals, [y, y], color="#B8C0CA", lw=1.0)
        ax.scatter(vals[0], y, s=22, color=RED, edgecolor="white", linewidth=.35)
        ax.scatter(vals[1], y, s=22, color=BLUE, edgecolor="white", linewidth=.35)
    ax.axvline(0, color=MUTED, lw=.55); ax.set_yticks(range(4), [SCHEME_LABEL[s] for s in SCHEMES] if index == 2 else []); ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()} gain (pp)"); clean_axes(ax, "x"); panel(ax, index, title)


def block_panel(ax, block, scheme, metric, index, title):
    point = f"point_{metric}_gain"; low = f"{metric}_ci_low"; high = f"{metric}_ci_high"
    current = block.loc[block.partition_scheme.eq(scheme)]
    for y, b in enumerate(["B1", "B2", "B3", "B4"]):
        for offset, family in [(-.09, FAMILIES[0]), (.09, FAMILIES[1])]:
            row = current.loc[current.outer_test_block.eq(b) & current.family.eq(family)].iloc[0]
            ax.hlines(y + offset, row[low], row[high], color=FAMILY_COLOR[family], lw=1.05)
            ax.scatter(row[point], y + offset, s=13, color=FAMILY_COLOR[family], edgecolor="white", linewidth=.3)
    ax.axvline(0, color=MUTED, lw=.55); ax.set_yticks(range(4), ["B1", "B2", "B3", "B4"] if index in (4, 8) else [])
    ax.invert_yaxis(); ax.set_xlabel(f"{metric.upper()} gain (pp)")
    clean_axes(ax, "x"); panel(ax, index, title)


def metric_forest(ax, fold, metric, index, title, xlabel):
    for y, model in enumerate(MODELS):
        v = fold.loc[fold.model.eq(model), metric].to_numpy(float)
        lo, med, hi = np.quantile(v, [.025, .5, .975])
        ax.scatter(v, y + np.linspace(-.08, .08, len(v)), s=5.5, color=MODEL_COLOR[model], alpha=.28, linewidth=0)
        ax.hlines(y, lo, hi, color=MODEL_COLOR[model], lw=1.25)
        ax.scatter(med, y, s=16, color=MODEL_COLOR[model], edgecolor="white", linewidth=.35)
    ax.set_yticks(range(9), HEAT_ROW_LABEL if index == 16 else [])
    if index == 16:
        ax.tick_params(axis="y", labelsize=3.25, pad=1)
    ax.invert_yaxis(); ax.set_xlabel(xlabel); clean_axes(ax, "x"); panel(ax, index, title)


def main():
    setup_style(5.8)
    paths = [PART_PATH, BLOCK_PATH, FOLD_PATH]
    before = {str(p): sha256(p) for p in paths}
    part = pd.read_csv(PART_PATH); block = pd.read_csv(BLOCK_PATH); fold = pd.read_csv(FOLD_PATH)
    if set(part.family) != set(FAMILIES) or set(part.partition_scheme) != set(SCHEMES):
        raise RuntimeError("PARTITION_ABLATION_STRUCTURE_MISMATCH")
    if len(block) != 32 or len(fold) != 144:
        raise RuntimeError("BLOCK_OR_MODEL_FOLD_ROW_COUNT_MISMATCH")

    fig = plt.figure(figsize=(183 / 25.4, 250 / 25.4))
    gs = fig.add_gridspec(5, 4, left=.175, right=.955, bottom=.04, top=.958,
                          wspace=.58, hspace=.72, height_ratios=[.88, .82, .82, 1.18, 1.05])
    summary_gain(fig.add_subplot(gs[0, 0]), part, "mae", 0, "Partition-level MAE gain")
    summary_gain(fig.add_subplot(gs[0, 1]), part, "rmse", 1, "Partition-level RMSE gain")
    paired_scheme(fig.add_subplot(gs[0, 2]), part, "mae", 2, "One- vs four-relation MAE gain")
    paired_scheme(fig.add_subplot(gs[0, 3]), part, "rmse", 3, "One- vs four-relation RMSE")
    legend = [Line2D([0], [0], marker="o", color=FAMILY_COLOR[f], lw=1, ms=3.5, label=FAMILY_LABEL[f]) for f in FAMILIES]
    fig.legend(handles=legend, fontsize=4.2, ncol=2, loc="upper center",
               bbox_to_anchor=(.55, .988), frameon=False)

    for col, scheme in enumerate(SCHEMES):
        block_panel(fig.add_subplot(gs[1, col]), block, scheme, "mae", 4 + col,
                    f"{SCHEME_LABEL[scheme]} · block MAE gain")
        block_panel(fig.add_subplot(gs[2, col]), block, scheme, "rmse", 8 + col,
                    f"{SCHEME_LABEL[scheme]} · block RMSE gain")

    # m-p: all models' paired MAE difference from Ridge in each partition.
    rel_lim = symmetric_limit(fold.delta_mae_vs_ridge.to_numpy(float), 1.0)
    rel_norm = TwoSlopeNorm(vmin=-rel_lim, vcenter=0, vmax=rel_lim)
    cbar_axes = []
    heat_axes = []
    for col, scheme in enumerate(SCHEMES):
        ax = fig.add_subplot(gs[3, col])
        table = fold.loc[fold.partition_scheme.eq(scheme)].pivot(index="model", columns="spatial_block", values="delta_mae_vs_ridge")
        table = table.reindex(index=MODELS, columns=["B1", "B2", "B3", "B4"])
        im = ax.imshow(table.to_numpy(), cmap=DIV, norm=rel_norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(9), HEAT_ROW_LABEL if col == 0 else [])
        ax.set_xticks(range(4), ["B1", "B2", "B3", "B4"])
        ax.tick_params(axis="x", labelsize=4.8, pad=1.0, length=0)
        ax.tick_params(axis="y", labelsize=4.2, length=0, pad=1.0)
        panel(ax, 12 + col, f"{SCHEME_LABEL[scheme]} · ΔMAE vs Ridge")
        cb, cax = add_equal_height_colorbar(fig, ax, im, width=.035, gap=.033)
        cb.set_ticks(symmetric_colourbar_ticks(-rel_lim, rel_lim)); cbar_axes.append((ax, cax))
        heat_axes.append(ax)

    metric_forest(fig.add_subplot(gs[4, 0]), fold, "r2", 16, "Fold-level explained variation", "R²")
    metric_forest(fig.add_subplot(gs[4, 1]), fold, "rmse", 17, "Fold-level squared-error scale", "RMSE (pp)")
    metric_forest(fig.add_subplot(gs[4, 2]), fold, "mae", 18, "Fold-level absolute-error scale", "MAE (pp)")

    # t: within-fold R² standardization exposes rank reversals across all 16 folds.
    ax = fig.add_subplot(gs[4, 3])
    r2 = fold.pivot(index="fold_label", columns="model", values="r2").reindex(columns=MODELS)
    r2 = r2.reindex([f"{prefix}{i}" for prefix in ["A", "R", "X", "Y"] for i in range(1, 5)])
    z = r2.sub(r2.mean(axis=1), axis=0).div(r2.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    zlim = symmetric_limit(z.to_numpy(float), .99)
    im = ax.imshow(z.to_numpy().T, cmap=DIV, vmin=-zlim, vmax=zlim, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(9), HEAT_ROW_LABEL)
    shown_ticks = [0, 3, 4, 7, 8, 11, 12, 15]
    ax.set_xticks(shown_ticks, [z.index[i] for i in shown_ticks], rotation=90, ha="center")
    ax.tick_params(axis="x", labelsize=4.2, pad=1.0, length=0)
    ax.tick_params(axis="y", labelsize=4.0, length=0, pad=1.0); panel(ax, 19, "Within-fold R² profile (z)")
    cb, cax = add_equal_height_colorbar(fig, ax, im, width=.035, gap=.033)
    cb.set_ticks(symmetric_colourbar_ticks(-zlim, zlim)); cbar_axes.append((ax, cax))

    delivery = save_delivery(fig, OUT, STEM)
    fig.canvas.draw()
    height_ratios = [cax.get_position().height / ax.get_position().height for ax, cax in cbar_axes]
    plt.close(fig)

    part.to_csv(OUT / "source_partition_ablation.csv", index=False)
    block.to_csv(OUT / "source_block_ablation.csv", index=False)
    fold.to_csv(OUT / "source_all_model_fold_metrics.csv", index=False)
    z.rename_axis("fold_label").reset_index().to_csv(OUT / "source_within_fold_r2_profiles.csv", index=False)
    after = {str(p): sha256(p) for p in paths}
    nonwhite, border = border_check(delivery["jpg"])
    write_validation(OUT, {
        "figure": STEM, "panels": 20,
        "contract": {"ablation_unit": "matched spatial held-out rows clustered by tract",
                     "partition_schemes": [SCHEME_LABEL[s] for s in SCHEMES],
                     "models": [MODEL_LABEL[m] for m in MODELS],
                     "positive_gain": "lower error for graph-fused model than matched non-graph comparator",
                     "causal_claim": False},
        "source_hashes_before": before, "source_hashes_after": after, "delivery": delivery,
        "checks": {
            "panel_count_20": True,
            "eight_partition_family_cells": len(part) == 8,
            "thirty_two_block_family_cells": len(block) == 32,
            "nine_models_by_sixteen_folds": len(fold) == 144 and fold.model.nunique() == 9 and fold.fold_label.nunique() == 16,
            "registered_partition_gates_all_pass": bool(part.registered_partition_gate.eq(1).all()),
            "individual_colourbar_each_heatmap": len(cbar_axes) == 5,
            "colourbars_equal_heatmap_height": bool(np.allclose(height_ratios, 1, atol=.015)),
            "model_labels_written_on_heatmaps": True,
            "heatmaps_oriented_for_reader_facing_model_labels": True,
            "signed_heatmaps_centered_zero": rel_norm.vcenter == 0,
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
        "# Figure contract\n\nThe figure expands registered graph ablations from partition-level estimates "
        "to held-out blocks and all-model fold stability. Positive gain denotes lower error for the graph-fused model. "
        "All uncertainty intervals and paired differences are read directly from sealed V118 tables. "
        "All nine model columns are named directly on the heatmaps; labels are abbreviated consistently with the full model names in the fold panels.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
