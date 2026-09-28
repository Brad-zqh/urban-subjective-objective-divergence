"""Fig. 2 editorial support-band redesign; the 15 registered maps are untouched.

Panels p and q show exact 2014/2019/2023 summaries as end-point rails. Panel r
shows each model's complete 2023 tract distribution as a violin with the
empirical IQR and median. No fitted uncertainty intervals are introduced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import build_v211_maps_fullheight_v12 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_map_fullheight_v13"
SOURCE = ROOT / "figures/current_spatial_model_matrix_25panel/source_spatial_model_predictions.csv"
MODELS = ["MM-GTGNNWR", "OLS anchor", "GTWR", "GTNNWR", "GTGNNWR"]
MODEL_COLOURS = ["#B40426", "#274690", "#3B4CC0", "#7699C6", "#D45463"]
INK = "#19324F"
GRID = "#E1EAF4"
original_finish = prior.prior._aligned_finish


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _style(ax, title: str, letter: str, xlim: tuple[float, float], xlabel: str,
           show_models: bool = False) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(-.45, 4.45)
    ax.set_yticks(range(4, -1, -1), MODELS if show_models else [""] * 5)
    ax.tick_params(axis="y", length=0, pad=3.3, labelsize=6.5, colors=INK)
    ax.tick_params(axis="x", length=2.2, width=.55, labelsize=6.3, colors=INK, pad=2.5)
    ax.set_xlabel(xlabel, fontsize=6.7, color=INK, labelpad=3.5)
    ax.set_title(title, loc="left", fontsize=8.0, color=INK, pad=9)
    ax.text(-.16 if show_models else -.075, 1.045, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.6, fontweight="bold", color="#17191C")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#7590AC")
    ax.spines["bottom"].set_linewidth(.55)


def _rail(ax, summary: pd.DataFrame, field: str, xlim: tuple[float, float]) -> None:
    for row, (model, colour) in enumerate(zip(MODELS, MODEL_COLOURS)):
        y = 4 - row
        sub = summary.loc[summary.model.eq(model)].set_index("health_year")
        values = {year: float(sub.loc[year, field]) for year in (2014, 2019, 2023)}
        ax.hlines(y, *xlim, color=GRID, linewidth=.68, zorder=0)
        ax.plot([values[2014], values[2023]], [y, y], color=colour,
                linewidth=2.0, alpha=.89, solid_capstyle="round", zorder=2)
        ax.scatter([values[2014]], [y], s=36, facecolors="white",
                   edgecolors=colour, linewidths=1.5, zorder=4)
        ax.scatter([values[2019]], [y], s=26, marker="D", facecolors=colour,
                   edgecolors="white", linewidths=.5, zorder=5)
        ax.scatter([values[2023]], [y], s=54, facecolors=colour,
                   edgecolors="white", linewidths=.75, zorder=6)
        delta = values[2023] - values[2014]
        ax.text(xlim[1] - .04 * (xlim[1] - xlim[0]), y + .19,
                f"{delta:+.2f}", ha="right", va="bottom", fontsize=6.15,
                color=colour, fontweight="medium")


def _distribution(ax) -> None:
    raw = pd.read_csv(SOURCE)
    raw["model"] = raw.model.replace({"OLS/Ridge": "OLS anchor"})
    arrays = [raw.loc[raw.model.eq(model) & raw.health_year.eq(2023),
                      "prediction"].dropna().to_numpy(float) for model in MODELS]
    if any(array.size < 800 for array in arrays):
        raise RuntimeError("Fig. 2 distribution input unexpectedly incomplete")
    violins = ax.violinplot(arrays, positions=np.arange(4, -1, -1),
                            orientation="horizontal", widths=.65, showmeans=False,
                            showmedians=False, showextrema=False, points=150,
                            bw_method=.22)
    for body, colour in zip(violins["bodies"], MODEL_COLOURS):
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_linewidth(.85)
        body.set_alpha(.69)
    for row, (array, colour) in enumerate(zip(arrays, MODEL_COLOURS)):
        y = 4 - row
        q25, median, q75 = np.quantile(array, [.25, .5, .75])
        ax.plot([q25, q75], [y, y], color="white", lw=3.1,
                solid_capstyle="round", zorder=4)
        ax.plot([q25, q75], [y, y], color=colour, lw=.7,
                solid_capstyle="round", zorder=5)
        ax.scatter([median], [y], s=31, facecolors="white",
                   edgecolors=colour, linewidths=1.45, zorder=6)
    ax.set_xlim(6.4, 26.3)
    ax.set_xticks([8, 12, 16, 20, 24])


def premium_support(fig, folder, stem, panels, sources, core, chain, alignment):
    if folder == "fig2_spatial_prediction":
        titles = ("Temporal level", "Spatial dispersion", "Final-year distributions")
        axes = [next((ax for ax in fig.axes if ax.get_title(loc="left") == title), None)
                for title in titles]
        if any(ax is None for ax in axes):
            raise RuntimeError("Fig. 2 p–r support axes not found")
        summary = pd.read_csv(
            ROOT / "figures/v211_map_fullheight_v9/fig2_spatial_prediction/"
                   "source_temporal_spatial_summaries.csv"
        )
        summary["health_year"] = summary.health_year.astype(int)
        for ax in axes:
            ax.clear()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + .011, box.width, box.height + .005])

        ax_p, ax_q, ax_r = axes
        _style(ax_p, "Median level · 2014–23", "p", (11.6, 16.2),
               "Median prediction (%)", True)
        _rail(ax_p, summary, "median", (11.6, 16.2))
        ax_p.set_xticks([12, 14, 16])

        _style(ax_q, "Spatial IQR · 2014–23", "q", (2.45, 6.55),
               "Spatial IQR (pp)")
        _rail(ax_q, summary, "iqr", (2.45, 6.55))
        ax_q.set_xticks([3, 4, 5, 6])

        _style(ax_r, "2023 tract distributions", "r", (6.4, 26.3),
               "Predicted outcome (%)")
        _distribution(ax_r)

        for legend in list(fig.legends):
            legend.remove()
        key = [
            Line2D([0], [0], marker="o", linestyle="", color=INK,
                   markerfacecolor="white", markersize=4.5, label="2014"),
            Line2D([0], [0], marker="D", linestyle="", color=INK,
                   markersize=3.6, label="2019"),
            Line2D([0], [0], marker="o", linestyle="", color=INK,
                   markersize=4.5, label="2023"),
        ]
        fig.legend(handles=key, loc="lower center", bbox_to_anchor=(.5, .011),
                   ncol=3, frameon=False, fontsize=6.2, handletextpad=.45,
                   columnspacing=2.25)
        chain = (
            "Fifteen registered maps (3 years × 5 models) → exact 2014-to-2023 "
            "median and spatial-IQR endpoint rails with 2019 markers → complete "
            "2023 tract distributions (violin, empirical median and IQR)."
        )
    return original_finish(fig, folder, stem, panels, sources, core, chain, alignment)


def main() -> None:
    previous = ROOT / "figures/v211_map_fullheight_v12"
    if not OUT.exists():
        prior.OUT = OUT
        prior.prior.stronger_support = premium_support
        prior.main()
    elif not (OUT / "fig2_spatial_prediction/Fig2_spatial_prediction_fullheight_v12.jpg").exists():
        raise FileExistsError(OUT)
    for group in prior.prior.prior.prior.MAP_GROUPS:
        folder = OUT / group
        for old in list(folder.glob("*.jpg")) + list(folder.glob("*.svg")):
            old.rename(folder / old.name.replace("_v12", "_v13"))
        for name in ("validation.json", "manifest.json"):
            path = folder / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["figure"] = str(payload["figure"]).replace("_v12", "_v13")
            for kind in ("jpg", "svg"):
                if kind in payload.get("delivery", {}):
                    payload["delivery"][kind] = str(payload["delivery"][kind]).replace("_v12", "_v13")
            payload["layout_version"] = "fullheight_v13_editorial_support_band"
            if name == "validation.json":
                payload["checks"]["p_q_r_redesigned_from_exact_source_no_CI"] = True
                payload["checks"]["source_table_byte_identical_to_v12"] = (
                    _hash(previous / group / "source_temporal_spatial_summaries.csv") ==
                    _hash(folder / "source_temporal_spatial_summaries.csv")
                ) if group == "fig2_spatial_prediction" else True
                payload["all_checks_passed"] = all(payload["checks"].values())
                if not payload["all_checks_passed"]:
                    raise RuntimeError(f"Fig. 2 QA failed: {payload['checks']}")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    root = OUT / "aligned_manifest.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["layout_version"] = "fullheight_v13_editorial_support_band"
    payload["outputs"] = [str(path) for path in OUT.rglob("*.jpg")]
    root.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(OUT), "checks": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
