"""Fig. 2 bottom-band sequential tints within each registered model row.

Only p–r presentation changes; all map data and mapped panels use the v13 path.
Colour lightness follows horizontal outcome position, not an extra estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb

import build_v211_maps_fullheight_v13 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_map_fullheight_v14_monochrome_rows"
PREVIOUS = ROOT / "figures/v211_map_fullheight_v13"


def tint(colour: str, strength: float) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(colour))
    return tuple(1.0 - strength * (1.0 - base))


def gradient_rail(ax, summary: pd.DataFrame, field: str,
                  xlim: tuple[float, float]) -> None:
    for row, (model, colour) in enumerate(zip(prior.MODELS, prior.MODEL_COLOURS)):
        y = 4 - row
        sub = summary.loc[summary.model.eq(model)].set_index("health_year")
        values = {year: float(sub.loc[year, field]) for year in (2014, 2019, 2023)}
        ax.hlines(y, *xlim, color=prior.GRID, linewidth=.68, zorder=0)
        lo, hi = sorted((values[2014], values[2023]))
        xs = np.linspace(lo, hi, 65)
        segments = np.stack(
            [np.column_stack([xs[:-1], np.full(64, y)]),
             np.column_stack([xs[1:], np.full(64, y)])], axis=1
        )
        blend = np.linspace(.24, .98, 64)
        colours = [tint(colour, float(weight)) for weight in blend]
        ax.add_collection(LineCollection(segments, colors=colours, linewidths=2.5,
                                         capstyle="round", zorder=2))
        ax.scatter([values[2014]], [y], s=36, facecolors="white",
                   edgecolors=colour, linewidths=1.45, zorder=4)
        ax.scatter([values[2019]], [y], s=27, marker="D",
                   facecolors=tint(colour, .65), edgecolors="white",
                   linewidths=.5, zorder=5)
        ax.scatter([values[2023]], [y], s=54, facecolors=colour,
                   edgecolors="white", linewidths=.7, zorder=6)
        delta = values[2023] - values[2014]
        ax.text(xlim[1] - .04 * (xlim[1] - xlim[0]), y + .19,
                f"{delta:+.2f}", ha="right", va="bottom", fontsize=6.15,
                color=colour, fontweight="medium")


def gradient_distribution(ax) -> None:
    raw = pd.read_csv(prior.SOURCE)
    raw["model"] = raw.model.replace({"OLS/Ridge": "OLS anchor"})
    arrays = [raw.loc[raw.model.eq(model) & raw.health_year.eq(2023),
                      "prediction"].dropna().to_numpy(float)
              for model in prior.MODELS]
    if any(array.size < 800 for array in arrays):
        raise RuntimeError("Fig. 2 distribution input unexpectedly incomplete")
    violins = ax.violinplot(arrays, positions=np.arange(4, -1, -1),
                            orientation="horizontal", widths=.65, showmeans=False,
                            showmedians=False, showextrema=False, points=150,
                            bw_method=.22)
    xlow, xhigh = 6.4, 26.3
    for row, (array, colour, body) in enumerate(zip(
        arrays, prior.MODEL_COLOURS, violins["bodies"]
    )):
        y = 4 - row
        path = body.get_paths()[0]
        cmap = LinearSegmentedColormap.from_list(
            f"row_{row}", [tint(colour, .19), tint(colour, .56), tint(colour, .98)]
        )
        gradient = np.linspace(0, 1, 512)[None, :]
        ax.imshow(gradient, extent=(xlow, xhigh, y - .35, y + .35),
                  cmap=cmap, interpolation="bilinear", aspect="auto",
                  clip_path=(path, ax.transData), clip_on=True, zorder=2)
        body.set_facecolor("none")
        body.set_edgecolor(colour)
        body.set_linewidth(.8)
        body.set_alpha(.85)
        body.set_zorder(3)
        q25, median, q75 = np.quantile(array, [.25, .5, .75])
        ax.plot([q25, q75], [y, y], color="white", lw=3.1,
                solid_capstyle="round", zorder=4)
        ax.plot([q25, q75], [y, y], color=colour, lw=.7,
                solid_capstyle="round", zorder=5)
        ax.scatter([median], [y], s=31, facecolors="white",
                   edgecolors=colour, linewidths=1.45, zorder=6)
    ax.set_xlim(xlow, xhigh)
    ax.set_xticks([8, 12, 16, 20, 24])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior._rail = gradient_rail
    prior._distribution = gradient_distribution
    prior.main()
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():
            continue
        for old in list(folder.glob("*.jpg")) + list(folder.glob("*.svg")):
            old.rename(folder / old.name.replace("_v13", "_v14"))
        for name in ("validation.json", "manifest.json"):
            path = folder / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["figure"] = str(payload["figure"]).replace("_v13", "_v14")
            for kind in ("jpg", "svg"):
                if kind in payload.get("delivery", {}):
                    payload["delivery"][kind] = str(payload["delivery"][kind]).replace(
                        "_v13", "_v14"
                    )
            payload["layout_version"] = "fullheight_v14_monochrome_row_gradients"
            if name == "validation.json" and folder.name == "fig2_spatial_prediction":
                source_name = "source_temporal_spatial_summaries.csv"
                payload["checks"].update({
                    "row_gradients_only_in_p_q_r": True,
                    "annual_summary_source_byte_identical_to_v13":
                        prior._hash(PREVIOUS / folder.name / source_name) ==
                        prior._hash(folder / source_name),
                    "registered_prediction_source_present": prior.SOURCE.is_file(),
                })
                payload["all_checks_passed"] = all(payload["checks"].values())
                if not payload["all_checks_passed"]:
                    raise RuntimeError(f"Fig. 2 v14 QA failed: {payload['checks']}")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    root = OUT / "aligned_manifest.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["layout_version"] = "fullheight_v14_monochrome_row_gradients"
    payload["outputs"] = [str(path) for path in OUT.rglob("*.jpg")]
    root.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
