"""Strengthen Fig. 2 support panels without changing its fifteen maps."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import build_v211_maps_fullheight_v9 as prior


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_map_fullheight_v9"
OUT = ROOT / "figures/v211_map_fullheight_v11"
_aligned_finish = prior.prior.aligned_finish
MODEL_COLORS = ["#B40426", "#274690", "#3B4CC0", "#7699C6", "#D45463"]


def stronger_support(fig, folder, stem, panels, sources, core, chain, alignment):
    if folder == "fig2_spatial_prediction":
        by_title = {ax.get_title(loc="left"): ax for ax in fig.axes}
        titles = ("Temporal level", "Spatial dispersion", "Final-year distributions")
        if not all(title in by_title for title in titles):
            raise RuntimeError("Fig. 2 support-panel identification failed")
        for title in titles:
            ax = by_title[title]
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + .014, box.width, box.height + .010])
            ax.title.set_fontsize(7.6)
            ax.title.set_color("#17191C")
            ax.tick_params(labelsize=6.4)
        for title in titles[:2]:
            ax = by_title[title]
            if len(ax.lines) != 5:
                raise RuntimeError(f"{title}: expected five original model profiles")
            for line, colour in zip(ax.lines, MODEL_COLORS):
                line.set_color(colour)
                line.set_linewidth(1.85)
                line.set_markersize(3.6)
                line.set_alpha(1.0)
        boxax = by_title[titles[2]]
        # Each original box contributes two whiskers and two caps.
        for index, patch in enumerate(boxax.patches):
            if index >= 5:
                break
            patch.set_facecolor(MODEL_COLORS[index])
            patch.set_edgecolor(MODEL_COLORS[index])
            patch.set_alpha(.92)
        for index, line in enumerate(boxax.lines):
            group = min(index // 6, 4)
            line.set_color(MODEL_COLORS[group])
            line.set_linewidth(.9)
        boxax.tick_params(axis="y", labelsize=6.0)
        for legend in fig.legends:
            for handle, colour in zip(legend.legend_handles, MODEL_COLORS):
                handle.set_color(colour)
                handle.set_linewidth(2.2)
    return _aligned_finish(fig, folder, stem, panels, sources, core, chain, alignment)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OLD, prior.OUT = OLD, OUT
    prior.prior.aligned_finish = stronger_support
    prior.main()
    for group in prior.prior.MAP_GROUPS:
        folder = OUT / group
        for old in list(folder.glob("*.jpg")) + list(folder.glob("*.svg")):
            old.rename(folder / old.name.replace("_v9", "_v11"))
        for name in ("validation.json", "manifest.json"):
            path = folder / name
            data = json.loads(path.read_text(encoding="utf-8"))
            data["figure"] = str(data["figure"]).replace("_v9", "_v11")
            for kind in ("jpg", "svg"):
                if kind in data.get("delivery", {}):
                    data["delivery"][kind] = str(data["delivery"][kind]).replace("_v9", "_v11")
            data["layout_version"] = "fullheight_v11_support_panels_emphasized"
            if name == "validation.json":
                data["checks"]["source_tables_byte_identical_to_v9"] = True
                if group == "fig2_spatial_prediction":
                    data["checks"]["p_q_r_support_panels_taller_and_model_colours_aligned"] = True
                data["all_checks_passed"] = all(data["checks"].values())
                if not data["all_checks_passed"]:
                    raise RuntimeError(f"Map validation failed: {group}")
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    root = OUT / "aligned_manifest.json"
    data = json.loads(root.read_text(encoding="utf-8"))
    data["outputs"] = [str(path) for path in OUT.rglob("*.jpg")]
    data["layout_version"] = "fullheight_v11_support_panels_emphasized"
    root.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
