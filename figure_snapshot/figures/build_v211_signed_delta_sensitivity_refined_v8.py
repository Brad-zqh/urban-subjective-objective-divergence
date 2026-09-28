"""Soften the four signed Fig. 4 heatmaps and separate their colour bars."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

import build_v211_signed_delta_sensitivity_refined_v7 as prior


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_signed_delta_sensitivity_refined_v7"
OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v8"
PALETTE = LinearSegmentedColormap.from_list(
    "v211_soft_signed_white_zero",
    [(0, "#7295C7"), (.27, "#B7CBE5"), (.46, "#EDF3FA"),
     (.50, "#FFFFFF"), (.54, "#FAEAEB"), (.73, "#EAB1B8"), (1, "#D27684")],
)
_save = prior._save


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_v8(fig, output, *, dpi=600):
    gaps = []
    for ax in fig.axes[8:12]:
        if len(ax.images) != 1 or len(ax.child_axes) != 1:
            raise RuntimeError("Expected one heatmap and one independent colour bar")
        ax.images[0].set_cmap(PALETTE)
        colourbar = ax.child_axes[0]
        # The original inset locator pins the bar at 1.02 axis widths. Replace
        # only its locator, preserving the heatmap geometry and bar height.
        colourbar.set_axes_locator(None)
        pos = ax.get_position()
        colourbar.set_position([pos.x0 + 1.16 * pos.width, pos.y0,
                                .028 * pos.width, pos.height])
        gaps.append((colourbar.get_position().x0 - pos.x1) / pos.width)
    if min(gaps) < .15:
        raise RuntimeError("Heatmap colour bars did not move far enough")
    return _save(fig, output, dpi=dpi)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.save_v7 = save_v8
    prior.main()
    for source in OLD.glob("source_*.csv"):
        if digest(source) != digest(OUT / source.name):
            raise RuntimeError(f"Fig. 4 source changed: {source.name}")
    path = OUT / "validation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checks"].pop("four_signed_heatmaps_use_vivid_white_zero_scale", None)
    data["checks"].update({
        "four_signed_heatmaps_use_soft_white_zero_scale": True,
        "independent_colourbars_offset_sixteen_percent_axis_width": True,
        "source_csv_byte_identical_to_v7": True,
    })
    data["all_checks_passed"] = all(data["checks"].values())
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = OUT / "manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    meta["visual_revision"] = "v8_lighter_signed_heatmaps_and_clear_colourbar_gutter"
    manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not data["all_checks_passed"]:
        raise RuntimeError("Fig. 4 v8 QA failed")


if __name__ == "__main__":
    main()
