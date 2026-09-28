"""Widen V154 map colourbars and clear summary-label/map-title collisions."""
from __future__ import annotations

import json

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.axes import Axes

import build_v154_known_truth_maintext_v7 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v154_known_truth_maintext_v8"
OLD = ROOT / "figures/v154_known_truth_maintext_v7"
ORIGINAL_INSET = Axes.inset_axes
ORIGINAL_SAVE = prior.ORIGINAL_SAVE


def wider_map_colourbar(self, bounds, *args, **kwargs):
    if (isinstance(bounds, (list, tuple)) and len(bounds) == 4 and
            abs(float(bounds[0]) - 1.085) < 1e-9 and
            abs(float(bounds[2]) - .026) < 1e-9):
        bounds = [bounds[0], bounds[1], .048, bounds[3]]
    return ORIGINAL_INSET(self, bounds, *args, **kwargs)


def clear_top_labels_then_save(fig):
    if len(fig.axes) != 20:
        raise RuntimeError("Expected 20 V154 panels")
    for ax in fig.axes[:2]:
        ax.xaxis.set_label_coords(.5, -.095)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes[:2]:
        label_box = ax.xaxis.label.get_window_extent(renderer)
        for map_ax in fig.axes[2:5]:
            if label_box.overlaps(map_ax.title.get_window_extent(renderer)):
                raise RuntimeError("V154 summary xlabel overlaps first-map title")
    colourbars = [ax.child_axes[0] for ax in fig.axes[2:]]
    if len(colourbars) != 18 or any(abs(cax.get_position().height - ax.get_position().height) > 1e-5
                                    for ax, cax in zip(fig.axes[2:], colourbars)):
        raise RuntimeError("V154 map colourbars must retain exact map height")
    return ORIGINAL_SAVE(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.ORIGINAL_SAVE = clear_top_labels_then_save
    Axes.inset_axes = wider_map_colourbar
    try:
        prior.main()
    finally:
        Axes.inset_axes = ORIGINAL_INSET
    for extension in ("jpg", "svg"):
        (OUT / f"Fig_v154_known_truth_maintext_v7.{extension}").rename(
            OUT / f"Fig_v154_known_truth_maintext_v8.{extension}")
    if sha256(OLD / "source_model_metric_intervals.csv") != sha256(
        OUT / "source_model_metric_intervals.csv"):
        raise RuntimeError("Sealed V154 interval source changed")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["figure"] = "Fig_v154_known_truth_maintext_v8"
        for extension in ("jpg", "svg"):
            data["delivery"][extension] = str(data["delivery"][extension]).replace("_v7.", "_v8.")
        data["v8_visual_change"] = (
            "All eighteen map colourbars widen from 0.026 to 0.048 map-axis widths; "
            "top summary x labels move upward to clear first map-row headings. "
            "No metric, sealed surface, or colour scale changed."
        )
        data["source_interval_sha256_equal_v7"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
