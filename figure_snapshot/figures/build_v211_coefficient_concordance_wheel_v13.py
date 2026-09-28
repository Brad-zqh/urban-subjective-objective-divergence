"""Add same-hue left-to-right tonal transitions in Fig. 8 support panels."""
from __future__ import annotations

import json

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.collections import LineCollection, PathCollection

import build_v211_coefficient_concordance_wheel_v12 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v13"
STEM = "Fig_v211_coefficient_concordance_wheel_v13"
OLD = ROOT / "figures/v211_coefficient_concordance_wheel_v12"


def tint(rgb, fraction):
    base = np.asarray(rgb[:3], dtype=float)
    return 1 - (.46 + .50 * fraction) * (1 - base)


def refine_support(ax) -> dict:
    xlo, xhi = map(float, ax.get_xlim())
    span = max(xhi - xlo, 1e-9)
    segment_count = 0
    for old in [c for c in ax.collections if isinstance(c, LineCollection)]:
        original = old.get_segments()
        raw_colors = old.get_colors()
        widths = old.get_linewidths()
        pieces, colors, piece_widths = [], [], []
        for index, segment in enumerate(original):
            if len(segment) != 2:
                continue
            base = raw_colors[min(index, len(raw_colors) - 1)]
            width = widths[min(index, len(widths) - 1)]
            for step in range(32):
                start = segment[0] + (segment[1] - segment[0]) * step / 32
                end = segment[0] + (segment[1] - segment[0]) * (step + 1) / 32
                xmid = (float(start[0]) + float(end[0])) / 2
                pieces.append([start, end])
                colors.append((*tint(base, np.clip((xmid - xlo) / span, 0, 1)), base[3]))
                piece_widths.append(width)
        if pieces:
            ax.add_collection(LineCollection(pieces, colors=colors,
                                             linewidths=piece_widths,
                                             capstyle="round", zorder=old.get_zorder()))
            old.remove()
            segment_count += len(original)

    recolored = 0
    for collection in [c for c in ax.collections if isinstance(c, PathCollection)]:
        positions = collection.get_offsets()
        facecolors = collection.get_facecolors()
        if len(positions) == 0 or len(facecolors) == 0:
            continue
        revised = []
        for index, point in enumerate(positions):
            original = facecolors[min(index, len(facecolors) - 1)]
            fraction = np.clip((float(point[0]) - xlo) / span, 0, 1)
            revised.append((*tint(original, fraction), original[3]))
        collection.set_facecolors(revised)
        recolored += len(positions)
    ax.set_xlim(xlo, xhi)
    return {"original_line_segments": segment_count, "recoloured_points": recolored}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    base = prior.v11.v10.base.base
    raw_save = base.save_jpg_svg
    statistics = {}

    def support_gradient_save(fig):
        if len(fig.axes) != 8:
            raise RuntimeError("Fig. 8 must retain six panels and two colourbars")
        for letter, ax in zip("cdef", fig.axes[4:8]):
            statistics[letter] = refine_support(ax)
        if not all(entry["recoloured_points"] > 0 for entry in statistics.values()):
            raise RuntimeError("Support panels lack their registered observations")
        return raw_save(fig)

    base.save_jpg_svg = support_gradient_save
    prior.OUT, prior.STEM = OUT, STEM
    prior.main()
    for path in OLD.glob("source_*.csv"):
        if sha256(path) != sha256(OUT / path.name):
            raise RuntimeError(f"Fig. 8 source changed: {path.name}")
    (OUT / "support_gradient_qa.json").write_text(
        json.dumps({"support_panels": statistics,
                    "same_hue_ordered_tint_only": True,
                    "source_tables_byte_identical_to_v12": True},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["support_panel_visual_revision"] = "v13_same_hue_left_to_right_gradients_c_to_f"
        payload["source_data_equal_to_v12"] = True
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
