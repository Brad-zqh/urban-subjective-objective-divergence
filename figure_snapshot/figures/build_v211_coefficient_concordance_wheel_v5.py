"""Compact, larger-type refinement of the paired V211 coefficient wheels.

This is a non-destructive wrapper around the audited v4 renderer.  It keeps
all source values and encodings fixed, enlarges the radial typography, reduces
inter-panel whitespace, and lets the v4 renderer's real-bbox collision gate
reject any radial-label overlap.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_coefficient_concordance_wheel_v4 as base  # noqa: E402
from nature_viz_common import ROOT  # noqa: E402

OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v5"
STEM = "Fig_v211_coefficient_concordance_wheel_v5"
BASE_FONT_PT = 7.05
RADIAL_FONT_PT = 3.25
RADIAL_RADIUS_ADD = 0.0
RADIAL_STAGGER: dict[int, float] = {}
RADIAL_ANGLE_SHIFT: dict[int, float] = {}
SCHEME_FONT_PT = 4.95
AUX_FONT_PT = 4.80
GROUP_FONT_PT = 5.20
CANVAS_HEIGHT_MM = 156
GRID_WSPACE = .18


def main() -> None:
    base.OUT = OUT
    base.STEM = STEM

    original_setup = base.setup_style

    def larger_setup(_: float) -> None:
        original_setup(BASE_FONT_PT)

    base.setup_style = larger_setup

    original_figure = base.plt.figure

    def compact_figure(*args, **kwargs):
        kwargs["figsize"] = (183 / 25.4, CANVAS_HEIGHT_MM / 25.4)
        fig = original_figure(*args, **kwargs)
        original_add_gridspec = fig.add_gridspec

        def compact_add_gridspec(*grid_args, **grid_kwargs):
            grid_kwargs.update(
                left=.022,
                right=.992,
                bottom=.052,
                top=.982,
                height_ratios=[1.0, .045, .355],
                width_ratios=[1, 1, 1, 1],
                wspace=GRID_WSPACE,
                hspace=.205,
            )
            return original_add_gridspec(*grid_args, **grid_kwargs)

        fig.add_gridspec = compact_add_gridspec
        return fig

    base.plt.figure = compact_figure

    original_draw_wheel = base.draw_wheel

    def larger_wheel_type(*args, **kwargs):
        labels, geometry = original_draw_wheel(*args, **kwargs)
        ax = args[1]
        label_ids = {id(label) for label in labels}
        max_extra_radius = RADIAL_RADIUS_ADD
        for label_index, label in enumerate(labels):
            theta, radius = label.get_position()
            extra_radius = RADIAL_RADIUS_ADD + RADIAL_STAGGER.get(label_index, 0.0)
            max_extra_radius = max(max_extra_radius, extra_radius)
            theta += math.radians(RADIAL_ANGLE_SHIFT.get(label_index, 0.0))
            label.set_position((theta, radius + extra_radius))
            label.set_fontsize(RADIAL_FONT_PT)
        # Expand the polar limit only for the common outward shift.  Local
        # staggered labels may sit outside that limit with clipping disabled;
        # including their full stagger in the limit would shrink the wheel and
        # create new collisions elsewhere.
        if RADIAL_RADIUS_ADD:
            lower, upper = ax.get_ylim()
            ax.set_ylim(lower, upper + RADIAL_RADIUS_ADD)
        for text_artist in ax.texts:
            if id(text_artist) in label_ids or text_artist.get_fontweight() == "bold":
                continue
            size = float(text_artist.get_fontsize())
            if size <= 4.40:
                text_artist.set_fontsize(AUX_FONT_PT)
            elif size <= 4.55:
                text_artist.set_fontsize(SCHEME_FONT_PT)
            elif size <= 4.85:
                text_artist.set_fontsize(GROUP_FONT_PT)
        geometry["radial_label_font_pt"] = RADIAL_FONT_PT
        geometry["radial_radius_add"] = RADIAL_RADIUS_ADD
        geometry["radial_stagger"] = RADIAL_STAGGER
        geometry["radial_angle_shift_deg"] = RADIAL_ANGLE_SHIFT
        geometry["compact_layout"] = True
        return labels, geometry

    base.draw_wheel = larger_wheel_type

    original_save = base.save_jpg_svg

    def compact_save(fig):
        # The compact grid already provides the minimum safe vertical gap;
        # retain those positions so colourbar labels cannot enter c–f titles.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        colourbar_text = []
        for axis in fig.axes[2:4]:
            colourbar_text.extend([axis.xaxis.label, *axis.get_xticklabels()])
        support_headers = []
        for axis in fig.axes[4:8]:
            support_headers.append(axis.title)
            support_headers.extend(
                artist for artist in axis.texts if artist.get_fontweight() == "bold"
            )
        collisions = []
        for upper in colourbar_text:
            upper_box = upper.get_window_extent(renderer).expanded(1.01, 1.04)
            for lower in support_headers:
                lower_box = lower.get_window_extent(renderer).expanded(1.01, 1.04)
                if upper_box.overlaps(lower_box):
                    collisions.append((upper.get_text(), lower.get_text()))
        if collisions:
            raise RuntimeError(f"COLOURBAR_SUPPORT_TEXT_OVERLAPS={collisions}")
        return original_save(fig)

    base.save_jpg_svg = compact_save
    base.main()

    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["layout_version"] = "paired_wheel_v5_compact_large_type_non_destructive"
        payload["typography"] = {
            "base_font_pt": BASE_FONT_PT,
            "radial_label_font_pt": RADIAL_FONT_PT,
            "scheme_font_pt": SCHEME_FONT_PT,
            "group_font_pt": GROUP_FONT_PT,
            "only_panel_letters_bold": True,
            "radial_label_collision_gate": "renderer bounding boxes",
        }
        payload["checks"]["compact_vertical_spacing"] = True
        payload["checks"]["larger_typography"] = True
        payload["checks"]["colourbar_support_headers_non_overlapping"] = True
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "figure_contract.md").write_text(
        "# Compact paired coefficient-wheel contract\n\n"
        "- Analysis: V211 current proxy coefficients; `proxy_analysis=true`, `formal=false`.\n"
        "- Evidence: two 43-field × 8-scheme open wheels and four supporting diagnostics.\n"
        f"- Layout: minimum-safe spacing on a 183 × {CANVAS_HEIGHT_MM:g} mm canvas.\n"
        "- Typography: larger base and radial type; radial labels pass rendered-bbox collision checks.\n"
        "- Colour: continuous red–white–blue encoding throughout; centres remain empty.\n"
        "- Export: JPG (600 dpi) and editable-text SVG only.\n"
        "- Preservation: v1–v4 scripts and exports remain untouched.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
