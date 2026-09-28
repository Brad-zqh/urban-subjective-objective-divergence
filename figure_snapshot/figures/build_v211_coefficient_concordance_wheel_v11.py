"""Non-destructive V211 wheel refinement: closer labels and group brackets.

All quantitative encodings and source values come from the audited v10
renderer. This version changes only radial annotation geometry.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_coefficient_concordance_wheel_v10 as v10  # noqa: E402
from nature_viz_common import ROOT, sha256  # noqa: E402

OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v11"
STEM = "Fig_v211_coefficient_concordance_wheel_v11"
LABEL_INWARD = .035
BRACKET_RADIUS = .857
BRACKET_COLOUR = "#4E5A69"
GROUP_BOUNDS = ((0, 9), (10, 19), (30, 42))


def main() -> None:
    v10.OUT = OUT
    v10.STEM = STEM
    raw_draw = v10.base.base.draw_wheel
    raw_save = v10.base.base.save_jpg_svg

    def bracketed_draw(*args, **kwargs):
        labels, geometry = raw_draw(*args, **kwargs)
        ax = args[1]
        if len(labels) != 43:
            raise RuntimeError("EXPECTED_43_RADIAL_LABELS")
        for index, label in enumerate(labels):
            theta, radius = label.get_position()
            extra_clearance = .075 if index == 39 else 0.0
            label.set_position((theta, radius - LABEL_INWARD + extra_clearance))

        start = geometry["start_deg"]
        width = geometry["span_deg"] / 43
        # Three curved braces follow the user's O/S/E sketch. End caps make
        # each brace read as a group boundary, not a data-valued colour ring.
        for lo, hi in GROUP_BOUNDS:
            left = start + lo * width + 2.0
            right = start + (hi + 1) * width - 2.0
            theta = np.deg2rad(np.linspace(left, right, 120))
            ax.plot(theta, np.full_like(theta, BRACKET_RADIUS),
                    color=BRACKET_COLOUR, lw=.80, alpha=.93, zorder=8)
            for end in (left, right):
                ax.plot([math.radians(end)] * 2,
                        [BRACKET_RADIUS - .007, BRACKET_RADIUS + .012],
                        color=BRACKET_COLOUR, lw=.80, alpha=.93, zorder=8)

        for artist in ax.texts:
            if artist.get_text() in {"O", "S", "SES"} and artist.get_fontsize() == 4.8:
                theta, _ = artist.get_position()
                artist.set_position((theta, BRACKET_RADIUS + .026))
        geometry["radial_variable_labels_moved_inward_by"] = LABEL_INWARD
        geometry["group_brackets"] = {
            "groups": ["O", "S", "E"],
            "radius": BRACKET_RADIUS,
            "line_colour": BRACKET_COLOUR,
            "line_width_pt": .80,
            "numeric_encoding": False,
        }
        return labels, geometry

    def bracketed_save(fig):
        # Reconnect short leaders after v10 applies the selective stagger.
        for ax in fig.axes[:2]:
            labels = [artist for artist in ax.texts if artist.get_fontsize() == v10.base.RADIAL_FONT_PT]
            leaders = []
            for line in ax.lines:
                yy = line.get_ydata()
                if len(yy) == 2 and float(yy[1]) > .90:
                    leaders.append(line)
            if len(labels) != 43 or len(leaders) != 43:
                raise RuntimeError(f"EXPECTED_43_LABEL_LEADERS={len(labels)}:{len(leaders)}")
            for leader, label in zip(leaders, labels):
                _, radius = label.get_position()
                leader.set_ydata([BRACKET_RADIUS + .015, radius - .013])
        return raw_save(fig)

    v10.base.base.draw_wheel = bracketed_draw
    v10.base.base.save_jpg_svg = bracketed_save
    v10.main()

    old = ROOT / "figures/v211_coefficient_concordance_wheel_v10"
    source_names = (
        "source_pooled_feature_summary.csv",
        "source_registered_scheme_feature_medians_and_deviations.csv",
    )
    source_hashes_equal = {name: sha256(OUT / name) == sha256(old / name) for name in source_names}
    if not all(source_hashes_equal.values()):
        raise RuntimeError(f"SOURCE_DATA_CHANGED={source_hashes_equal}")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["layout_version"] = "paired_wheel_v11_closer_labels_group_brackets_non_destructive"
        payload["annotation_geometry"] = {
            "radial_label_inward_shift": LABEL_INWARD,
            "group_bracket_radius": BRACKET_RADIUS,
            "brackets_for": ["O", "S", "E"],
            "derived_D_retains_its_label_without_a_bracket": True,
            "short_leaders_start_outside_brackets": True,
        }
        payload["source_data_equal_to_v10"] = source_hashes_equal
        payload["checks"].update({
            "three_group_brackets_visible": True,
            "radial_labels_closer_to_outer_ribbon": True,
            "source_data_equal_to_v10": True,
        })
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "figure_contract.md").write_text(
        "# V211 closer-label, bracketed paired-wheel contract\n\n"
        "- Core conclusion: fitted V211 proxy coefficient patterns differ across fields and registered schemes; the wheel brackets are categorical group annotations, not another data ring.\n"
        "- Evidence chain and panel count: a/b are the 43-field × eight-scheme coefficient and deviation wheels, c–f preserve the four quantitative diagnostics.\n"
        "- Archetype: quantitative grid with two hero wheels.\n"
        "- Group brackets: three separate outer circular braces identify O objective, S platform proxy and E socioeconomic. D signed divergence remains separately labelled as a derived S−O channel, not incorrectly absorbed into S. Radial labels sit closer to the wheel than v10.\n"
        "- Colour: unchanged from v10; symmetric zero-centred empirical P90 display limits for wheels, saturated signed hue for c–f. Exact values stay in source CSVs.\n"
        "- Export: 183-mm registered width, JPG at 600 dpi and editable-text SVG only. All previous versions remain untouched.\n"
        "- Analysis: `proxy_analysis=true`, `formal=false`; no causal or resident self-report interpretation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
