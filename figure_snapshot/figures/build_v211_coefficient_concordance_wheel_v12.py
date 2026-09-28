"""V211 paired wheel v12: four finer outer braces and closer colourbars.

This is a non-destructive layout wrapper around v11. It changes no data or
quantitative colour mapping; all previous renderers and exports stay intact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_coefficient_concordance_wheel_v11 as v11  # noqa: E402
from nature_viz_common import ROOT, sha256  # noqa: E402

OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v12"
STEM = "Fig_v211_coefficient_concordance_wheel_v12"
GROUP_BOUNDS = ((0, 9), (10, 19), (20, 29), (30, 42))
BRACKET_LINEWIDTH_PT = .54
COLOURBAR_UP_FIG_FRACTION = .016


def main() -> None:
    v11.OUT = OUT
    v11.STEM = STEM
    v11.GROUP_BOUNDS = GROUP_BOUNDS
    base = v11.v10.base.base
    raw_save = base.save_jpg_svg

    def thin_braces_and_close_colourbars(fig):
        for ax in fig.axes[:2]:
            arcs = 0
            caps = 0
            for line in ax.lines:
                yy = line.get_ydata()
                if len(yy) == 120 and all(abs(float(value) - v11.BRACKET_RADIUS) < 1e-9 for value in yy):
                    line.set_linewidth(BRACKET_LINEWIDTH_PT)
                    arcs += 1
                elif (len(yy) == 2 and
                      abs(float(yy[0]) - (v11.BRACKET_RADIUS - .007)) < 1e-9 and
                      abs(float(yy[1]) - (v11.BRACKET_RADIUS + .012)) < 1e-9):
                    line.set_linewidth(BRACKET_LINEWIDTH_PT)
                    caps += 1
            if arcs != 4 or caps != 8:
                raise RuntimeError(f"EXPECTED_FOUR_ARCS_EIGHT_CAPS={arcs}:{caps}")

        for ax in fig.axes[2:4]:
            p = ax.get_position()
            ax.set_position([p.x0, p.y0 + COLOURBAR_UP_FIG_FRACTION, p.width, p.height])

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        key = next((artist for artist in fig.texts if "objective" in artist.get_text() and "socioeconomic" in artist.get_text()), None)
        if key is None:
            raise RuntimeError("SHARED_GROUP_KEY_MISSING")
        key_box = key.get_window_extent(renderer).expanded(1.01, 1.02)
        for ax in fig.axes[2:4]:
            if key_box.overlaps(ax.get_window_extent(renderer).expanded(1.01, 1.02)):
                raise RuntimeError("COLOURBAR_OVERLAPS_GROUP_KEY")
        for bar in fig.axes[2:4]:
            upper = [bar.xaxis.label, *bar.get_xticklabels()]
            for support in fig.axes[4:8]:
                lower = [support.title, support._left_title, *support.texts]
                for top_artist in upper:
                    if not top_artist.get_visible() or not top_artist.get_text().strip():
                        continue
                    box_a = top_artist.get_window_extent(renderer).expanded(1.01, 1.02)
                    for low_artist in lower:
                        if low_artist.get_visible() and low_artist.get_text().strip() and box_a.overlaps(low_artist.get_window_extent(renderer).expanded(1.01, 1.02)):
                            raise RuntimeError(f"COLOURBAR_SUPPORT_TEXT_OVERLAP={top_artist.get_text()}:{low_artist.get_text()}")
        return raw_save(fig)

    base.save_jpg_svg = thin_braces_and_close_colourbars
    v11.main()

    previous = ROOT / "figures/v211_coefficient_concordance_wheel_v11"
    source_names = (
        "source_pooled_feature_summary.csv",
        "source_registered_scheme_feature_medians_and_deviations.csv",
    )
    equality = {name: sha256(OUT / name) == sha256(previous / name) for name in source_names}
    if not all(equality.values()):
        raise RuntimeError(f"SOURCE_DATA_CHANGED={equality}")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["layout_version"] = "paired_wheel_v12_four_fine_group_braces_closer_colourbars"
        payload["annotation_geometry"]["brackets_for"] = ["O", "S", "D", "E"]
        payload["annotation_geometry"]["derived_D_retains_its_label_without_a_bracket"] = False
        payload["annotation_geometry"]["bracket_line_width_pt"] = BRACKET_LINEWIDTH_PT
        payload["colourbar_upward_shift_figure_fraction"] = COLOURBAR_UP_FIG_FRACTION
        for geometry in payload["wheel_geometry"].values():
            geometry["group_brackets"]["groups"] = ["O", "S", "D", "E"]
            geometry["group_brackets"]["line_width_pt"] = BRACKET_LINEWIDTH_PT
        payload["source_data_equal_to_v11"] = equality
        payload["checks"].pop("three_group_brackets_visible", None)
        payload["checks"].update({
            "four_group_brackets_visible": True,
            "brace_arcs_and_end_caps_counted": True,
            "thin_bracket_lines": True,
            "colourbars_closer_to_wheels": True,
            "colourbars_do_not_overlap_group_key_or_support_titles": True,
            "source_data_equal_to_v11": True,
        })
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "figure_contract.md").write_text(
        "# V211 four-brace paired wheel contract\n\n"
        "- Scientific conclusion and six-panel evidence chain are unchanged from v10/v11. These fitted V211 proxy coefficients are not causal effects.\n"
        "- Four non-quantitative outer circular braces identify O objective, S platform proxy, D signed divergence and E socioeconomic fields; all braces use the same finer neutral line.\n"
        "- Variable names retain v11's closer radial placement; both colourbars move 0.016 figure-height units toward the hero wheels without crossing the group key or support-panel text.\n"
        "- Exact source CSVs, zero-centred P90 wheel display scales, supporting panels and red–blue sign semantics remain unchanged.\n"
        "- Export is JPG at 600 dpi plus editable-text SVG, no PDF; previous versions and embedded manuscript figures remain untouched.\n"
        "- `proxy_analysis=true`, `formal=false`; platform expression is not resident self-report.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
