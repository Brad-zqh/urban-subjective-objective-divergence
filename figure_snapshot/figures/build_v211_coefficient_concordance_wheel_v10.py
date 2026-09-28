"""V211 paired coefficient wheels with readable variable names (new version).

The full 43-feature × eight-scheme values, two wheels, and four support panels
are retained. Radial labels use reader-facing variable names instead of v9's
O1/S1/D1/E1 IDs. Group cues distinguish repeated names across channels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_coefficient_concordance_wheel_v5 as base  # noqa: E402
from nature_viz_common import ROOT  # noqa: E402

OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v10"
STEM = "Fig_v211_coefficient_concordance_wheel_v10"
GROUP_CODES = {"O", "S", "D", "E"}
BLUE = "#3B4CC0"
RED = "#B40426"
NEUTRAL = "#A7B0BC"


class _DisplayQuantiles:
    """Change only the two wheel display limits, never the source values."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def quantile(values, q, *args, **kwargs):
        if q == .98:
            q = .90
        return np.quantile(values, q, *args, **kwargs)


def main() -> None:
    base.OUT = OUT
    base.STEM = STEM
    base.BASE_FONT_PT = 8.0
    base.RADIAL_FONT_PT = 5.8
    base.RADIAL_RADIUS_ADD = 0.0
    base.RADIAL_STAGGER = {
        2: .145, 4: .175, 12: .120, 13: .150, 14: .300,
        16: .120, 26: .105, 29: .085, 32: .120, 41: .135,
    }
    base.RADIAL_ANGLE_SHIFT = {4: 3.0}
    base.SCHEME_FONT_PT = 6.5
    base.AUX_FONT_PT = 6.3
    base.GROUP_FONT_PT = 6.4
    base.CANVAS_HEIGHT_MM = 168
    base.GRID_WSPACE = .18
    base.base.np = _DisplayQuantiles()

    wide = pd.read_parquet(base.base.WIDE)
    all_features = [column for column in wide.columns if column.startswith(("o_", "s_", "delta_", "ses__"))]
    feature_order = (
        [f"o_d{i}_composite_z" for i in range(1, 11)]
        + [f"s_d{i}_augmented_mm_z" for i in range(1, 11)]
        + [f"delta_d{i}_augmented_signed_s_minus_o" for i in range(1, 11)]
        + [feature for feature in all_features if feature.startswith("ses__")]
    )
    if len(feature_order) != 43 or set(feature_order) != set(all_features):
        raise RuntimeError("V211_FEATURE_ORDER_MISMATCH")
    pooled_medians = np.array([float(wide[feature].median()) for feature in feature_order])
    pooled_median_absolute = np.array([float(np.median(np.abs(wide[feature].to_numpy(float)))) for feature in feature_order])
    near_zero_limit = float(np.quantile(np.abs(pooled_medians), .10))
    group_medians = [float(np.median(pooled_medians[lo:hi])) for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 43))]

    def sign_colour(value):
        if abs(float(value)) <= near_zero_limit:
            return NEUTRAL
        return RED if value > 0 else BLUE

    original_draw = base.base.draw_wheel

    def named_wheel(*args, **kwargs):
        labels, geometry = original_draw(*args, **kwargs)
        if len(labels) != 43:
            raise RuntimeError("EXPECTED_43_NAMED_FIELDS")
        ax = args[1]
        for artist in ax.texts:
            old = artist.get_text()
            if old == "registered schemes":
                artist.set_text("schemes")
            elif old == "S−O" and artist.get_fontsize() == 4.8:
                artist.set_text("D")
            elif old == "SES" and artist.get_fontsize() == 4.8:
                artist.set_text("E")
        geometry["reader_facing_variable_names"] = True
        return labels, geometry

    base.base.draw_wheel = named_wheel
    original_save = base.base.save_jpg_svg

    def audit_and_save(fig):
        # The support panels use saturated sign hues; magnitude remains on the
        # axis, so small signed medians are not falsely promoted by colour.
        strong_ax = fig.axes[4]
        sign_ax = fig.axes[5]
        dist_ax = fig.axes[6]
        scatter_ax = fig.axes[7]
        if len(strong_ax.collections) != 2 or len(sign_ax.collections) != 8 or len(dist_ax.collections) != 12 or len(scatter_ax.collections) != 1:
            raise RuntimeError("UNEXPECTED_SUPPORT_PANEL_ARTIST_COUNT")
        strong_order = np.argsort(pooled_median_absolute)[-10:]
        strong_order = strong_order[np.argsort(pooled_median_absolute[strong_order])]
        strong_colours = [sign_colour(pooled_medians[index]) for index in strong_order]
        strong_ax.collections[0].set_color(strong_colours)
        strong_ax.collections[1].set_facecolors(strong_colours)
        for group_index, median in enumerate(group_medians):
            colour = sign_colour(median)
            sign_ax.collections[2 * group_index].set_color(colour)
            sign_ax.collections[2 * group_index + 1].set_facecolor(colour)
        for group_index, (lo, hi) in enumerate(((0, 10), (10, 20), (20, 30), (30, 43))):
            colours = [sign_colour(value) for value in pooled_medians[lo:hi]]
            dist_ax.collections[3 * group_index].set_facecolors(colours)
            colour = sign_colour(group_medians[group_index])
            dist_ax.collections[3 * group_index + 1].set_color(colour)
            dist_ax.collections[3 * group_index + 2].set_facecolor(colour)
        scatter_ax.collections[0].set_facecolors([sign_colour(value) for value in pooled_medians])
        scatter_ax.collections[0].set_alpha(1.0)
        for handle, colour in zip(scatter_ax.get_legend().legend_handles, (BLUE, NEUTRAL, RED)):
            handle.set_markerfacecolor(colour)

        for ax in (fig.axes[5], fig.axes[6]):
            ax.set_yticklabels(["O", "S", "D", "SES"])
            ax.tick_params(axis="y", labelsize=6.6)
        # These rows line up exactly; label them once at d to avoid crowding
        # the boundary between d and e.
        fig.axes[6].set_yticklabels([])
        for ax in fig.axes[:2]:
            ax.set_title("", loc="left")
        for ax in (fig.axes[2], fig.axes[3]):
            ax.tick_params(axis="x", labelsize=6.3)
            ax.xaxis.label.set_fontsize(6.3)
            ax.xaxis.label.set_text(ax.xaxis.label.get_text() + " (P90 clip)")
        legend = fig.axes[7].get_legend()
        if legend is not None:
            for artist in legend.get_texts():
                artist.set_fontsize(6.3)
        group_key = fig.text(
            .5, .385,
            "O  objective    ·    S  platform proxy    ·    D  signed divergence    ·    E  socioeconomic",
            ha="center", va="center", fontsize=6.4, fontweight="normal",
        )
        left_ticks = fig.axes[2].get_xticklabels()
        right_ticks = fig.axes[3].get_xticklabels()
        left_ticks[-1].set_ha("right")
        right_ticks[0].set_ha("left")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if left_ticks[-1].get_window_extent(renderer).expanded(1.02, 1.04).overlaps(
            right_ticks[0].get_window_extent(renderer).expanded(1.02, 1.04)
        ):
            raise RuntimeError("INNER_COLOURBAR_ENDPOINT_LABELS_OVERLAP")
        key_box = group_key.get_window_extent(renderer).expanded(1.02, 1.04)
        for ax in fig.axes[:4]:
            for artist in [*ax.texts, ax.xaxis.label, *ax.get_xticklabels()]:
                if artist.get_visible() and artist.get_text().strip() and key_box.overlaps(artist.get_window_extent(renderer)):
                    raise RuntimeError(f"SHARED_GROUP_KEY_OVERLAP={artist.get_text()}")
        for ax in fig.axes[:2]:
            groups = [artist for artist in ax.texts if artist.get_text() in GROUP_CODES]
            labels = [artist for artist in ax.texts if artist not in groups and artist.get_fontsize() == base.RADIAL_FONT_PT]
            if len(groups) != 4 or len(labels) != 43:
                raise RuntimeError("EXPECTED_GROUP_CODES_AND_43_VARIABLE_NAMES")
            for name in labels:
                nbox = name.get_window_extent(renderer).expanded(1.01, 1.02)
                for group in groups:
                    if nbox.overlaps(group.get_window_extent(renderer).expanded(1.01, 1.02)):
                        raise RuntimeError(f"GROUP_VARIABLE_OVERLAP={name.get_text()}:{group.get_text()}")
        return original_save(fig)

    base.base.save_jpg_svg = audit_and_save
    base.main()

    pooled_path = OUT / "source_pooled_feature_summary.csv"
    pooled = pd.read_csv(pooled_path)
    if len(pooled) != 43 or pooled["feature"].nunique() != 43:
        raise RuntimeError("EXPECTED_43_UNIQUE_FEATURES")
    pooled[["feature", "label", "group"]].rename(columns={"label": "display_variable_name"}).to_csv(
        OUT / "source_variable_name_key.csv", index=False
    )

    ring = pd.read_csv(OUT / "source_registered_scheme_feature_medians_and_deviations.csv")
    if len(ring) != 43 * 8:
        raise RuntimeError("EXPECTED_344_COEFFICIENT_CELLS")
    coefficient_limit = float(np.quantile(np.abs(ring["median_local_coefficient"]), .90))
    deviation_limit = float(np.quantile(np.abs(ring["deviation_from_pooled_field_median"]), .90))
    coefficient_clipped = int((np.abs(ring["median_local_coefficient"]) > coefficient_limit).sum())
    deviation_clipped = int((np.abs(ring["deviation_from_pooled_field_median"]) > deviation_limit).sum())

    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["layout_version"] = "paired_wheel_v10_reader_facing_variable_names_non_destructive"
        payload["wheel_display_scale"] = {
            "rule": "zero-centred symmetric limits at the empirical 90th percentile of absolute values",
            "not_natural_breaks": "Jenks classes would obscure the signed continuous coefficient scale",
            "coefficient_limit": coefficient_limit,
            "deviation_limit": deviation_limit,
            "coefficient_cells_beyond_display_limit": coefficient_clipped,
            "deviation_cells_beyond_display_limit": deviation_clipped,
            "source_values_unchanged": True,
        }
        payload.pop("support_panel_colour_limit", None)
        payload["support_panel_colour_rule"] = "c–f categorical sign hue; near-zero if |pooled median| <= empirical P10; magnitude remains on axes"
        payload["support_panel_near_zero_threshold"] = near_zero_limit
        payload["variable_name_key"] = str((OUT / "source_variable_name_key.csv").relative_to(ROOT))
        payload["group_code_mapping"] = {
            "O": "Objective conditions",
            "S": "Platform expression",
            "D": "Signed divergence",
            "E": "Socioeconomic context",
        }
        payload["typography"] = {
            "base_font_pt": 8.0,
            "radial_label_font_pt": base.RADIAL_FONT_PT,
            "scheme_font_pt": 6.5,
            "group_font_pt": 6.4,
            "only_panel_letters_bold": True,
            "radial_label_collision_gate": "renderer bounding boxes",
        }
        payload["checks"].update({
            "reader_facing_variable_names_on_both_wheels": True,
            "radial_variables_and_group_codes_non_overlapping": True,
            "shared_group_key_clear_of_wheels_and_colourbars": True,
            "inner_colourbar_endpoint_labels_non_overlapping": True,
            "no_invented_dendrogram_or_p_values": True,
            "wheel_limits_are_zero_centred_p90": True,
            "support_colour_is_sign_not_magnitude": True,
            "wheel_titles_removed_to_clear_radial_variable_names": True,
            "channel_rows_labeled_once_to_avoid_support_overlap": True,
        })
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "figure_contract.md").write_text(
        "# V211 named-variable paired wheel contract\n\n"
        "- Core conclusion: registered V211 proxy coefficient levels and partition deviations vary across 43 fields and eight schemes; these are fitted-model diagnostics, not causal effects.\n"
        "- Archetype: quantitative grid, with two complementary hero wheels and four supporting panels.\n"
        "- Panels: a, coefficient levels; b, deviations from pooled field medians; c–f, magnitude, sign concordance, channel distributions and magnitude–concordance.\n"
        "- Labels: both wheels directly show reader-facing variable names, repeated within O/S/D only where those are different measured channels; the four channel cues are decoded in a shared line. Full internal field identifiers remain in `source_variable_name_key.csv`.\n"
        "- Colour: signed continuous red–white–blue wheels use symmetric empirical P90 limits around zero. Outlying cells are clipped in display only, with exact values preserved in source data; the colourbars explicitly say P90 clip. Jenks natural breaks are not used because they would disrupt a comparable signed coefficient scale.\n"
        f"- Supporting panels c–f: saturated red/blue hues encode sign, with neutral grey only for |pooled median| ≤ P10 ({near_zero_limit:.5f}); magnitude is positioned on axes.\n"
        "- Integrity: source data and all 43×8 data cells follow the audited v8/v9 renderer; no P-value ring or artificial clustering tree is added.\n"
        "- Export: 183-mm registered width, JPG 600 dpi and editable-text SVG only; v1–v9 outputs remain untouched.\n"
        "- Analysis limits: `proxy_analysis=true`, `formal=false`; platform expression is not resident self-report.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
