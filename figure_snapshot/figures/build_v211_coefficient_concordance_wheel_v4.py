"""Two-wheel V211 coefficient atlas with a disciplined red-white-blue grammar.

The v1-v3 scripts and exports remain untouched.  Panel a shows registered
scheme medians; panel b shows each scheme's deviation from the pooled field
median.  Four compact diagnostics retain quantitative support below the two
dominant, reference-inspired circular atlases.
"""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_coefficient_concordance_wheel_v1 as legacy  # noqa: E402
from nature_viz_common import ROOT, DIV, INK, GRID, MUTED, setup_style, clean_axes, sha256  # noqa: E402

WIDE = legacy.WIDE
SUMMARY = legacy.SUMMARY
OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v4"
STEM = "Fig_v211_coefficient_concordance_wheel_v4"
GROUP_ORDER = legacy.GROUP_ORDER


def panel(ax: plt.Axes, index: int, title: str = "", x: float = -0.08, y: float = 1.03) -> None:
    if title:
        ax.set_title(title, loc="left", pad=2.6, fontweight="normal")
    ax.text(x, y, string.ascii_lowercase[index], transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, fontweight="bold", color=INK, clip_on=False)


def radial_rotation(angle_deg: float) -> tuple[float, str]:
    rotation = -float(angle_deg)
    if 90 < rotation % 360 < 270:
        return rotation + 180, "right"
    return rotation, "left"


def save_jpg_svg(fig: plt.Figure) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    jpg = OUT / f"{STEM}.jpg"
    svg = OUT / f"{STEM}.svg"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(jpg, dpi=600, bbox_inches="tight", pil_kwargs={"quality": 95})
    with Image.open(jpg) as image:
        dpi = image.info.get("dpi", (0, 0))
        pixels = image.size
    raw = svg.read_text(encoding="utf-8", errors="ignore")
    return {"jpg": str(jpg), "svg": str(svg), "delivery_formats": ["jpg", "svg"],
            "jpg_dpi_metadata": [float(dpi[0]), float(dpi[1])],
            "jpg_pixel_size": [int(pixels[0]), int(pixels[1])], "svg_text_editable": "<text" in raw}


def draw_wheel(
    fig: plt.Figure,
    ax: plt.Axes,
    values: np.ndarray,
    feature_labels: list[str],
    groups: list[str],
    schemes: list[str],
    norm: TwoSlopeNorm,
    outer_values: np.ndarray,
    title: str,
    index: int,
) -> tuple[list[plt.Text], dict]:
    """Draw one open circular heatmap with a continuous data-valued outer ribbon."""
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_axis_off()
    gap_deg = 76.0
    start_deg = gap_deg / 2
    span_deg = 360.0 - gap_deg
    n_features = values.shape[1]
    width_deg = span_deg / n_features
    centres_deg = start_deg + (np.arange(n_features) + .5) * width_deg
    centres = np.deg2rad(centres_deg)
    inner = .315
    ring_h = .058
    cell_w = np.deg2rad(width_deg * .965)
    for ring_index, row in enumerate(values):
        bottom = inner + ring_index * ring_h
        for theta, value in zip(centres, row):
            clipped = float(np.clip(value, norm.vmin, norm.vmax))
            ax.bar(theta, ring_h * .94, width=cell_w, bottom=bottom, color=DIV(norm(clipped)),
                   edgecolor="white", linewidth=.22, align="center")

    outer = inner + len(schemes) * ring_h
    for theta, value in zip(centres, outer_values):
        clipped = float(np.clip(value, norm.vmin, norm.vmax))
        ax.bar(theta, .046, width=np.deg2rad(width_deg * 1.015), bottom=outer + .014,
               color=DIV(norm(clipped)), edgecolor="none", linewidth=0, align="center")

    # Four group boundaries structure the ring without introducing extra hues.
    boundaries = []
    for group in dict.fromkeys(groups):
        ids = [i for i, value in enumerate(groups) if value == group]
        boundaries.append((group, min(ids), max(ids)))
    codes = {"Objective conditions": "O", "Platform expression": "S",
             "Signed divergence": "S−O", "Socioeconomic context": "SES"}
    for group, lo, hi in boundaries:
        if lo > 0:
            theta = np.deg2rad(start_deg + lo * width_deg)
            ax.plot([theta, theta], [inner - .010, outer + .064], color="#7F8893", lw=.42, alpha=.70)
        centre_deg = float((centres_deg[lo] + centres_deg[hi]) / 2)
        rotation, ha = radial_rotation(centre_deg)
        ax.text(np.deg2rad(centre_deg), outer + .095, codes[group], rotation=rotation,
                rotation_mode="anchor", ha=ha, va="center", fontsize=4.8, fontweight="normal")

    # Scheme labels occupy the deliberate opening, as in the reference atlas.
    ax.text(.815, .675, "registered schemes", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=4.7, fontweight="normal")
    for ring_index, scheme in enumerate(schemes):
        y = .642 - ring_index * .034
        ax.plot([.801, .813], [y, y], transform=ax.transAxes, color=MUTED, lw=.62, clip_on=False)
        ax.text(.821, y, scheme, transform=ax.transAxes, ha="left", va="center",
                fontsize=4.45, fontweight="normal")
    ax.text(.815, .330, "inner → outer", transform=ax.transAxes, ha="left", va="top",
            fontsize=4.35, color=MUTED, fontweight="normal")

    artists: list[plt.Text] = []
    label_radius = outer + .190
    for feature_index, (theta, angle, label) in enumerate(zip(centres, centres_deg, feature_labels)):
        if feature_index == 0:
            adjustment = -2.8
        elif feature_index == 3:
            adjustment = 3.5
        elif feature_index == n_features - 2:
            adjustment = -1.2
        elif feature_index == n_features - 1:
            adjustment = 2.0
        else:
            adjustment = 0.0
        display_angle = float(angle) + adjustment
        display_theta = np.deg2rad(display_angle)
        display_radius = label_radius + (.026 if feature_index in {1, 3} else 0.0)
        rotation, ha = radial_rotation(display_angle)
        ax.plot([theta, display_theta], [outer + .066, display_radius - .015], color=GRID, lw=.28)
        artists.append(ax.text(display_theta, display_radius, label, rotation=rotation,
                               rotation_mode="anchor", ha=ha, va="center", fontsize=3.00,
                               fontweight="normal", color=INK))

    # Keep the centre deliberately empty; the information hierarchy lives in the rings.
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(theta, np.full_like(theta, inner - .012), color="#626A74", lw=.48)
    ax.set_ylim(0, label_radius + .055)
    panel(ax, index, title, x=-.015, y=1.012)
    return artists, {"inner": inner, "outer": outer, "label_radius": label_radius,
                     "start_deg": start_deg, "span_deg": span_deg}


def main() -> None:
    setup_style(6.35)
    wide = pd.read_parquet(WIDE)
    _ = pd.read_csv(SUMMARY)
    features = [column for column in wide.columns if column.startswith(("o_", "s_", "delta_", "ses__"))]
    if len(features) != 43 or len(wide) != 33760:
        raise RuntimeError("EXPECTED_43_FIELDS_AND_33760_EVALUATIONS")
    objective = [f"o_d{i}_composite_z" for i in range(1, 11)]
    subjective = [f"s_d{i}_augmented_mm_z" for i in range(1, 11)]
    divergence = [f"delta_d{i}_augmented_signed_s_minus_o" for i in range(1, 11)]
    socioeconomic = [feature for feature in features if feature.startswith("ses__")]
    feature_order = objective + subjective + divergence + socioeconomic
    if set(feature_order) != set(features):
        raise RuntimeError("FEATURE_ORDER_MISMATCH")

    scheme_table = wide[["partition_scheme", "heldout_boundary_vintage"]].drop_duplicates().copy()
    scheme_table["scheme"] = scheme_table.apply(legacy.scheme_label, axis=1)
    preferred = ["AR·10", "AR·20", "PC·10", "PC·20", "R45·10", "R45·20", "EQ·10", "EQ·20"]
    scheme_table["order"] = scheme_table.scheme.map({value: i for i, value in enumerate(preferred)})
    schemes = scheme_table.sort_values("order").scheme.tolist()
    temp = wide.copy(); temp["scheme"] = temp.apply(legacy.scheme_label, axis=1)
    ring = temp.groupby("scheme", sort=False)[feature_order].median().reindex(schemes)
    pooled = pd.DataFrame({
        "feature": feature_order,
        "label": [legacy.feature_label(feature) for feature in feature_order],
        "group": [legacy.feature_group(feature) for feature in feature_order],
        "median": [float(wide[feature].median()) for feature in feature_order],
        "q25": [float(wide[feature].quantile(.25)) for feature in feature_order],
        "q75": [float(wide[feature].quantile(.75)) for feature in feature_order],
        "median_absolute": [float(np.median(np.abs(wide[feature].to_numpy(float)))) for feature in feature_order],
    })
    pooled["label"] = pooled.label.replace({"Unemploy.": "Unemp.", "Higher education": "Higher ed.",
                                              "Black population": "Black pop.", "Air pollution": "Air"})
    pooled_sign = np.sign(pooled["median"].to_numpy(float))
    scheme_sign = np.sign(ring.to_numpy(float))
    pooled["scheme_sign_agreement"] = (scheme_sign == pooled_sign[None, :]).mean(axis=0)
    matrix = ring.to_numpy(float)
    deviation = matrix - pooled["median"].to_numpy(float)[None, :]
    coefficient_limit = max(float(np.quantile(np.abs(matrix), .98)), 1e-8)
    deviation_limit = max(float(np.quantile(np.abs(deviation), .98)), 1e-8)
    coefficient_norm = TwoSlopeNorm(vmin=-coefficient_limit, vcenter=0, vmax=coefficient_limit)
    deviation_norm = TwoSlopeNorm(vmin=-deviation_limit, vcenter=0, vmax=deviation_limit)
    max_dev_index = np.argmax(np.abs(deviation), axis=0)
    max_signed_deviation = deviation[max_dev_index, np.arange(deviation.shape[1])]

    fig = plt.figure(figsize=(183 / 25.4, 174 / 25.4), facecolor="white")
    grid = fig.add_gridspec(3, 4, left=.035, right=.985, bottom=.075, top=.975,
                            height_ratios=[1.0, .055, .34], width_ratios=[1, 1, 1, 1],
                            wspace=.35, hspace=.20)
    ax_a = fig.add_subplot(grid[0, 0:2], projection="polar")
    labels_a, geometry_a = draw_wheel(fig, ax_a, matrix, pooled.label.tolist(), pooled.group.tolist(),
                                       schemes, coefficient_norm, pooled["median"].to_numpy(float),
                                       "Registered coefficient levels", 0)
    ax_b = fig.add_subplot(grid[0, 2:4], projection="polar")
    labels_b, geometry_b = draw_wheel(fig, ax_b, deviation, pooled.label.tolist(), pooled.group.tolist(),
                                       schemes, deviation_norm, max_signed_deviation,
                                       "Partition deviation from pooled median", 1)

    cax_a = fig.add_subplot(grid[1, 0:2]); cax_b = fig.add_subplot(grid[1, 2:4])
    cb_a = fig.colorbar(ScalarMappable(norm=coefficient_norm, cmap=DIV), cax=cax_a, orientation="horizontal")
    cb_b = fig.colorbar(ScalarMappable(norm=deviation_norm, cmap=DIV), cax=cax_b, orientation="horizontal")
    for cb, limit, label in [(cb_a, coefficient_limit, "Median local coefficient"),
                             (cb_b, deviation_limit, "Scheme median − pooled field median")]:
        cb.set_ticks([-limit, 0, limit]); cb.set_ticklabels([f"{-limit:.3f}", "0", f"{limit:.3f}"])
        cb.ax.tick_params(labelsize=5.0, length=1.0, pad=.5)
        cb.outline.set_linewidth(.32); cb.set_label(label, fontsize=5.3, labelpad=1.0, fontweight="normal")

    # Support panels use a robust 90th-percentile display scale so the many
    # moderate fields remain chromatically legible; source values are intact.
    support_limit = max(float(np.quantile(np.abs(pooled["median"]), .90)), 1e-8)
    support_norm = TwoSlopeNorm(vmin=-support_limit, vcenter=0, vmax=support_limit)
    codes = {"Objective conditions": "O", "Platform expression": "S",
             "Signed divergence": "S−O", "Socioeconomic context": "SES"}

    ax = fig.add_subplot(grid[2, 0])
    ranked = pooled.nlargest(10, "median_absolute").sort_values("median_absolute")
    yy = np.arange(len(ranked))
    colours = [DIV(support_norm(float(v))) for v in ranked["median"]]
    ax.hlines(yy, 0, ranked.median_absolute, color=colours, lw=1.7)
    ax.scatter(ranked.median_absolute, yy, s=24, color=colours, edgecolor="white", linewidth=.45)
    ax.set_yticks(yy, [f"{label} · {codes[group]}" for label, group in zip(ranked.label, ranked.group)])
    ax.set_xlabel("|Median coefficient|"); clean_axes(ax, "x")
    panel(ax, 2, "Strongest fields", x=-.10, y=1.035)

    group_medians = {group: float(pooled.loc[pooled.group.eq(group), "median"].median()) for group in GROUP_ORDER}
    ax = fig.add_subplot(grid[2, 1])
    for y, group in enumerate(GROUP_ORDER):
        values = pooled.loc[pooled.group.eq(group), "scheme_sign_agreement"].to_numpy(float)
        q25, med, q75 = np.quantile(values, [.25, .5, .75])
        colour = DIV(support_norm(group_medians[group]))
        ax.hlines(y, q25, q75, color=colour, lw=4.0)
        ax.scatter(med, y, s=27, color=colour, edgecolor="white", linewidth=.5)
    ax.axvline(.5, color=GRID, lw=.65, ls="--")
    ax.set_xlim(.45, 1.02); ax.set_yticks(range(4), ["Objective", "Platform", "Divergence", "SES"])
    ax.invert_yaxis(); ax.set_xlabel("Schemes matching pooled sign"); clean_axes(ax, "x")
    panel(ax, 3, "Sign concordance", x=-.11, y=1.035)

    ax = fig.add_subplot(grid[2, 2])
    for y, group in enumerate(GROUP_ORDER):
        subset = pooled.loc[pooled.group.eq(group), "median"].to_numpy(float)
        point_colours = [DIV(support_norm(float(v))) for v in subset]
        ax.scatter(subset, np.full_like(subset, y, dtype=float), s=12, color=point_colours,
                   alpha=.75, edgecolor="white", linewidth=.2)
        q25, med, q75 = np.quantile(subset, [.25, .5, .75])
        colour = DIV(support_norm(float(med)))
        ax.hlines(y, q25, q75, color=colour, lw=3.0)
        ax.scatter(med, y, s=27, color=colour, edgecolor="white", linewidth=.5)
    ax.axvline(0, color=GRID, lw=.65)
    ax.set_yticks(range(4), ["Objective", "Platform", "Divergence", "SES"])
    ax.invert_yaxis(); ax.set_xlabel("Pooled signed coefficient"); clean_axes(ax, "x")
    panel(ax, 4, "Channel distributions", x=-.11, y=1.035)

    ax = fig.add_subplot(grid[2, 3])
    colours = [DIV(support_norm(float(v))) for v in pooled["median"]]
    ax.scatter(pooled.median_absolute, pooled.scheme_sign_agreement, s=20, color=colours,
               edgecolor="white", linewidth=.4, alpha=.92)
    ax.set_xlabel("|Median coefficient|"); ax.set_ylabel("Sign agreement"); ax.set_ylim(.45, 1.03)
    clean_axes(ax, "both"); panel(ax, 5, "Magnitude–concordance", x=-.11, y=1.035)
    ax.legend([Line2D([0], [0], marker="o", color="none", markerfacecolor=DIV(.12), markersize=4.6),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=DIV(.50), markersize=4.6),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=DIV(.88), markersize=4.6)],
              ["negative", "near zero", "positive"], loc="lower right", fontsize=4.3,
              handletextpad=.25, frameon=False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlap_report = {}
    for name, artists in [("coefficient_wheel", labels_a), ("deviation_wheel", labels_b)]:
        boxes = [artist.get_window_extent(renderer).expanded(1.005, 1.02) for artist in artists]
        overlaps = [(i, j, artists[i].get_text(), artists[j].get_text())
                    for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j])]
        overlap_report[name] = overlaps
    if any(overlap_report.values()):
        raise RuntimeError(f"RADIAL_LABEL_OVERLAPS={overlap_report}")

    delivery = save_jpg_svg(fig)
    plt.close(fig)
    ring_long = ring.reset_index().melt(id_vars="scheme", var_name="feature", value_name="median_local_coefficient")
    ring_long = ring_long.merge(pooled[["feature", "label", "group", "scheme_sign_agreement"]], on="feature", how="left")
    ring_long["pooled_field_median"] = ring_long.feature.map(pooled.set_index("feature")["median"])
    ring_long["deviation_from_pooled_field_median"] = ring_long.median_local_coefficient - ring_long.pooled_field_median
    ring_long.to_csv(OUT / "source_registered_scheme_feature_medians_and_deviations.csv", index=False)
    pooled.to_csv(OUT / "source_pooled_feature_summary.csv", index=False)
    payload = {
        "figure": STEM, "panels": 6, "analysis_version": "V211",
        "layout_version": "paired_wheel_v4_non_destructive", "proxy_analysis": True, "formal": False,
        "core_conclusion": "Coefficient level and partition-specific deviation are structured across fields and registered schemes.",
        "evidence_chain": "43×8 coefficient wheel → 43×8 deviation wheel → strength → sign concordance → channel distribution → magnitude-concordance.",
        "outer_ribbon": "continuous field-level values; no categorical group colours",
        "ring_order": schemes, "sector_groups": GROUP_ORDER,
        "coefficient_colour_limit": coefficient_limit, "deviation_colour_limit": deviation_limit,
        "support_panel_colour_limit": support_limit,
        "support_panel_colour_rule": "pooled signed-median absolute 90th percentile; display only",
        "source_hashes": {str(WIDE.relative_to(ROOT)): sha256(WIDE), str(SUMMARY.relative_to(ROOT)): sha256(SUMMARY)},
        "wheel_geometry": {"coefficient": geometry_a, "deviation": geometry_b},
        "delivery": delivery,
        "checks": {
            "registered_rows": len(wide) == 33760, "coefficient_fields": len(features) == 43,
            "registered_schemes": len(schemes) == 8, "six_panels": True,
            "radial_labels_non_overlapping": not any(overlap_report.values()),
            "outer_ribbons_use_continuous_red_blue_values": True,
            "centres_uncluttered": True, "only_panel_letters_bold_by_construction": True,
            "jpg_svg_only": delivery["delivery_formats"] == ["jpg", "svg"],
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
            "preserves_existing_versions": True, "proxy_only": True, "not_formal": True,
        },
    }
    (OUT / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\n"
        "- Panel a shows registered scheme-level coefficient medians.\n"
        "- Panel b shows scheme median minus the pooled field median.\n"
        "- The outer ribbons encode continuous field-level values in the same red-white-blue grammar.\n"
        "- The wheel centres are intentionally empty; only subplot letters are bold.\n"
        "- This is a new v4 export; earlier wheel versions remain untouched.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT / f"{STEM}.jpg"), "delivery": delivery}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
