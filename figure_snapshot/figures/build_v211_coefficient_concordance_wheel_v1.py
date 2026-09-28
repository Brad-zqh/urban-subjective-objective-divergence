"""Build a high-impact circular coefficient-concordance atlas for V211.

The circular panel uses all 43 non-intercept coefficient fields and all eight
registered spatial schemes.  Rings are registered scheme/vintage medians;
sectors are semantically ordered fields.  The open sector is reserved for ring
labels and interpretation, avoiding an ornamental dendrogram or unregistered
clustering.
"""
from __future__ import annotations

import json
import re
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
from matplotlib.patches import Circle

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

from nature_viz_common import (  # noqa: E402
    ROOT, DIV, INK, GRID, MUTED, DIMENSION_LABELS, SES_LABELS,
    setup_style, clean_axes, save_delivery, write_validation, sha256,
)

RUN = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered"
WIDE = RUN / "outer_test_local_coefficients_wide.parquet"
SUMMARY = RUN / "feature_local_coefficient_summary.csv"
OUT = ROOT / "figures/v211_coefficient_concordance_wheel_v2"
STEM = "Fig_v211_coefficient_concordance_wheel_v2"

BLUE = "#3B4CC0"
RED = "#B40426"
GROUP_ORDER = [
    "Objective conditions",
    "Platform expression",
    "Signed divergence",
    "Socioeconomic context",
]


def feature_group(feature: str) -> str:
    if feature.startswith("o_"):
        return "Objective conditions"
    if feature.startswith("s_"):
        return "Platform expression"
    if feature.startswith("delta_"):
        return "Signed divergence"
    return "Socioeconomic context"


def feature_label(feature: str) -> str:
    if feature.startswith(("o_d", "s_d", "delta_d")):
        match = re.search(r"_d(\d+)_", feature)
        number = int(match.group(1)) if match else 0
        label = DIMENSION_LABELS.get(number, f"Dimension {number}")
        return {
            "Green / blue": "Green",
            "Cleanliness": "Clean",
            "Air pollution": "Air",
            "Social": "Vitality",
        }.get(label, label)
    label = SES_LABELS.get(feature, feature.replace("ses__", "").replace("_", " "))
    return {
        "Age ≥65": "65+", "Age <18": "<18", "Hispanic / Latino": "Hisp./Lat.",
        "Black population": "Black", "Higher education": "Higher ed.",
        "Transit commute": "Transit", "Walk commute": "Walk",
        "Male population": "Male", "Population size": "Pop.",
        "Household income": "Income", "Log income": "Income", "Rent burden": "Rent",
        "Unemployment": "Unemploy.",
    }.get(label, label)


def scheme_label(row: pd.Series) -> str:
    prefix = {
        "axis_recursive": "AR",
        "polar_north_clockwise_equal_count": "PC",
        "rotated45_recursive": "R45",
        "y_equal_count_stripes": "EQ",
    }.get(str(row.partition_scheme), str(row.partition_scheme))
    vintage = "10" if int(row.heldout_boundary_vintage) == 2010 else "20"
    return f"{prefix}·{vintage}"


def label_rotation(angle_deg: float) -> tuple[float, str]:
    rotation = angle_deg - 90
    normalized = rotation % 360
    if 90 < normalized < 270:
        return rotation + 180, "right"
    return rotation, "left"


def main() -> None:
    setup_style(6.8)
    wide = pd.read_parquet(WIDE)
    summary = pd.read_csv(SUMMARY)
    features = [c for c in wide.columns if c.startswith(("o_", "s_", "delta_", "ses__"))]
    if len(features) != 43 or len(wide) != 33760:
        raise RuntimeError("EXPECTED_43_FIELDS_AND_33760_EVALUATIONS")

    objective = [f"o_d{i}_composite_z" for i in range(1, 11)]
    subjective = [f"s_d{i}_augmented_mm_z" for i in range(1, 11)]
    divergence = [f"delta_d{i}_augmented_signed_s_minus_o" for i in range(1, 11)]
    socioeconomic = [f for f in features if f.startswith("ses__")]
    feature_order = objective + subjective + divergence + socioeconomic
    if set(feature_order) != set(features):
        raise RuntimeError("FEATURE_ORDER_MISMATCH")

    scheme_table = wide[["partition_scheme", "heldout_boundary_vintage"]].drop_duplicates().copy()
    scheme_table["scheme"] = scheme_table.apply(scheme_label, axis=1)
    preferred = ["AR·10", "AR·20", "PC·10", "PC·20", "R45·10", "R45·20", "EQ·10", "EQ·20"]
    scheme_table["order"] = scheme_table.scheme.map({x: i for i, x in enumerate(preferred)})
    scheme_table = scheme_table.sort_values("order")
    schemes = scheme_table.scheme.tolist()

    temp = wide.copy()
    temp["scheme"] = temp.apply(scheme_label, axis=1)
    ring = temp.groupby("scheme", sort=False)[feature_order].median().reindex(schemes)
    pooled = pd.DataFrame({
        "feature": feature_order,
        "label": [feature_label(x) for x in feature_order],
        "group": [feature_group(x) for x in feature_order],
        "median": [float(wide[x].median()) for x in feature_order],
        "q25": [float(wide[x].quantile(.25)) for x in feature_order],
        "q75": [float(wide[x].quantile(.75)) for x in feature_order],
        "median_absolute": [float(np.median(np.abs(wide[x].to_numpy(float)))) for x in feature_order],
    })
    pooled_sign = np.sign(pooled.set_index("feature").loc[feature_order, "median"].to_numpy(float))
    scheme_sign = np.sign(ring[feature_order].to_numpy(float))
    pooled["scheme_sign_agreement"] = (scheme_sign == pooled_sign[None, :]).mean(axis=0)

    matrix = ring[feature_order].to_numpy(float)
    colour_limit = float(np.quantile(np.abs(matrix), .98))
    norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0, vmax=colour_limit)

    fig = plt.figure(figsize=(183 / 25.4, 126 / 25.4), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=.028, right=.975, bottom=.100, top=.965,
                            width_ratios=[1.88, .80], height_ratios=[1.12, .88],
                            wspace=.25, hspace=.42)
    ax = fig.add_subplot(grid[:, 0], projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_axis_off()

    n_features = len(feature_order)
    n_rings = len(schemes)
    gap_deg = 66.0
    start_deg = gap_deg / 2
    span_deg = 360.0 - gap_deg
    width_deg = span_deg / n_features
    centers_deg = start_deg + (np.arange(n_features) + .5) * width_deg
    centers = np.deg2rad(centers_deg)
    inner = 0.275
    ring_h = 0.070
    cell_w = np.deg2rad(width_deg * .93)

    for ring_index, scheme in enumerate(schemes):
        bottom = inner + ring_index * ring_h
        values = ring.loc[scheme, feature_order].to_numpy(float)
        for theta, value in zip(centers, values):
            clipped = float(np.clip(value, -colour_limit, colour_limit))
            ax.bar(theta, ring_h * .91, width=cell_w, bottom=bottom,
                   color=DIV(norm(clipped)), edgecolor="white", linewidth=.34, align="center")
    # The ring labels are stacked vertically inside the open sector.  A radial
    # theta=0 labelling scheme would place all eight strings on one baseline.
    ax.text(.800, .665, "rings", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.3, fontweight="bold", color=INK)
    for ring_index, scheme in enumerate(schemes):
        y = .628 - ring_index * .034
        ax.plot([.784, .798], [y, y], transform=ax.transAxes, color=MUTED, lw=.70, clip_on=False)
        ax.text(.806, y, scheme, transform=ax.transAxes, ha="left", va="center", fontsize=4.85, color=INK)
    ax.text(.800, .335, "inner → outer", transform=ax.transAxes, ha="left", va="top",
            fontsize=4.55, color=MUTED)

    outer = inner + n_rings * ring_h
    agreement = pooled.scheme_sign_agreement.to_numpy(float)
    for theta, agree, sign in zip(centers, agreement, pooled_sign):
        colour = RED if sign >= 0 else BLUE
        ax.bar(theta, .054, width=cell_w, bottom=outer + .018,
               color=colour, alpha=.18 + .82 * np.clip((agree - .5) / .5, 0, 1),
               edgecolor="white", linewidth=.35)

    groups = pooled.group.tolist()
    group_signed_medians = {
        group: float(pooled.loc[pooled.group.eq(group), "median"].median())
        for group in GROUP_ORDER
    }
    group_limit = max(abs(value) for value in group_signed_medians.values())
    group_norm = TwoSlopeNorm(vmin=-group_limit, vcenter=0, vmax=group_limit)
    group_colours = {group: DIV(group_norm(value)) for group, value in group_signed_medians.items()}
    boundaries: list[tuple[str, int, int]] = []
    start = 0
    for group in dict.fromkeys(groups):
        inds = [i for i, value in enumerate(groups) if value == group]
        boundaries.append((group, min(inds), max(inds)))
    group_codes = {
        "Objective conditions": "O",
        "Platform expression": "S",
        "Signed divergence": "S−O",
        "Socioeconomic context": "SES",
    }
    for group, lo, hi in boundaries:
        group_center = np.deg2rad((centers_deg[lo] + centers_deg[hi]) / 2)
        group_width = np.deg2rad((hi - lo + 1) * width_deg * .97)
        ax.bar(group_center, .078, width=group_width, bottom=outer + .086,
               color=group_colours[group], edgecolor="white", linewidth=.55)
        angle = np.rad2deg(group_center)
        rotation, ha = label_rotation(angle)
        ax.text(group_center, outer + .124, group_codes[group], rotation=rotation,
                rotation_mode="anchor", ha=ha, va="center", fontsize=6.0,
                fontweight="bold", color=INK)

    feature_artists = []
    for index, (theta, angle, label) in enumerate(zip(centers, centers_deg, pooled.label)):
        radius = outer + .155 + (.125 if index % 2 else 0.0) + .012 * ((index // 2) % 2)
        label_angle = float(angle) + (-1.25 if index % 2 == 0 else 1.25)
        label_theta = np.deg2rad(label_angle)
        rotation, ha = label_rotation(label_angle)
        ax.plot([theta, label_theta], [outer + .150, radius - .014], color=GRID, lw=.30, zorder=1)
        artist = ax.text(label_theta, radius, label, rotation=rotation, rotation_mode="anchor", ha=ha, va="center",
                         fontsize=4.10, color=INK)
        feature_artists.append(artist)

    ax.add_patch(Circle((0, 0), inner * .78, transform=ax.transData._b,
                        facecolor="white", edgecolor=INK, linewidth=.65, zorder=10))
    ax.text(0, 0, "MM-GTGNNWR\n43 fields · 8 schemes",
            ha="center", va="center", fontsize=6.35, fontweight="bold", linespacing=1.35, zorder=11)
    ax.text(-.025, 1.035, "a", transform=ax.transAxes, ha="right", va="top",
            fontsize=9.0, fontweight="bold", color=INK)
    ax.set_ylim(0, outer + .39)

    # Independent horizontal colour bar below the circular evidence field.
    cax = fig.add_axes([.115, .055, .405, .014])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=DIV), cax=cax, orientation="horizontal")
    cb.set_ticks([-colour_limit, 0, colour_limit])
    cb.set_ticklabels([f"{-colour_limit:.3f}", "0", f"{colour_limit:.3f}"])
    cb.ax.tick_params(labelsize=5.3, length=1.0, pad=.6)
    cb.outline.set_linewidth(.35)
    cb.set_label("Median local coefficient (registered outer-fold rows)", fontsize=5.7, labelpad=1.0)

    # b: readable ranking of the strongest fields, avoiding a second heatmap.
    b = fig.add_subplot(grid[0, 1])
    ranked = pooled.nlargest(12, "median_absolute").sort_values("median_absolute")
    yy = np.arange(len(ranked))
    ranked_colours = [RED if value >= 0 else BLUE for value in ranked["median"]]
    b.hlines(yy, 0, ranked.median_absolute, color=ranked_colours, lw=2.0)
    b.scatter(ranked.median_absolute, yy, s=31, color=ranked_colours, edgecolor="white", linewidth=.5)
    b.set_yticks(yy, [f"{label} · {group_codes[group]}" for label, group in zip(ranked.label, ranked.group)])
    b.set_xlabel("|Median local coefficient|")
    clean_axes(b, "x")
    b.set_title("Strongest fields", loc="left", pad=2.5)
    b.text(-.18, 1.04, "b", transform=b.transAxes, ha="right", va="top", fontsize=9.0, fontweight="bold")

    # c: interval summary of sign concordance by scientific channel.
    c = fig.add_subplot(grid[1, 1])
    group_order = GROUP_ORDER
    for y, group in enumerate(group_order):
        values = pooled.loc[pooled.group.eq(group), "scheme_sign_agreement"].to_numpy(float)
        q25, med, q75 = np.quantile(values, [.25, .5, .75])
        c.hlines(y, q25, q75, color=group_colours[group], lw=5.0)
        c.scatter(med, y, s=34, color=group_colours[group], edgecolor="white", linewidth=.55, zorder=3)
    c.axvline(.5, color=GRID, lw=.7, ls="--")
    c.set_xlim(.45, 1.02)
    c.set_yticks(range(len(group_order)), ["Objective", "Platform", "Divergence", "SES"])
    c.invert_yaxis()
    c.set_xlabel("Share of schemes matching pooled sign")
    clean_axes(c, "x")
    c.set_title("Sign concordance", loc="left", pad=2.5)
    c.text(-.18, 1.04, "c", transform=c.transAxes, ha="right", va="top", fontsize=9.0, fontweight="bold")
    c.legend(
        [Line2D([0], [0], color=BLUE, lw=2), Line2D([0], [0], color=RED, lw=2)],
        ["negative pooled median", "positive pooled median"],
        loc="lower center", bbox_to_anchor=(.5, -.53), frameon=False, fontsize=5.4, ncol=1,
    )

    # Explicit overlap gate for the 43 outer labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer).expanded(1.01, 1.04) for artist in feature_artists]
    overlap_pairs = [
        (i, j, feature_artists[i].get_text(), feature_artists[j].get_text())
        for i in range(len(boxes)) for j in range(i + 1, len(boxes))
        if boxes[i].overlaps(boxes[j])
    ]
    if overlap_pairs:
        raise RuntimeError(f"OUTER_FEATURE_LABEL_OVERLAPS={overlap_pairs}")

    delivery = save_delivery(fig, OUT, STEM, dpi=600)
    plt.close(fig)
    ring_long = ring.reset_index().melt(id_vars="scheme", var_name="feature", value_name="median_local_coefficient")
    ring_long = ring_long.merge(pooled[["feature", "label", "group", "scheme_sign_agreement"]], on="feature", how="left")
    ring_long.to_csv(OUT / "source_registered_scheme_feature_medians.csv", index=False)
    pooled.to_csv(OUT / "source_pooled_feature_summary.csv", index=False)
    payload = {
        "figure": STEM,
        "panels": 3,
        "analysis_version": "V211",
        "proxy_analysis": True,
        "formal": False,
        "core_conclusion": "Coefficient direction is structured across channels and largely concordant across the eight registered spatial schemes.",
        "evidence_chain": "43-field × 8-scheme circular coefficient atlas → strongest-field ranking → channel-level sign concordance.",
        "ring_order": schemes,
        "sector_groups": group_order,
        "group_signed_medians": group_signed_medians,
        "colour_limit": colour_limit,
        "colour_limit_rule": "98th percentile of absolute registered scheme medians; values are clipped for colour only",
        "source_hashes": {str(WIDE.relative_to(ROOT)): sha256(WIDE), str(SUMMARY.relative_to(ROOT)): sha256(SUMMARY)},
        "delivery": delivery,
        "checks": {
            "registered_rows": len(wide) == 33760,
            "coefficient_fields": len(features) == 43,
            "registered_schemes": len(schemes) == 8,
            "source_tables_written": (OUT / "source_registered_scheme_feature_medians.csv").exists() and (OUT / "source_pooled_feature_summary.csv").exists(),
            "five_delivery_formats": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
            "svg_text_editable": delivery["svg_text_editable"],
            "proxy_only": True,
            "not_formal": True,
        },
    }
    write_validation(OUT, payload)
    (OUT / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\n"
        "- Core conclusion: coefficient direction is structured across four scientific channels and is mostly concordant across the eight registered spatial schemes.\n"
        "- Ring encoding: scheme/vintage median local coefficient, ordered AR10, AR20, PC10, PC20, R45-10, R45-20, EQ10 and EQ20.\n"
        "- Sector encoding: 43 semantically ordered non-intercept fields; the red-blue sign track encodes pooled direction and sign agreement.\n"
        "- Outer group arcs use the same red-white-blue scale and encode each channel's median signed coefficient.\n"
        "- The open sector is used for labels, not an unregistered dendrogram.\n"
        "- Scientific status: V211 compact proxy (`proxy_analysis=true`, `formal=false`).\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT / f"{STEM}.png"), "delivery": delivery}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
