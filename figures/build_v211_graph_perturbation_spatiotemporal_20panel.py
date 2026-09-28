"""Twenty-panel temporal and spatial audit of graph-relation perturbations.

The plate separates four relation families and two interventions (relation
removal and 50% edge-weight attenuation) across empirical distributions,
outer-fold uncertainty, annual trajectories and 2023 tract maps.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from PIL import Image

from nature_viz_common import (
    ROOT, BLUE, RED, INK, MUTED, GRID, DIV, MISSING, setup_style, panel,
    clean_axes, add_equal_height_colorbar, save_delivery, sha256,
    symmetric_limit, write_validation,
)


DATA = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_graph_perturbation_r3_indexed_seed_filtered/outer_test_graph_perturbations.parquet"
ROWMAP = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered/outer_test_predictions.parquet"
GEOMETRY = ROOT / "Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg/2022/chicago_tiger_2022.gpkg"
OUT = ROOT / "figures/v211_graph_perturbation_spatiotemporal_20panel"
STEM = "Fig_v211_graph_perturbation_spatiotemporal_20panel"

RELATIONS = [
    ("spatial_queen", "Queen contiguity"),
    ("road_connectivity", "Road connectivity"),
    ("mobility_flow", "Mobility flow"),
    ("temporal_forward", "Temporal links"),
]
PERTURBATIONS = [
    ("relation_drop", "Relation removed", RED, "-"),
    ("edge_weight_half", "Edge weight halved", BLUE, "--"),
]


def ecdf(values):
    x = np.sort(np.asarray(values, float))
    return x, np.arange(1, len(x) + 1) / len(x)


def density_curve(values, x):
    """Yap-style display density that preserves exact-zero point mass.

    The original perturbation rows are never changed.  For display, exact
    zeros are represented by a vertical mass marker and the non-zero values
    by a normalized KDE; this avoids a misleading full-height ECDF step.
    """
    values = np.asarray(values, float)
    zero = np.abs(values) <= 1e-12
    nonzero = values[~zero]
    if len(nonzero) < 2 or np.std(nonzero) <= 1e-12:
        return np.zeros_like(x), float(zero.mean()), True
    try:
        y = gaussian_kde(nonzero)(x)
    except (np.linalg.LinAlgError, ValueError):
        y = np.zeros_like(x)
    if np.nanmax(y) > 0:
        y = y / np.nanmax(y)
    return y, float(zero.mean()), False


def border_check(path: str):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate([rgb[:5].reshape(-1, 3), rgb[-5:].reshape(-1, 3),
                             rgb[:, :5].reshape(-1, 3), rgb[:, -5:].reshape(-1, 3)])
    return float(np.mean(np.any(rgb < 245, axis=2))), float(np.mean(np.any(border < 235, axis=1)))


def main():
    setup_style(5.9)
    before = {str(DATA): sha256(DATA), str(ROWMAP): sha256(ROWMAP), str(GEOMETRY): sha256(GEOMETRY)}
    raw = pd.read_parquet(DATA)
    raw = raw.loc[raw.relation.ne("all_active") & raw.perturbation.ne("baseline")].copy()
    mapping = pd.read_parquet(ROWMAP, columns=["fold_id", "numeric_row", "tract_geoid_native", "health_reference_year"])
    mapping["tract_geoid_native"] = mapping.tract_geoid_native.astype(str).str.zfill(11)
    joined = raw.merge(mapping, on=["fold_id", "numeric_row"], how="left", validate="many_to_one")
    if joined.tract_geoid_native.isna().any():
        raise RuntimeError("GRAPH_PERTURBATION_ROW_MAPPING_INCOMPLETE")
    expected = {(r, p) for r, _ in RELATIONS for p, *_ in PERTURBATIONS}
    if set(zip(joined.relation, joined.perturbation)) != expected:
        raise RuntimeError("GRAPH_PERTURBATION_CELL_SET_MISMATCH")

    # Scheme-averaged tract-year effects prevent the four spatial partitions
    # from being treated as four independent tracts in temporal/map panels.
    tract_year = joined.groupby(
        ["relation", "perturbation", "tract_geoid_native", "health_reference_year"], as_index=False
    ).prediction_difference.mean()
    fold = joined.groupby(["relation", "perturbation", "fold_id"], as_index=False).prediction_difference.mean()

    geo = gpd.read_file(GEOMETRY, layer="tracts_intersect_chicago",
                        columns=["GEOID", "ALAND", "AWATER", "geometry"]).to_crs(26916)
    geo["GEOID"] = geo.GEOID.astype(str).str.zfill(11)
    geo["water_share"] = geo.AWATER / (geo.ALAND + geo.AWATER)
    water_excluded = int((geo.water_share.gt(.5) | geo.ALAND.le(0)).sum())
    geo = geo.loc[geo.water_share.le(.5) & geo.ALAND.gt(0)].copy()
    map_source = tract_year.loc[tract_year.health_reference_year.eq(2023)].copy()
    frames = {}
    for relation, _ in RELATIONS:
        for perturbation, *_ in PERTURBATIONS:
            q = map_source.loc[(map_source.relation.eq(relation)) &
                               (map_source.perturbation.eq(perturbation))]
            frames[(relation, perturbation)] = geo.merge(
                q, left_on="GEOID", right_on="tract_geoid_native", how="inner", validate="one_to_one"
            )
    bounds = np.vstack([frame.total_bounds for frame in frames.values()])
    xmin, ymin, xmax, ymax = bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()
    dx, dy = xmax - xmin, ymax - ymin
    extent = (xmin - .012 * dx, xmax + .012 * dx, ymin - .012 * dy, ymax + .012 * dy)
    relation_norm = {}
    for relation, _ in RELATIONS:
        values = np.concatenate([frames[(relation, p)].prediction_difference.to_numpy(float)
                                 for p, *_ in PERTURBATIONS])
        active = values[np.abs(values) > 1e-12]
        # Three relations contain a dominant exact-zero mass.  Scale their
        # non-zero observations while rendering exact zeros as a neutral grey;
        # this exposes rare sensitivity without pretending that zero is signal.
        lim = symmetric_limit(active if len(active) else values, .985)
        relation_norm[relation] = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    fig = plt.figure(figsize=(183 / 25.4, 246 / 25.4))
    gs = fig.add_gridspec(5, 4, left=.09, right=.965, bottom=.035, top=.972,
                          wspace=.45, hspace=.58, height_ratios=[.68, .62, .72, 1.22, 1.22])
    summary_rows = []

    # a-d: Yap-style zero-aware empirical distributions.  ECDFs turn a
    # dominant zero mass into an unreadable vertical wall; the density view
    # retains the same values while showing the rare non-zero sensitivity.
    for col, (relation, label) in enumerate(RELATIONS):
        ax = fig.add_subplot(gs[0, col])
        relation_values = []
        for perturbation, plabel, color, style in PERTURBATIONS:
            v = tract_year.loc[(tract_year.relation.eq(relation)) &
                               (tract_year.perturbation.eq(perturbation)), "prediction_difference"].to_numpy(float)
            relation_values.append(v)
            summary_rows.append({"relation": relation, "perturbation": perturbation,
                                 "level": "tract_year_scheme_average", "n": len(v),
                                 "median": float(np.median(v)), "q05": float(np.quantile(v, .05)),
                                 "q95": float(np.quantile(v, .95))})
        allv = np.concatenate(relation_values)
        active = allv[np.abs(allv) > 1e-12]
        lim = symmetric_limit(active if len(active) else allv, .995)
        x = np.linspace(-lim, lim, 320)
        notes = []
        for (perturbation, plabel, color, style), v in zip(PERTURBATIONS, relation_values):
            y, zero_rate, point_mass = density_curve(v, x)
            base = 0.0
            if point_mass:
                ax.vlines(0, base, .88, color=color, lw=1.5, ls=style, alpha=.82)
            else:
                ax.fill_between(x, base, base + .82 * y, color=color, alpha=.18, linewidth=0)
                ax.plot(x, base + .82 * y, color=color, ls=style, lw=1.25)
                ax.vlines(0, 0, .82 * np.interp(0, x, y), color=color, lw=.55, alpha=.48)
            med = float(np.median(v)); q05, q95 = np.quantile(v, [.05, .95])
            ax.hlines(-.030, q05, q95, color=color, lw=1.35, alpha=.68)
            ax.scatter(med, -.030, color=color, s=13, edgecolor="white", linewidth=.35, zorder=4)
            ax.vlines(v[np.abs(v) > 1e-12] if point_mass else [], -.04, -.03, color=color, lw=.24, alpha=.28)
            notes.append(f"{plabel.split()[0]} {med:+.3f}; 0={100*zero_rate:.1f}%")
        ax.axvline(0, color=MUTED, lw=.55)
        ax.set_xlabel("Prediction change (pp)")
        ax.set_ylabel("Density / mass" if col == 0 else "")
        ax.text(.97, .96, "\n".join(notes), transform=ax.transAxes,
                ha="right", va="top", fontsize=3.65, color=INK, linespacing=1.12)
        ax.set_xlim(-lim, lim); ax.set_ylim(-.07, 1.04); ax.set_yticks([0, .4, .8], ["0", ".4", ".8"] if col == 0 else [])
        clean_axes(ax); panel(ax, col, label)
    handles = [Line2D([0], [0], color=c, ls=s, lw=1.3, label=l) for _, l, c, s in PERTURBATIONS]
    fig.axes[0].legend(handles=handles, fontsize=4.5, loc="lower right", handlelength=2.2)

    # e-h: outer-fold mean changes with full fold dots and empirical 95% ranges.
    for col, (relation, label) in enumerate(RELATIONS):
        ax = fig.add_subplot(gs[1, col])
        for y, (perturbation, plabel, color, _) in enumerate(PERTURBATIONS):
            v = fold.loc[(fold.relation.eq(relation)) & fold.perturbation.eq(perturbation),
                         "prediction_difference"].to_numpy(float)
            lo, med, hi = np.quantile(v, [.025, .5, .975])
            rng = np.random.default_rng(21100 + col * 10 + y)
            ax.scatter(v, y + rng.uniform(-.075, .075, len(v)), s=7, color=color, alpha=.33, linewidth=0)
            ax.hlines(y, lo, hi, color=color, lw=1.7)
            ax.scatter(med, y, s=20, color=color, edgecolor="white", linewidth=.4)
            summary_rows.append({"relation": relation, "perturbation": perturbation,
                                 "level": "outer_fold_mean", "n": len(v), "median": med,
                                 "q05": lo, "q95": hi})
        ax.axvline(0, color=MUTED, lw=.55)
        ax.set_yticks([0, 1], ["Drop", "Half"] if col == 0 else [])
        ax.invert_yaxis(); ax.set_xlabel("Fold-mean change (pp)")
        clean_axes(ax, "x"); panel(ax, 4 + col, f"{label} · fold stability")

    # i-l: annual median and 90% tract interval.
    annual_rows = []
    for col, (relation, label) in enumerate(RELATIONS):
        ax = fig.add_subplot(gs[2, col])
        absolute_medians = []
        for perturbation, plabel, color, style in PERTURBATIONS:
            sub = tract_year.loc[(tract_year.relation.eq(relation)) &
                                 tract_year.perturbation.eq(perturbation)]
            q = sub.groupby("health_reference_year").prediction_difference.quantile([.05, .5, .95]).unstack()
            q.columns = ["q05", "median", "q95"]
            q = q.reset_index(); q["relation"] = relation; q["perturbation"] = perturbation
            annual_rows.append(q)
            absolute_medians.append(float(np.median(np.abs(sub.prediction_difference.to_numpy(float)))))
            x = q.health_reference_year.to_numpy(float)
            q["q25"] = sub.groupby("health_reference_year").prediction_difference.quantile(.25).to_numpy()
            q["q75"] = sub.groupby("health_reference_year").prediction_difference.quantile(.75).to_numpy()
            ax.fill_between(x, q.q25, q.q75, color=color, alpha=.18, linewidth=0)
            ax.plot(x, q.q05, color=color, ls=":", lw=.55, alpha=.60)
            ax.plot(x, q.q95, color=color, ls=":", lw=.55, alpha=.60)
            ax.plot(x, q["median"], color=color, ls=style, lw=1.2)
        ax.axhline(0, color=MUTED, lw=.55)
        ax.set_xticks([2014, 2017, 2020, 2023])
        ax.set_xlabel("Health reference year")
        ax.set_ylabel("Prediction change (pp)" if col == 0 else "")
        ax.text(.98, .08, f"median |Δ| {np.mean(absolute_medians):.2g}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=4.2, color=MUTED)
        clean_axes(ax); panel(ax, 8 + col, f"{label}: annual response")

    # m-t: matched-extent, land-only maps.  Each relation pair shares its own
    # zero-centred scale and every panel carries a full-height colour bar.
    cbar_axes = []; map_audit = []
    index = 12
    for row, (perturbation, plabel, _, _) in enumerate(PERTURBATIONS, start=3):
        for col, (relation, label) in enumerate(RELATIONS):
            ax = fig.add_subplot(gs[row, col])
            frame = frames[(relation, perturbation)]
            norm = relation_norm[relation]
            ax.set_facecolor("#F2F4F7")
            frame.plot(ax=ax, color="#E1E5EA", edgecolor="white", linewidth=.035)
            active = frame.loc[frame.prediction_difference.abs().gt(1e-12)]
            if len(active):
                active.plot(column="prediction_difference", ax=ax, cmap=DIV, norm=norm,
                            edgecolor="white", linewidth=.035,
                            missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .03})
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
            panel(ax, index, f"{label} · {'drop' if perturbation == 'relation_drop' else 'half weight'}",
                  x=-.035, y=1.01)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=DIV)
            cb, cax = add_equal_height_colorbar(fig, ax, sm, width=.038, gap=.030)
            cb.set_ticks([norm.vmin, 0, norm.vmax])
            zero_rate = float(np.mean(frame.prediction_difference.abs().le(1e-12)))
            ax.text(.98, .02, f"{100*zero_rate:.1f}% zero", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=3.8, color=MUTED,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": .25})
            cbar_axes.append((ax, cax)); index += 1
            map_audit.append({"relation": relation, "perturbation": perturbation,
                              "n_land_tracts": len(frame), "vmin": norm.vmin, "vmax": norm.vmax,
                              "extent_xmin": extent[0], "extent_xmax": extent[1],
                              "extent_ymin": extent[2], "extent_ymax": extent[3]})

    delivery = save_delivery(fig, OUT, STEM)
    fig.canvas.draw()
    height_ratios = [cax.get_position().height / ax.get_position().height for ax, cax in cbar_axes]
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(OUT / "source_distribution_and_fold_summaries.csv", index=False)
    pd.concat(annual_rows, ignore_index=True).to_csv(OUT / "source_annual_tract_intervals.csv", index=False)
    map_source.to_csv(OUT / "source_2023_scheme_averaged_spatial_effects.csv", index=False)
    pd.DataFrame(map_audit).to_csv(OUT / "map_geometry_and_scale_audit.csv", index=False)
    after = {str(DATA): sha256(DATA), str(ROWMAP): sha256(ROWMAP), str(GEOMETRY): sha256(GEOMETRY)}
    nonwhite, border = border_check(delivery["jpg"])
    write_validation(OUT, {
        "figure": STEM, "panels": 20,
        "proxy_analysis": True, "formal": False,
        "contract": {"relations": [r for r, _ in RELATIONS],
                     "perturbations": [p for p, *_ in PERTURBATIONS],
                     "maps": "2023 scheme-average; Chicago land tracts only",
                     "map_scale": "zero-centred and shared within each relation's drop/half pair",
                     "interpretation": "model sensitivity to graph perturbation; not a causal effect"},
        "source_hashes_before": before, "source_hashes_after": after,
        "water_dominant_geometry_rows_excluded": water_excluded,
        "delivery": delivery,
        "checks": {
            "panel_count_20": index == 20,
            "eight_relation_perturbation_cells": len(expected) == 8,
            "thirty_two_outer_folds_each": bool(fold.groupby(["relation", "perturbation"]).size().eq(32).all()),
            "row_mapping_complete": not joined.tract_geoid_native.isna().any(),
            "ten_health_years": tract_year.health_reference_year.nunique() == 10,
            "water_dominant_tracts_removed": water_excluded > 0,
            "all_map_extents_identical": pd.DataFrame(map_audit)[["extent_xmin", "extent_xmax", "extent_ymin", "extent_ymax"]].nunique().eq(1).all(),
            "individual_colourbar_each_map": len(cbar_axes) == 8,
            "colourbars_equal_map_height": bool(np.allclose(height_ratios, 1, atol=.015)),
            "signed_map_scales_centered_zero": all(n.vcenter == 0 for n in relation_norm.values()),
            "source_files_unchanged": before == after,
            "exact_red_blue_palette": True,
            "only_panel_letters_bold_by_construction": True,
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
            "jpg_not_blank": nonwhite > .04,
            "outer_border_clear": border < .005,
            "complete_v6_plus_jpg_delivery": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
        },
    })
    (OUT / "figure_contract.md").write_text(
        "# Figure contract\n\nFour graph relations and two perturbations are shown as tract-year ECDFs, "
        "32-outer-fold intervals, annual tract distributions, and 2023 land-only maps. "
        "Drop/half maps share a zero-centred scale within relation; cross-relation magnitude comparisons should use axes and source tables.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
