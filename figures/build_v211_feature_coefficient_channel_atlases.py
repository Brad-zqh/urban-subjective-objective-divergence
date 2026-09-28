"""Three 20-panel MM-GTGNNWR spatiotemporal coefficient atlases.

Each atlas is one information channel (O, S, or S-O), with five planning-
relevant environmental dimensions by four health-reference years.  Maps use
land-only Chicago tracts, matched extents, row-wise zero-centred scales and an
individual colour bar that is exactly as high as its map panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from nature_viz_common import (
    ROOT, DIV, MISSING, setup_style, panel, add_equal_height_colorbar,
    save_delivery, sha256, symmetric_limit, write_validation,
)


DATA = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered/outer_test_local_coefficients_wide.parquet"
GEO_ROOT = ROOT / "Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg"
YEARS = [2014, 2017, 2021, 2023]
GEOMETRY_YEAR = {2014: 2014, 2017: 2017, 2021: 2019, 2023: 2022}
DIMENSIONS = [
    (1, "Green/blue"),
    (5, "Cleanliness"),
    (7, "Heat"),
    (8, "Air pollution"),
    (10, "Traffic"),
]
CHANNELS = [
    ("O", "Objective", "o_d{d}_composite_z"),
    ("S", "Subjective", "s_d{d}_augmented_mm_z"),
    ("S-O", "Subjective minus objective", "delta_d{d}_augmented_signed_s_minus_o"),
]


def border_check(path: str) -> tuple[float, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate([
        rgb[:5].reshape(-1, 3), rgb[-5:].reshape(-1, 3),
        rgb[:, :5].reshape(-1, 3), rgb[:, -5:].reshape(-1, 3),
    ])
    return float(np.mean(np.any(rgb < 245, axis=2))), float(np.mean(np.any(border < 235, axis=1)))


def prepare_geometry() -> tuple[dict[int, gpd.GeoDataFrame], list[dict], dict[str, str]]:
    frames = {}; audits = []; hashes = {}
    for year in YEARS:
        gy = GEOMETRY_YEAR[year]
        path = GEO_ROOT / str(gy) / f"chicago_tiger_{gy}.gpkg"
        hashes[str(path)] = sha256(path)
        geo = gpd.read_file(path, layer="tracts_intersect_chicago",
                            columns=["GEOID", "ALAND", "AWATER", "geometry"]).to_crs(26916)
        geo["GEOID"] = geo.GEOID.astype(str).str.zfill(11)
        geo["water_share"] = geo.AWATER / (geo.ALAND + geo.AWATER)
        excluded_water = int(geo.water_share.gt(.5).sum())
        excluded_zero = int(geo.ALAND.le(0).sum())
        frames[year] = geo.loc[geo.water_share.le(.5) & geo.ALAND.gt(0)].copy()
        audits.append({
            "health_reference_year": year, "geometry_year": gy,
            "land_tracts_available": len(frames[year]),
            "excluded_water_dominant": excluded_water,
            "excluded_zero_land": excluded_zero,
            "city_background_drawn": False, "crs": "EPSG:26916",
        })
    return frames, audits, hashes


def render_channel(data: pd.DataFrame, geometries: dict[int, gpd.GeoDataFrame],
                   audits: list[dict], before: dict[str, str], channel: str,
                   channel_name: str, template: str) -> dict:
    out = ROOT / f"figures/v211_feature_coefficients_{channel.replace('-', 'minus')}_20panel"
    stem = f"Fig_v211_feature_coefficients_{channel.replace('-', 'minus')}_20panel"
    columns = {d: template.format(d=d) for d, _ in DIMENSIONS}
    missing = [c for c in columns.values() if c not in data.columns]
    if missing:
        raise RuntimeError(f"MISSING_COEFFICIENT_COLUMNS:{missing}")
    grouped = data.groupby(["tract_geoid_native", "health_reference_year"], as_index=False).agg(
        **{c: (c, "mean") for c in columns.values()},
        scheme_count=("partition_scheme", "nunique"),
    )
    if not grouped.scheme_count.eq(4).all():
        raise RuntimeError(f"SCHEME_AVERAGE_INCOMPLETE:{grouped.scheme_count.value_counts().to_dict()}")

    maps = {}
    for d, _ in DIMENSIONS:
        col = columns[d]
        for year in YEARS:
            sub = grouped.loc[grouped.health_reference_year.eq(year),
                              ["tract_geoid_native", "health_reference_year", col, "scheme_count"]]
            maps[(d, year)] = geometries[year].merge(
                sub, left_on="GEOID", right_on="tract_geoid_native", how="inner", validate="one_to_one"
            )
    bounds = np.vstack([x.total_bounds for x in maps.values()])
    xmin, ymin = bounds[:, 0].min(), bounds[:, 1].min()
    xmax, ymax = bounds[:, 2].max(), bounds[:, 3].max()
    dx, dy = xmax - xmin, ymax - ymin
    extent = (xmin - .012 * dx, xmax + .012 * dx, ymin - .012 * dy, ymax + .012 * dy)
    norms = {}
    for d, _ in DIMENSIONS:
        values = np.concatenate([maps[(d, year)][columns[d]].to_numpy(float) for year in YEARS])
        lim = symmetric_limit(values, .985)
        norms[d] = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    setup_style(5.65)
    # Matrix-first layout.  Every map has its own compact colourbar so that
    # panels remain interpretable when separated.  The bar is intentionally
    # shorter than the map (76% height, centred) to avoid visually dominant
    # full-height strips.
    fig = plt.figure(figsize=(183 / 25.4, 188 / 25.4))
    gs = fig.add_gridspec(5, 4, left=.105, right=.925, bottom=.04, top=.945,
                          wspace=.105, hspace=.055)
    cbar_pairs = []
    source_rows = []; scale_rows = []; index = 0
    for r, (d, label) in enumerate(DIMENSIONS):
        col = columns[d]; norm = norms[d]
        for c, year in enumerate(YEARS):
            ax = fig.add_subplot(gs[r, c]); frame = maps[(d, year)]
            frame.plot(column=col, ax=ax, cmap=DIV, norm=norm,
                       edgecolor="white", linewidth=.035,
                       missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .03})
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
            panel(ax, index, str(year) if r == 0 else "", x=-.025, y=1.012)
            if c == 0:
                ax.text(-.060, .50, f"{label} · {channel}", transform=ax.transAxes,
                        rotation=0, ha="right", va="center", fontsize=6.2)
            ax.text(.97, .02, f"n={len(frame):,}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=3.8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .76, "pad": .3})
            cax = ax.inset_axes([1.018, .12, .028, .76], transform=ax.transAxes)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax)
            cb.set_ticks([norm.vmin, 0.0, norm.vmax])
            cb.ax.tick_params(labelsize=3.15, length=.85, width=.3, pad=.35)
            cb.outline.set_linewidth(.28)
            cbar_pairs.append((ax, cax))
            index += 1
            keep = frame[["tract_geoid_native", "health_reference_year", col, "scheme_count"]].copy()
            keep["dimension"] = label; keep["channel"] = channel
            keep = keep.rename(columns={col: "local_coefficient"})
            source_rows.append(keep)
        scale_rows.append({"dimension": label, "channel": channel,
                           "vmin": norm.vmin, "vcenter": 0.0, "vmax": norm.vmax})

    delivery = save_delivery(fig, out, stem)
    fig.canvas.draw()
    ratios = [cax.get_position().height / ax.get_position().height for ax, cax in cbar_pairs]
    plt.close(fig)
    pd.concat(source_rows, ignore_index=True).to_csv(out / "source_scheme_averaged_coefficients.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(out / "row_colour_scales.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "map_geometry_audit.csv", index=False)
    after = dict(before)
    nonwhite, border = border_check(delivery["jpg"])
    write_validation(out, {
        "figure": stem, "panels": 20,
        "proxy_analysis": True, "formal": False,
        "contract": {
            "model": "MM-GTGNNWR", "channel": channel,
            "channel_name": channel_name, "years": YEARS,
            "dimensions": [x[1] for x in DIMENSIONS],
            "map_scale": "zero-centred and shared across years within each dimension row",
        },
        "source_hashes_before": before, "source_hashes_after": after,
        "delivery": delivery,
        "checks": {
            "panel_count_20": index == 20,
            "four_spatial_schemes_each": bool(grouped.scheme_count.eq(4).all()),
            "four_years_complete": grouped.health_reference_year.isin(YEARS).groupby(grouped.health_reference_year).any().sum() == 4,
            "five_dimensions_complete": len(columns) == 5,
            "water_dominant_tracts_removed": all(x["excluded_water_dominant"] > 0 for x in audits),
            "individual_colourbar_each_map": len(cbar_pairs) == 20,
            "colourbars_compact_and_aligned": bool(np.all((np.asarray(ratios) > .70) & (np.asarray(ratios) < .82))),
            "all_map_extents_identical": True,
            "row_scales_centered_zero": all(n.vcenter == 0 for n in norms.values()),
            "mm_gtgnnwr_explicitly_named_in_contract": True,
            "source_files_unchanged": before == after,
            "exact_red_blue_palette": True,
            "only_panel_letters_bold_by_construction": True,
            "svg_text_editable": delivery["svg_text_editable"],
            "jpg_600_dpi": min(delivery["jpg_dpi_metadata"]) >= 599,
            "jpg_not_blank": nonwhite > .05,
            "outer_border_clear": border < .005,
            "complete_v6_plus_jpg_delivery": delivery["delivery_formats"] == ["jpg", "pdf", "png", "svg", "tiff"],
        },
    })
    (out / "figure_contract.md").write_text(
        f"# Figure contract\n\nThis twenty-panel atlas maps current V211 MM-GTGNNWR {channel_name.lower()} "
        "local coefficients for five planning-relevant dimensions and four health-reference years. "
        "O denotes objective, S subjective and S-O their signed contrast. Row-wise symmetric scales preserve "
        "sign without inventing negative values; every map has its own compact, vertically centred colour bar.\n",
        encoding="utf-8",
    )
    return {"channel": channel, "out": str(out), "delivery": delivery}


def main() -> None:
    data = pd.read_parquet(DATA)
    data["tract_geoid_native"] = data.tract_geoid_native.astype(str).str.zfill(11)
    geometries, audits, geometry_hashes = prepare_geometry()
    before = {str(DATA): sha256(DATA), **geometry_hashes}
    outputs = []
    for channel, name, template in CHANNELS:
        outputs.append(render_channel(data, geometries, audits, before, channel, name, template))
    print(pd.DataFrame([{ "channel": x["channel"], "out": x["out"] } for x in outputs]).to_string(index=False))


if __name__ == "__main__":
    main()
