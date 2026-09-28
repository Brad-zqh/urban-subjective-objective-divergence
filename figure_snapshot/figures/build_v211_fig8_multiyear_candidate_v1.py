"""Four signed-coefficient dimensions across three years, from V211 R3 only."""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import build_v211_main_editorial_v2 as editorial
from nature_viz_common import ROOT, setup_style


INPUT = ROOT / "workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered/outer_test_local_coefficients_wide.parquet"
OUT = ROOT / "figures/v211_fig8_multiyear_candidate_v1"
STEM = "Fig8_signed_coefficient_multiyear_candidate_v1"
YEARS = (2014, 2019, 2023)
DIMENSIONS = ((4, "Safety and order"), (5, "Cleanliness and maintenance"), (9, "Noise"), (10, "Traffic pressure"))
INK = "#1C2733"
MISSING = "#E6EAF0"
MAP_CMAP = LinearSegmentedColormap.from_list(
    "v211_signed_coefficients",
    [(0, "#2247A5"), (.20, "#6F95C8"), (.42, "#DCE8F4"), (.50, "#FFFFFF"),
     (.58, "#F9E2E2"), (.80, "#D66D7D"), (1, "#AC1837")],
)
TREND_COLORS = {4: "#2F5CA8", 5: "#BE2A45", 9: "#655493", 10: "#61727F"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_data() -> tuple[pd.DataFrame, dict[int, gpd.GeoDataFrame], str]:
    original_hash = sha256(INPUT)
    columns = [f"delta_d{d}_augmented_signed_s_minus_o" for d, _ in DIMENSIONS]
    frame = pd.read_parquet(INPUT, columns=["health_reference_year", "tract_geoid_native", *columns])
    frame["tract_geoid_native"] = frame.tract_geoid_native.astype(str).str.zfill(11)
    assert sorted(frame.health_reference_year.dropna().astype(int).unique()) == list(range(2014, 2024))
    pooled = frame.groupby(["health_reference_year", "tract_geoid_native"], as_index=False)[columns].mean()
    geometry = {year: editorial._geo(year) for year in YEARS}
    for year, geo in geometry.items():
        geo["GEOID"] = geo.GEOID.astype(str).str.zfill(11)
        assert geo.GEOID.is_unique, year
    assert sha256(INPUT) == original_hash
    return pooled, geometry, original_hash


def plot() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    setup_style(6.7)
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    pooled, geometry, input_hash = source_data()
    columns = {d: f"delta_d{d}_augmented_signed_s_minus_o" for d, _ in DIMENSIONS}
    bounds = np.vstack([geo.total_bounds for geo in geometry.values()])
    xmin, ymin, xmax, ymax = bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()
    dx, dy = xmax - xmin, ymax - ymin
    extent = (xmin - .015 * dx, xmax + .015 * dx, ymin - .015 * dy, ymax + .015 * dy)

    fig = plt.figure(figsize=(190 / 25.4, 204 / 25.4), facecolor="white")
    fig.text(.5, .982, "Signed platform–objective local coefficients across reference years",
             ha="center", va="top", fontsize=8.7, fontweight="bold", color=INK)
    fig.text(.935, .948, "β (pp/SD)", ha="center", va="bottom", fontsize=6.0, color=INK)
    x_positions = (.155, .407, .659)
    y_positions = (.755, .560, .365, .170)
    map_width, map_height = .218, .181
    for x, year in zip(x_positions, YEARS):
        fig.text(x + map_width / 2, .946, str(year), ha="center", va="bottom",
                 fontsize=7.4, fontweight="bold", color=INK)
    map_rows, summaries, display_limits = [], [], {}
    maps_for_qa = []
    for row, (dimension, label) in enumerate(DIMENSIONS):
        column = columns[dimension]
        selected = pooled.loc[pooled.health_reference_year.isin(YEARS), column].dropna().to_numpy(float)
        limit = max(float(np.quantile(np.abs(selected), .98)), 1e-6)
        display_limits[f"D{dimension}"] = limit
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        fig.text(.075, y_positions[row] + map_height / 2, label,
                 ha="center", va="center", rotation=90, fontsize=7.2,
                 fontweight="bold", color=INK)
        for col, year in enumerate(YEARS):
            ax = fig.add_axes([x_positions[col], y_positions[row], map_width, map_height])
            frame = geometry[year].merge(
                pooled.loc[pooled.health_reference_year.eq(year), ["tract_geoid_native", column]],
                left_on="GEOID", right_on="tract_geoid_native", how="left", validate="one_to_one",
            ).rename(columns={column: "coefficient"})
            n = int(frame.coefficient.notna().sum())
            assert n > 700, (dimension, year, n)
            frame.plot(column="coefficient", ax=ax, cmap=MAP_CMAP, norm=norm,
                       edgecolor="white", linewidth=.009,
                       missing_kwds={"color": MISSING, "edgecolor": "white", "linewidth": .009})
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
            panel_letter = string.ascii_lowercase[row * 3 + col]
            ax.text(-.07, 1.025, panel_letter, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, fontweight="bold", color=INK)
            ax.text(.98, .025, f"n = {n}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.5, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": .16})
            maps_for_qa.append(ax)
            map_rows.append(frame[["GEOID", "coefficient"]].assign(
                health_reference_year=year, dimension=f"D{dimension}",
                dimension_name=label, direct_land_tracts=n))
        cax = fig.add_axes([.906, y_positions[row] + .012, .011, map_height - .024])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=MAP_CMAP), cax=cax)
        cb.set_ticks([-limit, 0, limit])
        cb.set_ticklabels([f"−{limit:.3f}", "0", f"+{limit:.3f}"])
        cb.ax.tick_params(labelsize=5.4, length=1.1, pad=1.0, colors=INK)
        cb.outline.set_linewidth(.3)

    trend = fig.add_axes([.17, .055, .69, .083])
    annual_rows = []
    for dimension, label in DIMENSIONS:
        column = columns[dimension]
        annual = pooled.groupby("health_reference_year")[column].agg(
            median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), n="count").reset_index()
        annual["dimension"] = f"D{dimension}"
        annual["dimension_name"] = label
        annual_rows.append(annual)
        trend.plot(annual.health_reference_year, annual["median"],
                   color=TREND_COLORS[dimension], lw=1.5, marker="o", markersize=2.1,
                   label=label)
    trend.axhline(0, color="#7F8996", lw=.6, ls="--")
    trend.set_xlim(2013.7, 2023.3)
    trend.set_xticks([2014, 2016, 2018, 2020, 2022, 2023])
    trend.set_ylabel("Median β (pp/SD)", fontsize=6.1)
    trend.set_xlabel("Health-reference year", fontsize=6.1)
    trend.tick_params(labelsize=5.7, length=2, colors=INK)
    trend.spines[["top", "right"]].set_visible(False)
    trend.spines[["left", "bottom"]].set_linewidth(.5)
    trend.grid(axis="y", color="#E3E8EF", lw=.5)
    trend.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.42),
                 frameon=False, fontsize=5.6, handlelength=1.7, columnspacing=1.1)
    trend.text(-.085, 1.32, "m", transform=trend.transAxes, ha="right", va="bottom",
               fontsize=8.5, fontweight="bold", color=INK)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in maps_for_qa:
        assert not ax.get_window_extent(renderer).overlaps(trend.get_window_extent(renderer))
    OUT.mkdir(parents=True)
    source_maps = pd.concat(map_rows, ignore_index=True)
    source_trend = pd.concat(annual_rows, ignore_index=True)
    source_maps.to_csv(OUT / "source_multiyear_maps.csv", index=False)
    source_trend.to_csv(OUT / "source_annual_median_coefficients.csv", index=False)
    jpg = OUT / f"{STEM}.jpg"
    for ext, kwargs in (("jpg", {"dpi": 600, "pil_kwargs": {"quality": 96, "subsampling": 0}}),
                        ("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{STEM}.{ext}", facecolor="white", **kwargs)
    plt.close(fig)
    contract = (
        "# Figure contract — multiyear signed coefficients\n\n"
        "Core conclusion: Fitted signed local coefficients vary spatially and across years, while annual medians can mask local sign heterogeneity.\n"
        "Archetype: image plate + quantitative trend. Backend: Python. Final size: 190 × 204 mm; Word placement 169 × 181 mm.\n"
        "Panels a–l: four prespecified dimensions across 2014, 2019 and 2023; m: annual median trajectory 2014–2023.\n"
        "Selection: safety, cleanliness and noise were among the three largest signed median-absolute coefficients; traffic is the registered mobility/planning comparator.\n"
        "Evidence hierarchy: maps are primary; annual medians are context; the complete 43-variable/four-year atlas remains in SI.\n"
        "Statistics: each map labels its nonmissing land-tract n. Zero-centred colour limits are the 98th percentile of absolute pooled coefficients, shared across years within a dimension.\n"
        "Source data: V211 R3 outer-test coefficient Parquet only, plus annual TIGER tract geometry. No V208 or synthetic values.\n"
        "Image integrity: polygon geometry and values are unchanged; only a common within-dimension display clip is applied. Missing tracts are grey.\n"
        "Reviewer risk: maps are fitted coefficients in pp per fold-training SD, not independently identified effects; selected dimensions are explicit and SI retains the full atlas.\n"
    )
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")
    manifest = {
        "figure": STEM, "analysis": "V211 R3 compact proxy", "proxy_analysis": True, "formal": False,
        "selected_dimensions": [f"D{d}" for d, _ in DIMENSIONS], "years": list(YEARS),
        "panels": 13, "map_panels": 12, "input_sha256": input_hash,
        "display_abs_98pct_limits": display_limits,
        "source_multiyear_maps_sha256": sha256(OUT / "source_multiyear_maps.csv"),
        "source_annual_median_coefficients_sha256": sha256(OUT / "source_annual_median_coefficients.csv"),
        "source_rows": len(source_maps), "annual_rows": len(source_trend),
        "map_nonmissing_n": source_maps.groupby(["dimension", "health_reference_year"]).coefficient.count().to_dict(),
        "qa": {"source_unchanged": sha256(INPUT) == input_hash,
               "all_maps_have_n_gt_700": bool(source_maps.groupby(["dimension", "health_reference_year"]).coefficient.count().gt(700).all()),
               "twelve_maps_and_one_trend": len(maps_for_qa) == 12},
    }
    manifest["map_nonmissing_n"] = {f"{d}_{y}": int(n) for (d, y), n in manifest["map_nonmissing_n"].items()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(manifest["qa"].values())
    return {"jpg": str(jpg), "input_sha256": input_hash, "n_panels": 13, "qa": manifest["qa"]}


if __name__ == "__main__":
    print(json.dumps(plot(), ensure_ascii=False, indent=2))
