"""V211 SES atlas: saturate signed data marks while preserving all 15 panels.

This is a display-only revision. The original 104 × 8 registered table and
panel order are unchanged; its byte-identical source tables are checked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgb

import build_v211_inequality_flagship_refined_v1 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_inequality_flagship_refined_v3"
OLD = ROOT / "figures/v211_inequality_flagship_refined_v2"
original_save = prior.save_jpg_svg
BLUE = "#2443A8"
RED = "#B40426"


def signed_color(value: float, limit: float) -> tuple[float, float, float]:
    """Retain honest sign and magnitude, with a visible near-zero colour floor."""
    base = np.asarray(to_rgb(RED if value > 0 else BLUE), float)
    strength = .26 + .72 * min(abs(value) / max(limit, 1e-9), 1)
    return tuple(1 - strength * (1 - base))


def polished_save(fig, out_dir):
    # Limits are the same 98th-percentile display limits as the original.
    source = prior.pd.read_csv(prior.SOURCE)
    limit = max(float(np.nanquantile(np.abs(source.standardized_prediction_gap), .98)), 1e-9)
    axes = fig.axes
    if len(axes) != 15:
        raise RuntimeError(f"Expected all 15 SES panels, got {len(axes)}")
    hero, robust, *facets = axes
    # One hue per sign across the entire composition.  Do not let gray
    # baselines or pale median glyphs obscure the 8-scheme comparisons.
    for ax in [robust, *facets]:
        for line in ax.lines:
            if len(line.get_xdata()) == 2:
                line.set_color("#203A68")
                line.set_linewidth(.85)
        ax.grid(axis="x", color="#DCE7F5", linewidth=.45)
        ax.tick_params(axis="x", labelsize=6.5, colors="#182B48")
        ax.tick_params(axis="y", colors="#182B48")
        ax.title.set_fontsize(8.1)
    robust.set_xlabel("Standardized gap", fontsize=7.1, color="#182B48")
    # Each robust row has its range, eight observations, and median.
    if len(robust.collections) != 39:
        raise RuntimeError("Across-scheme summary lost a row or glyph")
    for row in range(13):
        line, dots, median = robust.collections[row * 3:row * 3 + 3]
        vals = np.asarray(dots.get_offsets(), float)[:, 0]
        med = float(np.asarray(median.get_offsets(), float)[0, 0])
        line.set_color(signed_color(med, limit))
        line.set_linewidth(1.9)
        line.set_alpha(.88)
        dots.set_facecolors([signed_color(float(v), limit) for v in vals])
        dots.set_sizes(np.full(8, 11.0))
        dots.set_edgecolors("white")
        dots.set_linewidths(.25)
        dots.set_alpha(.92)
        median.set_facecolor(signed_color(med, limit))
        median.set_edgecolor("#182B48")
        median.set_linewidth(.45)
        median.set_sizes([32])
    for ax in facets:
        if len(ax.collections) != 2:
            raise RuntimeError("Expected eight signed stems and eight points per SES facet")
        stems, points = ax.collections
        if not isinstance(stems, LineCollection) or not isinstance(points, PathCollection):
            raise RuntimeError("Unexpected SES facet artist types")
        values = np.asarray(points.get_offsets(), float)[:, 0]
        if len(values) != 8 or len(stems.get_segments()) != 8:
            raise RuntimeError("SES scheme count changed")
        colors = [signed_color(float(v), limit) for v in values]
        stems.set_colors(colors)
        stems.set_linewidths(1.5)
        points.set_facecolors(colors)
        points.set_edgecolors("#203A68")
        points.set_linewidths(.28)
        points.set_sizes(np.full(8, 29.0))
        ax.tick_params(axis="y", labelsize=6.25)
        ax.tick_params(axis="x", labelsize=6.4)
        if ax.get_xlabel():
            ax.set_xlabel("Standardized gap", fontsize=7.0, color="#182B48")
    return original_save(fig, out_dir)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.save_jpg_svg = polished_save
    prior.main()
    for name in ("source_all_scheme_ses_gaps.csv", "source_cross_scheme_summary.csv"):
        if prior.sha256(OLD / name) != prior.sha256(OUT / name):
            raise RuntimeError(f"SES source table changed: {name}")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_revision"] = "v3_signed_stems_readable_15panel"
    manifest["source_tables_byte_identical_to_v2"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
