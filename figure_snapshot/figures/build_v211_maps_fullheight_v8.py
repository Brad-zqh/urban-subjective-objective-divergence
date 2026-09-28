"""Non-destructive map-header alignment after the full-height v7 contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import build_v211_maps_fullheight_v7 as v7


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_map_fullheight_v7"
OUT = ROOT / "figures/v211_map_fullheight_v8"
MAP_GROUPS = {
    "fig2_spatial_prediction": ["source_temporal_spatial_summaries.csv"],
    "fig8_coefficient_heterogeneity": ["source_cross_scheme_spread.csv", "source_pooled_2023_maps.csv"],
}
original_finish = v7.base.finish
headings = ["MM-GTGNNWR", "OLS anchor", "GTWR", "GTNNWR", "GTGNNWR"]


def aligned_finish(fig, folder, stem, panels, sources, core, chain, alignment):
    if stem.startswith("Fig2_"):
        titled_axes = {ax.get_title(): ax for ax in fig.axes if ax.get_title() in headings}
        if set(titled_axes) != set(headings):
            raise RuntimeError(f"Missing map-column headings: {set(headings) - set(titled_axes)}")
        for title in headings:
            ax = titled_axes[title]
            ax.set_title("")
            position = ax.get_position()
            fig.text(position.x0 + position.width / 2, .953, title, ha="center", va="bottom",
                     fontsize=7.2, fontweight="normal", color=v7.base.INK)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_artists = [next(item for item in fig.texts if item.get_text() == title)
                         for title in headings]
        boxes = [item.get_window_extent(renderer) for item in title_artists]
        same_baseline = max(box.y0 for box in boxes) - min(box.y0 for box in boxes) < 1.0
        separated = all(boxes[i].x1 + 2 < boxes[i + 1].x0 for i in range(4))
        inside = all(box.y1 < fig.bbox.y1 - 2 for box in boxes)
        if not (same_baseline and separated and inside):
            raise RuntimeError(f"Map header alignment failed: {same_baseline=}, {separated=}, {inside=}")
        title_qa = {"same_baseline": bool(same_baseline),
                    "headings_do_not_overlap": bool(separated), "headings_inside_canvas": bool(inside)}
    else:
        title_qa = {}
    new_stem = stem.replace("_v7", "_v8")
    output = original_finish(fig, folder, new_stem, panels, sources, core, chain, alignment)
    if title_qa:
        (output.parent / "layout_qa.json").write_text(
            json.dumps({"five_top_map_titles": headings, "checks": title_qa}, indent=2),
            encoding="utf-8",
        )
    return output


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v7.OUT_ROOT = OUT
    v7.base.finish = aligned_finish
    v7.main()
    hashes = {}
    for group, names in MAP_GROUPS.items():
        hashes[group] = {}
        for name in names:
            before, after = digest(OLD / group / name), digest(OUT / group / name)
            hashes[group][name] = {"v7_sha256": before, "v8_sha256": after,
                                    "identical": before == after}
            if before != after:
                raise RuntimeError(f"Map source table changed: {group}/{name}")
        validation_path = OUT / group / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        checks = validation["checks"]
        checks["source_tables_byte_identical_to_v7"] = True
        if group == "fig2_spatial_prediction":
            checks.update(json.loads((OUT / group / "layout_qa.json").read_text(encoding="utf-8"))["checks"])
        validation["layout_version"] = "fullheight_v8_aligned_titles"
        validation["all_checks_passed"] = all(checks.values())
        validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        if not validation["all_checks_passed"]:
            raise RuntimeError(f"Map validation failed: {group}")
    (OUT / "source_data_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    root_manifest_path = OUT / "aligned_manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["layout_version"] = "fullheight_v8_aligned_titles"
    root_manifest["source_tables_byte_identical_to_v7"] = True
    root_manifest_path.write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
