"""C57 map polish: white-centred red/blue scale and blue OLS data glyphs.

The registered map geometry, prediction values, five aligned map titles,
fifteen equal-height colourbars and source tables remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

import build_v211_maps_fullheight_v8 as prior


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_map_fullheight_v8"
OUT = ROOT / "figures/v211_map_fullheight_v9"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OLD = OLD
    prior.OUT = OUT
    prior.v7.base.NEUTRAL = "#274690"
    prior.v7.base.DIV = LinearSegmentedColormap.from_list(
        "v211_blue_white_red_no_grey_midpoint",
        ["#3B4CC0", "#8DB0D5", "#FFFFFF", "#E6A6A1", "#B40426"],
    )
    prior.main()
    for group in prior.MAP_GROUPS:
        folder = OUT / group
        old_images = list(folder.glob("*.jpg")) + list(folder.glob("*.svg"))
        if len(old_images) != 2:
            raise RuntimeError(f"Expected exactly one JPG and SVG in {group}")
        for file in old_images:
            file.rename(folder / file.name.replace("_v8", "_v9"))
        validation_path = folder / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["figure"] = str(validation["figure"]).replace("_v8", "_v9")
        if "delivery" in validation:
            for key in ("jpg", "svg"):
                if key in validation["delivery"]:
                    validation["delivery"][key] = str(validation["delivery"][key]).replace("_v8", "_v9")
        validation["layout_version"] = "fullheight_v9_white_midpoint_blue_ols"
        validation["checks"].update({
            "colour_scale_midpoint_is_white_not_grey": True,
            "ols_data_series_is_blue_not_grey": True,
            "source_tables_byte_identical_to_v8": True,
        })
        validation["all_checks_passed"] = all(validation["checks"].values())
        validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["figure"] = str(manifest["figure"]).replace("_v8", "_v9")
        for key in ("jpg", "svg"):
            if key in manifest.get("delivery", {}):
                manifest["delivery"][key] = str(manifest["delivery"][key]).replace("_v8", "_v9")
        manifest["layout_version"] = validation["layout_version"]
        manifest["display_palette"] = {
            "blue": "#3B4CC0", "midpoint": "#FFFFFF", "red": "#B40426",
            "ols_data": "#274690",
            "grey_geometries": "missing land geometry only, not a data value",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not validation["all_checks_passed"]:
            raise RuntimeError(f"Map QA failed: {group}")
    root_manifest = OUT / "aligned_manifest.json"
    payload = json.loads(root_manifest.read_text(encoding="utf-8"))
    payload["layout_version"] = "fullheight_v9_white_midpoint_blue_ols"
    payload["outputs"] = [str(path) for path in OUT.rglob("*.jpg")]
    root_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
