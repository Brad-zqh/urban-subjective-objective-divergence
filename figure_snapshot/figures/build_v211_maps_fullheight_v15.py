"""Slightly softer Fig. 2 p–r row hues; maps and data remain registered."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex, to_rgb

import build_v211_maps_fullheight_v14 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_map_fullheight_v15_soft_rows"
OLD = ROOT / "figures/v211_map_fullheight_v14_monochrome_rows"


def soften(value: str) -> str:
    rgb = np.asarray(to_rgb(value), float)
    return to_hex(.91 * rgb + .09)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.prior.MODEL_COLOURS = [soften(color) for color in prior.prior.MODEL_COLOURS]
    prior.main()
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():
            continue
        for old in list(folder.glob("*.jpg")) + list(folder.glob("*.svg")):
            old.rename(folder / old.name.replace("_v14", "_v15"))
        for name in ("validation.json", "manifest.json"):
            path = folder / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["figure"] = str(payload["figure"]).replace("_v14", "_v15")
            for kind in ("jpg", "svg"):
                if kind in payload.get("delivery", {}):
                    payload["delivery"][kind] = str(payload["delivery"][kind]).replace(
                        "_v14", "_v15"
                    )
            payload["layout_version"] = "fullheight_v15_soft_row_hues"
            if folder.name == "fig2_spatial_prediction":
                payload["row_colour_adjustment"] = "9% white mix of v14 model row hues only"
                source = "source_temporal_spatial_summaries.csv"
                payload["checks"]["annual_summary_source_byte_identical_to_v14"] = (
                    prior.prior._hash(OLD / folder.name / source) ==
                    prior.prior._hash(folder / source)
                )
                payload["all_checks_passed"] = all(payload["checks"].values())
                if not payload["all_checks_passed"]:
                    raise RuntimeError("Fig. 2 v15 QA failed")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    root = OUT / "aligned_manifest.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["layout_version"] = "fullheight_v15_soft_row_hues"
    payload["outputs"] = [str(path) for path in OUT.rglob("*.jpg")]
    root.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
