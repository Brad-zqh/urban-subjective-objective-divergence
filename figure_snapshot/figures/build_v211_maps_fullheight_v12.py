"""Fig. 2 support spacing correction after the v11 visual review."""
from __future__ import annotations

import json
from pathlib import Path

import build_v211_maps_fullheight_v11 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_map_fullheight_v12"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.main()
    for group in prior.prior.prior.MAP_GROUPS:
        folder = OUT / group
        for old in list(folder.glob("*.jpg")) + list(folder.glob("*.svg")):
            old.rename(folder / old.name.replace("_v11", "_v12"))
        for name in ("validation.json", "manifest.json"):
            path = folder / name
            data = json.loads(path.read_text(encoding="utf-8"))
            data["figure"] = str(data["figure"]).replace("_v11", "_v12")
            for kind in ("jpg", "svg"):
                if kind in data.get("delivery", {}):
                    data["delivery"][kind] = str(data["delivery"][kind]).replace("_v11", "_v12")
            data["layout_version"] = "fullheight_v12_support_panels_clear_of_legend"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    root = OUT / "aligned_manifest.json"
    data = json.loads(root.read_text(encoding="utf-8"))
    data["outputs"] = [str(path) for path in OUT.rglob("*.jpg")]
    data["layout_version"] = "fullheight_v12_support_panels_clear_of_legend"
    root.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
