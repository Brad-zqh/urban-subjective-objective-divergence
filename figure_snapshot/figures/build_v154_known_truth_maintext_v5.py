"""Separate Fig. 7 summary x labels from first map-row titles."""
from __future__ import annotations

import json
from pathlib import Path

import build_v154_known_truth_maintext_v4 as prior
from nature_viz_common import ROOT


OUT = ROOT / "figures/v154_known_truth_maintext_v5"
_save = prior.prior.base.save_jpg_svg


def clear_header_save(fig):
    for ax in fig.axes[:2]:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + .016, box.width, box.height - .016])
    return _save(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.prior.base.save_jpg_svg = clear_header_save
    prior.main()
    for extension in ("jpg", "svg"):
        old = OUT / f"Fig_v154_known_truth_maintext_v4.{extension}"
        old.rename(OUT / f"Fig_v154_known_truth_maintext_v5.{extension}")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["figure"] = "Fig_v154_known_truth_maintext_v5"
        for extension in ("jpg", "svg"):
            data["delivery"][extension] = str(data["delivery"][extension]).replace("_v4.", "_v5.")
        data["v5_visual_change"] = "Top summary axes raised to clear beta map titles"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
