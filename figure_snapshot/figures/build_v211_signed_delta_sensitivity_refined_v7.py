"""Vivid, white-centred signed heatmaps in the unchanged 20-panel atlas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

import build_v211_signed_delta_sensitivity_refined_v6 as prior


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_signed_delta_sensitivity_refined_v6"
OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v7"
PALETTE = LinearSegmentedColormap.from_list(
    "v211_vivid_signed_white_zero",
    [(0, "#2443A8"), (.25, "#4F79C4"), (.44, "#A9C1E5"),
     (.50, "#FFFFFF"), (.56, "#EBA5AB"), (.75, "#D45463"), (1, "#B40426")],
)
_save = prior.revised.revised.base.save_jpg_svg


def save_v7(fig, output, *, dpi=600):
    for ax in fig.axes[8:12]:
        if len(ax.images) != 1:
            raise RuntimeError("Expected one original data image per Fig. 4 heatmap")
        ax.images[0].set_cmap(PALETTE)
    return _save(fig, output, dpi=dpi)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.revised.revised.base.OUT = OUT
    prior.revised.revised.base.save_jpg_svg = save_v7
    prior.revised.revised.base.main()
    for path in OLD.glob("source_*.csv"):
        if digest(path) != digest(OUT / path.name):
            raise RuntimeError(f"Fig. 4 source changed: {path.name}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"]["four_signed_heatmaps_use_vivid_white_zero_scale"] = True
    validation["checks"]["all_source_csv_byte_identical_to_v6"] = True
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 4 visual QA failed")


if __name__ == "__main__":
    main()
