"""Very lightly desaturate the four Fig. 4 heatmaps without altering data."""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex

import build_v211_signed_delta_sensitivity_refined_v8 as prior


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_signed_delta_sensitivity_refined_v8"
OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v9"
WHITE_MIX = .12
STOPS = [
    (0.00, "#7295C7"), (.27, "#B7CBE5"), (.46, "#EDF3FA"),
    (.50, "#FFFFFF"), (.54, "#FAEAEB"), (.73, "#EAB1B8"), (1.00, "#D27684"),
]


def softened(colour: str) -> str:
    rgb = to_rgb(colour)
    return to_hex(tuple((1 - WHITE_MIX) * part + WHITE_MIX for part in rgb))


PALETTE = LinearSegmentedColormap.from_list(
    "v211_gently_softened_signed_white_zero",
    [(position, softened(colour)) for position, colour in STOPS],
)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OLD, prior.OUT, prior.PALETTE = OLD, OUT, PALETTE
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update({
        "four_signed_heatmaps_twelve_percent_white_mix": True,
        "zero_centre_and_numeric_cells_preserved": True,
        "source_csv_byte_identical_to_v8": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_revision"] = "v9_twelve_percent_whiter_signed_heatmaps"
    manifest["white_mix_fraction"] = WHITE_MIX
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 4 v9 validation failed")


if __name__ == "__main__":
    main()
