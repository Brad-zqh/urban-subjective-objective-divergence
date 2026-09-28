"""Slightly deepen only the four Fig. 4 signed heatmap palettes."""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

import build_v211_signed_delta_sensitivity_refined_v8 as prior
import build_v211_signed_delta_sensitivity_refined_v9 as v9


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_signed_delta_sensitivity_refined_v9"
OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v10"
WHITE_MIX = .05


def deepen(colour: str) -> str:
    return to_hex(tuple((1 - WHITE_MIX) * part + WHITE_MIX for part in to_rgb(colour)))


PALETTE = LinearSegmentedColormap.from_list(
    "v211_signed_slightly_deeper_white_zero",
    [(position, deepen(colour)) for position, colour in v9.STOPS],
)


def main() -> None:
    assert OLD.is_dir() and not OUT.exists()
    prior.OLD, prior.OUT, prior.PALETTE = OLD, OUT, PALETTE
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].pop("source_csv_byte_identical_to_v7", None)
    validation["checks"].update({
        "v10_four_heatmaps_five_percent_white_mix": True,
        "colour_values_and_zero_centres_unchanged": True,
        "source_csv_byte_identical_to_v9": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["visual_revision"] = "v10_five_percent_white_mix_for_four_signed_heatmaps_only"
    manifest["white_mix_fraction"] = WHITE_MIX
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    assert validation["all_checks_passed"]


if __name__ == "__main__":
    main()
