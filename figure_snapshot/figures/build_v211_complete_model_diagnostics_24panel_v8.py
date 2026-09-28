"""Make all seven residual histograms show a legible within-panel sign ramp.

Only whole-bin face colours change.  Occupied bins reaching at least 2% of
their panel's peak count define each sign's saturation endpoint; bin edges,
heights, source tables, model outputs and all other panels remain unchanged.
Colour depth is therefore not a shared numerical residual scale across models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb

import build_v211_complete_model_diagnostics_24panel_v3 as base
import build_v211_complete_model_diagnostics_24panel_v4 as v4
import build_v211_complete_model_diagnostics_24panel_v7 as prior
from nature_viz_common import sha256


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_complete_model_diagnostics_24panel_v7"
OUT = ROOT / "figures/v211_complete_model_diagnostics_24panel_v8"
STEM = "Fig_v211_complete_model_diagnostics_24panel_v8"
BLUE_DEEP = np.asarray(to_rgb("#2443A8"))
BLUE_PALE = np.asarray(to_rgb("#E8F0FB"))
RED_PALE = np.asarray(to_rgb("#FBE5E8"))
RED_DEEP = np.asarray(to_rgb("#B40426"))
VISIBLE_PEAK_FRACTION = .02
ENDPOINTS: list[dict[str, float]] = []


def colour_histogram(ax) -> tuple[int, list[float]]:
    bars = list(ax.patches)
    if len(bars) != 28:
        raise RuntimeError(f"Expected 28 registered residual bins, found {len(bars)}")
    centres = np.asarray([bar.get_x() + bar.get_width() / 2 for bar in bars], float)
    heights = np.asarray([bar.get_height() for bar in bars], float)
    original_geometry = [(bar.get_x(), bar.get_width(), bar.get_height()) for bar in bars]
    if not np.all(np.isfinite(centres)) or not np.all(np.isfinite(heights)) or heights.max() <= 0:
        raise RuntimeError("Invalid histogram geometry or empty residual panel")
    visible = heights >= VISIBLE_PEAK_FRACTION * heights.max()
    scales: dict[str, float] = {}
    for name, mask in (("negative", centres < 0), ("positive", centres >= 0)):
        active = mask & visible
        if not np.any(active):
            active = mask & (heights > 0)
        if not np.any(active):
            raise RuntimeError(f"Residual panel lacks {name} populated bins")
        scales[name] = float(np.max(np.abs(centres[active])))
    for centre, bar in zip(centres, bars):
        if centre < 0:
            fraction = min(1.0, abs(float(centre)) / scales["negative"])
            colour = BLUE_PALE + fraction * (BLUE_DEEP - BLUE_PALE)
        else:
            fraction = min(1.0, float(centre) / scales["positive"])
            colour = RED_PALE + fraction * (RED_DEEP - RED_PALE)
        bar.set_facecolor(colour)
        bar.set_alpha(1)
        bar.set_edgecolor("white")
        bar.set_linewidth(.22)
    if original_geometry != [(bar.get_x(), bar.get_width(), bar.get_height()) for bar in bars]:
        raise RuntimeError("Colour-only update altered histogram geometry")
    ENDPOINTS.append({"negative_pp": scales["negative"],
                      "positive_pp": scales["positive"]})
    return len(bars), centres.tolist()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    # Call the v3 renderer directly.  The v4->v5->v7 wrapper chain resets
    # v3.save_pair inside v4.main(), which silently bypasses v7's intended
    # whole-bin recolouring.  Binding immediately before v3.main() avoids it.
    base.OUT, base.OLD, base.STEM = OUT, OLD, STEM
    base.concise_panel = v4.readable_panel
    base.baseline.COLORS["OLS"] = "#274690"
    prior.colour_histogram = colour_histogram
    base.save_pair = prior.save_signed_ramp
    base.main()
    for source in OLD.glob("source_*.csv"):
        if sha256(source) != sha256(OUT / source.name):
            raise RuntimeError(f"Residual figure source changed: {source.name}")
    if len(ENDPOINTS) != 7:
        raise RuntimeError(f"Expected seven residual colour scales, found {len(ENDPOINTS)}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update({
        "seven_panels_use_visible_occupied_bin_colour_range": True,
        "all_residual_bin_geometry_unchanged": True,
        "source_tables_byte_equal_v7": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "residual_colour_mapping.json").write_text(json.dumps({
        "method": "separate sign ramps, saturation at outermost bin with count >= 2% of panel peak",
        "threshold_peak_fraction": VISIBLE_PEAK_FRACTION,
        "colour_depth_comparable_across_models": False,
        "panel_endpoints_i_to_o": ENDPOINTS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 3 v8 validation failed")


if __name__ == "__main__":
    main()
