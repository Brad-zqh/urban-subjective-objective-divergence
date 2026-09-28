"""Stronger signed residual colour while preserving the registered 24 panels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.colors import to_rgb

import build_v211_complete_model_diagnostics_24panel_v4 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_complete_model_diagnostics_24panel_v5"
OLD = ROOT / "figures/v211_complete_model_diagnostics_24panel_v4"
STEM = "Fig_v211_complete_model_diagnostics_24panel_v5"


def saturated_signed_bar(value: float, span: float):
    """Keep sign hue and magnitude ordering, with a visible saturation floor."""
    strength = min(abs(value) / max(span, 1e-9), 1.0)
    amount = .57 + .40 * np.sqrt(strength)
    target = np.asarray(to_rgb("#B40426" if value >= 0 else "#2443A8"))
    return tuple(1.0 - amount * (1.0 - target))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD, prior.STEM = OUT, OLD, STEM
    prior.signed_bar_colour = saturated_signed_bar
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"]["residual_saturation_floor_preserves_signed_magnitude_order"] = True
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 3 visual QA failed")


if __name__ == "__main__":
    main()
