"""Zero-collision axis-tick pass for the 14-panel planning atlas."""

from __future__ import annotations

from pathlib import Path

import build_v211_planning_strategy_refined_v4 as revised


ROOT = Path(__file__).resolve().parents[1]
revised.base.OUT = ROOT / "figures/v211_planning_strategy_refined_v10"
_save = revised.base.save_jpg_svg


def save_v10(fig, output, *, dpi=600):
    revised.base.panel(fig, "Tract–year response").set_xticks([-.5, 0, .5])
    revised.base.panel(fig, "Response by poverty").set_xticks([-.5, 0, .5])
    residual = revised.base.panel(fig, "Package non-additivity residual")
    residual.set_yticks([-1, 0, 1])
    residual.set_ylabel(r"Residual ($\times 10^{-7}$ pp)")
    return _save(fig, output, dpi=dpi)


revised.base.save_jpg_svg = save_v10


if __name__ == "__main__":
    revised.base.main()
