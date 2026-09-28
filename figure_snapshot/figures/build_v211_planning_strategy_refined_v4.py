"""Final compact-label pass for the full 14-panel planning candidate."""

from __future__ import annotations

from pathlib import Path

import build_v211_planning_strategy_refined_v3 as base


ROOT = Path(__file__).resolve().parents[1]
base.OUT = ROOT / "figures/v211_planning_strategy_refined_v4"
_save = base.save_jpg_svg


def save_v4(fig, output, *, dpi=600):
    summary = base.panel(fig, "Tract–year response")
    summary.set_yticks(range(7), ["Green", "Upkeep", "Cooling", "Air",
                                  "Quiet", "Traffic", "Retrofit"])
    corr = base.panel(fig, "Response correlation")
    numbers = [str(i) for i in range(1, 8)]
    corr.set_xticks(range(7), numbers)
    corr.set_yticks(range(7), numbers)
    corr.tick_params(labelsize=6.0, pad=1.0)
    for artist in corr.texts:
        try:
            value = float(artist.get_text())
        except ValueError:
            continue
        x, y = artist.get_position()
        artist.set_text("1" if round(x) == round(y) else f"{value:+.1f}")
        artist.set_fontsize(5.5)
    return _save(fig, output, dpi=dpi)


base.save_jpg_svg = save_v4


if __name__ == "__main__":
    base.main()
