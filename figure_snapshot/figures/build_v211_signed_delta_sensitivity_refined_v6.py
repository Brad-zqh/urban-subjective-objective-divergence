"""Final 20-panel collision and concise-axis pass."""

from __future__ import annotations

from pathlib import Path

import build_v211_signed_delta_sensitivity_refined_v5 as revised


ROOT = Path(__file__).resolve().parents[1]
revised.revised.base.OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v6"
_save = revised.revised.base.save_jpg_svg


def save_v6(fig, output, *, dpi=600):
    for ax in fig.axes:
        title = ax.get_title(loc="left")
        if title == "Paired fold ΔRMSE":
            ax.set_yticks([-.1, 0, .2, .4])
        if title in {"Explicit S−O representation", "O + S + SES representation"}:
            ax.set_xlabel("Observed MHLTH (%)")
    return _save(fig, output, dpi=dpi)


revised.revised.base.save_jpg_svg = save_v6


if __name__ == "__main__":
    revised.revised.base.main()
