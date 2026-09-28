"""Typography-collision pass with short partition labels and sparse ticks."""

from __future__ import annotations

from pathlib import Path

import build_v211_signed_delta_sensitivity_refined_v3 as revised


ROOT = Path(__file__).resolve().parents[1]
revised.base.OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v5"
_save = revised.base.save_jpg_svg


def save_v5(fig, output, *, dpi=600):
    for ax in fig.axes:
        title = ax.get_title(loc="left")
        if title.startswith("Paired fold "):
            ax.set_xticks([3.5, 11.5, 19.5, 27.5],
                          ["Axis", "R45", "Stripe", "Polar"])
            ax.tick_params(axis="x", labelrotation=0, labelsize=5.6)
        if title in {"Explicit S−O representation", "O + S + SES representation"}:
            ax.set_xticks([10, 20])
            ax.set_yticks([10, 20])
        if title == "Paired fold ΔMAE":
            ax.set_yticks([-.1, 0, .2, .4])
        if title == "Annual ΔR²":
            ax.set_yticks([-.08, -.04, 0])
        for artist in ax.texts:
            if len(artist.get_text()) == 1 and artist.get_text() in "abcdefghijklmnopqrst":
                artist.set_position((-.14, 1.07))
    return _save(fig, output, dpi=dpi)


revised.base.save_jpg_svg = save_v5


if __name__ == "__main__":
    revised.base.main()
