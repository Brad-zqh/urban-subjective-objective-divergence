"""Resolve residual title, annual-tick and parity-bar collisions in version 3."""

from __future__ import annotations

from pathlib import Path

import build_v211_signed_delta_sensitivity_refined_v1 as base


ROOT = Path(__file__).resolve().parents[1]
base.OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v3"
_base_export = base.refined_export


def export_v3(fig, output, stem, *, dpi=600):
    for ax in fig.axes:
        title = ax.get_title(loc="left")
        if title == "Directional coverage by fold":
            ax.set_title("Fold coverage", loc="left")
        elif title == "Row-level error shift by year":
            ax.set_title("Error shift by year", loc="left")
        if title.startswith("Annual "):
            ax.set_xticks([2014, 2017, 2020, 2023])
        for child in ax.child_axes:
            if child.get_ylabel() == "Count/bin":
                child.set_ylabel("")
                child.set_yticks([])
    return _base_export(fig, output, stem, dpi=dpi)


base.refined_export = export_v3


if __name__ == "__main__":
    base.main()
