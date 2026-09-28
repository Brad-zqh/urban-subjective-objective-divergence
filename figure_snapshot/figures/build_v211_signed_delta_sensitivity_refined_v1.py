"""Reflow the intact 20-panel V211 sensitivity atlas without altering source art."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.text import Text

import build_v211_signed_delta_sensitivity_20panel as legacy
from nature_viz_common import INK, setup_style
from v211_refinement_common import save_jpg_svg, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_signed_delta_sensitivity_refined_v1"
INPUTS = [
    legacy.FOLD_SOURCE, legacy.PARTITION_SOURCE, legacy.PREDICTION_SOURCE,
    legacy.SUMMARY_SOURCE, legacy.CHAIN_VALIDATION,
]


def refined_export(fig, output: Path, stem: str, *, dpi: int = 600) -> dict:
    # Remove microtype that physically occupies data regions. Its precise
    # statistics remain in the source tables and figure legend, not erased.
    for ax in fig.axes:
        for artist in list(ax.texts):
            label = artist.get_text().strip()
            if (
                label.startswith("points = folds")
                or label.startswith("median ")
                or label.startswith("positive favours")
                or label.startswith("negative favours")
                or label.startswith("pooled ")
            ):
                artist.remove()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.subplots_adjust(left=.090, right=.950, bottom=.103, top=.973,
                        wspace=.68, hspace=.66)
    handles = [
        Line2D([0], [0], marker="o", ls="none", color="none",
               markerfacecolor=legacy.SCHEME_COLORS[key], markeredgecolor="white",
               markersize=5, label=legacy.SCHEME_LABELS[key])
        for key in legacy.SCHEME_ORDER
    ]
    handles.append(Line2D([0], [0], marker="D", ls="none", color="none",
                          markerfacecolor=INK, markeredgecolor="white",
                          markersize=5, label="Pooled median"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .023),
               ncol=5, frameon=False, fontsize=6.0, columnspacing=1.1,
               handletextpad=.3)
    for text in fig.findobj(match=Text):
        if text.get_text().strip() and text.get_fontsize() < 5.6:
            text.set_fontsize(5.6)
        text.set_color("#111111")
    return save_jpg_svg(fig, OUT, dpi=dpi)


def main() -> None:
    before = {str(path): sha256(path) for path in INPUTS}
    if OUT.exists():
        raise FileExistsError(OUT)
    legacy.OUT = OUT
    legacy.STEM = OUT.name
    legacy.setup_style = lambda _: setup_style(7.0)
    legacy.save_delivery = refined_export
    legacy.main()
    after = {str(path): sha256(path) for path in INPUTS}
    if before != after:
        raise RuntimeError("Source data changed during figure refinement")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = validation["checks"]
    checks.pop("five_delivery_formats", None)
    checks["two_delivery_formats"] = validation["delivery"]["delivery_formats"] == ["jpg", "svg"]
    validation["all_checks_passed"] = all(checks.values())
    validation["source_hashes_before"] = before
    validation["source_hashes_after"] = after
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figure"] = OUT.name
    manifest["delivery_formats"] = ["jpg", "svg"]
    manifest["refinement"] = "data-label collision removal and shared legend; 20 panels retained"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Refined sensitivity validation failed")
    print(json.dumps({"output": str(OUT), "panels": 20,
                      "source_unchanged": before == after,
                      "all_checks_passed": validation["all_checks_passed"]}))


if __name__ == "__main__":
    main()
