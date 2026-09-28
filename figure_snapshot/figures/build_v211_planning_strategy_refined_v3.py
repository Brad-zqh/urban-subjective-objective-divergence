"""Collision-focused planning-atlas repair; retains the original 14-panel grid."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.text import Annotation, Text

import build_v211_planning_strategy_20panel as legacy
from nature_viz_common import setup_style
from v211_refinement_common import save_jpg_svg, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_planning_strategy_refined_v3"
INPUTS = [legacy.DATA, legacy.COEF, legacy.GEOMETRY]


def panel(fig, title: str):
    matches = [ax for ax in fig.axes if ax.get_title(loc="left") == title]
    if len(matches) != 1:
        raise RuntimeError(f"Panel {title!r} appeared {len(matches)} times")
    return matches[0]


def export_v3(fig, output: Path, stem: str, *, dpi: int = 600) -> dict:
    a = panel(fig, "Tract–year response")
    b = panel(fig, "Directional coverage")
    c = panel(fig, "Magnitude–heterogeneity")
    d = panel(fig, "Poverty-stratified response")
    a.set_xlabel("Change (pp)")
    b.set_xlabel("Tracts improved (%)")
    c.set_xlabel("Median change (pp)")
    d.set_xlabel("Change (pp)")
    d.set_yticklabels([])
    if d.get_legend() is not None:
        d.get_legend().remove()

    # Seven verbose direct labels cannot fit a 33-mm panel. Preserve every
    # point and use one-character indexed labels tied to an outside key.
    direct = []
    for artist in list(c.texts):
        if isinstance(artist, Annotation) and artist.get_text() in legacy.SHORT.values():
            direct.append((artist.get_text(), artist.xy))
            artist.remove()
    if len(direct) != 7:
        raise RuntimeError(f"Expected seven scenario annotations, got {len(direct)}")
    number = {legacy.SHORT[key]: index + 1 for index, key in enumerate(legacy.KEYS)}
    for label, xy in direct:
        c.annotate(str(number[label]), xy=xy, xytext=(2, 2),
                   textcoords="offset points", fontsize=6.4,
                   ha="left", va="bottom", color="black")
    c.set_title("Response spread", loc="left")
    d.set_title("Response by poverty", loc="left")
    fig.text(
        .5, .755,
        "1 Greening   2 Upkeep   3 Cooling   4 Clean-air   5 Quiet streets   6 Traffic calming   7 Integrated retrofit",
        ha="center", va="center", fontsize=6.1, color="black",
    )

    corr = panel(fig, "Strategy-response correlation")
    short = ["Green", "Upkeep", "Cooling", "Air", "Quiet", "Traffic", "Retrofit"]
    corr.set_xticks(range(7), short, rotation=48, ha="right")
    corr.set_yticks(range(7), short)
    corr.set_title("Response correlation", loc="left")
    for child in corr.child_axes:
        if child.get_ylabel() == "":
            child.set_yticks([-1, 0, 1])
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        if artist.get_text().strip() and artist.get_fontsize() < 5.5:
            artist.set_fontsize(5.5)
        artist.set_color("#111111")
    return save_jpg_svg(fig, OUT, dpi=dpi)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    before = {str(path): sha256(path) for path in INPUTS}
    legacy.OUT = OUT
    legacy.STEM = OUT.name
    legacy.setup_style = lambda _: setup_style(7.0)
    legacy.save_delivery = export_v3
    legacy.main()
    after = {str(path): sha256(path) for path in INPUTS}
    if before != after:
        raise RuntimeError("V211 input changed during figure rendering")
    validation_file = OUT / "validation.json"
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    manifest_file = OUT / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    validation["source_hashes_before"] = before
    validation["source_hashes_after"] = after
    validation["checks"]["two_delivery_formats"] = manifest["delivery"]["delivery_formats"] == ["jpg", "svg"]
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_file.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["figure"] = OUT.name
    manifest["delivery_formats"] = ["jpg", "svg"]
    manifest["refinement"] = "collision-free indexed scenario labels, shorter axis text, compact correlation key; original 14-panel geometry and map scales retained"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Planning-atlas validation failed")
    print(json.dumps({"output": str(OUT), "panels": 14,
                      "source_unchanged": before == after,
                      "all_checks_passed": validation["all_checks_passed"]}))


if __name__ == "__main__":
    main()
