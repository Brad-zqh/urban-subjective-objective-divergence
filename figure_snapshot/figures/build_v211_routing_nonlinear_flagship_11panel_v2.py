"""Non-destructive C57 Fig. 14 typographic and palette-consistency pass.

The original V211 renderer remains the calculation source.  All 11 panels and
registered V211 source tables are retained; only display objects are revised.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.text import Text
from PIL import Image

import build_v211_routing_nonlinear_flagship_11panel as baseline
from nature_viz_common import blacken_figure_text


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_routing_nonlinear_flagship_11panel"
OUT = ROOT / "figures/v211_routing_nonlinear_flagship_11panel_v2"
STEM = "Fig_v211_routing_nonlinear_flagship_11panel_v2"
SOURCE_FILES = (
    "source_fold_paired_comparison.csv",
    "source_faithfulness_summary.csv",
    "source_feature_mask_summary.csv",
    "source_ses_mobility_relation_summary.csv",
    "source_curvature_sign_audit.csv",
    "source_pooled_dose_response_surface.csv",
    "source_curvature_stability_surface.csv",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_pair(fig, out_dir: Path, stem: str, *, dpi: int = 600) -> dict:
    axes = fig.axes[:11]
    if len(axes) != 11:
        raise RuntimeError("The full 11-panel routing plate was not drawn")

    # Free the cramped top-right of panel i without hiding any observation.
    note = next(text for text in axes[8].texts if text.get_text().startswith("≥75%"))
    note.set_text(note.get_text().replace("≥75%:", "≥75% ").replace("≥90%:", "≥90% "))
    note.set_position((.975, .82))
    note.set_fontsize(5.7)
    note.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": .4})
    axes[2].set_xlabel("Paired change (pp) · blue RMSE / red MAE")
    axes[9].set_title("Curvature sign by SES", loc="left")
    for text in axes[6].texts:
        match = re.fullmatch(r"median ([0-9.]+) · exact zero ([0-9]+%)", text.get_text())
        if match:
            text.set_text(f"med {match.group(1)} · {match.group(2)} zero")

    panel_letters = {
        id(next(text for text in ax.texts if text.get_text() == letter))
        for ax, letter in zip(axes, "abcdefghijk")
    }
    for artist in fig.findobj(match=Text):
        if id(artist) not in panel_letters:
            artist.set_fontweight("normal")
    fig.canvas.draw()
    blacken_figure_text(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    note_box = note.get_window_extent(renderer)
    title_box = axes[8].title.get_window_extent(renderer)
    x_label_box = axes[8].xaxis.label.get_window_extent(renderer)
    g_notes = [text.get_window_extent(renderer) for text in axes[6].texts if text.get_text().startswith("med ")]
    h_row_labels = [text.get_window_extent(renderer) for text in axes[7].get_yticklabels()]
    checks = {
        "eleven_panels_preserved": len(axes) == 11,
        "panel_i_note_clear_of_title": not note_box.overlaps(title_box),
        "panel_i_note_clear_of_xlabel": not note_box.overlaps(x_label_box),
        "g_notes_clear_of_h_labels": not any(
            left.overlaps(right) for left in g_notes for right in h_row_labels
        ),
        "only_panel_letters_bold": all(
            artist.get_fontweight() in ("normal", 400, "bold", 700)
            and (artist.get_fontweight() in ("normal", 400) or id(artist) in panel_letters)
            for artist in fig.findobj(match=Text)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Fig. 14 layout/typography failed: {checks}")

    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{stem}.svg"
    jpg = out_dir / f"{stem}.jpg"
    fig.savefig(svg, facecolor="white")
    fig.savefig(jpg, dpi=dpi, facecolor="white",
                pil_kwargs={"quality": 96, "subsampling": 0, "optimize": True})
    with Image.open(jpg) as image:
        image.verify()
    with Image.open(jpg) as image:
        dimensions = list(image.size)
        stored_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
    svg_text = svg.read_text(encoding="utf-8")
    (out_dir / "layout_qa.json").write_text(
        json.dumps({"panel_count": 11, "checks": checks}, indent=2), encoding="utf-8"
    )
    return {
        "jpg": str(jpg), "svg": str(svg),
        "jpg_dimensions_px": dimensions, "jpg_dpi_metadata": stored_dpi,
        "jpg_nonempty": jpg.stat().st_size > 100_000,
        "svg_nonempty": svg.stat().st_size > 10_000,
        "svg_text_editable": "<text" in svg_text,
        "delivery_formats": ["jpg", "svg"],
    }


def main() -> None:
    baseline.OUT = OUT
    baseline.STEM = STEM
    baseline.save_delivery = save_pair
    baseline.main()

    hashes = {}
    for name in SOURCE_FILES:
        old_hash = digest(OLD / name)
        new_hash = digest(OUT / name)
        hashes[name] = {
            "baseline_sha256": old_hash,
            "v2_sha256": new_hash,
            "identical": old_hash == new_hash,
        }
    if not all(item["identical"] for item in hashes.values()):
        raise RuntimeError("A registered V211 Fig. 14 source table changed")
    (OUT / "source_data_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update({
        "all_seven_source_tables_byte_identical": True,
        "only_jpg_svg_exported": sorted(path.suffix.lower() for path in OUT.glob(STEM + ".*")) == [".jpg", ".svg"],
        "eleven_panels_preserved": True,
    })
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 14 validation failed")


if __name__ == "__main__":
    main()
