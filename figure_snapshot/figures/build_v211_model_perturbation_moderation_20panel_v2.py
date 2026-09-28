"""Versioned display refinement of the complete C57 Fig. 10 response atlas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.collections import PolyCollection
from matplotlib.text import Text
from PIL import Image

import build_v211_model_perturbation_moderation_20panel as baseline
from nature_viz_common import blacken_figure_text


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_model_perturbation_moderation_20panel"
OUT = ROOT / "figures/v211_model_perturbation_moderation_20panel_v2"
STEM = "Fig_v211_model_perturbation_moderation_20panel_v2"
SOURCE_FILES = (
    "source_cubic_response_curves.csv",
    "source_p10_p90_contrasts.csv",
    "scenario_dictionary.csv",
)
original_setup_style = baseline.setup_style
original_draw_curve = baseline.draw_curve
original_panel = baseline.panel


def larger_type(_requested: float):
    original_setup_style(6.05)


def vivid_curve(*args, **kwargs):
    ax = args[0]
    original_draw_curve(*args, **kwargs)
    ribbons = [artist for artist in ax.collections if isinstance(artist, PolyCollection)]
    if len(ribbons) >= 2:
        ribbons[0].set_alpha(.31)
        ribbons[1].set_alpha(.40)
    note = next(text for text in ax.texts if text.get_text().startswith("p90-p10 "))
    note.set_position((.98, .985))
    note.set_fontsize(5.5)
    note.set_color("#111111")
    note.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": .87, "pad": .25})


def short_panel(ax, index: int, title: str = "", **kwargs):
    title = title.replace(" · package", "")
    title = title.replace("Integrated retrofit", "Retrofit package")
    original_panel(ax, index, title, **kwargs)
    if index % 5 == 0:
        ax.set_ylabel("Model response (pp)")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_pair(fig, out_dir: Path, stem: str, *, dpi: int = 600) -> dict:
    axes = fig.axes[:20]
    if len(axes) != 20:
        raise RuntimeError("The complete 20-panel moderation atlas was not drawn")
    panel_letters = {
        id(next(text for text in ax.texts if text.get_text() == letter))
        for ax, letter in zip(axes, "abcdefghijklmnopqrst")
    }
    for artist in fig.findobj(match=Text):
        if id(artist) not in panel_letters:
            artist.set_fontweight("normal")
    fig.canvas.draw()
    blacken_figure_text(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    note_title_collision = []
    for index, ax in enumerate(axes):
        note = next(text for text in ax.texts if text.get_text().startswith("p90-p10 "))
        if note.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)):
            note_title_collision.append(index)
    checks = {
        "twenty_panels_preserved": len(axes) == 20,
        "notes_clear_of_panel_titles": not note_title_collision,
        "only_panel_letters_bold": all(
            text.get_fontweight() in ("normal", 400) or id(text) in panel_letters
            for text in fig.findobj(match=Text)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Fig. 10 typographic gate failed: {checks}")

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
        json.dumps({"panel_count": 20, "checks": checks}, indent=2), encoding="utf-8"
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
    baseline.setup_style = larger_type
    baseline.draw_curve = vivid_curve
    baseline.panel = short_panel
    baseline.save_delivery = save_pair
    baseline.main()

    hashes = {}
    for name in SOURCE_FILES:
        old_hash, new_hash = digest(OLD / name), digest(OUT / name)
        hashes[name] = {"baseline_sha256": old_hash, "v2_sha256": new_hash,
                        "identical": old_hash == new_hash}
    if not all(item["identical"] for item in hashes.values()):
        raise RuntimeError("A registered V211 Fig. 10 source table changed")
    (OUT / "source_data_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = validation["checks"]
    checks.pop("complete_v6_plus_jpg_delivery", None)  # User requests JPG/SVG only.
    checks.update({
        "twenty_panels_preserved": True,
        "three_source_tables_byte_identical": True,
        "requested_jpg_svg_only": sorted(path.suffix.lower() for path in OUT.glob(STEM + ".*")) == [".jpg", ".svg"],
    })
    validation["all_checks_passed"] = all(checks.values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 10 validation failed")


if __name__ == "__main__":
    main()
