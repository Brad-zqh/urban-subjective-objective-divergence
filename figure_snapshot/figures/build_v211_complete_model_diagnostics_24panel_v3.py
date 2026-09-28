"""Versioned title and spacing repair of the current C57 24-panel diagnostic atlas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.text import Text
from PIL import Image

import build_v211_complete_model_diagnostics_24panel_v2 as baseline
from nature_viz_common import blacken_figure_text


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_complete_model_diagnostics_24panel_v2"
OUT = ROOT / "figures/v211_complete_model_diagnostics_24panel_v3"
STEM = "Fig_v211_complete_model_diagnostics_24panel_v3"
SOURCE_FILES = (
    "source_annual_descriptive_metrics.csv",
    "source_descriptive_profile_metrics.csv",
    "source_locked_2023_metrics.csv",
    "source_mm_gtgnnwr_spatial_outer_fold_profiles.parquet",
    "source_registered_prediction_profiles.parquet",
    "source_residual_summary.csv",
)
original_panel = baseline.panel


def concise_panel(ax, index: int, title: str = "", **kwargs):
    title = title.replace(" · 2014–2023 profiles", " · 2014–23")
    title = title.replace(" · spatial folds", " · spatial CV")
    title = title.replace("Complete model comparison", "Model comparison")
    title = title.replace("Residual centre and spread", "Residual centre/spread")
    original_panel(ax, index, title, **kwargs)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_pair(fig, out_dir: Path, stem: str, *, dpi: int = 600) -> dict:
    axes = fig.axes
    if len(axes) != 24:
        raise RuntimeError(f"Expected 24 panels; found {len(axes)}")
    alphabet = "abcdefghijklmnopqrstuvwx"
    letters = {id(text) for ax in axes for text in ax.texts if text.get_text() in alphabet and len(text.get_text()) == 1}
    if len(letters) != 24:
        raise RuntimeError(f"Expected 24 panel letters; found {len(letters)}")
    by_letter = {text.get_text(): ax for ax in axes for text in ax.texts
                 if text.get_text() in alphabet and len(text.get_text()) == 1}
    # The sole overlong row key in h crossed g's mandatory equal-height
    # density colourbar.  The full model name remains in g, p, and q.
    h = by_letter["h"]
    h.set_yticklabels(["MM*" if "MM-GTGNNWR" in item.get_text() else item.get_text()
                       for item in h.get_yticklabels()])
    h_box = h.get_position()
    h.set_position([h_box.x0 + .05, h_box.y0, h_box.width - .05, h_box.height])
    for artist in fig.findobj(match=Text):
        if id(artist) not in letters:
            artist.set_fontweight("normal")
    fig.canvas.draw()
    blacken_figure_text(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    collisions = []
    outside = []
    for index, letter in enumerate(alphabet):
        title = by_letter[letter]._left_title
        title_box = title.get_window_extent(renderer)
        if title_box.x1 >= fig.bbox.x1 - 2:
            outside.append(letter)
        if index % 4 < 3:
            next_letter = alphabet[index + 1]
            next_text = next(text for text in by_letter[next_letter].texts if text.get_text() == next_letter)
            if title_box.overlaps(next_text.get_window_extent(renderer)):
                collisions.append(letter)
    g_bar = by_letter["g"].child_axes[0]
    gh_label_collisions = [
        (bar_label.get_text(), row_label.get_text())
        for bar_label in g_bar.get_yticklabels()
        for row_label in h.get_yticklabels()
        if bar_label.get_window_extent(renderer).overlaps(row_label.get_window_extent(renderer))
    ]
    checks = {
        "twenty_four_panels_preserved": len(axes) == 24,
        "titles_clear_of_next_panel_letters": not collisions,
        "titles_inside_canvas": not outside,
        "g_colourbar_clear_of_h_model_labels": not gh_label_collisions,
        "only_panel_letters_bold": all(
            text.get_fontweight() in ("normal", 400) or id(text) in letters
            for text in fig.findobj(match=Text)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Fig. 3 layout gate failed: {checks}; collisions={collisions}; outside={outside}; g/h={gh_label_collisions}")
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
    (out_dir / "layout_qa.json").write_text(
        json.dumps({"panel_count": 24, "checks": checks}, indent=2), encoding="utf-8"
    )
    return {"jpg": str(jpg), "svg": str(svg), "jpg_dimensions_px": dimensions,
            "jpg_dpi_metadata": stored_dpi, "jpg_nonempty": jpg.stat().st_size > 100_000,
            "svg_nonempty": svg.stat().st_size > 10_000,
            "svg_text_editable": "<text" in svg.read_text(encoding="utf-8"),
            "delivery_formats": ["jpg", "svg"]}


def main() -> None:
    baseline.OUT = OUT
    baseline.STEM = STEM
    baseline.panel = concise_panel
    baseline.save_delivery = save_pair
    baseline.main()
    hashes = {}
    for name in SOURCE_FILES:
        old_hash, new_hash = digest(OLD / name), digest(OUT / name)
        hashes[name] = {"baseline_sha256": old_hash, "v3_sha256": new_hash,
                        "identical": old_hash == new_hash}
    if not all(row["identical"] for row in hashes.values()):
        raise RuntimeError("Registered C57 Fig. 3 source tables differ")
    (OUT / "source_data_sha256.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = validation["checks"]
    checks.pop("complete_v6_plus_jpg_delivery", None)
    checks.update({
        "six_source_tables_byte_identical": True,
        "requested_jpg_svg_only": sorted(path.suffix.lower() for path in OUT.glob(STEM + ".*")) == [".jpg", ".svg"],
        **json.loads((OUT / "layout_qa.json").read_text(encoding="utf-8"))["checks"],
    })
    validation["all_checks_passed"] = all(checks.values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 3 validation failed")


if __name__ == "__main__":
    main()
