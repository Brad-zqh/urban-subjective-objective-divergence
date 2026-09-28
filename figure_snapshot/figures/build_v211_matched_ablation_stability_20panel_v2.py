"""Non-destructive typesetting refinement of current C57 Fig. 6 (20 panels)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PIL import Image

import build_v211_matched_ablation_stability_20panel as baseline
from nature_viz_common import blacken_figure_text


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_matched_ablation_stability_20panel"
OUT = ROOT / "figures/v211_matched_ablation_stability_20panel_v2"
STEM = "Fig_v211_matched_ablation_stability_20panel_v2"
SOURCE_FILES = (
    "source_partition_ablation.csv",
    "source_block_ablation.csv",
    "source_all_model_fold_metrics.csv",
    "source_within_fold_r2_profiles.csv",
)
original_style = baseline.setup_style
original_figure = baseline.plt.figure
original_panel = baseline.panel
original_colourbar = baseline.add_equal_height_colorbar


def larger_type(_requested: float):
    original_style(6.05)


def tighter_figure(*args, **kwargs):
    fig = original_figure(*args, **kwargs)
    original_gridspec = fig.add_gridspec

    def compact_gridspec(*gs_args, **gs_kwargs):
        gs_kwargs["hspace"] = .42
        gs_kwargs["wspace"] = .58
        return original_gridspec(*gs_args, **gs_kwargs)

    fig.add_gridspec = compact_gridspec
    return fig


def concise_panel(ax, index: int, title: str = "", **kwargs):
    title = title.replace("One- vs four-relation MAE gain", "One/four · MAE")
    title = title.replace("One- vs four-relation RMSE", "One/four · RMSE")
    title = title.replace(" · ΔMAE vs Ridge", " · ΔMAE")
    title = title.replace("Fold-level squared-error scale", "Fold-level RMSE")
    title = title.replace("Fold-level absolute-error scale", "Fold-level MAE")
    title = title.replace("Fold-level explained variation", "Fold-level R²")
    original_panel(ax, index, title, **kwargs)


def compact_colourbar(*args, **kwargs):
    cb, cax = original_colourbar(*args, **kwargs)
    original_set_ticks = cb.set_ticks

    def set_ticks_with_short_labels(ticks, *tick_args, **tick_kwargs):
        result = original_set_ticks(ticks, *tick_args, **tick_kwargs)
        cb.formatter = FuncFormatter(lambda value, _: f"{value:+.2f}" if abs(value) > .005 else "0")
        cb.update_ticks()
        return result

    cb.set_ticks = set_ticks_with_short_labels
    return cb, cax


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_pair(fig, out_dir: Path, stem: str, *, dpi: int = 600) -> dict:
    axes = fig.axes
    if len(axes) != 20:
        raise RuntimeError(f"Expected 20 panels; found {len(axes)}")
    panel_letters = {
        id(text) for ax in axes for text in ax.texts
        if len(text.get_text()) == 1 and text.get_text() in "abcdefghijklmnopqrst"
    }
    if len(panel_letters) != 20:
        raise RuntimeError(f"Expected 20 distinct panel letters; found {len(panel_letters)}")
    by_letter = {text.get_text(): ax for ax in axes for text in ax.texts
                 if len(text.get_text()) == 1 and text.get_text() in "abcdefghijklmnopqrst"}
    t_ax = by_letter["t"]
    t_ax.set_yticklabels([
        item.get_text().replace("Four-relation graph", "4-rel. graph")
        .replace("Fused four relations", "Fused 4-rel.")
        .replace("Mobility graph", "Mob. graph")
        .replace("Fused mobility", "Fused mob.")
        for item in t_ax.get_yticklabels()
    ])
    for artist in fig.findobj(match=Text):
        if id(artist) not in panel_letters:
            artist.set_fontweight("normal")
    fig.canvas.draw()
    blacken_figure_text(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # The five inset colourbars must retain map height and leave a real gap
    # before the next panel, including their formatted tick labels.
    heat_letters = "mnopt"
    colourbar_height = []
    tick_right_edges = []
    for letter in heat_letters:
        ax = by_letter[letter]
        if len(ax.child_axes) != 1:
            raise RuntimeError(f"Panel {letter} has no unique inset colourbar")
        cax = ax.child_axes[0]
        colourbar_height.append(cax.get_window_extent(renderer).height / ax.get_window_extent(renderer).height)
        tick_right_edges.append(max(label.get_window_extent(renderer).x1 for label in cax.get_yticklabels()))
    clear_of_next = all(tick_right_edges[j] + 4 < by_letter[heat_letters[j + 1]].get_window_extent(renderer).x0
                        for j in range(3))
    clear_of_border = all(tick_right_edges[j] < fig.bbox.x1 - 2 for j in (3, 4))
    s_right = by_letter["s"].get_window_extent(renderer).x1
    t_labels_clear_of_s = bool(min(item.get_window_extent(renderer).x0
                                   for item in t_ax.get_yticklabels()) > s_right + 2)
    title_letter_collisions = []
    for index, letter in enumerate("abcdefghijklmnopqrst"):
        if index % 4 == 3:
            continue
        title_box = by_letter[letter]._left_title.get_window_extent(renderer)
        next_letter = next(text for text in by_letter["abcdefghijklmnopqrst"[index + 1]].texts
                           if text.get_text() == "abcdefghijklmnopqrst"[index + 1])
        if title_box.overlaps(next_letter.get_window_extent(renderer)):
            title_letter_collisions.append(index)
    checks = {
        "twenty_panels_preserved": len(axes) == 20,
        "five_colourbars_equal_heatmap_height": all(abs(ratio - 1) < .01 for ratio in colourbar_height),
        "colourbar_ticks_clear_of_next_panel": clear_of_next,
        "last_colourbar_clear_of_border": clear_of_border,
        "last_heatmap_model_labels_clear_of_s_panel": t_labels_clear_of_s,
        "titles_clear_of_next_panel_letters": not title_letter_collisions,
        "only_panel_letters_bold": all(
            text.get_fontweight() in ("normal", 400) or id(text) in panel_letters
            for text in fig.findobj(match=Text)
        ),
    }
    if not all(checks.values()):
        next_left = [by_letter[heat_letters[j + 1]].get_window_extent(renderer).x0 for j in range(3)]
        raise RuntimeError(f"Fig. 6 layout gate failed: {checks}; tick edges={tick_right_edges}, next left={next_left}, fig right={fig.bbox.x1}")
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
        json.dumps({"panel_count": 20, "checks": checks,
                    "colourbar_to_panel_height": colourbar_height}, indent=2), encoding="utf-8"
    )
    return {
        "jpg": str(jpg), "svg": str(svg), "jpg_dimensions_px": dimensions,
        "jpg_dpi_metadata": stored_dpi, "jpg_nonempty": jpg.stat().st_size > 100_000,
        "svg_nonempty": svg.stat().st_size > 10_000,
        "svg_text_editable": "<text" in svg_text, "delivery_formats": ["jpg", "svg"],
    }


def main() -> None:
    baseline.OUT = OUT
    baseline.STEM = STEM
    baseline.setup_style = larger_type
    baseline.plt.figure = tighter_figure
    baseline.panel = concise_panel
    baseline.add_equal_height_colorbar = compact_colourbar
    baseline.save_delivery = save_pair
    baseline.main()
    hashes = {}
    for name in SOURCE_FILES:
        old_hash, new_hash = digest(OLD / name), digest(OUT / name)
        hashes[name] = {"baseline_sha256": old_hash, "v2_sha256": new_hash,
                        "identical": old_hash == new_hash}
    if not all(item["identical"] for item in hashes.values()):
        raise RuntimeError("Registered Fig. 6 source tables differ from the original")
    (OUT / "source_data_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = validation["checks"]
    checks.pop("complete_v6_plus_jpg_delivery", None)
    checks.update({
        "four_source_tables_byte_identical": True,
        "requested_jpg_svg_only": sorted(path.suffix.lower() for path in OUT.glob(STEM + ".*")) == [".jpg", ".svg"],
        **json.loads((OUT / "layout_qa.json").read_text(encoding="utf-8"))["checks"],
    })
    validation["all_checks_passed"] = all(checks.values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Fig. 6 validation failed")


if __name__ == "__main__":
    main()
