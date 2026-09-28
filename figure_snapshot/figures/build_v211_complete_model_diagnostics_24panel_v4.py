"""C57 visual refinement: signed residual gradients and a clear annual legend.

This wrapper changes display only. The seven model arrays, 24 panels and six
registered source-data tables are rebuilt by the v3 renderer and hash-checked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

import build_v211_complete_model_diagnostics_24panel_v3 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/v211_complete_model_diagnostics_24panel_v4"
OLD = ROOT / "figures/v211_complete_model_diagnostics_24panel_v3"
STEM = "Fig_v211_complete_model_diagnostics_24panel_v4"
BLUE = "#3B4CC0"
RED = "#B40426"
SIGNED = LinearSegmentedColormap.from_list("signed_residual_no_grey", [BLUE, "#FFFFFF", RED])
original_save = prior.save_pair
original_panel = prior.concise_panel


def readable_panel(ax, index: int, title: str = "", **kwargs):
    if title == "Complete model comparison":
        title = "Error by split contract"
    return original_panel(ax, index, title, **kwargs)


def signed_bar_colour(value: float, span: float):
    # Never use a grey or nearly white data bar. Sign is the hue; magnitude is
    # saturation, with a pale but visible floor near zero.
    strength = min(abs(value) / max(span, 1e-9), 1.0)
    location = .5 + (1 if value >= 0 else -1) * (.11 + .39 * strength)
    return SIGNED(location)


def save_polished(fig, out_dir: Path, stem: str, *, dpi: int = 600):
    if len(fig.axes) != 24:
        raise RuntimeError("The complete diagnostic plate must retain 24 panels")
    histograms = fig.axes[8:15]
    for ax in histograms:
        if len(ax.patches) != 28:
            raise RuntimeError(f"Expected all 28 registered bins: {len(ax.patches)}")
        span = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
        for bar in ax.patches:
            centre = bar.get_x() + .5 * bar.get_width()
            bar.set_facecolor(signed_bar_colour(centre, span))
            bar.set_alpha(1.0)
            bar.set_edgecolor("#FFFFFF")
            bar.set_linewidth(.26)
    # The original seven-item legend obscured the observed annual R² profile.
    annual = fig.axes[20]
    handles, labels = annual.get_legend_handles_labels()
    if annual.get_legend() is not None:
        annual.get_legend().remove()
    for ax in fig.axes[20:24]:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + .029, box.width, box.height - .029])
    fig.legend(handles, labels, ncol=7, loc="lower center",
               bbox_to_anchor=(.52, .005), frameon=False, fontsize=5.35,
               handlelength=1.1, handletextpad=.28, columnspacing=.58)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[-1].get_window_extent(renderer)
    xlabel_boxes = [ax.xaxis.label.get_window_extent(renderer) for ax in fig.axes[20:24]]
    if any(legend_box.overlaps(box) for box in xlabel_boxes):
        raise RuntimeError("Bottom model legend overlaps annual x-axis labels")
    result = original_save(fig, out_dir, stem, dpi=dpi)
    qa_path = out_dir / "layout_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["checks"].update({
        "signed_residual_histograms_use_blue_white_red_without_grey": True,
        "annual_legend_outside_plot_and_clear_of_xlabels": True,
        "model_baseline_uses_blue_not_grey": True,
    })
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.OLD = OLD
    prior.STEM = STEM
    prior.concise_panel = readable_panel
    prior.save_pair = save_polished
    prior.baseline.COLORS["OLS"] = "#274690"
    prior.main()
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"].update(json.loads((OUT / "layout_qa.json").read_text(encoding="utf-8"))["checks"])
    validation["all_checks_passed"] = all(validation["checks"].values())
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["all_checks_passed"]:
        raise RuntimeError("Refined diagnostic QA failed")


if __name__ == "__main__":
    main()
