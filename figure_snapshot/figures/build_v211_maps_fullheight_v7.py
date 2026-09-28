"""Full-height colourbar and black-text map contract for V211 map figures.

All quantitative inputs and map geometries come from the audited aligned-v5
renderer.  This non-destructive wrapper enforces one display rule: each map's
colourbar has exactly the map axes height.  Titles, explanatory header text,
tick labels and sample-size annotations are black.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

FIGURES = Path(__file__).resolve().parent
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import build_v211_nature_aligned_composites_v5 as base  # noqa: E402
from nature_viz_common import ROOT  # noqa: E402

OUT_ROOT = ROOT / "figures/v211_map_fullheight_v7"


def main() -> None:
    base.OUT_ROOT = OUT_ROOT
    base.MUTED = base.INK

    original_setup = base.setup_style

    def black_large_setup(_: float) -> None:
        original_setup(7.0)
        mpl.rcParams.update({
            "text.color": base.INK,
            "axes.labelcolor": base.INK,
            "axes.titlecolor": base.INK,
            "xtick.color": base.INK,
            "ytick.color": base.INK,
        })

    base.setup_style = black_large_setup

    original_figure = base.plt.figure

    def compact_figure(*args, **kwargs):
        requested = kwargs.get("figsize")
        is_fig8 = bool(requested and abs(float(requested[1]) - 115 / 25.4) < .02)
        if is_fig8:
            kwargs["figsize"] = (183 / 25.4, 102 / 25.4)
        fig = original_figure(*args, **kwargs)
        if is_fig8:
            original_add_gridspec = fig.add_gridspec

            def compact_add_gridspec(*grid_args, **grid_kwargs):
                grid_kwargs.update(
                    left=.080,
                    right=.925,
                    bottom=.115,
                    top=.925,
                    height_ratios=[1.28, 1.0],
                    hspace=.105,
                )
                return original_add_gridspec(*grid_args, **grid_kwargs)

            fig.add_gridspec = compact_add_gridspec
        return fig

    base.plt.figure = compact_figure

    original_map_cell = base.map_cell

    def protected_map_cell(*args, **kwargs):
        # Full-height bars need the panel letter inside the map so row-terminal
        # tick labels cannot collide with the next row.  Pull n left so an
        # inward terminal colourbar never covers it.
        kwargs["letter_inside"] = True
        ax, cax = original_map_cell(*args, **kwargs)
        for artist in ax.texts:
            if artist.get_text().startswith("n="):
                artist.set_x(.820)
                artist.set_color(base.INK)
        return ax, cax

    base.map_cell = protected_map_cell

    original_centre = base.centre_colourbars_on_maps

    def fullheight_colourbars(
        map_axes,
        cbar_axes,
        fraction=.70,
        final_inset=0.0,
        terminal_every=None,
    ):
        return original_centre(
            map_axes,
            cbar_axes,
            fraction=1.0,
            final_inset=final_inset,
            terminal_every=terminal_every,
        )

    base.centre_colourbars_on_maps = fullheight_colourbars

    original_finish = base.finish

    def fullheight_finish(fig, folder, stem, panels, sources, core, chain, alignment):
        new_stem = (
            "Fig2_spatial_prediction_fullheight_v7"
            if stem.startswith("Fig2_")
            else "Fig8_coefficient_heterogeneity_fullheight_v7"
        )
        out = original_finish(fig, folder, new_stem, panels, sources, core, chain, alignment)
        validation_path = out.parent / "validation.json"
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        payload["layout_version"] = "fullheight_colourbar_v7_black_text_non_destructive"
        payload["map_text_colour"] = "#17191C"
        payload["checks"].pop("short_centred_colourbars", None)
        payload["checks"]["colourbars_match_map_height"] = bool(
            abs(float(payload["alignment"]["mean_colourbar_to_map_height"]) - 1.0) < 1e-9
            and float(payload["alignment"]["colourbar_to_map_height_range"]) < 1e-9
        )
        payload["checks"]["map_titles_headers_ticks_and_n_black_by_construction"] = True
        validation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out.parent / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out.parent / "figure_contract.md").write_text(
            "# Full-height map colourbar contract\n\n"
            f"- Core conclusion: {core}\n"
            f"- Evidence chain: {chain}\n"
            "- Geometry: every colourbar matches the actual equal-aspect map axes height exactly.\n"
            "- Typography: titles, header text, ticks and `n=` labels are black; only panel letters are bold.\n"
            "- Export: JPG (600 dpi) and editable-text SVG only.\n"
            "- Preservation: aligned-v5/v6 and earlier exports remain untouched.\n",
            encoding="utf-8",
        )
        return out

    base.finish = fullheight_finish
    output_fig2 = base.build_fig2()
    output_fig8 = base.build_fig8()

    root_manifest = {
        "layout_version": "fullheight_colourbar_v7_black_text_non_destructive",
        "preserves_existing_versions": True,
        "delivery_formats": ["jpg", "svg"],
        "map_colourbar_rule": "colourbar height equals actual map axes height",
        "map_text_rule": "black",
        "outputs": [str(output_fig2), str(output_fig8)],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "aligned_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
