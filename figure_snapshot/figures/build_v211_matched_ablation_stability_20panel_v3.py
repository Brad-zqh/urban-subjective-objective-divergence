"""A wider, portrait-safe source canvas for the V211 matched-ablation plate.

This remains the compatibility layer used by the v4/v5 registered renderers.
Its historical v3 output is never overwritten by downstream versions.
"""
from __future__ import annotations

import json

import build_v211_matched_ablation_stability_20panel_v2 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v211_matched_ablation_stability_20panel_v3"
OLD = ROOT / "figures/v211_matched_ablation_stability_20panel_v2"
STEM = "Fig_v211_matched_ablation_stability_20panel_v3"


def wider_figure(*args, **kwargs):
    kwargs["figsize"] = (190 / 25.4, 245 / 25.4)
    fig = prior.original_figure(*args, **kwargs)
    original_add = fig.add_gridspec

    def wider_grid(*gargs, **gkwargs):
        gkwargs["hspace"] = .42
        gkwargs["wspace"] = .48
        return original_add(*gargs, **gkwargs)

    fig.add_gridspec = wider_grid
    return fig


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.STEM = OUT, STEM
    prior.tighter_figure = wider_figure
    prior.main()
    equality = {name: sha256(OLD / name) == sha256(OUT / name)
                for name in prior.SOURCE_FILES}
    if not all(equality.values()):
        raise RuntimeError(f"Matched-ablation source tables changed: {equality}")
    validation_path = OUT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["source_tables_byte_identical_to_prior"] = equality
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
