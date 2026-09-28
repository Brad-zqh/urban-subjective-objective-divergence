"""Increase within-facet top-to-bottom contrast without changing SES values."""
from __future__ import annotations

import json

import numpy as np
from matplotlib.colors import to_rgb

import build_v211_inequality_flagship_refined_v7 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v211_inequality_flagship_refined_v8"
OLD = ROOT / "figures/v211_inequality_flagship_refined_v7"


def stronger_single_hue(values: np.ndarray):
    if len(values) != 8 or not np.isfinite(values).all():
        raise RuntimeError("SES facet must contain eight finite registered values")
    target = np.asarray(to_rgb("#B40426" if float(np.median(values)) > 0 else "#2443A8"))
    return [tuple(1 - (.32 + .66 * row / 7) * (1 - target))
            for row in range(8)]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.one_hue_per_facet = stronger_single_hue
    prior.main()
    for path in OLD.glob("source_*.csv"):
        if sha256(path) != sha256(OUT / path.name):
            raise RuntimeError(f"Fig. 11 source changed: {path.name}")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "visual_revision": "v8_stronger_top_to_bottom_single_hue_SES_facets",
        "gradient_strength_top_to_bottom": [.32, .98],
        "source_tables_byte_identical_to_v7": True,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
