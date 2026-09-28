"""Use one ordered blue or red colour family within each SES facet."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from matplotlib.colors import to_rgb

import build_v211_inequality_flagship_refined_v6 as prior
from nature_viz_common import sha256


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "figures/v211_inequality_flagship_refined_v6"
OUT = ROOT / "figures/v211_inequality_flagship_refined_v7"


def one_hue_per_facet(values: np.ndarray):
    if len(values) != 8 or not np.isfinite(values).all():
        raise RuntimeError("SES facet must contain eight finite registered values")
    dominant = float(np.median(values))
    target = np.asarray(to_rgb("#B40426" if dominant > 0 else "#2443A8"))
    return [tuple(1 - (.43 + .54 * row / 7) * (1 - target))
            for row in range(8)]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.scheme_ordered_colours = one_hue_per_facet
    prior.main()
    for path in OLD.glob("source_*.csv"):
        if sha256(path) != sha256(OUT / path.name):
            raise RuntimeError(f"Fig. 11 source changed: {path.name}")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "visual_revision": "v7_single_hue_per_SES_facet_with_ordered_luminance",
        "gradient_scale": (
            "eight scheme positions set light-to-dark tint within each facet; "
            "facet hue follows median sign, while horizontal position shows every signed value"
        ),
        "source_tables_byte_identical_to_v6": True,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
