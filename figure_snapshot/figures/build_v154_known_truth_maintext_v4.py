"""Wider 20-panel V154 simulation atlas with original sealed data."""
from __future__ import annotations

import json
from pathlib import Path

import build_v154_known_truth_maintext_v3 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v154_known_truth_maintext_v4"
OLD = ROOT / "figures/v154_known_truth_maintext_v3"
_figure = prior.base.plt.figure


def wide_figure(*args, **kwargs):
    kwargs["figsize"] = (205 / 25.4, 260 / 25.4)
    fig = _figure(*args, **kwargs)
    add = fig.add_gridspec

    def compact_grid(*gargs, **gkwargs):
        gkwargs.update(left=.100, right=.960, bottom=.028, top=.975,
                       wspace=.46, hspace=.28)
        return add(*gargs, **gkwargs)

    fig.add_gridspec = compact_grid
    return fig


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.base.plt.figure = wide_figure
    prior.main()
    for extension in ("jpg", "svg"):
        old = OUT / f"Fig_v154_known_truth_maintext_v3.{extension}"
        old.rename(OUT / f"Fig_v154_known_truth_maintext_v4.{extension}")
    if sha256(OLD / "source_model_metric_intervals.csv") != sha256(OUT / "source_model_metric_intervals.csv"):
        raise RuntimeError("V154 source intervals changed")
    validation_path = OUT / "validation.json"
    data = json.loads(validation_path.read_text(encoding="utf-8"))
    if not all(data["checks"].values()):
        raise RuntimeError(f"V154 layout checks failed: {data['checks']}")
    data["figure"] = "Fig_v154_known_truth_maintext_v4"
    for extension in ("jpg", "svg"):
        data["delivery"][extension] = str(data["delivery"][extension]).replace("_v3.", "_v4.")
    data["v4_visual_change"] = "205×260 mm canvas; wider map columns and tighter gutters"
    data["source_interval_sha256_equal_v3"] = True
    validation_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figure"] = data["figure"]
    for extension in ("jpg", "svg"):
        manifest["delivery"][extension] = data["delivery"][extension]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
