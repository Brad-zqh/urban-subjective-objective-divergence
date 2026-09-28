"""Increase top V154 interval-panel height without changing sealed evidence."""
from __future__ import annotations

import json

import build_v154_known_truth_maintext_v4 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v154_known_truth_maintext_v6"
OLD = ROOT / "figures/v154_known_truth_maintext_v5"
ORIGINAL_FIGURE = prior._figure


def balanced_figure(*args, **kwargs):
    kwargs["figsize"] = (205 / 25.4, 275 / 25.4)
    fig = ORIGINAL_FIGURE(*args, **kwargs)
    add = fig.add_gridspec

    def balanced_grid(*gargs, **gkwargs):
        gkwargs.update(left=.100, right=.960, bottom=.028, top=.975,
                       wspace=.46, hspace=.30,
                       height_ratios=[1.14, 1, 1, 1, 1, 1, 1])
        return add(*gargs, **gkwargs)

    fig.add_gridspec = balanced_grid
    return fig


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT = OUT
    prior.wide_figure = balanced_figure
    prior.main()
    for extension in ("jpg", "svg"):
        old = OUT / f"Fig_v154_known_truth_maintext_v4.{extension}"
        old.rename(OUT / f"Fig_v154_known_truth_maintext_v6.{extension}")
    if sha256(OLD / "source_model_metric_intervals.csv") != sha256(
        OUT / "source_model_metric_intervals.csv"
    ):
        raise RuntimeError("V154 sealed interval source changed")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["figure"] = "Fig_v154_known_truth_maintext_v6"
        for extension in ("jpg", "svg"):
            data["delivery"][extension] = str(data["delivery"][extension]).replace(
                "_v4.", "_v6."
            )
        data["v6_visual_change"] = (
            "Top interval-row height ratio 0.68 to 1.14; 205×275 mm canvas; "
            "eighteen sealed coefficient fields unchanged"
        )
        data["source_interval_sha256_equal_v5"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
