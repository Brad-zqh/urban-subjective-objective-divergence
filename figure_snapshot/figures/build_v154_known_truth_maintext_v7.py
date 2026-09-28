"""Align V154 summary charts more closely with the three-column map plate."""
from __future__ import annotations

import json

import build_v154_known_truth_maintext_v6 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v154_known_truth_maintext_v7"
OLD = ROOT / "figures/v154_known_truth_maintext_v6"
BASE = prior.prior.prior.base
ORIGINAL_SAVE = BASE.save_jpg_svg


def narrow_summary_save(fig):
    if len(fig.axes) != 20:
        raise RuntimeError("V154 atlas must retain 20 evidence axes")
    widths = []
    for ax in fig.axes[:2]:
        box = ax.get_position()
        new_width = .75 * box.width
        ax.set_position([box.x0 + .5 * (box.width - new_width),
                         box.y0, new_width, box.height])
        widths.append(ax.get_position().width / box.width)
    if any(abs(value - .75) > 1e-9 for value in widths):
        raise RuntimeError("Top summary axes failed 0.75 width ratio")
    return ORIGINAL_SAVE(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    BASE.save_jpg_svg = narrow_summary_save
    prior.main()
    for extension in ("jpg", "svg"):
        (OUT / f"Fig_v154_known_truth_maintext_v6.{extension}").rename(
            OUT / f"Fig_v154_known_truth_maintext_v7.{extension}"
        )
    if sha256(OLD / "source_model_metric_intervals.csv") != sha256(
        OUT / "source_model_metric_intervals.csv"
    ):
        raise RuntimeError("Sealed V154 interval source changed")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["figure"] = "Fig_v154_known_truth_maintext_v7"
        for extension in ("jpg", "svg"):
            data["delivery"][extension] = str(data["delivery"][extension]).replace(
                "_v6.", "_v7."
            )
        data["v7_visual_change"] = (
            "Top a,b width 75% of v6, centred within original slots; "
            "all eighteen sealed coefficient matrices unchanged"
        )
        data["source_interval_sha256_equal_v6"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
