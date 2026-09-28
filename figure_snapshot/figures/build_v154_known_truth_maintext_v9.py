"""Move summary metric names into titles so no text crowds the first map row."""
from __future__ import annotations

import json

import build_v154_known_truth_maintext_v8 as prior
from nature_viz_common import ROOT, sha256


OUT = ROOT / "figures/v154_known_truth_maintext_v9"
OLD = ROOT / "figures/v154_known_truth_maintext_v8"
ORIGINAL_SAVE = prior.ORIGINAL_SAVE


def title_metric_then_save(fig):
    if len(fig.axes) != 20:
        raise RuntimeError("Expected the complete V154 20-panel plate")
    for ax, title in zip(fig.axes[:2], (
        "Spatial-extrapolation prediction · test R²",
        "Local-coefficient recovery · mean RMSE",
    )):
        ax.set_xlabel("")
        ax.set_title(title, loc="left", pad=2.8, fontweight="normal")
    return ORIGINAL_SAVE(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior.OUT, prior.OLD = OUT, OLD
    prior.clear_top_labels_then_save = title_metric_then_save
    prior.main()
    for extension in ("jpg", "svg"):
        (OUT / f"Fig_v154_known_truth_maintext_v8.{extension}").rename(
            OUT / f"Fig_v154_known_truth_maintext_v9.{extension}")
    if sha256(OLD / "source_model_metric_intervals.csv") != sha256(
        OUT / "source_model_metric_intervals.csv"):
        raise RuntimeError("V154 sealed interval data changed")
    for name in ("validation.json", "manifest.json"):
        path = OUT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["figure"] = "Fig_v154_known_truth_maintext_v9"
        for extension in ("jpg", "svg"):
            data["delivery"][extension] = str(data["delivery"][extension]).replace("_v8.", "_v9.")
        data["v9_visual_change"] = "Summary metric names are integrated into panel titles; no summary x-label can overlap map-row beta headings."
        data["source_interval_sha256_equal_v8"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
