"""Re-run the maintained quantitative V211 figure families in a fixed order.

The script is deliberately an orchestration layer: all plotting logic stays in
the auditable renderer scripts under ``figures/``. It never invokes the
Image-2 framework workflow.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Resolve the workspace from this file so the runner works after cloning the
# repository on Windows, macOS, or Linux.  The repo is two levels below the
# workspace root: ``reproducibility/v211_nature_figures_20260906``.
ROOT = Path(__file__).resolve().parents[2]
CURRENT_V211_RENDERERS = [
    ROOT / "figures/build_v211_signed_delta_sensitivity_12panel.py",
    ROOT / "figures/build_v211_explainability_flagship_20panel_v2.py",
    ROOT / "figures/build_v211_explainability_composite_20panel.py",
    ROOT / "figures/build_v211_full_coefficient_atlas_20panel.py",
    ROOT / "figures/build_v211_local_coefficient_map_20panel.py",
    ROOT / "figures/build_v211_spatiotemporal_20panel.py",
    ROOT / "figures/build_v211_scenario_inequality_20panel.py",
    ROOT / "figures/build_v211_scenario_flagship_15panel.py",
    ROOT / "figures/build_v211_inequality_flagship_15panel.py",
    ROOT / "figures/build_v211_modality_ablation_20panel.py",
    ROOT / "figures/build_v211_training_ablation_20panel.py",
    ROOT / "figures/build_v211_graph_perturbation_nature_16panel.py",
    ROOT / "figures/build_v211_graph_perturbation_spatiotemporal_20panel.py",
    ROOT / "figures/build_v211_model_perturbation_moderation_20panel.py",
    ROOT / "figures/build_v211_feature_coefficient_channel_atlases.py",
    ROOT / "figures/build_v211_planning_strategy_20panel.py",
]
REFERENCE_ONLY_RENDERERS = [
    ROOT / "figures/build_v211_nature_master_rebuild.py",
    ROOT / "figures/build_v154_known_truth_recovery_suite.py",
    ROOT / "figures/build_v211_model_benchmark_diagnostics_18panel.py",
    ROOT / "figures/build_v211_matched_ablation_stability_20panel.py",
    ROOT / "figures/build_v211_five_model_spatiotemporal_25panel.py",
    ROOT / "figures/build_v211_training_ablation_recovery_perturbation_atlas.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print renderers without running them")
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="also list/run historical or mixed reference renderers",
    )
    parser.add_argument(
        "--start-at",
        metavar="FILENAME",
        help="resume at this renderer filename (for example build_v211_modality_ablation_20panel.py)",
    )
    args = parser.parse_args()
    renderers = list(CURRENT_V211_RENDERERS)
    if args.include_reference:
        renderers.extend(REFERENCE_ONLY_RENDERERS)
    if args.start_at:
        names = [path.name for path in renderers]
        if args.start_at not in names:
            raise ValueError(f"Unknown --start-at renderer: {args.start_at}")
        renderers = renderers[names.index(args.start_at):]
    missing = [str(path) for path in renderers if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing renderer(s): " + "; ".join(missing))
    print("CURRENT_V211_RENDERERS")
    current_to_run = [path for path in renderers if path in CURRENT_V211_RENDERERS]
    for path in current_to_run:
        print(path)
        if not args.dry_run:
            subprocess.run([sys.executable, str(path)], cwd=str(path.parent), check=True)
    if args.include_reference:
        print("REFERENCE_ONLY_RENDERERS")
        reference_to_run = [path for path in renderers if path in REFERENCE_ONLY_RENDERERS]
        for path in reference_to_run:
            print(path)
            if not args.dry_run:
                subprocess.run([sys.executable, str(path)], cwd=str(path.parent), check=True)


if __name__ == "__main__":
    main()
