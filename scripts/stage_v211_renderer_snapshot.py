"""Stage the maintained V211 renderer snapshot into a release checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FILES = (
    "figures/build_v211_signed_delta_sensitivity_12panel.py",
    "figures/build_v211_explainability_flagship_20panel_v2.py",
    "figures/build_v211_explainability_composite_20panel.py",
    "figures/build_v211_full_coefficient_atlas_20panel.py",
    "figures/build_v211_local_coefficient_map_20panel.py",
    "figures/build_v211_spatiotemporal_20panel.py",
    "figures/build_v211_scenario_inequality_20panel.py",
    "figures/build_v211_scenario_flagship_15panel.py",
    "figures/build_v211_inequality_flagship_15panel.py",
    "figures/build_v211_modality_ablation_20panel.py",
    "figures/build_v211_training_ablation_20panel.py",
    "figures/build_v211_graph_perturbation_nature_16panel.py",
    "figures/build_v211_graph_perturbation_spatiotemporal_20panel.py",
    "figures/build_v211_model_perturbation_moderation_20panel.py",
    "figures/build_v211_feature_coefficient_channel_atlases.py",
    "figures/build_v211_planning_strategy_20panel.py",
    "figures/nature_viz_common.py",
    "reproducibility/v211_nature_figures_20260906/run_flagship_figures.py",
)

RELEASE_PATH_REWRITES = (
    ("ROOT/'MM-GTGNNWR/assets/", "ROOT/'assets/"),
    ("ROOT / 'MM-GTGNNWR/assets/", "ROOT / 'assets/"),
    ('ROOT / "MM-GTGNNWR/assets/', 'ROOT / "assets/'),
    ("ROOT/'MM-GTGNNWR/outputs/", "ROOT/'outputs/"),
    ("ROOT / 'MM-GTGNNWR/outputs/", "ROOT / 'outputs/"),
    ('ROOT / "MM-GTGNNWR/outputs/', 'ROOT / "outputs/'),
)


def make_release_relative(path: Path) -> None:
    """Rewrite monorepo-only asset paths in one staged Python source file."""
    content = path.read_text(encoding="utf-8")
    rewritten = content
    for old, new in RELEASE_PATH_REWRITES:
        rewritten = rewritten.replace(old, new)
    if rewritten != content:
        path.write_text(rewritten, encoding="utf-8", newline="\n")


def stage(source_root: Path, destination_root: Path, apply: bool) -> list[Path]:
    """Copy the fixed renderer list, or report what would be copied."""
    staged: list[Path] = []
    for relative in FILES:
        source = source_root / relative
        destination = destination_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing renderer source: {source}")
        if source.resolve() == destination.resolve():
            raise ValueError("Source and destination roots must be different")
        staged.append(destination)
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            make_release_relative(destination)
    return staged


def main() -> int:
    """Run the safe renderer staging command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--destination-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="copy files; without this flag the command is a dry-run",
    )
    args = parser.parse_args()
    staged = stage(args.source_root.resolve(), args.destination_root.resolve(), args.apply)
    mode = "staged" if args.apply else "dry-run"
    print(f"{mode}: {len(staged)} renderer files")
    for path in staged:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
