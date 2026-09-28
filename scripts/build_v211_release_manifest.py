"""Build a deterministic manifest for the V211 public-release handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

RENDERERS = (
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
MODEL_CONTRACT = "src/mm_gtgnnwr/model_contract.py"
MODEL_CONTRACT_SOURCE = "workstreams/RQ2_model/engineering_v207/v207_model_contract.py"
MODEL_CONTRACT_SOURCE_SHA256 = "30772daf98f31a65544858673900ca472c6ffefb95b96158c5e315387771a67c"
FONT_ASSETS = (
    "assets/fonts/nimbus-sans/NimbusSans-Bold.otf",
    "assets/fonts/nimbus-sans/NimbusSans-BoldItalic.otf",
    "assets/fonts/nimbus-sans/NimbusSans-Italic.otf",
    "assets/fonts/nimbus-sans/NimbusSans-Regular.otf",
    "assets/fonts/nimbus-sans/LICENSE",
    "assets/fonts/nimbus-sans/COPYING",
    "assets/fonts/nimbus-sans/SOURCE.md",
)
FROZEN_FONT_SHA256 = {
    "assets/fonts/nimbus-sans/NimbusSans-Bold.otf": "7f33328e6b4d4cd21b45fa625791928c9407dc702db6780e56b09ca9a3ecaa67",
    "assets/fonts/nimbus-sans/NimbusSans-BoldItalic.otf": "3f47fb34fcb7de09f8cbc9f305191340ddebf7a068419f4bb5f49287dea59b87",
    "assets/fonts/nimbus-sans/NimbusSans-Italic.otf": "7b0bef5686aa58c0fd0f0d01beeae56664208490e30ca6a25431281c9a0c6402",
    "assets/fonts/nimbus-sans/NimbusSans-Regular.otf": "7c25be4d78155523080ab85b10277150657ff7dabbcad7037bdd536c9b6d0d08",
    "assets/fonts/nimbus-sans/LICENSE": "b0b192fda6c754b02dd4346f6638dac9c2f84a249d4ed424510788daad6e147b",
    "assets/fonts/nimbus-sans/COPYING": "282751b8c98ee9e445346eb57a992c9ecbe25ed8dd554df046777313e19b10f9",
}
EXTERNAL_SOURCE_REGISTRY = "data_manifest/v211_external_figure_source_tables.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, object]:
    """Build a manifest and fail closed when a maintained renderer is absent."""
    entries: list[dict[str, object]] = []
    missing: list[str] = []
    for relative in RENDERERS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "restricted_data_embedded": False,
            }
        )
    if missing:
        raise FileNotFoundError("Missing maintained renderer(s): " + ", ".join(missing))
    model_contract_path = root / MODEL_CONTRACT
    if not model_contract_path.is_file():
        raise FileNotFoundError(f"Missing V211 model contract: {model_contract_path}")
    model_contract_sha256 = sha256_file(model_contract_path)
    if model_contract_sha256 != MODEL_CONTRACT_SOURCE_SHA256:
        raise RuntimeError("Vendored V211 model contract does not match the frozen source digest")
    font_entries: list[dict[str, object]] = []
    for relative in FONT_ASSETS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing licensed figure font asset: {path}")
        digest = sha256_file(path)
        expected = FROZEN_FONT_SHA256.get(relative)
        if expected is not None and digest != expected:
            raise RuntimeError(f"Licensed figure font asset drifted: {relative}")
        font_entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    external_registry_path = root / EXTERNAL_SOURCE_REGISTRY
    if not external_registry_path.is_file():
        raise FileNotFoundError(f"Missing external source-table registry: {external_registry_path}")
    external_registry = json.loads(external_registry_path.read_text(encoding="utf-8"))
    if not (
        external_registry.get("figures_registered") == 18
        and external_registry.get("tables_registered") == 46
        and external_registry.get("source_tables_external") is True
    ):
        raise RuntimeError("External source-table registry contract failed")
    return {
        "manifest_schema": "v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analysis_version": "V211",
        "proxy_analysis": True,
        "formal": False,
        "vendor_snapshot": True,
        "source_tables_external": True,
        "restricted_media_embedded": False,
        "model_contract": {
            "path": MODEL_CONTRACT,
            "bytes": model_contract_path.stat().st_size,
            "sha256": model_contract_sha256,
            "workspace_source": MODEL_CONTRACT_SOURCE,
            "workspace_source_sha256": MODEL_CONTRACT_SOURCE_SHA256,
            "source_name_retained_for_provenance": "V207TransparentMMGTGNNWR",
            "used_by_current_v211_r3": True,
        },
        "font_assets": {
            "upstream": "https://github.com/ArtifexSoftware/urw-base35-fonts",
            "upstream_commit": "3c0ba3b5687632dfc66526544a4e811fe0ec0cd9",
            "licence": "AGPL-3.0 with upstream font/document embedding exception",
            "files": font_entries,
        },
        "external_source_registry": {
            "path": EXTERNAL_SOURCE_REGISTRY,
            "bytes": external_registry_path.stat().st_size,
            "sha256": sha256_file(external_registry_path),
            "figures_registered": 18,
            "tables_registered": 46,
            "table_content_included": False,
        },
        "renderers": entries,
        "notes": [
            "This manifest records the current-only renderer code vendored into the release candidate.",
            "The model contract is an exact snapshot of the architecture class used by current V211 R3.",
            "The complete restricted-data training and post-processing closure is not yet vendored.",
            "Nimbus Sans files are unmodified third-party assets with separate upstream licence files.",
            "The external Source Data registry records 46 table hashes but includes no table content.",
            "Figure renderers must not be interpreted as full restricted-data retraining.",
            "Derived figure source tables remain external until licence and disclosure review is complete.",
        ],
    }


def main() -> int:
    """Write a JSON manifest from a workspace root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_manifest" / "v211_renderer_manifest.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing manifest against vendored renderer files without rewriting it",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.root.resolve())
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {args.output}")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        comparable_keys = (
            "manifest_schema",
            "analysis_version",
            "proxy_analysis",
            "formal",
            "vendor_snapshot",
            "source_tables_external",
            "restricted_media_embedded",
            "model_contract",
            "font_assets",
            "external_source_registry",
            "renderers",
            "notes",
        )
        mismatched = [key for key in comparable_keys if existing.get(key) != manifest.get(key)]
        if mismatched:
            raise RuntimeError("Release manifest is stale for keys: " + ", ".join(mismatched))
        print(f"Manifest matches {len(manifest['renderers'])} vendored files: {args.output.resolve()}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()} ({len(manifest['renderers'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
