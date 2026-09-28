"""Keep staging, manifest and runtime renderer registries synchronized."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    """Load one repository module without importing the legacy package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_registries_match_exactly() -> None:
    staging = load_module("stage_v211", ROOT / "scripts" / "stage_v211_renderer_snapshot.py")
    manifest = load_module("manifest_v211", ROOT / "scripts" / "build_v211_release_manifest.py")
    assert tuple(staging.FILES) == tuple(manifest.RENDERERS)
    assert len(staging.FILES) == 18


def test_model_contract_digest_is_frozen_across_staging_and_manifest() -> None:
    staging = load_module("stage_model_contract", ROOT / "scripts" / "stage_v211_model_contract.py")
    manifest = load_module("manifest_model_contract", ROOT / "scripts" / "build_v211_release_manifest.py")
    destination = ROOT / staging.DESTINATION
    assert staging.EXPECTED_SOURCE_SHA256 == manifest.MODEL_CONTRACT_SOURCE_SHA256
    assert staging.sha256_file(destination) == staging.EXPECTED_SOURCE_SHA256


def test_runtime_registry_contains_only_current_renderers() -> None:
    runner = load_module(
        "run_v211_figures",
        ROOT / "reproducibility" / "v211_nature_figures_20260906" / "run_flagship_figures.py",
    )
    current = {path.relative_to(ROOT).as_posix() for path in runner.CURRENT_V211_RENDERERS}
    staged = set(load_module(
        "stage_v211_runtime",
        ROOT / "scripts" / "stage_v211_renderer_snapshot.py",
    ).FILES)
    assert len(current) == 16
    assert current.issubset(staged)
    assert not any(
        token in path.lower()
        for path in current
        for token in ("v118", "v154", "benchmark", "five_model", "matched_ablation")
    )


def test_vendored_renderers_use_repository_relative_assets() -> None:
    staging = load_module("stage_v211_paths", ROOT / "scripts" / "stage_v211_renderer_snapshot.py")
    for relative in staging.FILES:
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "MM-GTGNNWR/assets/" not in content
        assert "MM-GTGNNWR/outputs/" not in content
