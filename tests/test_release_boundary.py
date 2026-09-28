from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_release_boundary.py"
SPEC = importlib.util.spec_from_file_location("check_release_boundary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
find_forbidden_paths = MODULE.find_forbidden_paths


def test_release_boundary_ignores_normal_source(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("pass\n", encoding="utf-8")
    assert find_forbidden_paths(tmp_path) == []


def test_release_boundary_detects_weights_and_raw_media(tmp_path: Path) -> None:
    (tmp_path / "weights.pt").write_bytes(b"x")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "image.tif").write_bytes(b"x")
    findings = find_forbidden_paths(tmp_path)
    assert {path.name for path in findings} == {"weights.pt", "image.tif"}


def test_release_boundary_detects_common_secret_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=do-not-commit\n", encoding="utf-8")
    (tmp_path / "private.pem").write_text("not-a-real-key\n", encoding="utf-8")
    findings = find_forbidden_paths(tmp_path)
    assert {path.name for path in findings} == {".env", "private.pem"}


def test_renderer_manifest_declares_external_source_tables() -> None:
    import json

    manifest_path = Path(__file__).resolve().parents[1] / "data_manifest" / "v211_renderer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis_version"] == "V211"
    assert manifest["proxy_analysis"] is True
    assert manifest["formal"] is False
    assert manifest["vendor_snapshot"] is True
    assert manifest["model_contract"]["path"] == "src/mm_gtgnnwr/model_contract.py"
    assert manifest["model_contract"]["used_by_current_v211_r3"] is True
    assert len(manifest["font_assets"]["files"]) == 7
    assert manifest["font_assets"]["upstream_commit"] == "3c0ba3b5687632dfc66526544a4e811fe0ec0cd9"
    assert manifest["external_source_registry"]["tables_registered"] == 46
    assert manifest["external_source_registry"]["table_content_included"] is False
    assert len(manifest["renderers"]) == 18
    paths = {entry["path"] for entry in manifest["renderers"]}
    assert "figures/build_v211_signed_delta_sensitivity_12panel.py" in paths
    assert not any("v118" in path.lower() or "v154" in path.lower() for path in paths)
