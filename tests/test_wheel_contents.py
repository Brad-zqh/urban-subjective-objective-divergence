"""Tests for the fail-closed wheel-content release gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    """Load the wheel checker without importing the legacy source tree."""
    path = ROOT / "scripts" / "check_wheel_contents.py"
    spec = importlib.util.spec_from_file_location("check_wheel_contents", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wheel(path: Path, members: tuple[str, ...]) -> None:
    """Write a minimal ZIP-form wheel fixture."""
    with ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "fixture\n")


def test_current_package_wheel_passes(tmp_path: Path) -> None:
    checker = load_checker()
    wheel = tmp_path / "mm_gtgnnwr-0.1.0-py3-none-any.whl"
    write_wheel(
        wheel,
        (
            "mm_gtgnnwr/__init__.py",
            "mm_gtgnnwr-0.1.0.dist-info/METADATA",
            "mm_gtgnnwr-0.1.0.dist-info/RECORD",
        ),
    )
    assert checker.inspect_wheel(wheel) == []


def test_legacy_src_package_is_rejected(tmp_path: Path) -> None:
    checker = load_checker()
    wheel = tmp_path / "mm_gtgnnwr-0.1.0-py3-none-any.whl"
    write_wheel(
        wheel,
        (
            "mm_gtgnnwr/__init__.py",
            "src/stable_models.py",
            "mm_gtgnnwr-0.1.0.dist-info/METADATA",
        ),
    )
    assert any("unexpected top-level" in item for item in checker.inspect_wheel(wheel))


def test_licence_decision_memo_is_rejected(tmp_path: Path) -> None:
    checker = load_checker()
    wheel = tmp_path / "mm_gtgnnwr-0.1.0-py3-none-any.whl"
    write_wheel(
        wheel,
        (
            "mm_gtgnnwr/__init__.py",
            "mm_gtgnnwr-0.1.0.dist-info/licenses/LICENSE_SELECTION.md",
            "mm_gtgnnwr-0.1.0.dist-info/METADATA",
        ),
    )
    findings = checker.inspect_wheel(wheel)
    assert any("decision memo" in item for item in findings)
