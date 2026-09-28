"""Contract tests for direct and platform-resolved figure locks."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ]+)")


def pins(path: Path) -> dict[str, str]:
    """Read exact package pins from one requirements file."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            result[match.group(1).lower()] = match.group(2)
    return result


def test_platform_full_locks_contain_direct_pins_and_hashes() -> None:
    direct_path = ROOT / "requirements-figures.lock.txt"
    direct = pins(direct_path)
    assert len(direct) == 8
    for filename in (
        "requirements-figures-win-py312.lock.txt",
        "requirements-figures-linux-py312.lock.txt",
    ):
        full_path = ROOT / filename
        full = pins(full_path)
        assert len(full) == 20
        assert all(full[name] == version for name, version in direct.items())
        package_lines = [
            line.strip()
            for line in full_path.read_text(encoding="utf-8").splitlines()
            if PIN.match(line.strip())
        ]
        assert all("--hash=sha256:" in line for line in package_lines)
