"""Stage the exact architecture contract used by the V211 R3 proxy run."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

SOURCE = Path("workstreams/RQ2_model/engineering_v207/v207_model_contract.py")
DESTINATION = Path("src/mm_gtgnnwr/model_contract.py")
EXPECTED_SOURCE_SHA256 = "30772daf98f31a65544858673900ca472c6ffefb95b96158c5e315387771a67c"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(source_root: Path, destination_root: Path, apply: bool) -> Path:
    """Copy the model contract byte-for-byte, or report its destination."""
    source = source_root / SOURCE
    destination = destination_root / DESTINATION
    if not source.is_file():
        raise FileNotFoundError(f"Missing V211 model-contract source: {source}")
    source_sha256 = sha256_file(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "V211 model-contract source drifted; audit and update the frozen digest before staging: "
            f"{source_sha256}"
        )
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different")
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("Staged V211 model contract is not byte-identical to its source")
    return destination


def main() -> int:
    """Run the maintainer-only model-contract staging command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--destination-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="copy the file; default is dry-run")
    args = parser.parse_args()
    destination = stage(args.source_root.resolve(), args.destination_root.resolve(), args.apply)
    mode = "staged" if args.apply else "dry-run"
    print(f"{mode}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
