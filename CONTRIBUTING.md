# Contributing

Contributions should preserve the scientific and release boundaries of the
V211 proxy analysis.

1. Create a focused branch and keep changes small enough to review.
2. Do not commit raw platform media, account-level records, credentials,
   restricted model weights or local absolute paths.
3. Add or update tests for every change to data contracts, graph relations,
   spatial folds, metrics or figure manifests.
4. Run `python -m pytest -q`, the same focused Ruff command declared in
   `.github/workflows/ci.yml`, `python scripts/check_release_boundary.py`, and
   `python scripts/build_v211_release_manifest.py --check`. From a clean
   checkout, also build a wheel and run
   `python scripts/check_wheel_contents.py <wheel-or-directory>`.
5. State whether the change affects `proxy_analysis`, `formal`, source hashes,
   fold roles, figure source data or manuscript claims.

Bug reports should include the operating system, Python version, command,
minimal traceback and whether restricted inputs were available. Never attach
restricted data to a public issue.
