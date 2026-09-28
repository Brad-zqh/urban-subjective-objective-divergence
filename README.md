# Urban environmental subjective–objective divergence and mental health

Python code for the MM-GTGNNWR model architecture and urban-environment
visualizations used in the study.

> **Scientific boundary.** Current V211 outputs are `proxy_analysis=true` and
> `formal=false`. Platform-derived features are tract-year place proxies, not
> resident self-report. Frozen-model sensitivity is not a causal effect, and
> scenario panels report model-projected responses rather than realised policy
> benefits. Restricted Google Maps, Flickr and Street View media are not
> redistributed.

## What this repository contains

The initial public code release contains:

- an importable package carrying the V211 proxy/formal boundary and the exact
  PyTorch architecture contract used by the current R3 run;
- current-only renderer code, release validation and boundary tooling;
- the current-only figure renderers, handoff metadata and release manifests;
- hash-verified Nimbus Sans OTF files with their upstream AGPL-3.0 licence,
  font/document embedding exception and full licence text;
- smoke/contract tests and CI configuration;
- data-access, code-availability and release-gate documentation.

Historical direct modules remain in the local parent workspace for provenance,
but `src/*.py` is excluded from the first public snapshot. They are neither the
current V211 implementation nor part of the release wheel.

It does not contain restricted raw media, private credentials, resident-level
records or non-redistributable model weights.

The architecture class is vendored byte-for-byte from its original V207-named
source because that same implementation was bound to the V211 inputs. The
complete R3 training runner, input builders, V208-named post-processing engines
and validators have not yet been made portable as one audited closure. Until
that code and its dependency/data contracts are staged and reviewed, this
release supports code/renderer manifests and smoke reproduction tiers, not
clean-clone full retraining.

The current visualization source closure contains 72 Python files under
`figure_snapshot/figures/`, including the multiyear coefficient maps and
signed-sensitivity visualizations. The known-truth simulation uses a separate
V154 source family. An earlier 18-file V211 renderer registry is retained for
its own outputs. Numerical rerendering requires the source data identified in
the [reproducibility guide](REPRODUCIBILITY.md).

The implementation-aligned technical route is maintained in the workspace
handoff at `workstreams/RQ2_model/TECHNICAL_ROUTE_MM_GTGNNWR_CODE_PATH_V1.md`;
the experiment-to-figure map is
`workstreams/RQ2_model/EXPERIMENT_PLAN_MM_GTGNNWR_V211_V1.md`.

## Install and test

Requires Python 3.11 or newer. The release-candidate figure environment used
Python 3.12.10. Exact direct-dependency versions are recorded in
`requirements-figures.lock.txt` and `requirements-dev.lock.txt`; the complete
20-package Windows CPython 3.12 figure environment, including transitive
dependencies and distribution hashes, is frozen in
`requirements-figures-win-py312.lock.txt`. The corresponding target-resolved
manylinux/universal wheel set is frozen separately in
`requirements-figures-linux-py312.lock.txt`; its hashes pass pip's Linux-target
resolution check and have also executed successfully in GitHub/Linux CI.

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/check_release_boundary.py
```

On Windows with CPython 3.12, the hash-locked figure environment can be
validated or installed with:

```powershell
python -m pip install --dry-run --ignore-installed --require-hashes -r requirements-figures-win-py312.lock.txt
```

The CI definition installs the separate Linux CPython 3.12 lock before running
the public-candidate suite. Platform lockfiles must not be interchanged.

For linting, install `python -m pip install -e ".[lint]"`. CI lints the release
and boundary tools plus their tests, then builds and inspects a wheel from the
clean checkout. Vendored renderer snapshots are not auto-formatted after
figure QA because formatting would change their registered hashes.

The wheel boundary can be checked explicitly:

```powershell
python -m pip wheel --no-deps --wheel-dir dist-ci .
python scripts/check_wheel_contents.py dist-ci
```

The checker rejects historical top-level packages, unsafe archive paths and a
licence-decision memo accidentally packaged as if it were a licence.

## Reproduction tiers

The project separates what can be reproduced from public artefacts:

1. **Tier 0 — figure audit:** inspect the 72-file visualization source closure
   and verify the separate earlier 18-file renderer manifest.
   Full figure rendering additionally requires the reviewed derived source
   tables; those are not yet included in the clean-clone candidate.
2. **Tier 1 — smoke contracts:** run synthetic preprocessing, graph and
   perturbation checks in a clean environment. The optional architecture test
   needs the separately declared PyTorch model extra; a skipped test is not a
   pass.
3. **Tier 2 — derived-data reproduction:** rerun legally shareable aggregation
   and evaluation tables.
4. **Tier 3 — full retraining:** requires authorised access to restricted
   inputs, exact fold registry and model weights; a public clone does not imply
   this capability.

The current handoff manifest is
[`data_manifest/v211_renderer_manifest.json`](data_manifest/v211_renderer_manifest.json).
It records the 18 vendored current-only renderer files plus the frozen model
contract, licensed font assets and the metadata-only registry of 46 external
figure source tables. `source_tables_external` remains true until those tables
pass disclosure and licence review.

## Figure handoff

The 72-file visualization source closure is under `figure_snapshot/figures/`.
The following earlier quantitative-suite runner is retained for its own
registered outputs:

```powershell
python reproducibility/v211_nature_figures_20260906/run_flagship_figures.py --dry-run
python reproducibility/v211_nature_figures_20260906/run_flagship_figures.py
```

The runner resolves the workspace root relative to its own file and executes a
fixed renderer order. The 16 current-only renderers produce 18 figures and 338
panels. Each figure has synchronized JPG/PDF/PNG/SVG/TIFF exports, 600-dpi
rasters, editable SVG text, source tables and a machine-readable validation
file without mutating the registered source tables. The framework diagram is maintained separately
and is intentionally excluded from this quantitative runner.

The exact current-only renderer snapshot is vendored. Before a public tag,
derived source tables must be reviewed and either staged with hashes or listed
as restricted/external with access instructions.

The staging command is safe by default:

```powershell
python scripts/stage_v211_renderer_snapshot.py
python scripts/stage_v211_renderer_snapshot.py --apply
```

Use `--apply` only after reviewing the dry-run paths. The command copies
current-only renderer code and never copies source tables, raw media or model
checkpoints.

Maintainers can similarly refresh the frozen architecture contract from the
parent workspace. The command fails closed if the audited source digest has
changed:

```powershell
python scripts/stage_v211_model_contract.py --source-root ..
python scripts/stage_v211_model_contract.py --source-root .. --apply
```

Install `.[model,test]` to run the optional PyTorch contract tests. Environments
without PyTorch report that module as skipped instead of pretending it ran.

## Data and code availability

- [Data manifest and access boundary](data_manifest/README.md)
- [External figure Source Data registry](data_manifest/v211_external_figure_source_tables.json)
- [Code Availability statement](docs/CODE_AVAILABILITY.md)
- [Data Availability statement](docs/DATA_AVAILABILITY.md)
- [Open-source release checklist](docs/OPEN_SOURCE_RELEASE_CHECKLIST.md)
- [V211 execution-closure audit](docs/V211_EXECUTION_CLOSURE_AUDIT_20260910.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Licensing and rights boundary](docs/LICENSING.md)
- [Reproducibility contract](REPRODUCIBILITY.md)

The original code is MIT-licensed; the bundled Nimbus Sans fonts retain their
separate AGPL-3.0 license and embedding exception. External data and media
are not covered by the code license.

## Historical engineering notes

Earlier MVP experiments used CDC PLACES/MHLTH, ACS covariates, Chicago
administrative event proxies and Queen adjacency to test the data and GPU
pipeline. Those exploratory results are retained in `STABLE_MODEL_V3_RESULTS.md`
and related reports for traceability. They are not substituted for the V211
compact proxy analysis and should not be cited as its final evidence.
