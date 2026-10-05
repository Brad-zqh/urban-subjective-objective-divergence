# Reproducibility contract

This document is the release-facing companion to the V211 Geo Mental Health
analysis. It separates a figure-only audit from a complete retraining run.

## Supported public run

The current visualization source closure contains 73 Python files in
`figure_snapshot/figures/`, including the revised signed-sensitivity and
multiyear coefficient map scripts. The known-truth simulation uses separate
V154 code. Selected figure source data are included under `source_data/`;
Chicago geometry remains in the authorised research workspace.

Inspect the earlier quantitative-suite registry separately:

```powershell
python reproducibility/v211_nature_figures_20260906/run_flagship_figures.py --dry-run
```

Selected derived source tables are now included under `source_data/`. The
complete 18-figure legacy runner remains unsupported on a clean clone because
its external inputs and geometry have not all been staged. Do not interpret a
source-data hash pass as a successful render of every current manuscript panel.

Each maintained family must finish with:

- one 600-dpi JPG;
- one editable SVG (`svg.fonttype=none`);
- the registered source tables unchanged;
- a `validation.json` with panel, source, geometry, colourbar and export checks.

Manuscript drafts, figure placement audits and editorial working files are
maintained in the authors' private workspace.

The clean-clone release includes selected derived figure source data, all
14 current table CSVs, and their table-generation inputs. The separate V154
known-truth Fig. 6 is rebuildable from its released synthetic inputs. Other
figures may still need Chicago geometry, upstream V211 result tables or earlier
visual revision outputs. Full current-figure rendering remains blocked until
those dependencies and their access terms are recorded.

`data_manifest/v211_external_figure_source_tables.json` registers all 46
historical `source_*` tables across the earlier 18-figure suite. Its entries
remain external. They are a different snapshot from the current manuscript's
Fig. 2–13 source files under `source_data/`; the two registries must not be
merged or treated as interchangeable numerical versions.

For the released source-data and table checks, run:

```bash
python scripts/verify_source_data.py
python scripts/rebuild_study_tables.py --output-dir /tmp/rebuilt-study-tables
python scripts/rebuild_figure06.py
```

The test suite checks that the rebuilt Table 1 and S1–S13 CSVs are byte-identical
to their released counterparts, and that the V154 Fig. 6 has 20 panels and an
unchanged numeric interval source. It does not rerun V211 training or validate
all main-figure renderers.

The same release manifest also freezes the byte-identical PyTorch architecture
contract used by the V211 R3 run. Its original `V207TransparentMMGTGNNWR` class
name is retained as provenance of code reuse; it does not make the V211 result
a V207 numerical run. This contract alone is insufficient for retraining
without the audited runner, fold/input registries and authorised inputs.

The current visualization code is published under `figure_snapshot/`
separately from that earlier suite. Complete clean-clone numerical rendering
requires external input tables, geometry and historical QA baselines; the
earlier suite's 46 source-table registry does not describe every input needed
by the newer visualizations.

## Figure environment

The eight direct figure dependencies are frozen in
`requirements-figures.lock.txt`. Separate complete CPython 3.12 graphs are
recorded for Windows x86-64 and Linux x86-64 in
`requirements-figures-win-py312.lock.txt` and
`requirements-figures-linux-py312.lock.txt`. Each platform file pins 20 direct
and transitive packages to the selected wheel SHA-256 hashes. They pass pip's
hash-enforcing platform resolution checks. The Windows environment was
exercised locally and the Linux lock executed in GitHub Actions.

These figure locks do not describe the restricted CUDA training environment.
Do not substitute one platform lock for the other or infer retraining
reproducibility from a renderer dependency check.

## Source-data policy

The public release may include derived tract-year tables, schemas, checksums,
synthetic smoke inputs and code. It must not include raw platform imagery or
text, private credentials, resident-level records or third-party weights whose
licence forbids redistribution. Access instructions must be written for each
restricted input instead of silently replacing it with an unrelated source.

## Scientific guardrails

- V211 remains `proxy_analysis=true`, `formal=false`.
- Platform-derived subjective features are place-level proxies, not resident
  self-report.
- Same-year spatial holdout is not prospective forecasting.
- Local coefficients, additive contributions and full-function derivatives are
  separate quantities.
- Graph perturbation is frozen-model message-path sensitivity, not a transport
  or causal effect.
- Scenario output is model-projected mitigation potential, not an implemented
  policy benefit.

## Full retraining

Full retraining requires authorised access to every input source, the exact
fold registry, environment lock and model checkpoints. A figure-only run does
not imply that restricted-data retraining is portable or complete. Before a
public tag, record the commit, environment, seed list, fold hash, source hashes,
hardware and licence review.
