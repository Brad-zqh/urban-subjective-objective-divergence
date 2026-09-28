# Reproducibility contract

This document is the release-facing companion to the V211 Geo Mental Health
analysis. It separates a figure-only audit from a complete retraining run.

## Supported public run

The current visualization source closure contains 72 Python files in
`figure_snapshot/figures/`, including the revised signed-sensitivity and
multiyear coefficient map scripts. The known-truth simulation uses separate
V154 code. Figure source data and Chicago geometry remain in the authorised
research workspace.

Inspect the earlier quantitative-suite registry separately:

```powershell
python reproducibility/v211_nature_figures_20260906/run_flagship_figures.py --dry-run
```

The full render command becomes supported only after reviewed derived source
tables are staged under the documented repository-relative paths.

Each maintained family must finish with:

- one 600-dpi JPG;
- one editable SVG (`svg.fonttype=none`);
- the registered source tables unchanged;
- a `validation.json` with panel, source, geometry, colourbar and export checks.

Manuscript drafts, figure placement audits and editorial working files are
maintained in the authors' private workspace.

The clean-clone code release currently vendors renderer code and its
hash-verified fonts, but not the derived source tables or Chicago geometry.
Therefore `--dry-run`, manifest checks and smoke tests are public-tier
operations; complete figure rendering remains blocked until table-level
disclosure/access and geometry packaging are recorded.

`data_manifest/v211_external_figure_source_tables.json` registers all 46
current `source_*` tables across the 18 figures by repository-relative intended
path, byte count and SHA-256. Every entry remains
`pending_disclosure_and_licence_review` and `included_in_release=false`; the
registry is evidence for table-by-table review, not permission to redistribute.

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
