# V211 public-release readiness audit — 2026-09-10

Last updated: 2026-09-28. This remains an audit of the first code-only release,
not a certification of full manuscript reproduction.

## Current outcome

The original code now has an MIT `LICENSE` and the Nimbus Sans third-party
licence remains separate. The candidate supports bounded code and manifest
reproduction. The complete R3 training snapshot and reviewed source-data
package remain blocking items for full manuscript reproduction.

## Passed checks

- The vendored code manifest contains 18 files: 16 current-only renderers,
  `nature_viz_common.py` and the fixed-order runner.
- The signed `S−O` matched-sensitivity renderer is present.
- No V118, V154, benchmark, five-model or matched-ablation renderer appears in
  the public manifest.
- Manifest hashes match all 18 vendored files.
- Public-candidate tests pass locally (12 tests, including dependency-lock,
  registry portability,
  frozen-digest and wheel-content contracts); the optional PyTorch module is
  explicitly skipped in the figure-only Python environment.
- The byte-identical architecture contract used by V211 R3 is vendored under
  `src/mm_gtgnnwr/model_contract.py`. The staged and workspace source hashes
  both equal `30772daf98f31a65544858673900ca472c6ffefb95b96158c5e315387771a67c`.
  The public smoke tests pass 3/3 and the original source contract tests pass
  8/8 in the project GPU environment (Python 3.10.18,
  PyTorch 2.12.1+cu130). These tests execute CPU tensors; PyTorch-enabled
  Python 3.11/3.12 CI remains pending.
- The four Nimbus Sans OTF files match Artifex Software's official
  `urw-base35-fonts` commit `3c0ba3b5687632dfc66526544a4e811fe0ec0cd9`
  byte-for-byte. Their hashes, AGPL-3.0 licence, font/document embedding
  exception, full licence text and third-party notice are included separately
  from the pending project code licence.
- A metadata-only registry covers all 46 `source_*` tables used by the 18
  current figures. It records relative intended paths, byte counts and SHA-256
  hashes, passes its structural contract, and keeps every table marked
  `included_in_release=false` pending table-level review.
- Ruff passes for release/boundary tooling and its tests. Vendored scientific
  renderers are snapshot code and are not auto-formatted after figure QA.
- The release-boundary check rejects raw/restricted paths, common credential
  files, private-key suffixes and model checkpoints.
- A Python wheel builds successfully from a fresh temporary source tree and
  passes an archive-content gate. The gate rejects the historical top-level
  `src` package and licence-decision memos packaged as licence files.
- Direct figure and development dependencies are pinned to the locally tested
  Python 3.12.10 environment. Separate Windows and Linux CPython 3.12 figure
  locks each contain 20 direct and transitive packages, each pinned with the
  correct platform-wheel SHA-256 hash; pip's hash-enforcing Windows and
  Linux-target dry-runs both resolve successfully.
- GitHub Actions is configured for Python 3.11 and 3.12 with read-only default
  permissions, pip caching, timeouts, tests, lint, boundary checks and manifest
  drift checks.
- `actions/checkout` v4 and `actions/setup-python` v5 are pinned to the commit
  SHAs resolved directly from their official Git tag refs, rather than floating
  major-version tags.
- A fresh 63-file Windows tree assembled from the actual Git candidate
  allowlist passes tests, Ruff, boundary and manifest checks, external registry
  validation, 16-renderer dry-run, compilation, absolute-path scanning, wheel
  build and wheel-content validation. Its 63 source files are byte-identical to
  the current candidate. This is not a claim that remote GitHub Actions or an
  independently installed Windows dependency environment has run.

## Blocking before a full manuscript-reproduction release

1. **Complete current execution closure:** the architecture contract is now
   vendored, but the R3 runner, input builders, V208-named post-processing
   engines and validators still require a path/data/provenance refactor before
   they can be presented as a portable V211 chain. Historical direct `src/*.py`
   modules remain excluded. See `docs/V211_EXECUTION_CLOSURE_AUDIT_20260910.md`.
2. **Source Data:** decide which derived figure tables can be redistributed,
   change only approved entries in the 46-table registry, stage those tables at
   their registered paths, and provide access instructions for the rest.
3. **Clean-clone verification:** run the configured CI on GitHub/Linux and an
   independently resolved Windows environment. The clean Windows file-tree
   simulation passed locally, but remote CI and independent dependency
   resolution have not run.
4. **Metadata:** verify named author identities/ORCIDs before adding them;
   deposit an archival DOI only after a citable release exists. The software
   citation record names the research team and does not invent a DOI.
5. **Runtime and training dependency closure:** execute the resolved Linux
   figure lock in remote CI and the Windows lock in an independently created
   environment; document the separate restricted CUDA-training environment
   and determinism caveats.

The first code-only release deliberately does not claim these five full
reproduction gates are closed. In particular, the latest manuscript Fig. 4
and Fig. 8 renderers remain outside the public 18-file snapshot.

## Scientific release boundary

Every public artifact must retain `proxy_analysis=true` and `formal=false`.
Platform-derived expression is not resident self-report, spatial holdout is not
prospective forecasting, graph perturbation is not a network effect, and
scenario response is not an implemented policy benefit. Transit remains
unavailable and must not be represented by a synthetic zero graph.
