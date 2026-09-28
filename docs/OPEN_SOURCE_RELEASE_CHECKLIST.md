# Open-source release checklist

Current evidence and blockers are recorded in
`docs/RELEASE_READINESS_AUDIT_20260910.md`. Checkboxes remain intentionally
unchecked until the corresponding public-tag condition is actually satisfied.

This checklist is the release gate for the Nature-level public repository.

## Scientific integrity

- [ ] Every numerical claim maps to a registered source table and validation file.
- [ ] V211 outputs remain labelled `proxy_analysis=true`, `formal=false`.
- [ ] Platform-derived subjective features are described as place-level proxies,
      not resident self-report.
- [ ] Scenario outputs are called model-projected responses, not causal policy
      effects or realised benefits.
- [ ] Transit is recorded as unavailable; no synthetic zero graph is created.

## Data and licensing

- [ ] No restricted raw media, credentials, resident-level records or
      non-redistributable weights are committed.
- [ ] Each external source has a licence/access note.
- [x] All 46 current figure source tables have metadata-only paths, byte counts,
      SHA-256 hashes and an explicit pending-disclosure status.
- [x] The Nimbus Sans files match an official upstream commit; hashes,
      provenance, licence, exception and full licence text are included.
- [ ] A separate Code Availability statement and Data Availability statement
      are included in the manuscript and repository release.
- [x] The original code licence is MIT; separate third-party font terms remain
      in `THIRD_PARTY_NOTICES.md` and `docs/LICENSING.md`.

## Reproducibility

- [x] Python version, direct locks and separate hash-complete Windows/Linux
      CPython 3.12 figure locks are recorded and resolve under pip's
      platform-specific dry-run checks.
- [ ] Random seeds, fold registry hash and source-table SHA-256 hashes are
      recorded.
- [ ] Figure renderers emit one JPG, one editable SVG and `validation.json`.
- [ ] Clean-environment smoke tests pass on Windows and Linux where relevant.
- [ ] The built wheel passes `scripts/check_wheel_contents.py` from a clean
      checkout.
- [ ] Hardware/CUDA and deterministic-operation caveats are documented.

## Release hygiene

- [x] GitHub Actions dependencies are pinned to immutable official tag SHAs.
- [ ] `git diff --check` is clean.
- [ ] CI test and lint jobs pass.
- [ ] Legacy research modules are either linted or explicitly documented as
      deferred lint debt before the first public tag.
- [ ] A tagged release points to the exact manuscript evidence snapshot.
- [ ] DOI and repository URL are added to `CITATION.cff` only after they exist.
