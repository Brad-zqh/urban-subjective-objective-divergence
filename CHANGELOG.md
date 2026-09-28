# Changelog

## Unreleased — V211 release candidate

- Vendor the 18-file current-only renderer snapshot, including the signed
  `S−O` matched-sensitivity figure.
- Vendor the exact PyTorch architecture contract used by V211 R3, with a
  frozen upstream digest and optional CPU/GPU smoke tests.
- Separate historical V118/V154 renderers from the public evidence manifest.
- Add fail-closed data/secret and wheel-content checks, a Linux CI definition,
  direct dependency locks and Windows/Linux clean-clone verification
  requirements.
- Add separate fully hash-locked 20-package figure environments for Windows
  and Linux CPython 3.12; actual remote Linux execution remains a release gate.
- Pin GitHub Actions dependencies to immutable SHAs resolved from official tag
  refs.
- Record `proxy_analysis=true`, `formal=false` and the restricted-data limits.

No public release, DOI or semantic version tag has been issued.
