# Data manifest and access boundary

The release candidate contains current-only renderer code, the byte-identical
PyTorch architecture contract used by V211 R3, hashes and reproducibility
instructions. The 46-table registry below describes an older figure-suite
snapshot and remains external. Selected current manuscript source data and
tables are documented separately under `source_data/`. It does **not**
redistribute Google Maps, Flickr or Street View media, resident-level records,
private credentials, or non-redistributable model
weights.

The Nimbus Sans figure fonts are separate third-party assets. Their official
upstream commit, SHA-256 hashes, AGPL-3.0 licence, font/document embedding
exception and full licence text are recorded under
`assets/fonts/nimbus-sans/` and in the release manifest.

## Reproduction tiers

1. **Tier 0:** verify released source-file hashes and rebuild Table 1/S1–S13.
2. **Tier 1:** regenerate the separate V154 Fig. 6 from synthetic inputs.
3. **Tier 2:** rerun the remaining current visualizations only after their
   additional geometry and upstream result inputs are released or obtained.
4. **Tier 3:** full V211 retraining requires authorised access to restricted inputs;
   it is never implied by this public package.

Every released file has a SHA-256 hash and schema/row metadata in the current
source-data manifest. Data-licence status remains subject to author confirmation.
Missing restricted inputs must fail
closed rather than being silently replaced by zeros.

The metadata-only registry
`v211_external_figure_source_tables.json` covers 46 historical `source_*` tables
for an earlier 18-figure suite. It contains paths, sizes and SHA-256 hashes but no table
content. All entries remain `pending_disclosure_and_licence_review` and
`included_in_release=false` until a rights/data review changes them explicitly.
