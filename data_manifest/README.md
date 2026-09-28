# Data manifest and access boundary

The release candidate contains current-only renderer code, the byte-identical
PyTorch architecture contract used by V211 R3, hashes and reproducibility
instructions. Derived figure source tables remain external until disclosure
review. It does **not** redistribute Google Maps, Flickr, Street View or
resident-level records, private credentials, or non-redistributable model
weights.

The Nimbus Sans figure fonts are separate third-party assets. Their official
upstream commit, SHA-256 hashes, AGPL-3.0 licence, font/document embedding
exception and full licence text are recorded under
`assets/fonts/nimbus-sans/` and in the release manifest.

## Reproduction tiers

1. **Tier 0:** rerun maintained quantitative figure renderers from registered
   derived tables and inspect `validation.json`.
2. **Tier 1:** run the synthetic/smoke contracts on a clean installation.
3. **Tier 2:** rerun shareable aggregation and evaluation from legally derived
   tables.
4. **Tier 3:** full retraining requires authorised access to restricted inputs;
   it is never implied by this public package.

Every released table must carry a SHA-256 hash, provenance note, licence/access
status and a matching validation file. Missing restricted inputs must fail
closed rather than being silently replaced by zeros.

The metadata-only registry
`v211_external_figure_source_tables.json` covers 46 current `source_*` tables
for all 18 figures. It contains paths, sizes and SHA-256 hashes but no table
content. All entries remain `pending_disclosure_and_licence_review` and
`included_in_release=false` until a rights/data review changes them explicitly.
