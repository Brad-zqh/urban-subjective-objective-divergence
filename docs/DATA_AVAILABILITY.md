# Data Availability

The [study code and data repository](https://github.com/Brad-zqh/urban-subjective-objective-divergence)
contains derived numerical source-data exports for selected
panels in Figs. 2–13, the inputs and outputs used to rebuild Table 1 and
Supplementary Tables S1–S13, and a separate synthetic dataset for Fig. 6.
File-level paths, schemas, row counts and SHA-256 hashes are in
`source_data/manifest_v211.json`. The authors' original rights in the released
derived data are licensed under CC BY 4.0; underlying third-party rights are
not relicensed. The earlier 46-table registry in
`data_manifest/` remains metadata-only and describes a different figure suite.
Raw Google Maps,
Flickr and Street View media, private credentials, resident-level records and
non-redistributable model weights are not included. The released numerical
tables do not by themselves support full V211 retraining or a clean-clone
re-render of every current figure; annual tract geometry and upstream result
tables remain external for some panels.

For every restricted input, a future full-reproduction release must provide the source name,
access date, licence/terms note, temporal and spatial coverage, preprocessing
contract, and a checksum for the derived table used by the figure renderer.
Unavailable inputs will fail closed; they will not be replaced with zeros or
unrelated sources.

The current V211 artefacts are a proxy analysis of tract-year records. They do
not constitute a formal resident-perception dataset or a causal intervention
dataset.
