# Data Availability

The code repository releases metadata-only provenance notes and SHA-256 hashes
for 46 external figure source tables; it does not release their contents.
Derived schemas and legally shareable tables may be added after table-level
disclosure and licence review. Raw Google Maps,
Flickr and Street View media, private credentials, resident-level records and
non-redistributable model weights are not included.

For every restricted input, a future full-reproduction release must provide the source name,
access date, licence/terms note, temporal and spatial coverage, preprocessing
contract, and a checksum for the derived table used by the figure renderer.
Unavailable inputs will fail closed; they will not be replaced with zeros or
unrelated sources.

The current V211 artefacts are a proxy analysis of tract-year records. They do
not constitute a formal resident-perception dataset or a causal intervention
dataset.
