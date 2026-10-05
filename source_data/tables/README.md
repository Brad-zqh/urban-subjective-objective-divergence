# Study table sources

Run `python scripts/rebuild_study_tables.py --output-dir <empty-directory>` from
the repository root. The test suite compares all 14 regenerated CSV files with
`output/` byte-for-byte. The public script produces table data; it does not
generate or release the manuscript Word files.

| Displayed table | Released inputs used by the table builder |
| --- | --- |
| Main Table 1 | `inputs/manuscript_table_display_contract.json`; the ten-dimension definitions are checked against the V211 measurement contract. |
| Supplementary S1 | `inputs/completion.json` and `inputs/semantic_proxy_completion.json`; text/GSV union, intersection and fallback counts are computed from the registered support totals. |
| Supplementary S2–S3 | Versioned dimension and processing definitions in the builder; reviewed display wording in the display contract. |
| Supplementary S4 | `inputs/source_model_metrics.csv`. |
| Supplementary S5 | `../fig04/source_panel_statistics.csv`. |
| Supplementary S6 | `inputs/source_modality_panel_statistics.csv`. |
| Supplementary S7 | `inputs/source_partition_panel_statistics.csv`. |
| Supplementary S8 | `inputs/feature_local_coefficient_summary.csv`; ranks and sign agreement are recomputed. |
| Supplementary S9 | `inputs/source_graph_perturbation_summaries.csv`. |
| Supplementary S10 | `../fig12/source_scenario_predictions.csv`; quantiles and fractions are recomputed from the individual model-prediction differences. |
| Supplementary S11 | `inputs/completion.json` and versioned architecture definitions in the builder. |
| Supplementary S12 | `inputs/compact_53_feature_dictionary.csv`, `inputs/TEN_DIMENSION_COMPONENT_REGISTRY_V51.csv` and reviewed display wording. |
| Supplementary S13 | Versioned measurement notation in the builder and reviewed display wording. |

The display contract affects descriptive text only. It cannot override numeric
result fields in S4–S10. Tables S4–S10 reproduce published summaries from
frozen V211 results; they do not reproduce model training from raw inputs.
