# Figure and table source data

This directory contains derived numerical data from the study, not raw Google
Maps, Flickr or Street View text/images. The empirical files are from the V211
R3 compact proxy (`proxy_analysis=true`, `formal=false`). Figure 6 is a separate
V154 sealed synthetic validation; its values must not be mixed with Chicago
empirical coefficients.

`manifest_v211.json` lists every released file, column name, row count where
applicable, byte size and SHA-256. The 54 figure files comprise 51 exports
registered against the current Fig. 2–13 images, plus the Fig. 2 spatial
prediction table and the two synthetic inputs needed to rebuild Fig. 6.
`tables/output/` contains Table 1 and Supplementary Tables S1–S13;
`tables/inputs/` contains ten additional inputs used by the table code,
including the V211 semantic-source support completion record used to compute
Supplementary Table S1's coverage values. The Table 1 display rows and
reviewed editorial wording in selected supplementary tables are frozen in
`manuscript_table_display_contract.json`; numerical result cells remain
computed from the registered result tables.

## Reproduce the released data and tables

From the repository root, with the figure dependencies installed:

```bash
python scripts/verify_source_data.py
python scripts/rebuild_study_tables.py --output-dir /tmp/rebuilt-study-tables
python -m pytest -q tests/test_study_source_data.py
```

The test rebuilds all 14 table CSVs from the released inputs and compares them
byte-for-byte with `tables/output/`. Tables S4–S10 summarise frozen model
outputs; the test does not rerun the underlying fits. Tables 1, S1–S3 and
S11–S13 include registered descriptive definitions in the table-building code.
The separate V154 Fig. 6 can be regenerated with
`python scripts/rebuild_figure06.py`; its 20-panel and numeric-source checks
run in the test suite. Rendering may differ by font/backend across systems,
so the scientific check compares the numeric source file rather than JPEG bytes.

## Limits of this release

The file set makes the plotted numerical exports inspectable but does not yet
make every Fig. 2–13 renderer self-contained. In particular, some map panels
also need annual Chicago tract geometry; several visual-refinement scripts
read earlier figure outputs or upstream R3 result tables that are not included
here. Only Fig. 6 and the 14 table CSVs currently have a verified clean-input
rebuild route in this package. A source-data hash check is not a retraining or
independent scientific validation. Full V211 model retraining still requires
authorised platform inputs, registered folds, checkpoints and the complete
training environment.

These are area-level and model-derived data, not resident self-reports or
individual diagnoses. Scenario values describe model-projected mitigation
potential, not causal or realised policy effects. The authors' original rights
in these released data are licensed under [CC BY 4.0](LICENSE.md); underlying
third-party rights remain with their respective providers. The repository's
MIT licence applies to software, not these source-data files.
