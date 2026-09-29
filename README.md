# Urban environmental subjective–objective divergence and mental health

This repository accompanies our study of how measured urban conditions and
place-linked expressions of those conditions relate to neighbourhood mental
health. We aligned ten objective environmental dimensions with ten dimensions
derived from place-linked text and imagery across 8,440 tract–years in 885
Chicago census tracts (2014–2023). Their signed subjective–objective contrast
was examined alongside the original measurements in a multimodal geographical–
temporal graph neural network weighted regression (MM-GTGNNWR).

The analysis addresses three questions: how to measure the contrast on a common
tract–year scale; how its association with area-level frequent mental distress
varies across space and time; and how socioeconomic context and modelled
environmental changes shape the distribution of predicted burden. In 32
outcome-blind spatial holdouts, adding the explicit signed contrast did not
improve pooled prediction. Local coefficients, graph-relation diagnostics and
scenario responses provide complementary descriptions of the fitted model.

## Code for the study

| Research component | Where to start |
| --- | --- |
| MM-GTGNNWR architecture and transparent local-coefficient head | [`src/mm_gtgnnwr/model_contract.py`](src/mm_gtgnnwr/model_contract.py) |
| Quantitative visualizations, including spatial predictions, signed coefficients, relation diagnostics and socioeconomic analyses | [`figure_snapshot/figures/`](figure_snapshot/figures/) |
| Earlier registered quantitative-figure suite | [`figures/`](figures/) and [`reproducibility/`](reproducibility/) |
| Figure-source provenance and external-data registry | [`data_manifest/`](data_manifest/) |

The quantitative figure code includes the multiyear coefficient maps and
signed-sensitivity panels used in the current analysis. Figure scripts and their
source-data requirements are described in the [reproducibility guide](REPRODUCIBILITY.md).

## Getting started

The package requires Python 3.11 or newer. Install the public code and run its
tests with:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The optional PyTorch model check requires `.[model,test]`. Figure dependencies
and platform-specific locked environments are listed in the
[reproducibility guide](REPRODUCIBILITY.md). The tests check the released code;
they do not rerun the study's spatial folds.

## Data and reproducibility

The figure renderers require derived source tables and Chicago tract geometry.
The public repository provides code and a [source-data registry](data_manifest/README.md)
but does not distribute these inputs or the complete training runner. See the
[Data Availability](docs/DATA_AVAILABILITY.md) and
[Code Availability](docs/CODE_AVAILABILITY.md) statements for the release scope
and access information. Place-linked text and imagery are analysed as
tract–year expressions; scenario outputs are model-projected mitigation
potential.

## Citation and licence

Please use [`CITATION.cff`](CITATION.cff) when citing the code. The original code
is released under the [MIT licence](LICENSE); bundled fonts retain their
[separate terms](THIRD_PARTY_NOTICES.md).
