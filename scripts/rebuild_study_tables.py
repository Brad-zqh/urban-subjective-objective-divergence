"""Rebuild the study's main and supplementary table CSVs from released inputs.

This is a data-table generator; manuscript/Word layout files are not required.
The published outputs can be compared byte-for-byte using ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "source_data/tables/output"

COMPLETION = ROOT / "source_data/tables/inputs/completion.json"
SEMANTIC_COMPLETION = ROOT / "source_data/tables/inputs/semantic_proxy_completion.json"
DISPLAY_CONTRACT = ROOT / "source_data/tables/inputs/manuscript_table_display_contract.json"
MODEL_BENCHMARK = ROOT / "source_data/tables/inputs/source_model_metrics.csv"
MATCHED = ROOT / "source_data/fig04/source_panel_statistics.csv"
MODALITY = ROOT / "source_data/tables/inputs/source_modality_panel_statistics.csv"
PARTITION = ROOT / "source_data/tables/inputs/source_partition_panel_statistics.csv"
COEFFICIENT_SUMMARY = ROOT / "source_data/tables/inputs/feature_local_coefficient_summary.csv"
GRAPH_PERTURBATION = ROOT / "source_data/tables/inputs/source_graph_perturbation_summaries.csv"
SCENARIO_PREDICTIONS = ROOT / "source_data/fig12/source_scenario_predictions.csv"
OBJECTIVE_REGISTRY = ROOT / "source_data/tables/inputs/TEN_DIMENSION_COMPONENT_REGISTRY_V51.csv"
COMPACT_FEATURE_DICTIONARY = ROOT / "source_data/tables/inputs/compact_53_feature_dictionary.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pct(count: int, total: int = 8440) -> str:
    return f"{100 * count / total:.2f}%"


def f3(value: Any) -> str:
    return f"{float(value):.3f}"


def f4(value: Any) -> str:
    return f"{float(value):.4f}"


def f6(value: Any) -> str:
    return f"{float(value):.6f}"


def coefficient_label(feature: str) -> tuple[str, str]:
    dimensions = {
        1: "Green and blue space", 2: "Density and openness", 3: "Access and walkability",
        4: "Safety and order", 5: "Cleanliness and maintenance", 6: "Social vitality",
        7: "Heat stress", 8: "Air pollution", 9: "Noise", 10: "Traffic pressure",
    }
    match = re.search(r"_d(\d+)_", feature)
    if feature.startswith("o_d") and match:
        return "O", dimensions[int(match.group(1))]
    if feature.startswith("s_d") and match:
        return "S", dimensions[int(match.group(1))]
    if feature.startswith("delta_d") and match:
        return "S−O", dimensions[int(match.group(1))]
    labels = {
        "ses__age_65plus_pct": "Age ≥65", "ses__age_under18_pct": "Age <18",
        "ses__bachelors_or_higher_pct_25plus": "Higher education",
        "ses__black_alone_pct": "Black population", "ses__commute_transit_pct": "Transit commute",
        "ses__commute_walk_pct": "Walk commute", "ses__gross_rent_30plus_pct": "Rent burden",
        "ses__hispanic_latino_pct": "Hispanic/Latino", "ses__log_income_nominal": "Log income",
        "ses__male_pct": "Male population", "ses__population_log1p": "Population size",
        "ses__poverty_pct": "Poverty", "ses__unemployment_pct": "Unemployment",
    }
    return "SES", labels.get(feature, feature.replace("ses__", "").replace("_", " "))


def write_csv(table: dict[str, Any]) -> None:
    path = OUT / table["source_file"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table["columns"])
        writer.writeheader()
        writer.writerows(table["rows"])


def main() -> None:
    for path in (COMPLETION, SEMANTIC_COMPLETION, DISPLAY_CONTRACT,
                 MODEL_BENCHMARK, MATCHED, MODALITY, PARTITION,
                 COEFFICIENT_SUMMARY, GRAPH_PERTURBATION, SCENARIO_PREDICTIONS,
                 OBJECTIVE_REGISTRY, COMPACT_FEATURE_DICTIONARY):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    if completion.get("rows") != 8440 or completion.get("primary_numeric_features") != 43:
        raise ValueError("V211 compact-input contract changed; review table definitions before rebuilding")
    if completion.get("proxy_analysis") is not True or completion.get("formal") is not False:
        raise ValueError("V211 interpretation flags changed")
    semantic_completion = json.loads(SEMANTIC_COMPLETION.read_text(encoding="utf-8"))
    if semantic_completion.get("rows") != completion["rows"]:
        raise ValueError("The semantic-support denominator differs from the study panel")
    text_rows = int(semantic_completion["text_rows"])
    gsv_rows = int(semantic_completion["gsv_image_rows"])
    observed_rows = int(semantic_completion["union_observed_rows"])
    unsupported_rows = int(semantic_completion["no_text_or_gsv_image_rows"])
    both_rows = text_rows + gsv_rows - observed_rows
    if observed_rows + unsupported_rows != completion["rows"] or both_rows < 0:
        raise ValueError("The semantic-source support counts are inconsistent")

    dimensions = [
        ("D1", "Green and blue space", "natural-green fraction; tree canopy; water fraction", "little greenery/water ↔ pleasant green-blue environment"),
        ("D2", "Density and openness", "housing density; impervious cover; population density", "crowded/enclosed ↔ open/uncrowded"),
        ("D3", "Access and walkability", "intersection density; largest-component share; walkable-edge density", "difficult walking/access ↔ easy, comfortable walking/access"),
        ("D4", "Safety and order", "all reported records; interpersonal reported records", "unsafe/disorderly ↔ safe/orderly"),
        ("D5", "Cleanliness and maintenance", "311 litter-event density; median close days", "dirty/poorly maintained ↔ clean/well maintained"),
        ("D6", "Social vitality and opportunity", "opportunity density; opportunity-group Shannon diversity", "inactive/few opportunities ↔ lively/many opportunities"),
        ("D7", "Heat stress", "summer land-surface-temperature mean and 90th percentile", "hot/stuffy ↔ cool/comfortable"),
        ("D8", "Air pollution", "IDW NO2, O3 and PM2.5 concentrations", "polluted/smog/fumes ↔ clean/fresh air"),
        ("D9", "Noise", "conditional mean dBA; share at or above 55 dBA", "loud/disturbing ↔ quiet"),
        ("D10", "Traffic pressure", "directional VMT/day; regional and segment inverse speed", "congested/stressful ↔ smooth/uncongested travel"),
    ]

    objective_sources = {
        "D1": "NLCD-derived land-cover fields",
        "D2": "ACS population/housing fields and NLCD impervious cover",
        "D3": "OpenStreetMap pedestrian-network extraction",
        "D4": "reported administrative public-safety records",
        "D5": "Chicago 311 service-request records",
        "D6": "OpenStreetMap opportunity objects",
        "D7": "Landsat summer land-surface temperature",
        "D8": "EPA monitor concentrations interpolated by IDW",
        "D9": "annual environmental-noise availability interface",
        "D10": "segment ADT/VMT, segment speed and regional congestion interfaces",
    }
    supportive_dimensions = {"D1", "D3", "D6"}

    display_contract = json.loads(DISPLAY_CONTRACT.read_text(encoding="utf-8"))
    main_display = display_contract["main_table_1"]
    main_columns = main_display["columns"]
    if main_columns != ["Dimension / variables", "Objective source", "Objective measure (O)",
                        "Platform source", "Platform measure (S)", "Signed discrepancy"]:
        raise ValueError("Current manuscript Table 1 columns changed")
    if len(main_display["rows"]) != len(dimensions):
        raise ValueError("Current manuscript Table 1 display contract changed")
    for index, row in enumerate(main_display["rows"], start=1):
        if len(row) != len(main_columns) or f"O{index} / S{index} / D{index}" not in row[0]:
            raise ValueError(f"Current manuscript Table 1 dimension {index} changed")
    main_rows = [dict(zip(main_columns, row, strict=True)) for row in main_display["rows"]]

    architecture_rows = [
        {"Component": "Mental-health outcome", "Source and support": "CDC PLACES frequent mental distress; 8,440 tract-years across 885 tracts, 2014–2023", "Scale": "tract-year percentage", "Analytical role": "proxy outcome for spatial prediction and local association", "Interpretation boundary": "area-level model-based burden; not individual diagnosis"},
        {"Component": "Objective account (O)", "Source and support": "administrative, environmental, remote-sensing and network anchors across ten dimensions", "Scale": "10 adverse-oriented, within-year standardized composites", "Analytical role": "measured physical or administrative conditions", "Interpretation boundary": "coverage and measurement error differ by dimension"},
        {"Component": "Platform-expression account (S)", "Source and support": f"Google Maps text, Flickr author/comment text and GSV imagery; either text or GSV for {observed_rows:,} rows", "Scale": "10 multimodal semantic scores", "Analytical role": "place-linked environmental expression proxy", "Interpretation boundary": "platform expression is not resident self-report"},
        {"Component": "Signed discrepancy (S−O)", "Source and support": "paired, orientation-aligned O and S scores", "Scale": "10 signed standardized contrasts", "Analytical role": "tests whether explicit discrepancy adds information beyond components", "Interpretation boundary": "sign is dimension-oriented; it is not a causal exposure"},
        {"Component": "Absolute discrepancy |S−O|", "Source and support": "derived from paired signed contrasts", "Scale": "10 robustness fields", "Analytical role": "diagnostic magnitude of disagreement", "Interpretation boundary": "retained for robustness, not part of the 43-feature primary representation"},
        {"Component": "Socioeconomic context", "Source and support": "13 registered tract-year covariates in the V211 assembly", "Scale": "age, education, race/ethnicity, commuting, rent, income, sex, population, poverty and unemployment", "Analytical role": "context controls and prespecified stratification variables", "Interpretation boundary": "area context does not identify individual socioeconomic status"},
        {"Component": "Spatial evaluation", "Source and support": "4 outcome-blind partition schemes × 2 boundary vintages × 4 outer blocks", "Scale": "32 registered outer folds", "Analytical role": "transportability and sensitivity assessment", "Interpretation boundary": "empirical fold distributions are not confidence intervals"},
    ]

    source_rows = [
        {"Source or construct": "Study panel", "Modality": "linked tract-year panel", "Analytical role": "common index for all current V211 analyses", "Supported records": "8,440", "Coverage": "100%", "Missingness or boundary": "885 tracts; 2014–2023"},
        {"Source or construct": "CDC PLACES frequent mental distress", "Modality": "model-based health estimate", "Analytical role": "area-level outcome", "Supported records": "8,440", "Coverage": "100%", "Missingness or boundary": "not an individual diagnosis"},
        {"Source or construct": "At least one text source", "Modality": "Google review text or Flickr author/comment text", "Analytical role": "platform-expression evidence", "Supported records": f"{text_rows:,}", "Coverage": pct(text_rows), "Missingness or boundary": "missingness retained in support flags"},
        {"Source or construct": "Google Street View imagery", "Modality": "street-level image", "Analytical role": "visible-environment evidence", "Supported records": f"{gsv_rows:,}", "Coverage": pct(gsv_rows), "Missingness or boundary": "source support retained"},
        {"Source or construct": "Either text or GSV imagery", "Modality": "cross-modal union", "Analytical role": "observed multimodal subjective proxy", "Supported records": f"{observed_rows:,}", "Coverage": pct(observed_rows), "Missingness or boundary": "primary observed-support tier"},
        {"Source or construct": "Both text and GSV imagery", "Modality": "cross-modal intersection", "Analytical role": "high-support multimodal subset", "Supported records": f"{both_rows:,}", "Coverage": pct(both_rows), "Missingness or boundary": "not required for inclusion"},
        {"Source or construct": "No text and no GSV imagery", "Modality": "no direct platform/image support", "Analytical role": "flagged fallback tier", "Supported records": f"{unsupported_rows:,}", "Coverage": pct(unsupported_rows), "Missingness or boundary": "explicit temporal/Queen fallback; never silently treated as observed"},
        {"Source or construct": "Raw multimodal channels", "Modality": "six separate channels", "Analytical role": "retained outside compact numeric reduction", "Supported records": "source-specific", "Coverage": "source-specific", "Missingness or boundary": "Google image/text; Flickr image/author/comment text; GSV image"},
    ]

    dimension_rows = [
        {"Dimension": concept, "Construct": concept, "Objective anchors (O)": objective, "Platform-expression polarity (S)": subjective, "Signed term": "S − O; positive values indicate a more adverse platform-linked expression than the oriented objective score"}
        for d, concept, objective, subjective in dimensions
    ]

    process_rows = [
        {"Stage": "1. Indexing", "Input": "source-specific observations", "Operation": "link to tract, health-reference year and boundary vintage", "Outcome access": "No", "Output or audit": "unique tract-year keys and source-support flags"},
        {"Stage": "2. Objective harmonization", "Input": "Ten physical and administrative dimension anchors", "Operation": "orient larger values toward the adverse pole; combine within dimension", "Outcome access": "No", "Output or audit": "10 objective composites"},
        {"Stage": "3. Multimodal semantic scoring", "Input": "Google Maps text, Flickr author/comment text and GSV imagery", "Operation": "fixed adverse-versus-favourable SigLIP2 prompt contrast; within-year z-scoring; equal source and available-modality means", "Outcome access": "No", "Output or audit": "10 platform-expression scores plus source support"},
        {"Stage": "4. Missingness handling", "Input": "text and GSV availability", "Operation": "use observed evidence first; flag only 75 no-source rows for temporal/Queen fallback", "Outcome access": "No", "Output or audit": "fallback tier retained for every dimension"},
        {"Stage": "5. Feature assembly", "Input": "O, S and 13 socioeconomic covariates", "Operation": "derive 10 signed S−O terms; retain |S−O| for robustness", "Outcome access": "No", "Output or audit": "43 primary numeric features; 10 non-primary absolute deviations"},
        {"Stage": "6. Spatial registration", "Input": "tract geometry and legal relation registry", "Operation": "construct four outcome-blind spatial partitions across two boundary vintages and four outer blocks", "Outcome access": "No", "Output or audit": "32 registered outer folds"},
        {"Stage": "7. Model fitting", "Input": "V211 rows, features and legal relations", "Operation": "fit compact MM-GTGNNWR proxy within each registered fold", "Outcome access": "Training rows only", "Output or audit": "checkpoints, predictions and exact local coefficients"},
        {"Stage": "8. Sensitivity analyses", "Input": "registered folds and fitted checkpoints", "Operation": "refit matched/modality comparators within the same fold protocol; replay relation, SES and bounded scenarios on frozen models", "Outcome access": "Training rows for refitted comparators; no outcome access for frozen replays", "Output or audit": "source-data tables and empirical fold summaries"},
    ]

    feature_dictionary = pd.read_csv(COMPACT_FEATURE_DICTIONARY)
    primary_mask = feature_dictionary.primary.astype(str).str.lower().eq("true")
    primary_fields = set(feature_dictionary.loc[primary_mask, "feature"].astype(str))
    expected_primary = {
        *(f"o_d{i}_composite_z" for i in range(1, 11)),
        *(f"s_d{i}_augmented_mm_z" for i in range(1, 11)),
        *(f"delta_d{i}_augmented_signed_s_minus_o" for i in range(1, 11)),
        "ses__age_65plus_pct", "ses__age_under18_pct", "ses__bachelors_or_higher_pct_25plus",
        "ses__black_alone_pct", "ses__commute_transit_pct", "ses__commute_walk_pct",
        "ses__gross_rent_30plus_pct", "ses__hispanic_latino_pct", "ses__log_income_nominal",
        "ses__male_pct", "ses__population_log1p", "ses__poverty_pct", "ses__unemployment_pct",
    }
    if primary_fields != expected_primary:
        raise ValueError("The current V211 primary feature dictionary no longer matches the 43-variable table contract")

    variable_rows: list[dict[str, str]] = []
    for dimension, construct, objective, subjective in dimensions:
        index = int(dimension[1:])
        orientation = (
            "supportive anchors multiplied by −1; each anchor within-year z-scored; available-anchor mean"
            if dimension in supportive_dimensions
            else "each adverse-oriented anchor within-year z-scored; available-anchor mean"
        )
        variable_rows.extend([
            {
                "Family / dimension": f"Objective · {construct}",
                "Variable": construct,
                "Model field": f"o_d{index}_composite_z",
                "Source and construction": f"{objective_sources[dimension]}; anchors: {objective}; {orientation}",
                "Unit / interpretation": "dimension score; larger = more adverse objective condition",
            },
            {
                "Family / dimension": f"Platform expression · {construct}",
                "Variable": f"{construct} platform-expression proxy",
                "Model field": f"s_d{index}_augmented_mm_z",
                "Source and construction": f"Google Maps review text, Flickr author/comment text and GSV imagery; polarity: {subjective}; fixed SigLIP2 contrast, within-year z-scoring and equal available-source/modality mean",
                "Unit / interpretation": "proxy score; larger = more adverse place-linked expression; not resident self-report",
            },
            {
                "Family / dimension": f"Signed discrepancy · {construct}",
                "Variable": f"Signed {construct.lower()} discrepancy",
                "Model field": f"delta_d{index}_augmented_signed_s_minus_o",
                "Source and construction": f"derived exactly as s_d{index}_augmented_mm_z − o_d{index}_composite_z after aligned adverse orientation",
                "Unit / interpretation": "signed standardized contrast; positive = S more adverse than O",
            },
        ])

    ses_definitions = [
        ("Age ≥65", "ses__age_65plus_pct", "population aged 65 years or older", "percent"),
        ("Age <18", "ses__age_under18_pct", "population younger than 18 years", "percent"),
        ("Higher education", "ses__bachelors_or_higher_pct_25plus", "population aged ≥25 with a bachelor's degree or higher", "percent"),
        ("Black population", "ses__black_alone_pct", "population reporting Black race alone", "percent"),
        ("Transit commute", "ses__commute_transit_pct", "workers commuting by public transport", "percent"),
        ("Walk commute", "ses__commute_walk_pct", "workers commuting by walking", "percent"),
        ("Rent burden", "ses__gross_rent_30plus_pct", "renter households spending ≥30% of income on gross rent", "percent"),
        ("Hispanic/Latino", "ses__hispanic_latino_pct", "population of Hispanic or Latino ethnicity", "percent"),
        ("Log income", "ses__log_income_nominal", "positive natural log of nominal median household income (ACS B19013)", "ln(nominal USD)"),
        ("Male population", "ses__male_pct", "male population share", "percent"),
        ("Population size", "ses__population_log1p", "log1p transformation of total population", "log count"),
        ("Poverty", "ses__poverty_pct", "population below the poverty threshold", "percent"),
        ("Unemployment", "ses__unemployment_pct", "unemployed share of the civilian labour force", "percent"),
    ]
    for label, field, definition, unit in ses_definitions:
        variable_rows.append({
            "Family / dimension": "SES",
            "Variable": label,
            "Model field": field,
            "Source and construction": f"ACS-derived tract-year socioeconomic panel; {definition}; availability flag retained",
            "Unit / interpretation": f"{unit}; area-level context covariate",
        })
    if len(variable_rows) != 43 or {row["Model field"] for row in variable_rows} != expected_primary:
        raise ValueError("The complete variable dictionary must contain each current V211 primary field exactly once")

    notation_rows = [
        {"Quantity": "Within-year standardisation", "Notation": "z_t(x) = (x − μ_t) / σ_t", "Implemented operation": "mean and population SD (ddof = 0) computed within health-reference year; zero-variance groups remain unavailable", "Role / interpretation": "places heterogeneous raw anchors on a common annual scale"},
        {"Quantity": "Objective orientation", "Notation": "x*_{jkt} = a_{jk} x_{jkt}", "Implemented operation": "a = −1 for supportive D1, D3 and D6 anchors; a = +1 for adverse-oriented anchors", "Role / interpretation": "larger values consistently denote a more adverse objective condition"},
        {"Quantity": "Objective composite", "Notation": "O_kt = mean_j[z_t(x*_{jkt})]", "Implemented operation": "arithmetic mean across available registered anchors in dimension k", "Role / interpretation": "10 primary objective fields"},
        {"Quantity": "Semantic contrast", "Notation": "r = sim(e, p_adverse) − sim(e, p_favourable)", "Implemented operation": "frozen SigLIP2 joint embedding and fixed prompt pair for each dimension", "Role / interpretation": "larger raw contrast denotes more adverse platform-linked content"},
        {"Quantity": "Text proxy", "Notation": "S_text,kt = mean_source[z_t(r)]", "Implemented operation": "equal mean across available Google text and Flickr author/comment-text sources; review count does not determine source weight", "Role / interpretation": "outcome-blind platform-text component"},
        {"Quantity": "Observed multimodal proxy", "Notation": "S_obs,kt = mean(S_text,kt, S_GSV,kt)", "Implemented operation": "GSV contrast is separately within-year z-scored; equal mean of available text and GSV modalities", "Role / interpretation": "observed S support for 8,365 of 8,440 rows"},
        {"Quantity": "Flagged fallback", "Notation": "S_kt = augment(S_obs,kt)", "Implemented operation": "for 75 no-source rows only: nearest year in the same tract (same vintage preferred; past wins ties), then weighted same-year Queen-neighbour mean", "Role / interpretation": "provenance retained; never relabelled as observed"},
        {"Quantity": "Signed discrepancy", "Notation": "D_kt = S_kt − O_kt", "Implemented operation": "exact row-wise subtraction after aligned adverse orientation", "Role / interpretation": "10 primary fields; positive = platform expression more adverse than objective context"},
        {"Quantity": "Absolute discrepancy", "Notation": "Q_kt = |D_kt|", "Implemented operation": "absolute value of the signed contrast", "Role / interpretation": "10 robustness fields; excluded from the 43-feature primary representation"},
        {"Quantity": "Primary numeric representation", "Notation": "10 O + 10 S + 10 D + 13 SES", "Implemented operation": "43 fields registered in the current compact feature dictionary; model standardisation learned within training roles", "Role / interpretation": "current V211 R3 compact proxy; formal = false"},
    ]

    benchmark_rows_raw = read_csv(MODEL_BENCHMARK)
    benchmark_rows = [
        {"Model": row["model"].replace("_GPU", ""), "RMSE": f3(row["rmse"]), "MAE": f3(row["mae"]), "R²": f3(row["r2"]), "Bias": f3(row["bias"]), "Device": "GPU" if "GPU" in row["model"] else "CPU", "Outer-test n": row["n_test"]}
        for row in sorted(benchmark_rows_raw, key=lambda item: float(item["rmse"]))
        if row["model"].replace("_GPU", "") != "GTCNNWR"
    ]

    matched = read_csv(MATCHED)
    if len(matched) != 1:
        raise ValueError("Matched sensitivity table must contain one pooled summary row")
    matched = matched[0]
    matched_rows = [
        {"Representation": "Full V211 primary representation", "Features": "43", "Pooled rows": matched["rows"], "R²": f4(matched["full_r2"]), "RMSE": f4(matched["full_rmse"]), "MAE": f4(matched["full_mae"]), "Fold-level comparison": f"better in {matched['full_better_r2_folds']}/32 R², {matched['full_better_rmse_folds']}/32 RMSE and {matched['full_better_mae_folds']}/32 MAE folds"},
        {"Representation": "Matched O + S + SES comparator", "Features": "33", "Pooled rows": matched["rows"], "R²": f4(matched["os_only_r2"]), "RMSE": f4(matched["os_only_rmse"]), "MAE": f4(matched["os_only_mae"]), "Fold-level comparison": "reference comparator; identical rows and folds"},
        {"Representation": "Full minus comparator", "Features": "+10 signed", "Pooled rows": matched["rows"], "R²": f4(matched["delta_r2_full_minus_os"]), "RMSE": f4(matched["delta_rmse_full_minus_os"]), "MAE": f4(matched["delta_mae_full_minus_os"]), "Fold-level comparison": f"median ΔR² {f4(matched['delta_r2_median'])}; ΔRMSE {f4(matched['delta_rmse_median'])}; ΔMAE {f4(matched['delta_mae_median'])}"},
    ]

    modality_rows = []
    condition_names = {
        "gsv_only": "GSV only",
        "google_only": "Google only",
        "flickr_only": "Flickr only",
        "image_only": "Image only",
        "text_only": "Text only",
    }
    for row in read_csv(MODALITY):
        if row["stage"] != "Outer test":
            continue
        modality_rows.append({
            "Condition": condition_names[row["condition"]],
            "n folds": row["n_folds"],
            "5th percentile": f3(row["q05"]),
            "25th percentile": f3(row["q25"]),
            "Median RMSE": f3(row["median"]),
            "75th percentile": f3(row["q75"]),
            "95th percentile": f3(row["q95"]),
        })

    partition_raw = read_csv(PARTITION)
    partition_names = {
        "axis_recursive": "Axis-recursive",
        "polar_north_clockwise_equal_count": "Polar equal-count",
        "rotated45_recursive": "Rotated 45° recursive",
        "y_equal_count_stripes": "Y-stripes equal-count",
    }
    partition_rows = []
    for scheme, public_name in partition_names.items():
        rows = [row for row in partition_raw if row["scheme"] == scheme]
        stage = next(row for row in rows if row["panel_type"] == "stage_trajectory")
        spread = next(row for row in rows if row["panel_type"] == "partition_heatmap")
        gap = next(row for row in rows if row["panel_type"] == "paired_train_test")
        support = next(row for row in rows if row["panel_type"] == "support_error_scatter")
        partition_rows.append({
            "Scheme": public_name,
            "n folds": stage["n_folds"],
            "Train median": f3(stage["train_median"]),
            "Validation median": f3(stage["validation_median"]),
            "Outer-test median": f3(stage["test_median"]),
            "Outer-test range": f"{f3(spread['test_min'])}–{f3(spread['test_max'])}",
            "Test−train median": f3(gap["gap_median"]),
            "Outer-test n range": f"{int(float(support['n_test_min']))}–{int(float(support['n_test_max']))}",
        })

    coefficient_source = pd.read_csv(COEFFICIENT_SUMMARY)
    coefficient_source = coefficient_source.loc[~coefficient_source.feature.eq("intercept")].copy()
    if len(coefficient_source) != 43:
        raise ValueError(f"Expected 43 non-intercept coefficients, found {len(coefficient_source)}")
    coefficient_source[["Family", "Readable feature"]] = coefficient_source.feature.apply(
        lambda value: pd.Series(coefficient_label(value))
    )
    coefficient_source["Magnitude"] = coefficient_source.median_local_coefficient.abs()
    coefficient_source["Sign agreement"] = coefficient_source[["positive_share", "negative_share"]].max(axis=1)
    coefficient_rows: list[dict[str, str]] = []
    for family in ("O", "S", "S−O", "SES"):
        family_frame = coefficient_source.loc[coefficient_source["Family"].eq(family)].sort_values(
            ["Magnitude", "feature"], ascending=[False, True]
        )
        for rank, (_, row) in enumerate(family_frame.iterrows(), start=1):
            coefficient_rows.append({
                "Family": family, "Rank": str(rank), "Feature": row["Readable feature"],
                "Median β": f4(row["median_local_coefficient"]), "2.5th percentile": f4(row["q025_local_coefficient"]),
                "97.5th percentile": f4(row["q975_local_coefficient"]),
                "Sign agreement": f"{100 * row['Sign agreement']:.1f}%",
                "Local estimates": f"{int(row['outer_test_evaluations']):,}",
            })

    relation_names = {
        "spatial_queen": "Queen contiguity", "road_connectivity": "Road connectivity",
        "mobility_flow": "Mobility flow", "temporal_forward": "Temporal forward",
    }
    perturbation_names = {"relation_drop": "Relation removed", "edge_weight_half": "Edge weight halved"}
    graph_rows = []
    for row in read_csv(GRAPH_PERTURBATION):
        graph_rows.append({
            "Relation": relation_names[row["relation"]],
            "Perturbation": perturbation_names[row["perturbation"]],
            "Summary level": "Tract-year scheme average" if row["level"] == "tract_year_scheme_average" else "Outer-fold mean",
            "n": row["n"], "Median Δprediction (pp)": f6(row["median"]),
            "5th percentile": f6(row["q05"]), "95th percentile": f6(row["q95"]),
        })

    scenario_names = {
        "D1_green_water_plus_1sd": "Green/blue space +1 SD",
        "D5_clean_maintenance_minus_1sd": "Maintenance burden −1 SD",
        "D7_heat_mitigation_minus_1sd": "Heat stress −1 SD",
        "D8_air_pollution_minus_1sd": "Air pollution −1 SD",
        "D9_noise_mitigation_minus_1sd": "Noise −1 SD",
        "D10_traffic_pressure_minus_1sd": "Traffic pressure −1 SD",
        "joint_six_dimensions": "Joint six-dimension shift",
    }
    scenario_source = pd.read_csv(SCENARIO_PREDICTIONS, usecols=["scenario", "prediction_difference"])
    scenario_rows = []
    for scenario, frame in scenario_source.groupby("scenario", sort=False):
        values = frame.prediction_difference.to_numpy(float)
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        scenario_rows.append({
            "Scenario": scenario_names[scenario], "Outer predictions": f"{len(values):,}",
            "Median Δprediction (pp)": f4(median), "2.5th percentile": f4(q025),
            "97.5th percentile": f4(q975),
            "Fraction with Δ < 0": f"{100 * np.mean(values < 0):.1f}%",
        })

    tables = [
        {"key": "main_table_1", "title": "Table 1 | Measurement framework for objective conditions, platform-linked expression and signed differences", "columns": main_columns, "column_widths": [17, 15, 25, 14, 20, 9], "rows": main_rows, "source_file": "Table_1_measurement_framework.csv", "note": "All objective anchors were oriented so larger values indicate a more adverse condition, z-standardised within health-reference year and averaged across available anchors. S used fixed SigLIP2 adverse-minus-favourable contrasts, within-year standardisation, equal text-source weighting and an equal available text/GSV modality mean. D = S − O. Platform expression is a place-linked proxy, not resident self-report; the 75 no-source rows used an explicitly flagged temporal/Queen fallback."},
        {"key": "supp_table_s1", "title": "Supplementary Table 1 | Data sources, support and missingness", "columns": ["Source or construct", "Modality", "Analytical role", "Supported records", "Coverage", "Missingness or boundary"], "column_widths": [19, 18, 20, 12, 10, 21], "rows": source_rows, "source_file": "Table_S1_data_sources_support.csv", "note": "Coverage percentages use 8,440 tract-year records as the denominator. Source-specific support remains in the V211 support registry."},
        {"key": "supp_table_s2", "title": "Supplementary Table 2 | Dimension definitions and direction conventions", "columns": ["Dimension", "Construct", "Objective anchors (O)", "Platform-expression polarity (S)", "Signed term"], "column_widths": [8, 17, 27, 25, 23], "rows": dimension_rows, "source_file": "Table_S2_variable_groups.csv", "note": "All components were oriented so larger scores correspond to the adverse pole before standardization. S is inferred from platform content and must not be interpreted as a resident survey measure."},
        {"key": "supp_table_s3", "title": "Supplementary Table 3 | Outcome-blind processing and evaluation pipeline", "columns": ["Stage", "Input", "Operation", "Outcome access", "Output or audit"], "column_widths": [13, 19, 34, 13, 21], "rows": process_rows, "source_file": "Table_S3_processing_pipeline.csv", "note": "The outcome-blind label means that data construction, semantic orientation, support handling and partition registration did not use the health outcome. Model fitting necessarily used training-role outcome values."},
        {"key": "supp_table_s4", "title": "Supplementary Table 4 | Current V211 2023 outer-test model benchmark", "columns": ["Model", "RMSE", "MAE", "R²", "Bias", "Device", "Outer-test n"], "column_widths": [22, 12, 12, 12, 12, 15, 15], "rows": benchmark_rows, "source_file": "Table_S4_current_model_benchmark.csv", "note": "This frozen 43-feature benchmark uses 2014–2021 training, 2022 validation and 2023 outer testing. It reports the six manuscript comparator rows retained for display and is distinct from the 32-fold pooled sensitivity analyses."},
        {"key": "supp_table_s5", "title": "Supplementary Table 5 | Matched representation performance", "columns": ["Representation", "Features", "Pooled rows", "R²", "RMSE", "MAE", "Fold-level comparison"], "column_widths": [24, 10, 12, 10, 10, 10, 24], "rows": matched_rows, "source_file": "Table_S5_matched_representation_metrics.csv", "note": "Differences are full minus comparator. Positive RMSE or MAE differences and negative R² differences favour the 33-feature comparator. Empirical fold summaries are not confidence intervals."},
        {"key": "supp_table_s6", "title": "Supplementary Table 6 | Modality-ablation outer-test RMSE", "columns": ["Condition", "n folds", "5th percentile", "25th percentile", "Median RMSE", "75th percentile", "95th percentile"], "column_widths": [22, 10, 14, 14, 14, 13, 13], "rows": modality_rows, "source_file": "Table_S6_modality_outer_test_rmse.csv", "note": "Each condition retains all 32 registered outer folds. Percentiles describe the empirical fold distribution and are not confidence intervals."},
        {"key": "supp_table_s7", "title": "Supplementary Table 7 | Spatial partition and training diagnostics", "columns": ["Scheme", "n folds", "Train median", "Validation median", "Outer-test median", "Outer-test range", "Test−train median", "Outer-test n range"], "column_widths": [20, 8, 12, 13, 13, 14, 12, 8], "rows": partition_rows, "source_file": "Table_S7_spatial_partition_diagnostics.csv", "note": "Values are RMSE in percentage points. Ranges and medians are empirical across the eight folds within each partition scheme."},
        {"key": "supp_table_s8", "title": "Supplementary Table 8 | Complete local-coefficient magnitude and sign summary", "columns": ["Family", "Rank", "Feature", "Median β", "2.5th percentile", "97.5th percentile", "Sign agreement", "Local estimates"], "column_widths": [8, 7, 23, 12, 13, 13, 13, 11], "rows": coefficient_rows, "source_file": "Table_S8_local_coefficient_summary.csv", "note": "β is measured in outcome percentage points per fold-training standard deviation. Percentiles describe the empirical local-coefficient distribution across 33,760 outer-test evaluations per feature and are not confidence intervals. Rank is within family by absolute median β."},
        {"key": "supp_table_s9", "title": "Supplementary Table 9 | Frozen graph-relation perturbation summaries", "columns": ["Relation", "Perturbation", "Summary level", "n", "Median Δprediction (pp)", "5th percentile", "95th percentile"], "column_widths": [17, 16, 21, 9, 15, 11, 11], "rows": graph_rows, "source_file": "Table_S9_graph_perturbation_summary.csv", "note": "Prediction changes come from frozen-model relation removal or edge-weight halving. Values are descriptive replay summaries, not causal network effects or confidence intervals."},
        {"key": "supp_table_s10", "title": "Supplementary Table 10 | Frozen environmental scenario response distributions", "columns": ["Scenario", "Outer predictions", "Median Δprediction (pp)", "2.5th percentile", "97.5th percentile", "Fraction with Δ < 0"], "column_widths": [28, 13, 17, 14, 14, 14], "rows": scenario_rows, "source_file": "Table_S10_scenario_response_distributions.csv", "note": "Each scenario is a bounded one-standard-deviation model replay on current V211 outer-test rows. Negative prediction differences indicate lower model-predicted mental-health burden. These values represent model-projected mitigation potential, not intervention effects."},
        {"key": "supp_table_s11", "title": "Supplementary Table 11 | Study data architecture and analytical roles", "columns": ["Component", "Source and support", "Scale", "Analytical role", "Interpretation boundary"], "column_widths": [15, 25, 18, 22, 20], "rows": architecture_rows, "source_file": "Table_S11_study_data_architecture.csv", "note": "All components belong to the current V211 R3 compact proxy. Google/Flickr image embeddings remain separate raw channels and are not mislabelled as constituents of the compact S score. Platform-derived S is a place-linked expression proxy, not resident self-report."},
        {"key": "supp_table_s12", "title": "Supplementary Table 12 | Complete current V211 43-variable dictionary", "columns": ["Family / dimension", "Variable", "Model field", "Source and construction", "Unit / interpretation"], "column_widths": [11, 17, 24, 31, 17], "rows": variable_rows, "source_file": "Table_S12_complete_43_variable_dictionary.csv", "note": "Every current primary numeric field appears exactly once. O, S and S−O scores are outcome-blind measurement fields; fold-training standardisation used by the fitted model is separate from the within-year measurement harmonisation shown here. S is not resident self-report. The underlying V211 compact dictionary and construction scripts are included in the registry hash contract."},
        {"key": "supp_table_s13", "title": "Supplementary Table 13 | Measurement notation and implemented transformations", "columns": ["Quantity", "Notation", "Implemented operation", "Role / interpretation"], "column_widths": [18, 23, 37, 22], "rows": notation_rows, "source_file": "Table_S13_measurement_notation.csv", "note": "Indices j, k and t denote component, dimension and health-reference year, respectively. Means are computed across available registered components or modalities. These are the implemented V210/V211 operations, not methods proposed only in the funding application."},
    ]

    allowed_display_columns = {
        "S1": {"Analytical role"},
        "S2": {"Signed term"},
        "S3": {"Input"},
        "S5": {"Representation"},
        "S11": {"Source and support"},
        "S12": {"Model field", "Source and construction"},
        "S13": {"Implemented operation", "Role / interpretation"},
    }
    overrides = display_contract["supplementary_text_overrides"]
    if set(overrides) != set(allowed_display_columns):
        raise ValueError("Unexpected supplementary-table text override set")
    for number, fields in allowed_display_columns.items():
        table = next(item for item in tables if item["key"] == f"supp_table_{number.lower()}")
        for override in overrides[number]:
            row, column = int(override["row"]), int(override["column"])
            if row <= 0 or row > len(table["rows"]) or column >= len(table["columns"]):
                raise ValueError(f"Invalid display cell in {number}: {row}, {column}")
            field = table["columns"][column]
            if field not in fields:
                raise ValueError(f"A numeric or unreviewed cell would be overwritten: {number} {field}")
            table["rows"][row - 1][field] = override["value"]

    for table in tables:
        if sum(table["column_widths"]) != 100:
            raise ValueError(f"Column widths for {table['key']} do not sum to 100")
        write_csv(table)

    registry = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analysis_version": "V211 R3 compact proxy",
        "proxy_analysis": True,
        "formal": False,
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in (COMPLETION, SEMANTIC_COMPLETION, DISPLAY_CONTRACT,
                         MODEL_BENCHMARK, MATCHED, MODALITY, PARTITION,
                         COEFFICIENT_SUMMARY, GRAPH_PERTURBATION, SCENARIO_PREDICTIONS,
                         OBJECTIVE_REGISTRY, COMPACT_FEATURE_DICTIONARY)
        },
        "tables": {table["key"]: table for table in tables},
    }
    (OUT / "table_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "tables": len(tables)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir.resolve()
    main()
