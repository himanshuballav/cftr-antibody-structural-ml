#!/usr/bin/env python3

"""
08_prepare_elisa_experimental_data.py

Prepare experimental ELISA phenotype data from manually entered
raw OD measurements.

For each Clone_ID and Species:
    - Maximum OD
    - Mean OD
    - Log10-dilution AUC using the trapezoidal rule

For each clone:
    - Human - Rhesus differences

IMPORTANT
---------
ELISA-derived values describe the experimental phenotype.

They are NOT intended to be predictor variables in the structural
machine-learning classifier when Cross_Reactive_Target is derived
from the same ELISA experiment.

Cross-reactivity labels should therefore be supplied separately.

Expected raw input format
-------------------------
Clone_ID,Species,Dilution,OD
Clone_001,Human,0.5,1.301
Clone_001,Human,0.166666667,1.260
...
Clone_001,Rhesus,0.5,1.637
...

Species must be:
    Human
    Rhesus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Clone_ID",
    "Species",
    "Dilution",
    "OD",
]


# ============================================================
# INPUT
# ============================================================

def read_elisa_data(path: Path) -> pd.DataFrame:
    """Read raw ELISA OD measurements."""

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)

    else:
        raise ValueError(
            "Input must be CSV or Excel (.csv, .xlsx, .xls)."
        )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# QC
# ============================================================

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw ELISA measurements."""

    df = df.copy()

    df["Clone_ID"] = (
        df["Clone_ID"]
        .astype(str)
        .str.strip()
    )

    df["Species"] = (
        df["Species"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    allowed_species = {"Human", "Rhesus"}

    bad_species = set(df["Species"]) - allowed_species

    if bad_species:
        raise ValueError(
            f"Unexpected Species values: {bad_species}. "
            "Use Human or Rhesus."
        )

    df["Dilution"] = pd.to_numeric(
        df["Dilution"],
        errors="coerce"
    )

    df["OD"] = pd.to_numeric(
        df["OD"],
        errors="coerce"
    )

    if df["Dilution"].isna().any():
        raise ValueError(
            "Non-numeric or missing dilution values detected."
        )

    if df["OD"].isna().any():
        raise ValueError(
            "Non-numeric or missing OD values detected."
        )

    if (df["Dilution"] <= 0).any():
        raise ValueError(
            "All dilution values must be > 0 "
            "because log10(dilution) is calculated."
        )

    duplicates = df.duplicated(
        subset=[
            "Clone_ID",
            "Species",
            "Dilution",
        ],
        keep=False,
    )

    if duplicates.any():

        duplicate_rows = df.loc[
            duplicates,
            [
                "Clone_ID",
                "Species",
                "Dilution",
            ],
        ]

        raise ValueError(
            "Duplicate Clone_ID / Species / Dilution "
            "combinations detected:\n"
            f"{duplicate_rows}"
        )

    return df


# ============================================================
# AUC
# ============================================================

def calculate_log10_auc(
    dilution: np.ndarray,
    od: np.ndarray,
) -> float:
    """
    Calculate ELISA AUC over log10(dilution).

    Explicit trapezoidal calculation:

        AUC =
        SUM [
            abs(x[i+1] - x[i])
            *
            (OD[i] + OD[i+1]) / 2
        ]

    where:

        x = log10(dilution)
    """

    dilution = np.asarray(
        dilution,
        dtype=float,
    )

    od = np.asarray(
        od,
        dtype=float,
    )

    # Sort from largest dilution value to smallest.
    order = np.argsort(dilution)[::-1]

    dilution = dilution[order]
    od = od[order]

    log_dilution = np.log10(dilution)

    auc = 0.0

    for i in range(len(od) - 1):

        width = abs(
            log_dilution[i + 1]
            -
            log_dilution[i]
        )

        mean_height = (
            od[i]
            +
            od[i + 1]
        ) / 2.0

        auc += width * mean_height

    return float(auc)


# ============================================================
# CURVE SUMMARY
# ============================================================

def summarize_curves(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate summary statistics for every ELISA curve."""

    records = []

    for (clone, species), group in df.groupby(
        ["Clone_ID", "Species"],
        sort=True,
    ):

        group = group.sort_values(
            "Dilution",
            ascending=False,
        )

        max_od = group["OD"].max()

        mean_od = group["OD"].mean()

        auc = calculate_log10_auc(
            group["Dilution"].to_numpy(),
            group["OD"].to_numpy(),
        )

        records.append(
            {
                "Clone_ID": clone,
                "Species": species,
                "N_Dilutions": len(group),
                "Max_OD": max_od,
                "Mean_OD": mean_od,
                "Log10_AUC": auc,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# HUMAN - RHESUS
# ============================================================

def calculate_paired_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert Human/Rhesus curve summaries into one row per clone.

    Delta convention:

        Delta = Human - Rhesus
    """

    metrics = [
        "Max_OD",
        "Mean_OD",
        "Log10_AUC",
    ]

    records = []

    for clone, group in summary.groupby(
        "Clone_ID",
        sort=True,
    ):

        species_present = set(
            group["Species"]
        )

        if species_present != {"Human", "Rhesus"}:

            print(
                f"WARNING: {clone} does not have both "
                "Human and Rhesus measurements. Skipping."
            )

            continue

        human = (
            group.loc[
                group["Species"] == "Human"
            ]
            .iloc[0]
        )

        rhesus = (
            group.loc[
                group["Species"] == "Rhesus"
            ]
            .iloc[0]
        )

        record = {
            "Clone_ID": clone,
        }

        for metric in metrics:

            h = float(human[metric])
            r = float(rhesus[metric])

            record[f"Human_{metric}"] = h
            record[f"Rhesus_{metric}"] = r
            record[f"Delta_{metric}"] = h - r

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# LABELS
# ============================================================

def merge_labels(
    paired: pd.DataFrame,
    labels_path: Path | None,
) -> pd.DataFrame:
    """
    Optionally merge manually assigned experimental
    cross-reactivity labels.

    Expected columns:

        Clone_ID
        Cross_Reactive_Target

    Encoding:

        0 = Non-cross-reactive / Human-specific
        1 = Cross-reactive
    """

    if labels_path is None:
        return paired

    suffix = labels_path.suffix.lower()

    if suffix == ".csv":
        labels = pd.read_csv(labels_path)

    elif suffix in {".xlsx", ".xls"}:
        labels = pd.read_excel(labels_path)

    else:
        raise ValueError(
            "Labels file must be CSV or Excel."
        )

    required = {
        "Clone_ID",
        "Cross_Reactive_Target",
    }

    missing = required - set(labels.columns)

    if missing:
        raise ValueError(
            f"Labels file missing columns: {missing}"
        )

    labels = labels.copy()

    labels["Clone_ID"] = (
        labels["Clone_ID"]
        .astype(str)
        .str.strip()
    )

    labels["Cross_Reactive_Target"] = (
        pd.to_numeric(
            labels["Cross_Reactive_Target"],
            errors="raise",
        )
    )

    bad_labels = (
        ~labels["Cross_Reactive_Target"]
        .isin([0, 1])
    )

    if bad_labels.any():
        raise ValueError(
            "Cross_Reactive_Target must contain only 0 or 1."
        )

    labels["Cross_Reactive_Label"] = (
        labels["Cross_Reactive_Target"]
        .map(
            {
                0: "Non-cross-reactive",
                1: "Cross-reactive",
            }
        )
    )

    result = paired.merge(
        labels[
            [
                "Clone_ID",
                "Cross_Reactive_Target",
                "Cross_Reactive_Label",
            ]
        ],
        on="Clone_ID",
        how="left",
        validate="one_to_one",
    )

    return result


# ============================================================
# OUTPUT
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calculate experimental ELISA curve summaries "
            "and Human-Rhesus differences."
        )
    )

    parser.add_argument(
        "--raw-elisa",
        required=True,
        type=Path,
        help="Raw ELISA CSV/XLSX file.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Optional manually annotated cross-reactivity "
            "CSV/XLSX file."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Excel workbook.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ELISA EXPERIMENTAL DATA PROCESSING")
    print("=" * 70)

    print(f"\nRaw ELISA: {args.raw_elisa}")

    raw = read_elisa_data(
        args.raw_elisa
    )

    raw = clean_and_validate(
        raw
    )

    print(
        f"Loaded {len(raw)} raw OD measurements "
        f"from {raw['Clone_ID'].nunique()} clones."
    )

    summary = summarize_curves(
        raw
    )

    paired = calculate_paired_summary(
        summary
    )

    paired = merge_labels(
        paired,
        args.labels,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        args.output,
        engine="openpyxl",
    ) as writer:

        raw.to_excel(
            writer,
            sheet_name="Raw_ELISA_OD",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="Curve_Summary",
            index=False,
        )

        paired.to_excel(
            writer,
            sheet_name="Experimental_Phenotype",
            index=False,
        )

        methodology = pd.DataFrame(
            {
                "Field": [
                    "Raw OD",
                    "Max_OD",
                    "Mean_OD",
                    "Log10_AUC",
                    "Delta convention",
                    "Cross_Reactive_Target = 0",
                    "Cross_Reactive_Target = 1",
                    "ML usage",
                ],
                "Definition": [
                    "Experimental ELISA optical density measurement",
                    "Maximum OD across dilution series",
                    "Mean OD across dilution series",
                    "Trapezoidal AUC across log10(dilution)",
                    "Human - Rhesus",
                    "Non-cross-reactive / Human-specific",
                    "Cross-reactive",
                    (
                        "ELISA values characterize experimental "
                        "phenotype and are not structural ML predictors"
                    ),
                ],
            }
        )

        methodology.to_excel(
            writer,
            sheet_name="Methodology",
            index=False,
        )

    print(f"\nSaved: {args.output}")

    print("\nCreated sheets:")
    print("  Raw_ELISA_OD")
    print("  Curve_Summary")
    print("  Experimental_Phenotype")
    print("  Methodology")

    print("\nDone.")


if __name__ == "__main__":
    main()