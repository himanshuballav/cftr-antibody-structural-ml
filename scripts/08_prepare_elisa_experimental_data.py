#!/usr/bin/env python3
"""
08_prepare_elisa_experimental_data.py

General ELISA experimental-phenotype processor.

What this script does
---------------------
1. Reads raw ELISA OD measurements.
2. Validates that each clone has matching dilution points for the two species.
3. Calculates, for each Clone_ID and Species:
       - Max OD
       - Mean OD
       - AUC across log10(dilution), using the trapezoidal rule
4. Pairs the reference and comparison species for each clone.
5. Calculates:
       - reference - comparison differences
       - comparison/reference AUC ratio
6. Optionally converts the continuous AUC ratio into a binary experimental
   cross-reactivity target using a USER-SUPPLIED threshold.

The script does NOT assume that 0.10 is a universal cutoff.

For the current Human-vs-Rhesus project, run with:
    --reference-species Human
    --comparison-species Rhesus
    --cr-ratio-threshold 0.10

Then:
    ratio < 0.10  -> 0 = Non-cross-reactive / reference-specific
    ratio >= 0.10 -> 1 = Cross-reactive

For another project, the species names and threshold can be changed without
editing the source code.

IMPORTANT FOR DOWNSTREAM ML
---------------------------
ELISA-derived quantities (AUC, delta AUC, AUC ratio, etc.) describe or define
the experimental phenotype. If Cross_Reactive_Target is derived from these
ELISA measurements, these ELISA variables must NOT be used as structural ML
predictors.

Expected raw input
------------------
Required columns:
    Clone_ID, Species, Dilution, OD

Example:
    Clone_001,Human,0.5,1.301
    Clone_001,Human,0.166666667,1.260
    Clone_001,Rhesus,0.5,1.637
    Clone_001,Rhesus,0.166666667,1.677

The script is not hard-coded to four dilution points. It works with any
number >= 2, provided the reference and comparison species use matching
dilution points for each clone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Clone_ID", "Species", "Dilution", "OD"]


# ============================================================
# HELPERS
# ============================================================

def safe_name(value: str) -> str:
    """Convert a species/condition label into a safe column-name token."""
    value = str(value).strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "", value)
    if not value:
        raise ValueError("Species/condition name cannot be empty.")
    return value


# ============================================================
# INPUT
# ============================================================

def read_elisa_data(path: Path) -> pd.DataFrame:
    """Read raw ELISA OD measurements from CSV or Excel."""

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Input must be CSV or Excel (.csv, .xlsx, .xls).")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ============================================================
# QC / CLEANING
# ============================================================

def clean_and_validate(
    df: pd.DataFrame,
    reference_species: str,
    comparison_species: str,
) -> pd.DataFrame:
    """Clean the raw data and validate paired dilution series."""

    df = df.copy()

    df["Clone_ID"] = df["Clone_ID"].astype(str).str.strip()
    df["Species"] = df["Species"].astype(str).str.strip()

    if (df["Clone_ID"] == "").any():
        raise ValueError("Blank Clone_ID values detected.")

    df["Dilution"] = pd.to_numeric(df["Dilution"], errors="coerce")
    df["OD"] = pd.to_numeric(df["OD"], errors="coerce")

    if df["Dilution"].isna().any():
        raise ValueError("Non-numeric or missing dilution values detected.")

    if df["OD"].isna().any():
        raise ValueError("Non-numeric or missing OD values detected.")

    if (df["Dilution"] <= 0).any():
        raise ValueError(
            "All dilution values must be > 0 because log10(dilution) "
            "is calculated."
        )

    if reference_species == comparison_species:
        raise ValueError(
            "Reference species and comparison species must be different."
        )

    expected = {reference_species, comparison_species}
    observed = set(df["Species"].unique())

    unexpected = observed - expected
    missing_species = expected - observed

    if unexpected:
        raise ValueError(
            f"Unexpected Species values: {sorted(unexpected)}. "
            f"Expected only: {sorted(expected)}"
        )

    if missing_species:
        raise ValueError(
            f"Missing required Species values: {sorted(missing_species)}"
        )

    duplicates = df.duplicated(
        subset=["Clone_ID", "Species", "Dilution"],
        keep=False,
    )

    if duplicates.any():
        bad = df.loc[
            duplicates,
            ["Clone_ID", "Species", "Dilution"],
        ]
        raise ValueError(
            "Duplicate Clone_ID / Species / Dilution combinations detected:\n"
            + bad.to_string(index=False)
        )

    # Require the same dilution grid for both species/conditions per clone.
    for clone, group in df.groupby("Clone_ID", sort=True):

        present = set(group["Species"])
        if present != expected:
            raise ValueError(
                f"{clone}: expected both {reference_species!r} and "
                f"{comparison_species!r}; found {sorted(present)}"
            )

        ref_dilutions = sorted(
            group.loc[
                group["Species"] == reference_species,
                "Dilution",
            ].astype(float)
        )

        comp_dilutions = sorted(
            group.loc[
                group["Species"] == comparison_species,
                "Dilution",
            ].astype(float)
        )

        if len(ref_dilutions) < 2 or len(comp_dilutions) < 2:
            raise ValueError(
                f"{clone}: at least two dilution points per species "
                "are required."
            )

        if len(ref_dilutions) != len(comp_dilutions):
            raise ValueError(
                f"{clone}: reference and comparison species have "
                "different numbers of dilution points."
            )

        if not np.allclose(
            ref_dilutions,
            comp_dilutions,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                f"{clone}: dilution points do not match.\n"
                f"{reference_species}: {ref_dilutions}\n"
                f"{comparison_species}: {comp_dilutions}"
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
    Calculate trapezoidal ELISA AUC across log10(dilution).

    AUC = sum(
        abs(log10(d[i+1]) - log10(d[i]))
        * (OD[i] + OD[i+1]) / 2
    )
    """

    dilution = np.asarray(dilution, dtype=float)
    od = np.asarray(od, dtype=float)

    if len(dilution) != len(od):
        raise ValueError("Dilution and OD arrays must have equal length.")

    if len(dilution) < 2:
        raise ValueError("At least two dilution points are required.")

    order = np.argsort(dilution)[::-1]
    dilution = dilution[order]
    od = od[order]

    x = np.log10(dilution)

    auc = 0.0
    for i in range(len(od) - 1):
        width = abs(x[i + 1] - x[i])
        mean_height = (od[i] + od[i + 1]) / 2.0
        auc += width * mean_height

    return float(auc)


# ============================================================
# PER-CURVE SUMMARY
# ============================================================

def summarize_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Max OD, Mean OD, and log10-dilution AUC for each curve."""

    records = []

    for (clone, species), group in df.groupby(
        ["Clone_ID", "Species"],
        sort=True,
    ):
        group = group.sort_values("Dilution", ascending=False)

        records.append(
            {
                "Clone_ID": clone,
                "Species": species,
                "N_Dilutions": int(len(group)),
                "Max_OD": float(group["OD"].max()),
                "Mean_OD": float(group["OD"].mean()),
                "Log10_AUC": calculate_log10_auc(
                    group["Dilution"].to_numpy(),
                    group["OD"].to_numpy(),
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# PAIRED EXPERIMENTAL PHENOTYPE
# ============================================================

def calculate_paired_summary(
    summary: pd.DataFrame,
    reference_species: str,
    comparison_species: str,
) -> pd.DataFrame:
    """
    Build one row per clone.

    Delta convention:
        Delta = reference - comparison

    Ratio convention:
        comparison/reference AUC
    """

    ref = safe_name(reference_species)
    comp = safe_name(comparison_species)

    metrics = ["Max_OD", "Mean_OD", "Log10_AUC"]
    records = []

    for clone, group in summary.groupby("Clone_ID", sort=True):

        ref_row = group.loc[group["Species"] == reference_species]
        comp_row = group.loc[group["Species"] == comparison_species]

        if len(ref_row) != 1 or len(comp_row) != 1:
            raise ValueError(
                f"{clone}: expected exactly one summarized curve for each "
                "species."
            )

        ref_row = ref_row.iloc[0]
        comp_row = comp_row.iloc[0]

        record = {"Clone_ID": clone}

        for metric in metrics:
            r = float(ref_row[metric])
            c = float(comp_row[metric])

            record[f"{ref}_{metric}"] = r
            record[f"{comp}_{metric}"] = c
            record[f"Delta_{metric}"] = r - c

        ref_auc = record[f"{ref}_Log10_AUC"]
        comp_auc = record[f"{comp}_Log10_AUC"]

        if ref_auc <= 0:
            raise ValueError(
                f"{clone}: {reference_species} Log10_AUC is <= 0, so the "
                "comparison/reference AUC ratio is undefined."
            )

        record["Comparison_to_Reference_AUC_Ratio"] = comp_auc / ref_auc

        # Keep a descriptive alias that makes the current species explicit.
        record[f"{comp}_to_{ref}_AUC_Ratio"] = comp_auc / ref_auc

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# OPTIONAL BINARY CLASSIFICATION
# ============================================================

def add_binary_cross_reactivity_target(
    paired: pd.DataFrame,
    threshold: float | None,
    reference_species: str,
    comparison_species: str,
) -> pd.DataFrame:
    """
    Optionally classify the continuous AUC ratio into 0/1.

    If threshold is None:
        no binary target is generated.

    If threshold is supplied:
        ratio < threshold  -> 0
        ratio >= threshold -> 1
    """

    result = paired.copy()

    if threshold is None:
        return result

    if threshold < 0:
        raise ValueError("Cross-reactivity ratio threshold must be >= 0.")

    ratio = result["Comparison_to_Reference_AUC_Ratio"]

    result["Cross_Reactivity_Ratio_Threshold"] = float(threshold)

    result["Cross_Reactive_Target"] = (
        ratio >= threshold
    ).astype(int)

    result["Cross_Reactive_Label"] = (
        result["Cross_Reactive_Target"].map(
            {
                0: f"Non-cross-reactive / {reference_species}-specific",
                1: "Cross-reactive",
            }
        )
    )

    result["Classification_Method"] = (
        f"Cross-reactive if {comparison_species}/{reference_species} "
        f"Log10-AUC ratio >= {threshold:g}"
    )

    return result


# ============================================================
# CLASSIFICATION SUMMARY
# ============================================================

def build_classification_summary(
    paired: pd.DataFrame,
    reference_species: str,
    comparison_species: str,
) -> pd.DataFrame:
    """Create a compact phenotype/classification table."""

    ref = safe_name(reference_species)
    comp = safe_name(comparison_species)

    cols = [
        "Clone_ID",
        f"{ref}_Log10_AUC",
        f"{comp}_Log10_AUC",
        "Delta_Log10_AUC",
        "Comparison_to_Reference_AUC_Ratio",
        f"{comp}_to_{ref}_AUC_Ratio",
    ]

    optional = [
        "Cross_Reactivity_Ratio_Threshold",
        "Cross_Reactive_Target",
        "Cross_Reactive_Label",
        "Classification_Method",
    ]

    cols += [c for c in optional if c in paired.columns]

    out = paired[cols].copy()

    if "Cross_Reactive_Target" in out.columns:
        out = out.sort_values(
            ["Cross_Reactive_Target", "Comparison_to_Reference_AUC_Ratio"],
            ascending=[True, True],
        )
    else:
        out = out.sort_values(
            "Comparison_to_Reference_AUC_Ratio",
            ascending=True,
        )

    return out


# ============================================================
# METHODOLOGY
# ============================================================

def build_methodology(
    reference_species: str,
    comparison_species: str,
    threshold: float | None,
) -> pd.DataFrame:
    """Document exactly how the experimental phenotype was calculated."""

    rows = [
        (
            "Reference species/condition",
            reference_species,
        ),
        (
            "Comparison species/condition",
            comparison_species,
        ),
        (
            "Raw OD",
            "Experimental ELISA optical density measurement.",
        ),
        (
            "Max_OD",
            "Maximum OD across the dilution series.",
        ),
        (
            "Mean_OD",
            "Mean OD across the dilution series.",
        ),
        (
            "Log10_AUC",
            "Trapezoidal AUC across log10(dilution).",
        ),
        (
            "Delta convention",
            f"{reference_species} - {comparison_species}.",
        ),
        (
            "Comparison_to_Reference_AUC_Ratio",
            f"{comparison_species}_Log10_AUC divided by "
            f"{reference_species}_Log10_AUC.",
        ),
    ]

    if threshold is None:
        rows.extend(
            [
                (
                    "Binary cross-reactivity classification",
                    "Not generated because no --cr-ratio-threshold was supplied.",
                ),
                (
                    "Recommended general use",
                    "Keep the AUC ratio as a continuous phenotype unless a "
                    "biologically justified, prespecified cutoff exists.",
                ),
            ]
        )
    else:
        rows.extend(
            [
                (
                    "Cross-reactivity threshold",
                    f"{threshold:g}",
                ),
                (
                    "Cross_Reactive_Target = 0",
                    f"AUC ratio < {threshold:g}: Non-cross-reactive / "
                    f"{reference_species}-specific.",
                ),
                (
                    "Cross_Reactive_Target = 1",
                    f"AUC ratio >= {threshold:g}: Cross-reactive.",
                ),
                (
                    "Threshold interpretation",
                    "Project-specific operational cutoff; not a universal "
                    "ELISA cross-reactivity constant.",
                ),
            ]
        )

    rows.append(
        (
            "Structural ML usage",
            "ELISA values, AUCs, AUC ratios, and labels derived from them "
            "describe/define the phenotype and must not be included as "
            "structural ML predictors.",
        )
    )

    return pd.DataFrame(rows, columns=["Field", "Definition"])


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "General paired-species ELISA phenotype processor with optional "
            "binary cross-reactivity classification."
        )
    )

    parser.add_argument(
        "--raw-elisa",
        required=True,
        type=Path,
        help="Raw ELISA CSV/XLSX file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Excel workbook.",
    )

    parser.add_argument(
        "--reference-species",
        default="Human",
        help="Reference species/condition name. Default: Human",
    )

    parser.add_argument(
        "--comparison-species",
        default="Rhesus",
        help="Comparison species/condition name. Default: Rhesus",
    )

    parser.add_argument(
        "--cr-ratio-threshold",
        type=float,
        default=None,
        help=(
            "Optional comparison/reference Log10-AUC ratio cutoff for "
            "binary classification. If omitted, the script calculates the "
            "continuous ratio but does not assign 0/1 labels."
        ),
    )

    args = parser.parse_args()

    reference_species = args.reference_species.strip()
    comparison_species = args.comparison_species.strip()

    print("=" * 80)
    print("ELISA EXPERIMENTAL PHENOTYPE PROCESSING")
    print("=" * 80)

    print(f"\nRaw ELISA:          {args.raw_elisa}")
    print(f"Reference species:  {reference_species}")
    print(f"Comparison species: {comparison_species}")

    if args.cr_ratio_threshold is None:
        print("Binary threshold:    None (continuous phenotype only)")
    else:
        print(f"Binary threshold:    {args.cr_ratio_threshold:g}")

    if not args.raw_elisa.exists():
        raise FileNotFoundError(
            f"Raw ELISA file not found:\n{args.raw_elisa}"
        )

    raw = read_elisa_data(args.raw_elisa)

    raw = clean_and_validate(
        raw,
        reference_species=reference_species,
        comparison_species=comparison_species,
    )

    print(
        f"\nLoaded {len(raw)} raw OD measurements "
        f"from {raw['Clone_ID'].nunique()} clones."
    )

    curve_summary = summarize_curves(raw)

    paired = calculate_paired_summary(
        curve_summary,
        reference_species=reference_species,
        comparison_species=comparison_species,
    )

    paired = add_binary_cross_reactivity_target(
        paired,
        threshold=args.cr_ratio_threshold,
        reference_species=reference_species,
        comparison_species=comparison_species,
    )

    classification_summary = build_classification_summary(
        paired,
        reference_species=reference_species,
        comparison_species=comparison_species,
    )

    methodology = build_methodology(
        reference_species=reference_species,
        comparison_species=comparison_species,
        threshold=args.cr_ratio_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:

        raw.to_excel(
            writer,
            sheet_name="Raw_ELISA_OD",
            index=False,
        )

        curve_summary.to_excel(
            writer,
            sheet_name="Curve_Summary",
            index=False,
        )

        paired.to_excel(
            writer,
            sheet_name="Experimental_Phenotype",
            index=False,
        )

        classification_summary.to_excel(
            writer,
            sheet_name="Classification_Summary",
            index=False,
        )

        methodology.to_excel(
            writer,
            sheet_name="Methodology",
            index=False,
        )

    print(f"\nSaved:\n{args.output}")

    if "Cross_Reactive_Target" in paired.columns:

        n_non_cr = int((paired["Cross_Reactive_Target"] == 0).sum())
        n_cr = int((paired["Cross_Reactive_Target"] == 1).sum())

        print("\nClassification counts:")
        print(f"  0 = Non-cross-reactive: {n_non_cr}")
        print(f"  1 = Cross-reactive:     {n_cr}")

        print("\nNon-cross-reactive clones:")
        print(
            paired.loc[
                paired["Cross_Reactive_Target"] == 0,
                [
                    "Clone_ID",
                    "Comparison_to_Reference_AUC_Ratio",
                ],
            ]
            .sort_values("Comparison_to_Reference_AUC_Ratio")
            .to_string(index=False)
        )

        print("\nCross-reactive clones:")
        print(
            paired.loc[
                paired["Cross_Reactive_Target"] == 1,
                [
                    "Clone_ID",
                    "Comparison_to_Reference_AUC_Ratio",
                ],
            ]
            .sort_values("Comparison_to_Reference_AUC_Ratio")
            .to_string(index=False)
        )

    else:
        print(
            "\nNo 0/1 labels were assigned because no "
            "--cr-ratio-threshold was supplied."
        )

    print("\nCreated sheets:")
    print("  Raw_ELISA_OD")
    print("  Curve_Summary")
    print("  Experimental_Phenotype")
    print("  Classification_Summary")
    print("  Methodology")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR")
        print("=" * 80)
        print(exc)
        sys.exit(1)
