#!/usr/bin/env python3
"""
09_build_ml_ready_dataset.py

Build the final ML-ready workbook by combining:

1) Structural delta features from script 07
2) Experimental phenotype + binary target from script 08

No separate label file is required.

EXPECTED INPUTS
---------------
Script 07 workbook:
    ML_Signed_Input
    ML_Absolute_Input

Script 08 workbook:
    Experimental_Phenotype

Script 08 must have been run WITH --cr-ratio-threshold so that
Experimental_Phenotype contains:
    Clone_ID
    Cross_Reactive_Target
    Cross_Reactive_Label

TARGET
------
Cross_Reactive_Target
    0 = Non-cross-reactive / reference-specific
    1 = Cross-reactive

PREDICTORS
----------
Signed model:
    Delta_* structural/computational columns only

Absolute model:
    AbsDelta_* structural/computational columns only

IMPORTANT
---------
ELISA-derived measurements are retained in the output workbook for
experimental provenance and interpretation, but they are NEVER ML predictors.

Outputs:
    ML_Ready_Signed
    ML_Ready_Absolute
    Experimental_Phenotype
    Label_Table
    Data_Dictionary
    QC_Summary
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ID_COL = "Clone_ID"
TARGET_COL = "Cross_Reactive_Target"
LABEL_COL = "Cross_Reactive_Label"

SIGNED_SHEET = "ML_Signed_Input"
ABSOLUTE_SHEET = "ML_Absolute_Input"
ELISA_SHEET = "Experimental_Phenotype"


def clean_clone_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def validate_unique_ids(df: pd.DataFrame, table_name: str) -> None:
    if ID_COL not in df.columns:
        raise ValueError(f"{table_name} is missing required column '{ID_COL}'.")

    if df[ID_COL].isna().any():
        raise ValueError(f"{table_name} contains missing {ID_COL} values.")

    duplicated = df.loc[df[ID_COL].duplicated(keep=False), ID_COL].tolist()
    if duplicated:
        raise ValueError(
            f"{table_name} contains duplicate clone IDs: {sorted(set(duplicated))}"
        )


def namespace_elisa_predictor_like_columns(elisa: pd.DataFrame) -> pd.DataFrame:
    """
    Rename ELISA phenotype columns that look like structural predictor names.

    Script 10 intentionally selects structural predictors by prefix:
        Delta_*
        AbsDelta_*

    ELISA phenotype tables can also contain names such as Delta_Log10_AUC,
    Delta_Max_OD, or Delta_Mean_OD. Those are experimental phenotype variables,
    NOT structural predictors. Prefix them with "Experimental_" so they remain
    in the workbook for provenance but can never enter X by prefix matching.
    """
    elisa = elisa.copy()

    rename_map = {}
    for col in elisa.columns:
        name = str(col)
        if name.startswith("Delta_") or name.startswith("AbsDelta_"):
            rename_map[col] = f"Experimental_{name}"

    if rename_map:
        elisa = elisa.rename(columns=rename_map)

    return elisa


def validate_target_from_elisa(elisa: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the target produced by script 08.

    This script does NOT recalculate or redefine cross-reactivity.
    It simply carries forward the experimental target from script 08.
    """
    elisa = elisa.copy()

    required = {ID_COL, TARGET_COL}
    missing = required - set(elisa.columns)
    if missing:
        raise ValueError(
            "Experimental_Phenotype is missing required columns: "
            + ", ".join(sorted(missing))
            + "\nRun script 08 with --cr-ratio-threshold to generate the binary target."
        )

    elisa[ID_COL] = clean_clone_id(elisa[ID_COL])
    validate_unique_ids(elisa, ELISA_SHEET)

    elisa[TARGET_COL] = pd.to_numeric(
        elisa[TARGET_COL], errors="raise"
    ).astype(int)

    if not elisa[TARGET_COL].isin([0, 1]).all():
        bad = elisa.loc[
            ~elisa[TARGET_COL].isin([0, 1]),
            [ID_COL, TARGET_COL],
        ]
        raise ValueError(
            f"{TARGET_COL} must contain only 0 or 1.\n"
            + bad.to_string(index=False)
        )

    if sorted(elisa[TARGET_COL].unique().tolist()) != [0, 1]:
        raise ValueError(
            f"{TARGET_COL} must contain both classes 0 and 1."
        )

    if LABEL_COL not in elisa.columns:
        elisa[LABEL_COL] = elisa[TARGET_COL].map(
            {
                0: "Non-cross-reactive / reference-specific",
                1: "Cross-reactive",
            }
        )

    return elisa


def identify_structural_features(
    df: pd.DataFrame,
    expected_prefix: str,
    table_name: str,
) -> list[str]:
    """
    Select only structural features with the expected prefix.

    Signed sheet:   Delta_*
    Absolute sheet: AbsDelta_*
    """
    features = [
        str(c) for c in df.columns
        if str(c).startswith(expected_prefix)
    ]

    if not features:
        raise ValueError(
            f"{table_name}: no predictors beginning with "
            f"'{expected_prefix}' were found."
        )

    # Defense against accidental experimental leakage.
    leakage_terms = [
        "ELISA",
        "EXPERIMENTAL_AUC",
        "HUMAN_LOG10_AUC",
        "RHESUS_LOG10_AUC",
        "AUC_RATIO",
        "CROSS_REACTIVE",
        "TARGET",
    ]

    bad = [
        c for c in features
        if any(term in c.upper() for term in leakage_terms)
    ]

    if bad:
        raise ValueError(
            f"{table_name}: potential phenotype leakage columns detected:\n"
            + "\n".join(bad)
        )

    return features


def prepare_structural_table(
    df: pd.DataFrame,
    table_name: str,
    expected_prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return Clone_ID + numeric structural feature matrix."""
    df = df.copy()

    if ID_COL not in df.columns:
        raise ValueError(f"{table_name} is missing '{ID_COL}'.")

    df[ID_COL] = clean_clone_id(df[ID_COL])
    validate_unique_ids(df, table_name)

    features = identify_structural_features(
        df,
        expected_prefix=expected_prefix,
        table_name=table_name,
    )

    X = df[features].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        missing = X.isna().sum()
        missing = missing[missing > 0]
        raise ValueError(
            f"{table_name} contains missing/non-numeric structural features:\n"
            + missing.to_string()
        )

    out = pd.concat(
        [
            df[[ID_COL]].reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    )

    return out, features


def compare_clone_sets(
    signed: pd.DataFrame,
    absolute: pd.DataFrame,
    elisa: pd.DataFrame,
) -> None:
    """Require exact clone-set agreement across scripts 07 and 08."""
    sets = {
        "signed structural": set(signed[ID_COL]),
        "absolute structural": set(absolute[ID_COL]),
        "ELISA phenotype": set(elisa[ID_COL]),
    }

    union = set.union(*sets.values())

    problems = []
    for name, clone_set in sets.items():
        missing = sorted(union - clone_set)
        if missing:
            problems.append(f"{name} is missing: {missing}")

    if problems:
        raise ValueError(
            "Clone sets do not match across inputs.\n"
            + "\n".join(problems)
        )


def merge_ml_table(
    structural: pd.DataFrame,
    elisa: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge target + phenotype provenance + structural predictors.

    ELISA columns are kept for reference but the classifier will select only
    Delta_* or AbsDelta_* columns.
    """
    merged = elisa.merge(
        structural,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    front = [ID_COL, TARGET_COL, LABEL_COL]

    front += [
        c for c in [
            "Comparison_to_Reference_AUC_Ratio",
            "Cross_Reactivity_Ratio_Threshold",
            "Classification_Method",
        ]
        if c in merged.columns and c not in front
    ]

    rest = [c for c in merged.columns if c not in front]

    return merged[front + rest]


def build_data_dictionary(
    signed_features: list[str],
    absolute_features: list[str],
    elisa_columns: list[str],
) -> pd.DataFrame:

    rows = [
        {
            "Column_or_Group": ID_COL,
            "Role": "Identifier",
            "Used_as_ML_Predictor": "No",
            "Description": "Antibody clone identifier.",
        },
        {
            "Column_or_Group": TARGET_COL,
            "Role": "Outcome y",
            "Used_as_ML_Predictor": "No",
            "Description": (
                "Binary experimental cross-reactivity target generated by "
                "script 08 from the prespecified ELISA AUC-ratio cutoff."
            ),
        },
        {
            "Column_or_Group": "ELISA phenotype columns",
            "Role": "Experimental phenotype/provenance",
            "Used_as_ML_Predictor": "No",
            "Description": (
                "OD/AUC/ratio/threshold columns are retained for provenance "
                "only and must never enter structural model X."
            ),
        },
        {
            "Column_or_Group": "Delta_*",
            "Role": "Signed structural predictors X",
            "Used_as_ML_Predictor": "Yes, signed model only",
            "Description": "Reference - comparison structural/computational difference.",
        },
        {
            "Column_or_Group": "AbsDelta_*",
            "Role": "Absolute structural predictors X",
            "Used_as_ML_Predictor": "Yes, absolute model only",
            "Description": "Absolute magnitude of the paired structural difference.",
        },
    ]

    for col in elisa_columns:
        if col == ID_COL:
            continue
        rows.append(
            {
                "Column_or_Group": col,
                "Role": "Experimental phenotype/reference",
                "Used_as_ML_Predictor": "No",
                "Description": "Column inherited from script 08.",
            }
        )

    for col in signed_features:
        rows.append(
            {
                "Column_or_Group": col,
                "Role": "Signed structural predictor",
                "Used_as_ML_Predictor": "Yes, signed model only",
                "Description": "Structural/computational signed paired difference.",
            }
        )

    for col in absolute_features:
        rows.append(
            {
                "Column_or_Group": col,
                "Role": "Absolute structural predictor",
                "Used_as_ML_Predictor": "Yes, absolute model only",
                "Description": "Absolute structural/computational paired difference.",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine script 07 structural deltas and script 08 experimental "
            "phenotype/target into the final ML-ready workbook."
        )
    )

    parser.add_argument(
        "--delta-workbook",
        required=True,
        type=Path,
        help="Output workbook from script 07.",
    )

    parser.add_argument(
        "--elisa-workbook",
        required=True,
        type=Path,
        help="Output workbook from script 08.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output ML-ready Excel workbook.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("BUILDING FINAL ML-READY DATASET")
    print("=" * 90)

    for path in [args.delta_workbook, args.elisa_workbook]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found:\n{path}")

    # ------------------------------------------------------------
    # Script 07 structural features
    # ------------------------------------------------------------
    signed_raw = pd.read_excel(
        args.delta_workbook,
        sheet_name=SIGNED_SHEET,
    )

    absolute_raw = pd.read_excel(
        args.delta_workbook,
        sheet_name=ABSOLUTE_SHEET,
    )

    signed, signed_features = prepare_structural_table(
        signed_raw,
        table_name=SIGNED_SHEET,
        expected_prefix="Delta_",
    )

    absolute, absolute_features = prepare_structural_table(
        absolute_raw,
        table_name=ABSOLUTE_SHEET,
        expected_prefix="AbsDelta_",
    )

    # ------------------------------------------------------------
    # Script 08 target / phenotype
    # ------------------------------------------------------------
    elisa_raw = pd.read_excel(
        args.elisa_workbook,
        sheet_name=ELISA_SHEET,
    )

    elisa = validate_target_from_elisa(elisa_raw)
    elisa = namespace_elisa_predictor_like_columns(elisa)

    renamed_elisa_deltas = [
        c for c in elisa.columns
        if str(c).startswith("Experimental_Delta_")
        or str(c).startswith("Experimental_AbsDelta_")
    ]
    if renamed_elisa_deltas:
        print("\nExperimental ELISA delta columns namespaced to prevent leakage:")
        for c in renamed_elisa_deltas:
            print(f"  {c}")

    compare_clone_sets(
        signed=signed,
        absolute=absolute,
        elisa=elisa,
    )

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------
    ml_signed = merge_ml_table(signed, elisa)
    ml_absolute = merge_ml_table(absolute, elisa)

    # Final hard predictor checks.
    actual_signed = [
        c for c in ml_signed.columns
        if str(c).startswith("Delta_")
    ]

    actual_absolute = [
        c for c in ml_absolute.columns
        if str(c).startswith("AbsDelta_")
    ]

    if actual_signed != signed_features:
        raise ValueError("Signed feature list changed unexpectedly during merge.")

    if actual_absolute != absolute_features:
        raise ValueError("Absolute feature list changed unexpectedly during merge.")

    # Ensure signed and absolute sheets have exactly the same target by clone.
    signed_targets = ml_signed.set_index(ID_COL)[TARGET_COL].sort_index()
    absolute_targets = ml_absolute.set_index(ID_COL)[TARGET_COL].sort_index()

    if not signed_targets.equals(absolute_targets):
        raise ValueError(
            "Target mismatch between signed and absolute ML-ready tables."
        )

    # ------------------------------------------------------------
    # Output support tables
    # ------------------------------------------------------------
    label_cols = [ID_COL, TARGET_COL, LABEL_COL]

    for extra in [
        "Comparison_to_Reference_AUC_Ratio",
        "Cross_Reactivity_Ratio_Threshold",
        "Classification_Method",
    ]:
        if extra in elisa.columns:
            label_cols.append(extra)

    label_table = elisa[label_cols].copy()

    qc = pd.DataFrame(
        {
            "Item": [
                "N clones",
                "N class 0",
                "N class 1",
                "N signed structural predictors",
                "N absolute structural predictors",
                "Signed missing predictor values",
                "Absolute missing predictor values",
                "Clone sets matched across inputs",
                "Target source",
                "ELISA used as ML predictor",
            ],
            "Value": [
                len(ml_signed),
                int((ml_signed[TARGET_COL] == 0).sum()),
                int((ml_signed[TARGET_COL] == 1).sum()),
                len(signed_features),
                len(absolute_features),
                int(ml_signed[signed_features].isna().sum().sum()),
                int(ml_absolute[absolute_features].isna().sum().sum()),
                "Yes",
                "Script 08 Experimental_Phenotype",
                "No",
            ],
        }
    )

    data_dictionary = build_data_dictionary(
        signed_features=signed_features,
        absolute_features=absolute_features,
        elisa_columns=elisa.columns.tolist(),
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        ml_signed.to_excel(
            writer,
            sheet_name="ML_Ready_Signed",
            index=False,
        )
        ml_absolute.to_excel(
            writer,
            sheet_name="ML_Ready_Absolute",
            index=False,
        )
        elisa.to_excel(
            writer,
            sheet_name="Experimental_Phenotype",
            index=False,
        )
        label_table.to_excel(
            writer,
            sheet_name="Label_Table",
            index=False,
        )
        data_dictionary.to_excel(
            writer,
            sheet_name="Data_Dictionary",
            index=False,
        )
        qc.to_excel(
            writer,
            sheet_name="QC_Summary",
            index=False,
        )

    print(f"\nSaved:\n{args.output}")

    print("\nFinal dataset:")
    print(f"  Clones:                     {len(ml_signed)}")
    print(f"  Class 0 (Non-CR):           {(ml_signed[TARGET_COL] == 0).sum()}")
    print(f"  Class 1 (CR):               {(ml_signed[TARGET_COL] == 1).sum()}")
    print(f"  Signed structural features: {len(signed_features)}")
    print(f"  Absolute features:          {len(absolute_features)}")

    print("\nIMPORTANT:")
    print("  Target comes directly from script 08.")
    print("  ELISA/AUC/ratio variables are NOT model predictors.")
    print("  Signed X = Delta_* only.")
    print("  Absolute X = AbsDelta_* only.")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR")
        print("=" * 90)
        print(exc)
        sys.exit(1)
