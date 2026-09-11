#!/usr/bin/env python3
"""
07_calculate_paired_deltas.py

Calculate paired Human-vs-Rhesus computational features from the master workbook
created by 06_build_master_workbook.py.

Outputs:
  - Signed_Deltas
  - Absolute_Deltas
  - ML_Signed_Input
  - ML_Absolute_Input
  - Formulas_Column
  - Validation_Report (when --reference-workbook is supplied)

Signed convention:
    Human - Rhesus

Absolute convention:
    abs(Human - Rhesus)

The exact Signed_Deltas schema mirrors the original "CDR Deltas" worksheet.
The ML_Signed_Input schema mirrors the computational columns of the original
ML_Ready worksheet, but EXCLUDES:
  - Cross_Reactive_Target
  - Experimental_Human_AUC
  - Experimental_Rhesus_AUC
  - Experimental_Selectivity

Therefore the ML input generated here contains structural/computational
predictors only. The target can be joined separately at modeling time.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SIGNED_COLUMNS = [
    "Clone",
    "Delta_Atomic_Contacts",
    "Delta_Ab_Interface_N",
    "Delta_Peptide_Interface_N",
    "Delta_HBond_N",
    "Delta_Total_CDR_Interface_N",
    "Delta_Framework_Interface_N",
    "Delta_CDR_Interface_Fraction",
    "Delta_CDR_HBond_N",
    "Delta_HCDR1_Interface_N",
    "Delta_HCDR2_Interface_N",
    "Delta_HCDR3_Interface_N",
    "Delta_LCDR1_Interface_N",
    "Delta_LCDR2_Interface_N",
    "Delta_LCDR3_Interface_N",
    "Delta_HCDR1_Engagement",
    "Delta_HCDR2_Engagement",
    "Delta_HCDR3_Engagement",
    "Delta_HCDR_Engagement",
    "Delta_LCDR1_Engagement",
    "Delta_LCDR2_Engagement",
    "Delta_LCDR3_Engagement",
    "Delta_LCDR_Engagement",
    "Delta_Total_CDR_Engagement",
    "Delta_HCDR1_HBond_N",
    "Delta_HCDR2_HBond_N",
    "Delta_HCDR3_HBond_N",
    "Delta_LCDR1_HBond_N",
    "Delta_LCDR2_HBond_N",
    "Delta_LCDR3_HBond_N",
    "Delta_Variant_Pos7_Contact",
    "Delta_Variant_Pos8_Contact",
    "Delta_Key_Peptide_Residue_Contact",
    "Delta_HADDOCK_Score",
    "Delta_vdW_Energy",
    "Delta_Electrostatic_Energy",
    "Delta_Desolvation_Energy",
    "Delta_Restraint_Violation_Energy",
    "Delta_BSA",
    "Delta_Z_Score",
    "Delta_PRODIGY_DeltaG",
    "Delta_PRODIGY_Kd_M",
    "Delta_PRODIGY_log10Kd",
    "Delta_PRODIGY_Intermolecular_Contacts",
    "Delta_PRODIGY_Charged_Charged",
    "Delta_PRODIGY_Charged_Polar",
    "Delta_PRODIGY_Charged_Apolar",
    "Delta_PRODIGY_Polar_Polar",
    "Delta_PRODIGY_Apolar_Polar",
    "Delta_PRODIGY_Apolar_Apolar",
    "Delta_PRODIGY_Apolar_NIS_pct",
    "Delta_PRODIGY_Charged_NIS_pct",
    "Human_Selectivity_PRODIGY_DeltaG",
    "Human_Selectivity_PRODIGY_log10Kd",
    "Human_Selectivity_HADDOCK_Score",
]

# Exact computational predictors used in the original ML_Ready worksheet.
# ELISA columns and Cross_Reactive_Target are intentionally absent.
ML_SIGNED_FEATURES = [
    "Delta_Atomic_Contacts",
    "Delta_Ab_Interface_N",
    "Delta_HBond_N",
    "Delta_Total_CDR_Interface_N",
    "Delta_CDR_Interface_Fraction",
    "Delta_CDR_HBond_N",
    "Delta_HCDR1_Engagement",
    "Delta_HCDR2_Engagement",
    "Delta_HCDR3_Engagement",
    "Delta_LCDR1_Engagement",
    "Delta_LCDR2_Engagement",
    "Delta_LCDR3_Engagement",
    "Delta_Total_CDR_Engagement",
    "Delta_HCDR1_HBond_N",
    "Delta_HCDR2_HBond_N",
    "Delta_HCDR3_HBond_N",
    "Delta_LCDR1_HBond_N",
    "Delta_LCDR2_HBond_N",
    "Delta_LCDR3_HBond_N",
    "Delta_Variant_Pos7_Contact",
    "Delta_Variant_Pos8_Contact",
    "Delta_HADDOCK_Score",
    "Delta_vdW_Energy",
    "Delta_Electrostatic_Energy",
    "Delta_Desolvation_Energy",
    "Delta_BSA",
    "Delta_PRODIGY_DeltaG",
    "Delta_PRODIGY_log10Kd",
    "Delta_PRODIGY_Intermolecular_Contacts",
    "Delta_PRODIGY_Charged_Charged",
    "Delta_PRODIGY_Charged_Polar",
    "Delta_PRODIGY_Charged_Apolar",
    "Delta_PRODIGY_Polar_Polar",
    "Delta_PRODIGY_Apolar_Polar",
    "Delta_PRODIGY_Apolar_Apolar",
    "Delta_PRODIGY_Apolar_NIS_pct",
    "Delta_PRODIGY_Charged_NIS_pct",
]

FEATURE_MAP = {
    "Delta_Atomic_Contacts": ("master", "ChimeraX_Atomic_Contacts_HeavyAtom"),
    "Delta_Ab_Interface_N": ("master", "Ab_Interface_Residues_4A_N_HeavyAtom"),
    "Delta_Peptide_Interface_N": ("master", "Peptide_Interface_Residues_4A_N_HeavyAtom"),
    "Delta_HBond_N": ("master", "ChimeraX_HBond_N_RawPDB"),
    "Delta_Total_CDR_Interface_N": ("master", "Total_CDR_Interface_N"),
    "Delta_Framework_Interface_N": ("master", "Framework_Interface_N"),
    "Delta_CDR_Interface_Fraction": ("master", "CDR_Interface_Fraction"),
    "Delta_CDR_HBond_N": ("master", "CDR_HBond_Events_N"),
    "Delta_HCDR1_Interface_N": ("master", "HCDR1_Interface_N"),
    "Delta_HCDR2_Interface_N": ("master", "HCDR2_Interface_N"),
    "Delta_HCDR3_Interface_N": ("master", "HCDR3_Interface_N"),
    "Delta_LCDR1_Interface_N": ("master", "LCDR1_Interface_N"),
    "Delta_LCDR2_Interface_N": ("master", "LCDR2_Interface_N"),
    "Delta_LCDR3_Interface_N": ("master", "LCDR3_Interface_N"),
    "Delta_HCDR1_Engagement": ("master", "HCDR1 Engagement %"),
    "Delta_HCDR2_Engagement": ("master", "HCDR2 Engagement %"),
    "Delta_HCDR3_Engagement": ("master", "HCDR3 Engagement %"),
    "Delta_HCDR_Engagement": ("master", "HCDR Engagement %"),
    "Delta_LCDR1_Engagement": ("master", "LCDR1 Engagement %"),
    "Delta_LCDR2_Engagement": ("master", "LCDR2 Engagement %"),
    "Delta_LCDR3_Engagement": ("master", "LCDR3 Engagement %"),
    "Delta_LCDR_Engagement": ("master", "LCDR Engagement %"),
    "Delta_Total_CDR_Engagement": ("master", "Total CDR Engagement %"),
    "Delta_HCDR1_HBond_N": ("master", "HCDR1_HBond_Events_N"),
    "Delta_HCDR2_HBond_N": ("master", "HCDR2_HBond_Events_N"),
    "Delta_HCDR3_HBond_N": ("master", "HCDR3_HBond_Events_N"),
    "Delta_LCDR1_HBond_N": ("master", "LCDR1_HBond_Events_N"),
    "Delta_LCDR2_HBond_N": ("master", "LCDR2_HBond_Events_N"),
    "Delta_LCDR3_HBond_N": ("master", "LCDR3_HBond_Events_N"),
    "Delta_Variant_Pos7_Contact": ("master", "Variant_Pos7_Contact"),
    "Delta_Variant_Pos8_Contact": ("master", "Variant_Pos8_Contact"),
    "Delta_Key_Peptide_Residue_Contact": ("master", "Key_Peptide_Residue_Contact"),
    "Delta_HADDOCK_Score": ("master", "HADDOCK Score"),
    "Delta_vdW_Energy": ("master", "van der Waals Energy"),
    "Delta_Electrostatic_Energy": ("master", "Electrostatic Energy"),
    "Delta_Desolvation_Energy": ("master", "Desolvation Energy"),
    "Delta_Restraint_Violation_Energy": ("master", "Restraint Violation Energy"),
    "Delta_BSA": ("master", "Buried Surface Area (BSA)"),
    "Delta_Z_Score": ("master", "Z-Score"),
    "Delta_PRODIGY_DeltaG": ("prodigy", "PRODIGY_DeltaG"),
    "Delta_PRODIGY_Kd_M": ("prodigy", "PRODIGY_Kd_M"),
    "Delta_PRODIGY_Intermolecular_Contacts": ("prodigy", "Intermolecular_Contacts"),
    "Delta_PRODIGY_Charged_Charged": ("prodigy", "Charged_Charged"),
    "Delta_PRODIGY_Charged_Polar": ("prodigy", "Charged_Polar"),
    "Delta_PRODIGY_Charged_Apolar": ("prodigy", "Charged_Apolar"),
    "Delta_PRODIGY_Polar_Polar": ("prodigy", "Polar_Polar"),
    "Delta_PRODIGY_Apolar_Polar": ("prodigy", "Apolar_Polar"),
    "Delta_PRODIGY_Apolar_Apolar": ("prodigy", "Apolar_Apolar"),
    "Delta_PRODIGY_Apolar_NIS_pct": ("prodigy", "Apolar_NIS_pct"),
    "Delta_PRODIGY_Charged_NIS_pct": ("prodigy", "Charged_NIS_pct"),
}

BINARY_FEATURES = {
    "Variant_Pos7_Contact",
    "Variant_Pos8_Contact",
    "Key_Peptide_Residue_Contact",
}

BINARY_MAP = {
    "yes": 1.0, "no": 0.0,
    "true": 1.0, "false": 0.0,
    "y": 1.0, "n": 0.0,
}

FORMULA_ROWS = [
    ["Feature", "Formula", "Notes"],
    ["All signed Delta_*", "Human - Rhesus", "Direction preserved"],
    ["All absolute AbsDelta_*", "abs(Human - Rhesus)", "Magnitude only"],
    ["Binary contact deltas", "Yes=1, No=0, then Human - Rhesus", "Possible signed values: -1, 0, +1"],
    ["Delta_PRODIGY_log10Kd", "log10(Human Kd) - log10(Rhesus Kd)", "Requires positive Kd values"],
    ["Human_Selectivity_PRODIGY_DeltaG", "-Delta_PRODIGY_DeltaG", "Positive means Human-favoring because more-negative DeltaG is stronger"],
    ["Human_Selectivity_PRODIGY_log10Kd", "-Delta_PRODIGY_log10Kd", "Positive means Human-favoring because lower Kd is stronger"],
    ["Human_Selectivity_HADDOCK_Score", "-Delta_HADDOCK_Score", "Positive means Human-favoring because more-negative HADDOCK score is stronger"],
    ["HADDOCK summary strings", 'First numeric value: "-77.9 +/- 5.2" -> -77.9', "Matches original delta logic"],
    ["Engagement/fraction fields", "Computed from exact counts/lengths by script 06", "Avoids 62.5% -> 62.5 parsing error"],
]


def normalize_clone(value) -> str:
    s = "" if pd.isna(value) else str(value).strip()
    s = re.sub(r"^C24048_LEM_", "", s, flags=re.IGNORECASE)
    return s.upper()

def normalize_condition(value) -> str:
    s = "" if pd.isna(value) else str(value).strip()
    low = s.lower()
    if "human" in low:
        return "Human"
    if "rhesus" in low or "monkey" in low or "nhp" in low:
        return "Rhesus"
    return s

def value_to_number(value, binary=False):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if not s:
        return np.nan

    low = s.lower()
    if binary and low in BINARY_MAP:
        return BINARY_MAP[low]

    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan

    try:
        return float(s)
    except Exception:
        pass

    # central value from HADDOCK strings such as "-77.9 +/- 5.2"
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan

def read_master(path: Path):
    master = pd.read_excel(
        path,
        sheet_name="Docking_Results_CFTRimmunogen",
        dtype=object,
    )
    prodigy = pd.read_excel(
        path,
        sheet_name="Interface_Analysis",
        header=1,
        dtype=object,
    )

    for df in (master, prodigy):
        df["Clone_norm"] = df["Clone"].map(normalize_clone)
        df["Species_norm"] = df["CFTR Species"].map(normalize_condition)

    return master, prodigy

def validate_one_pair(df: pd.DataFrame, label: str):
    counts = df.groupby(["Clone_norm", "Species_norm"]).size()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(f"{label}: duplicate clone/species rows:\n{bad.to_string()}")

    grouped = df.groupby("Clone_norm")["Species_norm"].agg(set)
    incomplete = grouped[grouped.apply(lambda x: not {"Human", "Rhesus"}.issubset(x))]
    if not incomplete.empty:
        raise ValueError(
            f"{label}: clones without one Human and one Rhesus row:\n"
            f"{incomplete.to_string()}"
        )

def row_for(df, clone, species):
    x = df[(df["Clone_norm"] == clone) & (df["Species_norm"] == species)]
    if len(x) != 1:
        raise ValueError(f"Expected exactly one row for {clone}/{species}; found {len(x)}")
    return x.iloc[0]

def compute_signed(master: pd.DataFrame, prodigy: pd.DataFrame) -> pd.DataFrame:
    validate_one_pair(master, "Docking_Results_CFTRimmunogen")
    validate_one_pair(prodigy, "Interface_Analysis")

    master_clones = list(dict.fromkeys(master["Clone_norm"].tolist()))
    prodigy_clones = set(prodigy["Clone_norm"])
    clones = [c for c in master_clones if c in prodigy_clones]

    rows = []

    for clone in clones:
        mh = row_for(master, clone, "Human")
        mr = row_for(master, clone, "Rhesus")
        ph = row_for(prodigy, clone, "Human")
        pr = row_for(prodigy, clone, "Rhesus")

        out = {"Clone": mh["Clone"]}

        for delta_name, (source, source_col) in FEATURE_MAP.items():
            hrow, rrow = (mh, mr) if source == "master" else (ph, pr)
            binary = source_col in BINARY_FEATURES

            hv = value_to_number(hrow[source_col], binary=binary)
            rv = value_to_number(rrow[source_col], binary=binary)

            out[delta_name] = (
                hv - rv
                if not pd.isna(hv) and not pd.isna(rv)
                else np.nan
            )

        hkd = value_to_number(ph["PRODIGY_Kd_M"])
        rkd = value_to_number(pr["PRODIGY_Kd_M"])

        if (
            not pd.isna(hkd)
            and not pd.isna(rkd)
            and hkd > 0
            and rkd > 0
        ):
            out["Delta_PRODIGY_log10Kd"] = math.log10(hkd) - math.log10(rkd)
        else:
            out["Delta_PRODIGY_log10Kd"] = np.nan

        out["Human_Selectivity_PRODIGY_DeltaG"] = -out["Delta_PRODIGY_DeltaG"]
        out["Human_Selectivity_PRODIGY_log10Kd"] = -out["Delta_PRODIGY_log10Kd"]
        out["Human_Selectivity_HADDOCK_Score"] = -out["Delta_HADDOCK_Score"]

        rows.append(out)

    signed = pd.DataFrame(rows)

    for col in SIGNED_COLUMNS:
        if col not in signed.columns:
            signed[col] = np.nan

    return signed[SIGNED_COLUMNS]

def compute_absolute(signed: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Clone": signed["Clone"]})

    # Only actual Delta_* quantities are transformed.
    for col in signed.columns:
        if col.startswith("Delta_"):
            out["Abs" + col] = pd.to_numeric(signed[col], errors="coerce").abs()

    return out

def build_ml_tables(signed: pd.DataFrame, absolute: pd.DataFrame):
    signed_ml = signed[["Clone"] + ML_SIGNED_FEATURES].copy()
    signed_ml = signed_ml.rename(columns={"Clone": "Clone_ID"})

    abs_cols = ["Abs" + c for c in ML_SIGNED_FEATURES]
    abs_ml = absolute[["Clone"] + abs_cols].copy()
    abs_ml = abs_ml.rename(columns={"Clone": "Clone_ID"})

    return signed_ml, abs_ml

def compare_numeric(a, b, atol=1e-9):
    aa = value_to_number(a)
    bb = value_to_number(b)
    if pd.isna(aa) and pd.isna(bb):
        return True, 0.0
    if pd.isna(aa) or pd.isna(bb):
        return False, np.nan
    return abs(aa - bb) <= atol, aa - bb

def validation_report(
    signed: pd.DataFrame,
    reference_workbook: Path,
    excluded: set[str],
    atol: float,
):
    ref = pd.read_excel(
        reference_workbook,
        sheet_name="CDR Deltas",
        dtype=object,
    )
    ref = ref.dropna(axis=0, how="all")
    ref["Clone_norm"] = ref["Clone"].map(normalize_clone)

    cur = signed.copy()
    cur["Clone_norm"] = cur["Clone"].map(normalize_clone)

    reports = []

    for _, crow in cur.iterrows():
        clone = crow["Clone_norm"]

        if clone in excluded:
            reports.append({
                "Clone": clone,
                "Column": "(all)",
                "Reference": "",
                "Current": "",
                "Difference": "",
                "Status": "EXPECTED_CHANGE_EXCLUDED",
            })
            continue

        rr = ref[ref["Clone_norm"] == clone]
        if len(rr) != 1:
            reports.append({
                "Clone": clone,
                "Column": "(row)",
                "Reference": f"{len(rr)} reference rows",
                "Current": "1 current row",
                "Difference": "",
                "Status": "REFERENCE_ROW_PROBLEM",
            })
            continue

        rrow = rr.iloc[0]

        for col in SIGNED_COLUMNS[1:]:
            if col not in rrow.index:
                continue

            ok, diff = compare_numeric(crow[col], rrow[col], atol)
            reports.append({
                "Clone": clone,
                "Column": col,
                "Reference": rrow[col],
                "Current": crow[col],
                "Difference": diff,
                "Status": "MATCH" if ok else "DIFFERENT",
            })

    return pd.DataFrame(reports)

def write_output(
    output: Path,
    signed: pd.DataFrame,
    absolute: pd.DataFrame,
    signed_ml: pd.DataFrame,
    abs_ml: pd.DataFrame,
    validation: pd.DataFrame | None,
):
    output.parent.mkdir(parents=True, exist_ok=True)

    formulas = pd.DataFrame(FORMULA_ROWS[1:], columns=FORMULA_ROWS[0])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        signed.to_excel(writer, sheet_name="Signed_Deltas", index=False)
        absolute.to_excel(writer, sheet_name="Absolute_Deltas", index=False)
        signed_ml.to_excel(writer, sheet_name="ML_Signed_Input", index=False)
        abs_ml.to_excel(writer, sheet_name="ML_Absolute_Input", index=False)
        formulas.to_excel(writer, sheet_name="Formulas_Column", index=False)

        if validation is not None:
            validation.to_excel(writer, sheet_name="Validation_Report", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-workbook", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p.add_argument(
        "--reference-workbook",
        type=Path,
        default=None,
        help="Optional original workbook used only to verify Signed_Deltas.",
    )
    p.add_argument(
        "--exclude-validation-clone",
        action="append",
        default=[],
        help="Clone allowed to differ, e.g. 15E5. May be supplied multiple times.",
    )
    p.add_argument("--validation-atol", type=float, default=1e-9)
    return p.parse_args()

def main():
    args = parse_args()

    if not args.input_workbook.exists():
        raise FileNotFoundError(args.input_workbook)

    master, prodigy = read_master(args.input_workbook)

    signed = compute_signed(master, prodigy)
    absolute = compute_absolute(signed)
    signed_ml, abs_ml = build_ml_tables(signed, absolute)

    validation = None
    if args.reference_workbook:
        if not args.reference_workbook.exists():
            raise FileNotFoundError(args.reference_workbook)
        excluded = {normalize_clone(x) for x in args.exclude_validation_clone}
        validation = validation_report(
            signed,
            args.reference_workbook,
            excluded,
            args.validation_atol,
        )

        n_match = int((validation["Status"] == "MATCH").sum())
        n_diff = int((validation["Status"] == "DIFFERENT").sum())
        print(f"Delta validation: {n_match} matching cells; {n_diff} differing cells.")
        if n_diff:
            print("Review Validation_Report before using the ML input sheets.")

    write_output(
        args.output,
        signed,
        absolute,
        signed_ml,
        abs_ml,
        validation,
    )

    print("\nDELTA WORKBOOK COMPLETE")
    print(f"Saved: {args.output}")
    print(f"Signed_Deltas: {len(signed)} rows x {len(signed.columns)} columns")
    print(f"Absolute_Deltas: {len(absolute)} rows x {len(absolute.columns)} columns")
    print(f"ML_Signed_Input: {len(signed_ml)} rows x {len(signed_ml.columns)} columns")
    print(f"ML_Absolute_Input: {len(abs_ml)} rows x {len(abs_ml.columns)} columns")
    print("Convention: Human - Rhesus")
    print("ELISA/experimental columns: excluded from ML inputs")


if __name__ == "__main__":
    main()
