#!/usr/bin/env python3
"""
06_build_master_workbook.py

Build the current computational master workbook from:
  1) HADDOCK/docking-results Excel workbook
  2) ChimeraX corrected heavy-atom interface CSV
  3) PRODIGY Human/Rhesus CSV

The output reproduces the computational organization of the reference workbook:
  - Docking_Results_CFTRimmunogen   (95 columns)
  - Interface_Analysis              (14 columns)
  - Formulas_Column
  - Validation_Report               (only when --reference-workbook is supplied)

Important:
- ELISA/experimental data are NOT used.
- Percentage strings such as "62.5%" are converted to fractions (0.625).
- CDR engagement/fraction fields are recalculated from counts and lengths so
  rounding in display strings does not propagate into delta calculations.
- Current ChimeraX values are preferred for interface/CDR/H-bond fields.
- Current docking workbook values are preferred for HADDOCK/docking fields.
- Optional validation compares all computational values to the original
  reference workbook, while allowing explicitly excluded redocked clones.

Example:
python 06_build_master_workbook.py ^
  --docking-workbook "CFTR_mAb1_ML_Docking_Results.xlsx" ^
  --chimerax-csv "ChimeraX_Human_Rhesus_Interface_CDR_CORRECTED.csv" ^
  --prodigy-csv "PRODIGY_Human_Rhesus_results_noH.csv" ^
  --output "CFTR_Peptide_Master_Workbook.xlsx"
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Exact reference schemas
# ---------------------------------------------------------------------

DOCKING_COLUMNS = [
    "Clone",
    "CFTR Species",
    "CFTR Immunogen Used",
    "Software",
    "HADDOCK Active Residues (Antibody)",
    "HADDOCK Score",
    "Cluster Number",
    "Cluster Size",
    "RMSD",
    "van der Waals Energy",
    "Electrostatic Energy",
    "Desolvation Energy",
    "Restraint Violation Energy",
    "Buried Surface Area (BSA)",
    "Z-Score",
    "PDB_File",
    "ChimeraX_Atomic_Contacts_HeavyAtom",
    "Ab_Interface_Residues_4A_N_HeavyAtom",
    "Ab_Interface_Residues_4A_List_HeavyAtom",
    "Peptide_Interface_Residues_4A_N_HeavyAtom",
    "Peptide_Interface_Residues_4A_List_HeavyAtom",
    "ChimeraX_HBond_N_RawPDB",
    "ChimeraX_HBond_Residues_RawPDB",
    "HCDR1_Interface_N",
    "HCDR1_Interface_List",
    "HCDR2_Interface_N",
    "HCDR2_Interface_List",
    "HCDR3_Interface_N",
    "HCDR3_Interface_List",
    "LCDR1_Interface_N",
    "LCDR1_Interface_List",
    "LCDR2_Interface_N",
    "LCDR2_Interface_List",
    "LCDR3_Interface_N",
    "LCDR3_Interface_List",
    "Total_CDR_Interface_N",
    "Total_CDR_Interface_List",
    "Framework_Interface_N",
    "Framework_Interface_List",
    "CDR_Interface_Fraction",
    "Dominant_CDR",
    "CDR_HBond_Events_N",
    "CDR_HBond_AbResidues_List",
    "HCDR1_HBond_Events_N",
    "HCDR2_HBond_Events_N",
    "HCDR3_HBond_Events_N",
    "LCDR1_HBond_Events_N",
    "LCDR2_HBond_Events_N",
    "LCDR3_HBond_Events_N",
    "Variant_Pos7_Contact",
    "Variant_Pos8_Contact",
    "Key_Peptide_Residue_Contact",
    "Mapping_QC",
    "QC_Flag",
    "HADDOCK Active Residues",
    "HCDR1 Start",
    "HCDR1 End",
    "HCDR1 Length",
    "HCDR1 Interface N",
    "HCDR1 Engagement %",
    "HCDR2 Start",
    "HCDR2 End",
    "HCDR2 Length",
    "HCDR2 Interface N",
    "HCDR2 Engagement %",
    "HCDR3 Start",
    "HCDR3 End",
    "HCDR3 Length",
    "HCDR3 Interface N",
    "HCDR3 Engagement %",
    "Total HCDR Length",
    "HCDR Interface N",
    "HCDR Engagement %",
    "LCDR1 Start",
    "LCDR1 End",
    "LCDR1 Length",
    "LCDR1 Interface N",
    "LCDR1 Engagement %",
    "LCDR2 Start",
    "LCDR2 End",
    "LCDR2 Length",
    "LCDR2 Interface N",
    "LCDR2 Engagement %",
    "LCDR3 Start",
    "LCDR3 End",
    "LCDR3 Length",
    "LCDR3 Interface N",
    "LCDR3 Engagement %",
    "Total LCDR Length",
    "LCDR Interface N",
    "LCDR Engagement %",
    "Total CDR Length",
    "Total CDR Interface N",
    "Total CDR Engagement %",
    "Existing CDR Interface Fraction",
]

PRODIGY_COLUMNS = [
    "Clone",
    "CFTR Species",
    "PDB_File",
    "PRODIGY_DeltaG",
    "PRODIGY_Kd_M",
    "Intermolecular_Contacts",
    "Charged_Charged",
    "Charged_Polar",
    "Charged_Apolar",
    "Polar_Polar",
    "Apolar_Polar",
    "Apolar_Apolar",
    "Apolar_NIS_pct",
    "Charged_NIS_pct",
]

DOCKING_META_COLUMNS = DOCKING_COLUMNS[:16]
INTERFACE_COLUMNS = DOCKING_COLUMNS[16:]

DERIVED_COLUMNS = {
    "CDR_Interface_Fraction",
    "HCDR1 Engagement %",
    "HCDR2 Engagement %",
    "HCDR3 Engagement %",
    "HCDR Engagement %",
    "LCDR1 Engagement %",
    "LCDR2 Engagement %",
    "LCDR3 Engagement %",
    "LCDR Engagement %",
    "Total CDR Engagement %",
    "Existing CDR Interface Fraction",
}

FORMULA_ROWS = [
    ["Field", "Formula logic", "Meaning"],
    ["CDR_Interface_Fraction",
     "Total_CDR_Interface_N / Ab_Interface_Residues_4A_N_HeavyAtom",
     "Fraction of antibody interface residues that are CDR residues"],
    ["HCDR1 Engagement %",
     "HCDR1 Interface N / HCDR1 Length",
     "Fraction of HCDR1 residues participating in the interface"],
    ["HCDR2 Engagement %",
     "HCDR2 Interface N / HCDR2 Length",
     "Fraction of HCDR2 residues participating in the interface"],
    ["HCDR3 Engagement %",
     "HCDR3 Interface N / HCDR3 Length",
     "Fraction of HCDR3 residues participating in the interface"],
    ["HCDR Engagement %",
     "HCDR Interface N / Total HCDR Length",
     "Fraction of all heavy-chain CDR residues participating in the interface"],
    ["LCDR1 Engagement %",
     "LCDR1 Interface N / LCDR1 Length",
     "Fraction of LCDR1 residues participating in the interface"],
    ["LCDR2 Engagement %",
     "LCDR2 Interface N / LCDR2 Length",
     "Fraction of LCDR2 residues participating in the interface"],
    ["LCDR3 Engagement %",
     "LCDR3 Interface N / LCDR3 Length",
     "Fraction of LCDR3 residues participating in the interface"],
    ["LCDR Engagement %",
     "LCDR Interface N / Total LCDR Length",
     "Fraction of all light-chain CDR residues participating in the interface"],
    ["Total CDR Engagement %",
     "Total CDR Interface N / Total CDR Length",
     "Fraction of all six CDR residues participating in the interface"],
    ["Existing CDR Interface Fraction",
     "Total CDR Interface N / Ab_Interface_Residues_4A_N_HeavyAtom",
     "Same biological quantity as CDR_Interface_Fraction"],
    ["Binary contact fields",
     "Yes = 1; No = 0 only during delta calculation",
     "Raw master workbook keeps Yes/No labels"],
    ["HADDOCK summary strings",
     'First/central numeric value used for delta calculations, e.g. "-77.9 +/- 5.2" -> -77.9',
     "Raw master workbook preserves the original summary string"],
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

ID_ALIASES = ["Clone", "Clone_ID", "Clone ID", "Sample_ID", "Sample ID", "Sample"]
COND_ALIASES = ["CFTR Species", "CFTR_Species", "Condition", "Species", "Dataset"]

def normalize_clone(value) -> str:
    """Normalize clone IDs only for joining/validation."""
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

def find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    lower = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
        if a.lower() in lower:
            return lower[a.lower()]
    return None

def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    id_col = find_column(x, ID_ALIASES)
    cond_col = find_column(x, COND_ALIASES)
    if id_col is None:
        raise ValueError(f"Could not find clone/sample ID column. Columns: {list(x.columns)}")
    if cond_col is None:
        raise ValueError(f"Could not find Human/Rhesus condition column. Columns: {list(x.columns)}")
    if id_col != "Clone":
        x = x.rename(columns={id_col: "Clone"})
    if cond_col != "CFTR Species":
        x = x.rename(columns={cond_col: "CFTR Species"})
    x["_join_clone"] = x["Clone"].map(normalize_clone)
    x["_join_species"] = x["CFTR Species"].map(normalize_condition)
    x["CFTR Species"] = x["_join_species"]
    return x

def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")

def detect_header_row(path: Path, sheet_name, required_terms=("Clone", "Sample_ID", "HADDOCK Score")) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=20, dtype=object)
    terms = {t.lower() for t in required_terms}
    for i in range(len(preview)):
        vals = {str(v).strip().lower() for v in preview.iloc[i] if pd.notna(v)}
        if vals & terms:
            return i
    return 0

def read_best_docking_table(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    candidates = []

    for sheet in xls.sheet_names:
        header = detect_header_row(path, sheet)
        df = drop_empty(pd.read_excel(path, sheet_name=sheet, header=header, dtype=object))
        cols = {str(c).strip() for c in df.columns}
        score = 0
        for wanted in [
            "HADDOCK Score",
            "Cluster Number",
            "van der Waals Energy",
            "Electrostatic Energy",
            "Buried Surface Area (BSA)",
        ]:
            if wanted in cols:
                score += 2
        if any(c in cols for c in ID_ALIASES):
            score += 1
        if any(c in cols for c in COND_ALIASES):
            score += 1
        candidates.append((score, sheet, df))

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_sheet, best_df = candidates[0]

    if best_score < 4:
        summary = [(s, sh, list(df.columns)[:12]) for s, sh, df in candidates]
        raise ValueError(
            "Could not identify a docking-results worksheet containing HADDOCK metrics.\n"
            f"Scanned sheets: {summary}"
        )

    print(f"Docking worksheet selected: {best_sheet!r}")
    return standardize_keys(best_df)

def read_csv_standard(path: Path) -> pd.DataFrame:
    return standardize_keys(drop_empty(pd.read_csv(path, dtype=object)))

def percentage_or_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return value

def numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def safe_div(numer, denom):
    n = numeric(numer)
    d = numeric(denom)
    if pd.isna(n) or pd.isna(d) or d == 0:
        return np.nan
    return n / d

def coalesce(row, candidates):
    for c in candidates:
        if c in row.index and not pd.isna(row[c]):
            return row[c]
    return np.nan

def validate_unique(df: pd.DataFrame, label: str):
    dup = df.groupby(["_join_clone", "_join_species"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        raise ValueError(f"{label} has duplicate clone/species rows:\n{dup.to_string()}")

def canonical_docking_name_map():
    """Aliases seen across old and new analysis outputs."""
    return {
        "CFTR Immunogen Used": ["CFTR Immunogen Used", "Immunogen", "Antigen"],
        "Software": ["Software", "Docking Software"],
        "HADDOCK Active Residues (Antibody)": [
            "HADDOCK Active Residues (Antibody)",
            "HADDOCK Active Residues",
            "Active Residues",
        ],
        "HADDOCK Score": ["HADDOCK Score", "HADDOCK_Score"],
        "Cluster Number": ["Cluster Number", "Cluster", "Cluster_Number"],
        "Cluster Size": ["Cluster Size", "Cluster_Size"],
        "RMSD": ["RMSD", "RMSD from overall lowest-energy structure"],
        "van der Waals Energy": ["van der Waals Energy", "Van der Waals Energy", "vdW Energy"],
        "Electrostatic Energy": ["Electrostatic Energy"],
        "Desolvation Energy": ["Desolvation Energy"],
        "Restraint Violation Energy": ["Restraint Violation Energy", "AIR Energy"],
        "Buried Surface Area (BSA)": ["Buried Surface Area (BSA)", "BSA", "Buried Surface Area"],
        "Z-Score": ["Z-Score", "Z Score", "Z_score"],
        "PDB_File": ["PDB_File", "PDB File", "PDB"],
    }

def build_master(docking: pd.DataFrame, chimerax: pd.DataFrame) -> pd.DataFrame:
    validate_unique(docking, "Docking workbook")
    validate_unique(chimerax, "ChimeraX CSV")

    docking_index = docking.set_index(["_join_clone", "_join_species"], drop=False)
    chim_index = chimerax.set_index(["_join_clone", "_join_species"], drop=False)

    common_keys = [k for k in chim_index.index if k in docking_index.index]
    missing_docking = [k for k in chim_index.index if k not in docking_index.index]
    if missing_docking:
        raise ValueError(
            "Some ChimeraX clone/species pairs are missing from the docking workbook:\n"
            + "\n".join(map(str, missing_docking[:20]))
        )

    aliases = canonical_docking_name_map()
    rows = []

    for key in common_keys:
        c = chim_index.loc[key]
        d = docking_index.loc[key]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        if isinstance(d, pd.DataFrame):
            d = d.iloc[0]

        row = {}

        # Preserve the ChimeraX short/current identifier for clean current outputs.
        row["Clone"] = c["Clone"]
        row["CFTR Species"] = c["CFTR Species"]

        for col in DOCKING_META_COLUMNS[2:]:
            if col == "PDB_File":
                # Prefer current ChimeraX PDB filename if present.
                val = coalesce(c, ["PDB_File", "PDB File", "PDB"])
                if pd.isna(val):
                    val = coalesce(d, aliases[col])
            else:
                val = coalesce(d, aliases.get(col, [col]))
            row[col] = val

        for col in INTERFACE_COLUMNS:
            row[col] = coalesce(c, [col])

        # If the current docking workbook has active residues but ChimeraX does not,
        # populate the downstream active-residue column too.
        if pd.isna(row.get("HADDOCK Active Residues")):
            row["HADDOCK Active Residues"] = row.get("HADDOCK Active Residues (Antibody)")

        # Exact derived fractions, replacing rounded percentage strings.
        row["CDR_Interface_Fraction"] = safe_div(
            row.get("Total_CDR_Interface_N"),
            row.get("Ab_Interface_Residues_4A_N_HeavyAtom"),
        )

        for cdr in ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]:
            row[f"{cdr} Engagement %"] = safe_div(
                row.get(f"{cdr} Interface N"),
                row.get(f"{cdr} Length"),
            )

        row["HCDR Engagement %"] = safe_div(
            row.get("HCDR Interface N"),
            row.get("Total HCDR Length"),
        )
        row["LCDR Engagement %"] = safe_div(
            row.get("LCDR Interface N"),
            row.get("Total LCDR Length"),
        )
        row["Total CDR Engagement %"] = safe_div(
            row.get("Total CDR Interface N"),
            row.get("Total CDR Length"),
        )
        row["Existing CDR Interface Fraction"] = safe_div(
            row.get("Total CDR Interface N"),
            row.get("Ab_Interface_Residues_4A_N_HeavyAtom"),
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    # Ensure exact schema/order and preserve missing columns as blank.
    for col in DOCKING_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[DOCKING_COLUMNS]
    return out

def build_prodigy(path: Path) -> pd.DataFrame:
    p = read_csv_standard(path)
    validate_unique(p, "PRODIGY CSV")

    out = pd.DataFrame()
    out["Clone"] = p["Clone"]
    out["CFTR Species"] = p["CFTR Species"]

    aliases = {
        "PDB_File": ["PDB_File", "PDB File", "PDB"],
        "PRODIGY_DeltaG": ["PRODIGY_DeltaG", "DeltaG", "Predicted Binding Affinity"],
        "PRODIGY_Kd_M": ["PRODIGY_Kd_M", "Kd", "Kd_M"],
        "Intermolecular_Contacts": ["Intermolecular_Contacts"],
        "Charged_Charged": ["Charged_Charged"],
        "Charged_Polar": ["Charged_Polar"],
        "Charged_Apolar": ["Charged_Apolar"],
        "Polar_Polar": ["Polar_Polar"],
        "Apolar_Polar": ["Apolar_Polar"],
        "Apolar_Apolar": ["Apolar_Apolar"],
        "Apolar_NIS_pct": ["Apolar_NIS_pct"],
        "Charged_NIS_pct": ["Charged_NIS_pct"],
    }

    for target in PRODIGY_COLUMNS[2:]:
        vals = []
        for _, row in p.iterrows():
            vals.append(coalesce(row, aliases.get(target, [target])))
        out[target] = vals

    return out[PRODIGY_COLUMNS]

def central_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan

def compare_value(a, b, atol=1e-9):
    # text/binary/list fields
    na = central_numeric(a)
    nb = central_numeric(b)
    if not pd.isna(na) and not pd.isna(nb):
        return abs(na - nb) <= atol, na - nb
    sa = "" if pd.isna(a) else str(a).strip()
    sb = "" if pd.isna(b) else str(b).strip()
    return sa == sb, "" if sa == sb else f"{sa} != {sb}"

def load_reference_sheet(path: Path, sheet: str, header=0) -> pd.DataFrame:
    return drop_empty(pd.read_excel(path, sheet_name=sheet, header=header, dtype=object))

def validation_report(
    current_master: pd.DataFrame,
    current_prodigy: pd.DataFrame,
    reference_path: Path,
    exclude_clones: set[str],
    atol: float,
) -> pd.DataFrame:

    ref_master = standardize_keys(
        load_reference_sheet(reference_path, "Docking Results_CFTRimmunogen", 0)
    )
    # Reference Interface Analysis has title row "PRODIGY" then headers.
    ref_prod = standardize_keys(
        load_reference_sheet(reference_path, "Interface Analysis", 1)
    )

    reports = []

    def compare_tables(current, reference, columns, source):
        ci = current.copy()
        ri = reference.copy()
        ci["_join_clone"] = ci["Clone"].map(normalize_clone)
        ci["_join_species"] = ci["CFTR Species"].map(normalize_condition)
        ri["_join_clone"] = ri["Clone"].map(normalize_clone)
        ri["_join_species"] = ri["CFTR Species"].map(normalize_condition)

        ci = ci.set_index(["_join_clone", "_join_species"], drop=False)
        ri = ri.set_index(["_join_clone", "_join_species"], drop=False)

        for key in ci.index:
            clone, species = key
            if clone in exclude_clones:
                reports.append({
                    "Source": source,
                    "Clone": clone,
                    "CFTR Species": species,
                    "Column": "(all)",
                    "Reference": "",
                    "Current": "",
                    "Difference": "",
                    "Status": "EXPECTED_CHANGE_EXCLUDED",
                })
                continue

            if key not in ri.index:
                reports.append({
                    "Source": source, "Clone": clone, "CFTR Species": species,
                    "Column": "(row)", "Reference": "missing",
                    "Current": "present", "Difference": "",
                    "Status": "MISSING_REFERENCE_ROW",
                })
                continue

            crow = ci.loc[key]
            rrow = ri.loc[key]
            if isinstance(crow, pd.DataFrame): crow = crow.iloc[0]
            if isinstance(rrow, pd.DataFrame): rrow = rrow.iloc[0]

            for col in columns:
                if col not in crow.index or col not in rrow.index:
                    continue
                ok, diff = compare_value(crow[col], rrow[col], atol)
                reports.append({
                    "Source": source,
                    "Clone": clone,
                    "CFTR Species": species,
                    "Column": col,
                    "Reference": rrow[col],
                    "Current": crow[col],
                    "Difference": diff,
                    "Status": "MATCH" if ok else "DIFFERENT",
                })

    compare_tables(
        current_master,
        ref_master,
        [c for c in DOCKING_COLUMNS if c not in {
            # filenames may legitimately differ after regeneration
            "PDB_File",
        }],
        "Docking_Results_CFTRimmunogen",
    )
    compare_tables(
        current_prodigy,
        ref_prod,
        PRODIGY_COLUMNS[3:],
        "Interface_Analysis",
    )

    return pd.DataFrame(reports)

def write_excel(
    output: Path,
    master: pd.DataFrame,
    prodigy: pd.DataFrame,
    validation: Optional[pd.DataFrame],
):
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Docking_Results_CFTRimmunogen", index=False)

        # Match the original organization: title row "PRODIGY", then headers.
        prodigy.to_excel(
            writer,
            sheet_name="Interface_Analysis",
            index=False,
            startrow=1,
        )
        ws = writer.book["Interface_Analysis"]
        ws["A1"] = "PRODIGY"

        pd.DataFrame(FORMULA_ROWS[1:], columns=FORMULA_ROWS[0]).to_excel(
            writer,
            sheet_name="Formulas_Column",
            index=False,
        )

        if validation is not None:
            validation.to_excel(writer, sheet_name="Validation_Report", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.font = cell.font.copy(bold=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docking-workbook", required=True, type=Path)
    p.add_argument("--chimerax-csv", required=True, type=Path)
    p.add_argument("--prodigy-csv", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p.add_argument(
        "--reference-workbook",
        type=Path,
        default=None,
        help="Optional original workbook used only for validation.",
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

    for path in [args.docking_workbook, args.chimerax_csv, args.prodigy_csv]:
        if not path.exists():
            raise FileNotFoundError(path)

    docking = read_best_docking_table(args.docking_workbook)
    chimerax = read_csv_standard(args.chimerax_csv)
    prodigy = build_prodigy(args.prodigy_csv)

    master = build_master(docking, chimerax)

    validation = None
    if args.reference_workbook:
        if not args.reference_workbook.exists():
            raise FileNotFoundError(args.reference_workbook)
        excluded = {normalize_clone(x) for x in args.exclude_validation_clone}
        validation = validation_report(
            master,
            prodigy,
            args.reference_workbook,
            excluded,
            args.validation_atol,
        )
        n_diff = int((validation["Status"] == "DIFFERENT").sum())
        n_match = int((validation["Status"] == "MATCH").sum())
        print(f"Validation: {n_match} matching cells; {n_diff} differing cells.")
        if n_diff:
            print("Review Validation_Report before using downstream ML inputs.")

    write_excel(args.output, master, prodigy, validation)

    print("\nMASTER WORKBOOK COMPLETE")
    print(f"Saved: {args.output}")
    print(f"Docking_Results_CFTRimmunogen: {len(master)} rows x {len(master.columns)} columns")
    print(f"Interface_Analysis: {len(prodigy)} rows x {len(prodigy.columns)} columns")
    print("ELISA/experimental data: excluded")


if __name__ == "__main__":
    main()
