#!/usr/bin/env python3
"""Batch PRODIGY analysis for paired Human and Rhesus docked PDB complexes.

The script:
1. Reads all PDB files from separate Human and Rhesus input directories.
2. Writes hydrogen-stripped copies of each structure.
3. Runs PRODIGY on the selected partner chains.
4. Parses binding/interface metrics into per-species and combined CSV files.
5. Writes a simple Human/Rhesus pairing-QC table.

No sequences, structures, or project-specific paths are embedded in this script.
PRODIGY must be installed and available as the ``prodigy`` command.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


FIELDNAMES = [
    "Sample_ID",
    "Species",
    "PDB_File",
    "NoH_PDB_File",
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
    "QC_Flag",
    "QC_Message",
]

REQUIRED_PRODIGY_FIELDS = [
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PRODIGY on paired Human and Rhesus docked PDB complexes "
            "and collect binding/interface metrics."
        )
    )
    parser.add_argument(
        "--human-dir",
        type=Path,
        required=True,
        help="Directory containing Human docked PDB complexes.",
    )
    parser.add_argument(
        "--rhesus-dir",
        type=Path,
        required=True,
        help="Directory containing Rhesus docked PDB complexes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which PRODIGY outputs will be written.",
    )
    parser.add_argument(
        "--partner1-chain",
        default="A",
        help="First PRODIGY interaction-partner chain ID (default: A).",
    )
    parser.add_argument(
        "--partner2-chain",
        default="B",
        help="Second PRODIGY interaction-partner chain ID (default: B).",
    )
    parser.add_argument(
        "--prodigy-command",
        default="prodigy",
        help="PRODIGY executable name/path (default: prodigy).",
    )
    return parser.parse_args()


def normalize_sample_id(filename: str, species: str) -> str:
    """Create a species-independent sample ID from a PDB filename.

    Only common terminal Human/Rhesus labels are removed. The function does not
    assume any private clone prefix or project-specific naming convention.
    """
    sample = Path(filename).stem

    suffixes = [
        f"_{species}",
        f"-{species}",
        "_Human",
        "-Human",
        "_Rhesus",
        "-Rhesus",
    ]

    for suffix in suffixes:
        if sample.lower().endswith(suffix.lower()):
            sample = sample[: -len(suffix)]
            break

    return sample


def strip_hydrogens(input_path: Path, output_path: Path) -> int:
    """Write a PDB copy with hydrogen atoms removed and return count removed."""
    removed = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as fin, \
            output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            remove = False

            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip() if len(line) >= 78 else ""

                if element:
                    remove = element.upper() == "H"
                else:
                    atom_name = line[12:16].strip()
                    remove = bool(re.match(r"^[0-9]*H", atom_name, re.IGNORECASE))

            if remove:
                removed += 1
            else:
                fout.write(line)

    return removed


def get_pdb_chains(pdb_path: Path) -> List[str]:
    """Return unique nonblank chain IDs found in ATOM/HETATM records."""
    chains = set()

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                chain = line[21].strip()
                if chain:
                    chains.add(chain)

    return sorted(chains)


def extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else ""


def blank_result(
    sample_id: str,
    species: str,
    filename: str,
    noh_filename: str,
    qc_flag: str,
    message: str = "",
) -> Dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "Sample_ID": sample_id,
            "Species": species,
            "PDB_File": filename,
            "NoH_PDB_File": noh_filename,
            "QC_Flag": qc_flag,
            "QC_Message": message,
        }
    )
    return row


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# PRODIGY execution
# -----------------------------------------------------------------------------

def run_species(
    species: str,
    input_dir: Path,
    output_root: Path,
    partner1_chain: str,
    partner2_chain: str,
    prodigy_command: str,
) -> List[Dict[str, str]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"{species} input directory does not exist: {input_dir}")

    species_output = output_root / species
    noh_dir = species_output / "noH_structures"
    species_output.mkdir(parents=True, exist_ok=True)
    noh_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdb"
    )

    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in: {input_dir}")

    results: List[Dict[str, str]] = []

    print("\n" + "=" * 78)
    print(f"PRODIGY — {species.upper()}")
    print("=" * 78)
    print(f"Input directory: {input_dir}")
    print(f"PDB files found: {len(pdb_files)}")

    for input_path in pdb_files:
        filename = input_path.name
        sample_id = normalize_sample_id(filename, species)
        noh_filename = f"{input_path.stem}_noH.pdb"
        noh_path = noh_dir / noh_filename

        print(f"\n[{species}] {filename}")

        chains = get_pdb_chains(input_path)
        missing_chains = [
            chain for chain in (partner1_chain, partner2_chain)
            if chain not in chains
        ]

        if missing_chains:
            message = (
                f"Expected partner chains {partner1_chain},{partner2_chain}; "
                f"found {','.join(chains) if chains else 'none'}"
            )
            print(f"  SKIPPED: {message}")
            results.append(
                blank_result(
                    sample_id,
                    species,
                    filename,
                    noh_filename,
                    "Fail",
                    message,
                )
            )
            continue

        removed = strip_hydrogens(input_path, noh_path)
        print(f"  Hydrogens removed: {removed}")

        command = [
            prodigy_command,
            str(noh_path),
            "--selection",
            partner1_chain,
            partner2_chain,
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            message = stderr.strip() or "PRODIGY returned a non-zero exit code."
            print(f"  FAILED: {message}")
            results.append(
                blank_result(
                    sample_id,
                    species,
                    filename,
                    noh_filename,
                    "Fail",
                    message,
                )
            )
            continue

        row = {
            "Sample_ID": sample_id,
            "Species": species,
            "PDB_File": filename,
            "NoH_PDB_File": noh_filename,
            "PRODIGY_DeltaG": extract(
                r"Predicted binding affinity.*?:\s*([-+0-9.eE]+)", stdout
            ),
            "PRODIGY_Kd_M": extract(
                r"Predicted dissociation constant.*?:\s*([-+0-9.eE]+)", stdout
            ),
            "Intermolecular_Contacts": extract(
                r"No\.\s*of intermolecular contacts:\s*(\d+)", stdout
            ),
            "Charged_Charged": extract(
                r"No\.\s*of charged-charged contacts:\s*([-+0-9.eE]+)", stdout
            ),
            "Charged_Polar": extract(
                r"No\.\s*of charged-polar contacts:\s*([-+0-9.eE]+)", stdout
            ),
            "Charged_Apolar": extract(
                r"No\.\s*of charged-apolar contacts:\s*([-+0-9.eE]+)", stdout
            ),
            "Polar_Polar": extract(
                r"No\.\s*of polar-polar contacts:\s*([-+0-9.eE]+)", stdout
            ),
            "Apolar_Polar": extract(
                r"No\.\s*of apolar-polar contacts:\s*([-+0-9.eE]+)", stdout
            ),
            "Apolar_Apolar": extract(
                r"No\.\s*of apolar-apolar contacts:\s*([-+0-9.eE]+)", stdout
            ),
            "Apolar_NIS_pct": extract(
                r"Percentage of apolar NIS residues:\s*([-+0-9.]+)", stdout
            ),
            "Charged_NIS_pct": extract(
                r"Percentage of charged NIS residues:\s*([-+0-9.]+)", stdout
            ),
            "QC_Flag": "Pass",
            "QC_Message": "",
        }

        missing = [field for field in REQUIRED_PRODIGY_FIELDS if row[field] == ""]
        if missing:
            row["QC_Flag"] = "Review"
            row["QC_Message"] = "Missing parsed fields: " + ", ".join(missing)

        results.append(row)

        print(
            f"  DeltaG={row['PRODIGY_DeltaG']} | "
            f"Kd={row['PRODIGY_Kd_M']} | "
            f"contacts={row['Intermolecular_Contacts']} | "
            f"QC={row['QC_Flag']}"
        )

    output_csv = species_output / f"PRODIGY_{species}_results_noH.csv"
    write_csv(output_csv, results, FIELDNAMES)
    print(f"\nSaved: {output_csv}")

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if shutil.which(args.prodigy_command) is None and not Path(args.prodigy_command).exists():
        raise RuntimeError(
            f"PRODIGY executable not found: {args.prodigy_command}\n"
            "Activate the PRODIGY environment or pass --prodigy-command with "
            "the executable path."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Human": args.human_dir,
        "Rhesus": args.rhesus_dir,
    }

    all_results: List[Dict[str, str]] = []

    for species, input_dir in datasets.items():
        all_results.extend(
            run_species(
                species=species,
                input_dir=input_dir,
                output_root=args.output_dir,
                partner1_chain=args.partner1_chain,
                partner2_chain=args.partner2_chain,
                prodigy_command=args.prodigy_command,
            )
        )

    combined_csv = args.output_dir / "PRODIGY_Human_Rhesus_results_noH.csv"
    write_csv(combined_csv, all_results, FIELDNAMES)

    species_by_sample: Dict[str, set] = {}
    for row in all_results:
        species_by_sample.setdefault(row["Sample_ID"], set()).add(row["Species"])

    pairing_rows = []
    for sample_id in sorted(species_by_sample):
        species_set = species_by_sample[sample_id]
        human_present = "Human" in species_set
        rhesus_present = "Rhesus" in species_set
        pairing_rows.append(
            {
                "Sample_ID": sample_id,
                "Human_Present": "Yes" if human_present else "No",
                "Rhesus_Present": "Yes" if rhesus_present else "No",
                "Pairing_QC": "Pass" if human_present and rhesus_present else "Review",
            }
        )

    pairing_csv = args.output_dir / "PRODIGY_Human_Rhesus_pairing_QC.csv"
    write_csv(
        pairing_csv,
        pairing_rows,
        ["Sample_ID", "Human_Present", "Rhesus_Present", "Pairing_QC"],
    )

    print("\n" + "=" * 78)
    print("PRODIGY HUMAN + RHESUS ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Structures processed: {len(all_results)}")
    print(f"Combined results: {combined_csv}")
    print(f"Pairing QC: {pairing_csv}")


if __name__ == "__main__":
    main()
