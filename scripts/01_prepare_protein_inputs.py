#!/usr/bin/env python3
"""
Prepare antibody protein inputs
===============================

Validates and prepares signal-peptide-trimmed heavy- and light-chain
protein FASTA files for downstream antibody structure prediction and
numbering workflows.

This script is intended to run AFTER signal peptide annotation/removal.

The script:

1. Reads heavy- and light-chain protein FASTA files.
2. Normalizes FASTA headers to sample IDs.
3. Checks that each sample has both a heavy and light chain.
4. Checks sequences for invalid amino-acid characters.
5. Removes terminal stop symbols (*) if present.
6. Reports sequence lengths.
7. Writes cleaned heavy- and light-chain FASTA files.
8. Writes a paired H/L FASTA file for downstream workflows.
9. Writes a QC summary table WITHOUT amino-acid sequences.

No antibody sequences are stored directly in this script.

Example
-------

python 01_prepare_protein_inputs.py \
    --heavy-fasta /path/to/private/heavy_trimmed.fasta \
    --light-fasta /path/to/private/light_trimmed.fasta \
    --output-dir /path/to/output
"""

import argparse
import csv
import re
from pathlib import Path


# ============================================================
# ALLOWED AMINO ACIDS
# ============================================================

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Allow X because some sequencing datasets may contain
# unresolved/ambiguous residues.
ALLOWED_AA = STANDARD_AA | {"X"}


# ============================================================
# FASTA I/O
# ============================================================

def read_fasta(path):
    """
    Read a multi-record FASTA file.

    Returns
    -------
    list of tuple
        [(header, sequence), ...]
    """

    records = []
    header = None
    seq_parts = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append(
                        (header, "".join(seq_parts))
                    )

                header = line[1:].strip()
                seq_parts = []

            else:
                if header is None:
                    raise ValueError(
                        f"Sequence encountered before FASTA header in {path}"
                    )

                seq_parts.append(line)

    if header is not None:
        records.append(
            (header, "".join(seq_parts))
        )

    if not records:
        raise ValueError(
            f"No FASTA records found in {path}"
        )

    return records


def write_fasta(records, path):
    """
    Write FASTA records.

    records should contain:
        [(header, sequence), ...]
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for header, sequence in records:

            f.write(f">{header}\n")

            for i in range(
                0,
                len(sequence),
                60
            ):
                f.write(
                    sequence[i:i + 60] + "\n"
                )


# ============================================================
# HEADER CLEANING
# ============================================================

def clean_sample_id(header):
    """
    Convert a FASTA header into a standardized sample ID.

    Only the first whitespace-separated field is used.

    Common terminal chain labels are removed:

        _H
        _L
        _HC
        _LC
        _VH
        _VL
        -H
        -L

    Examples
    --------
    sample01_H  -> sample01
    sample01_L  -> sample01
    sample01_VH -> sample01
    """

    sample = header.split()[0]

    sample = re.sub(
        r"(?i)([_\-](H|L|HC|LC|VH|VL|heavy|light))$",
        "",
        sample
    )

    return sample


# ============================================================
# SEQUENCE CLEANING / VALIDATION
# ============================================================

def clean_protein_sequence(sequence):
    """
    Normalize a protein sequence.

    Removes:
    - whitespace
    - gap characters
    - terminal stop symbols

    Converts sequence to uppercase.
    """

    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    sequence = sequence.replace("-", "")

    sequence = sequence.upper()

    # Remove terminal stop codon symbol if present
    sequence = sequence.rstrip("*")

    return sequence


def validate_sequence(sequence):
    """
    Return invalid amino-acid characters.

    X is permitted but reported separately.
    """

    observed = set(sequence)

    invalid = sorted(
        observed - ALLOWED_AA
    )

    return invalid


# ============================================================
# LOAD CHAIN FASTA
# ============================================================

def load_chain_fasta(path, chain):
    """
    Read, clean, and validate one chain FASTA.

    Returns
    -------
    dict
        sample_id -> sequence
    """

    records = read_fasta(path)

    sequences = {}

    for header, raw_sequence in records:

        sample = clean_sample_id(header)

        sequence = clean_protein_sequence(
            raw_sequence
        )

        if not sample:
            raise ValueError(
                f"Could not determine sample ID from header: {header}"
            )

        if sample in sequences:
            raise ValueError(
                f"Duplicate {chain}-chain sample ID: {sample}"
            )

        if not sequence:
            raise ValueError(
                f"{sample}_{chain} has an empty protein sequence."
            )

        invalid = validate_sequence(
            sequence
        )

        if invalid:
            raise ValueError(
                f"{sample}_{chain} contains invalid amino-acid "
                f"characters: {', '.join(invalid)}"
            )

        sequences[sample] = sequence

    return sequences


# ============================================================
# QC SUMMARY
# ============================================================

def write_qc_summary(
    samples,
    heavy,
    light,
    output_path
):
    """
    Write sequence metadata without storing the actual sequences.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Sample",
            "Heavy_Present",
            "Light_Present",
            "Heavy_Length",
            "Light_Length",
            "Heavy_Contains_X",
            "Light_Contains_X",
            "Pair_Status"
        ])

        for sample in samples:

            h_seq = heavy.get(sample)
            l_seq = light.get(sample)

            h_present = h_seq is not None
            l_present = l_seq is not None

            if h_present and l_present:
                status = "PAIRED"

            elif h_present:
                status = "MISSING_LIGHT"

            else:
                status = "MISSING_HEAVY"

            writer.writerow([
                sample,
                h_present,
                l_present,
                len(h_seq) if h_seq else "",
                len(l_seq) if l_seq else "",
                "X" in h_seq if h_seq else "",
                "X" in l_seq if l_seq else "",
                status
            ])


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--heavy-fasta",
        required=True,
        type=Path,
        help=(
            "Signal-peptide-trimmed heavy-chain "
            "protein FASTA"
        )
    )

    parser.add_argument(
        "--light-fasta",
        required=True,
        type=Path,
        help=(
            "Signal-peptide-trimmed light-chain "
            "protein FASTA"
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for cleaned output files"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not args.heavy_fasta.exists():
        raise FileNotFoundError(
            f"Heavy-chain FASTA not found:\n"
            f"{args.heavy_fasta}"
        )

    if not args.light_fasta.exists():
        raise FileNotFoundError(
            f"Light-chain FASTA not found:\n"
            f"{args.light_fasta}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load sequences
    # --------------------------------------------------------

    print("\nReading protein FASTA files...")

    heavy = load_chain_fasta(
        args.heavy_fasta,
        "H"
    )

    light = load_chain_fasta(
        args.light_fasta,
        "L"
    )

    print(
        f"Heavy-chain records: {len(heavy)}"
    )

    print(
        f"Light-chain records: {len(light)}"
    )

    # --------------------------------------------------------
    # Compare H/L sample sets
    # --------------------------------------------------------

    heavy_samples = set(heavy)
    light_samples = set(light)

    all_samples = sorted(
        heavy_samples | light_samples
    )

    paired_samples = sorted(
        heavy_samples & light_samples
    )

    missing_light = sorted(
        heavy_samples - light_samples
    )

    missing_heavy = sorted(
        light_samples - heavy_samples
    )

    print(
        f"Paired H/L samples: {len(paired_samples)}"
    )

    if missing_light:
        print(
            "\nWARNING: samples missing light chains:"
        )

        for sample in missing_light:
            print(
                f"  {sample}"
            )

    if missing_heavy:
        print(
            "\nWARNING: samples missing heavy chains:"
        )

        for sample in missing_heavy:
            print(
                f"  {sample}"
            )

    # --------------------------------------------------------
    # Warn about ambiguous residues
    # --------------------------------------------------------

    for sample in paired_samples:

        if "X" in heavy[sample]:
            print(
                f"WARNING: {sample}_H contains "
                f"ambiguous residue X"
            )

        if "X" in light[sample]:
            print(
                f"WARNING: {sample}_L contains "
                f"ambiguous residue X"
            )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    clean_heavy_path = (
        args.output_dir
        / "cleaned_HC_protein.fasta"
    )

    clean_light_path = (
        args.output_dir
        / "cleaned_LC_protein.fasta"
    )

    paired_path = (
        args.output_dir
        / "paired_HL_protein.fasta"
    )

    qc_path = (
        args.output_dir
        / "protein_input_QC.csv"
    )

    # --------------------------------------------------------
    # Write cleaned H FASTA
    # --------------------------------------------------------

    heavy_records = [
        (
            f"{sample}_H",
            heavy[sample]
        )
        for sample in paired_samples
    ]

    write_fasta(
        heavy_records,
        clean_heavy_path
    )

    # --------------------------------------------------------
    # Write cleaned L FASTA
    # --------------------------------------------------------

    light_records = [
        (
            f"{sample}_L",
            light[sample]
        )
        for sample in paired_samples
    ]

    write_fasta(
        light_records,
        clean_light_path
    )

    # --------------------------------------------------------
    # Write paired H/L FASTA
    #
    # Each sample contributes two independent FASTA records:
    #
    # >sample_H
    # ...
    #
    # >sample_L
    # ...
    #
    # Heavy and light sequences are NOT concatenated.
    # --------------------------------------------------------

    paired_records = []

    for sample in paired_samples:

        paired_records.append(
            (
                f"{sample}_H",
                heavy[sample]
            )
        )

        paired_records.append(
            (
                f"{sample}_L",
                light[sample]
            )
        )

    write_fasta(
        paired_records,
        paired_path
    )

    # --------------------------------------------------------
    # QC table
    # --------------------------------------------------------

    write_qc_summary(
        all_samples,
        heavy,
        light,
        qc_path
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\nPreparation complete.")

    print(
        f"\nClean heavy FASTA:\n"
        f"  {clean_heavy_path}"
    )

    print(
        f"\nClean light FASTA:\n"
        f"  {clean_light_path}"
    )

    print(
        f"\nPaired H/L FASTA:\n"
        f"  {paired_path}"
    )

    print(
        f"\nQC summary:\n"
        f"  {qc_path}"
    )

    print(
        f"\nSamples ready for downstream analysis: "
        f"{len(paired_samples)}"
    )

    if missing_heavy or missing_light:

        print(
            "\nNOTE: Only samples containing BOTH heavy and "
            "light chains were written to the cleaned "
            "downstream FASTA files."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()