#!/usr/bin/env python3
"""
Nucleotide -> protein translation
=================================

Reads per-sample nucleotide FASTA files from an input folder and
translates them to protein using the standard genetic code.

Expected filename format:
    <prefix>_<sample>_<chain>.fa

where <chain> is:
    H = heavy chain
    L = light chain

Example:
    project_001_H.fa
    project_001_L.fa

Outputs:
    individual_protein/<sample>_<chain>_protein.fasta
    combined_HC_protein.fasta
    combined_LC_protein.fasta

Usage:
    python translate_nucleotide_to_protein.py \
        --input-dir /path/to/private/input_sequences \
        --output-dir /path/to/output
"""

import argparse
import re
from pathlib import Path


# ============================================================
# STANDARD GENETIC CODE
# ============================================================

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ============================================================
# FASTA I/O
# ============================================================

def read_single_record_fasta(path):
    """Read the first FASTA record from a file."""
    header = None
    seq_parts = []
    n_records = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith(">"):
                n_records += 1

                if n_records == 1:
                    header = line[1:].strip()

                elif n_records == 2:
                    print(
                        f"WARNING: {path.name} contains multiple FASTA records; "
                        f"only the first record will be used."
                    )

            elif n_records <= 1:
                seq_parts.append(line.strip())

    if header is None:
        raise ValueError(f"No FASTA record found in {path}")

    return header, "".join(seq_parts).upper()


def write_fasta(records, path):
    """Write a list of (header, sequence) tuples to FASTA."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n")

            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


# ============================================================
# SAMPLE / CHAIN NAME PARSING
# ============================================================

FILE_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?P<sample>[A-Za-z0-9]+)_(?P<chain>H|L)$",
    re.IGNORECASE
)


def parse_sample_and_chain(filename_stem):
    """Extract sample name and heavy/light chain from filename."""
    match = FILE_PATTERN.match(filename_stem)

    if not match:
        return None, None

    return (
        match.group("sample"),
        match.group("chain").upper()
    )


# ============================================================
# TRANSLATION
# ============================================================

def translate_nucleotide(nt_seq, sample, chain):
    """Translate nucleotide sequence in reading frame 1."""
    nt_seq = re.sub(
        r"[^ACGTUacgtu]",
        "",
        nt_seq
    ).upper().replace("U", "T")

    remainder = len(nt_seq) % 3

    if remainder != 0:
        print(
            f"WARNING: {sample}_{chain}: nucleotide length "
            f"({len(nt_seq)}) is not a multiple of 3; "
            f"trimming {remainder} trailing base(s)."
        )

        nt_seq = nt_seq[:len(nt_seq) - remainder]

    protein_chars = []
    has_x = False

    for i in range(0, len(nt_seq), 3):
        codon = nt_seq[i:i + 3]
        aa = CODON_TABLE.get(codon)

        if aa is None:
            aa = "X"
            has_x = True

        if aa == "*":
            break

        protein_chars.append(aa)

    protein = "".join(protein_chars)

    if has_x:
        print(
            f"NOTE: {sample}_{chain}: translated sequence contains 'X'. "
            f"Check the source nucleotide sequence for ambiguous bases."
        )

    return protein


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing nucleotide FASTA files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where translated protein FASTA files will be written"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    individual_dir = (
        args.output_dir / "individual_protein"
    )

    fa_files = (
        sorted(args.input_dir.glob("*.fa"))
        + sorted(args.input_dir.glob("*.fasta"))
    )

    if not fa_files:
        raise FileNotFoundError(
            f"No .fa or .fasta files found in {args.input_dir}"
        )

    heavy = {}
    light = {}

    for fa_file in fa_files:
        sample, chain = parse_sample_and_chain(
            fa_file.stem
        )

        if sample is None:
            print(
                f"WARNING: could not parse sample/chain from "
                f"'{fa_file.name}', skipping."
            )
            continue

        _, nt_seq = read_single_record_fasta(
            fa_file
        )

        protein = translate_nucleotide(
            nt_seq,
            sample,
            chain
        )

        out_header = f"{sample}_{chain}"

        if chain == "H":
            heavy[sample] = (
                out_header,
                protein
            )
        else:
            light[sample] = (
                out_header,
                protein
            )

        write_fasta(
            [(out_header, protein)],
            individual_dir
            / f"{sample}_{chain}_protein.fasta"
        )

    print(
        f"\nTranslated {len(heavy)} heavy-chain and "
        f"{len(light)} light-chain sequences"
    )

    combined_hc = (
        args.output_dir
        / "combined_HC_protein.fasta"
    )

    combined_lc = (
        args.output_dir
        / "combined_LC_protein.fasta"
    )

    write_fasta(
        [heavy[s] for s in sorted(heavy)],
        combined_hc
    )

    write_fasta(
        [light[s] for s in sorted(light)],
        combined_lc
    )

    print(f"Saved: {combined_hc}")
    print(f"Saved: {combined_lc}")
    print(
        f"Individual protein FASTAs: "
        f"{individual_dir}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()