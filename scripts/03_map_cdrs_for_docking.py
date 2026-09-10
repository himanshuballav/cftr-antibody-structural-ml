#!/usr/bin/env python3
"""
03_map_cdrs_for_docking.py

Map IMGT DomainGapAlign CDR regions back onto the exact protein sequences
used for structure prediction, then derive PDB residue ranges suitable for
downstream docking setup.

Design goals
------------
- No project-specific paths, sample names, or biological sequences in code.
- Input data stay external/private.
- --check-only performs a read-only validation: no directories or files are created.
- Normal mode writes mapping/QC tables to the requested output directory.

Important
---------
This script parses SAVED TEXT from IMGT DomainGapAlign result pages.
It does not call IMGT/ANARCI over the network.

Expected PDB numbering assumption
---------------------------------
By default, this script assumes the structure predictor retained continuous
residue numbering across the antibody Fv:
    H: 1..H_length
    L: H_length+1..H_length+L_length

If your PDB instead restarts each chain at 1, use --pdb-numbering per-chain.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


CHAIN_SUFFIX_RE = re.compile(r"(?i)([_\-](H|L|HC|LC|VH|VL|heavy|light))$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map IMGT DomainGapAlign CDRs to sequence/PDB residue positions."
    )
    p.add_argument("--heavy-fasta", required=True, type=Path)
    p.add_argument("--light-fasta", required=True, type=Path)
    p.add_argument("--heavy-imgt", required=True, type=Path,
                   help="Saved IMGT DomainGapAlign heavy-chain results text.")
    p.add_argument("--light-imgt", required=True, type=Path,
                   help="Saved IMGT DomainGapAlign light-chain results text.")
    p.add_argument("--output-dir", type=Path,
                   help="Required unless --check-only is used.")
    p.add_argument(
        "--pdb-numbering",
        choices=["continuous", "per-chain"],
        default="continuous",
        help="continuous: H starts at 1 and L follows H; per-chain: both restart at 1.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Read and validate inputs only. Creates/modifies no files.",
    )
    return p.parse_args()


def clean_sample_name(raw_name: str) -> str:
    name = raw_name.split()[0].strip()
    return CHAIN_SUFFIX_RE.sub("", name)


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current_id = None
    parts: List[str] = []

    def flush() -> None:
        nonlocal current_id, parts
        if current_id is None:
            return
        sample = clean_sample_name(current_id)
        seq = "".join(parts).replace(" ", "").replace("\t", "").upper().rstrip("*")
        if not seq:
            raise ValueError(f"{path}: empty sequence for {current_id!r}")
        if "-" in seq:
            raise ValueError(f"{path}: {current_id!r} contains '-' gaps.")
        if "*" in seq:
            raise ValueError(f"{path}: {current_id!r} contains an internal stop '*'.")
        if sample in records:
            raise ValueError(f"{path}: duplicate normalized sample ID {sample!r}.")
        records[sample] = seq

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].strip()
                parts = []
            else:
                if current_id is None:
                    raise ValueError(f"{path}: sequence found before first FASTA header.")
                parts.append(line)
        flush()

    return records


def parse_alignment_rows(text: str) -> Dict[str, List[str]]:
    rows: Dict[str, List[str]] = {}
    chunks = text.split(" Sequence name: ")
    if len(chunks) == 1:
        chunks = text.split("Sequence name: ")

    for chunk in chunks[1:]:
        if "\n" not in chunk:
            continue
        first_line, rest = chunk.split("\n", 1)
        raw_name = first_line.strip()
        sample = clean_sample_name(raw_name)

        m = re.search(re.escape(raw_name) + r"\t([^\t\n]+)\t?\r?\n", rest)
        if not m:
            print(f"WARNING: no alignment row found for {raw_name!r}")
            continue

        tokens = m.group(1).split()
        if len(tokens) != 12:
            print(
                f"WARNING: {raw_name!r}: expected 12 IMGT alignment blocks, "
                f"found {len(tokens)}; skipping."
            )
            continue

        if sample in rows:
            print(f"WARNING: duplicate DomainGapAlign entry for {sample!r}; keeping first.")
            continue
        rows[sample] = tokens

    return rows


def region_sequences(tokens: List[str]) -> List[Tuple[str, str]]:
    strip_gap = lambda s: s.replace(".", "")
    return [
        ("FR1", strip_gap(tokens[0]) + strip_gap(tokens[1])),
        ("CDR1", strip_gap(tokens[2])),
        ("FR2", strip_gap(tokens[3]) + strip_gap(tokens[4])),
        ("CDR2", strip_gap(tokens[5])),
        ("FR3", strip_gap(tokens[6]) + strip_gap(tokens[7]) +
                strip_gap(tokens[8]) + strip_gap(tokens[9])),
        ("CDR3", strip_gap(tokens[10])),
        ("FR4", strip_gap(tokens[11])),
    ]


def map_regions(sample: str, chain: str, regions, real_seq: str):
    rows = []
    notes = []
    cursor = 0

    for region_name, region_seq in regions:
        if not region_seq:
            continue

        idx = real_seq.find(region_seq, cursor)
        if idx == -1:
            notes.append(
                f"{sample} {chain}: could not locate {region_name} from "
                f"sequence position {cursor + 1}; remaining regions skipped."
            )
            break

        if idx != cursor:
            notes.append(
                f"{sample} {chain}: {region_name} starts at {idx + 1}, "
                f"not expected {cursor + 1}; using actual match."
            )

        for offset, aa in enumerate(region_seq):
            rows.append(
                {
                    "Sample": sample,
                    "Chain": chain,
                    "Sequence_Position": idx + offset + 1,
                    "Amino_Acid": aa,
                    "IMGT_Region": f"{chain}{region_name}",
                }
            )

        cursor = idx + len(region_seq)

    return rows, notes


def positions_to_ranges(positions: Iterable[int]) -> str:
    vals = sorted(set(int(x) for x in positions))
    if not vals:
        return ""
    pieces = []
    start = prev = vals[0]
    for pos in vals[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        pieces.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = pos
    pieces.append(f"{start}-{prev}" if start != prev else str(start))
    return "+".join(pieces)


def validate_inputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input is not a file: {path}")


def main() -> int:
    args = parse_args()
    validate_inputs([args.heavy_fasta, args.light_fasta, args.heavy_imgt, args.light_imgt])

    if not args.check_only and args.output_dir is None:
        raise SystemExit("--output-dir is required unless --check-only is used.")

    heavy_seqs = read_fasta(args.heavy_fasta)
    light_seqs = read_fasta(args.light_fasta)
    heavy_align = parse_alignment_rows(
        args.heavy_imgt.read_text(encoding="utf-8", errors="ignore")
    )
    light_align = parse_alignment_rows(
        args.light_imgt.read_text(encoding="utf-8", errors="ignore")
    )

    all_rows = []
    notes = []

    for sample, seq in heavy_seqs.items():
        if sample not in heavy_align:
            notes.append(f"{sample} H: no DomainGapAlign row found.")
            continue
        mapped, local_notes = map_regions(
            sample, "H", region_sequences(heavy_align[sample]), seq
        )
        all_rows.extend(mapped)
        notes.extend(local_notes)

    for sample, seq in light_seqs.items():
        if sample not in light_align:
            notes.append(f"{sample} L: no DomainGapAlign row found.")
            continue
        mapped, local_notes = map_regions(
            sample, "L", region_sequences(light_align[sample]), seq
        )
        all_rows.extend(mapped)
        notes.extend(local_notes)

    if not all_rows:
        raise RuntimeError("No residues were mapped. Check FASTA IDs and DomainGapAlign text.")

    mapping = pd.DataFrame(all_rows)
    h_lengths = {sample: len(seq) for sample, seq in heavy_seqs.items()}
    mapping["H_Length"] = mapping["Sample"].map(h_lengths)

    if mapping["H_Length"].isna().any():
        missing = mapping.loc[mapping["H_Length"].isna(), "Sample"].unique().tolist()
        raise RuntimeError(f"Missing H-chain length for samples: {missing}")

    if args.pdb_numbering == "continuous":
        mapping["PDB_Residue_Number"] = mapping.apply(
            lambda r: int(r["Sequence_Position"])
            if r["Chain"] == "H"
            else int(r["Sequence_Position"]) + int(r["H_Length"]),
            axis=1,
        )
    else:
        mapping["PDB_Residue_Number"] = mapping["Sequence_Position"].astype(int)

    cdr = mapping[mapping["IMGT_Region"].str.contains("CDR", na=False)].copy()

    cdr_summary_rows = []
    cdr_range_rows = []
    for (sample, chain, region), group in cdr.groupby(
        ["Sample", "Chain", "IMGT_Region"], sort=True
    ):
        group = group.sort_values("Sequence_Position")
        seq_positions = group["Sequence_Position"].astype(int).tolist()
        pdb_positions = group["PDB_Residue_Number"].astype(int).tolist()

        cdr_summary_rows.append(
            {
                "Sample": sample,
                "Chain": chain,
                "CDR": region,
                "Sequence_Positions": ",".join(map(str, seq_positions)),
                "Sequence": "".join(group["Amino_Acid"].astype(str)),
            }
        )
        cdr_range_rows.append(
            {
                "Sample": sample,
                "Chain": chain,
                "CDR": region,
                "PDB_Range": positions_to_ranges(pdb_positions),
            }
        )

    cdr_summary = pd.DataFrame(cdr_summary_rows)
    cdr_ranges = pd.DataFrame(cdr_range_rows)

    expected_regions = {"HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"}
    haddock_rows = []

    for sample in sorted(set(heavy_seqs) & set(light_seqs)):
        sub = cdr_ranges[cdr_ranges["Sample"] == sample]
        found = set(sub["CDR"])
        missing = sorted(expected_regions - found)
        if missing:
            notes.append(f"{sample}: missing CDR region(s): {', '.join(missing)}")

        parts = []
        for region in ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]:
            hit = sub[sub["CDR"] == region]
            if not hit.empty:
                parts.append(f"{region[0]}:{hit.iloc[0]['PDB_Range']}")

        haddock_rows.append(
            {"Sample": sample, "HADDOCK_Active_Residues": ",".join(parts)}
        )

    haddock = pd.DataFrame(haddock_rows)

    print("=" * 72)
    print("CDR MAPPING VALIDATION")
    print("=" * 72)
    print(f"Heavy FASTA records:         {len(heavy_seqs)}")
    print(f"Light FASTA records:         {len(light_seqs)}")
    print(f"Heavy DomainGapAlign rows:   {len(heavy_align)}")
    print(f"Light DomainGapAlign rows:   {len(light_align)}")
    print(f"Mapped residue rows:         {len(mapping)}")
    print(f"Mapped CDR residue rows:     {len(cdr)}")
    print(f"Samples with docking ranges: {len(haddock)}")
    print(f"PDB numbering mode:          {args.pdb_numbering}")

    if notes:
        print("\nWarnings/notes:")
        for note in notes:
            print(f"  - {note}")
    else:
        print("\nNo mapping warnings detected.")

    if args.check_only:
        print("\nCHECK-ONLY MODE")
        print("No directories were created and no files were written.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping_out = args.output_dir / "residue_region_mapping.csv"
    cdr_out = args.output_dir / "cdr_residue_mapping.csv"
    summary_out = args.output_dir / "cdr_summary.csv"
    ranges_out = args.output_dir / "cdr_pdb_ranges.csv"
    haddock_out = args.output_dir / "haddock_active_residues.csv"

    mapping.to_csv(mapping_out, index=False)
    cdr.to_csv(cdr_out, index=False)
    cdr_summary.merge(haddock, on="Sample", how="left").to_csv(summary_out, index=False)
    cdr_ranges.merge(haddock, on="Sample", how="left").to_csv(ranges_out, index=False)
    haddock.to_csv(haddock_out, index=False)

    print("\nWritten:")
    for path in [mapping_out, cdr_out, summary_out, ranges_out, haddock_out]:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
