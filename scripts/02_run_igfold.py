#!/usr/bin/env python3
"""
Batch antibody structure prediction with IgFold
================================================

Predict antibody Fv structures from paired heavy- and light-chain
protein FASTA files using IgFold.

Intended workflow:

    nucleotide sequences
        -> translation
        -> signal peptide removal
        -> protein input validation
        -> IgFold structure prediction
        -> ANARCI / IMGT numbering

Input
-----
Two multi-record protein FASTA files:

    heavy-chain FASTA
    light-chain FASTA

Heavy and light sequences are paired using their FASTA record IDs
after removal of common chain suffixes such as:

    _H
    _L
    _HC
    _LC
    _VH
    _VL

Outputs
-------
For each antibody/sample:

    <sample>/
        <sample>_IgFold.pdb
        <sample>_per_residue_pRMSD.csv
        <sample>_pRMSD_plot.png

Master files:

    IgFold_summary_metrics.csv
    IgFold_all_residue_pRMSD.csv
    IgFold_run_status.csv

Notes
-----
IgFold's optional Chothia renumbering is disabled.

The original IgFold PDB numbering is preserved. Standardized IMGT
numbering should be applied separately downstream using ANARCI.

No antibody sequences are embedded in this script.

Example
-------

python 02_run_igfold.py \
    --heavy-fasta /path/to/private/cleaned_HC_protein.fasta \
    --light-fasta /path/to/private/cleaned_LC_protein.fasta \
    --output-dir /path/to/private/IgFold_results
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from igfold import IgFoldRunner


# ============================================================
# FASTA HEADER CLEANING
# ============================================================

def clean_sample_name(header):
    """
    Convert H/L FASTA headers to a common sample name.

    Examples
    --------
    sample01_H  -> sample01
    sample01_L  -> sample01
    sample01_VH -> sample01
    sample01_VL -> sample01
    """

    name = header.split()[0]

    name = re.sub(
        r"(?i)([_\-](H|L|HC|LC|VH|VL|heavy|light))$",
        "",
        name,
    )

    return name


# ============================================================
# READ FASTA
# ============================================================

def read_fasta(filepath):
    """
    Read protein FASTA and return:

        sample -> sequence
    """

    sequences = {}

    for record in SeqIO.parse(filepath, "fasta"):

        sample = clean_sample_name(record.id)

        sequence = (
            str(record.seq)
            .upper()
            .replace("-", "")
            .replace(" ", "")
            .replace("*", "")
        )

        if not sample:
            raise RuntimeError(
                f"Could not determine sample name from "
                f"FASTA header: {record.id}"
            )

        if sample in sequences:
            raise RuntimeError(
                f"Duplicate sample name after header cleaning: "
                f"{sample}\nFile: {filepath}"
            )

        if not sequence:
            raise RuntimeError(
                f"Empty sequence detected for sample: {sample}"
            )

        sequences[sample] = sequence

    if not sequences:
        raise RuntimeError(
            f"No FASTA records found in:\n{filepath}"
        )

    return sequences


# ============================================================
# pRMSD HANDLING
# ============================================================

def extract_prmsd_array(igfold_output):
    """
    Convert IgFold pRMSD output to NumPy array and remove
    the batch dimension when present.
    """

    prmsd = (
        igfold_output.prmsd
        .detach()
        .cpu()
        .numpy()
    )

    if prmsd.ndim == 3:

        if prmsd.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected IgFold batch dimension: "
                f"{prmsd.shape}"
            )

        prmsd = prmsd[0]

    return prmsd


def calculate_mean_prmsd(prmsd):
    """
    Calculate mean predicted RMSD for each residue.

    IgFold commonly returns atom-level pRMSD values with
    shape:

        residues x atoms
    """

    if prmsd.ndim == 2:
        return np.mean(
            prmsd,
            axis=1
        )

    if prmsd.ndim == 1:
        return prmsd

    raise RuntimeError(
        f"Unexpected pRMSD dimensions: {prmsd.shape}"
    )


# ============================================================
# PER-RESIDUE TABLE
# ============================================================

def create_residue_table(
    sample,
    heavy_sequence,
    light_sequence,
    prmsd
):
    """
    Build residue-level pRMSD table.

    Residue_Number refers to sequence position within each
    chain and therefore starts at 1 independently for H and L.

    Global_Position is used only for plotting H followed by L.

    This table does NOT assume that sequence numbering equals
    the residue numbering stored in the IgFold PDB file.
    """

    heavy_length = len(
        heavy_sequence
    )

    light_length = len(
        light_sequence
    )

    expected_length = (
        heavy_length
        + light_length
    )

    mean_prmsd = calculate_mean_prmsd(
        prmsd
    )

    if len(mean_prmsd) != expected_length:

        print(
            f"WARNING: {sample}: pRMSD length "
            f"does not match input sequence."
        )

        print(
            f"Expected: {expected_length}"
        )

        print(
            f"Observed: {len(mean_prmsd)}"
        )

    n = min(
        len(mean_prmsd),
        expected_length
    )

    chain = (
        ["H"] * heavy_length
        + ["L"] * light_length
    )

    residue_number = (
        list(
            range(
                1,
                heavy_length + 1
            )
        )
        +
        list(
            range(
                1,
                light_length + 1
            )
        )
    )

    amino_acid = (
        list(heavy_sequence)
        + list(light_sequence)
    )

    global_position = list(
        range(
            1,
            expected_length + 1
        )
    )

    df = pd.DataFrame(
        {
            "Sample": [sample] * n,
            "Chain": chain[:n],
            "Sequence_Position": residue_number[:n],
            "Global_Position": global_position[:n],
            "AA": amino_acid[:n],
            "Mean_pRMSD_A": mean_prmsd[:n],
        }
    )

    # --------------------------------------------------------
    # Atom-level pRMSD
    # --------------------------------------------------------

    if prmsd.ndim == 2:

        atom_count = prmsd.shape[1]

        atom_names = [
            "N",
            "CA",
            "C",
            "CB",
        ]

        for index, atom_name in enumerate(
            atom_names
        ):

            if atom_count > index:

                df[
                    f"pRMSD_{atom_name}_A"
                ] = prmsd[:n, index]

    return df


# ============================================================
# pRMSD PLOT
# ============================================================

def create_prmsd_plot(
    sample,
    df,
    heavy_length,
    light_length,
    output_file
):

    heavy_df = df[
        df["Chain"] == "H"
    ]

    light_df = df[
        df["Chain"] == "L"
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        heavy_df["Global_Position"],
        heavy_df["Mean_pRMSD_A"],
        label="Heavy chain",
        linewidth=1.5,
    )

    plt.plot(
        light_df["Global_Position"],
        light_df["Mean_pRMSD_A"],
        label="Light chain",
        linewidth=1.5,
    )

    plt.axvline(
        heavy_length + 0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Residue position (H followed by L)"
    )

    plt.ylabel(
        "Predicted RMSD (Å)"
    )

    plt.title(
        f"{sample} - IgFold per-residue predicted RMSD"
    )

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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
        help="Heavy-chain protein FASTA",
    )

    parser.add_argument(
        "--light-fasta",
        required=True,
        type=Path,
        help="Light-chain protein FASTA",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for IgFold outputs",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check inputs
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
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load sequences
    # --------------------------------------------------------

    print(
        "\nReading antibody sequences..."
    )

    heavy_sequences = read_fasta(
        args.heavy_fasta
    )

    light_sequences = read_fasta(
        args.light_fasta
    )

    print(
        f"Heavy-chain sequences: "
        f"{len(heavy_sequences)}"
    )

    print(
        f"Light-chain sequences: "
        f"{len(light_sequences)}"
    )

    # --------------------------------------------------------
    # Pair H/L
    # --------------------------------------------------------

    heavy_names = set(
        heavy_sequences
    )

    light_names = set(
        light_sequences
    )

    paired_samples = sorted(
        heavy_names
        & light_names
    )

    heavy_only = sorted(
        heavy_names
        - light_names
    )

    light_only = sorted(
        light_names
        - heavy_names
    )

    if heavy_only:

        print(
            "\nWARNING: heavy chain present "
            "without matching light chain:"
        )

        for sample in heavy_only:
            print(
                f"  {sample}"
            )

    if light_only:

        print(
            "\nWARNING: light chain present "
            "without matching heavy chain:"
        )

        for sample in light_only:
            print(
                f"  {sample}"
            )

    if not paired_samples:

        raise RuntimeError(
            "No paired heavy/light samples detected."
        )

    print(
        f"\nPaired antibodies: "
        f"{len(paired_samples)}"
    )

    # --------------------------------------------------------
    # Print lengths
    # --------------------------------------------------------

    print(
        "\nSequence lengths:"
    )

    print(
        "-----------------"
    )

    for sample in paired_samples:

        print(
            f"{sample:15s} "
            f"H={len(heavy_sequences[sample]):3d} aa  "
            f"L={len(light_sequences[sample]):3d} aa"
        )

    # --------------------------------------------------------
    # Load IgFold once
    # --------------------------------------------------------

    print(
        "\nLoading IgFold models..."
    )

    igfold = IgFoldRunner()

    # --------------------------------------------------------
    # Master outputs
    # --------------------------------------------------------

    all_residue_tables = []
    summary_rows = []

    # --------------------------------------------------------
    # Run each antibody
    # --------------------------------------------------------

    for index, sample in enumerate(
        paired_samples,
        start=1
    ):

        print(
            "\n============================================"
        )

        print(
            f"[{index}/{len(paired_samples)}] "
            f"Running {sample}"
        )

        print(
            "============================================"
        )

        heavy_seq = (
            heavy_sequences[sample]
        )

        light_seq = (
            light_sequences[sample]
        )

        heavy_length = len(
            heavy_seq
        )

        light_length = len(
            light_seq
        )

        total_length = (
            heavy_length
            + light_length
        )

        sequences = {
            "H": heavy_seq,
            "L": light_seq,
        }

        sample_dir = (
            args.output_dir
            / sample
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdb_file = (
            sample_dir
            / f"{sample}_IgFold.pdb"
        )

        # ----------------------------------------------------
        # Run IgFold
        # ----------------------------------------------------

        try:

            output = igfold.fold(
                str(pdb_file),
                sequences=sequences,

                # Direct IgFold prediction without
                # OpenMM/PyRosetta refinement.
                do_refine=False,

                # Do not apply IgFold/Chothia numbering.
                # IMGT numbering is performed separately.
                do_renum=False,
            )

        except Exception as exc:

            print(
                f"ERROR running IgFold for "
                f"{sample}: {exc}"
            )

            summary_rows.append(
                {
                    "Sample": sample,
                    "Status": "FAILED",
                    "Heavy_Length": heavy_length,
                    "Light_Length": light_length,
                    "Error": str(exc),
                }
            )

            continue

        # ----------------------------------------------------
        # Extract pRMSD
        # ----------------------------------------------------

        try:

            prmsd = extract_prmsd_array(
                output
            )

            print(
                f"Raw pRMSD shape: "
                f"{prmsd.shape}"
            )

        except Exception as exc:

            print(
                f"ERROR extracting pRMSD for "
                f"{sample}: {exc}"
            )

            summary_rows.append(
                {
                    "Sample": sample,
                    "Status": (
                        "PDB_SUCCESS_pRMSD_FAILED"
                    ),
                    "Heavy_Length": heavy_length,
                    "Light_Length": light_length,
                    "PDB_File": str(pdb_file),
                    "Error": str(exc),
                }
            )

            continue

        # ----------------------------------------------------
        # Residue table
        # ----------------------------------------------------

        df = create_residue_table(
            sample=sample,
            heavy_sequence=heavy_seq,
            light_sequence=light_seq,
            prmsd=prmsd,
        )

        residue_csv = (
            sample_dir
            / f"{sample}_per_residue_pRMSD.csv"
        )

        df.to_csv(
            residue_csv,
            index=False,
        )

        all_residue_tables.append(
            df
        )

        # ----------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------

        heavy_df = df[
            df["Chain"] == "H"
        ]

        light_df = df[
            df["Chain"] == "L"
        ]

        summary_rows.append(
            {
                "Sample": sample,
                "Status": "SUCCESS",
                "Heavy_Length": heavy_length,
                "Light_Length": light_length,
                "Total_Length": total_length,

                "Mean_pRMSD_A":
                    df["Mean_pRMSD_A"].mean(),

                "Median_pRMSD_A":
                    df["Mean_pRMSD_A"].median(),

                "SD_pRMSD_A":
                    df["Mean_pRMSD_A"].std(),

                "Min_pRMSD_A":
                    df["Mean_pRMSD_A"].min(),

                "Max_pRMSD_A":
                    df["Mean_pRMSD_A"].max(),

                "Heavy_Mean_pRMSD_A":
                    heavy_df[
                        "Mean_pRMSD_A"
                    ].mean(),

                "Heavy_Median_pRMSD_A":
                    heavy_df[
                        "Mean_pRMSD_A"
                    ].median(),

                "Light_Mean_pRMSD_A":
                    light_df[
                        "Mean_pRMSD_A"
                    ].mean(),

                "Light_Median_pRMSD_A":
                    light_df[
                        "Mean_pRMSD_A"
                    ].median(),

                "PDB_File":
                    str(pdb_file),

                "Residue_CSV":
                    str(residue_csv),
            }
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_file = (
            sample_dir
            / f"{sample}_pRMSD_plot.png"
        )

        create_prmsd_plot(
            sample=sample,
            df=df,
            heavy_length=heavy_length,
            light_length=light_length,
            output_file=plot_file,
        )

        print(
            f"Mean pRMSD: "
            f"{df['Mean_pRMSD_A'].mean():.3f} Å"
        )

        print(
            f"Saved PDB: {pdb_file}"
        )

        print(
            f"Saved residue metrics: "
            f"{residue_csv}"
        )

        print(
            f"Saved plot: {plot_file}"
        )

    # ========================================================
    # MASTER SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        args.output_dir
        / "IgFold_summary_metrics.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # MASTER PER-RESIDUE TABLE
    # ========================================================

    master_residue_file = None

    if all_residue_tables:

        master_residue_df = pd.concat(
            all_residue_tables,
            ignore_index=True,
        )

        master_residue_file = (
            args.output_dir
            / "IgFold_all_residue_pRMSD.csv"
        )

        master_residue_df.to_csv(
            master_residue_file,
            index=False,
        )

    # ========================================================
    # RUN STATUS
    # ========================================================

    status_file = (
        args.output_dir
        / "IgFold_run_status.csv"
    )

    if not summary_df.empty:

        status_columns = [
            column
            for column in [
                "Sample",
                "Status",
                "Heavy_Length",
                "Light_Length",
                "Error",
            ]
            if column
            in summary_df.columns
        ]

        summary_df[
            status_columns
        ].to_csv(
            status_file,
            index=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if not summary_df.empty:

        n_success = (
            summary_df["Status"]
            .eq("SUCCESS")
            .sum()
        )

        n_failed = (
            summary_df["Status"]
            .ne("SUCCESS")
            .sum()
        )

    else:

        n_success = 0
        n_failed = 0

    print(
        "\n============================================"
    )

    print(
        "IgFold BATCH RUN FINISHED"
    )

    print(
        "============================================"
    )

    print(
        f"Successful models: {n_success}"
    )

    print(
        f"Failed models: {n_failed}"
    )

    print(
        f"Total paired samples: "
        f"{len(paired_samples)}"
    )

    print(
        f"\nMaster summary:\n"
        f"{summary_file}"
    )

    if master_residue_file:

        print(
            f"\nMaster per-residue table:\n"
            f"{master_residue_file}"
        )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()