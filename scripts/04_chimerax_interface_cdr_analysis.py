"""
Generic antibody-peptide interface and CDR-engagement analysis for UCSF ChimeraX.

Run INSIDE ChimeraX, for example:
    runscript "C:\\path\\to\\05_chimerax_interface_cdr_analysis.py"

Purpose
-------
For each docked antibody-peptide PDB, this script calculates:
  - heavy-atom inter-chain contact counts
  - antibody and peptide interface residues within a distance cutoff
  - CDR-specific interface participation and engagement percentages
  - framework interface participation
  - hydrogen-bond counts using the original docked PDB
  - CDR-specific hydrogen-bond event counts
  - optional contact flags for user-defined peptide residue positions

Hydrogen handling
-----------------
The original docked PDBs are never modified. A temporary copy with explicit
hydrogen ATOM/HETATM records removed is used for contact and distance-based
interface calculations. The original PDB is re-opened for ChimeraX hydrogen-
bond analysis. Temporary files are deleted after each structure is analyzed.

Required CDR mapping CSV columns
--------------------------------
One identifier column: Sample, Clone, Sample_ID, or Clone_ID
CDR
PDB_Range
Optional: Chain

Expected CDR labels
-------------------
HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3

Expected docked PDB convention
------------------------------
By default:
  Chain A = antibody variable-region model
  Chain B = peptide
Change ANTIBODY_CHAIN and PEPTIDE_CHAIN below if your files use another scheme.
"""

import csv
import glob
import os
import re
import shutil
import tempfile

from chimerax.core.commands import run


# ============================================================
# 1. USER SETTINGS — EDIT THESE BEFORE RUNNING
# ============================================================

# Add one or more datasets. Labels are written to the output CSV.
# Use local/private paths here when running your analysis, but do not commit
# those edited paths back to a public repository.
DATASETS = [
    ("Condition_A", r"C:\path\to\docked_structures\condition_A"),
    ("Condition_B", r"C:\path\to\docked_structures\condition_B"),
]

CDR_RANGES_CSV = r"C:\path\to\cdr_pdb_ranges.csv"
OUTPUT_ROOT = r"C:\path\to\interface_analysis_output"

ANTIBODY_CHAIN = "A"
PEPTIDE_CHAIN = "B"
INTERFACE_DISTANCE = 4.0

MAKE_PNG = True
MAKE_MOVIE = False

# Optional peptide residue positions to flag for interface contact.
# Leave empty for a fully generic run.
KEY_PEPTIDE_POSITIONS = []

# Optional filename suffixes to strip before matching a PDB stem to the
# identifier in the CDR mapping CSV. These are generic examples only.
FILENAME_SUFFIX_REGEXES = [
    r"_condition_a$",
    r"_condition_b$",
]


# ============================================================
# 2. OUTPUT LOCATIONS
# ============================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

combined_csv = os.path.join(
    OUTPUT_ROOT,
    "ChimeraX_Interface_CDR_Analysis.csv",
)

image_folder = os.path.join(OUTPUT_ROOT, "Interface_PNGs")
movie_folder = os.path.join(OUTPUT_ROOT, "Interface_Movies")

if MAKE_PNG:
    os.makedirs(image_folder, exist_ok=True)
if MAKE_MOVIE:
    os.makedirs(movie_folder, exist_ok=True)


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

CDR_ORDER = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]


def normalize_sample_name(value):
    """Normalize a sample identifier without applying project-specific rules."""
    return str(value).strip()


def sample_from_filename(filename):
    """Derive a sample identifier from a PDB filename stem."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    for pattern in FILENAME_SUFFIX_REGEXES:
        stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
    return normalize_sample_name(stem)


def parse_range_text(text):
    """Parse 26-33, 26, H:26-33, etc. into integer start/end values."""
    if text is None:
        raise ValueError("Empty PDB range")

    value = str(text).strip()
    if ":" in value:
        value = value.split(":", 1)[1]

    match = re.fullmatch(r"\s*(-?\d+)\s*(?:-\s*(-?\d+))?\s*", value)
    if not match:
        raise ValueError(f"Cannot parse residue range: {text!r}")

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start

    if end < start:
        start, end = end, start

    return start, end


def residue_range_string(start, end):
    return str(start) if start == end else f"{start}-{end}"


def pct(numerator, denominator):
    if denominator in (None, 0, ""):
        return ""
    return f"{100.0 * float(numerator) / float(denominator):.1f}%"


# ============================================================
# 4. LOAD CDR RANGES
# ============================================================


def load_cdr_ranges(csv_path):
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"CDR mapping CSV not found:\n{csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError("CDR mapping CSV has no header.")

        field_lookup = {name.strip().lower(): name for name in reader.fieldnames}

        id_col = None
        for candidate in ("sample", "clone", "sample_id", "clone_id"):
            if candidate in field_lookup:
                id_col = field_lookup[candidate]
                break

        cdr_col = field_lookup.get("cdr")
        range_col = field_lookup.get("pdb_range")
        chain_col = field_lookup.get("chain")

        if id_col is None or cdr_col is None or range_col is None:
            raise RuntimeError(
                "CDR mapping CSV must contain an identifier column "
                "(Sample or Clone), plus CDR and PDB_Range."
            )

        mapping = {}

        for row in reader:
            sample = normalize_sample_name(row.get(id_col, ""))
            cdr = str(row.get(cdr_col, "")).strip().upper().replace("_", "")

            if cdr not in CDR_ORDER:
                continue

            start, end = parse_range_text(row.get(range_col, ""))
            source_chain = (
                str(row.get(chain_col, "")).strip().upper() if chain_col else ""
            )

            mapping.setdefault(sample, {})[cdr] = {
                "start": start,
                "end": end,
                "length": end - start + 1,
                "source_chain": source_chain,
            }

    return mapping


CDR_MAPPING = load_cdr_ranges(CDR_RANGES_CSV)


# ============================================================
# 5. PDB HYDROGEN HELPERS
# ============================================================


def atom_is_hydrogen(line):
    if not line.startswith(("ATOM", "HETATM")):
        return False

    element = line[76:78].strip()
    if element:
        return element.upper() == "H"

    atom_name = line[12:16].strip()
    return re.match(r"^[0-9]*H", atom_name, re.IGNORECASE) is not None


def count_hydrogens(pdb_path):
    count = 0
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if atom_is_hydrogen(line):
                count += 1
    return count


def strip_hydrogens(input_path, output_path):
    """Write a temporary copy of a PDB without explicit hydrogen records."""
    removed = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if atom_is_hydrogen(line):
                removed += 1
                continue
            dst.write(line)

    return removed


# ============================================================
# 6. CHIMERAX OUTPUT PARSERS
# ============================================================


def parse_residue_file(file_path, chain_id=None):
    residues = []

    if not os.path.exists(file_path):
        return residues

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = re.search(
                r"residue id .*?/([A-Za-z0-9?]+):(-?\d+).*?name\s+([A-Za-z0-9]+)",
                line,
            )
            if not match:
                continue

            chain = match.group(1)
            number = int(match.group(2))
            name = match.group(3)

            if chain_id is None or chain == chain_id:
                residues.append({"chain": chain, "number": number, "name": name})

    return residues


def residue_list_to_string(residues):
    return ",".join(
        f"{residue['chain']}:{residue['name']}{residue['number']}"
        for residue in residues
    )


def count_cross_chain_lines(file_path, chain_a, chain_b):
    if not os.path.exists(file_path):
        return 0

    count = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            has_a = re.search(rf"/{re.escape(chain_a)}:", line) is not None
            has_b = re.search(rf"/{re.escape(chain_b)}:", line) is not None

            if has_a and has_b:
                count += 1

    return count


def get_cross_chain_hbond_lines(file_path, chain_a, chain_b):
    if not os.path.exists(file_path):
        return []

    lines = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if (
                re.search(rf"/{re.escape(chain_a)}:", line)
                and re.search(rf"/{re.escape(chain_b)}:", line)
            ):
                lines.append(line.strip())

    return lines


def antibody_residue_numbers_from_hbond_line(line, antibody_chain):
    matches = re.findall(rf"/{re.escape(antibody_chain)}:(-?\d+)@", line)
    return sorted({int(value) for value in matches})


# ============================================================
# 7. CDR MAPPING HELPERS
# ============================================================


def cdr_for_residue(number, cdr_map):
    hits = []

    for cdr in CDR_ORDER:
        info = cdr_map.get(cdr)
        if info and info["start"] <= number <= info["end"]:
            hits.append(cdr)

    return hits


def annotate_interface_by_cdr(ab_residues, cdr_map):
    result = {}
    cdr_residue_keys = set()

    for cdr in CDR_ORDER:
        info = cdr_map.get(cdr)
        matched = []

        if info:
            matched = [
                residue
                for residue in ab_residues
                if info["start"] <= residue["number"] <= info["end"]
            ]

        result[f"{cdr}_Interface_N"] = len(matched)
        result[f"{cdr}_Interface_List"] = residue_list_to_string(matched)

        for residue in matched:
            cdr_residue_keys.add(
                (residue["chain"], residue["number"], residue["name"])
            )

    total_cdr_residues = [
        residue
        for residue in ab_residues
        if (residue["chain"], residue["number"], residue["name"])
        in cdr_residue_keys
    ]

    framework_residues = [
        residue
        for residue in ab_residues
        if (residue["chain"], residue["number"], residue["name"])
        not in cdr_residue_keys
    ]

    result["Total_CDR_Interface_N"] = len(total_cdr_residues)
    result["Total_CDR_Interface_List"] = residue_list_to_string(total_cdr_residues)
    result["Framework_Interface_N"] = len(framework_residues)
    result["Framework_Interface_List"] = residue_list_to_string(framework_residues)

    result["CDR_Interface_Fraction"] = (
        round(len(total_cdr_residues) / len(ab_residues), 3)
        if ab_residues
        else ""
    )

    counts = {cdr: result[f"{cdr}_Interface_N"] for cdr in CDR_ORDER}
    max_n = max(counts.values()) if counts else 0

    if max_n == 0:
        result["Dominant_CDR"] = "None"
    else:
        winners = [cdr for cdr, count in counts.items() if count == max_n]
        result["Dominant_CDR"] = winners[0] if len(winners) == 1 else ";".join(winners)

    return result


def annotate_hbonds_by_cdr(hbond_lines, cdr_map):
    event_counts = {cdr: 0 for cdr in CDR_ORDER}
    cdr_event_count = 0
    cdr_ab_residue_numbers = set()

    for line in hbond_lines:
        ab_numbers = antibody_residue_numbers_from_hbond_line(
            line,
            ANTIBODY_CHAIN,
        )

        event_cdrs = set()

        for number in ab_numbers:
            for cdr in cdr_for_residue(number, cdr_map):
                event_cdrs.add(cdr)
                cdr_ab_residue_numbers.add(number)

        if event_cdrs:
            cdr_event_count += 1
            for cdr in event_cdrs:
                event_counts[cdr] += 1

    return {
        "CDR_HBond_Events_N": cdr_event_count,
        "CDR_HBond_AbResidues_List": ",".join(
            f"{ANTIBODY_CHAIN}:{number}"
            for number in sorted(cdr_ab_residue_numbers)
        ),
        **{
            f"{cdr}_HBond_Events_N": event_counts[cdr]
            for cdr in CDR_ORDER
        },
    }


def mapping_qc(cdr_map):
    missing = [cdr for cdr in CDR_ORDER if cdr not in cdr_map]
    if missing:
        return "Review: missing " + ",".join(missing)

    ranges = [
        (cdr_map[cdr]["start"], cdr_map[cdr]["end"], cdr)
        for cdr in CDR_ORDER
    ]

    for i in range(len(ranges)):
        start_1, end_1, cdr_1 = ranges[i]
        for j in range(i + 1, len(ranges)):
            start_2, end_2, cdr_2 = ranges[j]
            if max(start_1, start_2) <= min(end_1, end_2):
                return f"Review: overlapping CDR ranges {cdr_1}/{cdr_2}"

    return "Pass"


def build_haddock_active_string(cdr_map):
    pieces = []

    for cdr in CDR_ORDER:
        info = cdr_map.get(cdr)
        if not info:
            continue

        label = "H" if cdr.startswith("H") else "L"
        pieces.append(
            f"{label}:{residue_range_string(info['start'], info['end'])}"
        )

    return ",".join(pieces)


def add_engagement_columns(row, cdr_map):
    for cdr in CDR_ORDER:
        info = cdr_map.get(cdr)
        start = info["start"] if info else ""
        end = info["end"] if info else ""
        length = info["length"] if info else ""
        interface_n = row.get(f"{cdr}_Interface_N", "")

        row[f"{cdr} Start"] = start
        row[f"{cdr} End"] = end
        row[f"{cdr} Length"] = length
        row[f"{cdr} Interface N"] = interface_n
        row[f"{cdr} Engagement %"] = pct(interface_n, length)

    heavy_cdrs = ("HCDR1", "HCDR2", "HCDR3")
    light_cdrs = ("LCDR1", "LCDR2", "LCDR3")

    total_h_len = sum(
        cdr_map[cdr]["length"] for cdr in heavy_cdrs if cdr in cdr_map
    )
    total_l_len = sum(
        cdr_map[cdr]["length"] for cdr in light_cdrs if cdr in cdr_map
    )

    total_h_int = sum(
        int(row.get(f"{cdr}_Interface_N", 0) or 0) for cdr in heavy_cdrs
    )
    total_l_int = sum(
        int(row.get(f"{cdr}_Interface_N", 0) or 0) for cdr in light_cdrs
    )

    total_cdr_len = total_h_len + total_l_len
    total_cdr_int = total_h_int + total_l_int

    row["Total HCDR Length"] = total_h_len
    row["HCDR Interface N"] = total_h_int
    row["HCDR Engagement %"] = pct(total_h_int, total_h_len)

    row["Total LCDR Length"] = total_l_len
    row["LCDR Interface N"] = total_l_int
    row["LCDR Engagement %"] = pct(total_l_int, total_l_len)

    row["Total CDR Length"] = total_cdr_len
    row["Total CDR Interface N"] = total_cdr_int
    row["Total CDR Engagement %"] = pct(total_cdr_int, total_cdr_len)

    fraction = row.get("CDR_Interface_Fraction", "")
    row["Existing CDR Interface Fraction"] = (
        f"{100.0 * float(fraction):.1f}%"
        if fraction not in ("", None)
        else ""
    )


# ============================================================
# 8. CORE ANALYSIS
# ============================================================


def analyze_structure(pdb_path, dataset_label):
    pdb_name = os.path.basename(pdb_path)
    sample_id = sample_from_filename(pdb_name)
    cdr_map = CDR_MAPPING.get(sample_id, {})

    temp_dir = tempfile.mkdtemp(prefix="interface_analysis_")
    no_h_path = os.path.join(temp_dir, "structure_noH.pdb")
    contact_file = os.path.join(temp_dir, "contacts.txt")
    ab_res_file = os.path.join(temp_dir, "antibody_interface_residues.txt")
    pep_res_file = os.path.join(temp_dir, "peptide_interface_residues.txt")
    hbond_file = os.path.join(temp_dir, "hbonds.txt")

    qc_flag = "Pass"
    qc_notes = []

    try:
        raw_h_count = count_hydrogens(pdb_path)
        stripped_h_count = strip_hydrogens(pdb_path, no_h_path)
        no_h_count = count_hydrogens(no_h_path)

        if no_h_count != 0:
            qc_flag = "Review"
            qc_notes.append(
                f"Temporary no-H PDB still contains {no_h_count} hydrogen records."
            )

        map_qc = (
            mapping_qc(cdr_map)
            if cdr_map
            else "Fail: sample not found in CDR mapping CSV"
        )

        if map_qc != "Pass":
            qc_flag = "Review"
            qc_notes.append(map_qc)

        # ----------------------------------------------------
        # A. Heavy-atom structure
        # ----------------------------------------------------
        run(session, "close all")
        run(session, f'open "{no_h_path}"')

        atomic_models = [
            model for model in session.models.list() if hasattr(model, "chains")
        ]

        if not atomic_models:
            raise RuntimeError(f"No atomic model detected in {pdb_path}")

        model = atomic_models[0]
        chain_ids = [chain.chain_id for chain in model.chains]

        if ANTIBODY_CHAIN not in chain_ids:
            qc_flag = "Review"
            qc_notes.append(f"Expected antibody chain {ANTIBODY_CHAIN} missing.")

        if PEPTIDE_CHAIN not in chain_ids:
            qc_flag = "Review"
            qc_notes.append(f"Expected peptide chain {PEPTIDE_CHAIN} missing.")

        # ----------------------------------------------------
        # B. Heavy-atom atomic contacts
        # ----------------------------------------------------
        try:
            run(
                session,
                f'contacts #1/{ANTIBODY_CHAIN} restrict #1/{PEPTIDE_CHAIN} '
                f'reveal false log false saveFile "{contact_file}" '
                f'namingStyle command',
            )
            atomic_contacts = count_cross_chain_lines(
                contact_file,
                ANTIBODY_CHAIN,
                PEPTIDE_CHAIN,
            )
        except Exception as exc:
            atomic_contacts = ""
            qc_flag = "Review"
            qc_notes.append(f"Heavy-atom contact analysis failed: {exc}")

        try:
            run(session, "contacts delete")
        except Exception:
            pass

        # ----------------------------------------------------
        # C. Antibody interface residues
        # ----------------------------------------------------
        try:
            run(session, "select clear")
            run(
                session,
                f"select zone #1/{PEPTIDE_CHAIN} {INTERFACE_DISTANCE} "
                f"#1/{ANTIBODY_CHAIN} residues true",
            )
            run(session, f'info residues sel saveFile "{ab_res_file}"')

            ab_residues = parse_residue_file(ab_res_file, ANTIBODY_CHAIN)
            ab_interface_n = len(ab_residues)
            ab_interface_list = residue_list_to_string(ab_residues)
        except Exception as exc:
            ab_residues = []
            ab_interface_n = ""
            ab_interface_list = ""
            qc_flag = "Review"
            qc_notes.append(f"Antibody interface selection failed: {exc}")

        # ----------------------------------------------------
        # D. Peptide interface residues
        # ----------------------------------------------------
        try:
            run(session, "select clear")
            run(
                session,
                f"select zone #1/{ANTIBODY_CHAIN} {INTERFACE_DISTANCE} "
                f"#1/{PEPTIDE_CHAIN} residues true",
            )
            run(session, f'info residues sel saveFile "{pep_res_file}"')

            pep_residues = parse_residue_file(pep_res_file, PEPTIDE_CHAIN)
            peptide_interface_n = len(pep_residues)
            peptide_interface_list = residue_list_to_string(pep_residues)
        except Exception as exc:
            pep_residues = []
            peptide_interface_n = ""
            peptide_interface_list = ""
            qc_flag = "Review"
            qc_notes.append(f"Peptide interface selection failed: {exc}")

        peptide_numbers = {residue["number"] for residue in pep_residues}
        key_position_contact = (
            "Yes"
            if KEY_PEPTIDE_POSITIONS
            and any(position in peptide_numbers for position in KEY_PEPTIDE_POSITIONS)
            else "No"
            if KEY_PEPTIDE_POSITIONS
            else ""
        )

        # ----------------------------------------------------
        # E. CDR interface mapping
        # ----------------------------------------------------
        cdr_metrics = annotate_interface_by_cdr(ab_residues, cdr_map)

        # ----------------------------------------------------
        # F. H-bonds on ORIGINAL docked PDB
        # ----------------------------------------------------
        run(session, "close all")
        run(session, f'open "{pdb_path}"')

        try:
            run(session, "select clear")
            run(
                session,
                f'hbonds #1/{ANTIBODY_CHAIN} restrict #1/{PEPTIDE_CHAIN} '
                f'reveal false log false saveFile "{hbond_file}" '
                f'namingStyle command',
            )

            hbond_lines = get_cross_chain_hbond_lines(
                hbond_file,
                ANTIBODY_CHAIN,
                PEPTIDE_CHAIN,
            )
            hbond_n = len(hbond_lines)
            hbond_residues = " | ".join(hbond_lines)
        except Exception as exc:
            hbond_lines = []
            hbond_n = ""
            hbond_residues = ""
            qc_flag = "Review"
            qc_notes.append(f"H-bond analysis failed: {exc}")

        cdr_hbond_metrics = annotate_hbonds_by_cdr(hbond_lines, cdr_map)

        # ----------------------------------------------------
        # G. Optional PNG
        # ----------------------------------------------------
        if MAKE_PNG:
            try:
                base_name = os.path.splitext(pdb_name)[0]
                png_path = os.path.join(
                    image_folder,
                    f"{dataset_label}_{base_name}_interface.png",
                )

                run(session, "select clear")
                run(session, "hide #1 atoms")
                run(session, f"cartoon #1/{ANTIBODY_CHAIN}")
                run(session, f"color #1/{ANTIBODY_CHAIN} cornflowerblue")
                run(session, f"show #1/{PEPTIDE_CHAIN} atoms")
                run(session, f"style #1/{PEPTIDE_CHAIN} stick")
                run(session, f"color #1/{PEPTIDE_CHAIN} orange")
                run(session, "select clear")
                run(
                    session,
                    f"select zone #1/{PEPTIDE_CHAIN} {INTERFACE_DISTANCE} "
                    f"#1/{ANTIBODY_CHAIN} residues true",
                )
                run(session, "show sel atoms")
                run(session, "style sel stick")
                run(session, "color sel red")
                run(session, "set bgColor white")
                run(session, "lighting soft")
                run(session, "graphics silhouettes true")
                run(session, "view")
                run(
                    session,
                    f'save "{png_path}" width 2400 height 1800 supersample 3',
                )
            except Exception as exc:
                qc_flag = "Review"
                qc_notes.append(f"PNG generation failed: {exc}")

        # ----------------------------------------------------
        # H. Optional movie
        # ----------------------------------------------------
        if MAKE_MOVIE:
            try:
                base_name = os.path.splitext(pdb_name)[0]
                movie_path = os.path.join(
                    movie_folder,
                    f"{dataset_label}_{base_name}_interface_rotation.mp4",
                )

                run(session, "movie reset")
                run(session, "movie record supersample 2 size 1600,1200")
                run(session, "turn y 2 180")
                run(session, "wait 180")
                run(session, f'movie encode output "{movie_path}" framerate 30')
            except Exception as exc:
                qc_flag = "Review"
                qc_notes.append(f"Movie generation failed: {exc}")

        result = {
            "Sample_ID": sample_id,
            "Dataset": dataset_label,
            "PDB_File": pdb_name,
            "ChimeraX_Atomic_Contacts_HeavyAtom": atomic_contacts,
            "Ab_Interface_Residues_4A_N_HeavyAtom": ab_interface_n,
            "Ab_Interface_Residues_4A_List_HeavyAtom": ab_interface_list,
            "Peptide_Interface_Residues_4A_N_HeavyAtom": peptide_interface_n,
            "Peptide_Interface_Residues_4A_List_HeavyAtom": peptide_interface_list,
            "ChimeraX_HBond_N_RawPDB": hbond_n,
            "ChimeraX_HBond_Residues_RawPDB": hbond_residues,
            **cdr_metrics,
            **cdr_hbond_metrics,
            "Key_Peptide_Position_Contact": key_position_contact,
            "Mapping_QC": map_qc,
            "QC_Flag": qc_flag,
            "HADDOCK Active Residues": build_haddock_active_string(cdr_map),
        }

        add_engagement_columns(result, cdr_map)

        print(f"  Explicit H atoms in original PDB: {raw_h_count}")
        print(f"  H atoms removed in temporary copy: {stripped_h_count}")
        print(f"  H atoms remaining in temporary copy: {no_h_count}")
        if qc_notes:
            print("  QC notes: " + " | ".join(qc_notes))

        return result

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# 9. OUTPUT COLUMNS
# ============================================================

fieldnames = [
    "Sample_ID",
    "Dataset",
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
    "Key_Peptide_Position_Contact",
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


# ============================================================
# 10. RUN DATASETS
# ============================================================

all_results = []

print("\n" + "=" * 80)
print("ANTIBODY-PEPTIDE HEAVY-ATOM INTERFACE + CDR ENGAGEMENT ANALYSIS")
print("=" * 80)
print(f"CDR mapping CSV:\n{CDR_RANGES_CSV}")
print(f"Output root:\n{OUTPUT_ROOT}")

for dataset_label, input_folder in DATASETS:
    print("\n" + "-" * 80)
    print(f"DATASET: {dataset_label}")
    print("-" * 80)

    if not os.path.isdir(input_folder):
        raise RuntimeError(f"Input folder not found:\n{input_folder}")

    pdb_files = sorted(glob.glob(os.path.join(input_folder, "*.pdb")))
    pdb_files = [
        path
        for path in pdb_files
        if "_noH.pdb" not in os.path.basename(path)
    ]

    print(f"Input folder:\n{input_folder}")
    print(f"PDB structures found: {len(pdb_files)}")

    if not pdb_files:
        raise RuntimeError(f"No PDB files found for dataset {dataset_label}.")

    for index, pdb_path in enumerate(pdb_files, start=1):
        print(
            f"\n[{index}/{len(pdb_files)}] "
            f"{dataset_label}: {os.path.basename(pdb_path)}"
        )

        result = analyze_structure(pdb_path, dataset_label)
        all_results.append(result)

        print(
            "  Heavy-atom contacts = "
            f"{result['ChimeraX_Atomic_Contacts_HeavyAtom']}"
        )
        print(
            "  Antibody interface residues = "
            f"{result['Ab_Interface_Residues_4A_N_HeavyAtom']}"
        )
        print(
            "  Peptide interface residues = "
            f"{result['Peptide_Interface_Residues_4A_N_HeavyAtom']}"
        )
        print(f"  H-bonds = {result['ChimeraX_HBond_N_RawPDB']}")
        print(f"  Total CDR interface residues = {result['Total_CDR_Interface_N']}")
        print(f"  Dominant CDR = {result['Dominant_CDR']}")
        print(f"  Mapping QC = {result['Mapping_QC']}")
        print(f"  QC = {result['QC_Flag']}")


# ============================================================
# 11. WRITE CSV OUTPUT
# ============================================================


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(combined_csv, all_results)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

run(session, "close all")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Total structures analyzed: {len(all_results)}")
print(
    "QC Review/Fail rows: "
    f"{sum(1 for row in all_results if row['QC_Flag'] != 'Pass')}"
)
print("\nOutput CSV:")
print(combined_csv)

if MAKE_PNG:
    print("\nPNG folder:")
    print(image_folder)

if MAKE_MOVIE:
    print("\nMovie folder:")
    print(movie_folder)

print("\nContact/interface features use temporary hydrogen-stripped copies.")
print("Hydrogen-bond features use the original docked PDBs.")
print("=" * 80 + "\n")
