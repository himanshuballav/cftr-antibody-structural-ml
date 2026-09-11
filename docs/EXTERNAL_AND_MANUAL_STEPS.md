# External and Manual Pipeline Steps

Some parts of this workflow were performed with external tools or web interfaces rather than as locally executable Python scripts. They are documented here so that the repository reflects the complete analysis workflow without pretending that every step was automated locally.

## Why these steps are documented instead of scripted

The repository contains Python scripts for the parts of the workflow that could be reproduced locally.

A few upstream steps depended on external software, web services, or interactive workflows. Those steps produced inputs that were subsequently consumed by the local Python pipeline.

They should therefore be treated as **external preprocessing or modeling steps**, not as missing Python calculations.

---

## 1. Signal-peptide removal / mature antibody sequence preparation

### Purpose

Antibody heavy- and light-chain protein sequences used for structure prediction were prepared without their N-terminal signal peptides.

The signal-peptide-trimmed sequences were then used consistently for downstream steps such as:

```text
IgFold structure prediction
IMGT/DomainGapAlign CDR annotation
CDR-to-PDB residue mapping
docking preparation
```

The downstream CDR-mapping workflow assumes that the FASTA sequences correspond to the actual sequences used to generate the antibody structures.

### Repository treatment

Signal-peptide removal was an upstream preprocessing step and is not represented as a locally executable Python script in this repository.

A reproducible implementation should record:

```text
- the signal-peptide prediction/annotation tool used
- tool or database version
- date accessed
- cleavage position for each heavy chain
- cleavage position for each light chain
- whether any cleavage calls were manually reviewed
```

Do not commit private antibody sequences to a public repository.

Instead, users should provide their own signal-peptide-trimmed FASTA files to the downstream workflow.

### Required downstream condition

The FASTA files used for CDR mapping should be the same mature sequences used to generate the antibody PDB structures.

This avoids residue-number shifts between sequence annotation and structural analysis.

---

## 2. IMGT / DomainGapAlign annotation

CDR annotation was obtained externally using IMGT/DomainGapAlign.

The downstream mapping code uses the IMGT-defined CDR sequences and maps those literal sequences back onto the actual antibody-chain sequences used for structure prediction.

This is important because insertions/deletions can make simple region-length counting unreliable.

For reproducibility, retain:

```text
- the DomainGapAlign result text
- the corresponding mature heavy-chain FASTA
- the corresponding mature light-chain FASTA
```

Private sequences and raw annotation files should remain outside the public repository unless they have been appropriately sanitized.

---

## 3. HADDOCK docking

### Purpose

Antibody-antigen complexes were generated using HADDOCK rather than by a local Python docking script.

The Python workflow prepares or organizes information needed for docking and later analyzes the resulting complexes, but the actual docking calculation itself was performed externally.

### Docking software

The project used:

```text
HADDOCK 2.4
```

The public repository therefore does **not** include a script that claims to reproduce HADDOCK docking locally.

### Inputs prepared before docking

The docking workflow requires project-specific structural inputs such as:

```text
- antibody structure
- antigen or peptide structure
- chain assignments
- active/interface residue definitions
- docking restraints/settings
```

Antibody CDR residue ranges were mapped from sequence annotation onto the actual antibody PDB numbering before docking.

### External docking step

Conceptually:

```text
mature antibody sequence
        |
        v
antibody structure prediction
        |
        v
CDR annotation and residue mapping
        |
        v
HADDOCK input preparation
        |
        v
EXTERNAL HADDOCK 2.4 docking
        |
        v
ranked docked complexes / docking metrics
        |
        v
local structural interface analysis
```

The exact HADDOCK submission settings used for a project should be recorded with the analysis metadata.

Examples include:

```text
- HADDOCK version
- receptor/ligand chain definitions
- active residues
- passive residue handling
- restraints
- number of generated/refined models
- model/cluster selection rule
- date of submission
```

### Why docking output is an input to this repository

After docking, the resulting complex structures and docking summary metrics are treated as upstream inputs.

Subsequent local steps can then calculate or integrate features such as:

```text
- interface contacts
- hydrogen bonds
- CDR-interface participation
- PRODIGY-derived interaction metrics
- paired Human-versus-Rhesus structural differences
```

The repository should not include private docked structures or unpublished project-specific output unless release is intentional.

---

## 4. ChimeraX-dependent structural analysis

Some structural measurements require UCSF ChimeraX and are executed inside the ChimeraX Python environment rather than with standard local Python.

These scripts should be labeled accordingly in the repository.

A command-line Python interpreter may not have access to the ChimeraX `session` object or ChimeraX-specific modules.

Therefore:

```text
standard Python script != ChimeraX Python script
```

A ChimeraX script should be run from within ChimeraX using the workflow documented for that script.

---

## 5. Local Python pipeline boundary

A useful way to interpret the repository is:

```text
External / interactive steps
----------------------------
sequence preprocessing
signal-peptide trimming
IMGT/DomainGapAlign annotation
HADDOCK docking
some ChimeraX execution

            |
            v

Local scripted pipeline
-----------------------
sequence/input preparation
structure organization
CDR mapping
interface feature extraction
PRODIGY integration
master workbook construction
paired delta calculation
ELISA phenotype processing
ML-ready dataset construction
nested cross-validated classification
```

The goal is not to imply that every scientific step was locally automated. The goal is to document exactly where external tools were used and make the locally reproducible portion of the workflow explicit.

## Suggested citation / software reporting

When publishing or sharing results, report the external tools separately from the local Python analysis and cite the appropriate software or web-service publications.

Exact versions, access dates, and project-specific parameters should be recorded in the analysis notes or methods section.
