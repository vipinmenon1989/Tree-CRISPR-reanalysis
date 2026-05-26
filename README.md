# Tree-CRISPR Reanalysis

A comprehensive reanalysis pipeline for CRISPR knockout and CRISPRa (activation) screens across multiple cancer cell lines, integrating MAGeCK statistical analysis, sgRNA-level hit calling, trajectory plotting, and epigenetic feature extraction.

## Overview

This pipeline processes pooled CRISPR screen data (from MAGeCK count outputs) through three analytical layers:

1. **Screen reanalysis** — MAGeCK-based negative selection, resistance hit calling, and custom FDR cutoffs across multiple drug conditions
2. **Visualisation** — density plots, heatmaps, trajectory plots, violin plots, and gene-level summaries
3. **Epigenetic integration** — TSS distance calculation, BigWig signal extraction, binning, and IGV-ready BED file generation

## Repository Layout

```
Tree-CRISPR-reanalysis/
├── epigenetic_pipeline/         # Genomic coordinate & epigenetic feature extraction scripts
├── A375_reanalysis/             # A375 melanoma cell line — KO + CRISPRa screens
│   ├── CRISPRa/                 # CRISPRa screen (SAM, T300, TP conditions)
│   ├── gene_summary/            # Gene-level essentiality and filtering
│   └── sgrna_summary/           # sgRNA-level hit calling (ALK conditions + K)
├── A549_reanalysis/             # A549 lung cancer cell line
│   ├── gene_summary/
│   └── sgrna_summary/           # ALK conditions + K
├── K562_reanalysis/             # K562 leukemia cell line — negative selection + CRISPRa
│   ├── gene_summary/
│   ├── sgRNA_summary/           # DMSO and RIG conditions
│   └── gene_tree_complete_pipeline.py
├── K562_resistance_screen/      # K562 resistance screen — ALK conditions, CRISPRa, RIG
│   ├── CRISPRa/                 # CRISPRa resistance (SAM, T300, TP, WT)
│   ├── Resistance_RIG_no_DMSO/  # RIG-I resistance (no DMSO baseline)
│   └── custom_heatmap/
├── TreeCRISPRa_library/         # CRISPRa library epigenetic extraction
├── TreeCRISPRi_library/         # CRISPRi library — sgRNA mapping, BED generation, epigenetic overlap
└── Comprehensive_Report/        # Cross-cell-line comprehensive summary plots
```

## Drug / Condition Abbreviations

| Code | Meaning |
|------|---------|
| K | Kinase inhibitor (baseline) |
| ALKE | ALK inhibitor — Early timepoint |
| ALKL | ALK inhibitor — Late timepoint |
| ALKH | ALK inhibitor — High dose |
| ALKME | ALK inhibitor + MEK — Early |
| ALKML | ALK inhibitor + MEK — Late |
| ALKMH | ALK inhibitor + MEK — High dose |
| WT | Wild-type (no treatment) |
| SAM | S-adenosylmethionine (CRISPRa condition) |
| T300 | T300 compound condition |
| TP | TP condition |
| DMSO | Vehicle control |
| RIG | RIG-I agonist |

## Standard Folder Structure per Condition

Each condition folder (e.g. `A375_reanalysis/sgrna_summary/ALKE/`) follows this structure:

```
ALKE/
├── process_neg_selection.py         # MAGeCK negative selection preprocessing
├── high_confidence_hits.py          # Filter sgRNAs to high-confidence hits
├── plot_density.py                  # sgRNA score density plot
└── custom_cutoff/
    ├── custom_cutoff.py             # Apply custom FDR/LFC thresholds
    └── plot_density.py              # Density plot at custom cutoff
```

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn pyBigWig pyfaidx
```

External tools required:
- [MAGeCK](https://sourceforge.net/projects/mageck/) — CRISPR screen statistical analysis
- [Bowtie](http://bowtie-bio.sourceforge.net/) — sgRNA genome mapping (used in `epigenetic_pipeline/mapper.py`)

## Quick Start

```bash
# 1. Clone and enter repo
git clone --branch master https://github.com/vipinmenon1989/Tree-CRISPR-reanalysis.git
cd Tree-CRISPR-reanalysis

# 2. Run MAGeCK (example for K562)
bash K562_reanalysis/mageck_K562.sh

# 3. Process negative selection hits
python K562_reanalysis/sgRNA_summary/dmso/ALKE/process_neg_selection.py

# 4. Call high-confidence hits
python K562_reanalysis/sgRNA_summary/dmso/ALKE/high_confidence_hits.py

# 5. Plot results
python K562_reanalysis/sgRNA_summary/dmso/ALKE/plot_density.py

# 6. Run epigenetic feature extraction
cd epigenetic_pipeline
python calc_distance_tss_revised.py library.txt tss_annotation.csv step1_output.txt
python windomaker.py step1_output.txt step2_output.txt
python all_bigwig.py step2_output.txt bigwig_folder/ final_matrix.txt
```

## Cell Line Summary

| Cell Line | Cancer Type | Screens Available |
|-----------|-------------|-------------------|
| A375 | Melanoma | KO negative selection, CRISPRa resistance |
| A549 | Lung adenocarcinoma | KO negative selection |
| K562 | CML leukemia | KO negative selection, CRISPRa resistance, RIG-I |
