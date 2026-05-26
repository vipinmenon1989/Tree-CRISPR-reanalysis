# A375_reanalysis — A375 Melanoma CRISPR Screen Reanalysis

Reanalysis of CRISPR knockout and CRISPRa screens in the A375 melanoma cell line across multiple ALK inhibitor conditions and CRISPRa delivery methods.

## Directory Structure

```
A375_reanalysis/
├── mageck_A375.sh               # MAGeCK count + test pipeline (KO screen)
├── CRISPRa/                     # CRISPRa activation screen
│   ├── mageck_A375.sh           # MAGeCK pipeline for CRISPRa
│   ├── mageck_A375_dmso_rig.sh  # MAGeCK with DMSO vs RIG-I comparison
│   ├── Trajectory_plot.py       # Guide-level trajectory across timepoints
│   ├── heatmap_matrix.py        # Heatmap matrix of sgRNA scores
│   ├── check_depth_for_days.py  # Sequencing depth QC per timepoint
│   ├── A375_heatmap/            # Custom heatmap analysis
│   ├── SAM/                     # SAM condition (gene + sgRNA summary)
│   ├── T300/                    # T300 condition
│   ├── TP/                      # TP condition
│   └── output/                  # Processed outputs with custom cutoffs
├── gene_summary/                # Gene-level analysis
│   ├── gene_essentiality_dmso.py    # Essentiality scoring under DMSO
│   └── gene_filtered_dmso.py        # Filtered gene hits under DMSO
└── sgrna_summary/               # sgRNA-level hit calling
    ├── process_neg_selection.py     # Global negative selection preprocessing
    ├── plot_density.py              # Global density plot
    ├── Trajectory_plot.py           # Trajectory across all conditions
    ├── Trajectory_plot_custom_cutoff_heatap.py
    ├── heatmap_matrix.py
    ├── voilin_heatmap.py
    ├── batch_list.py                # Batch correction list
    ├── custom_heatmap/              # Custom heatmap analysis
    ├── ALKE/                        # ALK-Early condition
    ├── ALKH/                        # ALK-High condition
    ├── ALKL/                        # ALK-Late condition
    ├── ALKME/                       # ALK+MEK-Early condition
    ├── ALKMH/                       # ALK+MEK-High condition
    ├── ALKML/                       # ALK+MEK-Late condition
    └── K/                           # Kinase inhibitor baseline
```

## Standard Analysis Per Condition

Each condition folder (e.g. `sgrna_summary/ALKE/`) contains:

| Script | Purpose |
|--------|---------|
| `process_neg_selection.py` | Preprocess MAGeCK negative selection output |
| `high-confidence_hits.py` / `high_confidence_hits.py` | Filter to statistically robust hits |
| `plot_density.py` | Density plot of sgRNA LFC scores |
| `custom_cutoff/custom_cutoff.py` | Apply custom FDR/LFC thresholds |
| `custom_cutoff/plot_density.py` | Density plot at custom cutoff |

## CRISPRa Conditions

The `CRISPRa/` folder mirrors the same structure for activation screens:

| Condition | Description |
|-----------|-------------|
| SAM | S-adenosylmethionine-dependent activation |
| T300 | T300 compound condition |
| TP | TP condition |

Each has `gene_summary/` and `sgrna_summary/` subdirectories.

## Running the Pipeline

```bash
# 1. Run MAGeCK
bash mageck_A375.sh

# 2. Process sgRNA-level hits for a condition
python sgrna_summary/ALKE/process_neg_selection.py
python sgrna_summary/ALKE/high-confidence_hits.py
python sgrna_summary/ALKE/plot_density.py

# 3. Apply custom cutoffs
python sgrna_summary/ALKE/custom_cutoff/custom_cutoff.py
python sgrna_summary/ALKE/custom_cutoff/plot_density.py

# 4. Gene-level analysis
python gene_summary/gene_essentiality_dmso.py
python gene_summary/gene_filtered_dmso.py

# 5. Trajectory and heatmap visualisation
python sgrna_summary/Trajectory_plot.py
python sgrna_summary/heatmap_matrix.py
```
