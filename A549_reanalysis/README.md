# A549_reanalysis — A549 Lung Cancer CRISPR Screen Reanalysis

Reanalysis of CRISPR knockout screens in the A549 lung adenocarcinoma cell line across ALK inhibitor and combination treatment conditions.

## Directory Structure

```
A549_reanalysis/
├── mageck_A549.sh            # MAGeCK count + test pipeline
├── gene_summary/
│   ├── gene_essentiality_dmso.py    # Gene essentiality scoring under DMSO
│   └── gene_filtered_dmso.py        # Filtered gene hits
└── sgrna_summary/
    ├── heatmap_matrix.py            # Heatmap of sgRNA scores across conditions
    ├── voilin_heatmap.py            # Violin + heatmap combined plot
    ├── Trajectory_plot.py           # Trajectory plot across timepoints
    ├── Trajectory_cutsom_heatmap.py # Custom trajectory heatmap
    ├── batch_list.py                # Batch correction inputs
    ├── custom_heatmap/
    │   └── analysis.py              # Custom heatmap analysis
    ├── ALKE/                        # ALK-Early
    ├── ALKH/                        # ALK-High
    ├── ALKL/                        # ALK-Late
    ├── ALKME/                       # ALK+MEK-Early
    ├── ALKMH/                       # ALK+MEK-High
    ├── ALKML/                       # ALK+MEK-Late
    └── K/                           # Kinase inhibitor baseline
```

## Standard Analysis Per Condition

Each condition folder contains:

| Script | Purpose |
|--------|---------|
| `process_neg_selection.py` | Preprocess MAGeCK negative selection output |
| `high_confidence_hits.py` | Filter to high-confidence sgRNA hits |
| `plot_density.py` | Density plot of sgRNA scores |
| `custom_cutoff/custom_cutoff.py` | Custom FDR/LFC threshold application |
| `custom_cutoff/plot_density.py` | Density plot at custom cutoff |

## Running the Pipeline

```bash
# 1. Run MAGeCK
bash mageck_A549.sh

# 2. Per-condition processing (example: ALKL)
python sgrna_summary/ALKL/process_neg_selection.py
python sgrna_summary/ALKL/high_confidence_hits.py
python sgrna_summary/ALKL/plot_density.py

# 3. Gene-level summary
python gene_summary/gene_essentiality_dmso.py
python gene_summary/gene_filtered_dmso.py

# 4. Visualisation
python sgrna_summary/heatmap_matrix.py
python sgrna_summary/Trajectory_plot.py
```
