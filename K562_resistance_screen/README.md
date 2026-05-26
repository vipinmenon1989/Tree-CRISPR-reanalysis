# K562_resistance_screen — K562 Resistance Screen Analysis

Analysis of drug resistance screens in K562 cells, covering ALK inhibitor conditions, CRISPRa-based resistance, RIG-I agonist conditions (without DMSO baseline), and wild-type controls.

## Directory Structure

```
K562_resistance_screen/
├── mageck_K562.sh (via CRISPRa/)    # MAGeCK pipeline
├── resistance_hits.py               # Global resistance hit calling
├── heatmap_matrix.py                # Cross-condition heatmap
├── Trajectory_plot.py               # Cross-condition trajectory
├── Trajecotry_plot_heatmap_custom.py
├── preprocessing_resistance_screen_custom.py
├── custom_heatmap/
│   ├── analysis.py
│   └── myawk.sh
│
├── ALKE/  ALKH/  ALKL/  ALKME/  ALKMH/  ALKML/  K/  WT/
│   └── (each has gene_summary/ + sgrna_summary/ with custom_cutoff/)
│
├── CRISPRa/                         # CRISPRa-based resistance screen
│   ├── analysis.py
│   ├── gene_plot_summary_filtered.py
│   ├── gene_plot_summary_unfiltered.py
│   ├── gene_tree_complete_pipeline.py
│   ├── resistance_hits.py
│   ├── mageck_K562.sh
│   ├── K562/                        # Baseline K562 dropout (SAM, T300, TP, WT)
│   └── K562_resistance_screen/      # Resistance comparisons
│       ├── SAM/  T300/  TP/  WT/
│       ├── Resistance_no_DMSO/      # Resistance without DMSO baseline
│       ├── Trajectory_plot.py
│       ├── Trajectory_plot_heatmap_custom.py
│       ├── heatmap_matrix.py
│       └── custom_heatmap/
│
└── Resistance_RIG_no_DMSO/          # RIG-I resistance (no DMSO)
    ├── Trajectory_plot.py
    ├── heatmap_matrix.py
    └── ALKE/  ALKH/  ALKL/  ALKME/  ALKMH/  ALKML/  K/
        └── (gene_summary/ + sgrna_summary/)
```

## Standard Per-Condition Structure

Each ALK condition folder (e.g. `ALKE/`) contains:

```
ALKE/
├── gene_summary/
│   └── resistance_hits.py           # Gene-level resistance hits
├── resistance_hits.py               # Top-level resistance hit script
└── sgrna_summary/
    ├── preprocessing_resistance_screen.py     # MAGeCK preprocessing
    ├── preprocessing_resistance_screen_custom.py
    ├── high_confidence_hits.py
    ├── density_plot.py
    └── custom_cutoff/
        ├── custom_cutoff.py
        └── density_plot.py / denisty_plot.py
```

## Running the Pipeline

```bash
# 1. Preprocess resistance screen for a condition
python ALKE/sgrna_summary/preprocessing_resistance_screen.py

# 2. Call high-confidence resistance hits
python ALKE/sgrna_summary/high_confidence_hits.py

# 3. Apply custom cutoffs
python ALKE/sgrna_summary/custom_cutoff/custom_cutoff.py

# 4. Gene-level resistance summary
python ALKE/gene_summary/resistance_hits.py

# 5. Cross-condition visualisation
python Trajectory_plot.py
python heatmap_matrix.py

# 6. CRISPRa resistance pipeline
cd CRISPRa
python gene_tree_complete_pipeline.py
```

## Notes

- `Resistance_RIG_no_DMSO/` contains resistance screens using RIG-I agonist as the selective pressure, without a DMSO baseline comparison — analysis uses a different reference condition
- `validate_methods.py` (in some ALKE/ALKL/ALKME/ALKML subfolders) performs method validation comparing two hit-calling approaches
