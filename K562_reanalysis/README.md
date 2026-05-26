# K562_reanalysis — K562 Leukemia CRISPR Screen Reanalysis

Reanalysis of CRISPR knockout and CRISPRa screens in the K562 chronic myeloid leukemia (CML) cell line, including DMSO-treated negative selection and RIG-I agonist conditions.

## Directory Structure

```
K562_reanalysis/
├── mageck_K562.sh                   # MAGeCK count + test pipeline
├── gene_based_plots.py              # Gene-level visualisation
├── gene_filter.py                   # Gene hit filtering
├── gene_tree_complete_pipeline.py   # Complete gene-level tree pipeline
├── sgrna_direct.py                  # Direct sgRNA-level analysis utility
├── gene_summary/
│   ├── gene_essentiality_dmso.py
│   ├── gene_plot_dmso_filtered.py
│   ├── gene_plot_dmso_unfiltered.py
│   ├── gene_plot_high_hits_esg_res_filtered.py
│   ├── gene_plot_high_hits_esg_res_unfiltered.py
│   └── gene_tree_complete_pipeline.py
└── sgRNA_summary/
    ├── check_depth.py               # Sequencing depth QC
    ├── dmso/                        # DMSO (vehicle control) condition
    │   ├── process_neg_selection.py
    │   ├── plot_density.py
    │   ├── heatmap_matrix.py
    │   ├── check_depth.py
    │   ├── Trajectory_plot_custom_heatmap.py
    │   ├── custom_heatmp/
    │   ├── ALKE/ ALKH/ ALKL/ ALKME/ ALKMH/ ALKML/ K/ WT/
    │   └── K562_heatmap/            # K562-specific heatmap analysis
    │       ├── Trajectory_plot.py
    │       ├── Trajectoyr_plot_custom_k_means_heirarchial_cluster.py
    │       ├── heatmap.py
    │       ├── heatmap_matrix.py
    │       ├── voilin_and_heatmap.py
    │       ├── batch_list.py
    │       ├── gene_extractor.py
    │       └── gene_list/
    └── rig/
        └── check_depth.py           # Depth QC for RIG-I condition
```

## Running the Pipeline

```bash
# 1. Run MAGeCK
bash mageck_K562.sh

# 2. Check sequencing depth
python sgRNA_summary/check_depth.py

# 3. Process DMSO negative selection (example: ALKE)
python sgRNA_summary/dmso/ALKE/process_neg_selection.py
python sgRNA_summary/dmso/ALKE/high_confidence_hits.py
python sgRNA_summary/dmso/ALKE/plot_density.py

# 4. Gene-level analysis
python gene_summary/gene_essentiality_dmso.py
python gene_summary/gene_plot_dmso_filtered.py

# 5. Heatmap and trajectory visualisation
python sgRNA_summary/dmso/K562_heatmap/heatmap_matrix.py
python sgRNA_summary/dmso/K562_heatmap/Trajectory_plot.py
```
