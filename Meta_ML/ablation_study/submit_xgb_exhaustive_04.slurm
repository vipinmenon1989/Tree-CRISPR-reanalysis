#!/bin/bash
#SBATCH --job-name=xgb_04
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=18:00:00
#SBATCH --output=/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/ablation_study/xgb_ablation_04_%j.log
#SBATCH --error=/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/ablation_study/xgb_ablation_04_%j.err

# Navigate to execution folder
cd /local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/ablation_study/

# Initialize and activate Conda environment
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate ML

# Create ablation output directory
#mkdir -p /local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/ablation_study/

# Execute 6-part XGBoost classification ablation study sequentially (threshold 0.4)
echo "Starting Part 1: Baseline (No Epigenetics) at 0.4 threshold"
python3 train_xgb_baseline_04.py

echo "Starting Part 2: Gene-Level Epigenetics Only at 0.4 threshold"
python3 train_xgb_gene_epi_04.py

echo "Starting Part 3: Guide-Level RAW Epigenetics Only at 0.4 threshold"
python3 train_xgb_guide_raw_04.py

echo "Starting Part 4: Guide-Level AGGREGATED Epigenetics Only at 0.4 threshold"
python3 train_xgb_guide_agg_04.py

echo "Starting Part 5: Gene + Guide RAW Epigenetics at 0.4 threshold"
python3 train_xgb_gene_plus_guide_raw_04.py

echo "Starting Part 6: Gene (RAW) + Guide (AGGREGATED) Epigenetics at 0.4 threshold"
python3 train_xgb_gene_plus_guide_agg_04.py

echo "All threshold 0.4 ablation training runs successfully completed!"
