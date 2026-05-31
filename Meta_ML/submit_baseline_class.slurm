#!/bin/bash
#SBATCH --job-name=enlr_base
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/baseline_class_%j.log
#SBATCH --error=/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/baseline_class_%j.err

# Navigate to execution folder
cd /local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/

# Initialize and activate Conda environment
source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate ML

# Execute Elastic Net Logistic Regression Baseline Classification training
python3 train_enlr_baseline_class.py
