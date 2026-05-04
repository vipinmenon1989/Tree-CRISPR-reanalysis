#!/bin/bash

# Input files
COUNT_FILE="A375_TreeCRISPRa.txt"
CONTROL_GENE="non-target-A375.txt"
OUT_DIR="/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/A375_reanalysis/CRISPRa/output"
# Create output directory
mkdir -p "$OUT_DIR"

echo "[INFO] Running MAGeCK: PLX vs Day 0 Baseline..."
echo "------------------------------------------------------------"

# 1. SAM Architecture (Day 27 PLX vs Day 0 SAM)
echo "[RUN 1] Processing SAM Architecture..."
mageck test -k $COUNT_FILE \
    -t d27_SAM_plx_R1,d27_SAM_plx_R2 \
    -c d0_SAM_R1,d0_SAM_R2 \
    -n ${OUT_DIR}/SAM_Res_vs_d0 \
    --norm-method control --control-gene $CONTROL_GENE

# 2. TP Architecture (Day 27 PLX vs Day 0 TP)
echo "[RUN 2] Processing TP Architecture..."
mageck test -k $COUNT_FILE \
    -t d27_TP_plx_R1,d27_TP_plx_R2 \
    -c d0_TP_R1,d0_TP_R2 \
    -n ${OUT_DIR}/TP_Res_vs_d0 \
    --norm-method control --control-gene $CONTROL_GENE

# 3. T300 Architecture (Day 30 PLX vs Day 0 T300)
# Note: Sticking with Day 30 to avoid the extreme Day 50 bottleneck
echo "[RUN 3] Processing T300 Architecture..."
mageck test -k $COUNT_FILE \
    -t d30_T300_plx_R1,d30_T300_plx_R2 \
    -c d0_T300_R1,d0_T300_R2 \
    -n ${OUT_DIR}/T300_Res_vs_d0 \
    --norm-method control --control-gene $CONTROL_GENE

echo "------------------------------------------------------------"
echo "[SUCCESS] Generated 3 summary files using Day 0 baselines."
