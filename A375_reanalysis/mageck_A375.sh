#!/bin/bash

# Input files
COUNT_FILE="A375.count.txt"
CONTROL_GENE="non-target-A375.txt"
OUTPUT="/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/A375_reanalysis/output"

# Start command construction
CMD="mageck test -k $COUNT_FILE -n $OUTPUT --norm-method control --control-gene $CONTROL_GENE"

# Manually pairing Treatment (-t) to its specific Baseline (-c)
# This is exactly what you did for K562
CMD+=" -t d15_K_R1,d15_K_R2           -c d0_K_R1,d0_K_R2"
CMD+=" -t d15_ALK_L_R1,d15_ALK_L_R2   -c d0_ALK_L_R1,d0_ALK_L_R2"
CMD+=" -t d15_ALKM_L_R1,d15_ALKM_L_R2 -c d0_ALKM_L_R1,d0_ALKM_L_R2"
CMD+=" -t d15_ALK_E_R1,d15_ALK_E_R2   -c d0_ALK_E_R1,d0_ALK_E_R2"
CMD+=" -t d15_ALKM_E_R1,d15_ALKM_E_R2 -c d0_ALKM_E_R1,d0_ALKM_E_R2"
CMD+=" -t d15_ALK_H_R1,d15_ALK_H_R2   -c d0_ALK_H_R1,d0_ALK_H_R2"
CMD+=" -t d15_ALKM_H_R1,d15_ALKM_H_R2 -c d0_ALKM_H_R1,d0_ALKM_H_R2"

# Display full command
echo "[INFO] Running full mageck test command:"
echo "$CMD"
echo "-----------------------------------------"

# Run the full command once
eval "$CMD"
