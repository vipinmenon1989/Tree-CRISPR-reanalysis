#!/bin/bash

# Input files
COUNT_FILE="K562_a.count_normalized.txt"
CONTROL_GENE="non-target.txt"
OUTPUT="/local/projects-t3/lilab/vmenon/CRISPRi/K562/K562_reanalysis/CRISPRa/output"

# Start command construction
CMD="mageck test -k $COUNT_FILE -n $OUTPUT --norm-method control --control-gene $CONTROL_GENE"

# Append all test-control groups
CMD+=" -t a_d14_WT_dmso_R1,a_d14_WT_dmso_R2      -c a_d0_WT_R1,a_d0_WT_R2"   
CMD+=" -t d14_SAM_dmso_R1,d14_SAM_dmso_R2        -c d0_SAM_R1,d0_SAM_R2"
CMD+=" -t d14_TP_dmso_R1,d14_TP_dmso_R2          -c d0_TP_R1,d0_TP_R2"
CMD+=" -t d14_T300_dmso_R1,d14_T300_dmso_R2      -c d0_T300_R1,d0_T300_R2"
CMD+=" -t a_d14_WT_rig_R1,a_d14_WT_rig_R2        -c a_d0_WT_R1,a_d0_WT_R2"
CMD+=" -t d14_SAM_rig_R1,d14_SAM_rig_R2          -c d0_SAM_R1,d0_SAM_R2"
CMD+=" -t d14_TP_rig_R1,d14_TP_rig_R2            -c d0_TP_R1,d0_TP_R2"
CMD+=" -t d14_T300_rig_R1,d14_T300_rig_R2        -c d0_T300_R1,d0_T300_R2"

# Display full command
echo "[INFO] Running full mageck test command:"
echo "$CMD"
echo "-----------------------------------------"

# Run the full command once
eval "$CMD"

