#!/bin/bash

# Input files
COUNT_FILE="K562_i.count_normalized.txt"
CONTROL_GENE="non-target.txt"
OUTPUT="/local/projects-t3/lilab/vmenon/CRISPRi/K562/K562_reanalysis/output"

# Start command construction
CMD="mageck test -k $COUNT_FILE -n $OUTPUT --norm-method control --control-gene $CONTROL_GENE"

# Append all test-control groups
CMD+=" -t i_d14_WT_dmso_R1,i_d14_WT_dmso_R2     -c i_d0_WT_R1,i_d0_WT_R2"   
CMD+=" -t d14_K_dmso_R1,d14_K_dmso_R2           -c d0_K_R1,d0_K_R2"
CMD+=" -t d14_ALKM_L_dmso_R1,d14_ALKM_L_dmso_R2 -c d0_ALKM_L_R1,d0_ALKM_L_R2"
CMD+=" -t d14_ALKM_E_dmso_R1,d14_ALKM_E_dmso_R2 -c d0_ALKM_E_R1,d0_ALKM_E_R2"
CMD+=" -t d14_ALKM_H_dmso_R1,d14_ALKM_H_dmso_R2 -c d0_ALKM_H_R1,d0_ALKM_H_R2"
CMD+=" -t d14_ALK_L_dmso_R1,d14_ALK_L_dmso_R2   -c d0_ALK_L_R1,d0_ALK_L_R2"
CMD+=" -t d14_ALK_E_dmso_R1,d14_ALK_E_dmso_R2   -c d0_ALK_E_R1,d0_ALK_E_R2"
CMD+=" -t d14_ALK_H_dmso_R1,d14_ALK_H_dmso_R2   -c d0_ALK_H_R1,d0_ALK_H_R2"
CMD+=" -t i_d14_WT_rig_R1,i_d14_WT_rig_R2        -c i_d0_WT_R1,i_d0_WT_R2" 
CMD+=" -t d14_K_rig_R1,d14_K_rig_R2             -c d0_K_R1,d0_K_R2"
CMD+=" -t d14_ALKM_L_rig_R1,d14_ALKM_L_rig_R2   -c d0_ALKM_L_R1,d0_ALKM_L_R2"
CMD+=" -t d14_ALKM_E_rig_R1,d14_ALKM_E_rig_R2   -c d0_ALKM_E_R1,d0_ALKM_E_R2"
CMD+=" -t d14_ALKM_H_rig_R1,d14_ALKM_H_rig_R2   -c d0_ALKM_H_R1,d0_ALKM_H_R2"
CMD+=" -t d14_ALK_L_rig_R1,d14_ALK_L_rig_R2     -c d0_ALK_L_R1,d0_ALK_L_R2"
CMD+=" -t d14_ALK_E_rig_R1,d14_ALK_E_rig_R2     -c d0_ALK_E_R1,d0_ALK_E_R2"
CMD+=" -t d14_ALK_H_rig_R1,d14_ALK_H_rig_R2     -c d0_ALK_H_R1,d0_ALK_H_R2"

# Display full command
echo "[INFO] Running full mageck test command:"
echo "$CMD"
echo "-----------------------------------------"

# Run the full command once
eval "$CMD"
