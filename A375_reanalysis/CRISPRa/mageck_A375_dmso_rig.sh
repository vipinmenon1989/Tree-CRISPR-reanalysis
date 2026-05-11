#!/bin/bash

# 1. Input files and directories
COUNT_FILE="A375_TreeCRISPRa.txt"
CONTROL_GENE="non-target-A375.txt"
OUTPUT_DIR="/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/A375_reanalysis/CRISPRa/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# 2. Define the Analytical Matrix: "Output_Prefix | Test_Columns | Control_Columns"
COMPARISONS=(
    # --- SAM System (Day 27) ---
    "SAM_d27_dmso  | d27_SAM_dmso_R1,d27_SAM_dmso_R2 | d0_SAM_R1,d0_SAM_R2"
    "SAM_d27_plx   | d27_SAM_plx_R1,d27_SAM_plx_R2   | d0_SAM_R1,d0_SAM_R2"
    
    # --- TP System (Day 27) ---
    "TP_d27_dmso   | d27_TP_dmso_R1,d27_TP_dmso_R2   | d0_TP_R1,d0_TP_R2"
    "TP_d27_plx    | d27_TP_plx_R1,d27_TP_plx_R2     | d0_TP_R1,d0_TP_R2"
    
    # --- T300 System (Day 30) ---
    "T300_d30_dmso | d30_T300_dmso_R1,d30_T300_dmso_R2 | d0_T300_R1,d0_T300_R2"
    "T300_d30_plx  | d30_T300_plx_R1,d30_T300_plx_R2   | d0_T300_R1,d0_T300_R2"
    
    # --- T300 System (Day 41) ---
    "T300_d41_dmso | d41_T300_dmso_R1,d41_T300_dmso_R2 | d0_T300_R1,d0_T300_R2"
    "T300_d41_plx  | d41_T300_plx_R1,d41_T300_plx_R2   | d0_T300_R1,d0_T300_R2"
    
    # --- T300 System (Day 50) ---
    "T300_d50_dmso | d50_T300_dmso_R1,d50_T300_dmso_R2 | d0_T300_R1,d0_T300_R2"
    "T300_d50_plx  | d50_T300_plx_R1,d50_T300_plx_R2   | d0_T300_R1,d0_T300_R2"
)

echo "Starting MAGeCK analysis..."

# 3. Execution Loop
for COMP in "${COMPARISONS[@]}"; do
    # Parse the string
    PREFIX=$(echo "$COMP" | awk -F' \\| ' '{print $1}' | xargs)
    TEST=$(echo "$COMP" | awk -F' \\| ' '{print $2}' | xargs)
    CTRL=$(echo "$COMP" | awk -F' \\| ' '{print $3}' | xargs)
    
    OUT_PATH="${OUTPUT_DIR}/${PREFIX}"
    
    CMD="mageck test -k $COUNT_FILE -n $OUT_PATH -t $TEST -c $CTRL --norm-method control --control-gene $CONTROL_GENE"
    
    echo "-----------------------------------------"
    echo "[INFO] Running comparison: $PREFIX"
    echo "[CMD]  $CMD"
    
    # Run the command
    eval "$CMD"
done

echo "-----------------------------------------"
echo "Success! All MAGeCK runs completed."
