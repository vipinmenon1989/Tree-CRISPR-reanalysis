import os
import sys
import math
import numpy as np
import pandas as pd
import RNA  # Ensure ViennaRNA is correctly installed in your environment

# ======================================================================
# FORCE HEADLESS HPC RENDER BACKEND
# ======================================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ======================================================================

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

# ======================================================================
# BIOPHYSICAL EQUATION (LINDEL/CGDi)
# ======================================================================
def LINDELicalculatescore(seq):
    # Ensure inputs are clean uppercase strings
    if pd.isna(seq) or len(str(seq)) != 30:
        return 0.0
    seq = str(seq).upper()

    positional_features = [
        ('A',18,0.0815450954),('T',3,0.1251411781),('T',14,-0.05601527),
        ('T',16,-0.1374303579),('T',26,0.6129942086),('G',3,-0.1584387612),
        ('C',22,-0.0687106001),('C',23,-0.1450156038),('C',27,0.1564440206),
        ('C',28,-0.1007689224),('AA',9,-0.1547718709),('AA',11,-0.3012529992),
        ('AA',22,0.1655011092),('TA',8,-0.1481922885),('GA',13,0.1244079638),
        ('GA',18,0.1880705815),('CA',20,-0.3439729695),('CA',21,0.1416105185),
        ('CA',22,0.1522284385),('CA',27,0.1847169166),('CA',28,-0.1975980794),
        ('AT',2,0.3944690605),('AT',8,-0.1972874534),('AT',11,-0.2336100532),
        ('AT',22,-0.2545299014),('AT',27,-0.3396280329),('TT',5,-0.3974796405),
        ('TT',6,-0.0900360596),('TT',7,-0.2787390725),('TT',9,-0.1649851365),
        ('TT',10,-0.2572279591),('TT',11,-0.302645449),('TT',13,-0.3616339187),
        ('TT',14,-0.265153483),('TT',20,-0.293255811),('TT',21,-0.4450813624),
        ('TT',23,-0.2013282932),('GT',0,0.1335707585),('GT',1,-0.1045879319),
        ('GT',7,0.1046889318),('CT',15,-0.1706747529),('CT',23,-0.2440952346),
        ('AG',13,0.0674383094),('AG',21,-0.0548264226),('TG',16,-0.0621481838),
        ('GG',18,-0.3465682018),('GG',19,-0.1770629408),('GG',26,-0.207419321),
        ('GG',27,-0.2534565649),('CG',1,0.1439811675),('CG',7,0.1884952631),
        ('CG',10,0.1179399121),('CG',16,-0.2754010247),('CG',17,-0.1893117011),
        ('AC',15,0.2027064141),('AC',18,0.3311347746),('AC',20,0.2303312739),
        ('AC',28,0.1772425831),('TC',9,-0.1755042319),('GC',1,0.1250831227),
        ('GC',5,0.1045485972),('GC',11,0.0851007713),('GC',19,-0.1945267787),
        ('GC',20,-0.1458484707),('GC',21,-0.5363020499),('GC',22,-0.5256951798),
        ('GC',23,0.1888301689),('GC',25,-0.7042919152),('CC',18,0.1522234528),
        ('CC',22,-0.1439032381),('CC',28,-0.1504334951)
    ]

    intercept = -1.3484915738
    Free_energy = 0.0584654915
    Entropy = 0.4056274813
    GChigh = 0.7542669585
    GClow = -0.0065689225
    TT = -0.1045974512
    AT = -0.0957803804
    AG = 0.1051405001
    GG = 0.0459548209
    GT = 0.0463509282
    AA = -0.0437729377
    TA = 0.1324070584

    score = intercept

    # Free energy extraction strictly across core 20nt space (coordinates 4 to 24)
    guide = seq[4:24]
    _, mfe = RNA.fold(guide)
    mfe = round(mfe, 0)
    score += mfe * Free_energy

    # Dinucleotide frequency parsing across whole 30nt string context
    score += AG * seq.count('AG')
    score += AT * seq.count('AT')
    score += GG * seq.count('GG')
    score += TT * seq.count('TT')
    score += TA * seq.count('TA')
    score += AA * seq.count('AA')
    score += GT * seq.count('GT')

    # Shannon Entropy Vector Calculation
    freqs = {nt: guide.count(nt)/len(guide) for nt in 'ATGC'}
    entropy = -sum(v * np.log2(v) for v in freqs.values() if v > 0)
    score += round(entropy, 1) * Entropy

    # Local GC Content Weight Application
    gc_count = guide.count('G') + guide.count('C')
    gc_weight = GChigh if gc_count > 10 else GClow
    score += abs(10 - gc_count) * gc_weight

    # Structural Flanking Positional Feature Intersections
    for bp, pos, wt in positional_features:
        if seq[pos:pos+len(bp)] == bp:
            score += wt

    return 1.0 / (1.0 + math.exp(-score))

# ======================================================================
# MAIN EVALUATION PIPELINE
# ======================================================================
def main():
    # 1. Configuration
    working_dir = "./"
    
    # Path to your independent unseen guide partition
    input_file = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_unseen_guides.csv") 
    
    # Output paths
    output_prefix = "cgdi_independent"
    metrics_txt_path = os.path.join(working_dir, f"{output_prefix}_metrics.txt")
    metrics_csv_path = os.path.join(working_dir, f"{output_prefix}_metrics.csv")
    predictions_csv_path = os.path.join(working_dir, f"{output_prefix}_predictions.csv")
    curves_png_path = os.path.join(working_dir, f"{output_prefix}_curves.png")
    curves_pdf_path = os.path.join(working_dir, f"{output_prefix}_curves.pdf")
    
    # Hardcoded threshold based on standard logic
    threshold = 0.0

    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Independent test file missing at {input_file}")
        sys.exit(1)

    # 2. Data Ingestion & Sanitization
    print(f"[*] Ingesting independent dataset from: {input_file}")
    df = pd.read_csv(input_file, sep=',')
    df.columns = [col.lower().strip() for col in df.columns]

    # Verify structural columns exist
    required_cols = ['extended_sequence', 'sigmoid_score', 'unique_sgrna_id', 'sgrna sequence']
    for col in required_cols:
        if col not in df.columns:
            print(f"CRITICAL ERROR: Required column '{col}' missing from input file header!")
            sys.exit(1)

    # Enforce strict 0.50 threshold to match the established ground truth
    df['class'] = (df['sigmoid_score'] > 0.5).astype(int)

    # 3. Apply Biophysical Scoring Engine
    print("[*] Processing biophysical properties... Generating CGDi_Scores...")
    df['cgdi_score'] = df['extended_sequence'].apply(LINDELicalculatescore)

    # Generate binary predictions based on the equation's score
    df['prediction_binary'] = (df['cgdi_score'] >= threshold).astype(int)

    # 4. Extract Granular Predictions
    strand_col = 'strand' if 'strand' in df.columns else ('guide_strand' if 'guide_strand' in df.columns else None)
    
    export_cols = [
        'unique_sgrna_id', 
        'gene', 
        'sgrna sequence', 
        'extended_sequence', 
        'start', 
        'end', 
        strand_col, 
        'sigmoid_score', 
        'class',
        'cgdi_score',
        'prediction_binary'
    ]
    
    # Filter valid columns for export
    valid_cols = [col for col in export_cols if col and col in df.columns]
    df_output = df[valid_cols].copy()

    # Save to disk
    df_output.to_csv(predictions_csv_path, index=False)
    print(f"--> Granular predictions saved to: {predictions_csv_path}")

    # 5. Evaluate Performance Metrics
    print("[*] Calculating performance metrics...")
    y_true = df['class'].values
    y_scores = df['cgdi_score'].values
    y_pred = df['prediction_binary'].values

    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 6. Export Metrics Logging
    print(f"--> Exporting evaluation logs...")
    
    # Text format
    with open(metrics_txt_path, 'w') as f:
        f.write("=== CGDi Biophysical Equation Evaluation (Independent Split) ===\n")
        f.write(f"Testing Pool Size: {len(y_true)} rows\n")
        f.write(f"Class 1 (Hit) Count: {np.sum(y_true == 1)}\n")
        f.write(f"Class 0 (Noise) Count: {np.sum(y_true == 0)}\n")
        f.write("-" * 65 + "\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Precision (at >= {threshold}): {precision:.4f}\n")
        f.write(f"Recall (at >= {threshold}): {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("-" * 65 + "\n")

    # CSV Format (for easy programmatic loading later if needed)
    metrics_df = pd.DataFrame({
        'ROC_AUC': [roc_auc],
        'AUPR': [pr_auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1]
    })
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 7. Generate Performance Curves
    print("[*] Rendering evaluation curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax1.plot(fpr, tpr, color='crimson', lw=2.5, label=f'CGDi (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('CGDi ROC Curve', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    baseline_pr = np.sum(y_true == 1) / len(y_true)
    ax2.plot(recall_curve, precision_curve, color='dodgerblue', lw=2.5, label=f'CGDi (PR-AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_pr, color='gray', lw=1.5, linestyle='--', label=f'Baseline Class Ratio (PR = {baseline_pr:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('CGDi PR Curve', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(curves_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"\n[-->] CGDi independent evaluation completed successfully.")
    print(f" -> ROC AUC:   {roc_auc:.4f}")
    print(f" -> PR AUC:    {pr_auc:.4f}")
    print(f" -> Precision: {precision:.4f} (at threshold {threshold})")
    print(f" -> Recall:    {recall:.4f}")
    print(f" -> F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
