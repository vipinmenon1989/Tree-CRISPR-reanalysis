import os
import sys
import math
import numpy as np
import pandas as pd
import ViennaRNA as RNA  # Standard package mapping for modern python-RNA wrappers
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

def LINDELicalculatescore(seq):
    # Ensure inputs are clean uppercase strings
    if pd.isna(seq) or len(seq) != 30:
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

def cgdscoring(input_file, output_prefix, threshold=0.5):
    print(f"[*] Ingesting predictions file from target matrix path: {input_file}")
    df = pd.read_csv(input_file, sep="\t")
    df.columns = [col.lower().strip() for col in df.columns]

    # Validate presence of the raw 30nt genomic context column
    if 'prediction_values' not in df.columns:
        print("CRITICAL ERROR: 'prediction_values' (30nt context) missing from input file header!")
        sys.exit(1)

    print("[*] Processing biophysical properties... Generating CGDi_Scores...")
    df['cgdi_score'] = df['prediction_values'].apply(LINDELicalculatescore)

    # Re-align headers to match casing requested for final delivery
    df_output = pd.DataFrame({
        'unique_sgrna_id': df['unique_sgrna_id'].values,
        'sgrna sequence': df['sgrna sequence'].values,
        'class': df['class'].values,
        'prediction_probability': df['prediction_probability'].values,
        'prediction_binary': df['prediction_binary'].values,
        'prediction values': df['prediction_values'].values,
        'CGDi_Score': df['cgdi_score'].values
    })

    # Save target independent predictions matrix
    scored_file = os.path.join(os.path.dirname(input_file), "independent_predictions.txt")
    df_output.to_csv(scored_file, sep="\t", index=False)
    print(f"[✔] Successfully exported scored prediction roster → {scored_file}")

    # Initialize evaluations vector
    metrics_dict = {
        'ROC_AUC': [None],
        'AUPR': [None],
        'Precision': [None],
        'Recall': [None],
        'F1': [None]
    }

    if 'class' in df.columns:
        try:
            y_true = df['class'].astype(int)
            y_scores = df['cgdi_score']

            metrics_dict['ROC_AUC'][0] = roc_auc_score(y_true, y_scores)
            metrics_dict['AUPR'][0] = average_precision_score(y_true, y_scores)

            # Apply user discrete decision threshold
            y_pred = (y_scores >= threshold).astype(int)

            metrics_dict['Precision'][0] = precision_score(y_true, y_pred, zero_division=0)
            metrics_dict['Recall'][0] = recall_score(y_true, y_pred, zero_division=0)
            metrics_dict['F1'][0] = f1_score(y_true, y_pred, zero_division=0)

            print("\n" + "="*50)
            print("  BIOPHYSICAL MODEL EVALUATION SUMMARY")
            print("="*50)
            print(f" -> ROC AUC:   {metrics_dict['ROC_AUC'][0]:.4f}")
            print(f" -> AUPR:      {metrics_dict['AUPR'][0]:.4f}")
            print(f" -> Precision: {metrics_dict['Precision'][0]:.4f} (at threshold {threshold})")
            print(f" -> Recall:    {metrics_dict['Recall'][0]:.4f}")
            print(f" -> F1 Score:  {metrics_dict['F1'][0]:.4f}")
            print("="*50)

        except Exception as e:
            print(f"[ERROR] Scoring metric matrix evaluation aborted: {e}")

    # Export clean metric evaluation sheet
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_file = f"{output_prefix}_biophysical_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"[✔] Saved performance reports successfully → {metrics_file}")

    return scored_file


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python your_script.py <prediction.txt> <output_prefix> [threshold]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_prefix = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 0.5

    result_file = cgdscoring(input_file, output_prefix, threshold)