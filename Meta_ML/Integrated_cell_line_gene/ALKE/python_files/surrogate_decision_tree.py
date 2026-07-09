import matplotlib
matplotlib.use('Agg') # HPC headless mode

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import sys

def run_dynamic_holdout_and_export(file_path):
    print("======================================================================")
    print("[*] 80/20 HOLDOUT VALIDATION & RULE EXPORT")
    print("======================================================================\n")
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
    except FileNotFoundError:
        print(f"[-] Error: File {file_path} not found.")
        sys.exit(1)

    # 2. Strict 80/20 Stratified Split
    y = df['class'].values
    drop_cols = [c for c in df.columns if c in ['id', 'gene', 'mean_sigmoid_score', 'class']]
    X_df = df.drop(columns=drop_cols)
    feature_names = X_df.columns.tolist()
    
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_df, y, df.index, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Train Fresh Tree STRICTLY on 80% Training Data
    tree_model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=15, class_weight='balanced', random_state=42)
    tree_model.fit(X_train, y_train)
    
    # 4. DYNAMIC MATHEMATICAL EXTRACTION
    tree_obj = tree_model.tree_
    
    root_node = 0
    root_feat = feature_names[tree_obj.feature[root_node]]
    root_thresh = tree_obj.threshold[root_node]
    
    left_node = tree_obj.children_left[root_node]
    left_feat = feature_names[tree_obj.feature[left_node]]
    left_thresh = tree_obj.threshold[left_node]
    
    right_node = tree_obj.children_right[root_node]
    right_feat = feature_names[tree_obj.feature[right_node]]
    right_thresh = tree_obj.threshold[right_node]

    # =====================================================================
    # 5. EXPORT RULES TO .TXT FILE
    # =====================================================================
    raw_tree_text = export_text(tree_model, feature_names=feature_names)
    
    text_output = f"""======================================================================
ALKE MACRO-ENVIRONMENT SURROGATE RULES
Extracted dynamically from 80% Training Data Split
======================================================================

1. THE ROOT ROUTER (Determines the physical vulnerability zone):
   - IF {root_feat} <= {root_thresh:.4f}  --> Proceed to Panel A (Vulnerable Chromatin)
   - IF {root_feat} >  {root_thresh:.4f}  --> Proceed to Panel B (Forced-Open Chromatin)

2. PANEL A VETO RULE (The General Methylation Blanket):
   - IF {left_feat} > {left_thresh:.4f} --> VETO Triggered (Class 0: Guaranteed Failure)
   - ELSE --> RUNWAY CLEAR (Class 1)

3. PANEL B VETO RULE (The Targeted CpG Methylation Lock):
   - IF {right_feat} > {right_thresh:.4f} --> VETO Triggered (Class 0: Guaranteed Failure)
   - ELSE --> RUNWAY CLEAR (Class 1)

======================================================================
RAW SCIKIT-LEARN DECISION PATH OUTPUT:
======================================================================
{raw_tree_text}
"""
    
    output_txt = "alke_surrogate_rules.txt"
    with open(output_txt, "w") as f:
        f.write(text_output)
    
    print(f"[+] Rules successfully exported to: {output_txt}")

    # =====================================================================
    # 6. Generate Plot Using Dynamic Thresholds
    # =====================================================================
    test_df = df.loc[indices_test].copy()
    mod_zone_test = test_df[test_df[root_feat] <= root_thresh]
    high_zone_test = test_df[test_df[root_feat] > root_thresh]

    print("[*] Generating Holdout Plot on Unseen 20%...")
    plt.style.use('default') 
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    dot_colors = {0: '#34495e', 1: '#2980b9'}

    # Panel A
    ax = axes[0]
    ax.axvspan(0, left_thresh, color='#2ecc71', alpha=0.12, lw=0) 
    ax.axvspan(left_thresh, 1.05, color='#e74c3c', alpha=0.12, lw=0) 
    ax.axvline(left_thresh, color='black', linestyle='--', linewidth=2)
    
    sns.scatterplot(
        data=mod_zone_test, x=left_feat, y=root_feat, 
        hue='class', palette=dot_colors, ax=ax, s=85, edgecolor='white', 
        linewidth=1, alpha=0.9, legend=False
    )
    
    ax.set_title(f"Panel A: Holdout Validation\n({root_feat} $\leq$ {root_thresh:.2f})", fontsize=14, weight='bold')
    ax.set_xlabel(left_feat, fontsize=12)
    ax.set_ylabel(root_feat, fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, root_thresh + 0.05) 

    # Panel B
    ax = axes[1]
    ax.axvspan(0, right_thresh, color='#2ecc71', alpha=0.12, lw=0) 
    ax.axvspan(right_thresh, 1.05, color='#e74c3c', alpha=0.12, lw=0) 
    ax.axvline(right_thresh, color='black', linestyle='--', linewidth=2)
    
    sns.scatterplot(
        data=high_zone_test, x=right_feat, y=root_feat, 
        hue='class', palette=dot_colors, ax=ax, s=85, edgecolor='white', 
        linewidth=1, alpha=0.9
    )
    
    ax.set_title(f"Panel B: Holdout Validation\n({root_feat} > {root_thresh:.2f})", fontsize=14, weight='bold')
    ax.set_xlabel(right_feat, fontsize=12)
    ax.set_ylabel(root_feat, fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(root_thresh - 0.05, 1.05) 
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Failed (Class 0)', 'Edited (Class 1)'], 
              title="Unseen Test Data (20%)", loc='upper right', framealpha=1, edgecolor='black')

    plt.tight_layout()
    output_pdf = "alke_dynamic_holdout.pdf"
    output_png = "alke_dynamic_holdout.png"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Dynamic Holdout plots saved successfully: {output_png}")

if __name__ == "__main__":
    run_dynamic_holdout_and_export("merged_alke_dataset.csv")
