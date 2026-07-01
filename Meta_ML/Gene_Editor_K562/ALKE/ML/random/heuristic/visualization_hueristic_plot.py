import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def generate_quadrant_collaborator_plots(weights_path, clash_path, exclusivity_path, output_png_path, output_pdf_path):
    print("======================================================================")
    print("[*] STARTING MATRICES VISUALIZATION ENGINE: 2x2 QUADRANT SYSTEM")
    print("======================================================================\n")

    if not os.path.exists(weights_path) or not os.path.exists(clash_path) or not os.path.exists(exclusivity_path):
        print("CRITICAL ERROR: Source matrix views for visualization are missing.")
        sys.exit(1)

    # Set professional presentation styling
    sns.set_theme(style="whitegrid", context="talk")
    
    # Structure into a 2x2 grid framework
    fig, axes = plt.subplots(2, 2, figsize=(26, 22))
    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]
    
    # -------------------------------------------------------------------------
    # PLOT 1 (Panel A): Spatial Bin Weight Heatmap
    # -------------------------------------------------------------------------
    print("[*] Drawing Panel A (Top-Left): Spatial Weight Heatmap...")
    df_weights = pd.read_csv(weights_path)
    df_weights = df_weights.set_index('spatial_bin')
    df_weights.columns = [c.replace('weight_editor_', '').upper() for c in df_weights.columns]
    
    sns.heatmap(
        df_weights.T, 
        cmap="RdBu_r", 
        center=0.0, 
        cbar_kws={'label': 'Spatial Weight Coefficient', 'orientation': 'horizontal', 'pad': 0.12}, 
        ax=ax_a,
        xticklabels=False 
    )
    ax_a.set_title("Panel A: High-Resolution Spatial Bin Weight Matrix Profile (Train Pool)", fontsize=16, fontweight='bold', pad=10)
    ax_a.set_xlabel("Genomic Tracking Spatial Bins (70 Continuous Features)", fontsize=13)
    ax_a.set_ylabel("Editor Architectures", fontsize=13)

    # -------------------------------------------------------------------------
    # PLOT 2 (Panel B): Gene-Centric Multi-Editor Clash Grouped Bar
    # -------------------------------------------------------------------------
    print("[*] Drawing Panel B (Top-Right): Multi-Editor Overlap Bar Plot...")
    df_clash = pd.read_csv(clash_path)
    df_clash.columns = [col.lower() for col in df_clash.columns]
    
    top_genes_subset = df_clash.head(6) 
    score_cols = [c for c in df_clash.columns if 'estsigmoid_under_' in c]
    
    df_melted_clash = top_genes_subset.melt(
        id_vars=['gene_name'], 
        value_vars=score_cols,
        var_name='Editor', 
        value_name='Estimated_Sigmoid_Score'
    )
    df_melted_clash['Editor'] = df_melted_clash['Editor'].str.replace('estsigmoid_under_', '').str.upper()
    df_melted_clash['gene_name'] = df_melted_clash['gene_name'].str.upper()

    sns.barplot(
        data=df_melted_clash, 
        x='gene_name', 
        y='Estimated_Sigmoid_Score', 
        hue='Editor', 
        palette='muted', 
        ax=ax_b
    )
    ax_b.set_title("Panel B: Multi-Editor Clashing Profiles across Overlapping Loci (Holdout Pool)", fontsize=16, fontweight='bold', pad=10)
    ax_b.set_xlabel("Target Gene Loci", fontsize=13)
    ax_b.set_ylabel("Estimated Repression Sigmoid Score", fontsize=13)
    ax_b.legend(title="Editor Variants", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=11, title_fontsize=12)

    # -------------------------------------------------------------------------
    # PLOT 3 (Panel C): Distribution Density of Selectivity Index Lifts
    # -------------------------------------------------------------------------
    print("[*] Drawing Panel C (Bottom-Left): Selectivity Lift Distribution...")
    for editor_col in score_cols:
        clean_label = editor_col.replace('estsigmoid_under_', '').upper()
        sns.kdeplot(df_clash[editor_col], label=clean_label, ax=ax_c, lw=2.5, fill=True, alpha=0.03)
        
    ax_c.set_title("Panel C: Population Kernel Density Spread of Predicted Scores", fontsize=16, fontweight='bold', pad=10)
    ax_c.set_xlabel("Calculated High-Resolution Predictive Sigmoid Outputs", fontsize=13)
    ax_c.set_ylabel("Density Scale", fontsize=13)
    ax_c.legend(title="Editor Ensembles", fontsize=11, title_fontsize=12)

    # -------------------------------------------------------------------------
    # PLOT 4 (Panel D): Hyper-Exclusive Targets WITH HIGHLIGHT & DIRECT ANNOTATION
    # -------------------------------------------------------------------------
    print("[*] Drawing Panel D (Bottom-Right): Text-Annotated Exclusivity Plot...")
    df_excl = pd.read_csv(exclusivity_path)
    df_excl.columns = [col.lower() for col in df_excl.columns]
    
    df_excl['dominant_editor'] = df_excl['dominant_editor'].str.replace('EDITOR_', '', case=False).str.upper()
    excl_score_cols = [c for c in df_excl.columns if 'score_under_' in c]
    editor_order = sorted([c.replace('score_under_', '').upper() for c in excl_score_cols])
    
    # Extract the top 1 absolute most exclusive locus target per editor
    selected_excl_genes = []
    for ed in editor_order:
        ed_subset = df_excl[df_excl['dominant_editor'] == ed]
        if not ed_subset.empty:
            top_1_excl = ed_subset.head(1)
            selected_excl_genes.extend(top_1_excl['gene_name'].tolist())
        
    df_excl_subset = df_excl[df_excl['gene_name'].isin(selected_excl_genes)].copy()
    unique_genes = [g.upper() for g in df_excl_subset['gene_name'].unique()]
    
    n_genes = len(unique_genes)
    n_editors = len(editor_order)
    bar_width = 0.13
    x_indices = np.arange(n_genes)

    # Loop manually to individualize every bar color and stamp identity labels natively
    for i, ed in enumerate(editor_order):
        # Align rows cleanly matching global unique index structures
        score_vals = []
        bar_colors = []
        
        for g in unique_genes:
            gene_row = df_excl_subset[df_excl_subset['gene_name'].str.upper() == g].iloc[0]
            score_vals.append(gene_row[f'score_under_{ed.lower()}'])
            
            # Highlight skyblue if it matches the calculated optimal champion, otherwise mute to grey
            is_dominant = (ed == gene_row['dominant_editor'])
            bar_colors.append('#3498db' if is_dominant else '#d1d5db')
            
        bars = ax_d.bar(
            x_indices + i * bar_width, 
            score_vals, 
            width=bar_width, 
            color=bar_colors, 
            edgecolor='white', 
            linewidth=0.7
        )
        
        # Overlay short editor text labels directly on top of the bars to remove legend confusion
        ax_d.bar_label(bars, labels=[ed]*len(bars), rotation=90, label_type='edge', padding=4, fontsize=10, fontweight='semibold')

    # Final axis configuration for Panel D
    ax_d.set_xticks(x_indices + (n_editors - 1) * bar_width / 2)
    ax_d.set_xticklabels(unique_genes, rotation=30, ha='right')
    ax_d.set_title("Panel D: Hyper-Exclusive Loci (Sky-Blue Outshines Muted Grey Alternatives)", fontsize=16, fontweight='bold', pad=10, color='darkblue')
    ax_d.set_xlabel("Exclusivity Anchored Gene Loci (Top Target per Variant)", fontsize=13)
    ax_d.set_ylabel("Estimated Repression Sigmoid Score", fontsize=13)
    ax_d.set_ylim(0, ax_d.get_ylim()[1] + 0.15) # Add headroom for vertical text tags

    # Create clean, descriptive, non-confusing functional legend manually
    legend_elements = [
        Patch(facecolor='#3498db', label='Dominant Editor (Winner at Locus)'),
        Patch(facecolor='#d1d5db', label='Alternative Variants (Runner-up)')
    ]
    ax_d.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Tight layout preservation & Double-format asset logging
    plt.tight_layout()
    
    print("\n[*] Exporting updated quadrant graphics files...")
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    print(f"      --> [SAVED OK] Grid PNG: {output_png_path}")
    
    plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
    print(f"      --> [SAVED OK] Grid PDF: {output_pdf_path}")
    print("\n[-->] Complete. New 2x2 presentation dashboard compiled successfully.")

if __name__ == "__main__":
    working_dir = "./"
    weights_csv_file = os.path.join(working_dir, "highres_calculated_bin_weights.csv")
    clash_csv_file = os.path.join(working_dir, "gene_multi_editor_clash_matrix.csv")
    excl_csv_file = os.path.join(working_dir, "gene_editor_exclusivity_matrix.csv")
    
    final_png_dashboard = os.path.join(working_dir, "crispri_comprehensive_exclusivity_dashboard.png")
    final_pdf_dashboard = os.path.join(working_dir, "crispri_comprehensive_exclusivity_dashboard.pdf")
    
    generate_quadrant_collaborator_plots(
        weights_path=weights_csv_file,
        clash_path=clash_csv_file,
        exclusivity_path=excl_csv_file,
        output_png_path=final_png_dashboard,
        output_pdf_path=final_pdf_dashboard
    )
