import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def main():
    print("[INFO] Loading comprehensive reports...")
    # 1. Load Data
    try:
        df_a = pd.read_csv('Comprehensive_CRISPRa_report.csv').dropna()
        df_i = pd.read_csv('Comprehensive_report_CRISPRi.csv').dropna()
    except Exception as e:
        print(f"[ERROR] Could not load files: {e}")
        sys.exit(1)

    # 2. Melt DataFrames to separate sgRNA and Gene counts for faceting
    df_a_melt = df_a.melt(id_vars=['Editor', 'Cell line', 'Screen'], 
                          value_vars=['sgRNA', 'gene'], 
                          var_name='Metric', value_name='Count')
    df_a_melt['Condition'] = df_a_melt['Cell line'] + " (" + df_a_melt['Screen'] + ")"

    df_i_melt = df_i.melt(id_vars=['Editor', 'Cell line', 'Screen'], 
                          value_vars=['sgRNA', 'gene'], 
                          var_name='Metric', value_name='Count')
    df_i_melt['Condition'] = df_i_melt['Cell line'] + " (" + df_i_melt['Screen'] + ")"

    sns.set_style("whitegrid")

    # ==========================================
    # FIGURE 1: CRISPRa
    # ==========================================
    print("[INFO] Generating CRISPRa Visualizations...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=df_a_melt[df_a_melt['Metric']=='sgRNA'], x='Condition', y='Count', hue='Editor', ax=axes1[0], palette='viridis')
    axes1[0].set_title('CRISPRa: Total Validated sgRNAs', fontweight='bold')
    axes1[0].set_ylabel('Number of sgRNAs')
    axes1[0].set_xlabel('')

    sns.barplot(data=df_a_melt[df_a_melt['Metric']=='gene'], x='Condition', y='Count', hue='Editor', ax=axes1[1], palette='viridis')
    axes1[1].set_title('CRISPRa: Total Validated Genes', fontweight='bold')
    axes1[1].set_ylabel('Number of Genes')
    axes1[1].set_xlabel('')

    plt.tight_layout()
    # Save both formats
    plt.savefig('CRISPRa_Summary_Plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('CRISPRa_Summary_Plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ==========================================
    # FIGURE 2: CRISPRi
    # ==========================================
    print("[INFO] Generating CRISPRi Visualizations...")
    # Force 'K' to be the baseline color, followed by engineered ALK variants
    editor_order = ['K', 'ALKE', 'ALKL', 'ALKH', 'ALKME', 'ALKML', 'ALKMH']
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 12))

    sns.barplot(data=df_i_melt[df_i_melt['Metric']=='sgRNA'], x='Condition', y='Count', hue='Editor', hue_order=editor_order, ax=axes2[0], palette='coolwarm')
    axes2[0].set_title('CRISPRi: Total Validated sgRNAs', fontweight='bold', fontsize=14)
    axes2[0].set_ylabel('Number of sgRNAs', fontsize=12)
    axes2[0].set_xlabel('')
    axes2[0].legend(title='Architecture', bbox_to_anchor=(1.01, 1), loc='upper left')

    sns.barplot(data=df_i_melt[df_i_melt['Metric']=='gene'], x='Condition', y='Count', hue='Editor', hue_order=editor_order, ax=axes2[1], palette='coolwarm')
    axes2[1].set_title('CRISPRi: Total Validated Genes', fontweight='bold', fontsize=14)
    axes2[1].set_ylabel('Number of Genes', fontsize=12)
    axes2[1].set_xlabel('Screen Condition', fontsize=12)
    axes2[1].legend(title='Architecture', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    # Save both formats
    plt.savefig('CRISPRi_Summary_Plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('CRISPRi_Summary_Plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print("------------------------------------------------------------")
    print("[SUCCESS] All plots successfully saved in both .pdf and .png formats.")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
