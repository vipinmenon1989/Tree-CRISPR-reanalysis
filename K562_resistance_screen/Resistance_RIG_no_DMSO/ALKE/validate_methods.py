import pandas as pd
from scipy.stats import hypergeom
import argparse
import sys
try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False

def run_essentiality_test(disputed_genes, core_essentials, total_genome_size=20000):
    """Test 2: Hypergeometric test for essential gene overlap."""
    overlap = disputed_genes.intersection(core_essentials)
    
    M = total_genome_size        # Total population
    n = len(core_essentials)     # Total successes in population
    N = len(disputed_genes)      # Sample size drawn
    k = len(overlap)             # Successes drawn

    # P(X >= k) using survival function
    p_value = hypergeom.sf(k - 1, M, n, N)
    
    return len(overlap), p_value

def run_pathway_test(gene_list, list_name):
    """Test 1: GO Biological Process Enrichment using Enrichr."""
    if not HAS_GSEAPY or len(gene_list) == 0:
        return None
    
    try:
        # Query Enrichr for Biological Processes
        enr = gp.enrichr(gene_list=list(gene_list),
                         gene_sets='GO_Biological_Process_2023',
                         organism='human',
                         outdir=None)
        
        # Filter for adjusted p-value < 0.05 and sort
        results = enr.results
        sig_results = results[results['Adjusted P-value'] < 0.05].head(5)
        return sig_results[['Term', 'Adjusted P-value', 'Overlap']]
    except Exception as e:
        return str(e)

def main():
    parser = argparse.ArgumentParser(description='Compare GI Normalization vs Raw RIG pipelines.')
    parser.add_argument('--gi', required=True, help='Path to your GI Hits file')
    parser.add_argument('--raw', required=True, help='Path to PI\'s Raw Hits file')
    parser.add_argument('--essentials', required=True, help='Path to text file of core essential genes')
    parser.add_argument('--bg', type=int, default=20000, help='Total genes in screen (Background size)')
    args = parser.parse_args()

    # 1. Load Data
    try:
        gi_df = pd.read_csv(args.gi, sep='\t')
        raw_df = pd.read_csv(args.raw, sep='\t')
        with open(args.essentials, 'r') as f:
            essentials_list = set([line.strip() for line in f if line.strip()])
    except Exception as e:
        print(f"CRITICAL ERROR loading files: {e}")
        sys.exit(1)

    set_gi = set(gi_df['Gene'].dropna().tolist())
    set_raw = set(raw_df['Gene'].dropna().tolist())

    if len(set_gi) != len(set_raw):
        print("WARNING: Lists are not the same length! Set subtraction metrics may be skewed.")

    # 2. Isolate the Disputed Cohorts
    pi_noise = set_raw - set_gi     # Genes PI kept, but you threw away (Suspected Proliferators)
    your_rescue = set_gi - set_raw  # Genes you kept, but PI missed (Suspected True Resistance)

    print("=" * 60)
    print(" PIPELINE SHOWDOWN: GI SCORE vs RAW LFC")
    print("=" * 60)
    print(f"Total GI Hits: {len(set_gi)}")
    print(f"Total Raw Hits: {len(set_raw)}")
    print(f"Disputed Genes (Unique to PI's Raw List): {len(pi_noise)}")
    print(f"Disputed Genes (Unique to Your GI List): {len(your_rescue)}")
    
    # ---------------------------------------------------------
    # TEST 2: THE ESSENTIALITY TEST
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" TEST 2: THE BASAL PROLIFERATION TEST (ESSENTIALITY)")
    print("=" * 60)
    
    overlap_pi, p_val_pi = run_essentiality_test(pi_noise, essentials_list, args.bg)
    print("Analyzing the genes unique to the PI's Raw method:")
    print(f"Overlap with Core Essentials: {overlap_pi} out of {len(pi_noise)}")
    print(f"Hypergeometric P-Value: {p_val_pi:.2e}")
    
    if p_val_pi < 0.05:
        print("-> CONCLUSION: PI's method is mathematically polluted with generic essential genes.")
    else:
        print("-> CONCLUSION: PI's method is NOT significantly polluted by essential genes.")

    # ---------------------------------------------------------
    # TEST 1: THE PATHWAY PURITY TEST
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" TEST 1: THE PATHWAY PURITY TEST (GO ENRICHMENT)")
    print("=" * 60)

    if not HAS_GSEAPY:
        print("Please install 'gseapy' to run automatic pathway enrichment.")
        print("Alternatively, copy the unique genes into the Enrichr website (https://maayanlab.cloud/Enrichr/).")
    else:
        print("Top 5 Enriched Pathways in PI's Raw List (Total List):")
        raw_pw = run_pathway_test(set_raw, "Raw")
        print(raw_pw if isinstance(raw_pw, pd.DataFrame) else "No significant pathways found.")
        
        print("\nTop 5 Enriched Pathways in Your GI List (Total List):")
        gi_pw = run_pathway_test(set_gi, "GI")
        print(gi_pw if isinstance(gi_pw, pd.DataFrame) else "No significant pathways found.")

if __name__ == "__main__":
    main()
