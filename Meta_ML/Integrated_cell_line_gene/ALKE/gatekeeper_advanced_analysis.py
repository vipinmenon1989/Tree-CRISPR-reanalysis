#!/usr/bin/env python3
"""
Promoter Epigenetic Features vs. CRISPRi Screening Results Analysis Pipeline

This script implements a publication-ready statistical workflow evaluating the 
relationship between binned epigenetic promoter modifications and gene-level 
CRISPRi repression efficiency scores.

Optimized for HPC headless execution and modern Seaborn categorical mappings.
Author: Bioinformatics Analytics Pipeline
Year: 2026
"""

import os
import re
import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd

# Enforce headless mode for HPC cluster environments before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# --- CONFIGURATION & LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_execution.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Plotting aesthetics for high-impact journals
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# --- STATISTICAL ENGINE ---

def calculate_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Computes Cliff's Delta effect size directly using NumPy vectorization.
    Formula: d = (2 * U / (n1 * n2)) - 1
    """
    n1 = len(group1)
    n2 = len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    u_stat, _ = mannwhitneyu(group1, group2, alternative='two-sided')
    delta = (2.0 * u_stat / (n1 * n2)) - 1.0
    return delta


def parse_feature_name(feature_name: str) -> Tuple[str, int]:
    """
    Dynamically extracts the epigenetic mark and bin number from a feature name string.
    Example: 'h3k27ac_bin_10' -> ('h3k27ac', 10)
    """
    match = re.match(r"(.+)_bin_(\d+)$", feature_name)
    if match:
        mark = match.group(1)
        bin_num = int(match.group(2))
        return mark, bin_num
    else:
        return feature_name, 0


# --- PIPELINE ENGINE ---

class EpigeneticAnalysisPipeline:
    def __init__(self, data_path: str, output_dir: str = "publication_outputs"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.results_df: Optional[pd.DataFrame] = None
        
        self.dirs = {
            "base": output_dir,
            "tables": os.path.join(output_dir, "summary_tables"),
            "threshold_plots": os.path.join(output_dir, "plots", "threshold_analysis"),
            "class_plots": os.path.join(output_dir, "plots", "efficiency_analysis"),
            "scatter_plots": os.path.join(output_dir, "plots", "scatter_plots"),
            "heatmaps": os.path.join(output_dir, "plots", "heatmaps"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def load_and_validate_data(self) -> None:
        """Loads and filters target analysis features from dataset."""
        logger.info(f"Loading dataset from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Input file {self.data_path} not found.")
            
        self.df = pd.read_csv(self.data_path)
        
        required_cols = ['ID', 'gene', 'mean_sigmoid_score', 'class']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing critical metadata column: {col}")
        
        # Isolate variables and enforce clean data typing
        self.df['class'] = self.df['class'].astype(int)
        self.df['mean_sigmoid_score'] = self.df['mean_sigmoid_score'].astype(float)
        
        self.features = [c for c in self.df.columns if c not in required_cols]
        logger.info(f"Successfully tracked {len(self.features)} epigenetic track variables.")

    def run_statistical_analyses(self) -> None:
        """Executes targeted multi-tier statistical tests across all isolated features."""
        logger.info("Initiating core statistical engine processing loops...")
        raw_results = []
        
        for feat in self.features:
            subset = self.df[[feat, 'class', 'mean_sigmoid_score']].dropna().copy()
            if len(subset) < 5:
                logger.warning(f"Feature {feat} has insufficient non-null instances. Skipping.")
                continue
                
            subset['class'] = subset['class'].astype(int)
            mark, bin_num = parse_feature_name(feat)
            
            # --- Analysis 1: Continuous Association (Spearman) ---
            rho, spearman_p = spearmanr(subset[feat], subset['mean_sigmoid_score'])
            
            # --- Analysis 2: Threshold Analysis (<=0.5 vs >0.5) ---
            low_grp = subset[subset[feat] <= 0.5]['mean_sigmoid_score'].values
            high_grp = subset[subset[feat] > 0.5]['mean_sigmoid_score'].values
            
            n_low, n_high = len(low_grp), len(high_grp)
            med_low, med_high = np.median(low_grp) if n_low > 0 else np.nan, np.median(high_grp) if n_high > 0 else np.nan
            
            if n_low > 0 and n_high > 0:
                u_thresh, p_thresh = mannwhitneyu(low_grp, high_grp, alternative='two-sided')
                delta_thresh = calculate_cliffs_delta(low_grp, high_grp)
            else:
                u_thresh, p_thresh, delta_thresh = np.nan, np.nan, np.nan

            # --- Analysis 3: Efficient vs Inefficient Analysis (Class 0 vs Class 1) ---
            c0_grp = subset[subset['class'] == 0][feat].values
            c1_grp = subset[subset['class'] == 1][feat].values
            
            n_c0, n_c1 = len(c0_grp), len(c1_grp)
            med_c0, med_c1 = np.median(c0_grp) if n_c0 > 0 else np.nan, np.median(c1_grp) if n_c1 > 0 else np.nan
            
            if n_c0 > 0 and n_c1 > 0:
                u_class, p_class = mannwhitneyu(c0_grp, c1_grp, alternative='two-sided')
                delta_class = calculate_cliffs_delta(c0_grp, c1_grp)
            else:
                u_class, p_class, delta_class = np.nan, np.nan, np.nan
                
            raw_results.append({
                'Feature': feat, 'Histone_Mark': mark, 'Bin': bin_num,
                'Spearman_Rho': rho, 'Spearman_P': spearman_p,
                'Thresh_U': u_thresh, 'Thresh_P': p_thresh, 'Thresh_Delta': delta_thresh,
                'Thresh_Med_Low': med_low, 'Thresh_Med_High': med_high, 'Thresh_N_Low': n_low, 'Thresh_N_High': n_high,
                'Class_U': u_class, 'Class_P': p_class, 'Class_Delta': delta_class,
                'Class_Med_0': med_c0, 'Class_Med_1': med_c1, 'Class_N_0': n_c0, 'Class_N_1': n_c1
            })

        res_df = pd.DataFrame(raw_results)
        
        # Apply corrections completely independently per test tier
        for p_col, fdr_col in [('Spearman_P', 'Spearman_FDR'), 
                               ('Thresh_P', 'Thresh_FDR'), 
                               ('Class_P', 'Class_FDR')]:
            valid_mask = res_df[p_col].notna()
            if valid_mask.any():
                _, adjusted_p, _, _ = multipletests(res_df.loc[valid_mask, p_col], method='fdr_bh')
                res_df.loc[valid_mask, fdr_col] = adjusted_p
            else:
                res_df[fdr_col] = np.nan
                
        self.results_df = res_df
        logger.info("Statistical tests complete. Multiple testing corrections applied.")

    def export_summary_tables(self) -> None:
        """Generates required output matrices."""
        logger.info("Exporting summary data tables...")
        
        master_file = os.path.join(self.dirs["tables"], "master_analysis_report.csv")
        self.results_df.to_csv(master_file, index=False)
        
        sig_mask = (self.results_df['Spearman_FDR'] < 0.05) | \
                   (self.results_df['Thresh_FDR'] < 0.05) | \
                   (self.results_df['Class_FDR'] < 0.05)
        self.results_df[sig_mask].to_csv(os.path.join(self.dirs["tables"], "significant_features.csv"), index=False)
        
        ranked_df = self.results_df.copy()
        ranked_df['Abs_Thresh_Delta'] = ranked_df['Thresh_Delta'].abs()
        ranked_df['Abs_Spearman_Rho'] = ranked_df['Spearman_Rho'].abs()
        ranked_df = ranked_df.sort_values(
            by=['Thresh_FDR', 'Abs_Thresh_Delta', 'Abs_Spearman_Rho'],
            ascending=[True, False, False]
        ).drop(columns=['Abs_Thresh_Delta', 'Abs_Spearman_Rho'])
        
        ranked_df.to_csv(os.path.join(self.dirs["tables"], "top_ranked_features.csv"), index=False)

    def generate_publication_plots(self) -> None:
        """Produces all metric visualization frames."""
        logger.info("Rendering publication-quality visualization arrays...")
        
        # Bulletproof categorical type mapping to satisfy strict Seaborn palette parsing
        class_palette = {0: "#E74C3C", 1: "#3498DB", '0': "#E74C3C", '1': "#3498DB"}
        
        for _, row in self.results_df.iterrows():
            feat = row['Feature']
            subset = self.df[[feat, 'class', 'mean_sigmoid_score']].dropna().copy()
            
            # Explicit string cast to resolve "Using categorical units..." warning log anomalies
            subset['class_str'] = subset['class'].astype(str)
            subset['Intensity_Category'] = subset[feat].apply(lambda x: 'Low (≤ 0.5)' if x <= 0.5 else 'High (> 0.5)')
            
            # --- Figure 1: Threshold Impact Boxplot ---
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            sns.boxplot(
                x='Intensity_Category', y='mean_sigmoid_score', hue='Intensity_Category', data=subset,
                order=['Low (≤ 0.5)', 'High (> 0.5)'], palette="Blues", width=0.5, ax=ax, showfliers=False, legend=False
            )
            sns.stripplot(x='Intensity_Category', y='mean_sigmoid_score', data=subset, order=['Low (≤ 0.5)', 'High (> 0.5)'], color="black", alpha=0.15, size=3, jitter=0.2, ax=ax)
            
            title_t = (
                f"{feat}\n"
                f"MW p={row['Thresh_P']:.2e} | FDR={row['Thresh_FDR']:.2e}\n"
                f"Cliff's Δ={row['Thresh_Delta']:.2f} | ρ={row['Spearman_Rho']:.2f} (p={row['Spearman_P']:.2e})\n"
                f"Medians: Low={row['Thresh_Med_Low']:.2f}, High={row['Thresh_Med_High']:.2f}\n"
                f"N: Low={int(row['Thresh_N_Low'])}, High={int(row['Thresh_N_High'])}"
            )
            ax.set_title(title_t, fontsize=8, weight='bold')
            ax.set_ylabel("CRISPRi Repression Score (mean_sigmoid_score)")
            ax.set_xlabel("Epigenetic Intensity")
            plt.tight_layout()
            
            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(self.dirs["threshold_plots"], f"{feat}_threshold_boxplot.{fmt}"), dpi=300)
            plt.close(fig)
            
            # --- Figure 2: Efficient vs Inefficient Boxplot ---
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            sns.boxplot(
                x='class_str', y=feat, hue='class_str', data=subset, order=['0', '1'], 
                palette=class_palette, width=0.5, ax=ax, showfliers=False, legend=False
            )
            sns.stripplot(x='class_str', y=feat, data=subset, order=['0', '1'], color="black", alpha=0.15, size=3, jitter=0.2, ax=ax)
            
            title_c = (
                f"Class Separation: {feat}\n"
                f"MW p={row['Class_P']:.2e} | FDR={row['Class_FDR']:.2e}\n"
                f"Cliff's Δ={row['Class_Delta']:.2f}\n"
                f"Medians: C0={row['Class_Med_0']:.2f}, C1={row['Class_Med_1']:.2f}\n"
                f"N: C0={int(row['Class_N_0'])}, C1={int(row['Class_N_1'])}"
            )
            ax.set_title(title_c, fontsize=8, weight='bold')
            ax.set_ylabel("Normalized Feature Value")
            ax.set_xlabel("CRISPRi Class Designation")
            plt.tight_layout()
            
            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(self.dirs["class_plots"], f"{feat}_class_boxplot.{fmt}"), dpi=300)
            plt.close(fig)
            
            # --- Figure 3: Scatter Plots ---
            if row['Thresh_FDR'] < 0.05 and abs(row['Spearman_Rho']) >= 0.20:
                fig, ax = plt.subplots(figsize=(5, 4.5))
                sns.regplot(
                    x=feat, y='mean_sigmoid_score', data=subset, ax=ax,
                    scatter_kws={'alpha': 0.3, 'color': '#2C3E50', 's': 15},
                    line_kws={'color': '#E74C3C', 'linewidth': 2}
                )
                ax.set_title(f"{feat} Association\nρ = {row['Spearman_Rho']:.2f} (p = {row['Spearman_P']:.2e})", weight='bold')
                ax.set_xlabel("Normalized Epigenetic Feature")
                ax.set_ylabel("CRISPRi Repression Score")
                plt.tight_layout()
                
                for fmt in ['png', 'pdf']:
                    fig.savefig(os.path.join(self.dirs["scatter_plots"], f"{feat}_scatter.{fmt}"), dpi=300)
                plt.close(fig)

    def generate_positional_heatmaps(self) -> None:
        """Generates dynamic promoter position matrix heatmaps across all features."""
        logger.info("Structuring relational metric positional matrix heatmaps...")
        pivot_base = self.results_df.copy()
        pivot_base['-log10_Thresh_FDR'] = -np.log10(pivot_base['Thresh_FDR'] + 1e-300)
        
        metrics = {
            'Spearman_Rho': ('Spearman Correlation (ρ)', 'vlag', self.dirs["heatmaps"]),
            'Thresh_Delta': ("Cliff's Delta Effect Size", 'coolwarm', self.dirs["heatmaps"]),
            '-log10_Thresh_FDR': ('-log10(Threshold FDR Adjusted P-values)', 'rocket', self.dirs["heatmaps"])
        }
        
        for col, (label, cmap, path) in metrics.items():
            heatmap_data = pivot_base.pivot(index='Histone_Mark', columns='Bin', values=col)
            heatmap_data = heatmap_data.reindex(columns=sorted(heatmap_data.columns))
            
            if heatmap_data.empty:
                continue
                
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'label': label}, ax=ax)
            ax.set_title(f"Promoter Feature Architecture Profile Map: {label}", weight='bold')
            ax.set_xlabel("Promoter Bins (Directional TSS 1-10)")
            ax.set_ylabel("Epigenetic Factor")
            plt.tight_layout()
            
            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(path, f"promoter_heatmap_{col.lower()}.{fmt}"), dpi=300)
            plt.close(fig)

    def compile_final_summary_report(self) -> None:
        """Generates and prints the final analytical synthesis report."""
        df_res = self.results_df
        total_feats = len(df_res)
        sig_before = (df_res['Thresh_P'] < 0.05).sum()
        sig_after = (df_res['Thresh_FDR'] < 0.05).sum()
        
        strongest_pos = df_res.loc[df_res['Spearman_Rho'].idxmax()] if total_feats > 0 else None
        strongest_neg = df_res.loc[df_res['Spearman_Rho'].idxmin()] if total_feats > 0 else None
        largest_effect = df_res.loc[df_res['Thresh_Delta'].abs().idxmax()] if total_feats > 0 else None
        
        grouped = df_res.groupby('Histone_Mark')
        informative_bins = {}
        consistency_trends = {}
        
        for mark, group in grouped:
            best_bin_idx = group['Thresh_FDR'].idxmin()
            informative_bins[mark] = group.loc[best_bin_idx, 'Bin']
            rhos = group['Spearman_Rho'].values
            if all(r >= 0 for r in rhos):
                consistency_trends[mark] = "Consistently Positive across all bins"
            elif all(r <= 0 for r in rhos):
                consistency_trends[mark] = "Consistently Negative across all bins"
            else:
                consistency_trends[mark] = "Variable / Non-monotonic trend profile across bins"

        report_path = os.path.join(self.output_dir, "executive_summary_report.txt")
        with open(report_path, "w") as f:
            f.write("========================================================================\n")
            f.write("   EXECUTIVE BIOINFORMATICS PIPELINE RESULTS REPORT (PUBLICATION READY) \n")
            f.write("========================================================================\n\n")
            f.write(f"1. FEATURE DIMENSIONALITY ASSESSMENT:\n")
            f.write(f"   * Number of analyzed promoter tracking features: {total_feats}\n")
            f.write(f"   * Number of significant features before FDR correction (α=0.05): {sig_before}\n")
            f.write(f"   * Number of significant features after FDR correction (FDR<0.05): {sig_after}\n\n")
            
            if strongest_pos is not None:
                f.write(f"2. CORRELATION & EFFECT SIZE HIGHLIGHTS:\n")
                f.write(f"   * Strongest Positive Correlation Track: {strongest_pos['Feature']} (ρ = {strongest_pos['Spearman_Rho']:.3f}, FDR = {strongest_pos['Spearman_FDR']:.2e})\n")
                f.write(f"   * Strongest Negative Correlation Track: {strongest_neg['Feature']} (ρ = {strongest_neg['Spearman_Rho']:.3f}, FDR = {strongest_neg['Spearman_FDR']:.2e})\n")
                f.write(f"   * Largest Threshold Spatial Effect Size: {largest_effect['Feature']} (Cliff's Δ = {largest_effect['Thresh_Delta']:.3f}, FDR = {largest_effect['Thresh_FDR']:.2e})\n\n")
            
            f.write(f"3. DYNAMIC TRANSCRIPTION START SITE (TSS) BIN ARCHITECTURE ANALYSIS:\n")
            for mark in informative_bins:
                f.write(f"   * Histone/Modification Track [{mark}]:\n")
                f.write(f"     - Most Informative / Potent Bin Segment: Bin {informative_bins[mark]}\n")
                f.write(f"     - Positional Directionality Trend: {consistency_trends[mark]}\n")
            f.write("\n========================================================================\n")

        with open(report_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    TARGET_DATASET = "merged_alke_dataset.csv"
    OUTPUT_DIRECTORY = "publication_plots"
    
    try:
        pipeline = EpigeneticAnalysisPipeline(data_path=TARGET_DATASET, output_dir=OUTPUT_DIRECTORY)
        pipeline.load_and_validate_data()
        pipeline.run_statistical_analyses()
        pipeline.export_summary_tables()
        pipeline.generate_publication_plots()
        pipeline.generate_positional_heatmaps()
        pipeline.compile_final_summary_report()
    except Exception as e:
        logger.error(f"Fatal execution interruption in analysis pipeline runtime loop: {str(e)}", exc_info=True)
