import pandas as pd
import numpy as np
import sys
import argparse
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
)

def standardize_test_columns(col, cell_prefix='K562_'):
    if col.startswith(cell_prefix):
        return col.replace(cell_prefix, "", 1)
    elif col.startswith(f"guide_{cell_prefix}"):
        return col.replace(f"guide_{cell_prefix}", "guide_", 1)
    return col

def resolve_biological_synonyms(col):
    if 'DNA_methylation_coverage_bin' in col:
        return col.replace('DNA_methylation_coverage_bin', 'DNA_methylation_bin')
    return col

def run_ensemble_undersampled_testing(data_path, model_path, n_cohorts=10):
    try:
        df = pd.read_csv(data_path, sep=',')
    except Exception as e:
        print(f"Error reading file {data_path}: {e}")
        sys.exit(1)

    # Clean up column prefixes on the fly to match training layout
    df.columns = [standardize_test_columns(c, "K562_") for c in df.columns]
    df.columns = [resolve_biological_synonyms(c) for c in df.columns]

    if 'class' not in df.columns:
        print("Error: 'class' column missing from your independent data.")
        sys.exit(1)

    class_1_pool = df[df['class'] == 1].copy()
    class_0_pool = df[df['class'] == 0].copy()

    n_minus = len(class_0_pool)
    n_plus = len(class_1_pool)

    # Load native XGBoost model structure
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        expected_features = model.get_booster().feature_names
    except Exception as e:
        print(f"Error loading native XGBoost model: {e}")
        sys.exit(1)

    metrics_log = {
        'roc_auc': [], 'aupr': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': [],
        'tp': [], 'tn': [], 'fp': [], 'fn': [], 'specificity': []
    }

    # Iterate over the 10 balanced chunks
    for i in range(n_cohorts):
        sampled_class_1 = class_1_pool.sample(n=n_minus, random_state=42 + i)
        cohort_df = pd.concat([sampled_class_1, class_0_pool], axis=0).reset_index(drop=True)
        
        y_true = cohort_df['class'].astype(int).values
        
        try:
            X_features = cohort_df[expected_features].copy()
        except KeyError:
            missing = set(expected_features) - set(cohort_df.columns)
            print(f"Feature alignment error. Missing columns: {list(missing)}")
            sys.exit(1)

        y_probs = model.predict_proba(X_features)[:, 1]
        y_pred = model.predict(X_features)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_probs)
        aupr_val = auc(r_arr, p_arr)

        metrics_log['roc_auc'].append(roc_auc_score(y_true, y_probs))
        metrics_log['aupr'].append(aupr_val)
        metrics_log['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        metrics_log['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics_log['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics_log['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics_log['specificity'].append(spec)
        metrics_log['tp'].append(tp)
        metrics_log['tn'].append(tn)
        metrics_log['fp'].append(fp)
        metrics_log['fn'].append(fn)

    print("\n" + "="*60)
    print(f"   ENSEMBLE SUMMARY REPORT ({n_cohorts} BALANCED COHORTS)")
    print("="*60)
    print(f"Cohort Matrix Size: {n_minus * 2} rows (1:1 Split: {n_minus} Class 1 / {n_minus} Class 0)")
    print("-"*60)
    print(f"Mean Performance Metrics (± Standard Deviation):")
    print(f"  --> ROC-AUC:              {np.mean(metrics_log['roc_auc']):.4f}  ± {np.std(metrics_log['roc_auc']):.4f}")
    print(f"  --> AUPR (PR-AUC):         {np.mean(metrics_log['aupr']):.4f}  ± {np.std(metrics_log['aupr']):.4f}  [Random Baseline: 0.5000]")
    print(f"  --> Accuracy:             {np.mean(metrics_log['accuracy']):.4f}  ± {np.std(metrics_log['accuracy']):.4f}")
    print(f"  --> F1-Score:             {np.mean(metrics_log['f1']):.4f}  ± {np.std(metrics_log['f1']):.4f}")
    print(f"  --> Precision:            {np.mean(metrics_log['precision']):.4f}  ± {np.std(metrics_log['precision']):.4f}")
    print(f"  --> Recall (Sensitivity): {np.mean(metrics_log['recall']):.4f}  ± {np.std(metrics_log['recall']):.4f}")
    print(f"  --> Specificity (TNR):    {np.mean(metrics_log['specificity']):.4f}  ± {np.std(metrics_log['specificity']):.4f}")
    print("-"*60)
    print(f"Mean Confusion Matrix Cell Counts (Per Cohort Average):")
    print(f"  --> True Positives:  {np.mean(metrics_log['tp']):.1f} / {n_minus}")
    print(f"  --> True Negatives:  {np.mean(metrics_log['tn']):.1f} / {n_minus}")
    print(f"  --> False Positives: {np.mean(metrics_log['fp']):.1f}")
    print(f"  --> False Negatives: {np.mean(metrics_log['fn']):.1f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ensemble Testing via Majority Undersampling')
    parser.add_argument('-d', '--data', default='ML_delta_rig_lower_added.csv')
    parser.add_argument('-m', '--model', default='full_model_xgboost_model.json')
    parser.add_argument('-c', '--cohorts', type=int, default=10)
    
    args = parser.parse_args()
    run_ensemble_undersampled_testing(args.data, args.model, args.cohorts)
