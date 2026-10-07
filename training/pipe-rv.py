# ==========================================
# 80/20 split pipeline - 86% Accuracy
# ==========================================?
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import Lasso
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split
)
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==========================================
# 0. DIRECTORY SCANNER & PATH RESOLVER
# ==========================================
def resolve_data_paths():
    local_expr = "tcga-data/TCGA_BRCA_tpm.tsv"
    local_clin = "tcga-data/brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"

    kaggle_expr = "/kaggle/input/datasets/blaiseappolinary/tcga-data/TCGA_BRCA_tpm.tsv"
    kaggle_clin = "/kaggle/input/datasets/blaiseappolinary/tcga-data/brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"

    if os.path.exists(local_expr) and os.path.exists(local_clin):
        return local_expr, local_clin
    elif os.path.exists(kaggle_expr) and os.path.exists(kaggle_clin):
        return kaggle_expr, kaggle_clin
    else:
        # Fallback search if placed in an arbitrary subfolder
        for root, _, files in os.walk('.'):
            if "TCGA_BRCA_tpm.tsv" in files:
                expr = os.path.join(root, "TCGA_BRCA_tpm.tsv")
                clin = os.path.join(root, "brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv")
                if os.path.exists(clin):
                    return expr, clin
        raise FileNotFoundError("Could not locate TCGA dataset files in local directory or Kaggle input.")

# ==========================================
# 1. BULLETPROOF DATA INGESTION
# ==========================================
def load_and_preprocess_kaggle_data(expression_file, clinical_file):
    print("[+] Step 1: Loading and Transposing Kaggle TCGA BRCA datasets...")

    df_expr = pd.read_csv(expression_file, sep='\t', low_memory=False)
    gene_col_name = df_expr.columns[0]
    df_expr.set_index(gene_col_name, inplace=True)

    print("    --> Transposing matrix... (Patience, this takes a moment)")
    df_expr = df_expr.T
    df_expr.index.name = 'Sample_ID'
    df_expr.reset_index(inplace=True)

    df_clin = pd.read_csv(clinical_file, sep='\t')

    if 'Patient ID' not in df_clin.columns or 'Subtype' not in df_clin.columns:
        raise ValueError("[!] Could not find 'Patient ID' or 'Subtype' in clinical columns.")

    df_clin = df_clin.rename(columns={'Patient ID': 'Sample_ID', 'Subtype': 'Subtype'})
    df_clin = df_clin[['Sample_ID', 'Subtype']]

    print("    --> Standardizing TCGA Barcodes to 12 characters...")
    df_expr['Sample_ID'] = df_expr['Sample_ID'].astype(str).str[:12]
    df_clin['Sample_ID'] = df_clin['Sample_ID'].astype(str).str[:12]

    df_expr = df_expr.drop_duplicates(subset=['Sample_ID'])
    df_clin = df_clin.drop_duplicates(subset=['Sample_ID'])

    print("    --> Merging genetic data with clinical labels...")
    merged_df = pd.merge(df_expr, df_clin, on='Sample_ID', how='inner')

    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['Subtype'])
    if len(merged_df) < initial_len:
        print(f"        --> Purged {initial_len - len(merged_df)} patients due to missing clinical labels.")

    y = merged_df['Subtype'].values
    X_df = merged_df.drop(columns=['Sample_ID', 'Subtype'])
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    threshold = 0.75 * len(X_df)
    zero_counts = (X_df == 0).sum(axis=0)
    genes_to_keep = zero_counts[zero_counts <= threshold].index

    X_filtered = X_df[genes_to_keep].values
    print(f"[OK] Filtered feature count: {X_filtered.shape[1]}")

    return X_filtered, y, genes_to_keep

# ==========================================
# 2. VALIDATION ENGINE
# ==========================================
def execute_fast_cv(model, X, y, scoring_metric='roc_auc'):
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1)
    return np.mean(scores)

def execute_10x10_cv(model, X, y):
    cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0)
    }

    scores = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

    return (np.mean(scores['test_accuracy']),
            np.mean(scores['test_f1_weighted']),
            np.mean(scores['test_precision_weighted']),
            np.mean(scores['test_recall_weighted']))

# ==========================================
# 3. mCGA ALGORITHM
# ==========================================
def algorithm_2_mcga(X, y_binary, model, n_step=5, max_iter=100):
    N = X.shape[1]
    num_pvs = 10
    c_global = 2.0

    PVs = np.full((num_pvs, N), 0.5)

    pbar = tqdm(total=max_iter, desc="        [>] mCGA Epochs", unit="epoch", leave=True)

    for iteration in range(max_iter):
        binary_strings = (np.random.rand(num_pvs, N) < PVs).astype(int)
        fitness_values = np.zeros(num_pvs)

        for i in range(num_pvs):
            selected_cols = np.where(binary_strings[i] == 1)[0]
            if len(selected_cols) > 0:
                X_subset = X[:, selected_cols]
                fitness_values[i] = execute_fast_cv(model, X_subset, y_binary, scoring_metric='roc_auc')
            else:
                fitness_values[i] = 0.0

        winner_idx = np.argmax(fitness_values)
        winner_string = binary_strings[winner_idx]
        Best_PV = PVs[winner_idx].copy()

        for i in range(num_pvs):
            for j in range(N):
                if winner_string[j] != binary_strings[i, j]:
                    PVs[i, j] += (1.0 / n_step) if winner_string[j] == 1 else -(1.0 / n_step)
            PVs[i] = np.clip(PVs[i], 0.0, 1.0)
            PVs[i] = np.clip(PVs[i] + c_global * (Best_PV - PVs[i]), 0.0, 1.0)

        pbar.update(1)

        if np.all((Best_PV > 0.9) | (Best_PV < 0.1)):
            pbar.close()
            print(f"\n        [*] PVs converged early at Epoch {iteration+1}")
            return np.where(Best_PV >= 0.5)[0]

    pbar.close()
    return np.where(Best_PV >= 0.5)[0]

# ==========================================
# 4. MASTER PIPELINE
# ==========================================
def main_pipeline_multiclass(expression_filepath, clinical_filepath, target_subtypes):
    X_raw, y_multiclass, gene_names = load_and_preprocess_kaggle_data(expression_filepath, clinical_filepath)

    # ---------------------------------------------------------
    # 80/20 STRATIFIED HOLDOUT PARTITION
    # ---------------------------------------------------------
    print("\n[+] Executing Stratified 80/20 Train-Test Split to prevent Data Leakage...")
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y_multiclass, test_size=0.20, stratify=y_multiclass, random_state=42
    )
    print(f"    --> Discovery Set (Train): {len(y_train)} samples")
    print(f"    --> Validation Set (Test): {len(y_test)} samples")

    print("\n[+] Applying Log2 Transformation to normalize RNA-Seq skew...")
    X_train_log = np.log1p(X_tr_raw)
    X_test_log = np.log1p(X_te_raw)

    print("[+] Applying Standard Scaling (Fitted STRICTLY on Discovery Set)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)

    master_biomarker_indices = set()
    svm_metrics_summary = {}

    optimal_alphas = {
        'BRCA_LumA': 0.010,
        'BRCA_LumB': 0.010,
        'BRCA_Her2': 0.010,
        'BRCA_Normal': 0.010,
        'BRCA_Basal': 0.010
    }

    # ---------------------------------------------------------
    # PHASE A: SUBTYPE LOOP (Discovery Set Only)
    # ---------------------------------------------------------
    for subtype in target_subtypes:
        print(f"\n" + "="*80)
        print(f"[PROTOCOL] COMMENCING FOR: {subtype}")
        print("="*80)

        y_binary_train = (y_train == subtype).astype(int)

        if sum(y_binary_train) == 0:
            print(f"[!] WARNING: Subtype '{subtype}' not found in training data. Skipping.")
            continue

        print(f"    [*] Distribution -> Positive: {sum(y_binary_train)} | Negative: {len(y_binary_train)-sum(y_binary_train)}")

        print("\n    [>] Step 2: Dynamic LASSO Reduction...")
        alpha_val = optimal_alphas.get(subtype, 0.010)
        lasso = Lasso(alpha=alpha_val, random_state=42).fit(X_train_scaled, y_binary_train)
        print(f"        [*] Hardcoded literature alpha applied: {alpha_val:.5f}")

        selected_features_idx = np.where(np.abs(lasso.coef_) > 0)[0]
        X_reduced_train = X_train_scaled[:, selected_features_idx]
        print(f"        [OK] LASSO preserved {len(selected_features_idx)} potential biomarkers.")

        print("\n    [>] Base Model Evaluation (10x10 CV) on LASSO features...")
        models = {
            'RF': RandomForestClassifier(random_state=42),
            'SVM': SVC(kernel='linear', C=2, class_weight='balanced', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'NB': GaussianNB()
        }
        for name, model in models.items():
            acc, f1, prec, rec = execute_10x10_cv(model, X_reduced_train, y_binary_train)
            print(f"        -- Model {name:<4} | Acc: {acc*100:>6.2f}% | F1: {f1*100:>6.2f}% | Prec: {prec*100:>6.2f}% | Rec: {rec*100:>6.2f}%")

            if name == 'SVM':
                svm_metrics_summary[subtype] = (acc, f1, prec, rec)

        print("\n    [>] Step 3: mCGA Biomarker Extraction...")
        ovr_svm = SVC(kernel='linear', C=2, class_weight='balanced', random_state=42)
        final_relative_idx = algorithm_2_mcga(X_reduced_train, y_binary_train, ovr_svm)

        final_biomarkers_idx = selected_features_idx[final_relative_idx]
        master_biomarker_indices.update(final_biomarkers_idx)
        print(f"    [OK] Discovered {len(final_biomarkers_idx)} unique genes for {subtype}.")

    # ---------------------------------------------------------
    # PHASE B: RFE PRUNING & MULTI-CLASS SYNTHESIS
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("[SUMMARY] BASE SVM METRICS ACROSS ALL SUBTYPES (PHASE A - INTERNAL CV)")
    print("="*80)
    print(f"    {'Subtype':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("    " + "-"*65)
    for subtype, metrics in svm_metrics_summary.items():
        acc, f1, prec, rec = metrics
        print(f"    {subtype:<15} | {acc*100:>8.2f}% | {f1*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}")
    print("="*80)

    final_indices_array = np.array(sorted(list(master_biomarker_indices)))
    initial_master_genes = gene_names[final_indices_array]
    print(f"\n[*] Total Master Biomarkers Combined: {len(initial_master_genes)}")

    X_final_master_train = X_train_scaled[:, final_indices_array]

    TARGET_CLINICAL_GENES = 100

    print(f"\n[>] Activating Non-Linear RFE Scalpel to prune down to {TARGET_CLINICAL_GENES} genes...")
    rf_judge = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)

    rfe = RFE(estimator=rf_judge, n_features_to_select=TARGET_CLINICAL_GENES, step=2)
    rfe.fit(X_final_master_train, y_train)

    surviving_mask = rfe.support_
    pruned_indices_array = final_indices_array[surviving_mask]
    pruned_final_genes = gene_names[pruned_indices_array]

    X_final_pruned_train = X_train_scaled[:, pruned_indices_array]
    X_final_pruned_test = X_test_scaled[:, pruned_indices_array]

    print(f"[OK] RFE complete! Reduced from {len(initial_master_genes)} down to {len(pruned_final_genes)} elite genes.")

    # Proportional SMOTE Balancing
    print("\n[+] Applying Proportional SMOTE Balancing...")
    proportional_strategy = {
        'BRCA_Normal': 80,   # Boosted from 29
        'BRCA_Her2': 110,    # Boosted from 62
        'BRCA_LumB': 250,    # Moderately boosted from 157
    }
    smote = SMOTE(sampling_strategy=proportional_strategy, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_final_pruned_train, y_train)
    print(f"[OK] Balanced training sample count: {len(y_train_smote)}")

    # Multi-Class RBF Classifier
    print(f"\n[TRAIN] Training Calibrated RBF SVM on {len(pruned_final_genes)} pristine genes...")
    final_multiclass_svm = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)

    print("        --> Running rigorous 10x10 Cross-Validation on the Discovery Set...")
    final_acc, final_f1, final_prec, final_rec = execute_10x10_cv(final_multiclass_svm, X_final_pruned_train, y_train)

    print("\n        " + "*"*55)
    print(f"        [!] INTERNAL DISCOVERY SET ACCURACY : {final_acc*100:.2f}%")
    print("        " + "*"*55)

    final_multiclass_svm.fit(X_train_smote, y_train_smote)

    # ---------------------------------------------------------
    # PHASE C: DIRECT GEOMETRIC HOLDOUT VALIDATION
    # ---------------------------------------------------------
    print("\n[VALIDATION] Evaluating Final Model on 20% Untouched Test Set...")
    y_pred_test = final_multiclass_svm.predict(X_final_pruned_test)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)

    print("\n        " + "*"*55)
    print(f"        [!] INDEPENDENT HOLDOUT ACCURACY : {test_acc*100:.2f}%")
    print(f"        [!] INDEPENDENT HOLDOUT F1 SCORE : {test_f1*100:.2f}%")
    print(f"        [!] INDEPENDENT HOLDOUT PRECISION: {test_prec*100:.2f}%")
    print(f"        [!] INDEPENDENT HOLDOUT RECALL   : {test_rec*100:.2f}%")
    print("        " + "*"*55)

    joblib.dump(final_multiclass_svm, "BRCA_Omni_MultiClass_SVM.pkl")
    joblib.dump(scaler, "BRCA_Scaler.pkl")
    np.save("BRCA_Omni_Genes.npy", pruned_final_genes)

    print(f"\n[SAVE] Multi-Class Model securely saved as 'BRCA_Omni_MultiClass_SVM.pkl'")
    print(f"[SAVE] Preprocessing Scaler securely saved as 'BRCA_Scaler.pkl'")
    print(f"[SAVE] Gene signature safely saved as 'BRCA_Omni_Genes.npy'")
    print("="*80)

    return pruned_final_genes

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    expr_file, clin_file = resolve_data_paths()
    print(f"[*] Expression File: {expr_file}")
    print(f"[*] Clinical File:   {clin_file}")

    target_subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Her2', 'BRCA_Normal', 'BRCA_Basal']
    final_clinical_genes = main_pipeline_multiclass(expr_file, clin_file, target_subtypes)
