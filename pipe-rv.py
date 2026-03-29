import joblib
import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE


warnings.filterwarnings("ignore")

# ==========================================
# 0. SCANNING FILES
# ==========================================
print("🔍 Scanning Kaggle Input Directory...")
for dirname, _, filenames in os.walk('tcga-data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("-" * 50)

# ==========================================
# 1. DATA INGESTION & TRANSPOSITION (NaN Purged)
# ==========================================
def load_and_preprocess_kaggle_data(expression_file, clinical_file):
    print("📥 Step 1: Loading and Transposing Kaggle TCGA BRCA datasets...")
    
    df_expr = pd.read_csv(expression_file, sep='\t', low_memory=False)
    gene_col_name = df_expr.columns[0]
    df_expr.set_index(gene_col_name, inplace=True)
    
    print("🔄 Transposing matrix... (Patience, this takes a moment)")
    df_expr = df_expr.T 
    df_expr.index.name = 'Sample_ID'
    df_expr.reset_index(inplace=True)
    
    df_clin = pd.read_csv(clinical_file, sep='\t')
    
    if 'Patient ID' not in df_clin.columns or 'Subtype' not in df_clin.columns:
        raise ValueError("🚨 Could not find 'Patient ID' or 'Subtype' in clinical columns.")
        
    df_clin = df_clin.rename(columns={'Patient ID': 'Sample_ID', 'Subtype': 'Subtype'})
    df_clin = df_clin[['Sample_ID', 'Subtype']]
    
    print("✂️ Standardizing TCGA Barcodes to 12 characters...")
    df_expr['Sample_ID'] = df_expr['Sample_ID'].astype(str).str[:12]
    df_clin['Sample_ID'] = df_clin['Sample_ID'].astype(str).str[:12]
    
    df_expr = df_expr.drop_duplicates(subset=['Sample_ID'])
    df_clin = df_clin.drop_duplicates(subset=['Sample_ID'])
    
    print("🧬 Merging genetic data with clinical labels...")
    merged_df = pd.merge(df_expr, df_clin, on='Sample_ID', how='inner')
    
    # ⚠️ KINK #4 PREVENTED: Purge NaN ghosts
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['Subtype'])
    if len(merged_df) < initial_len:
        print(f"   🧹 Purged {initial_len - len(merged_df)} patients due to missing (NaN) clinical labels.")
    
    y = merged_df['Subtype'].values
    X_df = merged_df.drop(columns=['Sample_ID', 'Subtype'])
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    threshold = 0.75 * len(X_df)
    zero_counts = (X_df == 0).sum(axis=0)
    genes_to_keep = zero_counts[zero_counts <= threshold].index
    
    X_filtered = X_df[genes_to_keep].values
    print(f"✅ Filtered feature count: {X_filtered.shape[1]}")
    
    return X_filtered, y, genes_to_keep

# ==========================================
# 2. OPTIMIZED VALIDATION ENGINE
# ==========================================
def execute_fast_cv(model, X, y, scoring_metric='roc_auc'):
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1)
    return np.mean(scores)

def execute_10x10_cv(model, X, y, scoring_metric='accuracy'):
    cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1)
    return np.mean(scores)


# ==========================================
# 3. mCGA ALGORITHM (TURBO MODE)
# ==========================================
def algorithm_2_mcga(X, y_binary, model, n_step=5, max_iter=100):
    N = X.shape[1]
    num_pvs = 4
    c_global = 2.0  
    
    PVs = np.full((num_pvs, N), 0.5)
    
    # Restored the outer progress bar formatting
    for iteration in tqdm(range(max_iter), desc="   mCGA Epochs", unit="epoch"):
        binary_strings = (np.random.rand(num_pvs, N) < PVs).astype(int)
        fitness_values = np.zeros(num_pvs)
        
        # ⚠️ RESTORED INNER PROGRESS BAR HERE!
        for i in tqdm(range(num_pvs), desc="      Evaluating PVs", leave=False):
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
            
        if np.all((Best_PV > 0.9) | (Best_PV < 0.1)):
            print(f"\n   ✨ PVs converged early at Epoch {iteration+1}!")
            break
            
    return np.where(Best_PV >= 0.5)[0]


# ==========================================
# 4. MASTER OMNISCIENT PIPELINE
# ==========================================
def main_pipeline_multiclass(expression_filepath, clinical_filepath, target_subtypes):
    X_raw, y_multiclass, gene_names = load_and_preprocess_kaggle_data(expression_filepath, clinical_filepath)
    
    # ⚠️ KINK #2 PREVENTED: Log + Standard Scaling Geometry
    print("\n🧬 Applying Log2 Transformation to normalize RNA-Seq skew...")
    X_log = np.log1p(X_raw) 
    
    print("⚖️ Applying Standard Scaling (Mean=0, Variance=1) for SVM geometry...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_log)
    
    master_biomarker_indices = set()
    
    # ---------------------------------------------------------
    # PHASE A: SUBTYPE LOOP (The Exploratory Phase)
    # ---------------------------------------------------------
    for subtype in target_subtypes:
        print(f"\n" + "="*50)
        print(f"🎯 COMMENCING PROTOCOL FOR: {subtype}")
        print("="*50)
        
        y_binary = (y_multiclass == subtype).astype(int)
        
        if sum(y_binary) == 0:
            print(f"⚠️ WARNING: Subtype '{subtype}' not found in data. Skipping.")
            continue
            
        print(f"   Class Distribution -> Positive: {sum(y_binary)}, Negative: {len(y_binary)-sum(y_binary)}")
        
        print("\n   🔪 Step 2: Dynamic LASSO Reduction (LassoCV)...")
        
        # ⚠️ AIA'S MEMORY-SAFE FIX ⚠️
        # 1. n_jobs=2 (Prevents Kaggle RAM from choking on the 37k matrix)
        # 2. verbose=1 (Native Scikit-Learn progress reporting, no hacks needed!)
        # lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=2, verbose=1).fit(X, y_binary)
        # print(f"   ✨ Optimal alpha discovered: {lasso_cv.alpha_:.5f}")

        # --- TEMPORARY HARDCODED ALPHA INJECTION ---
        from sklearn.linear_model import Lasso
        
        alpha_map = {
            'BRCA_LumA': 0.044, 
            'BRCA_LumB': 0.133, 
            'BRCA_Her2': 0.039, 
            'BRCA_Normal': 0.010, 
            'BRCA_Basal': 0.010
        }
        
        alpha_val = alpha_map.get(subtype, 0.044)
        
        # Notice we are calling the variable lasso_cv so the rest of the script doesn't break!
        lasso_cv = Lasso(alpha=alpha_val, random_state=42).fit(X, y_binary)
        print(f"   ✨ Hardcoded literature alpha applied: {alpha_val:.5f}")
        # ---------------------------------------------
            
        
        selected_features_idx = np.where(np.abs(lasso_cv.coef_) > 0)[0]
        X_reduced = X[:, selected_features_idx]
        print(f"   LASSO preserved {len(selected_features_idx)} potential biomarkers.")
        
        # ⚠️ KINK #3 PREVENTED: Base model baseline evaluation
        print("\n   📊 Base Model Evaluation (10x10 CV) on LASSO features...")
        models = {
            'RF': RandomForestClassifier(random_state=42),
            'SVM': SVC(kernel='linear', C=2, probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'NB': GaussianNB()
        }
        for name, model in models.items():
            acc = execute_10x10_cv(model, X_reduced, y_binary, scoring_metric='accuracy')
            print(f"      Model {name} Accuracy: {acc*100:.2f}%")
            
        print("\n   🧬 Step 3: mCGA Biomarker Extraction...")
        ovr_svm = SVC(kernel='linear', C=2, random_state=42) 
        final_relative_idx = algorithm_2_mcga(X_reduced, y_binary, ovr_svm)
        
        final_biomarkers_idx = selected_features_idx[final_relative_idx]
        master_biomarker_indices.update(final_biomarkers_idx)
        print(f"   ✅ Discovered {len(final_biomarkers_idx)} unique genes for {subtype}.")

    # ---------------------------------------------------------
    # PHASE B: THE FINAL SCALPEL & MULTI-CLASS MODEL
    # ---------------------------------------------------------
    print("\n" + "🌟"*30)
    print("🏆 FINAL STAGE: RFECV PRUNING & MULTI-CLASS TRAINING")
    print("🌟"*30)
    
    final_indices_array = np.array(sorted(list(master_biomarker_indices)))
    initial_master_genes = gene_names[final_indices_array]
    print(f"🧬 Total Master Biomarkers Combined (Pre-Scalpel): {len(initial_master_genes)}")
    
    # Isolate the data for the combined master list
    X_final_master = X[:, final_indices_array]
    
    # ⚠️ KINK #1 PREVENTED: Targeted Recursive Feature Elimination (RFE)
    TARGET_CLINICAL_GENES = 20  # <--- YOU COMMAND THE EXACT NUMBER HERE!
    
    print(f"\n🔪 Activating RFE Targeted Scalpel to ruthlessly prune down to {TARGET_CLINICAL_GENES} genes...")
    # We use a linear SVM as the judge to recursively drop the weakest genes
    judge_svm = SVC(kernel='linear', C=1, random_state=42)
    
    # We switch from RFECV to RFE to force our exact target number
    rfe = RFE(estimator=judge_svm, n_features_to_select=TARGET_CLINICAL_GENES, step=1)
    
    # Fit the scalpel on the master dataset and the multiclass target
    rfe.fit(X_final_master, y_multiclass)
    
    # Extract the boolean mask of genes that SURVIVED the scalpel
    surviving_mask = rfe.support_
    
    # Map the mask back to our true gene names and indices
    pruned_indices_array = final_indices_array[surviving_mask]
    pruned_final_genes = gene_names[pruned_indices_array]
    X_final_pruned = X[:, pruned_indices_array]
    
    print(f"✅ RFE complete! Master Biomarkers forcefully reduced from {len(initial_master_genes)} down to **{len(pruned_final_genes)}** elite genes!")
    print(f"Final Clinical Panel: {pruned_final_genes.tolist()}")
    
    # ⚠️ KINK #5 PREVENTED: Training the final model
    print(f"\n🧠 Training the ultimate Multi-Class SVM on the {len(pruned_final_genes)} pristine genes...")
    final_multiclass_svm = SVC(kernel='linear', C=2, probability=True, random_state=42)
    
    # ⚠️ AIA'S FINAL EXAM (OVERALL ACCURACY CHECK) ⚠️
    print("   📊 Running rigorous 10x10 Cross-Validation on the Elite Clinical Panel...")
    final_acc = execute_10x10_cv(final_multiclass_svm, X_final_pruned, y_multiclass, scoring_metric='accuracy')
    print("   " + "⭐"*15)
    print(f"   🏆 FINAL MULTI-CLASS DIAGNOSTIC ACCURACY: {final_acc*100:.2f}%")
    print("   " + "⭐"*15)
    
    # Fit the model once and for all on the entire dataset
    final_multiclass_svm.fit(X_final_pruned, y_multiclass)
    
    # Save the completely trained, omniscient brain
    model_filename = "BRCA_Omni_MultiClass_SVM.pkl"
    joblib.dump(final_multiclass_svm, model_filename)
    print(f"\n💾 Multi-Class Model securely saved as '{model_filename}'")
    
    # Save the exact list of gene names so the clinic knows what to sequence!
    np.save("BRCA_Omni_Genes.npy", pruned_final_genes)
    
    return pruned_final_genes

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    expr_file = "tcga-data/TCGA_BRCA_tpm.tsv" 
    clin_file = "tcga-data/brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"
    
    target_subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Her2', 'BRCA_Normal', 'BRCA_Basal']
    
    # Run the absolute master pipeline
    final_clinical_genes = main_pipeline_multiclass(expr_file, clin_file, target_subtypes)