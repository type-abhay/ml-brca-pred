import joblib
import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("🔍 Scanning Kaggle Input Directory...")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("-" * 50)

# ==========================================
# 1. DATA INGESTION & TRANSPOSITION
# ==========================================
def load_and_preprocess_kaggle_data(expression_file, clinical_file):
    print("📥 Step 1: Loading and Transposing Kaggle TCGA BRCA datasets...")
    
    # 1. Load expression data
    df_expr = pd.read_csv(expression_file, sep='\t', low_memory=False)
    gene_col_name = df_expr.columns[0]
    df_expr.set_index(gene_col_name, inplace=True)
    
    print("🔄 Transposing matrix... (Patience, this takes a moment)")
    df_expr = df_expr.T 
    df_expr.index.name = 'Sample_ID'
    df_expr.reset_index(inplace=True)
    
    # 2. Load clinical data
    df_clin = pd.read_csv(clinical_file, sep='\t')
    
    # Explicitly rename to match
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
    
    y = merged_df['Subtype'].values
    X_df = merged_df.drop(columns=['Sample_ID', 'Subtype'])
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"Total features before filtering: {X_df.shape[1]}")
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
    """Fast 5-fold CV for mCGA fitness evaluation."""
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring_metric, n_jobs=-1)
    return np.mean(scores)

def execute_10x10_cv(model, X, y, scoring_metric='accuracy'):
    """Rigorous 100-fold CV for baseline evaluation."""
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
    
    print("\n   🚀 Initiating Turbo mCGA Convergence Protocol...")
    for iteration in tqdm(range(max_iter), desc="mCGA Epochs", unit="epoch"):
        binary_strings = (np.random.rand(num_pvs, N) < PVs).astype(int)
        fitness_values = np.zeros(num_pvs)
        
        for i in tqdm(range(num_pvs), desc="   Evaluating PVs", leave=False):
            selected_cols = np.where(binary_strings[i] == 1)[0]
            if len(selected_cols) > 0:
                X_subset = X[:, selected_cols]
                # Using the fast CV to prevent bottlenecks!
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
            print(f"\n   ✨ Probability Vectors converged at Epoch {iteration+1}!")
            break
            
    return np.where(Best_PV >= 0.5)[0]

# ==========================================
# 4. MASTER PIPELINE
# ==========================================
def main_pipeline(expression_filepath, clinical_filepath, target_subtype='BRCA_LumA'):
    X_raw, y_multiclass, gene_names = load_and_preprocess_kaggle_data(expression_filepath, clinical_filepath)
    
    # ⚠️ AIA'S CLINICAL SCALING INJECTION ⚠️
    print("\n🧬 Applying Log2 Transformation to normalize RNA-Seq skew...")
    # We use log1p which calculates log(1 + x) to safely handle zeros
    X_log = np.log1p(X_raw) 
    
    print("⚖️ Applying Standard Scaling (Mean=0, Variance=1) for SVM/KNN geometry...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_log)
    # ----------------------------------------

    y_binary = (y_multiclass == target_subtype).astype(int)
    print(f"\n🎯 Target Subtype: {target_subtype} (Positive: {sum(y_binary)}, Negative: {len(y_binary)-sum(y_binary)})")
    
    alpha_map = {'BRCA_LumA': 0.044, 'BRCA_LumB': 0.133, 'BRCA_HER2': 0.039, 'BRCA_Basal': 0.010}
    # Fallback to 0.044 if target_subtype spelling is slightly off from our map
    alpha_val = alpha_map.get(target_subtype, 0.044) 
    
    print(f"\n🔪 Step 2: LASSO Reduction (alpha={alpha_val})")
    lasso = Lasso(alpha=alpha_val, random_state=42).fit(X, y_binary)
    selected_features_idx = np.where(np.abs(lasso.coef_) > 0)[0]
    X_reduced = X[:, selected_features_idx]
    
    print(f"LASSO preserved {len(selected_features_idx)} potential biomarkers.")
    
    print("\n📊 Step 3: Base Model Evaluation (10x10 CV)")
    models = {
        'RF': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='linear', C=2, probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'NB': GaussianNB()
    }
    
    for name, model in models.items():
        acc = execute_10x10_cv(model, X_reduced, y_binary, scoring_metric='accuracy')
        print(f"   Model {name} Accuracy: {acc*100:.2f}%")
        
    print("\n🧬 Step 4: mCGA Biomarker Extraction")
    best_model = models['SVM'] 
    final_relative_idx = algorithm_2_mcga(X_reduced, y_binary, best_model)
    final_biomarkers_idx = selected_features_idx[final_relative_idx]
    
    final_genes = gene_names[final_biomarkers_idx]
    print(f"\n✅ Pipeline Complete! Discovered {len(final_genes)} final biomarkers for {target_subtype}.")
    print("Final Genes:", final_genes.tolist())

    # ⚠️ AIA'S FINAL CLINICAL TRAINING FIX ⚠️
    print("\n🧠 Training the final Clinical Model on the discovered biomarkers...")
    # Extract ONLY the data for the winning genes
    X_final_biomarkers = X_reduced[:, final_relative_idx]
    
    # Train the SVM once and for all on this pristine dataset
    best_model.fit(X_final_biomarkers, y_binary)
    
    # NOW we save the fully trained brain!
    model_filename = f"{target_subtype}_svm_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"💾 Trained model securely saved as {model_filename}")
    
    return final_genes

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Ensure these paths match what the Kaggle scanner prints at the top!
    expr_file = "/kaggle/input/datasets/blaiseappolinary/tcga-data/TCGA_BRCA_tpm.tsv" 
    clin_file = "/kaggle/input/datasets/blaiseappolinary/tcga-data/brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"
    
    final_luma_genes = main_pipeline(expr_file, clin_file, target_subtype='BRCA_LumA')