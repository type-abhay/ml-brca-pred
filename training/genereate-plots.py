from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, train_test_split

# ==========================================
# 0. DYNAMIC DIRECTORY ANCHORS
# ==========================================
# Resolves the exact project root regardless of terminal working directory
SCRIPT_DIR = Path(__file__).resolve().parent          # .../brca-pred/training
PROJECT_ROOT = SCRIPT_DIR.parent                     # .../brca-pred

# Set and create output directory for figures
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

print("[*] Loading Models and Genomic Data...")

# Model and gene signature paths inside root/backend/
MODEL_PATH = PROJECT_ROOT / "backend" / "BRCA_Omni_MultiClass_SVM.pkl"
GENES_PATH = PROJECT_ROOT / "backend" / "BRCA_Omni_Genes.npy"

# Fallback: check script folder or root if moved
if not MODEL_PATH.exists():
    for alt in [SCRIPT_DIR, PROJECT_ROOT, SCRIPT_DIR / "backend"]:
        if (alt / "BRCA_Omni_MultiClass_SVM.pkl").exists():
            MODEL_PATH = alt / "BRCA_Omni_MultiClass_SVM.pkl"
            GENES_PATH = alt / "BRCA_Omni_Genes.npy"
            break

try:
    svm_model = joblib.load(MODEL_PATH)
    target_genes = np.load(GENES_PATH, allow_pickle=True)
    print(f"[+] Successfully loaded {len(target_genes)} genes and model from: {MODEL_PATH}")
except Exception as e:
    print(f"[!] Error loading .pkl or .npy artifacts from {MODEL_PATH}: {e}")
    exit(1)

# ==========================================
# 1. DATASET RESOLUTION & INGESTION
# ==========================================
def resolve_data_paths():
    candidates = [
        (PROJECT_ROOT / "tcga-data" / "TCGA_BRCA_tpm.tsv",
         PROJECT_ROOT / "tcga-data" / "brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"),
        (SCRIPT_DIR / "tcga-data" / "TCGA_BRCA_tpm.tsv",
         SCRIPT_DIR / "tcga-data" / "brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"),
        (Path("/kaggle/input/datasets/blaiseappolinary/tcga-data/TCGA_BRCA_tpm.tsv"),
         Path("/kaggle/input/datasets/blaiseappolinary/tcga-data/brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"))
    ]
    for expr, clin in candidates:
        if expr.exists() and clin.exists():
            return expr, clin

    # Recursive search fallback across project folder
    for root, _, files in os.walk(PROJECT_ROOT):
        if "TCGA_BRCA_tpm.tsv" in files:
            expr = Path(root) / "TCGA_BRCA_tpm.tsv"
            clin = Path(root) / "brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"
            if clin.exists():
                return expr, clin
    raise FileNotFoundError(f"Could not locate TCGA dataset files under {PROJECT_ROOT}")

expr_file, clin_file = resolve_data_paths()
print(f"[+] Ingesting Expression Data: {expr_file}")
print(f"[+] Ingesting Clinical Data:   {clin_file}")

df_expr = pd.read_csv(expr_file, sep='\t', low_memory=False)
df_expr.set_index(df_expr.columns[0], inplace=True)
df_expr = df_expr.T
df_expr.index.name = 'Sample_ID'
df_expr.reset_index(inplace=True)

df_clin = pd.read_csv(clin_file, sep='\t')
df_clin = df_clin.rename(columns={'Patient ID': 'Sample_ID', 'Subtype': 'Subtype'})
df_clin = df_clin[['Sample_ID', 'Subtype']]

df_expr['Sample_ID'] = df_expr['Sample_ID'].astype(str).str[:12]
df_clin['Sample_ID'] = df_clin['Sample_ID'].astype(str).str[:12]

df_expr = df_expr.drop_duplicates(subset=['Sample_ID'])
df_clin = df_clin.drop_duplicates(subset=['Sample_ID'])

merged_df = pd.merge(df_expr, df_clin, on='Sample_ID', how='inner')
merged_df = merged_df.dropna(subset=['Subtype'])

# Filter strictly to the 100 final panel genes
X_df = merged_df[target_genes].apply(pd.to_numeric, errors='coerce').fillna(0)
y_all = merged_df['Subtype'].values

# ==========================================
# 2. REPLICATE 80/20 QUARANTINED SPLIT
# ==========================================
print("[+] Replicating Stratified 80/20 Partition (random_state=42)...")
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_df.values, y_all, test_size=0.20, stratify=y_all, random_state=42
)
print(f"    --> Discovery Set (Train): {len(y_train)} samples")
print(f"    --> Holdout Set (Test):    {len(y_test)} samples")

# Standard scaling fitted strictly on the train set (precluding data leakage)
X_train_log = np.log1p(X_tr_raw)
X_test_log = np.log1p(X_te_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled = scaler.transform(X_test_log)

print("\n[+] Generating Predictions on 20% Independent Holdout Set (N = 196)...")
y_pred_test = svm_model.predict(X_test_scaled)
classes = svm_model.classes_

print("\n--- Independent Holdout Classification Report ---")
print(classification_report(y_test, y_pred_test, digits=4))

# ==========================================
# PLOT 1: CONFUSION MATRIX (FIGURE 2)
# Strictly on the 20% Independent Holdout Set
# ==========================================
print("[>] Generating Figure 2: Multi-Class Confusion Matrix (Holdout)...")
cm = confusion_matrix(y_test, y_pred_test, labels=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=classes, yticklabels=classes,
    annot_kws={"size": 14}
)
plt.title(f'Multi-Class Confusion Matrix (100-Gene Panel - Independent Holdout N={len(y_test)})', fontsize=15, pad=18)
plt.ylabel('True Clinical Subtype', fontsize=13)
plt.xlabel('SVM Predicted Subtype', fontsize=13)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
fig1_path = PLOTS_DIR / "Fig1_Confusion_Matrix.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"    --> Saved to '{fig1_path}'")

# ==========================================
# PLOT 2: PCA SCATTER PLOT (FIGURE 3)
# ==========================================
print("[>] Generating Figure 3: PCA 2D Projection...")
pca = PCA(n_components=2, random_state=42)
X_pca_test = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(12, 9))
sns.scatterplot(
    x=X_pca_test[:, 0], y=X_pca_test[:, 1],
    hue=y_test, style=y_pred_test,
    palette='Set1', s=100, alpha=0.85,
    markers=['o', 's', 'D', '^', 'v']
)

plt.title(f'PCA 2D Projection: Transcriptomic Clusters ({len(target_genes)} Genes)', fontsize=16, pad=20)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=13)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=13)
plt.legend(title='True Subtype vs Predicted Marker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig2_path = PLOTS_DIR / "Fig2_PCA_Clusters.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"    --> Saved to '{fig2_path}'")

# ==========================================
# PLOT 3: SVM LEARNING CURVE (FIGURE 4)
# Computed on the Discovery Cohort
# ==========================================
print("[>] Generating Figure 4: SVM Learning Curve on Discovery Cohort...")
train_sizes, train_scores, test_scores = learning_curve(
    svm_model, X_train_scaled, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)

plt.plot(train_sizes, test_mean, label='Cross-Validation Accuracy', color='green', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.15)

plt.title('SVM Learning Curve: Accuracy vs. Clinical Sample Size', fontsize=15, pad=18)
plt.xlabel('Number of Training Patients', fontsize=13)
plt.ylabel('Accuracy Score', fontsize=13)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
fig3_path = PLOTS_DIR / "Fig3_Learning_Curve.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"    --> Saved to '{fig3_path}'")

print(f"\n[✓] All 3 publication-ready figures have been generated and saved inside: {PLOTS_DIR}")
