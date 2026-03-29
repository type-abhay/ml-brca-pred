import numpy as np
import joblib

# 1. Load your model to check the exact required gene count
model = joblib.load("D:\\Desktop Files\\Coding\\miniproject\\brca-pred\\backend\\BRCA_Omni_MultiClass_SVM.pkl")
required_genes = model.n_features_in_

# 2. Generate a perfectly scaled mock patient (Standard Normal Distribution)
mock_patient = np.random.randn(1, required_genes)

# 3. Export directly to a clean CSV file (no headers, no index)
np.savetxt("mock_patient_upload.csv", mock_patient, delimiter=",", fmt="%.5f")

print(f"✅ Successfully created 'mock_patient_upload.csv' containing {required_genes} scaled gene expression values!")