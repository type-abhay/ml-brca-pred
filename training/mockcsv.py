import numpy as np
import joblib

model = joblib.load("D:\\Desktop Files\\Coding\\miniproject\\brca-pred\\backend\\BRCA_Omni_MultiClass_SVM.pkl")
required_genes = model.n_features_in_

mock_patient = np.random.randn(1, required_genes)

np.savetxt("mock_patient_upload.csv", mock_patient, delimiter=",", fmt="%.5f")
print(f"Successfully created 'mock_patient_upload.csv' containing {required_genes} scaled gene expression values!")