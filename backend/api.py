from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import io

app = FastAPI()

# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Omniscient Brain
MODEL_FILE = "BRCA_Omni_MultiClass_SVM.pkl"
GENE_FILE = "BRCA_Omni_Genes.npy"

print("⏳ Booting Clinical Inference API...")
try:
    model = joblib.load(MODEL_FILE)
    required_genes = model.n_features_in_
    gene_names = np.load(GENE_FILE, allow_pickle=True)
    classes = model.classes_
    print(f"✅ Brain Loaded. Awaiting {required_genes}-gene vectors.")
except Exception as e:
    print(f"🚨 CRITICAL ERROR: {e}")

def get_predictions(patient_matrix):
    prediction = model.predict(patient_matrix)[0]
    probabilities = model.predict_proba(patient_matrix)[0]
    
    # Package results into a clean dictionary for the frontend
    breakdown = [{"subtype": cls, "confidence": round(prob * 100, 2)} for cls, prob in zip(classes, probabilities)]
    breakdown.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {"diagnosis": prediction, "breakdown": breakdown}


@app.get("/info")
async def get_info():
    """Provides the frontend with the required gene panel."""
    return {
        "required_genes": required_genes, 
        "gene_names": gene_names.tolist()
    }

@app.get("/predict/demo")
async def predict_demo():
    """Generates a perfectly scaled mock patient and returns the sequence."""
    mock_matrix = np.random.randn(1, required_genes)
    results = get_predictions(mock_matrix)
    # Append the raw generated sequence to the results dictionary!
    results["input_data"] = mock_matrix[0].tolist()
    return results

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Parses an uploaded CSV file containing RNA-Seq data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files are permitted.")
    
    contents = await file.read()
    try:
        # Assumes a simple CSV with 1 row of values matching the gene count
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None)
        patient_matrix = df.iloc[0].values.astype(float).reshape(1, -1)
        
        if patient_matrix.shape[1] != required_genes:
            raise ValueError(f"Expected {required_genes} columns, got {patient_matrix.shape[1]}")
            
        return get_predictions(patient_matrix)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))