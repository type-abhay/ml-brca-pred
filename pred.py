import sys
import joblib
import numpy as np
import warnings

# Suppress sklearn warnings about feature names for a clean terminal
warnings.filterwarnings("ignore")

# ==========================================
# 1. CORE INFERENCE ENGINE
# ==========================================
def predict_multiclass_patient(model_path, patient_expression_data):
    """
    Aia's Omniscient Multi-Class Diagnostic Engine
    """
    print(f"\n🩺 Booting Diagnostic Brain: {model_path}...")
    
    # 1. Load Model & Dimension Check
    model = joblib.load(model_path)
    expected_features = model.n_features_in_
    
    if len(patient_expression_data) != expected_features:
        raise ValueError(f"🚨 Dimension Mismatch! Our SVM expects {expected_features} master biomarkers, but received {len(patient_expression_data)}.")
    
    # 2. Reshape array for Scikit-Learn geometry
    patient_matrix = np.array(patient_expression_data).reshape(1, -1)
    
    # 3. Execute the Multi-Class SVM Hyperplane mathematics
    prediction = model.predict(patient_matrix)[0]
    probabilities = model.predict_proba(patient_matrix)[0]
    classes = model.classes_
    
    # 4. Elegant Output Formatting
    print("\n" + "=" * 55)
    print(f"⚠️ CLINICAL DIAGNOSIS: {prediction.upper()}")
    print("=" * 55)
    print("Probability Breakdown Across Subtypes:")
    
    # Sort the classes by their confidence probabilities
    class_probs = list(zip(classes, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    
    for cls, prob in class_probs:
        marker = "👉 " if cls == prediction else "   "
        print(f"{marker}{cls}: {prob * 100:>6.2f}%")
    print("-" * 55)
        
    return prediction

# ==========================================
# 2. INTERACTIVE CLINICAL TERMINAL
# ==========================================
if __name__ == "__main__":
    MODEL_FILE = "BRCA_Omni_MultiClass_SVM.pkl"
    GENE_FILE = "BRCA_Omni_Genes.npy"
    
    print("⏳ Powering up the Diagnostic Engine...")
    
    # ⚠️ AIA'S DYNAMIC FIX: Load the brain and the gene panel!
    try:
        omni_brain = joblib.load(MODEL_FILE)
        REQUIRED_GENES = omni_brain.n_features_in_
        required_gene_names = np.load(GENE_FILE, allow_pickle=True)
    except FileNotFoundError:
        print(f"🚨 CRITICAL ERROR: Could not find the model or gene files.")
        print("Please ensure your Kaggle pipeline has finished running and the .pkl / .npy files are in this folder.")
        sys.exit()
        
    print("=" * 55)
    print("🔬 Welcome to the BRCA Subtype Diagnostic Terminal")
    print(f"🧠 Model Loaded. The Omniscient Brain requires a {REQUIRED_GENES}-gene signature.")
    print("=" * 55)
    print("1. Run Demonstration (Simulate Random Patient Data)")
    print("2. Input Real Patient Data (Manual Clinical Mode)")
    
    choice = input("\nSelect your protocol (1 or 2): ").strip()
    
    if choice == '1':
        print("\n🧪 Generating mock RNA-Seq profile...")
        # Dynamically generating the exact right amount of mock data
        mock_patient_rna = np.random.randn(REQUIRED_GENES)
        predict_multiclass_patient(MODEL_FILE, mock_patient_rna)
        
    elif choice == '2':
        print(f"\n🩺 CLINICAL MODE ACTIVE.")
        print(f"Please input the RNA-Seq values for the following {REQUIRED_GENES} genes in this EXACT order:")
        print(f"🧬 {required_gene_names.tolist()}")
        print("\n(⚠️ Note: Input MUST be Log2-transformed and Standard Scaled!)")
        
        user_input = input("\nPatient RNA Array (comma-separated): ").strip()
        
        try:
            raw_data = [float(x.strip()) for x in user_input.split(',')]
            patient_data = np.array(raw_data)
            predict_multiclass_patient(MODEL_FILE, patient_data)
            
        except ValueError:
            print("🚨 Input Error: Please ensure you only entered numeric values separated by commas.")
    else:
        print("🚨 Invalid selection. Closing the laboratory.")