import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any
from datetime import datetime
import uuid
from pydantic import BaseModel

# Classification model and scaler paths
STACKING_MODEL_PATH = "models/stacking_model.sav"
SCALER_PATH = "models/scaler.sav"

# Clinical knowledge base for classification
CLINICAL_INFO = {
    0: {
        "name": "Healthy Liver",
        "description": "Normal liver function with no signs of damage or inflammation.",
        "markers": "ALT <40 U/L, AST <40 U/L, ALB 3.5-5.0 g/dL",
        "action": "Routine monitoring recommended."
    },
    1: {
        "name": "General Hepatitis",
        "description": "Liver inflammation from viral, autoimmune or metabolic causes.",
        "markers": "ALT 2-10x ULN, elevated bilirubin",
        "action": "Requires viral serology tests and lifestyle modification."
    },
    2: {
        "name": "Acute Hepatitis C",
        "description": "Early-stage HCV infection (<6 months) with 75% chronicity risk.",
        "markers": "Anti-HCV+, RNA viral load+, ALT 5-20x ULN",
        "action": "Urgent referral for DAAs (Direct-Acting Antivirals)."
    },
    3: {
        "name": "Fibrosis",
        "description": "Scar tissue formation (METAVIR F1-F4), potentially reversible.",
        "markers": "FibroScan 7-14 kPa, APRI >1.0, ELF >9.8",
        "action": "Treat underlying cause, monitor progression every 6 months."
    },
    4: {
        "name": "Cirrhosis",
        "description": "Irreversible scarring with decompensation risk.",
        "markers": "Platelets <150K, INR >1.1, MELD ≥15",
        "action": "HCC surveillance q6mo, transplant evaluation if MELD≥15."
    }
}

# Define request model for classification
class ClassificationRequest(BaseModel):
    age: float
    alb: float
    che: float
    chol: float
    prot: float
    alp: float
    alt: float
    ast: float
    bil: float
    ggt: float

def load_classification_model():
    """Load the classification model and scaler"""
    with open(STACKING_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler
    

def preprocess_data(raw_data):
    """Apply all required transformations"""
    df = pd.DataFrame({k: [v] for k, v in raw_data.items()})
    
    # Rename keys to match expected format
    rename_map = {
        'age': 'Age',
        'alb': 'ALB',
        'che': 'CHE',
        'chol': 'CHOL',
        'prot': 'PROT',
        'alp': 'ALP',
        'alt': 'ALT',
        'ast': 'AST',
        'bil': 'BIL',
        'ggt': 'GGT'
    }
    df = df.rename(columns=rename_map)
    
    # Log transforms (same as training)
    df['Logged ALT'] = np.log(df['ALT'])
    df['Logged BIL'] = np.log(df['BIL'])
    try:
        df['TLogged AST'] = np.log(np.log(np.log(df['AST'])))
    except:
        # Handle potential numerical issues
        df['TLogged AST'] = np.log(np.log(df['AST'] + 1) + 1)
    try:
        df['DLogged GGT'] = np.log(np.log(df['GGT']))
    except:
        # Handle potential numerical issues
        df['DLogged GGT'] = np.log(np.log(df['GGT'] + 1) + 1)
    
    # Extract only the features needed by the scaler
    _, scaler = load_classification_model()
    return df[scaler.feature_names_in_]

def generate_classification_report(patient_id, features, prediction, probabilities):
    """Generate a structured classification report"""
    cls_info = CLINICAL_INFO[prediction]
    prob_percent = probabilities[prediction] * 100
    
    # Build list of recommended actions
    actions = [cls_info['action']]
    if prediction >= 2:  # For Hepatitis+
        actions.append("❗ Urgent hepatology referral")
    if prediction >= 3:  # For Fibrosis+
        actions.append("📅 Schedule FibroScan/ELF test")
    if prediction == 4:  # Cirrhosis
        actions.append("🔄 MELD Score: Calculate for transplant priority")
    
    # Format probabilities for display
    prob_display = {CLINICAL_INFO[i]['name']: prob * 100 for i, prob in enumerate(probabilities)}
    
    return {
        "patient_id": patient_id,
        "prediction": int(prediction),
        "diagnosis_name": cls_info['name'],
        "confidence": prob_percent,
        "description": cls_info['description'],
        "markers": cls_info['markers'],
        "actions": actions,
        "probabilities": prob_display
    }

def process_classification(data):
    """Process classification request and return results"""
    # Load classification model and scaler
    clf_model, scaler = load_classification_model()
    
    # Convert request data to dictionary
    raw_data = data.dict()
    
    # Preprocess data
    X = preprocess_data(raw_data)
    X_scaled = scaler.transform(X)
    
    # Make prediction
    pred = clf_model.predict(X_scaled)[0]
    proba = clf_model.predict_proba(X_scaled)[0]
    
    # Generate patient ID
    patient_id = f"PAT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4]}"
    
    # Generate report
    report = generate_classification_report(patient_id, X.iloc[0], pred, proba)
    
    return report