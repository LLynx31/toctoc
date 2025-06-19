from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from model.model_utils import (
    load_model, 
    load_scaler, 
    load_columns, 
    load_target_encoder, 
    load_onehot_encoder
)
from data.preprocess import preprocess_uploaded_df
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Détection du type de fichier
    filename = file.filename.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture du fichier : {e}")

    # Chargement du modèle et preprocessing
    model = load_model()
    scaler = load_scaler()
    target_encoder = load_target_encoder()
    onehot_encoder = load_onehot_encoder()
    columns = load_columns()


    # Debug des colonnes
    logger.info(f"Colonnes du DataFrame: {df.columns.tolist()}")
    logger.info(f"Colonnes du scaler: {scaler.feature_names_in_.tolist()}")

    # Prétraitement et prédiction
    X_pred = preprocess_uploaded_df( df.copy(), 
            columns, 
            scaler,
            target_encoder,
            onehot_encoder)
    predictions = model.predict(X_pred)

    # Ajout des scores dans le DataFrame original
    df['score'] = predictions

    # Construction des événements pour le frontend
    events = df.fillna("").to_dict(orient="records")

    # Calcul des métriques
    metrics = {
        "total": len(df),
        "false_positives": int((df['score'] == 0).sum()),
        "true_positives": int((df['score'] == 1).sum()),
        "incidents": int((df['score'] == 2).sum())
    }

    return {
        "events": events,
        "metrics": metrics
    }