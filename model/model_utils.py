import joblib
import os

# Définition des chemins des modèles
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model2.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler1.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "columns1.pkl")
TARGET_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "target_encoder1.pkl")
ONEHOT_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "onehot_encoder1.pkl")

def load_model():
    """Charge le modèle KNN"""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")

def load_scaler():
    """Charge le StandardScaler"""
    try:
        return joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du scaler: {str(e)}")

def load_columns():
    """Charge la liste des colonnes du modèle"""
    try:
        return joblib.load(COLUMNS_PATH)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des colonnes: {str(e)}")

def load_target_encoder():
    """Charge le TargetEncoder"""
    try:
        return joblib.load(TARGET_ENCODER_PATH)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du target encoder: {str(e)}")

def load_onehot_encoder():
    """Charge le OneHotEncoder"""
    try:
        return joblib.load(ONEHOT_ENCODER_PATH)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du onehot encoder: {str(e)}")