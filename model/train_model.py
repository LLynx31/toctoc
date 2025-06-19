import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition des colonnes catégorielles pour chaque type d'encodage
COLS_ONE_HOT = ['description', 'feed_name', 'os_type', 'direction']
COLS_TARGET_ENCODE = ['process_name', 'created_time']

# Liste des colonnes à supprimer
COLS_TO_DROP = [
    'unique_id', 'process_id', 'process_unique_id', 'link', 'md5', 'sha256',
    'ioc_value', 'feed_id', 'watchlist_id', 'segment_id', 'sensor_id',
    'Unnamed: 0', 'watchlist_name', 'process_path', 'comms_ip', 'interface_ip',
    'hostname', "alert_severity", "alert_type", "feed_rating", "group",
    "ioc_confidence", "report_ignored", "report_score", "sensor_criticality",
    "status", "total_hosts"
]

def load_and_preprocess_data(filepath):
    logger.info("Chargement des données...")
    df = pd.read_excel(filepath)
    
    # Suppression des colonnes inutiles
    df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], 
            inplace=True, errors='ignore')

    # Traitement de ioc_attr
    logger.info("Traitement de la colonne ioc_attr...")
    df['ioc_attr'] = df['ioc_attr'].fillna('{}')
    df['ioc_attr'] = df['ioc_attr'].apply(ast.literal_eval)
    df_ioc = pd.json_normalize(df['ioc_attr'])
    df = df.drop(columns=['ioc_attr']).join(df_ioc)

    # Suppression des colonnes techniques
    for col in ['dns_name', 'port', 'remote_port', 'local_port', 'local_ip', 
                'protocol', 'remote_ip']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Traitement de ioc_type
    if 'ioc_type' in df.columns:
        df = df[df["ioc_type"] != "query"]
        df.drop(columns=['ioc_type'], inplace=True)

    # Calcul du score
    logger.info("Calcul des scores...")
    df['score'] = df.apply(lambda row: 2 if row['labelisation'] and row['incident'] else
                                      1 if row['labelisation'] else 0, axis=1)
    df.drop(columns=['labelisation', 'incident'], inplace=True)

    # Transformation de created_time
    if 'created_time' in df.columns:
        df['created_time'] = pd.to_datetime(df['created_time'], 
                                          errors='coerce').dt.time
    
    return df

def train_model(df):
    logger.info("Préparation des données d'entraînement...")
    X = df.drop(columns=['score'])
    y = df['score']

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Target Encoding
    logger.info("Application du Target Encoding...")
    target_encoder = TargetEncoder(cols=COLS_TARGET_ENCODE)
    X_train = target_encoder.fit_transform(X_train, y_train)
    X_test = target_encoder.transform(X_test)

    # One-Hot Encoding
    logger.info("Application du One-Hot Encoding...")
    onehot_encoder = OneHotEncoder(
        handle_unknown='ignore',
        drop='first',
        sparse_output=False
    )
    
    # Transformation des colonnes catégorielles
    cat_encoded_train = onehot_encoder.fit_transform(X_train[COLS_ONE_HOT])
    cat_encoded_test = onehot_encoder.transform(X_test[COLS_ONE_HOT])
    
    # Récupération des noms de features
    feature_names = onehot_encoder.get_feature_names_out(COLS_ONE_HOT)
    
    # Création des DataFrames encodés
    cat_encoded_train_df = pd.DataFrame(
        cat_encoded_train, 
        columns=feature_names, 
        index=X_train.index
    )
    cat_encoded_test_df = pd.DataFrame(
        cat_encoded_test, 
        columns=feature_names, 
        index=X_test.index
    )
    
    # Fusion des données
    X_train = pd.concat(
        [X_train.drop(columns=COLS_ONE_HOT), cat_encoded_train_df], 
        axis=1
    )
    X_test = pd.concat(
        [X_test.drop(columns=COLS_ONE_HOT), cat_encoded_test_df], 
        axis=1
    )

    # Standardisation
    logger.info("Standardisation des features numériques...")
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Application de SMOTE
    logger.info("Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Entraînement du modèle KNN
    logger.info("Entraînement du modèle KNN...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    return model, scaler, target_encoder, onehot_encoder, list(X_train.columns)

def main():
    try:
        # Chargement et prétraitement
        df = load_and_preprocess_data("../Labelisation_Alert_EDR_v3.xlsx")
        
        # Entraînement
        model, scaler, target_encoder, onehot_encoder, columns = train_model(df)

        # Sauvegarde des artefacts
        logger.info("Sauvegarde des artefacts du modèle...")
        joblib.dump(model, "model2.pkl")
        joblib.dump(scaler, "scaler1.pkl")
        joblib.dump(target_encoder, "target_encoder1.pkl")
        joblib.dump(onehot_encoder, "onehot_encoder1.pkl")
        joblib.dump(columns, "columns1.pkl")
        
        logger.info("Entraînement terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {str(e)}")
        raise

if __name__ == "__main__":
    main()