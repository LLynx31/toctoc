import pandas as pd
import ast
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Définition des colonnes catégorielles pour chaque type d'encodage
COLS_ONE_HOT = ['description', 'feed_name', 'os_type', 'direction']
COLS_TARGET_ENCODE = ['process_name', 'created_time']

def preprocess_uploaded_df(df: pd.DataFrame, 
                         columns: list, 
                         scaler,
                         target_encoder=None,
                         onehot_encoder=None):
    """
    Prétraite les données uploadées en suivant le même processus que l'entraînement
    """
    try:
        # Faire une copie et garder l'index original
        original_index = df.index
        df = df.copy()
        
        # Réinitialiser l'index temporairement
        df = df.reset_index(drop=True)
        
        # Suppression des colonnes inutiles
        colonnes_a_supprimer = [
            'unique_id', 'process_id', 'process_unique_id', 'link', 'md5', 'sha256',
            'ioc_value', 'feed_id', 'watchlist_id', 'segment_id', 'sensor_id',
            'Unnamed: 0', 'watchlist_name', 'process_path', 'comms_ip', 'interface_ip',
            'hostname', "alert_severity", "alert_type", "feed_rating", "group",
            "ioc_confidence", "report_ignored", "report_score", "sensor_criticality",
            "status", "total_hosts"
        ]
        df = df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns], 
                    errors='ignore')

        # Expansion de la colonne 'ioc_attr'
        if 'ioc_attr' in df.columns:
            df['ioc_attr'] = df['ioc_attr'].fillna('{}')
            df['ioc_attr'] = df['ioc_attr'].apply(lambda x: ast.literal_eval(str(x)))
            try:
                df_ioc = pd.json_normalize(df['ioc_attr'])
                df = df.drop(columns=['ioc_attr']).join(df_ioc)
            except Exception as e:
                logger.error(f"Erreur lors de l'expansion de ioc_attr: {e}")
                df = df.drop(columns=['ioc_attr'])

        # Suppression des colonnes techniques
        cols_to_drop = ['dns_name', 'port', 'remote_port', 'local_port', 
                       'local_ip', 'protocol', 'remote_ip','ioc_type']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], 
                    errors='ignore')

        # # Marquage des lignes à prédire (filtrage des queries)
        # if 'ioc_type' in df.columns:
        #     df['to_predict'] = df["ioc_type"] != "query"
        #     df = df.drop(columns=['ioc_type'], errors='ignore')
        # else:
        #     df['to_predict'] = True

        if 'labelisation' in df.columns and 'incident' in df.columns:
            df.drop(columns=['labelisation', 'incident'], inplace=True, errors='ignore')

        # Transformation de created_time
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'], 
                                              errors='coerce').dt.time

        # Application du Target Encoding
        if target_encoder:
            df = target_encoder.transform(df)

        # Application du One-Hot Encoding
        if onehot_encoder:
            cat_encoded = onehot_encoder.transform(df[COLS_ONE_HOT])
            feature_names = onehot_encoder.get_feature_names_out(COLS_ONE_HOT)
            cat_encoded_df = pd.DataFrame(cat_encoded, 
                                        columns=feature_names, 
                                        index=df.index)
            df = pd.concat([df.drop(columns=COLS_ONE_HOT), cat_encoded_df], 
                            axis=1)

        # Alignement des colonnes avec celles du modèle
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        df = df[columns]

        # Standardisation des features numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Restauration de l'index original
        df.index = original_index

        logger.info("Prétraitement terminé avec succès")
        return df

    except Exception as e:
        logger.error(f"Erreur durant le prétraitement: {str(e)}")
        raise RuntimeError(f"Erreur de prétraitement: {str(e)}")