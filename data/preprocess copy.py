import pandas as pd
import ast
import numpy as np

def preprocess_uploaded_df(df, columns, scaler):
    # Faire une copie et garder l'index original
    original_index = df.index
    df = df.copy()
    
    # Réinitialiser l'index temporairement
    df = df.reset_index(drop=True)
    
    # Suppression des colonnes inutiles
    colonnes_a_supprimer = [
        'unique_id', 'process_id', 'process_unique_id', 'link', 'md5', 'sha256', 'ioc_value',
        'feed_id', 'watchlist_id', 'segment_id', 'sensor_id', 'Unnamed: 0', 'watchlist_name',
        'process_path', 'comms_ip', 'interface_ip', 'hostname'
    ]
    df = df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns], errors='ignore')

    # Expansion de la colonne 'ioc_attr'
    if 'ioc_attr' in df.columns:
        df['ioc_attr'] = df['ioc_attr'].fillna('{}')
        df['ioc_attr'] = df['ioc_attr'].apply(lambda x: ast.literal_eval(str(x)))
        try:
            df_ioc = pd.json_normalize(df['ioc_attr'])
            df = df.drop(columns=['ioc_attr']).join(df_ioc)
        except Exception as e:
            print(f"Erreur lors de l'expansion de ioc_attr: {e}")
            df = df.drop(columns=['ioc_attr'])

    # Suppression des colonnes techniques inutiles issues de l'expansion
    cols_to_drop = ['dns_name', 'port', 'remote_port', 'local_port', 'local_ip', 'protocol', 'remote_ip']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Au lieu de supprimer les lignes, marquons-les pour le filtrage
    if 'ioc_type' in df.columns:
        df['to_predict'] = df["ioc_type"] != "query"
        df = df.drop(columns=['ioc_type'], errors='ignore')
    else:
        df['to_predict'] = True

    # Transformation de la colonne 'created_time'
    if 'created_time' in df.columns:
        df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce').dt.time

    # Encodage des variables catégorielles
    df = pd.get_dummies(df, dummy_na=True)

    # Alignement des colonnes avec celles du modèle
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Standardisation
    numericols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numericols] = scaler.transform(df[numericols])

    # Retourner le DataFrame prétraité et le masque des lignes à prédire
    return df, df['to_predict'] if 'to_predict' in df.columns else pd.Series(True, index=df.index)