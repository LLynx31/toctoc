import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Chargement et prétraitement des données
df = pd.read_excel("Labelisation_Alert_EDR_v3.xlsx")

# Suppression des colonnes inutiles
colonnes_a_supprimer = [
    'unique_id','process_id','process_unique_id','link','md5','sha256','ioc_value',
    'feed_id','watchlist_id','segment_id','sensor_id','Unnamed: 0','watchlist_name',
    'process_path','comms_ip','interface_ip','hostname'
]
df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns], inplace=True, errors='ignore')

# Expansion de la colonne 'ioc_attr'
df['ioc_attr'] = df['ioc_attr'].fillna('{}')
df['ioc_attr'] = df['ioc_attr'].apply(ast.literal_eval)
df_ioc = pd.json_normalize(df['ioc_attr'])
df = df.drop(columns=['ioc_attr']).join(df_ioc)

# Suppression des colonnes techniques inutiles issues de l'expansion
for col in ['dns_name','port','remote_port','local_port','local_ip', 'protocol', 'remote_ip']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Suppression des lignes où ioc_type == "query" et de la colonne ioc_type
if 'ioc_type' in df.columns:
    df = df[df["ioc_type"] != "query"]
    df.drop(columns=['ioc_type'], inplace=True, errors='ignore')

# Création de la colonne score
def calculeScore(ligne):
    if 'labelisation' in ligne and 'incident' in ligne:
        if ligne['labelisation'] == True and ligne['incident'] == 1:
            return 2
        elif ligne['labelisation'] == True and ligne['incident'] == 0:
            return 1
    return 0

if 'labelisation' in df.columns and 'incident' in df.columns:
    df["score"] = df.apply(calculeScore, axis=1)
    df.drop(columns=['labelisation', 'incident'], inplace=True, errors='ignore')

# Transformation de la colonne 'created_time'
if 'created_time' in df.columns:
    df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce').dt.time

# Encodage des variables catégorielles
df = pd.get_dummies(df)

# Séparation features/cible
X = df.drop(columns=['score'])
y = df['score']

# Standardisation
scaler = StandardScaler()
numericols = X.select_dtypes(include=['float64', 'int64']).columns
X[numericols] = scaler.fit_transform(X[numericols])

# Entraînement du modèle
model = RandomForestClassifier(random_state=42, n_estimators=80)
model.fit(X, y)

# Sauvegarde du modèle, du scaler et des colonnes
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "columns.pkl")