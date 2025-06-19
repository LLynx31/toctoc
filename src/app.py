import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_utils import load_model, load_scaler, load_columns
from data.preprocess import preprocess_uploaded_df
import matplotlib.pyplot as plt

st.set_page_config(page_title="Security Event Classifier", layout="wide")

# Styles CSS globaux
st.markdown("""
<style>
    /* Styles généraux */
    .main {
        padding: 2rem;
    }
    
    /* Styles pour les titres */
    h1 {
        color: #1E3D59;
        padding: 1rem 0;
        border-bottom: 2px solid #FF6B6B;
    }
    
    h2 {
        color: #1E3D59;
        margin-top: 2rem;
    }
    
    /* Styles pour les métriques */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3D59;
    }
    
    /* Styles pour le tableau */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Styles pour la sidebar */
    .sidebar {
        background-color: #F7F9FC;
        padding: 1rem;
    }
    
    /* Styles pour les boutons */
    .stButton button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Styles pour les filtres */
    .filter-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Gestion de l'historique des uploads ---
UPLOAD_HISTORY_FILE = "upload_history.csv"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_history(filename):
    if os.path.exists(UPLOAD_HISTORY_FILE):
        hist = pd.read_csv(UPLOAD_HISTORY_FILE)
    else:
        hist = pd.DataFrame(columns=["filename"])
    if filename not in hist["filename"].values:
        hist = pd.concat([hist, pd.DataFrame([{"filename": filename}])], ignore_index=True)
        hist.to_csv(UPLOAD_HISTORY_FILE, index=False)

def get_upload_history():
    if os.path.exists(UPLOAD_HISTORY_FILE):
        return pd.read_csv(UPLOAD_HISTORY_FILE)["filename"].tolist()
    return []

# --- PAGE D'ACCUEIL : UPLOAD ---
if "page" not in st.session_state:
    st.session_state.page = "home"

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

if st.session_state.page == "home":
    st.markdown("<h1 style='text-align:center;'>SECURITY EVENT CLASSIFIER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload your Excel file to analyze and prioritize security events.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        df = pd.read_excel(file_path)
        st.session_state.df = df
        st.session_state.uploaded_filename = uploaded_file.name
        save_upload_history(uploaded_file.name)
        st.session_state.page = "dashboard"
        st.rerun()

# --- DASHBOARD ---
if st.session_state.page == "dashboard":
    # Sidebar améliorée
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-shield.png", width=50)
        st.title("Security Dashboard")
        st.markdown("---")
    
    # Bouton pour nouveau upload
    if st.sidebar.button("➕ Nouveau fichier"):
        st.session_state.page = "home"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Sélection du fichier dans l'historique
    st.subheader("📋 Upload History")
    history = get_upload_history()
    if "current_file" not in st.session_state:
        st.session_state.current_file = st.session_state.uploaded_filename
    
    selected_file = st.sidebar.selectbox(
        "Sélectionner un fichier",
        history,
        index=history.index(st.session_state.current_file)
    )
    
    # Charger le fichier sélectionné si différent
    if selected_file != st.session_state.current_file:
        file_path = os.path.join(UPLOAD_DIR, selected_file)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            st.session_state.df = df
            st.session_state.current_file = selected_file
            st.rerun()
    
    # Afficher le fichier actuel
    st.sidebar.markdown(f"**Fichier actuel:** {st.session_state.current_file}")
    
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("Aucun fichier chargé.")
        st.session_state.page = "home"
        st.rerun()


    # Prédiction automatique si modèle disponible

    model = load_model()
    scaler = load_scaler()
    columns = load_columns()
    # Dans la partie dashboard, remplacer le bloc de prédiction par :
    if model is not None and scaler is not None and columns is not None:
        try:
            X_pred, to_predict = preprocess_uploaded_df(df.copy(), columns, scaler)
            # Initialiser tous les scores à 0
            df['score'] = 0
            # Faire la prédiction uniquement sur les lignes valides
            predictions = model.predict(X_pred[to_predict])
            # Mettre à jour les scores uniquement pour les lignes prédites
            df.loc[to_predict, 'score'] = predictions
            st.success("Prédictions ajoutées avec succès!")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")
    
    # Affichage des métriques
    if 'score' in df.columns:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        cols = st.columns(4)
        
        # Métriques avec icônes et couleurs
        with cols[0]:
            st.metric("📊 Total Events", 
                     f"{len(df):,}",
                     delta=None)
        
        with cols[1]:
            st.metric("✅ False Positives", 
                     f"{(df['score']==0).sum():,}",
                     f"{((df['score']==0).sum()/len(df))*100:.1f}%")
        
        with cols[2]:
            st.metric("⚠️ True Positives",
                     f"{(df['score']==1).sum():,}",
                     f"{((df['score']==1).sum()/len(df))*100:.1f}%")
        
        with cols[3]:
            st.metric("🚨 Incidents",
                     f"{(df['score']==2).sum():,}",
                     f"{((df['score']==2).sum()/len(df))*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Ajout de visualisations
        st.subheader("Analyse des scores")
        col1, col2 = st.columns(2)
        
        with col1:
            # Diagramme circulaire des scores avec texte en blanc
            fig_pie = plt.figure(figsize=(8, 8))
            score_counts = df['score'].value_counts().sort_index()
            wedges, texts, autotexts = plt.pie(
                score_counts, 
                labels=[f'Score {i} ({v:,})' for i, v in score_counts.items()],
                autopct='%1.1f%%', 
                colors=['#4CAF50', '#e67e22', 'red'],
                explode=(0.1, 0.1, 0.2),
                textprops={'color': 'white', 'fontsize': 14}
            )
            plt.title('Distribution des scores', color='white')
            plt.legend(['Faux positifs', 'Vrai positifs', 'Incidents'], facecolor='white')
            # Mettre aussi les labels de légende en blanc
            for text in texts + autotexts:
                text.set_color('white')
            st.pyplot(fig_pie)
            
    else:
        st.warning("Pas de prédictions disponibles")

    # --- Recherche et filtres ---
    st.subheader("Filtrer et rechercher")
    search = st.text_input("Recherche")
    score_filter = st.multiselect("Filtrer par score", options=[0,1,2], default=[0,1,2])

    filtered_df = df[df['score'].isin(score_filter)]
    if search:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
    
    # --- Tableau paginé avec couleurs ---
    st.subheader("Évènements détectés")
    page_size = st.selectbox("Évènements par page", [10, 20, 50], index=0)
    page_num = st.number_input("Page", min_value=1, max_value=(len(filtered_df)//page_size)+1, value=1)
    start = (page_num-1)*page_size
    end = start+page_size
    page_df = filtered_df.iloc[start:end].copy()

    def color_score(val, props=''):
        color = {
            0: 'background-color: #4CAF50; color: white;',
            1: 'background-color: #e67e22; color: white;',
            2: 'background-color: red; color: white;'
        }
        return color.get(val, '')

    st.markdown("""
    <style>
    .stDataFrame {
        height: 600px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if "score" in page_df.columns:
        styled_df = page_df.style.apply(
            lambda row: [color_score(row['score'])] * len(row),
            axis=1
        ).set_properties(**{
            'height': '50px',
            'font-size': '14px',
            'padding': '10px'
        })
        st.dataframe(styled_df, use_container_width=True, height=600)
    else:
        st.dataframe(page_df, use_container_width=True, height=600)

    # Pagination améliorée
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"Page {page_num} sur {(len(filtered_df)//page_size)+1}")