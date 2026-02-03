"""
Application Streamlit pour l'analyse de la criminalité en France
"""
import streamlit as st
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Analyse Criminalité France",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre du menu latéral
st.sidebar.title("Navigation")

# Informations personnelles
st.sidebar.title("Informations personnelles")
st.sidebar.text_input("Nom", value="Soumano", disabled=True)
st.sidebar.text_input("Prénom", value="Seydou", disabled=True)
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/seydou-soumano/)")
st.sidebar.markdown("[GitHub](https://github.com/Ssoumano)")

# Ajout des logos (si disponibles)
logo_efrei_path = Path("LOGO_EFREI-PRINT_EFREI-WEB.png")
logo_other_path = Path("téléchargement.png")

if logo_efrei_path.exists():
    st.sidebar.image(str(logo_efrei_path), use_column_width=True)
if logo_other_path.exists():
    st.sidebar.image(str(logo_other_path), use_column_width=True)

# Navigation entre les pages
page = st.sidebar.radio(
    "Choisir une page",
    ["🏠 Accueil", "📊 Analyses Détaillées", "🗺️ Cartographie"]
)

# Import et affichage de la page sélectionnée
if page == "🏠 Accueil":
    from pages import home
    home.show()
elif page == "📊 Analyses Détaillées":
    from pages import analyses
    analyses.show()
elif page == "🗺️ Cartographie":
    from pages import cartography
    cartography.show()
