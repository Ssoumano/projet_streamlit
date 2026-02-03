"""
Page d'accueil de l'application
"""
import streamlit as st
from utils.data_loader import (
    load_crime_data,
    preprocess_data,
    get_regional_data
)
from utils.visualizations import (
    create_pie_chart,
    create_time_series_plot,
    create_ranking_table
)


def show():
    """
    Affiche la page d'accueil
    """
    # Titre principal
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4;'>📊 Visualisation de la criminalité en France</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ### 👋 Bienvenue
    
    Cette application permet d'analyser l'évolution de la criminalité en France à travers différentes régions.
    Les données proviennent de **Data.gouv** et couvrent plusieurs années.
    
    #### 🎯 Objectifs
    - Identifier les régions les plus sûres et les plus dangereuses
    - Analyser les tendances temporelles
    - Prédire l'évolution future de la criminalité
    """)
    
    st.markdown("---")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        df_raw = load_crime_data()
        
        if df_raw is None:
            st.error("Impossible de charger les données. Veuillez vérifier les fichiers.")
            return
        
        df_processed = preprocess_data(df_raw)
        df_regional = get_regional_data(df_processed)
    
    # Section: Aperçu des données
    st.header("📋 Aperçu des données")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de régions", df_regional['nom_zone'].nunique())
    with col2:
        years = sorted(df_regional['Unite temps'].unique())
        st.metric("Période couverte", f"{years[0]} - {years[-1]}")
    with col3:
        total_crimes = df_regional['Valeurs'].sum()
        st.metric("Total crimes enregistrés", f"{int(total_crimes):,}")
    
    with st.expander("Voir un échantillon des données"):
        st.dataframe(df_processed.head(20))
    
    st.markdown("---")
    
    # Section: Distribution par année
    st.header("🥧 Distribution des crimes par région")
    
    available_years = sorted(df_regional['Unite temps'].unique(), reverse=True)
    selected_year = st.selectbox(
        "Sélectionner une année",
        available_years,
        key='home_year_select'
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_pie = create_pie_chart(df_regional, selected_year, top_n=4)
        st.pyplot(fig_pie)
    
    with col2:
        st.markdown("#### 📊 Top 5 des régions")
        ranking_table = create_ranking_table(df_regional, selected_year)
        st.dataframe(
            ranking_table.head(5),
            hide_index=True,
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Section: Évolution temporelle
    st.header("📈 Évolution temporelle par région")
    
    regions = sorted(df_regional['nom_zone'].unique())
    selected_region = st.selectbox(
        "Sélectionner une région",
        regions,
        key='home_region_select'
    )
    
    fig_time = create_time_series_plot(df_regional, selected_region)
    st.pyplot(fig_time)
    
    # Statistiques pour la région sélectionnée
    region_stats = df_regional[df_regional['nom_zone'] == selected_region]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Crimes en 2021",
            f"{int(region_stats[region_stats['Unite temps'] == '2021']['Valeurs'].iloc[0]):,}" 
            if len(region_stats[region_stats['Unite temps'] == '2021']) > 0 else "N/A"
        )
    with col2:
        avg_crimes = region_stats['Valeurs'].mean()
        st.metric("Moyenne annuelle", f"{int(avg_crimes):,}")
    with col3:
        max_year = region_stats.loc[region_stats['Valeurs'].idxmax(), 'Unite temps']
        st.metric("Année pic", max_year)
    
    st.markdown("---")
    
    # Section: Guide d'utilisation
    with st.expander("ℹ️ Guide d'utilisation"):
        st.markdown("""
        #### Navigation
        - **🏠 Accueil** : Vue d'ensemble et statistiques générales
        - **📊 Analyses Détaillées** : Tableaux de classement et comparaisons
        - **🗺️ Cartographie** : Visualisation spatiale et prédictions
        
        #### Fonctionnalités
        - Sélectionnez différentes années pour voir l'évolution
        - Comparez les régions entre elles
        - Consultez les prédictions pour les années futures
        
        #### Sources de données
        Les données proviennent de [Data.gouv](https://www.data.gouv.fr/) et sont 
        mises à jour régulièrement par les autorités compétentes.
        """)
