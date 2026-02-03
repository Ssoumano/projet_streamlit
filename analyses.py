"""
Page d'analyses détaillées
"""
import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_crime_data,
    preprocess_data,
    get_regional_data,
    create_crime_classes
)
from utils.visualizations import create_ranking_table


def show():
    """
    Affiche la page d'analyses détaillées
    """
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4;'>📊 Analyses Détaillées</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        df_raw = load_crime_data()
        
        if df_raw is None:
            st.error("Impossible de charger les données.")
            return
        
        df_processed = preprocess_data(df_raw)
        df_regional = get_regional_data(df_processed)
        df_with_classes = create_crime_classes(df_regional)
    
    # Section: Classement des régions
    st.header("🏆 Classement des régions")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        available_years = sorted(df_regional['Unite temps'].unique(), reverse=True)
        selected_year = st.selectbox(
            "Année",
            available_years,
            key='analyses_year_select'
        )
        
        sort_order = st.radio(
            "Ordre de tri",
            ["Du plus élevé au plus bas", "Du plus bas au plus élevé"],
            key='sort_order'
        )
    
    with col2:
        ascending = (sort_order == "Du plus bas au plus élevé")
        ranking_df = create_ranking_table(df_regional, selected_year, ascending=ascending)
        
        st.dataframe(
            ranking_df,
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        # Option de téléchargement
        csv = ranking_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger le classement (CSV)",
            data=csv,
            file_name=f'classement_criminalite_{selected_year}.csv',
            mime='text/csv'
        )
    
    st.markdown("---")
    
    # Section: Classification par niveaux
    st.header("🎯 Classification par niveaux de criminalité")
    
    if df_with_classes is not None:
        year_for_classes = st.selectbox(
            "Sélectionner une année",
            sorted(df_with_classes['Unite temps'].unique(), reverse=True),
            key='classes_year_select'
        )
        
        df_year = df_with_classes[df_with_classes['Unite temps'] == year_for_classes].copy()
        
        # Statistiques par classe
        col1, col2, col3 = st.columns(3)
        
        for i, (col, classe) in enumerate(zip([col1, col2, col3], ['Classe 1', 'Classe 2', 'Classe 3'])):
            count = len(df_year[df_year['classe'] == classe])
            label = ['Faible', 'Moyen', 'Élevé'][i]
            color = ['green', 'orange', 'red'][i]
            
            with col:
                st.markdown(
                    f"""<div style='padding: 20px; background-color: {color}22; border-radius: 10px; text-align: center;'>
                    <h3 style='color: {color};'>{label}</h3>
                    <h2>{count} régions</h2>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        st.markdown("#### Détail par classe")
        
        tabs = st.tabs(["🟢 Classe 1 (Faible)", "🟡 Classe 2 (Moyen)", "🔴 Classe 3 (Élevé)"])
        
        for i, (tab, classe) in enumerate(zip(tabs, ['Classe 1', 'Classe 2', 'Classe 3'])):
            with tab:
                df_classe = df_year[df_year['classe'] == classe][['nom_zone', 'Valeurs']].copy()
                df_classe = df_classe.sort_values('Valeurs', ascending=False)
                df_classe.columns = ['Région', 'Nombre de crimes']
                df_classe['Nombre de crimes'] = df_classe['Nombre de crimes'].apply(
                    lambda x: f'{int(x):,}'
                )
                
                if len(df_classe) > 0:
                    st.dataframe(df_classe, hide_index=True, use_container_width=True)
                else:
                    st.info("Aucune région dans cette classe pour l'année sélectionnée.")
    
    st.markdown("---")
    
    # Section: Comparaison multi-années
    st.header("📅 Évolution multi-années")
    
    col1, col2 = st.columns(2)
    
    with col1:
        region_to_compare = st.selectbox(
            "Sélectionner une région",
            sorted(df_regional['nom_zone'].unique()),
            key='compare_region'
        )
    
    with col2:
        # Récupérer les données pour cette région
        region_data = df_regional[df_regional['nom_zone'] == region_to_compare].copy()
        region_data = region_data.sort_values('Unite temps')
        
        if len(region_data) > 1:
            first_year = region_data.iloc[0]
            last_year = region_data.iloc[-1]
            
            evolution = ((last_year['Valeurs'] - first_year['Valeurs']) / first_year['Valeurs']) * 100
            
            st.metric(
                f"Évolution {first_year['Unite temps']} → {last_year['Unite temps']}",
                f"{evolution:+.1f}%",
                delta=f"{int(last_year['Valeurs'] - first_year['Valeurs']):+,} crimes"
            )
    
    # Tableau d'évolution
    if len(region_data) > 0:
        pivot_data = region_data.copy()
        pivot_data['Valeurs'] = pivot_data['Valeurs'].apply(lambda x: f'{int(x):,}')
        pivot_data = pivot_data[['Unite temps', 'Valeurs']].set_index('Unite temps').T
        
        st.dataframe(pivot_data, use_container_width=True)
    
    st.markdown("---")
    
    # Section: Statistiques globales
    st.header("📈 Statistiques globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_by_year = df_regional.groupby('Unite temps')['Valeurs'].sum()
    
    with col1:
        max_year = total_by_year.idxmax()
        st.metric(
            "Année la plus élevée",
            max_year,
            f"{int(total_by_year.max()):,} crimes"
        )
    
    with col2:
        min_year = total_by_year.idxmin()
        st.metric(
            "Année la plus basse",
            min_year,
            f"{int(total_by_year.min()):,} crimes"
        )
    
    with col3:
        avg_total = total_by_year.mean()
        st.metric(
            "Moyenne annuelle",
            f"{int(avg_total):,}"
        )
    
    with col4:
        std_dev = total_by_year.std()
        st.metric(
            "Écart-type",
            f"{int(std_dev):,}"
        )
    
    # Graphique d'évolution totale
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(total_by_year.index, total_by_year.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Nombre total de crimes', fontsize=12)
    ax.set_title('Évolution totale de la criminalité en France', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(total_by_year.values):
        ax.text(i, v, f'{int(v):,}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
