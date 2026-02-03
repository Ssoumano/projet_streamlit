"""
Page de cartographie et prédictions
"""
import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_crime_data,
    load_shapefile,
    preprocess_data,
    get_regional_data,
    create_crime_classes
)
from utils.visualizations import create_choropleth_map
from utils.predictions import (
    predict_crime_classes,
    predict_crime_values,
    create_forecast_visualization
)
from config import PREDICTION_YEARS


def show():
    """
    Affiche la page de cartographie
    """
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4;'>🗺️ Cartographie et Prédictions</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Chargement des données
    with st.spinner("Chargement des données cartographiques..."):
        df_raw = load_crime_data()
        gdf = load_shapefile()
        
        if df_raw is None or gdf is None:
            st.error("Impossible de charger les données nécessaires.")
            st.info("""
            Veuillez vous assurer que les fichiers suivants sont présents:
            - `data/data-gouv-series-chrono.xlsx`
            - `data/regions-20180101-shp/regions-20180101.shp`
            """)
            return
        
        df_processed = preprocess_data(df_raw)
        df_regional = get_regional_data(df_processed)
        df_with_classes = create_crime_classes(df_regional)
    
    # Onglets pour séparer les visualisations
    tab1, tab2, tab3 = st.tabs(["🗺️ Carte des régions", "🔮 Prédictions", "📊 Prévisions détaillées"])
    
    # TAB 1: Carte des régions
    with tab1:
        st.header("Répartition géographique de la criminalité")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            available_years = sorted(df_with_classes['Unite temps'].unique(), reverse=True)
            map_year = st.selectbox(
                "Année à visualiser",
                available_years,
                key='map_year_select'
            )
            
            st.markdown("""
            #### Légende
            - 🟢 **Faible** : Niveau de criminalité bas
            - 🟡 **Moyen** : Niveau de criminalité modéré
            - 🔴 **Élevé** : Niveau de criminalité important
            
            *La classification est basée sur des tertiles calculés pour chaque année.*
            """)
        
        with col2:
            fig_map = create_choropleth_map(gdf, df_with_classes, map_year)
            st.pyplot(fig_map)
        
        # Informations complémentaires
        with st.expander("ℹ️ À propos de la carte"):
            st.markdown("""
            Cette carte représente les régions françaises métropolitaines colorées selon leur niveau 
            de criminalité relatif. Les territoires d'outre-mer sont exclus de cette visualisation.
            
            La classification en trois niveaux est recalculée pour chaque année, ce qui permet de 
            voir l'évolution relative entre les régions plutôt que des valeurs absolues.
            """)
    
    # TAB 2: Prédictions
    with tab2:
        st.header("Prédictions des classes de criminalité")
        
        st.info("""
        💡 Ces prédictions sont basées sur une régression linéaire des tendances historiques. 
        Elles donnent une estimation de l'évolution probable mais ne tiennent pas compte 
        d'événements futurs imprévisibles.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            pred_year = st.selectbox(
                "Année à prédire",
                list(PREDICTION_YEARS),
                key='pred_year_select'
            )
            
            if st.button("🔮 Lancer la prédiction", type="primary"):
                st.session_state['prediction_done'] = True
                st.session_state['pred_year'] = pred_year
        
        with col2:
            if st.session_state.get('prediction_done', False):
                with st.spinner("Calcul des prédictions en cours..."):
                    df_predictions = predict_crime_classes(
                        df_with_classes,
                        st.session_state['pred_year']
                    )
                
                st.success(f"✅ Prédictions pour l'année {st.session_state['pred_year']} calculées !")
                
                # Afficher les résultats
                st.markdown("### Résultats")
                
                # Compter par classe
                class_counts = df_predictions['Classe prédite'].value_counts()
                
                col_a, col_b, col_c = st.columns(3)
                
                colors = {'Classe 1': 'green', 'Classe 2': 'orange', 'Classe 3': 'red'}
                labels = {'Classe 1': 'Faible', 'Classe 2': 'Moyen', 'Classe 3': 'Élevé'}
                
                for col, classe in zip([col_a, col_b, col_c], ['Classe 1', 'Classe 2', 'Classe 3']):
                    count = class_counts.get(classe, 0)
                    with col:
                        st.markdown(
                            f"""<div style='padding: 15px; background-color: {colors[classe]}22; 
                            border-radius: 10px; text-align: center;'>
                            <h4 style='color: {colors[classe]};'>{labels[classe]}</h4>
                            <h2>{count} régions</h2>
                            </div>""",
                            unsafe_allow_html=True
                        )
                
                # Tableau complet
                st.markdown("### Tableau des prédictions")
                st.dataframe(
                    df_predictions.sort_values('Région'),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Téléchargement
                csv = df_predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv,
                    file_name=f'predictions_{st.session_state["pred_year"]}.csv',
                    mime='text/csv'
                )
    
    # TAB 3: Prévisions détaillées
    with tab3:
        st.header("Prévisions détaillées par région")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            region_forecast = st.selectbox(
                "Sélectionner une région",
                sorted(df_regional['nom_zone'].unique()),
                key='forecast_region'
            )
            
            years_to_predict = st.multiselect(
                "Années à prédire",
                list(PREDICTION_YEARS),
                default=list(PREDICTION_YEARS)[:3]
            )
            
            show_forecast = st.button("📈 Générer les prévisions", type="primary")
        
        with col2:
            if show_forecast and years_to_predict:
                with st.spinner("Génération des prévisions..."):
                    # Prédictions pour toutes les régions
                    df_all_predictions = predict_crime_values(df_regional, years_to_predict)
                    
                    # Créer la visualisation
                    fig_forecast = create_forecast_visualization(
                        df_regional,
                        df_all_predictions,
                        region_forecast
                    )
                    
                    st.pyplot(fig_forecast)
                
                # Tableau des valeurs prédites
                pred_region = df_all_predictions[
                    df_all_predictions['nom_zone'] == region_forecast
                ].copy()
                
                pred_region = pred_region.sort_values('Unite temps')
                pred_region['Valeurs_predites'] = pred_region['Valeurs_predites'].apply(
                    lambda x: f'{int(x):,}'
                )
                pred_region.columns = ['Région', 'Année', 'Nombre de crimes prédit']
                
                st.markdown("### Valeurs prédites")
                st.dataframe(
                    pred_region[['Année', 'Nombre de crimes prédit']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Avertissement
                st.warning("""
                ⚠️ **Avertissement** : Ces prévisions sont basées uniquement sur les tendances 
                historiques et ne prennent pas en compte les changements de politique, 
                les événements sociaux, ou d'autres facteurs externes qui pourraient 
                influencer la criminalité.
                """)
            elif show_forecast:
                st.info("Veuillez sélectionner au moins une année à prédire.")
    
    st.markdown("---")
    
    # Section: Méthodologie
    with st.expander("📖 Méthodologie des prédictions"):
        st.markdown("""
        ### Modèle de prédiction
        
        Les prédictions sont générées à l'aide d'un modèle de **régression linéaire** qui analyse 
        les tendances historiques pour chaque région.
        
        #### Processus
        1. **Entraînement** : Le modèle apprend à partir des données historiques (2016-2022)
        2. **Prédiction** : Application du modèle aux années futures
        3. **Classification** : Attribution d'une classe (Faible/Moyen/Élevé) basée sur les tertiles
        
        #### Limitations
        - Le modèle suppose une continuité des tendances passées
        - Ne prend pas en compte les changements de politique ou événements majeurs
        - La précision diminue avec l'éloignement dans le temps
        
        #### Utilisation recommandée
        Ces prédictions sont utiles pour identifier des tendances générales mais ne doivent pas 
        être considérées comme des certitudes. Elles servent d'outil d'aide à la décision.
        """)
