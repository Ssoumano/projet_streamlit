"""
Module de chargement et traitement des données
"""
import pandas as pd
import geopandas as gpd
import streamlit as st
from config import EXCEL_FILE, SHAPEFILE_PATH, REGIONS_TO_EXCLUDE


@st.cache_data
def load_crime_data():
    """
    Charge les données de criminalité depuis le fichier Excel
    
    Returns:
        pd.DataFrame: DataFrame avec les données de criminalité
    """
    try:
        df = pd.read_excel(EXCEL_FILE)
        return df
    except FileNotFoundError:
        st.error(f"Fichier non trouvé: {EXCEL_FILE}")
        st.info("Veuillez placer le fichier 'data-gouv-series-chrono.xlsx' dans le dossier 'data/'")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None


@st.cache_data
def load_shapefile():
    """
    Charge le shapefile des régions françaises
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame avec les géométries des régions
    """
    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
        # Filtrer les régions d'outre-mer
        gdf_filtered = gdf[~gdf['nom'].isin(REGIONS_TO_EXCLUDE)].reset_index(drop=True)
        return gdf_filtered
    except FileNotFoundError:
        st.error(f"Fichier shapefile non trouvé: {SHAPEFILE_PATH}")
        st.info("Veuillez placer le dossier 'regions-20180101-shp' dans le dossier 'data/'")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du shapefile: {e}")
        return None


def extract_nom_zone(row):
    """
    Extrait le nom de la zone géographique
    
    Args:
        row: Ligne du DataFrame
        
    Returns:
        str: Nom de la zone
    """
    zone = row['Zone_geographique']
    if '-' in zone:
        return zone.split('-', 1)[1].split('(')[0].strip()
    else:
        return zone.split('(')[0].strip()


def preprocess_data(df):
    """
    Prétraite les données de criminalité
    
    Args:
        df (pd.DataFrame): DataFrame brut
        
    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    if df is None:
        return None
    
    # Création des colonnes dérivées
    df['nom_zone'] = df.apply(extract_nom_zone, axis=1)
    df['type_zone'] = df['Zone_geographique'].str.extract(r'\((.*?)\)')
    
    # Filtrer uniquement les statistiques de type "Nombre"
    df_filtered = df[df['Statistique'] == 'Nombre'].copy()
    
    return df_filtered


def get_regional_data(df):
    """
    Filtre et agrège les données par région
    
    Args:
        df (pd.DataFrame): DataFrame prétraité
        
    Returns:
        pd.DataFrame: Données agrégées par région et année
    """
    if df is None:
        return None
    
    # Filtrer les données régionales
    regional_data = df[df['type_zone'] == 'région']
    
    # Agréger par année et région
    grouped_data = regional_data.groupby(
        ['Unite temps', 'nom_zone']
    )['Valeurs'].sum().reset_index()
    
    return grouped_data


def create_crime_classes(df, n_classes=3):
    """
    Crée des classes de criminalité basées sur les quantiles
    
    Args:
        df (pd.DataFrame): DataFrame avec les données de criminalité
        n_classes (int): Nombre de classes à créer
        
    Returns:
        pd.DataFrame: DataFrame avec la colonne 'classe' ajoutée
    """
    if df is None:
        return None
    
    def assign_classes(group):
        labels = [f'Classe {i+1}' for i in range(n_classes)]
        group['classe'] = pd.qcut(
            group['Valeurs'], 
            q=n_classes, 
            labels=labels,
            duplicates='drop'
        )
        return group
    
    df_with_classes = df.groupby('Unite temps', group_keys=False).apply(assign_classes)
    return df_with_classes
