"""
Module de visualisation des données
"""
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from config import CRIME_CLASSES, FIGURE_SIZE


def create_pie_chart(df, year, top_n=4):
    """
    Crée un diagramme circulaire pour les régions avec le plus de crimes
    
    Args:
        df (pd.DataFrame): Données agrégées par région
        year (str): Année à visualiser
        top_n (int): Nombre de régions à afficher séparément
        
    Returns:
        matplotlib.figure.Figure: Figure du graphique
    """
    # Filtrer par année et trier
    data_year = df[df['Unite temps'] == year].copy()
    sorted_data = data_year.sort_values(by='Valeurs', ascending=False)
    
    # Top N régions
    top_regions = sorted_data.head(top_n)
    
    # Regrouper les autres
    other_sum = sorted_data.iloc[top_n:]['Valeurs'].sum()
    other = pd.DataFrame({
        'nom_zone': ['Autres régions'],
        'Valeurs': [other_sum]
    })
    
    combined_data = pd.concat([top_regions, other], ignore_index=True)
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(
        combined_data['Valeurs'],
        labels=combined_data['nom_zone'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title(f'Distribution des crimes en {year}', fontsize=14, fontweight='bold')
    
    return fig


def create_time_series_plot(df, region):
    """
    Crée un graphique d'évolution temporelle pour une région
    
    Args:
        df (pd.DataFrame): Données filtrées par région
        region (str): Nom de la région
        
    Returns:
        matplotlib.figure.Figure: Figure du graphique
    """
    # Filtrer par région
    region_data = df[df['nom_zone'] == region].copy()
    region_data = region_data.sort_values('Unite temps')
    
    x = region_data['Unite temps'].values
    y = region_data['Valeurs'].values
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=8, color='steelblue')
    
    # Annotations
    for i, j in zip(x, y):
        ax.annotate(
            f'{int(j):,}',
            xy=(i, j),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9
        )
    
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Nombre de crimes', fontsize=12)
    ax.set_title(f'Évolution du nombre de crimes - {region}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def create_choropleth_map(gdf, df_crime, year='2021'):
    """
    Crée une carte choroplèthe des régions françaises
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame des régions
        df_crime (pd.DataFrame): DataFrame avec les classes de criminalité
        year (str): Année à visualiser
        
    Returns:
        matplotlib.figure.Figure: Figure de la carte
    """
    # Filtrer par année
    df_year = df_crime[df_crime['Unite temps'] == year].copy()
    
    # Fusion avec les géométries
    gdf_merged = gdf.merge(df_year, left_on='nom', right_on='nom_zone', how='left')
    
    # Mapper les couleurs
    color_map = {k: v['color'] for k, v in CRIME_CLASSES.items()}
    gdf_merged['color'] = gdf_merged['classe'].map(color_map)
    
    # Création de la carte
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    gdf_merged.plot(
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
        color=gdf_merged['color'],
        legend=False
    )
    
    # Ajout des noms des régions
    for _, row in gdf_merged.iterrows():
        if pd.notna(row['geometry']):
            centroid = row['geometry'].centroid
            ax.text(
                centroid.x,
                centroid.y,
                row['nom'],
                fontsize=8,
                ha='center',
                va='center'
            )
    
    # Légende personnalisée
    legend_labels = {k: v['label'] for k, v in CRIME_CLASSES.items()}
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color_map[k], label=legend_labels[k])
        for k in sorted(CRIME_CLASSES.keys())
    ]
    
    ax.legend(
        handles=legend_elements,
        title='Niveau de criminalité',
        loc='upper left',
        bbox_to_anchor=(1.05, 1)
    )
    
    ax.set_title(f'Criminalité par région en {year}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig


def create_ranking_table(df, year, ascending=False):
    """
    Crée un tableau de classement des régions
    
    Args:
        df (pd.DataFrame): Données agrégées
        year (str): Année à visualiser
        ascending (bool): Ordre du tri
        
    Returns:
        pd.DataFrame: DataFrame trié
    """
    df_year = df[df['Unite temps'] == year].copy()
    df_sorted = df_year.sort_values(by='Valeurs', ascending=ascending)
    df_sorted['Rang'] = range(1, len(df_sorted) + 1)
    
    # Reformater pour l'affichage
    df_display = df_sorted[['Rang', 'nom_zone', 'Valeurs']].copy()
    df_display.columns = ['Rang', 'Région', 'Nombre de crimes']
    df_display['Nombre de crimes'] = df_display['Nombre de crimes'].apply(lambda x: f'{int(x):,}')
    
    return df_display
