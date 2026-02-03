"""
Module de prédiction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st


def train_linear_model(df, region):
    """
    Entraîne un modèle de régression linéaire pour une région
    
    Args:
        df (pd.DataFrame): Données d'entraînement
        region (str): Nom de la région
        
    Returns:
        LinearRegression: Modèle entraîné
    """
    df_region = df[df['nom_zone'] == region].copy()
    
    if len(df_region) < 2:
        return None
    
    # Préparation des données
    X = df_region['Unite temps'].values.reshape(-1, 1)
    y = df_region['Valeurs'].values
    
    # Entraînement
    model = LinearRegression()
    model.fit(X, y)
    
    return model


def predict_crime_values(df, years_to_predict):
    """
    Prédit les valeurs de criminalité pour des années futures
    
    Args:
        df (pd.DataFrame): Données historiques
        years_to_predict (list): Liste des années à prédire
        
    Returns:
        pd.DataFrame: DataFrame avec les prédictions
    """
    regions = df['nom_zone'].unique()
    predictions = []
    
    for region in regions:
        model = train_linear_model(df, region)
        
        if model is None:
            continue
        
        for year in years_to_predict:
            predicted_value = model.predict([[year]])[0]
            # S'assurer que la prédiction n'est pas négative
            predicted_value = max(0, predicted_value)
            
            predictions.append({
                'nom_zone': region,
                'Unite temps': year,
                'Valeurs_predites': predicted_value
            })
    
    df_predictions = pd.DataFrame(predictions)
    return df_predictions


def predict_crime_classes(df_with_classes, target_year):
    """
    Prédit les classes de criminalité pour une année future
    
    Args:
        df_with_classes (pd.DataFrame): Données avec classes historiques
        target_year (int): Année cible pour la prédiction
        
    Returns:
        pd.DataFrame: DataFrame avec les classes prédites
    """
    # Convertir les classes en valeurs numériques
    df_train = df_with_classes.copy()
    df_train['classe_num'] = df_train['classe'].str.extract('(\d+)').astype(float)
    df_train['Unite temps'] = pd.to_numeric(df_train['Unite temps'])
    
    regions = df_train['nom_zone'].unique()
    predictions = []
    
    for region in regions:
        df_region = df_train[df_train['nom_zone'] == region]
        
        if len(df_region) < 2:
            continue
        
        # Entraînement
        X = df_region['Unite temps'].values.reshape(-1, 1)
        y = df_region['classe_num'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédiction
        predicted_class_num = model.predict([[target_year]])[0]
        
        # Arrondir et contraindre entre 1 et 3
        predicted_class_num = np.clip(round(predicted_class_num), 1, 3)
        predicted_class = f'Classe {int(predicted_class_num)}'
        
        predictions.append({
            'Région': region,
            'Classe prédite': predicted_class,
            'Année': target_year
        })
    
    df_predictions = pd.DataFrame(predictions)
    return df_predictions


def create_forecast_visualization(df_historical, df_predictions, region):
    """
    Crée une visualisation des prédictions avec les données historiques
    
    Args:
        df_historical (pd.DataFrame): Données historiques
        df_predictions (pd.DataFrame): Prédictions
        region (str): Nom de la région
        
    Returns:
        matplotlib.figure.Figure: Figure du graphique
    """
    import matplotlib.pyplot as plt
    
    # Filtrer par région
    hist = df_historical[df_historical['nom_zone'] == region].copy()
    pred = df_predictions[df_predictions['nom_zone'] == region].copy()
    
    hist = hist.sort_values('Unite temps')
    pred = pred.sort_values('Unite temps')
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Données historiques
    ax.plot(
        hist['Unite temps'],
        hist['Valeurs'],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=8,
        color='steelblue',
        label='Données historiques'
    )
    
    # Prédictions
    ax.plot(
        pred['Unite temps'],
        pred['Valeurs_predites'],
        marker='s',
        linestyle='--',
        linewidth=2,
        markersize=8,
        color='coral',
        label='Prédictions'
    )
    
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Nombre de crimes', fontsize=12)
    ax.set_title(f'Historique et prédictions - {region}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
