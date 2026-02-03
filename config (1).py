"""
Configuration de l'application
Centralise les chemins de fichiers et les constantes
"""
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Fichiers de données
EXCEL_FILE = DATA_DIR / "data-gouv-series-chrono.xlsx"
SHAPEFILE_PATH = DATA_DIR / "regions-20180101-shp" / "regions-20180101.shp"

# Régions à exclure (territoires d'outre-mer)
REGIONS_TO_EXCLUDE = [
    'Martinique', 
    'Guyane', 
    'Mayotte', 
    'Guadeloupe', 
    'La Réunion'
]

# Configuration des classes de criminalité
CRIME_CLASSES = {
    'Classe 1': {'label': 'Low', 'color': 'green'},
    'Classe 2': {'label': 'Medium', 'color': 'yellow'},
    'Classe 3': {'label': 'High', 'color': 'red'}
}

# Configuration des graphiques
FIGURE_SIZE = (12, 8)
DEFAULT_YEAR = '2021'
PREDICTION_YEARS = range(2023, 2029)
