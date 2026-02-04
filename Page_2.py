import geopandas as gpd
import cartopy.crs as ccrs
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

st.sidebar.markdown("<h1 style='text-align: right; text-transform: lowercase;'>Your text here</h1>", unsafe_allow_html=True)

# Chemin vers le fichier shapefile des régions
shapefile_path = '/Users/seydou/Downloads/regions-20180101-shp/regions-20180101.shp'

# Charger le shapefile dans un GeoDataFrame
gdf = gpd.read_file(shapefile_path)

# Accès au fichier
file_path = "/Users/seydou/Downloads/process_2/data-gouv-series-chrono.xlsx"

# Charger le fichier Excel
base1 = pd.read_excel(file_path)
st.title("(Which regions have the highest crime rates?)")

# Création de nouvelles colonnes
def extract_nom_zone(row):
    if '-' in row['Zone_geographique']:
        return row['Zone_geographique'].split('-', 1)[1].split('(')[0].strip()
    else:
        return row['Zone_geographique'].split('(')[0].strip()

base1['nom_zone'] = base1.apply(extract_nom_zone, axis=1)
base1['type_zone'] = base1['Zone_geographique'].str.extract(r'\((.*?)\)')

# Création base 2
base2 = base1[base1['Statistique'] == 'Nombre']
filtered_data = base2[base2['type_zone'] == 'région']

# Création de la représentation du nombre de crimes sur la carte
grouped_data = filtered_data.groupby(['Unite temps', 'nom_zone'])['Valeurs'].sum().reset_index()
wx = pd.DataFrame(grouped_data)

def create_classes(x):
    x['classe'] = pd.qcut(x['Valeurs'], q=3, labels=['Classe 1', 'Classe 2', 'Classe 3'])
    return x

df = wx.groupby('Unite temps').apply(create_classes)

wx2021 = wx[wx['Unite temps'] == '2021']
wx2021 = wx2021.sort_values(by='Valeurs', ascending=False)
wx2021['Classe'] = pd.qcut(wx2021['Valeurs'], q=3, labels=['Low crimes number', 'Medium crimes number', 'High crimes number'])

# Filtrer les régions à exclure
regions_to_exclude = ['Martinique', 'Guyane', 'Mayotte', 'Guadeloupe', 'La Réunion']
gdf_filtered = gdf[~gdf['nom'].isin(regions_to_exclude)].reset_index(drop=True)

# Fusionner le GeoDataFrame filtré avec les classes
gdf_merged = gdf_filtered.merge(df, left_on='nom', right_on='nom_zone')

# Création de la carte
fig, ax = plt.subplots(figsize=(12, 8))

# Définir les couleurs pour chaque classe
colors = {'Classe 1': 'green', 'Classe 2': 'yellow', 'Classe 3': 'red'}

# Mapper les couleurs aux classes
gdf_merged['color'] = gdf_merged['classe'].map(colors)

# Dessiner les régions avec les couleurs correspondantes
gdf_merged.plot(ax=ax, edgecolor='black', linewidth=0.5, facecolor=gdf_merged['color'], legend=True)

# Ajout des noms des régions
for _, row in gdf_merged.iterrows():
    region = row['nom']
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, region, fontsize=8, ha='center', va='center')

# Ajout de la légende
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='green', label='Low'),
    plt.Rectangle((0, 0), 1, 1, fc='yellow', label='Medium'),
    plt.Rectangle((0, 0), 1, 1, fc='red', label='High')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

# Affichage de la carte
st.pyplot(fig)

st.title("(Ranking of French regions by number of crimes in 2021)")
st.markdown("<p style='text-align: center; font-size: small;'> </p>", unsafe_allow_html=True)

# Classement des régions par nombre de crimes
filtered_data = base2[base2['type_zone'] == 'région']
grouped_data = filtered_data.groupby(['Unite temps', 'nom_zone'])['Valeurs'].sum().reset_index()
grouped_data_df = grouped_data

wx = pd.DataFrame(grouped_data)

def create_classes(x):
    x['classe'] = pd.qcut(x['Valeurs'], q=3, labels=['1', '2', '3'])
    return x

df = grouped_data.groupby('Unite temps').apply(create_classes)

years = range(2023, 2029)

# Menu pour la prédiction des classes
selected_year = st.selectbox("Select a year to predict", years)

# Filtrer sur les données des années précédentes
df_train = df[df['Unite temps'] != str(selected_year)]
df_train['Unite temps'] = pd.to_numeric(df_train['Unite temps'])

predicted_classes = []
for region in df['nom_zone'].unique():
    df_region = df_train[df_train['nom_zone'] == region]
    X_train = df_region['Unite temps'].values.reshape(-1, 1)
    y_train = df_region['classe']
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    predicted_class = regression_model.predict([[selected_year]])
    predicted_classes.append(predicted_class[0])

df_predict = pd.DataFrame({'nom_zone': df['nom_zone'].unique(), 'Predicted Classe': predicted_classes})

st.dataframe(df_predict)
