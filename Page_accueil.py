import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys

import geopandas as gpd
import cartopy.crs as ccrs
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import io


# Ajouter un onglet pour le deuxième code Python
option = st.sidebar.selectbox("Pages", ("Page d'accueil", "Plus de données"))
# Titre du menu
st.sidebar.title("Page d'accueil")
st.sidebar.title("Informations personnelles")

# Ajouter des champs pour saisir les informations
name = st.sidebar.text_input("Nom", "Soumano")
firstname = st.sidebar.text_input("Prénom", "Seydou")
linkedin = st.sidebar.text_input("LinkedIn", "https://www.linkedin.com/in/seydou-soumano/")
github = st.sidebar.text_input("GitHub", "https://github.com/Ssoumano")
if option == "Plus de données":
    exec(open('Page_2.py').read())
    sys.exit()

elif option == "Page d'accueil":
    # Afficher les informations personnelles
    x=0

# Ajouter les logos
st.sidebar.image("LOGO_EFREI-PRINT_EFREI-WEB.png", use_column_width=True)
st.sidebar.image("téléchargement.png", use_column_width=True)

title = "<h1 style='text-align: center; color: blue; text-decoration: underline;'>Visualization of the number of crimes in France</h1>"
st.markdown(title, unsafe_allow_html=True)
st.write("Here we have a dataset of crimes committed in France in the past years. This dataset has been downloaded from Data.gouv and will help us analyzing the level of danger in different regions, identify safer regions, and make predictions for the coming years.")
st.title("(Overview of the used database)")
st.markdown("<p style='text-align: center; font-size: small;'> </p>", unsafe_allow_html=True)

# Téléchargement des données
file_path = "data-gouv-series-chrono.xlsx"
# conversion en Dataframe
base1 = pd.read_excel(file_path)

# Création de nouvelles colonnes
def extract_nom_zone(row):
    if '-' in row['Zone_geographique']:
        return row['Zone_geographique'].split('-', 1)[1].split('(')[0].strip()
    else:
        return row['Zone_geographique'].split('(')[0].strip()

base1['nom_zone'] = base1.apply(extract_nom_zone, axis=1)

base1['type_zone'] = base1['Zone_geographique'].str.extract(r'\((.*?)\)')

# Creation de la base2 pour l'analyse des données nécéssaires
base2 = base1[base1['Statistique'] == 'Nombre']
st.dataframe(base2.head())


# calcul du nombre de crimes
filtered_data = base2[base2['type_zone'] == 'région']
grouped_data = filtered_data.groupby(['Unite temps', 'nom_zone'])['Valeurs'].sum().reset_index()
selected_year = st.selectbox("Select a year", grouped_data['Unite temps'].unique())
data_year = grouped_data[grouped_data['Unite temps'] == selected_year]
sorted_data = data_year.sort_values(by='Valeurs', ascending=False)

# création du Camenbert
top_4 = sorted_data.head(4)
other = pd.DataFrame({
    'nom_zone': ['Autre'],
     'Valeurs': [sorted_data.iloc[4:, sorted_data.columns.get_loc('Valeurs')].sum()]
})
combined_data = pd.concat([top_4, other])

# Creation du pie chart
labels = combined_data['nom_zone'].tolist()
values = combined_data['Valeurs'].tolist()

fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title(f"Crimes distribution in {selected_year}")
st.pyplot(fig)

# création du scatter plot
filtered_data = base2[base2['type_zone'] == 'région']
regions = filtered_data['nom_zone'].unique()
region = st.selectbox("Select a region", regions, key='region_dropdown')
filtered_data = filtered_data[filtered_data['nom_zone'] == region]
grouped_data = filtered_data.groupby(['Unite temps', 'nom_zone'])['Valeurs'].sum()
x = grouped_data.index.get_level_values('Unite temps')
y = grouped_data.values
fig, ax = plt.subplots()
ax.plot(x, y, marker='o', linestyle='-', label='Evolution of the number of crimes')
ax.scatter(x, y)
ax.set_xlabel('Unite temps')
ax.set_ylabel('Sum of Valeurs')
ax.set_title('Evolution of the number of crimes')

for i, j in zip(x, y):
     ax.annotate(str(j), xy=(i, j), xytext=(5, -10), textcoords='offset points')
st.pyplot(fig)
st.sidebar.markdown("<h1 style='text-align: right; text-transform: lowercase;'>Your text here</h1>", unsafe_allow_html=True)

# Accès au fichier
file_path = "data-gouv-series-chrono.xlsx"

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
