import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import numpy as np

# Menu latéral pour sélectionner les pages
option = st.sidebar.selectbox("Pages", ("Page d'accueil", "Plus de données"))

# Titre du menu
st.sidebar.title("Page d'accueil")
st.sidebar.title("Informations personnelles")

# Champs pour saisir les informations personnelles
name = st.sidebar.text_input("Nom", "Soumano")
firstname = st.sidebar.text_input("Prénom", "Seydou")
linkedin = st.sidebar.text_input("LinkedIn", "https://www.linkedin.com/in/seydou-soumano/")
github = st.sidebar.text_input("GitHub", "https://github.com/Ssoumano")

# Redirection vers une autre page si "Plus de données" est sélectionné
if option == "Plus de données":
    exec(open('Page_2.py').read())
    sys.exit()
elif option == "Page d'accueil":
    # Placeholder si "Page d'accueil" est sélectionnée
    x = 0

# Ajout des logos
st.sidebar.image("LOGO_EFREI-PRINT_EFREI-WEB.png", use_column_width=True)
st.sidebar.image("téléchargement.png", use_column_width=True)

# Titre principal
title = "<h1 style='text-align: center; color: blue; text-decoration: underline;'>Visualization of the number of crimes in France</h1>"
st.markdown(title, unsafe_allow_html=True)
st.write("Here we have a dataset of crimes committed in France in the past years. This dataset has been downloaded from Data.gouv and will help us analyzing the level of danger in different regions, identify safer regions, and make predictions for the coming years.")
st.title("(Overview of the used database)")

# Téléchargement des données
file_path = "data-gouv-series-chrono.xlsx"
base1 = pd.read_excel(file_path) 

# Création de nouvelles colonnes pour extraire les noms et types de zones
def extract_nom_zone(row):
    if '-' in row['Zone_geographique']:
        return row['Zone_geographique'].split('-', 1)[1].split('(')[0].strip()
    else:
        return row['Zone_geographique'].split('(')[0].strip()

base1['nom_zone'] = base1.apply(extract_nom_zone, axis=1)
base1['type_zone'] = base1['Zone_geographique'].str.extract(r'\((.*?)\)')

# Filtrage des données pour ne garder que les statistiques de type "Nombre"
base2 = base1[base1['Statistique'] == 'Nombre']
st.dataframe(base2.head())

# Calcul du nombre de crimes par région et par année
filtered_data = base2[base2['type_zone'] == 'région']
grouped_data = filtered_data.groupby(['Unite temps', 'nom_zone'])['Valeurs'].sum().reset_index()

# Sélection de l'année pour afficher les données
selected_year = st.selectbox("Select a year", grouped_data['Unite temps'].unique())
data_year = grouped_data[grouped_data['Unite temps'] == selected_year]
sorted_data = data_year.sort_values(by='Valeurs', ascending=False)

# Création d'un camembert pour les 4 régions avec le plus de crimes
top_4 = sorted_data.head(4)
other = pd.DataFrame({
    'nom_zone': ['Autre'],
    'Valeurs': [sorted_data.iloc[4:, sorted_data.columns.get_loc('Valeurs')].sum()]
})
combined_data = pd.concat([top_4, other])

# Création du pie chart
labels = combined_data['nom_zone'].tolist()
values = combined_data['Valeurs'].tolist()

fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%')
ax.axis('equal')
plt.title(f"Crimes distribution in {selected_year}")
st.pyplot(fig)

# Création d'un scatter plot pour l'évolution du nombre de crimes par région
region = st.selectbox("Select a region", filtered_data['nom_zone'].unique(), key='region_dropdown')
region_data = filtered_data[filtered_data['nom_zone'] == region]
grouped_region_data = region_data.groupby(['Unite temps', 'nom_zone'])['Valeurs'].sum()
x = grouped_region_data.index.get_level_values('Unite temps')
y = grouped_region_data.values

fig, ax = plt.subplots()
ax.plot(x, y, marker='o', linestyle='-', label='Evolution of the number of crimes')
ax.scatter(x, y)
ax.set_xlabel('Unite temps')
ax.set_ylabel('Sum of Valeurs')
ax.set_title('Evolution of the number of crimes')

for i, j in zip(x, y):
    ax.annotate(str(j), xy=(i, j), xytext=(5, -10), textcoords='offset points')
st.pyplot(fig)

# Utilisation d'une API pour obtenir les données géographiques des régions françaises
gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Vérifiez les colonnes disponibles dans gdf
st.write(gdf.columns)

# Filtrer pour obtenir uniquement la France
# Utilisez la colonne appropriée, par exemple 'name' ou 'admin'
if 'name' in gdf.columns:
    france = gdf[gdf['name'] == 'France']
elif 'admin' in gdf.columns:
    france = gdf[gdf['admin'] == 'France']
else:
    st.error("La colonne pour filtrer les données françaises est introuvable dans le dataset géographique.")

# Classement des régions par nombre de crimes pour l'année 2021
wx2021 = grouped_data[grouped_data['Unite temps'] == '2021']
wx2021 = wx2021.sort_values(by='Valeurs', ascending=False)
wx2021['Classe'] = pd.qcut(wx2021['Valeurs'], q=3, labels=['Low crimes number', 'Medium crimes number', 'High crimes number'])

# Assurez-vous que les données géographiques et les données de criminalité peuvent être fusionnées correctement
# Si la colonne 'name' ne correspond pas aux noms des régions, ajustez en conséquence
france_regions = france.copy()
france_regions['nom'] = france_regions['name']  # Assumons que 'name' est la colonne avec les noms des régions

# Fusionner les données
gdf_merged = france_regions.set_index('nom').join(wx2021.set_index('nom_zone'), how='inner')

# Vérification et correction des géométries
gdf_merged = gdf_merged[~gdf_merged.is_empty & gdf_merged.is_valid]

# Création de la carte
fig, ax = plt.subplots(figsize=(12, 8))
colors = {'Low crimes number': 'green', 'Medium crimes number': 'yellow', 'High crimes number': 'red'}
gdf_merged['color'] = gdf_merged['Classe'].map(colors)
gdf_merged.plot(ax=ax, edgecolor='black', linewidth=0.5, facecolor=gdf_merged['color'], legend=True)

# Ajout des noms des régions sur la carte
for _, row in gdf_merged.iterrows():
    x, y = row['geometry'].centroid.x, row['geometry'].centroid.y
    ax.text(x, y, row['name'], fontsize=8, ha='center', va='center')

# Ajout de la légende
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='green', label='Low'),
    plt.Rectangle((0, 0), 1, 1, fc='yellow', label='Medium'),
    plt.Rectangle((0, 0), 1, 1, fc='red', label='High')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
st.pyplot(fig)

# Prédiction des classes pour les années futures
years = range(2023, 2029)
selected_year = st.selectbox("Select a year to predict", years)
df_train = wx2021[wx2021['Unite temps'] != str(selected_year)]
df_train['Unite temps'] = pd.to_numeric(df_train['Unite temps'])

predicted_classes = []
for region in df_train['nom_zone'].unique():
    df_region = df_train[df_train['nom_zone'] == region]
    X_train = df_region['Unite temps'].values.reshape(-1, 1)
    y_train = df_region['Classe']
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    predicted_class = regression_model.predict([[selected_year]])
    predicted_classes.append(predicted_class[0])

df_predict = pd.DataFrame({'nom_zone': df_train['nom_zone'].unique(), 'Predicted Classe': predicted_classes})
st.dataframe(df_predict)
