import streamlit as st
import pandas as pd

# Titre de l'application
st.title("Dashboard Salariés")

# Affichage de texte
st.write("Etudes sur les différences de salaires en IA")

# Chargement de données 
data = pd.read_csv("data/ds_salaries.csv")

# Affichage du dataframe
st.write(data)

# Affichage d'un graphique
st.line_chart(data, x='job_title', y='salary_in_usd')
