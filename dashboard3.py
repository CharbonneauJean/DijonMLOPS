import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("data/ds_salaries.csv")

data = load_data()

# Titre et introduction
st.title("Dashboard Salariés")
st.write("Ce dashboard interactif permet d'explorer les données sur les données de salariés.")

# Création des onglets
tab1, tab2 = st.tabs(["Dataframe", "Graphiques"])

with tab1:
    # Sélection des colonnes à afficher
    cols = ["work_year", "experience_level", "employment_type","job_title","salary",  "salary_currency", "salary_in_usd", "employee_residence", "remote_ratio", "company_location", "company_size"]
    selected_cols = st.multiselect("Sélectionnez les colonnes à afficher :", cols, default=cols)

    # Affichage du dataframe filtré
    st.dataframe(data[selected_cols], use_container_width=True)

with tab2:
    # Sélection d'une variable pour les graphiques  
    cols_for_graphs = ["work_year", "experience_level", "employment_type", "remote_ratio", "company_size"]
    selected_var = st.selectbox("Sélectionnez une variable pour les graphiques :", cols_for_graphs)

    # Graphiques dans des colonnes
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 8), tight_layout=True)
        data[selected_var].value_counts().plot(kind="bar", ax=ax1) 
        ax1.set_title(f"Distribution de {selected_var}", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 8), tight_layout=True)
        labels = data[selected_var].value_counts().index
        sizes = data[selected_var].value_counts().values  
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 10})
        ax2.set_title(f"Proportion de {selected_var}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
