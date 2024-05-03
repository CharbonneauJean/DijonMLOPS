import streamlit as st
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

usd_to_eur = 0.94  # 1 USD = 0.94 EUR
gbp_to_eur = 1.14  # 1 GBP = 1.14 EUR
inr_to_eur = 0.011 # 1 INR = 0.011 EUR
cad_to_eur = 0.70  # 1 CAD = 0.70 EUR
aud_to_eur = 0.64  # 1 AUD = 0.64 EUR
sgd_to_eur = 0.71  # 1 SGD = 0.71 EUR [[10]]
brl_to_eur = 0.18  # 1 BRL = 0.18 EUR [[1]]
pln_to_eur = 0.22  # 1 PLN = 0.22 EUR [[2]] 
chf_to_eur = 1.01  # 1 CHF = 1.01 EUR
huf_to_eur = 0.0027 # 1 HUF = 0.0027 EUR [[2]]
dkk_to_eur = 0.13  # 1 DKK = 0.13 EUR [[2]]
jpy_to_eur = 0.0073 # 1 JPY = 0.0073 EUR [[2]]
try_to_eur = 0.051 # 1 TRY = 0.051 EUR [[2]]
thb_to_eur = 0.027 # 1 THB = 0.027 EUR 
ils_to_eur = 0.26  # 1 ILS = 0.26 EUR
hkd_to_eur = 0.12  # 1 HKD = 0.12 EUR
czk_to_eur = 0.041 # 1 CZK = 0.041 EUR [[2]]
mxn_to_eur = 0.052 # 1 MXN = 0.052 EUR
clp_to_eur = 0.0011 # 1 CLP = 0.0011 EUR

def predict_salary(dt):
    url = 'http://localhost:7000/invocations'
    headers = {'Content-Type': 'application/json'}
    
    # Convertir les données en format compatible avec MLflow
    data_dict = {
        'dataframe_split': {
            'columns': ct.get_feature_names_out().tolist(),
            'data': dt.toarray().tolist()  # Convertir en liste de listes
        }
    }
    
    response = requests.post(url, json=data_dict, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
        raise Exception(f'Request failed with status {response.status_code}')


    

# Chargement des données
data = pd.read_csv("data/ds_salaries_train.csv")

# create a new column "salary_eur" that contains the salary in EUR
data.loc[:, 'salary_eur'] = data['salary'].copy()

# Votre code de prétraitement des données ici
data = data[data['salary_currency'].isin(['USD', 'EUR', 'GBP'])]

# set type of data['salary_eur'] to float
data.loc[:, 'salary_eur'] = data['salary_eur'].astype(float)

# Conversion des salaires en EUR
data.loc[data['salary_currency'] == 'USD', 'salary_eur'] = data.loc[data['salary_currency'] == 'USD', 'salary'] * usd_to_eur
data.loc[data['salary_currency'] == 'GBP', 'salary_eur'] = data.loc[data['salary_currency'] == 'GBP', 'salary'] * gbp_to_eur
data.loc[data['salary_currency'] == 'INR', 'salary_eur'] = data.loc[data['salary_currency'] == 'INR', 'salary'] * inr_to_eur
data.loc[data['salary_currency'] == 'CAD', 'salary_eur'] = data.loc[data['salary_currency'] == 'CAD', 'salary'] * cad_to_eur
data.loc[data['salary_currency'] == 'AUD', 'salary_eur'] = data.loc[data['salary_currency'] == 'AUD', 'salary'] * aud_to_eur
data.loc[data['salary_currency'] == 'SGD', 'salary_eur'] = data.loc[data['salary_currency'] == 'SGD', 'salary'] * sgd_to_eur
data.loc[data['salary_currency'] == 'BRL', 'salary_eur'] = data.loc[data['salary_currency'] == 'BRL', 'salary'] * brl_to_eur
data.loc[data['salary_currency'] == 'PLN', 'salary_eur'] = data.loc[data['salary_currency'] == 'PLN', 'salary'] * pln_to_eur
data.loc[data['salary_currency'] == 'CHF', 'salary_eur'] = data.loc[data['salary_currency'] == 'CHF', 'salary'] * chf_to_eur
data.loc[data['salary_currency'] == 'HUF', 'salary_eur'] = data.loc[data['salary_currency'] == 'HUF', 'salary'] * huf_to_eur
data.loc[data['salary_currency'] == 'DKK', 'salary_eur'] = data.loc[data['salary_currency'] == 'DKK', 'salary'] * dkk_to_eur
data.loc[data['salary_currency'] == 'JPY', 'salary_eur'] = data.loc[data['salary_currency'] == 'JPY', 'salary'] * jpy_to_eur
data.loc[data['salary_currency'] == 'TRY', 'salary_eur'] = data.loc[data['salary_currency'] == 'TRY', 'salary'] * try_to_eur
data.loc[data['salary_currency'] == 'THB', 'salary_eur'] = data.loc[data['salary_currency'] == 'THB', 'salary'] * thb_to_eur
data.loc[data['salary_currency'] == 'ILS', 'salary_eur'] = data.loc[data['salary_currency'] == 'ILS', 'salary'] * ils_to_eur
data.loc[data['salary_currency'] == 'HKD', 'salary_eur'] = data.loc[data['salary_currency'] == 'HKD', 'salary'] * hkd_to_eur
data.loc[data['salary_currency'] == 'CZK', 'salary_eur'] = data.loc[data['salary_currency'] == 'CZK', 'salary'] * czk_to_eur
data.loc[data['salary_currency'] == 'MXN', 'salary_eur'] = data.loc[data['salary_currency'] == 'MXN', 'salary'] * mxn_to_eur
data.loc[data['salary_currency'] == 'CLP', 'salary_eur'] = data.loc[data['salary_currency'] == 'CLP', 'salary'] * clp_to_eur

X = data.drop(['salary_eur','salary', 'salary_currency', 'salary_in_usd'], axis=1)

cat_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
ohe = OneHotEncoder(handle_unknown='ignore')
ct = ColumnTransformer(transformers=[('encoder', ohe, cat_cols)], remainder='passthrough')
X = ct.fit_transform(X)

scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

df = pd.DataFrame(X)

# Sélection d'une ligne du dataframe
selected_row = st.selectbox("Sélectionnez une ligne :", options=df.index)

# Récupération du salaire en EUR pour le selected_row
salary = data.loc[selected_row, 'salary_eur']
st.write("Salaire en EUR réel :", salary)

if st.button("Prédire le salaire"):
    # Récupération de la ligne sélectionnée
    input_data = data.loc[selected_row]
    
    # Conversion des données en DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Transformation des données avec ColumnTransformer
    input_transformed = ct.transform(input_df)
    
    # Mise à l'échelle des données avec StandardScaler
    input_scaled = scaler.transform(input_transformed)
    
    # Appel de la fonction de prédiction
    prediction = predict_salary(input_scaled)
    
    # Affichage de la prédiction
    st.write("Salaire prédit :", prediction)


