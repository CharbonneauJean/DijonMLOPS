import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
import mlflow

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
    
def preprocess_data(df):
    # create a new column "salary_eur" that contains the salary in EUR
    df.loc[:, 'salary_eur'] = df['salary'].copy()
    
    # Votre code de prétraitement des données ici
    df = df[df['salary_currency'].isin(['USD', 'EUR', 'GBP'])]

    # set type of df['salary_eur'] to float
    df.loc[:, 'salary_eur'] = df['salary_eur'].astype(float)

    df.loc[df['salary_currency'] == 'USD', 'salary_eur'] = df.loc[df['salary_currency'] == 'USD', 'salary'] * usd_to_eur
    df.loc[df['salary_currency'] == 'GBP', 'salary_eur'] = df.loc[df['salary_currency'] == 'GBP', 'salary'] * gbp_to_eur
    df.loc[df['salary_currency'] == 'INR', 'salary_eur'] = df.loc[df['salary_currency'] == 'INR', 'salary'] * inr_to_eur
    df.loc[df['salary_currency'] == 'CAD', 'salary_eur'] = df.loc[df['salary_currency'] == 'CAD', 'salary'] * cad_to_eur
    df.loc[df['salary_currency'] == 'AUD', 'salary_eur'] = df.loc[df['salary_currency'] == 'AUD', 'salary'] * aud_to_eur
    df.loc[df['salary_currency'] == 'SGD', 'salary_eur'] = df.loc[df['salary_currency'] == 'SGD', 'salary'] * sgd_to_eur
    df.loc[df['salary_currency'] == 'BRL', 'salary_eur'] = df.loc[df['salary_currency'] == 'BRL', 'salary'] * brl_to_eur
    df.loc[df['salary_currency'] == 'PLN', 'salary_eur'] = df.loc[df['salary_currency'] == 'PLN', 'salary'] * pln_to_eur
    df.loc[df['salary_currency'] == 'CHF', 'salary_eur'] = df.loc[df['salary_currency'] == 'CHF', 'salary'] * chf_to_eur
    df.loc[df['salary_currency'] == 'HUF', 'salary_eur'] = df.loc[df['salary_currency'] == 'HUF', 'salary'] * huf_to_eur
    df.loc[df['salary_currency'] == 'DKK', 'salary_eur'] = df.loc[df['salary_currency'] == 'DKK', 'salary'] * dkk_to_eur
    df.loc[df['salary_currency'] == 'JPY', 'salary_eur'] = df.loc[df['salary_currency'] == 'JPY', 'salary'] * jpy_to_eur
    df.loc[df['salary_currency'] == 'TRY', 'salary_eur'] = df.loc[df['salary_currency'] == 'TRY', 'salary'] * try_to_eur
    df.loc[df['salary_currency'] == 'THB', 'salary_eur'] = df.loc[df['salary_currency'] == 'THB', 'salary'] * thb_to_eur
    df.loc[df['salary_currency'] == 'ILS', 'salary_eur'] = df.loc[df['salary_currency'] == 'ILS', 'salary'] * ils_to_eur
    df.loc[df['salary_currency'] == 'HKD', 'salary_eur'] = df.loc[df['salary_currency'] == 'HKD', 'salary'] * hkd_to_eur
    df.loc[df['salary_currency'] == 'CZK', 'salary_eur'] = df.loc[df['salary_currency'] == 'CZK', 'salary'] * czk_to_eur
    df.loc[df['salary_currency'] == 'MXN', 'salary_eur'] = df.loc[df['salary_currency'] == 'MXN', 'salary'] * mxn_to_eur
    df.loc[df['salary_currency'] == 'CLP', 'salary_eur'] = df.loc[df['salary_currency'] == 'CLP', 'salary'] * clp_to_eur

    df = df.drop(columns=['salary', 'salary_currency', 'salary_in_usd'])

    X = df.drop('salary_eur', axis=1)
    y = df['salary_eur']
  
    cat_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
    ohe = OneHotEncoder(handle_unknown='ignore')
    ct = ColumnTransformer(transformers=[('encoder', ohe, cat_cols)], remainder='passthrough')
    X = ct.fit_transform(X)

    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
  
    return train_test_split(X, y, test_size=0.2, random_state=42)

