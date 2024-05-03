# DijonMLOPS

Lignes de commandes utilisées :

```bash
mlflow ui # pour lancer l'interface web de mlflow
```

```bash 
mlflow models serve -m "models:/RandomForestRegressor/4" --env-manager local --port 7000 # pour "servir" le modèle
```

```bash	
streamlit run dashboard4.py # pour lancer l'application web
```

Lorsqu'un modèle est enregistrer dans MLFlow, il suffit de le "promote" en lui donnant un nom. Si ce nom existe déjà, la promotion augmentera sa version. Ici l'exemple donné permet de "servir" le modèle Random Forest Regressor en version 4.
