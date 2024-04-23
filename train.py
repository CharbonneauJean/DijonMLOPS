
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from preprocess_data import preprocess_data

from logs import currentTimestamp

def train_xgboost(X_train, y_train, X_test, y_test):
    param_grid_xgb = {
        'max_depth': [10,15],
        'learning_rate': [0.05,1,10],
        'n_estimators': [300,400],
    }

    grid_search_xgb = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror'), 
                           param_grid=param_grid_xgb,
                           cv=5,  # 5-fold cross-validation
                           n_jobs=-1,  # utiliser tous les CPU
                           verbose=2)  # afficher les logs
    
    grid_search_xgb.fit(X_train, y_train)

    best_model_xgb = grid_search_xgb.best_estimator_
    y_pred = best_model_xgb.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model_xgb, rmse, mae, r2, grid_search_xgb.best_params_

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid_rf = { 
        'n_estimators': [100,200,400],
        'max_features': [20,40,80],
        'max_depth' : [10,14,20],
        'criterion' :['squared_error']
    }

    rf = RandomForestRegressor()

    grid_search_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_rf,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search_rf.fit(X_train, y_train)

    best_model_rf = grid_search_rf.best_estimator_

    y_pred = best_model_rf.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return best_model_rf, rmse, mae, r2, grid_search_rf.best_params_

def train_svr(X_train, y_train, X_test, y_test):
    param_grid_svr = {
        'kernel': ['linear'],
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [1, 2, 2.5, 5]
    }

    svr = SVR()

    grid_search_svr = GridSearchCV(
        estimator=svr,
        param_grid=param_grid_svr,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search_svr.fit(X_train, y_train)

    best_model_svr = grid_search_svr.best_estimator_

    y_pred = best_model_svr.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model_svr, rmse, mae, r2, grid_search_svr.best_params_

