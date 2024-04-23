
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

def train_xgboost(X_train, y_train, X_test, y_test):
    param_grid_xgb = {
        'max_depth': [2,3,4,5,6,7],
        'learning_rate': [0.05,0.06,0.07,0.08,0.09],
        'n_estimators': [200,300,400,500],
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

    return best_model_xgb, rmse, mae, r2

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid_rf = { 
        'n_estimators': [90,100,110],
        'max_features': [35,40,45],
        'max_depth' : [13,14,15],
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
    
    return best_model_rf, rmse, mae, r2

def train_svr(X_train, y_train, X_test, y_test):
    param_grid_svr = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.010,0.011,0.012],
        'epsilon': [2.49,2.5,2.51]
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

    return best_model_svr, rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv('data/ds_salaries.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    mlflow.set_experiment("salary_prediction")
    
    with mlflow.start_run(run_name="xgboost"):
        best_model_xgb, rmse, mae, r2 = train_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(best_model_xgb, "model")
        
    with mlflow.start_run(run_name="random_forest"):
        best_model_rf, rmse, mae, r2 = train_random_forest(X_train, y_train, X_test, y_test) 
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model_rf, "model")
        
    with mlflow.start_run(run_name="svr"):
        best_model_svr, rmse, mae, r2 = train_svr(X_train, y_train, X_test, y_test)
        mlflow.log_metric("rmse", rmse) 
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model_svr, "model")
