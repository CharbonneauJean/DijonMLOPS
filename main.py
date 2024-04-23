import mlflow
from preprocess_data import preprocess_data
from logs import logFileName
from data_read_and_split import data_read_and_split
from train import train_xgboost, train_random_forest, train_svr
import pandas as pd

if __name__ == "__main__":
    mlflow.set_experiment("salary_prediction")

    # On commence par terminer un run actif s'il y en a un
    activeRun = mlflow.active_run()
    if activeRun:
        mlflow.end_run(activeRun.info.status)

    datasetFileStr = 'data/ds_salaries.csv'

    with mlflow.start_run(run_name="data_read_and_split"):
        mlflow.log_text("Début de la tâche de récupération du dataset", logFileName)
        df = data_read_and_split(datasetFileStr)

    with mlflow.start_run(run_name="data_preprocessing"):
        X_train, X_test, y_train, y_test = preprocess_data(df)
    
    with mlflow.start_run(run_name="xgboost"):
        best_model_xgb, rmse, mae, r2, gridSearchParams = train_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_params(gridSearchParams)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(best_model_xgb, "model")
        
    with mlflow.start_run(run_name="random_forest"):
        best_model_rf, rmse, mae, r2, gridSearchParams = train_random_forest(X_train, y_train, X_test, y_test)
        mlflow.log_params(gridSearchParams)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model_rf, "model")
        
    with mlflow.start_run(run_name="svr"):
        best_model_svr, rmse, mae, r2, gridSearchParams = train_svr(X_train, y_train, X_test, y_test)
        mlflow.log_params(gridSearchParams)
        mlflow.log_metric("rmse", rmse) 
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model_svr, "model")
