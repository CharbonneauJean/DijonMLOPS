import mlflow
from preprocess_data import preprocess_data
from logs import logFileName
from data_read_and_split import data_read_and_split
from train import train_xgboost, train_random_forest, train_svr
import pandas as pd
from logs import currentTimestamp

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
        best_model_xgb, rmse, mae, r2, bestModelParams = train_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_params(bestModelParams)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(best_model_xgb, "xgboost-model")
        logged_model = 'runs:/{}/xgboost-model'.format(mlflow.active_run().info.run_id)
        mlflow.register_model(logged_model,"xgboost-" + currentTimestamp.strftime("%Y-%m-%d-%H-%M-%S"))
        
    with mlflow.start_run(run_name="random_forest"):
        best_model_rf, rmse, mae, r2, bestModelParams = train_random_forest(X_train, y_train, X_test, y_test)
        mlflow.log_params(bestModelParams)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model_rf, "randomforestregressor-model")
        logged_model = 'runs:/{}/randomforestregressor-model'.format(mlflow.active_run().info.run_id)
        mlflow.register_model(logged_model,"randomforestregressor-" + currentTimestamp.strftime("%Y-%m-%d-%H-%M-%S"))
        
    with mlflow.start_run(run_name="svr"):
        best_model_svr, rmse, mae, r2, bestModelParams = train_svr(X_train, y_train, X_test, y_test)
        mlflow.log_params(bestModelParams)
        mlflow.log_metric("rmse", rmse) 
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model_svr, "svr-model")
        logged_model = 'runs:/{}/svr-model'.format(mlflow.active_run().info.run_id)
        mlflow.register_model(logged_model,"svr-" + currentTimestamp.strftime("%Y-%m-%d-%H-%M-%S"))
