import pandas as pd
import mlflow
from logs import logFileName

def data_read_and_split(fileStr):
    df = pd.read_csv(fileStr)

    mlflow.log_text("Imported dataset", logFileName)

    try:
        df_added = pd.read_csv('data/ds_salaries_added.csv')
        df_train = pd.read_csv('data/ds_salaries_train.csv')
    except:
        # Split the df dataframe into two dataframes : df_train and df_added (1000 lines in df_added, the rest in df_train. Chose lines randomly)
        df_added = df.sample(1000)
        df_train = df.drop(df_added.index)

        df_added.to_csv('data/ds_salaries_added.csv', index=False)
        df_train.to_csv('data/ds_salaries_train.csv', index=False)
        
        with mlflow.start_run(run_name="data_preprocessing"):
            mlflow.log_artifact("data/ds_salaries.csv", artifact_path="data/ds_salaries.csv")
            mlflow.log_artifact("data/ds_salaries_train.csv", artifact_path="data/ds_salaries_train.csv")
            mlflow.log_artifact("data/ds_salaries_added.csv", artifact_path="data/ds_salaries_added.csv")

    return df_train
