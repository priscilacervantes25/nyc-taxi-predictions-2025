import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pathlib
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from mlflow.tracking import MlflowClient
from prefect import flow, task


# Carga y preparación de los datos 
# Conversión de fechas a sus formatos correspondientes, filtra viajes 1-60min, conversión de ids a strings
@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


# Se crean variables y transformación de datos a formato numérico
# Combina pickup y dropoff -> PU_DO
# Separación de var categóricas y numéricas
@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


# Optimización de hiperparámetros con Optuna y mlfow
# Definición de función obj. Optuna
# Registro de resultados de c/trial
# Se regresan los mejores hiperparámetros encontrados.
@task(name="Hyperparameter Tunning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv):
    
    mlflow.xgboost.autolog()
    
    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")
    
    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")
    
    train = xgb.DMatrix(X_train, label=y_train)
    
    valid = xgb.DMatrix(X_val, label=y_val)
    
    def objective(trial: optuna.trial.Trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-3), 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha",   math.exp(-5), math.exp(-1), log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", math.exp(-6), math.exp(-1), log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", math.exp(-1), math.exp(3), log=True),
            "objective": "reg:squarederror",  
            "seed": 42,                      
        }

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "xgboost")  
            mlflow.log_params(params)                  

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=10,
            )

            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)

            mlflow.log_metric("rmse", rmse)

            signature = infer_signature(X_val, y_pred)

            mlflow.xgboost.log_model(
                booster,
                artifact_path="model",
                input_example=X_val[:5],
                signature=signature
            )

        return rmse

    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    
    with mlflow.start_run(run_name="XGBoost Hyperparameter Optimization (Optuna)", nested=True):
        study.optimize(objective, n_trials=3)


    best_params = study.best_params
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["seed"] = 42
    best_params["objective"] = "reg:squarederror"

    return best_params

# Entrenamiento del modelo final y se usan los mejores hiperparámetros
# Registro de métricas, parám., metadatos.
# Se guarda el preprocesador DictVect.
# Se registra el modelo final en mlflow
@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best model ever"):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(best_params)

        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        })

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=10,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)


        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)

        signature = infer_signature(input_example, y_val[:5])

        mlflow.xgboost.log_model(
            booster,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )
    return None

# Se registra el mejor modelo en el registro de modelos
# Se busca el run con el menor rmse en el experimentos.
# Se asigna el alias 'champion' a la versión que se registró 
@task(name="Register Best Model")
def register_best_model(experiment_name: str, registered_model_name: str = "workspace.default.nyc-taxi-model-prefect"):

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No se encontró el experimento con nombre: {experiment_name}")

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print("Champion:")
    print(f"id: {best_run_id}")
    print(f"validación rmse {best_run.data.metrics.get('rmse')}")

    run_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(model_uri=run_uri, name=registered_model_name)
    print(f"Modelo: '{registered_model.name}' , versión: {registered_model.version})")

    client.set_registered_model_alias(
        name=registered_model.name,
        alias="Champion",
        version=registered_model.version
    )
    print("Se asignó el alias '@Champion'.")

# Pipeline principal de entrenamiento
# Se definen rutas de datos y nombre dek experimento
# Se cargan datos de entrenamiento/val.
# Transformaciones y extracciónes de carac.
# Entrenamiento de modelo final y se registra el modelo en registry.
@flow(name="Main Flow")
def main_flow(year: int, month_train: str, month_val: str) -> None:
    """The main training pipeline"""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True) 
    EXPERIMENT_NAME = "/Users/priscila.cervantes@iteso.mx/nyc-taxi-experiment-prefect"

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    df_train = read_data(train_path)
    df_val = read_data(val_path)


    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    

    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    register_best_model(experiment_name=EXPERIMENT_NAME)

if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02")




