import math
import optuna
import pathlib
import pickle
import mlflow
import pandas as pd
import xgboost as xgb
from datetime import datetime
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlflow.tracking import MlflowClient
from prefect import flow, task
from dotenv import load_dotenv


# Lectura de datos
@task
def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    return df

# Creación de features
@task
def build_feature_matrix(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dv = DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values
    return X_train, X_val, y_train, y_val, dv

# Se calcula el rmse
def rmse(y_true, y_pred):
    return float(root_mean_squared_error(y_true, y_pred))

# Gradient boostong con optina+mlflow
@task
def tune_and_train_gb(X_train, X_val, y_train, y_val, dv, n_trials=10):
    mlflow.sklearn.autolog(log_models=False)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-7), 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 80),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "random_state": 42,
        }
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "gradient_boosting")
            mlflow.log_params(params)
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metric = rmse(y_val, y_pred)
            mlflow.log_metric("rmse", metric)

            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_val[:5], signature=signature)

        return metric

    with mlflow.start_run(run_name="GradientBoost Hyperopt (Optuna)"):
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_state"] = 42

        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run(run_name="Best Model GradientBoost"):
            mlflow.log_params(best_params)
            final_model = GradientBoostingRegressor(**best_params)
            final_model.fit(X_train, y_train)
            y_pred = final_model.predict(X_val)
            final_rmse = rmse(y_val, y_pred)
            mlflow.log_metric("rmse", final_rmse)

            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

            feature_names = dv.get_feature_names_out()
            input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
            signature = infer_signature(input_example, y_pred[:5])
            mlflow.sklearn.log_model(final_model, artifact_path="model", input_example=input_example, signature=signature)

            return {"run_id": mlflow.active_run().info.run_id, "rmse": final_rmse}

# Random Forest con optuna + mlflow
@task
def tune_and_train_rf(X_train, X_val, y_train, y_val, dv, n_trials=10):
    mlflow.sklearn.autolog(log_models=False)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 80),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "random_forest")
            mlflow.log_params(params)
            model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metric = rmse(y_val, y_pred)
            mlflow.log_metric("rmse", metric)

            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(model, "model", input_example=X_val[:5], signature=signature)
        return metric

    with mlflow.start_run(run_name="RandomForest Hyperopt (Optuna)"):
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_state"] = 42

        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run(run_name="Best Model RandomForest"):
            mlflow.log_params(best_params)
            final_model = RandomForestRegressor(**best_params, n_jobs=-1)
            final_model.fit(X_train, y_train)
            y_pred = final_model.predict(X_val)
            final_rmse = rmse(y_val, y_pred)
            mlflow.log_metric("rmse", final_rmse)

            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

            feature_names = dv.get_feature_names_out()
            input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
            signature = infer_signature(input_example, y_pred[:5])
            mlflow.sklearn.log_model(final_model, artifact_path="model", input_example=input_example, signature=signature)

            return {"run_id": mlflow.active_run().info.run_id, "rmse": final_rmse}

# Registro y promoción de modelos
@task
def register_best_as_challenger(run_infos: list, registered_model: str):
    client = MlflowClient()
    best = min(run_infos, key=lambda r: r["rmse"])
    run_id = best["run_id"]
    run_uri = f"runs:/{run_id}/model"

    mv = mlflow.register_model(model_uri=run_uri, name=registered_model)
    client.set_registered_model_alias(name=registered_model, alias="Challenger", version=mv.version)
    return {"registered_version": mv.version, "registered_name": mv.name, "run_id": run_id, "rmse": best["rmse"]}


@task
def evaluate_and_promote(registered_model: str, march_parquet_path: str, challenger_info: dict):
    client = MlflowClient()

    champ_mv = client.get_model_version_by_alias(registered_model, "Champion")
    champ_version = champ_mv.version
    champ_run_id = champ_mv.run_id

    challenger_version = challenger_info["registered_version"]

    run_id = challenger_info["run_id"]
    client.download_artifacts(run_id=run_id, path="preprocessor/preprocessor.b", dst_path=".")
    with open("preprocessor/preprocessor.b", "rb") as f_in:
        dv = pickle.load(f_in)

    df_march = read_dataframe(march_parquet_path)
    df_march['PU_DO'] = df_march['PULocationID'].astype(str) + '_' + df_march['DOLocationID'].astype(str)
    X_march = dv.transform(df_march[['PU_DO', 'trip_distance']].to_dict(orient='records'))
    y_march = df_march['duration'].values

    feature_names = dv.get_feature_names_out()
    X_march_df = pd.DataFrame(X_march.toarray(), columns=feature_names)

    champ_model = mlflow.pyfunc.load_model(f"models:/{registered_model}@Champion")
    chall_model = mlflow.pyfunc.load_model(f"models:/{registered_model}@Challenger")

    champ_preds = champ_model.predict(X_march_df)
    chall_preds = chall_model.predict(X_march_df)
    champ_rmse = rmse(y_march, champ_preds)
    chall_rmse = rmse(y_march, chall_preds)

    print(f"Champion RMSE de la evaluación en Marzo {champ_rmse:.6f}, Challenger rmse = {chall_rmse:.6f}")


    if chall_rmse < champ_rmse:
       
        client.set_registered_model_alias(name=registered_model, alias="Champion", version=challenger_version)
        if champ_version is not None:
            client.set_registered_model_alias(name=registered_model, alias="Challenger", version=champ_version)
        promoted = True
        print(f"Version promovida {challenger_version} a Champion, de {champ_rmse} a {chall_rmse})")
    else:
        client.set_registered_model_alias(name=registered_model, alias="Challenger", version=challenger_version)
        promoted = False
        print("Sin cambio de Champion.")

    return {
        "champion_rmse": champ_rmse,
        "challenger_rmse": chall_rmse,
        "promoted": promoted,
        "champion_version_after": client.get_model_version_by_alias(registered_model, "Champion").version
    }

# Flujo principal
@flow(name="challenger_vs_champion")
def challenger_vs_champion(registered_model: str = "workspace.default.nyc-taxi-model-prefect"):
    load_dotenv(override=True)
    EXPERIMENT_NAME = "/Users/priscila.cervantes@iteso.mx/nyc-taxi-experiment-prefect"

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    train_path = "../data/green_tripdata_2025-01.parquet"
    val_path = "../data/green_tripdata_2025-02.parquet"
    data_march = "../data/green_tripdata_2025-03.parquet"

    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    X_train, X_val, y_train, y_val, dv = build_feature_matrix(df_train, df_val)

    gb_info = tune_and_train_gb(X_train, X_val, y_train, y_val, dv)
    rf_info = tune_and_train_rf(X_train, X_val, y_train, y_val, dv, wait_for=[gb_info])

    registered = register_best_as_challenger([gb_info, rf_info], registered_model, wait_for=[gb_info, rf_info])

    promotion_result = evaluate_and_promote(registered_model, data_march, registered, wait_for=[registered])

    return {"registered": registered, "promotion_result": promotion_result}


if __name__ == "__main__":
    challenger_vs_champion()

