"""
Entrenamiento Multi-Modelo con Mejores Prácticas de Pipeline de Scikit-Learn.

Este script sigue mejores prácticas de scikit-learn y MLOps:
- Todo el preprocesamiento va dentro del Pipeline (evita fugas y facilita despliegue)
- ColumnTransformer para numéricas/categóricas
- FunctionTransformer para ingeniería de características sin cambiar el # de filas
- GridSearchCV ajusta todo el Pipeline
- Integración con MLflow (parámetros, métricas, artefactos y registro de modelo)
- Compatible con DVC: acepta rutas/params por CLI y guarda salidas donde el pipeline las espera
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import yaml


# Utilidades
def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

def detect_feature_types(df: pd.DataFrame, target_col: str):
    cols = [c for c in df.columns if c != target_col]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat

# Clipping de outliers (NO elimina filas para no romper scikit-learn)
def clip_outliers_df(X):
    df = pd.DataFrame(X).copy()
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(low, high)
    return df.values


def add_simple_features(X):
    df = pd.DataFrame(X).copy()
    if "temp" in df.columns:
        df["temp2"] = df["temp"] ** 2
    return df

# Pipeline

def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ]
    )

def build_feature_engineering():
    return FunctionTransformer(add_simple_features, feature_names_out="one-to-one", validate=False)

def build_clip_outliers():
    return FunctionTransformer(clip_outliers_df, validate=False)

# Configuración de modelos + grid
def get_model_spaces(random_state: int = 42):
    spaces = {
        "random_forest": {
            "estimator": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [10, 12, None],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "param_grid": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.05, 0.1],
            },
        },
        "ridge": {
            "estimator": Ridge(),  # Ridge no acepta random_state
            "param_grid": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        },
    }
    return spaces

# Entrenamiento y evaluación
def evaluate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Evitar división por cero en MAPE
    y_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_safe)) * 100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def train_one_model(name, estimator, param_grid, X_train, y_train, X_test, y_test,
                    numeric_features, categorical_features, cv=5, random_state=42):
    # Pipeline completo
    pipe = Pipeline(
        steps=[
            ("fe", build_feature_engineering()),
            ("clip", build_clip_outliers()),
            ("prep", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    best_pipe = gs.best_estimator_

    # Inference time (ms por muestra) en TEST
    import time as _t
    t0 = _t.time()
    _ = best_pipe.predict(X_test)
    inf_time_ms = ((_t.time() - t0) / len(X_test)) * 1000.0

    preds = best_pipe.predict(X_test)
    metrics = evaluate_metrics(y_test, preds)
    metrics.update({"best_params": gs.best_params_, "inference_time_ms": inf_time_ms})
    return best_pipe, metrics

# Visualización simple
def plot_model_comparison(df_comp: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    df_comp.plot(x="model", y="rmse", kind="bar", legend=False)
    plt.ylabel("RMSE (lower is better)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento multimodelo con Pipeline + MLflow + DVC")
    parser.add_argument("--raw", default=None, help="Ruta a CSV crudo (opcional si ya existe processed)")
    parser.add_argument("--data", default=None, help="Ruta a parquet procesado (si existe)")
    parser.add_argument("--params", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()

    P = load_params(args.params)

    raw_path = args.raw or P["data"]["raw_path"]
    proc_path = args.data or P["data"]["processed_path"]
    random_state = P.get("train", {}).get("random_state", 42)
    test_size = P.get("train", {}).get("test_size", 0.2)

    ensure_dirs()

    # Carga de datos
    if args.data and Path(proc_path).exists():
        df = pd.read_parquet(proc_path)
    else:
        df = pd.read_csv(raw_path)
        # Asegura nombre homogéneo para objetivo si tu dataset lo requiere
        if "cnt" in df.columns and "target" not in df.columns:
            df = df.rename(columns={"cnt": "target"})
        Path(proc_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(proc_path, index=False)

    if "target" not in df.columns:
        raise ValueError("No encuentro la columna objetivo 'target'. Renómbrala en build_features o aquí.")

    X = df.drop(columns=["target"])
    y = df["target"].values

    num_cols, cat_cols = detect_feature_types(df, target_col="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Configuración de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("exp_pipeline_multimodels")

    spaces = get_model_spaces(random_state=random_state)

    results = []
    best = {"name": None, "rmse": float("inf"), "pipe": None, "run_id": None}

    for name, spec in spaces.items():
        with mlflow.start_run(run_name=name) as run:
            pipe, metrics = train_one_model(
                name=name,
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                numeric_features=num_cols,
                categorical_features=cat_cols,
                cv=P.get("eval", {}).get("cv_folds", 5),
                random_state=random_state,
            )

            # Log params/metrics
            mlflow.log_params(metrics.get("best_params", {}))
            mlflow.log_metrics(
                {
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "mape": metrics["mape"],
                    "inference_time_ms": metrics["inference_time_ms"],
                }
            )

            # Guarda pipeline individual y registra el modelo
            joblib.dump(pipe, f"models/{name}_pipeline.pkl")
            mlflow.sklearn.log_model(
                pipe, artifact_path="model", registered_model_name="molas_regressor"
            )

            # Acumula resultados
            results.append({"model": name, **{k: metrics[k] for k in ["rmse", "mae", "r2", "mape"]}})

            if metrics["rmse"] < best["rmse"]:
                best.update({"name": name, "rmse": metrics["rmse"], "pipe": pipe, "run_id": run.info.run_id})

    # Guardados globales
    # 1) Mejor pipeline y modelo estándar para DVC
    joblib.dump(best["pipe"], "models/best_pipeline.pkl")
    joblib.dump(best["pipe"], "models/model.pkl")  # para cumplir outs: models/model.pkl

    # 2) Métricas agregadas
    all_metrics = {row["model"]: {k: row[k] for k in ["rmse", "mae", "r2", "mape"]} for row in results}
    with open("models/all_models_metrics_pipeline.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Comparativa y figura
    comp_df = pd.DataFrame(results).sort_values("rmse")
    fig_path = "reports/figures/model_comparison_pipeline.png"
    plot_model_comparison(comp_df, fig_path)

    # Adjunta artefactos al run ganador
    if best["run_id"]:
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.log_artifact("models/all_models_metrics_pipeline.json")
            mlflow.log_artifact(fig_path)

    # También deja un metrics.json genérico (para una etapa evaluate si lo requiere)
    with open("reports/metrics.json", "w") as f:
        json.dump({"best_model": best["name"], "rmse": float(best["rmse"])}, f, indent=2)

    print(f"Mejor modelo: {best['name']}  RMSE={best['rmse']:.4f}")
    print("Artefactos:")
    print(" - models/model.pkl (estándar para pipeline DVC)")
    print(" - models/best_pipeline.pkl y models/<model>_pipeline.pkl")
    print(" - models/all_models_metrics_pipeline.json")
    print(" - reports/metrics.json")
    print(" - reports/figures/model_comparison_pipeline.png")


if __name__ == "__main__":
    main()
