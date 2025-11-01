"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Model:
    """Representa un modelo individual con su configuración y procesos.

    Atributos:
    name: Nombre del modelo (p.ej. "RandomForest", "Ridge").
    estimator: Instancia del estimador sklearn.
    param_grid: Grid de hiperparámetros para búsqueda (dict).
    pipeline: Pipeline con preprocesamiento + modelo.
    best_estimator_: Mejor estimador tras la búsqueda.
    best_params_: Mejores hiperparámetros encontrados.
    training_time_: Tiempo de entrenamiento (segundos).
    metrics_: Métricas calculadas tras la evaluación.
    """

    def __init__(self, name, estimator, param_grid=None):
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid or {}
        self.pipeline = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.training_time_ = None
        self.metrics_ = None

    def build_pipeline(self, preprocessor):
        """Crea y retorna un pipeline con (preprocessor -> model)."""
        

    def train(self, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=None, verbose=0):
        """Ejecuta la búsqueda de hiperparámetros/ajuste del pipeline.

        Returns:
        dict con llaves como best_estimator_, best_params_, training_time_.
        """
        

    def evaluate(self, X_train, y_train, X_test, y_test):
        """Calcula métricas (RMSE, MAE, R2, MAPE) en train y test.

        Returns:
        dict con métricas por split.
        """
        

    def predict(self, X):
        """Genera predicciones con el mejor modelo entrenado."""
    

    def summary(self):
        """Devuelve un resumen del modelo: nombre, mejores params y métricas."""
    


class Evaluator:
    """Agrega y compara resultados de múltiples modelos."""

    results_: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def compare(self, models: List[Model]) -> pd.DataFrame:
        """Construye una tabla con métricas/tiempos por modelo."""
        pass