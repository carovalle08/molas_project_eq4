"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Nota: las importaciones de sklearn están aquí solo para type hints.
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 1) DataLoader -----------------------------------------------------------------
@dataclass
class DataLoader:
    """Carga y particiona datos desde CSV.

    Atributos:
        data_path: Ruta al archivo CSV.
        target_col: Nombre de la variable objetivo (p.ej. "cnt").
        drop_cols: Columnas a eliminar antes del modelado.
    """

    data_path: Path | str
    target_col: str = "cnt"
    drop_cols: List[str] = field(default_factory=list)

    def load(self) -> pd.DataFrame:
        """Lee el CSV, elimina columnas no deseadas y retorna un DataFrame listo.

        Returns:
            pd.DataFrame: Datos cargados.
        """
        pass

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide en conjuntos de entrenamiento y prueba.

        Args:
            df: DataFrame completo.
            test_size: Proporción para el conjunto de prueba.
            random_state: Semilla de aleatoriedad.

        Returns:
            X_train, X_test, y_train, y_test
        """
        pass


# 2) Preprocessor ---------------------------------------------------------------
@dataclass
class Preprocessor:
    """Crea y aplica transformaciones (escalado, one-hot, etc.).

    Atributos:
        numerical_features: Nombres de columnas numéricas.
        categorical_features: Nombres de columnas categóricas.
        column_transformer: Transformador de columnas (se construye con build()).
    """

    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    column_transformer: Optional[ColumnTransformer] = None

    def build(self) -> ColumnTransformer:
        """Construye y retorna el ColumnTransformer."""
        pass

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Aplica el transformador a df y retorna matriz transformada."""
        pass


# 3) Model (unifica ModelZoo + ModelSpec) --------------------------------------
@dataclass
class Model:
    """Representa un modelo individual con su configuración y procesos.

    Atributos:
        name: Nombre del modelo (p.ej. "RandomForest", "Ridge").
        estimator: Instancia del estimador sklearn.
        param_grid: Grid de hiperparámetros para búsqueda.
        pipeline: Pipeline con preprocesamiento + modelo.
        best_estimator_: Mejor estimador tras la búsqueda.
        best_params_: Mejores hiperparámetros encontrados.
        training_time_: Tiempo de entrenamiento (segundos).
        metrics_: Métricas calculadas tras la evaluación.
    """

    name: str
    estimator: RegressorMixin
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    pipeline: Optional[Pipeline] = None

    # Resultados post-entrenamiento/evaluación
    best_estimator_: Optional[RegressorMixin] = None
    best_params_: Optional[Dict[str, Any]] = None
    training_time_: Optional[float] = None
    metrics_: Optional[Dict[str, Any]] = None

    def build_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Crea y retorna un pipeline con (preprocessor -> model)."""
        pass

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = "neg_root_mean_squared_error",
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """Ejecuta la búsqueda de hiperparámetros/ajuste del pipeline.

        Returns:
            Dict con llaves como best_estimator_, best_params_, training_time_.
        """
        pass

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Calcula métricas (RMSE, MAE, R2, MAPE) en train y test.

        Returns:
            Dict con métricas por split.
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones con el mejor modelo entrenado."""
        pass

    def summary(self) -> Dict[str, Any]:
        """Devuelve un resumen del modelo: nombre, mejores params y métricas."""
        pass


# 4) Evaluator ------------------------------------------------------------------
@dataclass
class Evaluator:
    """Agrega y compara resultados de múltiples modelos."""

    results_: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def compare(self, models: List[Model]) -> pd.DataFrame:
        """Construye una tabla con métricas/tiempos por modelo."""
        pass


# 5) Visualizer -----------------------------------------------------------------
@dataclass
class Visualizer:
    """Genera visualizaciones a partir de resultados y métricas."""

    out_dir: Path = Path("reports")

    def plot_metrics(self, df_results: pd.DataFrame) -> Path:
        """Genera y guarda gráficos comparativos de métricas."""
        pass

    def plot_predictions(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        title: str | None = None,
    ) -> Path:
        """Grafica valores reales vs. predichos y guarda la figura."""
        pass


# 6) Orchestrator ---------------------------------------------------------------
@dataclass
class Orchestrator:
    """Coordina el flujo E2E: carga → preprocesa → entrena → evalúa → visualiza."""

    loader: DataLoader
    preprocessor: Preprocessor
    models: List[Model]
    evaluator: Evaluator = field(default_factory=Evaluator)
    visualizer: Visualizer = field(default_factory=Visualizer)

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Ejecuta el pipeline completo y devuelve un diccionario de resultados."""
        pass


if __name__ == "__main__":
    # Punto de entrada opcional para ejecutar el orquestador.
    pass
