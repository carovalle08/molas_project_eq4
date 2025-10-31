"""
Esqueleto OOP para pipeline de Bike Sharing (sólo estructura, sin implementación de métodos).
Clases incluidas: DataLoader, Preprocessor, Model, Evaluator, Visualizer, Orchestrator.
"""
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns



class DataLoader:
    """Carga y particiona datos desde CSV.

    Atributos:
        data_path: Ruta al archivo CSV.
        target_col: Nombre de la variable objetivo (p.ej. "cnt").
        drop_cols: Columnas a eliminar antes del modelado.
    """
    def __init__(self, data_path, target_col="cnt", drop_cols=None):
        self.data_path = data_path
        self.target_col = target_col
        self.drop_cols = drop_cols

    def load(self):
        """Lee el CSV, elimina columnas no deseadas y retorna un DataFrame listo.
        Returns:
            pd.DataFrame: Datos cargados.
        """
        df = pd.read_csv(self.data_path)
        print(f"Original shape: {df.shape}")
        
        # Eliminar columnas basadas en análisis de correlación
        df = df.drop(columns=self.drop_cols)
        print(f"Shape after dropping columns: {df.shape}")
        print(f"Remaining columns: {list(df.columns)}")

        if df.isnull().sum().any():
            print("\nWarning: Missing values found!")
            print(df.isnull().sum())
        else:
            print("\nNo missing values found.")

        return df

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Divide en conjuntos de entrenamiento y prueba.

        Args:
            df: DataFrame completo.
            test_size: Proporción para el conjunto de prueba.
            random_state: Semilla de aleatoriedad.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(self.target_col)
        y = df[self.target_col]

        print(f"\nSplitting data (test_size=0.2)...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test


class Preprocessor:
    """Crea y aplica transformaciones (escalado, one-hot, etc.).

    Atributos:
    numerical_features: Nombres de columnas numéricas.
    categorical_features: Nombres de columnas categóricas.
    column_transformer: Transformador de columnas (se construye con build()).
    """

    def __init__(self, numerical_features=None, categorical_features=None):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.column_transformer = None

    def clean_cat_variables(X):
        """Limpiar variables categoricas:
        - Reemplazar valores nulos por la moda
        - Eliminar valores no validos obvios (no numericos)
        - Reemplazar valores no validos pero si numericos, fuera del conjunto válido, por la moda."""

    def clean_num_variables(X):
        """Limpiar varibales numericas:
        - Reemplazar valores nulos por la mediana
        - Quitar outliers """

    def build(self):
        """Construye y retorna el ColumnTransformer."""
    

    def transform(self, df):
        """Aplica el transformador a df y retorna matriz transformada."""
    

