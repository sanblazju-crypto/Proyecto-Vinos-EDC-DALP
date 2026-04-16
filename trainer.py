"""
trainer.py
Módulo responsable de la división train/test y el entrenamiento de los modelos.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from logger_config import get_logger

logger = get_logger(__name__)

# Catálogo de modelos disponibles para el entrenamiento
MODEL_CATALOG: dict = {
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "decision_tree": DecisionTreeClassifier,
}


class Trainer:
    """
    Divide los datos en conjuntos de entrenamiento y prueba,
    instancia los modelos seleccionados y los entrena.
    """

    def __init__(self, train_size: float = 0.7, random_state: int = 25):
        """
        Args:
            train_size: Proporción de datos para entrenamiento (entre 0 y 1).
            random_state: Semilla para reproducibilidad.
        """
        self.train_size = train_size
        self.random_state = random_state
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.trained_models: dict = {}

    def split(self, X: np.ndarray, y: pd.Series) -> tuple:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            X: Características escaladas.
            y: Etiquetas.

        Returns:
            Tuple (X_train, X_test, y_train, y_test).

        Raises:
            ValueError: Si train_size no está en el rango válido.
            RuntimeError: Si la división falla.
        """
        logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        try:
            if not (0 < self.train_size < 1):
                raise ValueError(f"train_size debe estar entre 0 y 1, recibido: {self.train_size}")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=self.train_size, random_state=self.random_state
            )

            train_pct = round(len(self.X_train) / len(X) * 100)
            test_pct = round(len(self.X_test) / len(X) * 100)
            logger.info(f"División completada → Entrenamiento: {train_pct}% | Prueba: {test_pct}%")
            logger.debug(f"Tamaño X_train: {self.X_train.shape} | X_test: {self.X_test.shape}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except ValueError as ve:
            logger.error(f"Parámetro inválido en split: {ve}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error inesperado durante la división: {e}", exc_info=True)
            raise RuntimeError(f"Fallo en la división de datos: {e}") from e

    def train(self, model_keys: list | None = None, model_params: dict | None = None) -> dict:
        """
        Instancia y entrena los modelos especificados.

        Args:
            model_keys: Lista de claves del MODEL_CATALOG a entrenar.
                        Si es None, se entrenan todos los modelos del catálogo.
            model_params: Diccionario con hiperparámetros por modelo.
                          Ejemplo: {"svm": {"C": 10, "kernel": "rbf"}}

        Returns:
            dict: Diccionario {nombre_modelo: modelo_entrenado}.

        Raises:
            RuntimeError: Si split() no fue llamado antes o si algún modelo falla.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Los datos no han sido divididos. Llama a split() antes de train().")

        model_keys = model_keys or list(MODEL_CATALOG.keys())
        model_params = model_params or {}

        logger.info(f"Iniciando entrenamiento de {len(model_keys)} modelo(s): {model_keys}")

        for key in model_keys:
            try:
                if key not in MODEL_CATALOG:
                    logger.warning(f"Modelo '{key}' no encontrado en el catálogo. Se omite.")
                    continue

                params = model_params.get(key, {})
                logger.debug(f"Entrenando '{key}' con parámetros: {params or 'por defecto'}")

                model_class = MODEL_CATALOG[key]
                model = model_class(**params)
                model.fit(self.X_train, self.y_train)

                self.trained_models[key] = model
                logger.info(f"Modelo '{key}' entrenado correctamente.")

            except Exception as e:
                logger.error(f"Error al entrenar el modelo '{key}': {e}", exc_info=True)
                # Continuamos con el resto de modelos en lugar de abortar
                continue

        if not self.trained_models:
            raise RuntimeError("Ningún modelo pudo ser entrenado correctamente.")

        logger.info(f"Entrenamiento finalizado. Modelos disponibles: {list(self.trained_models.keys())}")
        return self.trained_models
